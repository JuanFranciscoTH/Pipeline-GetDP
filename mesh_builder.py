"""mesh_builder.py -- Gmsh mesh of the domain box + the coil, with the air/rock
interface following the block steps of the CSV.

  1. Mesh a plain box (the outer faces of the block grid) with the coil torus
     carved out.  Element sizes come from the size zones of config.yaml -> mesh
     (combined with a Min field, so the finest zone wins at every point).
  2. Classify every box tetrahedron by the block that contains its CENTROID:
     air block -> air (tag 1); rock block -> rock, grouped in material classes
     (tags 3, 4, ...): one class per distinct (sigma, mu_r, eps_r) of the rock
     blocks, sigma rounded to SIGMA_CLASS_DECADES in log10.  GetDP gets one region
     per class, so every block keeps (almost exactly) its own properties.
No surface is fitted: the interface is stepped at the mesh resolution, which the
'interface' zone refines near the coil.  The CSV is only read.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import gmsh
import numpy as np

AIR_TAG, COIL_TAG, FIRST_ROCK_TAG = 1, 2, 3
OUTER_TAG = 1000               # outer boundary surface (kept clear of the volume tags)
SIGMA_CLASS_DECADES = 0.02     # log10(sigma) resolution of the rock classes (~4.7 %)
NEAR_TRANSITION = 0.75         # near zone: size-transition width / zone half-size
INTERFACE_RAMP = 100.0         # interface size -> far_size within this distance [m]


@dataclass
class MeshInfo:
    msh: str
    n_nodes: int
    n_tets: int
    coil_z: float
    rock_classes: list = field(default_factory=list)   # [(tag, sigma, mu_r, eps_r, n_tets)]


def build_mesh(env, coil: dict, mesh: dict, coil_z: float, msh: str,
               verbosity: int = 2) -> MeshInfo:
    xmin, ymin, zmin, xmax, ymax, zmax = env.bounds
    cx, cy = (float(v) for v in coil["center_xy"])
    R, r = float(coil["major_radius"]), float(coil["minor_radius"])
    far = float(mesh["far_size"])

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("General.Verbosity", verbosity)
    gmsh.model.add("model")
    occ = gmsh.model.occ
    box = occ.addBox(xmin, ymin, zmin, xmax - xmin, ymax - ymin, zmax - zmin)
    torus = occ.addTorus(cx, cy, coil_z, R, r)          # horizontal loop (z axis)
    _, children = occ.fragment([(3, box)], [(3, torus)])
    occ.synchronize()
    coil_v = sorted({t for _, t in children[1]})
    air_v = sorted({t for _, t in children[0]} - set(coil_v))
    gmsh.model.addPhysicalGroup(3, air_v, tag=AIR_TAG, name="air")
    gmsh.model.addPhysicalGroup(3, coil_v, tag=COIL_TAG, name="coil")

    # outer boundary = the box faces
    tol = 1e-3
    outer = []
    for _, t in gmsh.model.getEntities(2):
        b = gmsh.model.getBoundingBox(2, t)
        if any(abs(b[a] - b[a + 3]) < tol and min(abs(b[a] - lo), abs(b[a] - hi)) < tol
               for a, (lo, hi) in enumerate(((xmin, xmax), (ymin, ymax), (zmin, zmax)))):
            outer.append(t)
    gmsh.model.addPhysicalGroup(2, outer, tag=OUTER_TAG, name="outer")

    # ---- size zones (all combined with Min) ---------------------------------
    F = gmsh.model.mesh.field
    fields = []

    # (1) grading from the coil: coil_size on the conductor -> far_size at grading_distance
    coil_surfs = sorted({abs(t) for v in coil_v
                         for _, t in gmsh.model.getBoundary([(3, v)], oriented=False)})
    fd = F.add("Distance"); F.setNumbers(fd, "SurfacesList", coil_surfs)
    ft = F.add("Threshold"); F.setNumber(ft, "InField", fd)
    F.setNumber(ft, "SizeMin", float(mesh["coil_size"])); F.setNumber(ft, "SizeMax", far)
    F.setNumber(ft, "DistMin", r); F.setNumber(ft, "DistMax", float(mesh["grading_distance"]))
    fields.append(ft)

    # (2) near zone: sphere (diameter = size) or cube (side = size) centred on the coil
    nz_cfg = mesh.get("near_zone")
    if nz_cfg:
        near_half = float(nz_cfg["size"]) / 2.0
        shape = str(nz_cfg.get("shape", "sphere")).lower()
        if shape == "sphere":
            f = F.add("Ball")
            for k, v in (("XCenter", cx), ("YCenter", cy), ("ZCenter", coil_z),
                         ("Radius", near_half)):
                F.setNumber(f, k, v)
        elif shape == "cube":
            f = _box_field(F, cx - near_half, cx + near_half, cy - near_half, cy + near_half,
                           coil_z - near_half, coil_z + near_half)
        else:
            raise SystemExit(f"mesh.near_zone.shape must be 'sphere' or 'cube', got {shape!r}")
        F.setNumber(f, "VIn", float(nz_cfg["element_size"])); F.setNumber(f, "VOut", far)
        F.setNumber(f, "Thickness", NEAR_TRANSITION * near_half)
        fields.append(f)

    # (3) interface zone: fine band along the stepped rock surface, only for the
    #     block columns within 'radius' (horizontal) of the coil axis.  The block-top
    #     points are NOT embedded in the mesh (that creates slivers that break the
    #     solve); the Distance field only reads their coordinates.
    iz = mesh.get("interface")
    if iz:
        rad = float(iz["radius"])
        XX, YY = np.meshgrid(env.x, env.y, indexing="ij")
        rho = np.hypot(XX - cx, YY - cy)
        d_tube = np.hypot(rho - R, env.surface - coil_z)
        sel = (rho <= rad) & (d_tube > r + 1.5) & env.rock.any(axis=2)
        pts = [occ.addPoint(float(px), float(py), float(pz))
               for px, py, pz in zip(XX[sel], YY[sel], env.surface[sel])]
        occ.synchronize()
        fdi = F.add("Distance"); F.setNumbers(fdi, "PointsList", pts)
        fti = F.add("Threshold"); F.setNumber(fti, "InField", fdi)
        F.setNumber(fti, "SizeMin", float(iz["element_size"])); F.setNumber(fti, "SizeMax", far)
        # flat band out to ~1 block step (covers the gaps between block-top points)
        F.setNumber(fti, "DistMin", 0.9 * max(env.dx, env.dy))
        F.setNumber(fti, "DistMax", INTERFACE_RAMP)
        fields.append(fti)

    fmin = F.add("Min"); F.setNumbers(fmin, "FieldsList", fields)
    F.setAsBackgroundMesh(fmin)
    for opt in ("MeshSizeFromPoints", "MeshSizeFromCurvature", "MeshSizeExtendFromBoundary"):
        gmsh.option.setNumber("Mesh." + opt, 0)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.model.mesh.generate(3)
    for _ in range(2):
        gmsh.model.mesh.optimize("Netgen")             # extra sliver-removal passes
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(msh)
    n_nodes = len(gmsh.model.mesh.getNodes()[0])
    gmsh.finalize()

    n_tets, classes = _classify(msh, env)
    return MeshInfo(msh, n_nodes, n_tets, coil_z, classes)


def _box_field(F, x0, x1, y0, y1, z0, z1):
    f = F.add("Box")
    for k, v in (("XMin", x0), ("XMax", x1), ("YMin", y0), ("YMax", y1),
                 ("ZMin", z0), ("ZMax", z1)):
        F.setNumber(f, k, v)
    return f


def _classify(msh: str, env) -> tuple:
    """Re-tag the air tets whose centroid lies in a rock block with the tag of
    its (sigma, mu_r, eps_r) class; rewrite the .msh.  Returns (n_tets, classes)."""
    import meshio
    mm = meshio.read(msh)
    P = np.asarray(mm.points)
    n_tets, keys_all, blocks = 0, [], []
    for ci, cb in enumerate(mm.cells):
        if cb.type != "tetra":
            continue
        n_tets += len(cb.data)
        phys = np.asarray(mm.cell_data["gmsh:physical"][ci]).astype(int)
        c = P[cb.data].mean(axis=1)
        i, j, k = env.index(c[:, 0], c[:, 1], c[:, 2])
        sel = (phys == AIR_TAG) & env.rock[i, j, k]
        s = env.sigma[i[sel], j[sel], k[sel]]
        m = env.mu_r[i[sel], j[sel], k[sel]]
        e = env.eps_r[i[sel], j[sel], k[sel]]
        lg = np.where(s > 0, np.rint(np.log10(np.where(s > 0, s, 1.0)) / SIGMA_CLASS_DECADES), -1e9)
        keys_all.append(np.column_stack([lg, np.round(m, 6), np.round(e, 6)]))
        blocks.append((ci, sel, s))
    keys = np.vstack(keys_all) if keys_all else np.zeros((0, 3))
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.ravel()
    allsig = np.concatenate([b[2] for b in blocks]) if blocks else np.zeros(0)
    classes = []
    for c_id, (lg, mu, eps) in enumerate(uniq):
        members = allsig[inv == c_id]
        sig = float(np.exp(np.mean(np.log(members)))) if lg > -1e8 else 0.0  # geometric mean
        classes.append((FIRST_ROCK_TAG + c_id, sig, float(mu), float(eps), int(len(members))))
    pos = 0
    for ci, sel, s in blocks:
        phys = np.asarray(mm.cell_data["gmsh:physical"][ci]).astype(int)
        phys[sel] = FIRST_ROCK_TAG + inv[pos:pos + int(sel.sum())]
        pos += int(sel.sum())
        mm.cell_data["gmsh:physical"][ci] = phys
    mm.field_data = {"air": np.array([AIR_TAG, 3]), "coil": np.array([COIL_TAG, 3]),
                     "outer": np.array([OUTER_TAG, 2])}
    for tag, *_ in classes:
        mm.field_data[f"rock_{tag}"] = np.array([tag, 3])
    meshio.write(msh, mm, file_format="gmsh22", binary=False)
    return n_tets, classes
