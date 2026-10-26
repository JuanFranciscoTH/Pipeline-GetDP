"""export_vtu.py -- convert GetDP's results/b.pos into results/B_field.vtu.

GetDP's `Print[ b, OnElementsOf Domain ]` writes a list-based Gmsh view: each
tetrahedron carries its own 4 vertex coordinates and the field at them for the
2 harmonic components (step 0 = real, step 1 = imaginary):

    VS(x0,y0,z0, ..., x3,y3,z3){ re(12), im(12) };

This rebuilds a conforming tetrahedral grid with NODAL fields (coincident
vertices merged, values averaged over the tets sharing a node):
    B_re, B_im  [T] (vectors)   and   Bamp = sqrt(|B_re|^2 + |B_im|^2)  [T]
B_field.vtu opens in ParaView and is what every post-processing script reads.
run.py calls it after the solve; run it by hand only to redo the conversion:
    python export_vtu.py
"""
from __future__ import annotations

import os

import numpy as np

POS = os.path.join("results", "b.pos")
VTU = os.path.join("results", "B_field.vtu")


def parse_pos(path: str) -> tuple:
    """-> vertices (M,4,3), B_re (M,4,3), B_im (M,4,3) of the M tetrahedra."""
    verts, bre, bim = [], [], []
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            if not ln.startswith("VS("):
                continue
            p2 = ln.index(")"); b1 = ln.index("{", p2); b2 = ln.index("}", b1)
            coords = np.array(ln[3:p2].split(","), float)            # 4 nodes x 3
            vals = np.array(ln[b1 + 1:b2].split(","), float)         # 2 steps x 4 x 3
            if coords.size != 12 or vals.size != 24:
                continue
            verts.append(coords.reshape(4, 3))
            vals = vals.reshape(2, 4, 3)
            bre.append(vals[0]); bim.append(vals[1])
    return np.asarray(verts), np.asarray(bre), np.asarray(bim)


def convert(pos: str = POS, out: str = VTU) -> str:
    import pyvista as pv
    if not os.path.exists(pos):
        raise SystemExit(f"{pos} not found -- it is written by the GetDP solve (python run.py)")
    print(f"[vtu] parsing {pos} ...", flush=True)
    verts, bre, bim = parse_pos(pos)
    nel = len(verts)
    V, Re, Im = verts.reshape(-1, 3), bre.reshape(-1, 3), bim.reshape(-1, 3)
    uniq, inv = np.unique(V.round(5), axis=0, return_inverse=True)
    inv = inv.ravel()
    acc_re = np.zeros((len(uniq), 3)); acc_im = np.zeros((len(uniq), 3))
    cnt = np.zeros(len(uniq))
    np.add.at(acc_re, inv, Re); np.add.at(acc_im, inv, Im); np.add.at(cnt, inv, 1)
    cnt[cnt == 0] = 1
    cells = np.empty((nel, 5), np.int64); cells[:, 0] = 4; cells[:, 1:] = inv.reshape(nel, 4)
    grid = pv.UnstructuredGrid(cells.ravel(), np.full(nel, pv.CellType.TETRA, np.uint8), uniq)
    grid.point_data["B_re"] = acc_re / cnt[:, None]
    grid.point_data["B_im"] = acc_im / cnt[:, None]
    grid.point_data["Bamp"] = np.sqrt((grid.point_data["B_re"] ** 2).sum(1)
                                      + (grid.point_data["B_im"] ** 2).sum(1))
    grid.save(out)
    print(f"[vtu] saved {out}: {len(uniq):,} nodes, {nel:,} tets, "
          f"|B| max {grid.point_data['Bamp'].max():.3e} T")
    return out


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    convert()
