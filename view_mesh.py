"""view_mesh.py -- views of the solve mesh model.msh (run.py --mesh-only is enough).

Outputs (results/):
  mesh_3d.png          3D view: air (light blue, translucent), rock (brown), coil
  mesh_cut_XY.png, mesh_cut_YZ.png, mesh_cut_XZ.png
                       the three planes through the coil with the element edges
                       (shows the refinement zones of config.yaml -> mesh)
  mesh_zoom_coil.png   XZ cut zoomed on the coil
The cuts always show the whole domain.

Run:  python view_mesh.py  [--interactive]   (--interactive opens rotatable windows)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pyvista as pv
from matplotlib.colors import ListedColormap

import common
import mesh_builder

INTERACTIVE = "--interactive" in sys.argv
AIR, COIL, ROCK = 0, 1, 2                       # display categories
COLORS = ["#5DADE2", common.COPPER, "#8B7355"]
NAMES = ["air", "coil", "rock"]

try:                                            # benign VTK console notes
    from vtkmodules.vtkCommonCore import vtkObject
    vtkObject.GlobalWarningDisplayOff()
except Exception:
    pass


def load_grid():
    """model.msh -> pyvista grid with cell data 'category' (air / coil / rock)."""
    import meshio
    if not os.path.exists("model.msh"):
        raise SystemExit("model.msh not found -- build it with: python run.py --mesh-only")
    mm = meshio.read("model.msh")
    tets, tags = [], []
    for i, cb in enumerate(mm.cells):
        if cb.type == "tetra":
            tets.append(cb.data); tags.append(mm.cell_data["gmsh:physical"][i])
    tets = np.vstack(tets); tags = np.concatenate(tags).astype(int)
    cells = np.column_stack([np.full(len(tets), 4), tets]).ravel()
    grid = pv.UnstructuredGrid(cells, np.full(len(tets), pv.CellType.TETRA, np.uint8),
                               mm.points)
    grid.cell_data["category"] = np.where(tags == mesh_builder.AIR_TAG, AIR,
                                          np.where(tags == mesh_builder.COIL_TAG, COIL, ROCK))
    return grid


def part(grid, cat):
    return grid.threshold([cat - 0.5, cat + 0.5], scalars="category")


def shot(p, name):
    out = os.path.join(common.RESULTS, name)
    for k in range(5):                          # off-screen grabs can fail once in a while
        try:
            p.render(); p.screenshot(out); break
        except OSError:
            if k == 4:
                raise
    print("saved", out)
    if INTERACTIVE:
        p.show()
    p.close()


def slice_view(grid, normal, center, name, title, zoom_box=None):
    src = grid.clip_box(zoom_box, invert=False) if zoom_box else grid
    sl = src.slice(normal=normal, origin=center)
    if not sl.n_cells:
        return
    p = pv.Plotter(off_screen=not INTERACTIVE, window_size=(1100, 950))
    p.add_mesh(sl, scalars="category", cmap=ListedColormap(COLORS), clim=[-0.5, 2.5],
               show_edges=True, edge_color="gray", line_width=0.3,
               annotations={float(i): n for i, n in enumerate(NAMES)},
               scalar_bar_args={"title": "", "n_labels": 0})
    p.add_text(title, font_size=9)
    p.show_axes()
    p.show_bounds(location="outer", xtitle="x [m]", ytitle="y [m]", ztitle="z [m]",
                  font_size=10, color="dimgray")
    # XZ cut seen from -y so that x grows to the right (as in the field maps)
    p.view_vector({"x": (1, 0, 0), "y": (0, -1, 0), "z": (0, 0, 1)}[normal])
    p.camera.zoom(1.2 if zoom_box is None else 1.3)
    shot(p, name)


def main():
    cfg = common.load_config()
    os.makedirs(common.RESULTS, exist_ok=True)
    info = common.run_info(cfg)
    c, R = info["coil_center"], info["major_radius"]
    grid = load_grid()
    counts = [int((grid["category"] == k).sum()) for k in range(3)]
    print(f"model.msh: {grid.n_points:,} nodes, {grid.n_cells:,} tets "
          f"(air {counts[0]:,}, coil {counts[1]:,}, rock {counts[2]:,})")

    p = pv.Plotter(off_screen=not INTERACTIVE, window_size=(1300, 1000))
    for cat, opacity in ((AIR, 0.12), (ROCK, 0.55), (COIL, 1.0)):
        body = part(grid, cat)
        if body.n_cells:
            p.add_mesh(body, color=COLORS[cat], opacity=opacity, label=NAMES[cat])
    p.add_legend(bcolor="white", size=(0.16, 0.12), loc="upper right")
    p.add_text("Solve mesh: air (light blue), rock (brown), coil (copper)", font_size=10)
    p.show_axes()
    p.show_bounds(location="outer", xtitle="x [m]", ytitle="y [m]", ztitle="z [m]",
                  font_size=11, color="dimgray")
    p.camera_position = "iso"; p.reset_camera(); p.camera.zoom(common.ZOOM_3D)
    shot(p, "mesh_3d.png")

    slice_view(grid, "z", c, "mesh_cut_XY.png", f"XY cut at z = {c[2]:g} (coil plane)")
    slice_view(grid, "x", c, "mesh_cut_YZ.png", f"YZ cut at x = {c[0]:g}")
    slice_view(grid, "y", c, "mesh_cut_XZ.png", f"XZ cut at y = {c[1]:g}")
    pad = 4.0 * R
    slice_view(grid, "y", c, "mesh_zoom_coil.png", "XZ cut zoomed on the coil",
               zoom_box=(c[0] - R - pad, c[0] + R + pad, c[1] - R - pad, c[1] + R + pad,
                         c[2] - R - pad, c[2] + R + pad))


if __name__ == "__main__":
    main()
