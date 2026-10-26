"""view_environment.py -- 3D view of the block-model CSV (config.yaml ->
environment.csv), as the pipeline reads it.  Air is transparent; rock blocks are
solid, coloured by conductivity (default) or by elevation.

Run:
  python view_environment.py             # sigma, interactive window (rotate / zoom, q = close)
  python view_environment.py z           # elevation z
  python view_environment.py sigma --png # save results/environment_sigma_{iso,xz,xy}.png
  python view_environment.py z --png     #      results/environment_z_{iso,xz,xy}.png
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pyvista as pv

import common
import environment

ARGS = [a.lower().lstrip("-") for a in sys.argv[1:]]
PNG = "png" in ARGS
MODE = "z" if "z" in ARGS else "sigma"
for a in ARGS:
    if a not in ("z", "sigma", "png"):
        raise SystemExit(f"unknown argument {a!r}; use: [sigma | z] [--png]")


def main():
    cfg = common.load_config()
    env = environment.load(cfg)
    nx, ny, nz = env.rock.shape
    grid = pv.ImageData(dimensions=(nx + 1, ny + 1, nz + 1), spacing=(env.dx, env.dy, env.dz),
                        origin=env.bounds[:3])
    grid.cell_data["rock"] = env.rock.ravel(order="F").astype(np.int8)
    grid.cell_data["sigma"] = env.sigma.ravel(order="F")
    rock = grid.threshold(0.5, scalars="rock")
    surf = rock.extract_surface()
    edges = dict(show_edges=True, edge_color="#3a2f1a", line_width=0.3) if max(nx, ny) <= 60 \
        else dict(show_edges=False)                 # edges only on small grids
    bar = {"vertical": True, "position_x": 0.85, "position_y": 0.18, "width": 0.045,
           "height": 0.62, "title_font_size": 20, "label_font_size": 16, "color": "black"}

    p = pv.Plotter(off_screen=PNG, window_size=(1500, 1100))
    p.set_background("white")
    if MODE == "z":
        surf["z [m]"] = surf.points[:, 2]
        lo, hi = float(surf.points[:, 2].min()), float(surf.points[:, 2].max())
        p.add_mesh(surf, scalars="z [m]", cmap="terrain", clim=(lo, hi), ambient=0.3,
                   diffuse=0.8, scalar_bar_args={"title": "z [m]", **bar}, **edges)
        desc = f"elevation z [{lo:g}, {hi:g}] m"
    else:
        s = np.asarray(rock.cell_data["sigma"], float)
        lo, hi = float(s[s > 0].min()) if (s > 0).any() else 1e-12, float(s.max())
        p.add_mesh(surf, scalars="sigma", cmap="plasma", log_scale=True, clim=(lo, hi),
                   ambient=0.3, diffuse=0.8, scalar_bar_args={"title": "sigma [S/m]", **bar},
                   **edges)
        desc = f"conductivity [{lo:.2g}, {hi:.2g}] S/m"
    air = grid.threshold(0.5, scalars="rock", invert=True).extract_surface()
    p.add_mesh(air, color="#8ecae6", opacity=0.05)
    p.add_mesh(grid.outline(), color="black", line_width=2)
    p.add_axes(line_width=3)
    p.show_bounds(grid="back", location="outer", xtitle="x [m]", ytitle="y [m]",
                  ztitle="z [m]", font_size=11, color="#333333")
    p.add_text(f"{os.path.basename(env.csv)}: rock coloured by {desc}; air transparent",
               font_size=10, color="black", position="upper_left")
    p.camera_position = "iso"; p.reset_camera(); p.camera.zoom(common.ZOOM_3D)
    if PNG:
        os.makedirs(common.RESULTS, exist_ok=True)
        for view in ("iso", "xz", "xy"):
            p.camera_position = view; p.reset_camera(); p.camera.zoom(common.ZOOM_3D)
            out = os.path.join(common.RESULTS, f"environment_{MODE}_{view}.png")
            p.screenshot(out); print("saved", out)
        p.close()
    else:
        print("rotate with the mouse, zoom with the wheel, press q to close")
        p.show()


if __name__ == "__main__":
    main()
