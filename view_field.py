"""view_field.py -- |B| maps of the solved field (reads results/B_field.vtu).

Outputs (results/):
  B_cut_XY.png, B_cut_YZ.png, B_cut_XZ.png  |B| (log colour scale) on the three
      planes through the coil centre, with the coil and the air/rock interface
      (dotted, stepped)
  B_3d.png      3D view coloured by |B|, coil in copper
  markers_B.csv the field at config.yaml -> markers (also printed)
Window (cuts and 3D): a cube of +- plots.window_half_size around the coil, or the
whole domain with --full / plots.full_domain: true (files *_full.png).

Run:  python view_field.py  [--full]
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.colors import LogNorm

import common
import environment

cfg = common.load_config()
env = environment.load(cfg, verbose=False)
info = common.run_info(cfg)
C, R, FREQ = info["coil_center"], info["major_radius"], info["frequency"]
FULL = common.full_domain(cfg)
SUFFIX = "_full" if FULL else ""
BOX = common.window_box(cfg, C, env.bounds, FULL)
plots = cfg.get("plots") or {}
VMIN, VMAX = float(plots.get("B_min", 1e-13)), float(plots.get("B_max", 1e-5))

mesh = common.read_field()
print(f"|B| max {np.nanmax(mesh['Bamp']):.3e} T;  colour scale [{VMIN:.0e}, {VMAX:.0e}] T")


def cut(normal: str, fname: str, title: str) -> None:
    A, Bg, Bm = common.sample_plane(mesh, normal, C, BOX, "Bamp")
    (a0, a1), (b0, b1) = common.plane_ranges(normal, BOX)
    fig, ax = plt.subplots(figsize=(8.6, 8.0))
    pcm = ax.pcolormesh(A, Bg, np.clip(Bm, VMIN, None), norm=LogNorm(VMIN, VMAX),
                        cmap="turbo", shading="auto")
    fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.03).set_label("|B| [T]")
    xl, yl = common.draw_overlay(ax, normal, C, R, env, (a0, a1))
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_aspect("equal")
    ax.set_xlim(a0, a1); ax.set_ylim(b0, b1)
    ax.set_title(f"{title}   |   f = {FREQ:g} Hz")
    fig.tight_layout()
    out = os.path.join(common.RESULTS, fname)
    fig.savefig(out, dpi=130); plt.close(fig)
    print("saved", out)


def view_3d() -> None:
    src = mesh if FULL else mesh.clip_box(BOX, invert=False)
    surf = (src if src.n_cells else mesh).extract_surface()
    b = surf.bounds
    p = pv.Plotter(off_screen=True, window_size=(1350, 1050))
    p.add_mesh(surf, scalars="Bamp", cmap="turbo", clim=(VMIN, VMAX), log_scale=True,
               opacity=0.9, scalar_bar_args=common.BAR_3D | {"title": "B amplitude [T]"})
    p.add_mesh(common.coil_tube(C, R, info["minor_radius"]), color=common.COPPER)
    # (VTK's font has no '|' glyph: plain text in the 3D titles)
    p.add_text(f"B amplitude, coil in copper ({b[1] - b[0]:.0f} x {b[3] - b[2]:.0f} x "
               f"{b[5] - b[4]:.0f} m)   f = {FREQ:g} Hz", position="upper_edge", font_size=10)
    p.show_axes()
    p.show_bounds(location="outer", xtitle="x [m]", ytitle="y [m]", ztitle="z [m]",
                  font_size=10, color="dimgray")
    p.camera_position = "iso"; p.reset_camera(); p.camera.zoom(common.ZOOM_3D)
    out = os.path.join(common.RESULTS, f"B_3d{SUFFIX}.png")
    p.screenshot(out); p.close()
    print("saved", out)


if __name__ == "__main__":
    cut("z", f"B_cut_XY{SUFFIX}.png", f"|B| on the XY plane (z = {C[2]:g}, coil plane)")
    cut("x", f"B_cut_YZ{SUFFIX}.png", f"|B| on the YZ plane (x = {C[0]:g})")
    cut("y", f"B_cut_XZ{SUFFIX}.png", f"|B| on the XZ plane (y = {C[1]:g})")
    view_3d()
    common.print_markers(common.sample_markers(cfg, mesh, env),
                         csv_path=os.path.join(common.RESULTS, "markers_B.csv"))
