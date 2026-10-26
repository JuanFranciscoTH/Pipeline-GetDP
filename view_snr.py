"""view_snr.py -- SNR maps and curves from the solved |B|.

Model (config.yaml -> receiver, noise), at every point of results/B_field.vtu:
    K       = Q * N * omega * A               receiver gain [V/T]
    V_ind   = K * |B|                         induced EMF amplitude [V]
    V_noise = V_cal * 10^((level_db - calibration_db)/20) * sqrt(BW / bin)
              (V_cal = calibration_volts, x sqrt(2) if calibration_rms: noise as amplitude)
    B_noise = V_noise / K                     noise floor referred to field [T]
    SNR     = 20 log10( |B| / B_noise ) = 20 log10( V_ind / V_noise )  [dB]
|B| (GetDP) and the noise level (spectrum of amplitude^2 per bin) are both
amplitudes, so the SNR is the height of the tone above the noise floor of a
spectrum with bins of spectrum_bin_hz, scaled to the detection bandwidth.

Outputs (results/):
  SNR_cut_XY/YZ/XZ.png   SNR maps with the plots.SNR_contour_db contour (0 dB)
  SNR_3d.png             3D view coloured by SNR
  curve_<name>_B_SNR.png |B| and SNR along every config.yaml -> curves line
  markers_SNR.csv        markers with V_ind and SNR (also printed)
Window as in view_field.py (--full = whole domain, files *_full.png).

Run:  python view_snr.py  [--full]
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.colors import Normalize

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
SNR_MIN, SNR_MAX = float(plots.get("SNR_min_db", 0.0)), float(plots.get("SNR_max_db", 120.0))
SNR_LINE = float(plots.get("SNR_contour_db", 0.0))
BW = float(cfg["noise"]["bandwidth_hz"])

K = common.receiver_gain(cfg, FREQ)
V_NOISE, B_NOISE = common.noise_floor(cfg, K)
mesh = common.read_field()
mesh.point_data["SNRdB"] = common.snr_db(mesh["Bamp"], B_NOISE)
rx = cfg["receiver"]
_dbm = 10.0 * np.log10((V_NOISE / np.sqrt(2.0)) ** 2 / float(rx["input_impedance"]) * 1e3)
print(f"receiver: N={rx['turns']}, D={rx['diameter']} m, Q={rx['quality_factor']} "
      f"-> K = {K:.4e} V/T at {FREQ:g} Hz")
print(f"noise: {cfg['noise']['level_db']:g} dB/bin -> V_noise = {V_NOISE:.3e} V "
      f"({_dbm:.1f} dBm) -> B_noise = {B_NOISE:.3e} T  (BW = {BW:g} Hz)")
print(f"SNR over the mesh: [{np.nanmin(mesh['SNRdB']):.1f}, {np.nanmax(mesh['SNRdB']):.1f}] dB")


def cut(normal: str, fname: str, title: str) -> None:
    A, Bg, S = common.sample_plane(mesh, normal, C, BOX, "SNRdB")
    (a0, a1), (b0, b1) = common.plane_ranges(normal, BOX)
    fig, ax = plt.subplots(figsize=(8.6, 8.0))
    pcm = ax.pcolormesh(A, Bg, S, norm=Normalize(SNR_MIN, SNR_MAX), cmap="turbo",
                        shading="auto")
    try:                                            # detection-limit contour
        cs = ax.contour(A, Bg, S, levels=[SNR_LINE], colors="k", linewidths=1.6)
        ax.clabel(cs, fmt="%.0f dB", fontsize=8)
    except Exception:
        pass
    fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.03).set_label("SNR [dB]")
    xl, yl = common.draw_overlay(ax, normal, C, R, env, (a0, a1))
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_aspect("equal")
    ax.set_xlim(a0, a1); ax.set_ylim(b0, b1)
    ax.set_title(f"{title}   |   f = {FREQ:g} Hz, BW = {BW:g} Hz")
    fig.tight_layout()
    out = os.path.join(common.RESULTS, fname)
    fig.savefig(out, dpi=130); plt.close(fig)
    print("saved", out)


def view_3d() -> None:
    src = mesh if FULL else mesh.clip_box(BOX, invert=False)
    surf = (src if src.n_cells else mesh).extract_surface()
    b = surf.bounds
    p = pv.Plotter(off_screen=True, window_size=(1350, 1050))
    p.add_mesh(surf, scalars="SNRdB", cmap="turbo", clim=(SNR_MIN, SNR_MAX), opacity=0.9,
               scalar_bar_args=common.BAR_3D | {"title": "SNR [dB]"})
    p.add_mesh(common.coil_tube(C, R, info["minor_radius"]), color=common.COPPER)
    p.add_text(f"SNR, coil in copper ({b[1] - b[0]:.0f} x {b[3] - b[2]:.0f} x "
               f"{b[5] - b[4]:.0f} m)   f = {FREQ:g} Hz, BW = {BW:g} Hz",
               position="upper_edge", font_size=10)
    p.show_axes()
    p.show_bounds(location="outer", xtitle="x [m]", ytitle="y [m]", ztitle="z [m]",
                  font_size=10, color="dimgray")
    p.camera_position = "iso"; p.reset_camera(); p.camera.zoom(common.ZOOM_3D)
    out = os.path.join(common.RESULTS, f"SNR_3d{SUFFIX}.png")
    p.screenshot(out); p.close()
    print("saved", out)


def curve_plots() -> None:
    """|B| and SNR along every config.yaml -> curves line (2-panel figure)."""
    cc = cfg.get("curves") or {}
    for ln in cc.get("lines") or []:
        name = str(ln.get("name", "curve"))
        p0, p1 = np.asarray(ln["p0"], float), np.asarray(ln["p1"], float)
        pts = p0 + np.linspace(0.0, 1.0, int(ln.get("n_points", cc.get("n_points", 1000)))
                               )[:, None] * (p1 - p0)
        s = pv.PolyData(pts).sample(mesh)
        B = np.asarray(s["Bamp"], float)
        B[~np.asarray(s["vtkValidPointMask"], bool)] = np.nan
        axis = int(np.argmax(np.abs(p1 - p0)))
        xv = pts[:, axis]
        vertical = axis == 2
        z_if = float(env.surface_at(C[0], C[1]))
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 8.8), sharex=True)
        for a in (a1, a2):
            if vertical:
                a.axvspan(float(np.nanmin(xv)), z_if, color="#8B7355", alpha=0.12)
                a.axvline(z_if, color="k", ls=":", lw=1.3, label=f"rock top z = {z_if:g}")
                a.axvline(C[2], color=common.COPPER, ls="--", lw=1.3, label=f"coil z = {C[2]:g}")
        a1.semilogy(xv, B, color="#C0392B", lw=1.9)
        a1.set_ylabel("|B| [T]"); a1.grid(alpha=0.3, which="both")
        a1.set_title(f"|B| and SNR along '{name}'   |   f = {FREQ:g} Hz, BW = {BW:g} Hz")
        a2.plot(xv, common.snr_db(B, B_NOISE), color="#1F77B4", lw=1.9)
        a2.axhline(SNR_LINE, color="gray", ls="--", lw=1.0, label=f"{SNR_LINE:g} dB")
        a2.set_ylabel("SNR [dB]"); a2.set_xlabel(f"{'xyz'[axis]} [m]"); a2.grid(alpha=0.3)
        for a in (a1, a2):
            if a.get_legend_handles_labels()[0]:
                a.legend(loc="best", fontsize=8)
        fig.tight_layout()
        out = os.path.join(common.RESULTS, f"curve_{name}_B_SNR.png")
        fig.savefig(out, dpi=140); plt.close(fig)
        print("saved", out)


if __name__ == "__main__":
    for normal, plane, at in (("z", "XY", f"z = {C[2]:g}"), ("x", "YZ", f"x = {C[0]:g}"),
                              ("y", "XZ", f"y = {C[1]:g}")):
        cut(normal, f"SNR_cut_{plane}{SUFFIX}.png", f"SNR on the {plane} plane ({at})")
    view_3d()
    curve_plots()
    rows = common.sample_markers(cfg, mesh, env)
    if rows:
        b = np.array([r["Bamp"] for r in rows], float)
        common.print_markers(rows, extra={"V_ind [V]": K * b, "SNR [dB]": common.snr_db(b, B_NOISE)},
                             csv_path=os.path.join(common.RESULTS, "markers_SNR.csv"))
