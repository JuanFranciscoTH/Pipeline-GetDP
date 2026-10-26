"""common.py -- helpers shared by run.py and the post-processing scripts
(view_*.py, export_curves.py): config, solved-run info, receiver/noise model,
plot window, plane sampling, plot overlays and the markers table."""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import yaml

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = "results"
VTU = os.path.join(RESULTS, "B_field.vtu")
RUN_INFO = os.path.join(RESULTS, "run_info.json")
COPPER = "#B87333"
# 3D framing: vertical colour bar on the right + camera zoomed out (no clipped corners)
BAR_3D = {"vertical": True, "position_x": 0.86, "position_y": 0.2, "width": 0.05,
          "height": 0.6, "title_font_size": 16, "label_font_size": 14}
ZOOM_3D = 0.85

# cosmetic warnings of pyvista/VTK in off-screen plots
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=r".*extract_surface.*")
try:
    from pyvista.core.errors import PyVistaDeprecationWarning, PyVistaFutureWarning
    warnings.filterwarnings("ignore", category=PyVistaFutureWarning)
    warnings.filterwarnings("ignore", category=PyVistaDeprecationWarning)
except Exception:
    pass


def load_config() -> dict:
    """config.yaml of the project folder (also makes it the working directory)."""
    os.chdir(PROJECT_DIR)
    with open("config.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def run_info(cfg: dict) -> dict:
    """Parameters of the SOLVED run (written by run.py): coil centre (with the
    computed z), radii, N*I, frequency.  Falls back to config.yaml if absent."""
    if os.path.exists(RUN_INFO):
        with open(RUN_INFO, encoding="utf-8") as fh:
            return json.load(fh)
    import environment
    c = cfg["coil"]
    cx, cy = (float(v) for v in c["center_xy"])
    z, _, _ = environment.coil_z_on_rock(environment.load(cfg, verbose=False), cx, cy,
                                         float(c["major_radius"]), float(c["minor_radius"]))
    return {"coil_center": [cx, cy, z], "major_radius": float(c["major_radius"]),
            "minor_radius": float(c["minor_radius"]),
            "NI": float(c["turns"]) * float(c["current"]), "frequency": float(c["frequency"])}


def read_field():
    """results/B_field.vtu with B_re, B_im (vectors) and Bamp = |B| amplitude [T]."""
    import pyvista as pv
    if not os.path.exists(VTU):
        raise SystemExit(f"{VTU} not found -- run the solve first: python run.py")
    mesh = pv.read(VTU)
    if "Bamp" not in mesh.point_data:
        re, im = (np.asarray(mesh.point_data[k], float) for k in ("B_re", "B_im"))
        mesh.point_data["Bamp"] = np.sqrt((re ** 2).sum(1) + (im ** 2).sum(1))
    return mesh


# --------------------------------------------------------------------------
#  Receiver + noise (config.yaml -> receiver, noise)
# --------------------------------------------------------------------------
def receiver_gain(cfg: dict, freq: float) -> float:
    """K = Q * N * omega * A  [V/T]: induced EMF amplitude per tesla, V = K |B|."""
    rx = cfg["receiver"]
    area = np.pi * (float(rx["diameter"]) / 2.0) ** 2
    return float(rx["quality_factor"]) * float(rx["turns"]) * 2.0 * np.pi * freq * area


def noise_floor(cfg: dict, K: float) -> tuple:
    """(V_noise [V], B_noise [T]): the measured noise level (dB per spectrum bin)
    converted to volts with the calibration, scaled to the detection bandwidth,
    and referred to field through the receiver: B_noise = V_noise / K.
    V_noise is an AMPLITUDE, like the induced voltage K*|B|: the spectrum scale is
    a tone amplitude, so an RMS calibration voltage (calibration_rms: true) is
    turned into its amplitude (x sqrt(2)) first."""
    n = cfg["noise"]
    v_cal = float(n["calibration_volts"]) * (np.sqrt(2.0) if n.get("calibration_rms", False) else 1.0)
    v = (v_cal
         * 10.0 ** ((float(n["level_db"]) - float(n["calibration_db"])) / 20.0)
         * np.sqrt(float(n["bandwidth_hz"]) / float(n["spectrum_bin_hz"])))
    return float(v), float(v / K)


def snr_db(B, B_noise):
    """SNR [dB] = 20 log10(|B| / B_noise), floored at -60 dB (NaN stays NaN)."""
    B = np.asarray(B, float)
    return 20.0 * np.log10(np.maximum(B, B_noise * 1e-3) / B_noise)


# --------------------------------------------------------------------------
#  Plot window (config.yaml -> plots; --full on the command line)
# --------------------------------------------------------------------------
def full_domain(cfg: dict) -> bool:
    return bool((cfg.get("plots") or {}).get("full_domain", False)) or "--full" in sys.argv


def window_box(cfg: dict, center, bounds, full: bool) -> tuple:
    """(x0, x1, y0, y1, z0, z1) drawn in the cuts AND the 3D view: the whole
    domain, or a cube of +- plots.window_half_size centred on the coil."""
    xmin, ymin, zmin, xmax, ymax, zmax = bounds
    if full:
        return (xmin, xmax, ymin, ymax, zmin, zmax)
    h = float((cfg.get("plots") or {}).get("window_half_size", 200.0))
    cx, cy, cz = center
    return (cx - h, cx + h, cy - h, cy + h, cz - h, cz + h)


def plane_ranges(normal: str, box: tuple) -> tuple:
    """(horizontal range, vertical range) of the cut with this normal."""
    x0, x1, y0, y1, z0, z1 = box
    return {"x": ((y0, y1), (z0, z1)), "y": ((x0, x1), (z0, z1)),
            "z": ((x0, x1), (y0, y1))}[normal]


def sample_plane(mesh, normal: str, center, box, key: str, n: int = 340):
    """Field `key` on an n x n grid of the cut plane through `center`."""
    import pyvista as pv
    (a0, a1), (b0, b1) = plane_ranges(normal, box)
    A, Bg = np.meshgrid(np.linspace(a0, a1, n), np.linspace(b0, b1, n))
    cx, cy, cz = center
    if normal == "x":
        pts = np.column_stack([np.full(A.size, cx), A.ravel(), Bg.ravel()])
    elif normal == "y":
        pts = np.column_stack([A.ravel(), np.full(A.size, cy), Bg.ravel()])
    else:
        pts = np.column_stack([A.ravel(), Bg.ravel(), np.full(A.size, cz)])
    s = pv.PolyData(pts).sample(mesh)
    v = np.asarray(s[key], float)
    v[~np.asarray(s["vtkValidPointMask"], bool)] = np.nan
    return A, Bg, v.reshape(A.shape)


def draw_overlay(ax, normal: str, center, R: float, env, a_range) -> tuple:
    """Coil outline + air/rock interface (dotted, stepped) on a cut.
    Returns the (x label, y label) of the cut."""
    import matplotlib.pyplot as plt
    cx, cy, cz = center
    if normal == "z":
        ax.add_patch(plt.Circle((cx, cy), R, fill=False, ec="white", lw=1.3, ls=":"))
        X, Y = np.meshgrid(env.x, env.y, indexing="ij")
        try:
            ax.contour(X, Y, env.surface, levels=[cz], colors="white",
                       linestyles=":", linewidths=1.8)
        except Exception:
            pass
        return "x [m]", "y [m]"
    h0 = cy if normal == "x" else cx
    ax.plot([h0 - R, h0 + R], [cz, cz], "o", mfc="none", mec="white", mew=1.8, ms=11)
    s = np.linspace(a_range[0], a_range[1], 1400)
    zt = env.surface_at(np.full_like(s, cx), s) if normal == "x" else \
        env.surface_at(s, np.full_like(s, cy))
    ax.plot(s, zt, color="white", lw=1.8, ls=":", alpha=0.95, drawstyle="steps-mid")
    return ("y [m]", "z [m]") if normal == "x" else ("x [m]", "z [m]")


def coil_tube(center, R: float, r: float):
    """The coil as a copper torus surface (for the 3D views)."""
    import pyvista as pv
    th = np.linspace(0, 2 * np.pi, 160)
    ring = np.column_stack([center[0] + R * np.cos(th), center[1] + R * np.sin(th),
                            np.full_like(th, center[2])])
    return pv.lines_from_points(ring, close=True).tube(radius=r, n_sides=16)


# --------------------------------------------------------------------------
#  Markers (config.yaml -> markers)
# --------------------------------------------------------------------------
def sample_markers(cfg: dict, mesh, env) -> list:
    """Field at every marker -> [{name, x, y, z, medium, Bamp, Bx, By, Bz}]
    (harmonic amplitudes [T]); NaN and medium 'outside' off the mesh."""
    import pyvista as pv
    mks = cfg.get("markers") or []
    if not mks:
        return []
    names = [str(m.get("name", f"M{i}")) for i, m in enumerate(mks, 1)]
    pts = np.array([[float(c) for c in m["xyz"]] for m in mks], float)
    s = pv.PolyData(pts).sample(mesh)
    ok = np.asarray(s["vtkValidPointMask"], bool)
    re, im = np.asarray(s["B_re"], float), np.asarray(s["B_im"], float)
    comp = np.sqrt(re ** 2 + im ** 2)                    # amplitude per component
    bamp = np.sqrt((comp ** 2).sum(1))
    comp[~ok] = np.nan; bamp[~ok] = np.nan
    rock = env.is_rock(pts[:, 0], pts[:, 1], pts[:, 2])
    return [{"name": nm, "x": p[0], "y": p[1], "z": p[2],
             "medium": "outside" if not ok[k] else ("rock" if rock[k] else "air"),
             "Bamp": bamp[k], "Bx": comp[k, 0], "By": comp[k, 1], "Bz": comp[k, 2]}
            for k, (nm, p) in enumerate(zip(names, pts))]


def print_markers(rows: list, extra: dict | None = None, csv_path: str | None = None):
    """Console table of the markers (+ `extra` = {header: [value per marker]})
    and the same table as CSV."""
    if not rows:
        return
    extra = extra or {}
    print("\nfield at the markers (config.yaml -> markers), harmonic amplitude:")
    print(f"  {'marker':16s} {'x':>8s} {'y':>8s} {'z':>8s} {'medium':>7s} "
          f"{'|B| [T]':>10s} {'|Bx|':>10s} {'|By|':>10s} {'|Bz|':>10s}"
          + "".join(f" {h:>11s}" for h in extra))
    for k, r in enumerate(rows):
        print(f"  {r['name']:16s} {r['x']:8.1f} {r['y']:8.1f} {r['z']:8.1f} {r['medium']:>7s} "
              f"{r['Bamp']:10.3e} {r['Bx']:10.3e} {r['By']:10.3e} {r['Bz']:10.3e}"
              + "".join(f" {v[k]:11.4g}" for v in extra.values()))
    if csv_path:
        import csv
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["name", "x", "y", "z", "medium", "B_amp_T", "Bx_amp_T", "By_amp_T",
                        "Bz_amp_T"] + list(extra))
            for k, r in enumerate(rows):
                w.writerow([r["name"], r["x"], r["y"], r["z"], r["medium"], r["Bamp"],
                            r["Bx"], r["By"], r["Bz"]] + [v[k] for v in extra.values()])
        print(f"  -> {csv_path}")
