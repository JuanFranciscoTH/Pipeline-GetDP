"""export_curves.py -- the field along straight lines, to CSV.

Samples results/B_field.vtu along every line of config.yaml -> curves -> lines
(p0 -> p1, n_points samples; a line may set its own n_points) and writes
results/curve_<name>.csv with the columns
    x, y, z [m], Bx_re, By_re, Bz_re, Bx_im, By_im, Bz_im, B_amp [T], SNR_dB
(phasor B(t) = Re[(B_re + j B_im) e^{j w t}]; SNR as in view_snr.py).
Points outside the domain are written as NaN.

Run:  python export_curves.py
"""
from __future__ import annotations

import os

import numpy as np
import pyvista as pv

import common

COLUMNS = ["x", "y", "z", "Bx_re", "By_re", "Bz_re", "Bx_im", "By_im", "Bz_im",
           "B_amp", "SNR_dB"]


def main() -> None:
    cfg = common.load_config()
    info = common.run_info(cfg)
    mesh = common.read_field()
    _, b_noise = common.noise_floor(cfg, common.receiver_gain(cfg, info["frequency"]))
    cc = cfg.get("curves") or {}
    lines = cc.get("lines") or []
    if not lines:
        raise SystemExit("no curves.lines in config.yaml -- nothing to export")
    for ln in lines:
        name = str(ln.get("name", "curve"))
        p0, p1 = np.asarray(ln["p0"], float), np.asarray(ln["p1"], float)
        n = int(ln.get("n_points", cc.get("n_points", 1000)))
        pts = p0 + np.linspace(0.0, 1.0, n)[:, None] * (p1 - p0)
        s = pv.PolyData(pts).sample(mesh)
        ok = np.asarray(s["vtkValidPointMask"], bool)
        re, im = np.asarray(s["B_re"], float), np.asarray(s["B_im"], float)
        re[~ok] = np.nan; im[~ok] = np.nan
        bamp = np.sqrt((re ** 2).sum(1) + (im ** 2).sum(1))
        M = np.column_stack([pts, re, im, bamp, common.snr_db(bamp, b_noise)])
        out = os.path.join(common.RESULTS, f"curve_{name}.csv")
        header = (f"curve '{name}' from {common.VTU}; f = {info['frequency']:g} Hz, "
                  f"N*I = {info['NI']:g}, coil centre {info['coil_center']}\n" + ",".join(COLUMNS))
        np.savetxt(out, M, delimiter=",", header=header, comments="# ")
        print(f"saved {out}  [{n} points, {p0} -> {p1}"
              + (f", {int((~ok).sum())} outside the domain]" if (~ok).any() else "]"))


if __name__ == "__main__":
    main()
