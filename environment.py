"""environment.py -- read the block-model CSV (the simulation environment).

The CSV is a regular grid of blocks, one row per block:

    x, y, z, material_label, sigma, eps_r, mu_r

x, y, z are the block CENTRE in local coordinates (the site block is 0, 0, 0).
Blocks whose material_label equals config.yaml -> environment.rock_label are
rock; any other label is air.  Each rock block keeps its own sigma [S/m], mu_r
and eps_r: the rock gets the complex conductivity sigma + j*omega*eps0*eps_r
(conduction + displacement current, see formulation.pro).

Everything else the pipeline needs is derived from the CSV, with no manual step:
  * the domain box   = outer faces of the block grid,
  * the rock surface = top face of the highest rock block in each column,
  * rock flag, sigma, mu_r and eps_r per block (looked up for every mesh element).
The arrays are cached in results/environment.npz (rebuilt when the CSV changes).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

CACHE = os.path.join("results", "environment.npz")
CACHE_VERSION = 2          # bump when the cached arrays change (2: + eps_r)


@dataclass
class Environment:
    x: np.ndarray              # block-centre coordinates along each axis [m]
    y: np.ndarray
    z: np.ndarray
    dx: float
    dy: float
    dz: float
    rock: np.ndarray           # (nx, ny, nz) bool, True = rock block
    sigma: np.ndarray          # (nx, ny, nz) conductivity [S/m] (0 in air)
    mu_r: np.ndarray           # (nx, ny, nz) relative permeability (1 in air)
    eps_r: np.ndarray          # (nx, ny, nz) relative permittivity (1 in air)
    surface: np.ndarray        # (nx, ny) z of the rock top face per column [m]
    csv: str

    @property
    def bounds(self) -> tuple:
        """Domain box (xmin, ymin, zmin, xmax, ymax, zmax) = outer block faces."""
        return (float(self.x[0] - self.dx / 2), float(self.y[0] - self.dy / 2),
                float(self.z[0] - self.dz / 2), float(self.x[-1] + self.dx / 2),
                float(self.y[-1] + self.dy / 2), float(self.z[-1] + self.dz / 2))

    def index(self, px, py, pz=None):
        """Indices of the block(s) containing the point(s); clipped to the grid."""
        def idx(v, axis, d):
            return np.clip(np.rint((np.asarray(v, float) - axis[0]) / d).astype(int),
                           0, len(axis) - 1)
        i, j = idx(px, self.x, self.dx), idx(py, self.y, self.dy)
        return (i, j) if pz is None else (i, j, idx(pz, self.z, self.dz))

    def surface_at(self, px, py):
        """Stepped rock-top elevation under (px, py) (nearest block column)."""
        i, j = self.index(px, py)
        return self.surface[i, j]

    def is_rock(self, px, py, pz):
        i, j, k = self.index(px, py, pz)
        return self.rock[i, j, k]


def load(cfg: dict, verbose: bool = True) -> Environment:
    """Environment of config.yaml -> environment (from the cache when valid)."""
    ecfg = cfg["environment"]
    csv = str(ecfg["csv"])
    label = str(ecfg.get("rock_label", "rock"))
    if not os.path.exists(csv):
        raise SystemExit(f"environment CSV not found: {csv}  (config.yaml -> environment.csv)")
    key = (f"v{CACHE_VERSION}|{os.path.abspath(csv)}|{os.path.getsize(csv)}|"
           f"{os.path.getmtime(csv)}|{label}")
    if os.path.exists(CACHE):
        c = np.load(CACHE, allow_pickle=False)
        if str(c["key"]) == key:
            return Environment(c["x"], c["y"], c["z"], float(c["dx"]), float(c["dy"]),
                               float(c["dz"]), c["rock"], c["sigma"], c["mu_r"],
                               c["eps_r"], c["surface"], csv)
    env = _read_csv(csv, label, verbose)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, key=key, x=env.x, y=env.y, z=env.z, dx=env.dx, dy=env.dy,
                        dz=env.dz, rock=env.rock, sigma=env.sigma, mu_r=env.mu_r,
                        eps_r=env.eps_r, surface=env.surface)
    return env


def _axis(vals: np.ndarray, name: str) -> tuple:
    """Sorted block-centre coordinates of one axis + the (uniform) step."""
    u = np.unique(vals)
    if len(u) < 2:
        raise SystemExit(f"CSV: only one {name} value -- not a 3D block grid")
    d = float(np.median(np.diff(u)))
    if not np.allclose(u, u[0] + d * np.arange(len(u)), atol=1e-3 * d):
        raise SystemExit(f"CSV: the {name} coordinates are not a regular grid (step {d:g} m)")
    return u, d


def _read_csv(csv: str, label: str, verbose: bool) -> Environment:
    import pandas as pd
    df = pd.read_csv(csv, usecols=["x", "y", "z", "material_label", "sigma", "eps_r", "mu_r"])
    (x, dx), (y, dy), (z, dz) = (_axis(df[c].to_numpy(float), c) for c in "xyz")
    nx, ny, nz = len(x), len(y), len(z)
    i = np.rint((df["x"].to_numpy(float) - x[0]) / dx).astype(int)
    j = np.rint((df["y"].to_numpy(float) - y[0]) / dy).astype(int)
    k = np.rint((df["z"].to_numpy(float) - z[0]) / dz).astype(int)
    is_rock = (df["material_label"].astype(str) == label).to_numpy()
    if not is_rock.any():
        raise SystemExit(f"CSV: no block with material_label == {label!r} "
                         "(config.yaml -> environment.rock_label)")
    sig = df["sigma"].to_numpy(float)
    mu = df["mu_r"].to_numpy(float)
    eps = df["eps_r"].to_numpy(float)
    if np.any(~np.isfinite(sig[is_rock])) or np.any(sig[is_rock] < 0):
        raise SystemExit("CSV: rock sigma must be finite and >= 0")
    if np.any(~np.isfinite(eps[is_rock])) or np.any(eps[is_rock] < 1):
        raise SystemExit("CSV: rock eps_r must be finite and >= 1")

    rock = np.zeros((nx, ny, nz), bool)
    sigma = np.zeros((nx, ny, nz))
    mu_r = np.ones((nx, ny, nz))
    eps_r = np.ones((nx, ny, nz))
    rock[i, j, k] = is_rock
    r = (i[is_rock], j[is_rock], k[is_rock])
    sigma[r] = sig[is_rock]
    mu_r[r] = mu[is_rock]
    eps_r[r] = eps[is_rock]

    # rock surface: top face of the highest rock block of each column
    has = rock.any(axis=2)
    ktop = nz - 1 - np.argmax(rock[:, :, ::-1], axis=2)
    surface = np.where(has, z[ktop] + dz / 2.0, z[0] - dz / 2.0)

    env = Environment(x, y, z, dx, dy, dz, rock, sigma, mu_r, eps_r, surface, csv)
    if verbose:
        b = env.bounds
        sr = sigma[rock]
        print(f"[env] {csv}: {len(df):,} blocks = {nx} x {ny} x {nz} of "
              f"{dx:g} x {dy:g} x {dz:g} m, {int(rock.sum()):,} rock ('{label}')")
        print(f"[env] domain x[{b[0]:g}, {b[3]:g}] y[{b[1]:g}, {b[4]:g}] z[{b[2]:g}, {b[5]:g}] m;"
              f"  rock surface z[{surface.min():g}, {surface.max():g}] m")
        print(f"[env] rock sigma [{sr.min():.3g}, {sr.max():.3g}] S/m, "
              f"mu_r [{mu_r[rock].min():g}, {mu_r[rock].max():g}], "
              f"eps_r [{eps_r[rock].min():g}, {eps_r[rock].max():g}]")
        if len(df) != nx * ny * nz:
            print(f"[env] warning: {nx * ny * nz - len(df):,} grid positions have no row "
                  "in the CSV -> treated as air")
    return env


def coil_z_on_rock(env: Environment, cx: float, cy: float, R: float, r: float) -> tuple:
    """Elevation of a HORIZONTAL coil (major radius R, tube radius r) centred at
    (cx, cy) resting on the rock: tube bottom on the rock top under the centre.

    Raises SystemExit if the centre is outside the grid, or if the stepped rock
    under the ring (radii R-r, R, R+r) rises above the tube bottom, i.e. the coil
    would pierce neighbouring blocks.  Returns (z_coil, z_rock_centre, z_rock_ring_max)."""
    b = env.bounds
    if not (b[0] < cx < b[3] and b[1] < cy < b[4]):
        raise SystemExit(f"coil centre (x={cx:g}, y={cy:g}) is outside the block grid "
                         f"x[{b[0]:g}, {b[3]:g}] y[{b[1]:g}, {b[4]:g}]")
    z_centre = float(env.surface_at(cx, cy))
    z_coil = z_centre + r
    th = np.linspace(0.0, 2.0 * math.pi, 180, endpoint=False)
    rr = np.repeat([R - r, R, R + r], len(th))
    tt = np.tile(th, 3)
    ring = env.surface_at(cx + rr * np.cos(tt), cy + rr * np.sin(tt))
    k = int(np.argmax(ring))
    if ring[k] > z_coil - r + 1e-6:
        raise SystemExit(
            f"cannot place the coil at (x={cx:g}, y={cy:g}): resting on the rock under its "
            f"centre (top z={z_centre:g} m) the rock under the ring rises to z={ring[k]:g} m "
            f"at (x={cx + rr[k] * math.cos(tt[k]):.1f}, y={cy + rr[k] * math.sin(tt[k]):.1f}) "
            f"and would cross the coil. Pick a flatter site or a smaller major_radius.")
    return z_coil, z_centre, float(ring.max())
