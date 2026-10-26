from __future__ import annotations

# === Geoespacial ===
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
from pyproj import Transformer

# === Datos ===
import pandas as pd
import numpy as np
import duckdb

# === Visualización ===
import matplotlib.pyplot as plt
import pyvista as pv
from IPython.display import display

# === Archivos ===
from pathlib import Path

# === DXF / CAD ===
import ezdxf

# === Raster ===
import rasterio

# === Optimización ===
from scipy.optimize import differential_evolution, minimize

# === Utilidades ===
import math
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class AxisMode:
    name: str
    matrix: np.ndarray


AXIS_MODES: list[AxisMode] = [
    AxisMode("identity", np.array([[1.0, 0.0], [0.0, 1.0]])),
    AxisMode("flip_x", np.array([[-1.0, 0.0], [0.0, 1.0]])),
    AxisMode("flip_y", np.array([[1.0, 0.0], [0.0, -1.0]])),
    AxisMode("flip_xy", np.array([[-1.0, 0.0], [0.0, -1.0]])),
    AxisMode("swap_xy", np.array([[0.0, 1.0], [1.0, 0.0]])),
    AxisMode("swap_flip_x", np.array([[0.0, -1.0], [1.0, 0.0]])),
    AxisMode("swap_flip_y", np.array([[0.0, 1.0], [-1.0, 0.0]])),
    AxisMode("swap_flip_xy", np.array([[0.0, -1.0], [-1.0, 0.0]])),
]


def ensure_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def load_dxf_points(path: Path, sample_step: int = 40) -> tuple[np.ndarray, np.ndarray]:
    path = ensure_path(path)
    doc = ezdxf.readfile(path)
    msp = doc.modelspace()
    pts: list[tuple[float, float]] = []
    zs: list[float] = []
    for entity in msp:
        if entity.dxftype() != "POLYLINE":
            continue
        for idx, vertex in enumerate(entity.vertices):
            if idx % sample_step != 0:
                continue
            loc = vertex.dxf.location
            pts.append((loc.x, loc.y))
            zs.append(loc.z)
    if not pts:
        raise ValueError(f"No POLYLINE vertices found in {path}.")
    return np.array(pts, dtype=float), np.array(zs, dtype=float)


def load_dxf_plan_view(path: Path, sample_step: int = 20) -> tuple[list[np.ndarray], np.ndarray]:
    path = ensure_path(path)
    doc = ezdxf.readfile(path)
    msp = doc.modelspace()
    segments: list[np.ndarray] = []
    all_points: list[tuple[float, float]] = []
    for entity in msp:
        if entity.dxftype() != "POLYLINE":
            continue
        pts: list[tuple[float, float]] = []
        for idx, vertex in enumerate(entity.vertices):
            loc = vertex.dxf.location
            all_points.append((loc.x, loc.y))
            if idx % sample_step == 0:
                pts.append((loc.x, loc.y))
        if len(pts) >= 2:
            segments.append(np.array(pts, dtype=float))
    if not all_points:
        raise ValueError(f"No POLYLINE vertices found in {path}.")
    return segments, np.array(all_points, dtype=float)


def read_dem(path: Path) -> dict[str, Any]:
    path = ensure_path(path)
    with rasterio.open(path) as ds:
        return {
            "array": ds.read(1),
            "transform": ds.transform,
            "nodata": ds.nodata,
            "bounds": ds.bounds,
            "crs": ds.crs,
        }


def bilinear_sample(
    dem: np.ndarray,
    transform: rasterio.Affine,
    nodata: float | None,
    xs: np.ndarray,
    ys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    inv = ~transform
    cols, rows = inv * (xs, ys)
    valid = (
        (cols >= 0)
        & (rows >= 0)
        & (cols < dem.shape[1] - 1)
        & (rows < dem.shape[0] - 1)
    )
    vals = np.full(xs.shape, np.nan, dtype=float)
    if not np.any(valid):
        return vals, valid

    c = cols[valid]
    r = rows[valid]
    c0 = np.floor(c).astype(int)
    r0 = np.floor(r).astype(int)
    dc = c - c0
    dr = r - r0

    z00 = dem[r0, c0]
    z10 = dem[r0, c0 + 1]
    z01 = dem[r0 + 1, c0]
    z11 = dem[r0 + 1, c0 + 1]

    if nodata is not None:
        valid_local = (z00 != nodata) & (z10 != nodata) & (z01 != nodata) & (z11 != nodata)
    else:
        valid_local = np.ones_like(z00, dtype=bool)

    interp = (
        z00 * (1 - dc) * (1 - dr)
        + z10 * dc * (1 - dr)
        + z01 * (1 - dc) * dr
        + z11 * dc * dr
    )
    tmp = vals[valid]
    tmp[valid_local] = interp[valid_local]
    vals[valid] = tmp

    valid2 = valid.copy()
    valid2[valid] = valid_local
    return vals, valid2


def rotate(theta_deg: float) -> np.ndarray:
    theta = math.radians(theta_deg)
    return np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=float,
    )


def transform_points(
    xy_centered: np.ndarray,
    axis_matrix: np.ndarray,
    theta_deg: float,
    tx: float,
    ty: float,
    scale: float,
) -> np.ndarray:
    rot = rotate(theta_deg)
    transformed = scale * (xy_centered @ axis_matrix.T) @ rot.T
    transformed[:, 0] += tx
    transformed[:, 1] += ty
    return transformed


def make_objective(
    xy_centered: np.ndarray,
    z_local: np.ndarray,
    dem: np.ndarray,
    dem_transform: rasterio.Affine,
    nodata: float | None,
    axis_matrix: np.ndarray,
    allow_scale: bool,
    min_valid_fraction: float = 0.4,
) -> Any:
    def objective(params: np.ndarray) -> float:
        if allow_scale:
            theta_deg, tx, ty, scale = params
        else:
            theta_deg, tx, ty = params
            scale = 1.0

        xy = transform_points(xy_centered, axis_matrix, theta_deg, tx, ty, scale)
        dem_z, valid = bilinear_sample(dem, dem_transform, nodata, xy[:, 0], xy[:, 1])
        valid_count = int(valid.sum())
        if valid_count < max(100, int(min_valid_fraction * len(valid))):
            return 1e9

        residuals = dem_z[valid] - z_local[valid]
        rms = float(np.sqrt(np.mean(residuals**2)))
        coverage_penalty = 5000.0 * (1.0 - valid_count / len(valid))
        return rms + coverage_penalty

    return objective


def search_mode(
    mode: AxisMode,
    xy_centered: np.ndarray,
    z_local: np.ndarray,
    bounds_xy: tuple[float, float, float, float],
    dem: np.ndarray,
    dem_transform: rasterio.Affine,
    nodata: float | None,
    allow_scale: bool = False,
    seed: int = 42,
    de_maxiter: int = 40,
) -> tuple[float, np.ndarray]:
    left, bottom, right, top = bounds_xy
    objective = make_objective(
        xy_centered=xy_centered,
        z_local=z_local,
        dem=dem,
        dem_transform=dem_transform,
        nodata=nodata,
        axis_matrix=mode.matrix,
        allow_scale=allow_scale,
    )
    bounds = [(-180.0, 180.0), (left, right), (bottom, top)]
    if allow_scale:
        bounds.append((0.98, 1.02))

    result_de = differential_evolution(
        objective,
        bounds=bounds,
        seed=seed,
        maxiter=de_maxiter,
        popsize=10,
        polish=False,
        updating="deferred",
        workers=1,
    )
    result_local = minimize(
        objective,
        x0=result_de.x,
        method="Powell",
        options={"maxiter": 200},
    )
    return float(result_local.fun), np.array(result_local.x, dtype=float)


def affine_coefficients_from_result(
    centroid: np.ndarray,
    axis_matrix: np.ndarray,
    theta_deg: float,
    tx: float,
    ty: float,
    scale: float,
) -> dict[str, float]:
    rot = rotate(theta_deg)
    matrix = scale * axis_matrix.T @ rot.T

    p0 = centroid
    p1 = centroid + np.array([1.0, 0.0])
    p2 = centroid + np.array([0.0, 1.0])

    def to_utm(point: np.ndarray) -> np.ndarray:
        return (point - centroid) @ matrix + np.array([tx, ty])

    q0 = to_utm(p0)
    q1 = to_utm(p1)
    q2 = to_utm(p2)

    a = q1[0] - q0[0]
    b = q2[0] - q0[0]
    d = q1[1] - q0[1]
    e = q2[1] - q0[1]
    c = q0[0] - a * p0[0] - b * p0[1]
    f = q0[1] - d * p0[0] - e * p0[1]
    return {"a": float(a), "b": float(b), "c": float(c), "d": float(d), "e": float(e), "f": float(f)}


def run_dem_fit(
    dxf_path: str | Path,
    dem_path: str | Path,
    sample_step: int = 60,
    de_maxiter: int = 15,
    seed: int = 42,
    allow_scale: bool = True,
) -> dict[str, Any]:
    dxf_path = ensure_path(dxf_path)
    dem_path = ensure_path(dem_path)
    xy_local, z_local = load_dxf_points(dxf_path, sample_step=sample_step)
    centroid = xy_local.mean(axis=0)
    xy_centered = xy_local - centroid

    dem_info = read_dem(dem_path)
    bounds_xy = (
        dem_info["bounds"].left,
        dem_info["bounds"].bottom,
        dem_info["bounds"].right,
        dem_info["bounds"].top,
    )

    best_score = float("inf")
    best_mode = None
    best_params = None
    mode_results: list[dict[str, Any]] = []

    for idx, mode in enumerate(AXIS_MODES, start=1):
        score, params = search_mode(
            mode=mode,
            xy_centered=xy_centered,
            z_local=z_local,
            bounds_xy=bounds_xy,
            dem=dem_info["array"],
            dem_transform=dem_info["transform"],
            nodata=dem_info["nodata"],
            allow_scale=allow_scale,
            seed=seed + idx,
            de_maxiter=de_maxiter,
        )
        mode_results.append({"mode": mode, "score": score, "params": params})
        if score < best_score:
            best_score = score
            best_mode = mode
            best_params = params

    if best_mode is None or best_params is None:
        raise RuntimeError("No valid DXF/DEM fit found.")

    if allow_scale:
        theta_deg, tx, ty, scale = best_params
    else:
        theta_deg, tx, ty = best_params
        scale = 1.0

    xy_utm = transform_points(xy_centered, best_mode.matrix, theta_deg, tx, ty, scale)
    dem_z, valid = bilinear_sample(
        dem_info["array"], dem_info["transform"], dem_info["nodata"], xy_utm[:, 0], xy_utm[:, 1]
    )
    residuals = dem_z[valid] - z_local[valid]
    coef = affine_coefficients_from_result(centroid, best_mode.matrix, theta_deg, tx, ty, scale)

    return {
        "xy_local": xy_local,
        "z_local": z_local,
        "centroid": centroid,
        "xy_centered": xy_centered,
        "dem_info": dem_info,
        "mode_results": mode_results,
        "best_mode": best_mode,
        "best_score": float(best_score),
        "best_params": np.array(best_params, dtype=float),
        "theta_deg": float(theta_deg),
        "tx": float(tx),
        "ty": float(ty),
        "scale": float(scale),
        "xy_utm": xy_utm,
        "valid": valid,
        "residuals": residuals,
        "rms_z": float(np.sqrt(np.mean(residuals**2))),
        "mae_z": float(np.mean(np.abs(residuals))),
        "coef": coef,
    }


def make_objective_fixed_center(
    xy_centered: np.ndarray,
    z_local: np.ndarray,
    dem: np.ndarray,
    dem_transform: rasterio.Affine,
    nodata: float | None,
    axis_matrix: np.ndarray,
    center_x: float,
    center_y: float,
    min_valid_fraction: float = 0.5,
) -> Any:
    def objective(params: np.ndarray) -> float:
        theta_deg, scale = params
        xy = transform_points(xy_centered, axis_matrix, theta_deg, center_x, center_y, scale)
        dem_z, valid = bilinear_sample(dem, dem_transform, nodata, xy[:, 0], xy[:, 1])
        valid_count = int(valid.sum())
        if valid_count < max(100, int(min_valid_fraction * len(valid))):
            return 1e9

        residuals = dem_z[valid] - z_local[valid]
        rms = float(np.sqrt(np.mean(residuals**2)))
        coverage_penalty = 5000.0 * (1.0 - valid_count / len(valid))
        return rms + coverage_penalty

    return objective


def run_fixed_center_fit(
    dxf_path: str | Path,
    dem_path: str | Path,
    lat: float,
    lon: float,
    sample_step: int = 60,
    de_maxiter: int = 15,
    seed: int = 42,
    scale_bounds: tuple[float, float] = (0.90, 1.10),
) -> dict[str, Any]:
    dxf_path = ensure_path(dxf_path)
    dem_path = ensure_path(dem_path)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32719", always_xy=True)
    center_x, center_y = transformer.transform(lon, lat)

    xy_local, z_local = load_dxf_points(dxf_path, sample_step=sample_step)
    centroid = xy_local.mean(axis=0)
    xy_centered = xy_local - centroid
    dem_info = read_dem(dem_path)

    best_score = float("inf")
    best_mode = None
    best_theta = None
    best_scale = None
    mode_results: list[dict[str, Any]] = []

    for idx, mode in enumerate(AXIS_MODES, start=1):
        objective = make_objective_fixed_center(
            xy_centered=xy_centered,
            z_local=z_local,
            dem=dem_info["array"],
            dem_transform=dem_info["transform"],
            nodata=dem_info["nodata"],
            axis_matrix=mode.matrix,
            center_x=center_x,
            center_y=center_y,
        )
        result_de = differential_evolution(
            objective,
            bounds=[(-180.0, 180.0), scale_bounds],
            seed=seed + idx,
            maxiter=de_maxiter,
            popsize=10,
            polish=False,
            updating="deferred",
            workers=1,
        )
        result_local = minimize(
            objective,
            x0=result_de.x,
            method="Powell",
            options={"maxiter": 200},
        )
        score = float(result_local.fun)
        theta_deg, scale = result_local.x
        mode_results.append({"mode": mode, "score": score, "theta_deg": theta_deg, "scale": scale})
        if score < best_score:
            best_score = score
            best_mode = mode
            best_theta = float(theta_deg)
            best_scale = float(scale)

    if best_mode is None or best_theta is None or best_scale is None:
        raise RuntimeError("No valid fixed-center DXF/DEM fit found.")

    xy_utm = transform_points(xy_centered, best_mode.matrix, best_theta, center_x, center_y, best_scale)
    dem_z, valid = bilinear_sample(
        dem_info["array"], dem_info["transform"], dem_info["nodata"], xy_utm[:, 0], xy_utm[:, 1]
    )
    residuals = dem_z[valid] - z_local[valid]
    coef = affine_coefficients_from_result(centroid, best_mode.matrix, best_theta, center_x, center_y, best_scale)

    return {
        "xy_local": xy_local,
        "z_local": z_local,
        "centroid": centroid,
        "xy_centered": xy_centered,
        "dem_info": dem_info,
        "center_utm": (float(center_x), float(center_y)),
        "mode_results": mode_results,
        "best_mode": best_mode,
        "best_score": float(best_score),
        "theta_deg": float(best_theta),
        "scale": float(best_scale),
        "xy_utm": xy_utm,
        "valid": valid,
        "residuals": residuals,
        "rms_z": float(np.sqrt(np.mean(residuals**2))),
        "mae_z": float(np.mean(np.abs(residuals))),
        "coef": coef,
    }


def coeff_to_lines(coef: dict[str, float], prefix: str = "") -> list[str]:
    label = f"{prefix} " if prefix else ""
    return [
        f"{label}UTM_E = {coef['a']} * X_local + {coef['b']} * Y_local + {coef['c']}",
        f"{label}UTM_N = {coef['d']} * X_local + {coef['e']} * Y_local + {coef['f']}",
    ]


def local_to_utm_xy(x: float, y: float, coef: dict[str, float]) -> tuple[float, float]:
    utm_e = coef["a"] * x + coef["b"] * y + coef["c"]
    utm_n = coef["d"] * x + coef["e"] * y + coef["f"]
    return utm_e, utm_n


def utm_a_local_xy(utm_e: float, utm_n: float, coef: dict[str, float]) -> tuple[float, float]:
    """Inversa de local_to_utm_xy: UTM 19S (E, N) -> coordenadas locales de la mina.

    Resuelve [a b; d e] @ [x; y] = [E - c; N - f]  (la misma transformación afín)."""
    a, b, c = coef["a"], coef["b"], coef["c"]
    d, e, f = coef["d"], coef["e"], coef["f"]
    det = a * e - b * d
    du, dv = utm_e - c, utm_n - f
    return (e * du - b * dv) / det, (a * dv - d * du) / det


def transform_polyline(entity: Any, coef: dict[str, float]) -> None:
    for vertex in entity.vertices:
        loc = vertex.dxf.location
        new_x, new_y = local_to_utm_xy(loc.x, loc.y, coef)
        vertex.dxf.location = (new_x, new_y, loc.z)


def transform_lwpolyline(entity: Any, coef: dict[str, float]) -> None:
    points = []
    for point in entity.get_points():
        new_x, new_y = local_to_utm_xy(point[0], point[1], coef)
        updated = list(point)
        updated[0] = new_x
        updated[1] = new_y
        points.append(tuple(updated))
    entity.set_points(points, format="xyseb")


def transform_line(entity: Any, coef: dict[str, float]) -> None:
    start = entity.dxf.start
    end = entity.dxf.end
    sx, sy = local_to_utm_xy(start.x, start.y, coef)
    ex, ey = local_to_utm_xy(end.x, end.y, coef)
    entity.dxf.start = (sx, sy, start.z)
    entity.dxf.end = (ex, ey, end.z)


def transform_point_like(entity: Any, attr_name: str, coef: dict[str, float]) -> None:
    loc = getattr(entity.dxf, attr_name)
    new_x, new_y = local_to_utm_xy(loc.x, loc.y, coef)
    setattr(entity.dxf, attr_name, (new_x, new_y, loc.z))


def georeference_dxf(
    input_path: str | Path,
    output_path: str | Path,
    coef: dict[str, float],
) -> tuple[int, dict[str, int]]:
    input_path = ensure_path(input_path)
    output_path = ensure_path(output_path)
    doc = ezdxf.readfile(input_path)
    msp = doc.modelspace()
    transformed = 0
    skipped: dict[str, int] = {}

    for entity in msp:
        etype = entity.dxftype()
        if etype == "POLYLINE":
            transform_polyline(entity, coef)
            transformed += 1
        elif etype == "LWPOLYLINE":
            transform_lwpolyline(entity, coef)
            transformed += 1
        elif etype == "LINE":
            transform_line(entity, coef)
            transformed += 1
        elif etype == "POINT":
            transform_point_like(entity, "location", coef)
            transformed += 1
        elif etype in {"TEXT", "MTEXT", "INSERT"}:
            transform_point_like(entity, "insert", coef)
            transformed += 1
        else:
            skipped[etype] = skipped.get(etype, 0) + 1

    doc.header["$INSUNITS"] = 6
    doc.saveas(output_path)
    return transformed, skipped


def infer_block_resolution(df: pd.DataFrame) -> tuple[float, float]:
    x_vals = np.sort(df["EASTING"].dropna().unique())
    y_vals = np.sort(df["NORTHING"].dropna().unique())
    dx_vals = np.diff(x_vals)
    dy_vals = np.diff(y_vals)
    dx_vals = dx_vals[dx_vals > 0]
    dy_vals = dy_vals[dy_vals > 0]
    dx = float(pd.Series(np.round(dx_vals, 6)).mode().iloc[0])
    dy = float(pd.Series(np.round(dy_vals, 6)).mode().iloc[0])
    return dx, dy


def block_polygon_utm(
    x: float,
    y: float,
    coef: dict[str, float],
    dx: float,
    dy: float,
) -> Polygon:
    half_x = dx / 2.0
    half_y = dy / 2.0
    corners_local = [
        (x - half_x, y - half_y),
        (x + half_x, y - half_y),
        (x + half_x, y + half_y),
        (x - half_x, y + half_y),
        (x - half_x, y - half_y),
    ]
    corners_utm = [local_to_utm_xy(px, py, coef) for px, py in corners_local]
    return Polygon(corners_utm)


def load_top_blocks_with_duckdb(
    csv_path: str | Path,
    material_columns: list[str] | None = None,
) -> pd.DataFrame:
    import duckdb

    csv_path = ensure_path(csv_path)
    columns = material_columns
    csv_posix = csv_path.as_posix()
    filters = " OR ".join(
        [
            "CLASE <> -1",
            "CUTMP <> -1",
            "AU <> -1",
            "AG <> -1",
            "ZN <> -1",
            "MIN <> 0",
        ]
    )
    query = f"""
    WITH base AS (
        SELECT *
        FROM read_csv_auto('{csv_posix}', HEADER=TRUE)
        WHERE {filters}
    ),
    ranked AS (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY EASTING, NORTHING
                ORDER BY ELEVATION DESC
            ) AS rn,
            COUNT(*) OVER (
                PARTITION BY EASTING, NORTHING
            ) AS n_blocks
        FROM base
    )
    SELECT *
    FROM ranked
    WHERE rn = 1
    """
    _ = columns  # keeps signature explicit for notebook use
    return duckdb.query(query).df()


def export_top_blocks_gpkg(
    csv_path: str | Path,
    output_gpkg: str | Path,
    layer_name: str,
    coef: dict[str, float],
    crs: str = "EPSG:32719",
) -> gpd.GeoDataFrame:
    csv_path = ensure_path(csv_path)
    output_gpkg = ensure_path(output_gpkg)
    df_top = load_top_blocks_with_duckdb(csv_path)
    dx, dy = infer_block_resolution(df_top)

    utm_coords = [local_to_utm_xy(x, y, coef) for x, y in zip(df_top["EASTING"], df_top["NORTHING"])]
    df_top["utm_e"] = [coord[0] for coord in utm_coords]
    df_top["utm_n"] = [coord[1] for coord in utm_coords]

    geometries = [
        block_polygon_utm(x, y, coef, dx=dx, dy=dy)
        for x, y in zip(df_top["EASTING"], df_top["NORTHING"])
    ]

    if "rn" in df_top.columns:
        df_top = df_top.drop(columns=["rn"])
    df_export = df_top.rename(columns={column: column.lower() for column in df_top.columns})

    gdf = gpd.GeoDataFrame(df_export, geometry=geometries, crs=crs)
    gdf.to_file(output_gpkg, layer=layer_name, driver="GPKG")
    return gdf


def print_dem_fit_report(result: dict[str, Any]) -> None:
    xy_local = result["xy_local"]
    centroid = result["centroid"]
    dem_info = result["dem_info"]
    best_mode = result["best_mode"]
    xy_utm = result["xy_utm"]
    valid = result["valid"]
    print(f"DXF points used: {len(xy_local)}")
    print(f"Local centroid: x={centroid[0]:.3f}, y={centroid[1]:.3f}")
    print(f"DEM CRS: {dem_info['crs']}")
    print(f"DEM bounds: {dem_info['bounds']}")
    for mode_result in result["mode_results"]:
        params = mode_result.get("params")
        if params is not None:
            print(f"mode={mode_result['mode'].name} score={mode_result['score']:.3f} params={params.tolist()}")
        else:
            print(
                f"mode={mode_result['mode'].name} score={mode_result['score']:.3f} "
                f"theta={mode_result['theta_deg']:.6f} scale={mode_result['scale']:.9f}"
            )
    print()
    print("Best solution")
    print(f"axis_mode: {best_mode.name}")
    if "tx" in result:
        print(f"theta_deg: {result['theta_deg']}")
        print(f"tx: {result['tx']}")
        print(f"ty: {result['ty']}")
    else:
        print(f"theta_deg: {result['theta_deg']}")
    print(f"scale: {result['scale']}")
    print(f"score: {result['best_score']}")
    print(f"valid_points: {int(valid.sum())}/{len(valid)}")
    print(f"rms_z: {result['rms_z']}")
    print(f"mae_z: {result['mae_z']}")
    print(
        "utm_bbox: "
        f"left={xy_utm[:,0].min()}, right={xy_utm[:,0].max()}, "
        f"bottom={xy_utm[:,1].min()}, top={xy_utm[:,1].max()}"
    )
    print()
    print("Formula directa:")
    for line in coeff_to_lines(result["coef"]):
        print(line)


def ejecutar_ajuste_libre(
    dxf_path: str | Path,
    dem_path: str | Path,
    sample_step: int = 60,
    de_maxiter: int = 15,
    seed: int = 42,
    allow_scale: bool = True,
) -> dict[str, Any]:
    result = run_dem_fit(
        dxf_path=dxf_path,
        dem_path=dem_path,
        sample_step=sample_step,
        de_maxiter=de_maxiter,
        seed=seed,
        allow_scale=allow_scale,
    )
    print_dem_fit_report(result)
    return result


def ejecutar_ajuste_centro_fijado(
    lat: float,
    lon: float,
    dxf_path: str | Path,
    dem_path: str | Path,
    sample_step: int = 60,
    de_maxiter: int = 15,
    seed: int = 42,
) -> dict[str, Any]:
    result = run_fixed_center_fit(
        dxf_path=dxf_path,
        dem_path=dem_path,
        lat=lat,
        lon=lon,
        sample_step=sample_step,
        de_maxiter=de_maxiter,
        seed=seed,
    )
    center_e, center_n = result["center_utm"]
    print("Centro conocido en UTM 19S:")
    print("Easting:", center_e)
    print("Northing:", center_n)
    print_dem_fit_report(result)
    return result


def plot_centro_dxf(
    dxf_path: str | Path,
    sample_step: int = 20,
    figsize: tuple[int, int] = (12, 12),
) -> dict[str, Any]:
    segments, points = load_dxf_plan_view(dxf_path, sample_step=sample_step)
    center_x = points[:, 0].mean()
    center_y = points[:, 1].mean()
    print("Centro del DXF original (coordenadas locales):")
    print("X centro:", center_x)
    print("Y centro:", center_y)

    fig, ax = plt.subplots(figsize=figsize)
    for seg in segments:
        ax.plot(seg[:, 0], seg[:, 1], color="steelblue", linewidth=0.4, alpha=0.8)
    ax.scatter(center_x, center_y, color="red", s=180, marker="*", label="Centro del DXF", zorder=5)
    ax.annotate(
        f"Centro\nX={center_x:.2f}\nY={center_y:.2f}",
        xy=(center_x, center_y),
        xytext=(15, 15),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.9),
        color="red",
    )
    ax.set_aspect("equal")
    ax.set_xlabel("X local")
    ax.set_ylabel("Y local")
    ax.set_title("DXF original")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
#    display(fig)
    plt.show()


def exportar_dxf_georreferenciado(
    coef: dict[str, float],
    input_path: str | Path,
    output_path: str | Path,
) -> tuple[int, dict[str, int]]:
    transformed, skipped = georeference_dxf(input_path, output_path, coef)
    print("Archivo salida:", ensure_path(output_path))
    print("Entidades transformadas:", transformed)
    print("Entidades omitidas:", skipped)
    print("CRS esperado en QGIS: EPSG:32719")
    return transformed, skipped


def exportar_bloques_identificar(
    coef: dict[str, float],
    csv_path: str | Path,
    output_gpkg: str | Path,
    layer_name: str,
) -> gpd.GeoDataFrame:
    print("Leyendo bloque superior desde CSV y exportando GeoPackage...")
    gdf = export_top_blocks_gpkg(
        csv_path=csv_path,
        output_gpkg=output_gpkg,
        layer_name=layer_name,
        coef=coef,
    )
    print("Listo.")
    print("Salida:", ensure_path(output_gpkg))
    print("Capa:", layer_name)
    print("Features:", len(gdf))
    print("Columnas:", len(gdf.columns))
    return gdf


def exportar_antena_con_halos(
    x: float,
    y: float,
    output_gpkg: str | Path,
    halos_radios_m: list[float],
    layer_name: str = "antena_cobertura",
    coord_type: str = "local",
    coef: dict[str, float] | None = None,
    crs: str = "EPSG:32719"
) -> gpd.GeoDataFrame:
    """
    Crea una capa espacial con un punto central (antena) y n halos cuadrados concéntricos.
    
    Parámetros:
    - x, y: Coordenadas de la antena.
    - output_gpkg: Ruta del archivo GeoPackage de salida.
    - halos_radios_m: Lista con las distancias (en metros) para cada halo cuadrado (ej: [50, 100]).
    - layer_name: Nombre de la capa dentro del GeoPackage.
    - coord_type: 'local' (coordenadas mina) o 'global' (coordenadas UTM).
    - coef: Diccionario con coeficientes afines (requerido si coord_type es 'local').
    """
    # 1. Resolución del sistema de coordenadas
    if coord_type.lower() == "local":
        if coef is None:
            raise ValueError("Se requieren los 'coef' calculados previamente para transformar coordenadas locales a UTM.")
        utm_x, utm_y = local_to_utm_xy(x, y, coef)
    elif coord_type.lower() in ["global", "utm"]:
        utm_x, utm_y = x, y
    else:
        raise ValueError("coord_type debe ser 'local' o 'global'.")

    records = []

    # 2. Generar el punto central (La Antena)
    records.append({
        "tipo_geometria": "Antena Central",
        "distancia_m": 0.0,
        "geometry": Point(utm_x, utm_y)
    })

    # 3. Generar los halos cuadrados (Líneas)
    for radio in halos_radios_m:
        # Un cuadrado de "radio" r significa que desde el centro hay 'r' metros hacia N, S, E, O.
        # Por lo tanto, el lado total del cuadrado es 2*r.
        esquinas = [
            (utm_x - radio, utm_y + radio), # Noroeste
            (utm_x + radio, utm_y + radio), # Noreste
            (utm_x + radio, utm_y - radio), # Sureste
            (utm_x - radio, utm_y - radio), # Suroeste
            (utm_x - radio, utm_y + radio)  # Cierre al Noroeste
        ]
        records.append({
            "tipo_geometria": f"Halo Cuadrado {radio}m",
            "distancia_m": float(radio),
            "geometry": LineString(esquinas)
        })

    # 4. Construir GeoDataFrame y Exportar
    df = pd.DataFrame(records)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=crs)
    
    output_path = ensure_path(output_gpkg)
    gdf.to_file(output_path, layer=layer_name, driver="GPKG")
    
    print(f"Capa '{layer_name}' guardada exitosamente en: {output_path}")
    print(f"Punto central (UTM): E={utm_x:.2f}, N={utm_y:.2f}")
    
    return gdf
    

def consultar_bloque_especifico(
    csv_path: str | Path,
    x: float,
    y: float,
    z: float,
    search_radius_m: float = 20.0,
    coord_type: str = "local",
    coef: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Busca el bloque más cercano a una coordenada (X, Y, Z) dada y extrae 
    sus variables geoquímicas y categóricas.
    """
    # 1. Transformación a coordenadas locales de la mina (si es necesario)
    if coord_type.lower() in ["global", "utm"]:
        if coef is None:
            raise ValueError("Se requieren los 'coef' para transformar UTM a Local.")
        # Reutilizamos la función inversa que definimos en el paso anterior
        x_local, y_local = utm_a_local_xy(x, y, coef)
        z_local = z  # Asumimos que la elevación no sufre deformación afín
    else:
        x_local, y_local, z_local = x, y, z
        
    csv_posix = ensure_path(csv_path).as_posix()
    
    # 2. Consulta SQL: Filtro de caja espacial + Distancia Euclidiana 3D
    query = f"""
    WITH cajita_espacial AS (
        SELECT 
            EASTING, NORTHING, ELEVATION, 
            AU, AG, ZN, CUTMP, CLASE, MIN,
            (EASTING - {x_local}) AS dx,
            (NORTHING - {y_local}) AS dy,
            (ELEVATION - {z_local}) AS dz
        FROM read_csv_auto('{csv_posix}', HEADER=TRUE)
        -- Filtro crudo para no leer toda la mina (muy rápido)
        WHERE EASTING BETWEEN {x_local - search_radius_m} AND {x_local + search_radius_m}
          AND NORTHING BETWEEN {y_local - search_radius_m} AND {y_local + search_radius_m}
          AND ELEVATION BETWEEN {z_local - search_radius_m} AND {z_local + search_radius_m}
    )
    SELECT 
        EASTING, NORTHING, ELEVATION, 
        AU, AG, ZN, CUTMP, CLASE, MIN,
        -- Cálculo de la hipotenusa en 3D (Distancia real)
        sqrt(pow(dx, 2) + pow(dy, 2) + pow(dz, 2)) AS DISTANCIA_M
    FROM cajita_espacial
    ORDER BY DISTANCIA_M ASC
    LIMIT 1
    """
    
    df = duckdb.query(query).df()
    
    if df.empty:
        print(f"No se encontró ningún bloque a menos de {search_radius_m}m de la coordenada solicitada.")
    
    return df


def bloque_superficie(
    csv_path: str | Path,
    dem_path: str | Path,
    coef: dict[str, float],
    x: float | None = None,
    y: float | None = None,
    coord_type: str = "local",
    lat: float | None = None,
    lon: float | None = None,
    dx: float = 12.5,
    dy: float = 12.5,
    dz: float = 15.0,
    n_bloques: int | None = None,
) -> dict[str, Any]:
    """
    Bloque de SUPERFICIE de la columna más cercana a un punto -> centro de la geometría.

    Usa EXACTAMENTE el criterio de clasificar_bloques_topografia: un bloque es roca si
    su base (ELEVATION - dz/2) queda bajo la topografía del DEM. El bloque de superficie
    es el MÁS ALTO de la columna que cumple eso, así que en el CSV del simulador ese
    bloque sale 'roca', el de encima 'aire', y queda en z = 0 del sistema relativo.

    Punto de entrada (elige uno):
        coord_type="local"  -> x, y = EASTING, NORTHING de la mina (modelo de bloques)
        coord_type="utm"    -> x, y = UTM 19S (EPSG:32719)
        coord_type="latlon" -> lat, lon (WGS84)
    n_bloques : si se da, informa cuántos metros de modelo hay hacia cada lado
                (si en Z no caben n_bloques no importa: lo relevante es la superficie).

    Retorna dict: easting, northing, elevation (centro del bloque de superficie),
    topografia, techo_roca, utm_e, utm_n, lat, lon, vecinos (3x3), plano, margenes.
    """
    # 1. Punto -> coordenadas locales de la mina
    ct = coord_type.lower()
    if ct == "latlon":
        if lat is None or lon is None:
            raise ValueError("coord_type='latlon' requiere lat y lon.")
        x, y = Transformer.from_crs("EPSG:4326", "EPSG:32719", always_xy=True).transform(lon, lat)
        ct = "utm"
    if ct in ("utm", "global"):
        x, y = utm_a_local_xy(x, y, coef)
    elif ct != "local":
        raise ValueError("coord_type debe ser 'local', 'utm' o 'latlon'.")
    x, y = float(x), float(y)

    # 2. Columnas reales del modelo alrededor del punto (la más cercana + sus 8 vecinas)
    csv_posix = ensure_path(csv_path).as_posix()
    df = duckdb.query(f"""
        SELECT EASTING, NORTHING, ELEVATION
        FROM read_csv_auto('{csv_posix}', HEADER=TRUE)
        WHERE EASTING  BETWEEN {x - 2 * dx} AND {x + 2 * dx}
          AND NORTHING BETWEEN {y - 2 * dy} AND {y + 2 * dy}
    """).df()
    if df.empty:
        raise ValueError(f"El punto local ({x:.2f}, {y:.2f}) está fuera del modelo de bloques.")
    cols = df[["EASTING", "NORTHING"]].drop_duplicates().reset_index(drop=True)
    k = int(np.argmin(np.hypot(cols["EASTING"] - x, cols["NORTHING"] - y)))
    e0, n0 = float(cols.loc[k, "EASTING"]), float(cols.loc[k, "NORTHING"])

    # 3. Topografía (DEM) en cada columna y su bloque de superficie
    dem = read_dem(dem_path)

    def superficie(e: float, n: float) -> tuple[float, float | None, float]:
        ue, un = local_to_utm_xy(e, n, coef)
        z, _ = bilinear_sample(dem=dem["array"], transform=dem["transform"],
                               nodata=dem["nodata"], xs=np.array([ue]), ys=np.array([un]))
        topo = float(z[0])
        zcol = df.loc[(df["EASTING"] == e) & (df["NORTHING"] == n), "ELEVATION"].to_numpy(float)
        if zcol.size == 0:
            return topo, None, np.nan
        roca = zcol[(zcol - dz / 2.0) <= topo]
        return topo, (float(roca.max()) if roca.size else None), float(zcol.max())

    topo0, z0, ztop_col = superficie(e0, n0)
    if np.isnan(topo0):
        raise ValueError("El DEM no cubre ese punto.")
    if z0 is None:
        raise ValueError(f"No hay roca en la columna ({e0}, {n0}): la topografía ({topo0:.1f}) "
                         "queda bajo el fondo del modelo.")
    if z0 == ztop_col and topo0 > ztop_col + dz / 2.0:
        print(f"AVISO: la topografía ({topo0:.1f}) supera el techo del modelo; se usa el bloque más alto.")

    # 4. Vecinos 3x3: ¿el bloque central es el más alto? (bobina horizontal cabe)
    vecinos = {}
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            vecinos[(i, j)] = superficie(e0 + i * dx, n0 + j * dy)[1]
    plano = all(v is None or v <= z0 for v in vecinos.values())

    # 5. Coordenadas globales del bloque elegido
    utm_e, utm_n = local_to_utm_xy(e0, n0, coef)
    lon0, lat0 = Transformer.from_crs("EPSG:32719", "EPSG:4326", always_xy=True).transform(utm_e, utm_n)

    # 6. Margen de modelo hacia cada lado (informativo)
    margenes = None
    if n_bloques is not None:
        mn = duckdb.query(f"""
            SELECT MIN(EASTING), MAX(EASTING), MIN(NORTHING), MAX(NORTHING),
                   MIN(ELEVATION), MAX(ELEVATION)
            FROM read_csv_auto('{csv_posix}', HEADER=TRUE)
        """).fetchone()
        margenes = {"oeste": e0 - mn[0], "este": mn[1] - e0, "sur": n0 - mn[2],
                    "norte": mn[3] - n0, "abajo": z0 - mn[4], "arriba": mn[5] - z0}

    # 7. Reporte
    print("--- BLOQUE DE SUPERFICIE (centro de la geometría) ---")
    print(f"Punto pedido (local): E={x:.2f}  N={y:.2f}")
    print(f"Columna elegida     : EASTING={e0}  NORTHING={n0}")
    print(f"Topografía DEM      : {topo0:.2f} m  ->  bloque de superficie ELEVATION={z0}"
          f"  (techo de roca {z0 + dz / 2.0})")
    print(f"UTM 19S             : E={utm_e:.2f}  N={utm_n:.2f}   |   lat={lat0:.7f}  lon={lon0:.7f}")
    sup_v = "  ".join(f"{v:.1f}" if v is not None else "--" for v in vecinos.values())
    print(f"Superficie 3x3      : {sup_v}  ->  "
          + ("PLANO: el bloque central es el más alto (cabe una bobina horizontal)" if plano
             else "AVISO: hay un vecino más alto que el centro (la bobina podría atravesar roca)"))
    if margenes is not None:
        need_xy, need_z = n_bloques * dx, n_bloques * dz
        def tag(v, need): return "OK" if v >= need - 1e-6 else f"FALTAN {need - v:.0f} m"
        print(f"Margen de modelo    : O={margenes['oeste']:.0f} ({tag(margenes['oeste'], need_xy)})  "
              f"E={margenes['este']:.0f} ({tag(margenes['este'], need_xy)})  "
              f"S={margenes['sur']:.0f} ({tag(margenes['sur'], need_xy)})  "
              f"N={margenes['norte']:.0f} ({tag(margenes['norte'], need_xy)}) m")
        print(f"                      abajo={margenes['abajo']:.0f}  arriba={margenes['arriba']:.0f} m "
              f"(en Z no importa si no caben {n_bloques} bloques)")

    return {"easting": e0, "northing": n0, "elevation": z0, "topografia": topo0,
            "techo_roca": z0 + dz / 2.0, "utm_e": utm_e, "utm_n": utm_n,
            "lat": lat0, "lon": lon0, "vecinos": vecinos, "plano": plano,
            "margenes": margenes}


def extraer_vecindad_bloques(
    csv_path: str | Path,
    x_centro: float,
    y_centro: float,
    z_centro: float,
    n_bloques: int = 3,
    dx: float = 12.5,  # Dimensión del bloque en el eje X (Metros)
    dy: float = 12.5,  # Dimensión del bloque en el eje Y (Metros)
    dz: float = 15.0   # Dimensión del bloque en el eje Z (Metros)
) -> pd.DataFrame:
    """
    Extrae un volumen de bloques extendiéndose 'n_bloques' en cada dirección 
    desde una coordenada central.
    """
    csv_posix = Path(csv_path).as_posix()
    
    # 1. Calcular los límites usando las dimensiones del bloque.
    # Agregamos un pequeño margen (0.1m) para evitar que por problemas de 
    # redondeo decimal queden bloques en el límite por fuera.
    x_min = x_centro - (n_bloques * dx) - 0.1
    x_max = x_centro + (n_bloques * dx) + 0.1
    
    y_min = y_centro - (n_bloques * dy) - 0.1
    y_max = y_centro + (n_bloques * dy) + 0.1
    
    z_min = z_centro - (n_bloques * dz) - 0.1
    z_max = z_centro + (n_bloques * dz) + 0.1

    # 2. Ejecutar la consulta filtrada directamente al disco
    query = f"""
        SELECT *
        FROM read_csv_auto('{csv_posix}', HEADER=TRUE)
        WHERE EASTING BETWEEN {x_min} AND {x_max}
          AND NORTHING BETWEEN {y_min} AND {y_max}
          AND ELEVATION BETWEEN {z_min} AND {z_max}
    """
    
    # Bloque de reporte visual en consola
    print(f"Extrayendo vecindad (N={n_bloques})...")
    print(f"Rango X: {x_min:.1f} a {x_max:.1f}")
    print(f"Rango Y: {y_min:.1f} a {y_max:.1f}")
    print(f"Rango Z: {z_min:.1f} a {z_max:.1f}")
    
    df = duckdb.query(query).df()
    
    volumen_teorico = (2 * n_bloques + 1) ** 3
    print(f"Extracción completada: {len(df)} bloques encontrados de un máximo posible de {volumen_teorico}.")
    
    return df


def clasificar_bloques_topografia(
    df_bloques: pd.DataFrame, 
    path_dem: str | Path, 
    coef: dict[str, float], 
    dz: float = 15.0
) -> pd.DataFrame:
    """
    Proyecta bloques a UTM, evalúa su posición respecto a la topografía (DEM)
    y los clasifica según si están subterráneos o en el aire.
    
    Parámetros:
    -----------
    df_bloques : pd.DataFrame
        DataFrame con las columnas 'EASTING', 'NORTHING' y 'ELEVATION'.
    path_dem : str o Path
        Ruta del archivo de topografía (.tif).
    coef : dict
        Diccionario con los coeficientes afines para la transformación local_to_utm_xy.
    dz : float, por defecto 15.0
        Altura de los bloques en el eje Z para el cálculo de la base del bloque.
        
    Retorna:
    --------
    pd.DataFrame: El DataFrame original con las columnas 'Z_TOPOGRAFIA' y 'ESTA_EN_AIRE' añadidas.
    """
    # Clonamos el DataFrame para no modificar el original por fuera de la función de forma inesperada
    df_resultado = df_bloques.copy()

    # 1. Proyectar las coordenadas locales de los bloques a UTM
    utm_coords = [
        local_to_utm_xy(x, y, coef) 
        for x, y in zip(df_resultado["EASTING"], df_resultado["NORTHING"])
    ]
    utm_e = np.array([c[0] for c in utm_coords])
    utm_n = np.array([c[1] for c in utm_coords])

    # 2. Cargar el Modelo Digital de Elevación (tu topografía base)
    dem_info = read_dem(path_dem)

    # 3. Extraer la altitud de la superficie para cada X, Y usando tu función de interpolación
    z_superficie, validos = bilinear_sample(
        dem=dem_info["array"],
        transform=dem_info["transform"],
        nodata=dem_info["nodata"],
        xs=utm_e,
        ys=utm_n
    )

    # 4. Anexar la información al DataFrame para compararla
    df_resultado["Z_TOPOGRAFIA"] = z_superficie.round(2)

    # La matemática: Si la elevación del bloque es mayor a la superficie, está en el aire.
    # Le restamos la mitad de la altura del bloque al centroide para evaluar si la base está flotando.
    mitad_dz = dz / 2.0  
    df_resultado["ESTA_EN_AIRE"] = (df_resultado["ELEVATION"] - mitad_dz) > df_resultado["Z_TOPOGRAFIA"]

    # 5. Mostrar el veredicto por consola
    columnas_ver = ["EASTING", "NORTHING", "ELEVATION", "Z_TOPOGRAFIA", "ESTA_EN_AIRE"]
    print("--- VEREDICTO DE UBICACIÓN ESPACIAL ---")
    print(df_resultado[columnas_ver])

    # Un resumen rápido
    total_aire = df_resultado["ESTA_EN_AIRE"].sum()
    print(f"\nResumen: {total_aire} de {len(df_resultado)} bloques están por encima de la topografía.")
    
    return df_resultado


def visualizar_modelo_bloques(
    df_mineral: pd.DataFrame, 
    df_vacios: pd.DataFrame = None, 
    col_ley: str = 'CUTMP',
    dx: float = 12.5, 
    dy: float = 12.5, 
    dz: float = 15.0
):
    """
    Renderiza un modelo de bloques 3D interactivo.
    Permite visualizar bloques con ley como sólidos y bloques vacíos como contornos (wireframe).
    """
    plotter = pv.Plotter(title="Modelo de Bloques 3D")
    
    # ==========================================
    # 1. RENDERIZAR BLOQUES CON MINERAL (Sólidos)
    # ==========================================
    # Asegurarnos de que solo graficamos bloques con ley positiva
    df_min = df_mineral[df_mineral[col_ley] > 0].copy()
    
    if not df_min.empty:
        puntos_min = df_min[['EASTING', 'NORTHING', 'ELEVATION']].values
        valores_ley = df_min[col_ley].values
        
        nube_min = pv.PolyData(puntos_min)
        nube_min[col_ley] = valores_ley
        
        malla_min = nube_min.glyph(
            geom=pv.Cube(x_length=dx, y_length=dy, z_length=dz),
            orient=False, scale=False
        )
        
        plotter.add_mesh(
            malla_min,
            scalars=col_ley,
            cmap='turbo',
            show_edges=True,
            edge_color='black',
            opacity=0.9,
            scalar_bar_args={'vertical': True, 'title': f'Ley ({col_ley} %)'}
        )

    # ==========================================
    # 2. RENDERIZAR BLOQUES VACÍOS (Contornos/Wireframe)
    # ==========================================
    if df_vacios is not None and not df_vacios.empty:
        puntos_vacios = df_vacios[['EASTING', 'NORTHING', 'ELEVATION']].values
        nube_vacios = pv.PolyData(puntos_vacios)
        
        malla_vacios = nube_vacios.glyph(
            geom=pv.Cube(x_length=dx, y_length=dy, z_length=dz),
            orient=False, scale=False
        )
        
        # El parámetro clave aquí es style='wireframe'
        plotter.add_mesh(
            malla_vacios,
            style='wireframe',    # Solo dibuja las líneas
            color='lightgray',    # Un gris claro para que parezca "fantasmal"
            line_width=2,         # Grosor de la línea
            opacity=0.6           # Ligeramente transparente para no distraer
        )

    # ==========================================
    # 3. CONFIGURACIÓN DEL ENTORNO
    # ==========================================
    plotter.show_bounds(grid='front', location='outer', all_edges=True)
    plotter.show_axes()
    
    # Lanzar la visualización interactiva en el notebook
    plotter.show(jupyter_backend='trame')


def exportar_bloques_a_gpkg(
    df_bloques: pd.DataFrame, 
    coef: dict[str, float], 
    ruta_salida: str | Path = "data/mi_cubo_mineral.gpkg",
    layer_name: str = "bloques_cutmp",
    crs: str = "EPSG:32719"
) -> gpd.GeoDataFrame:
    """
    Convierte un DataFrame de bloques a coordenadas UTM tridimensionales 
    y lo exporta como un archivo GeoPackage (.gpkg).
    
    Parámetros:
    -----------
    df_bloques : pd.DataFrame
        DataFrame que contiene 'EASTING', 'NORTHING' y 'ELEVATION'.
    coef : dict
        Diccionario con los coeficientes afines para la transformación UTM.
    ruta_salida : str o Path
        Ruta donde se guardará el archivo .gpkg.
    layer_name : str
        Nombre de la capa dentro del archivo GeoPackage.
    crs : str
        Sistema de referencia de coordenadas global (ej. "EPSG:32719").
        
    Retorna:
    --------
    gpd.GeoDataFrame: El GeoDataFrame creado con geometrías 3D.
    """
    # 1. Trabajar con una copia segura
    df_exportar = df_bloques.copy()

    # 2. Transformar las coordenadas locales a UTM
    utm_coords = [
        local_to_utm_xy(row["EASTING"], row["NORTHING"], coef) 
        for _, row in df_exportar.iterrows()
    ]

    df_exportar["UTM_E"] = [c[0] for c in utm_coords]
    df_exportar["UTM_N"] = [c[1] for c in utm_coords]

    # 3. Crear Geometrías Espaciales Tridimensionales (Point Z)
    geometrias = [
        Point(row["UTM_E"], row["UTM_N"], row["ELEVATION"]) 
        for _, row in df_exportar.iterrows()
    ]

    # 4. Construir el GeoDataFrame
    gdf_bloques = gpd.GeoDataFrame(
        df_exportar, 
        geometry=geometrias, 
        crs=crs
    )

    # 5. Asegurar que el directorio de salida exista y exportar
    ruta_path = Path(ruta_salida)
    ruta_path.parent.mkdir(parents=True, exist_ok=True)
    
    gdf_bloques.to_file(ruta_path, layer=layer_name, driver="GPKG")

    print(f"✅ ¡Exportación exitosa! Archivo guardado en: {ruta_path}")
    
    return gdf_bloques