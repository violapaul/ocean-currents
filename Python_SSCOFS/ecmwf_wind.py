"""
ecmwf_wind.py
-------------

Open-Meteo ECMWF 9 km wind pipeline for route planning.

Two phases:
1) Node discovery: sample candidate lat/lon points, snap to native ECMWF
   nodes via Open-Meteo, dedupe, and save node coordinates to CSV.
2) Forecast fetch: query those cached nodes in batches and unpack hourly
   wind fields into tidy tabular data and an xarray Dataset.

Design notes
------------
- Default model is ``ecmwf_ifs`` (native 9 km IFS HRES), available since
  ECMWF went fully open-data on 2025-10-01.  The legacy ``ecmwf_ifs04``
  (0.4-degree) was discontinued in early 2024 and now returns all nulls.
- Requests include elevation=nan, cell_selection=nearest for raw-node behavior.
- Wind outputs use hourly variables:
  wind_speed_10m, wind_direction_10m, wind_gusts_10m.
- Times are stored in UTC in outputs for stable cross-system use.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import requests
import xarray as xr


OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_MAX_COORDS = 100
DEFAULT_TIMEZONE = "America/Los_Angeles"
DEFAULT_CENTER_LAT = 47.6062
DEFAULT_CENTER_LON = -122.3321
DEFAULT_DISCOVERY_RADIUS_DEG = 0.75
DEFAULT_DISCOVERY_STEP_DEG = 0.08
DEFAULT_WIND_MODELS = ("ecmwf_ifs",)
DEFAULT_MIN_NON_NULL_COVERAGE = 0.05

WIND_VARS = (
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
)


def _chunked(items: Sequence, chunk_size: int):
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def _coord_param(values: Iterable[float]) -> str:
    return ",".join(f"{float(v):.6f}" for v in values)


def _model_label(model: str | None) -> str:
    return "auto" if model is None else str(model)


def _normalise_model_candidates(models: str | Sequence[str] | None) -> list[str | None]:
    if models is None:
        raw = list(DEFAULT_WIND_MODELS)
    elif isinstance(models, str):
        raw = [part.strip() for part in models.split(",") if part.strip()]
    else:
        raw = []
        for item in models:
            token = str(item).strip()
            if token:
                raw.append(token)

    out: list[str | None] = []
    seen: set[str | None] = set()
    for item in raw:
        model = None if str(item).lower() == "auto" else str(item)
        if model not in seen:
            out.append(model)
            seen.add(model)
    if not out:
        raise ValueError("At least one Open-Meteo wind model must be specified")
    return out


def _base_params(timezone: str, model: str | None) -> dict:
    params = {
        "cell_selection": "nearest",
        "hourly": ",".join(WIND_VARS),
        "wind_speed_unit": "kn",
        "timezone": timezone,
    }
    if model is not None:
        params["models"] = str(model)
    return params


def _normalise_location_payload(payload) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if payload.get("error") is True:
            reason = payload.get("reason", "unknown Open-Meteo error")
            raise RuntimeError(f"Open-Meteo API error: {reason}")
        if "latitude" in payload and "longitude" in payload:
            return [payload]
    raise ValueError("Unexpected Open-Meteo response format")


def request_open_meteo_batch(latitudes: Sequence[float], longitudes: Sequence[float],
                             timezone: str = DEFAULT_TIMEZONE,
                             timeout_s: float = 60.0,
                             start_date: str | None = None,
                             end_date: str | None = None,
                             forecast_days: int | None = None,
                             past_days: int | None = None,
                             model: str | None = "ecmwf_ifs",
                             session: requests.Session | None = None) -> list[dict]:
    """Request one Open-Meteo batch (up to 100 coordinates)."""
    if len(latitudes) != len(longitudes):
        raise ValueError("latitudes and longitudes must have equal length")
    if len(latitudes) == 0:
        return []
    if len(latitudes) > OPEN_METEO_MAX_COORDS:
        raise ValueError(
            f"Open-Meteo max coordinates per request is {OPEN_METEO_MAX_COORDS}, "
            f"got {len(latitudes)}"
        )

    params = _base_params(timezone=timezone, model=model)
    params["latitude"] = _coord_param(latitudes)
    params["longitude"] = _coord_param(longitudes)
    # Open-Meteo multi-coordinate calls require one elevation value per location.
    if len(latitudes) == 1:
        params["elevation"] = "nan"
    else:
        params["elevation"] = ",".join(["nan"] * len(latitudes))
    if start_date is not None:
        params["start_date"] = start_date
    if end_date is not None:
        params["end_date"] = end_date
    if forecast_days is not None:
        params["forecast_days"] = int(forecast_days)
    if past_days is not None:
        params["past_days"] = int(past_days)

    sess = session or requests.Session()
    resp = sess.get(OPEN_METEO_FORECAST_URL, params=params, timeout=timeout_s)
    resp.raise_for_status()
    return _normalise_location_payload(resp.json())


def build_latlon_grid(center_lat: float = DEFAULT_CENTER_LAT,
                      center_lon: float = DEFAULT_CENTER_LON,
                      radius_deg: float = DEFAULT_DISCOVERY_RADIUS_DEG,
                      step_deg: float = DEFAULT_DISCOVERY_STEP_DEG) -> list[tuple[float, float]]:
    """Build a square candidate grid around a center point."""
    lats = np.arange(center_lat - radius_deg,
                     center_lat + radius_deg + 0.5 * step_deg,
                     step_deg, dtype=np.float64)
    lons = np.arange(center_lon - radius_deg,
                     center_lon + radius_deg + 0.5 * step_deg,
                     step_deg, dtype=np.float64)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    return list(zip(lat_grid.ravel(), lon_grid.ravel()))


def build_route_bbox_grid(waypoints_latlon: Sequence[tuple[float, float]],
                          padding_deg: float = 0.25,
                          step_deg: float = DEFAULT_DISCOVERY_STEP_DEG) -> list[tuple[float, float]]:
    """Build candidate discovery points over the waypoint bounding box."""
    if not waypoints_latlon:
        raise ValueError("waypoints_latlon cannot be empty")
    lats = np.array([float(wp[0]) for wp in waypoints_latlon], dtype=np.float64)
    lons = np.array([float(wp[1]) for wp in waypoints_latlon], dtype=np.float64)
    lat_min = float(np.min(lats) - padding_deg)
    lat_max = float(np.max(lats) + padding_deg)
    lon_min = float(np.min(lons) - padding_deg)
    lon_max = float(np.max(lons) + padding_deg)
    lat_vals = np.arange(lat_min, lat_max + 0.5 * step_deg, step_deg, dtype=np.float64)
    lon_vals = np.arange(lon_min, lon_max + 0.5 * step_deg, step_deg, dtype=np.float64)
    lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)
    return list(zip(lat_grid.ravel(), lon_grid.ravel()))


def discover_ecmwf_nodes(candidate_points: Sequence[tuple[float, float]],
                         output_csv: str | Path | None = None,
                         timezone: str = DEFAULT_TIMEZONE,
                         chunk_size: int = OPEN_METEO_MAX_COORDS,
                         timeout_s: float = 60.0,
                         verbose: bool = True) -> pd.DataFrame:
    """Snap candidate points to ECMWF nodes and dedupe lat/lon coordinates."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_size > OPEN_METEO_MAX_COORDS:
        raise ValueError(f"chunk_size cannot exceed {OPEN_METEO_MAX_COORDS}")

    points = [(float(lat), float(lon)) for lat, lon in candidate_points]
    if len(points) == 0:
        raise ValueError("candidate_points cannot be empty")

    rows: list[dict] = []
    with requests.Session() as sess:
        for idx, batch in enumerate(_chunked(points, chunk_size), start=1):
            lats = [p[0] for p in batch]
            lons = [p[1] for p in batch]
            locations = request_open_meteo_batch(
                lats, lons,
                timezone=timezone,
                timeout_s=timeout_s,
                session=sess,
            )
            for loc in locations:
                rows.append({
                    "latitude": float(loc["latitude"]),
                    "longitude": float(loc["longitude"]),
                })
            if verbose:
                print(f"Discovery batch {idx}: requested={len(batch)} "
                      f"received={len(locations)}")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Node discovery returned no locations")

    unique = (
        df.dropna(subset=["latitude", "longitude"])
          .drop_duplicates(subset=["latitude", "longitude"])
          .sort_values(["latitude", "longitude"])
          .reset_index(drop=True)
    )

    if output_csv is not None:
        out = Path(output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        unique.to_csv(out, index=False)
        if verbose:
            print(f"Saved {len(unique)} unique nodes to {out}")

    return unique


def load_nodes_csv(nodes_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(nodes_csv)
    required = {"latitude", "longitude"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Nodes CSV is missing columns: {sorted(missing)}")
    out = df.loc[:, ["latitude", "longitude"]].copy()
    out["latitude"] = out["latitude"].astype(np.float64)
    out["longitude"] = out["longitude"].astype(np.float64)
    out = out.dropna().drop_duplicates(["latitude", "longitude"]).reset_index(drop=True)
    out["node"] = np.arange(len(out), dtype=np.int32)
    return out


def _rows_from_location(node_id: int, fallback_lat: float, fallback_lon: float,
                        loc_payload: dict, timezone: str) -> pd.DataFrame:
    hourly = loc_payload.get("hourly", {})
    if not hourly:
        raise ValueError("Location payload missing hourly block")
    for key in ("time",) + WIND_VARS:
        if key not in hourly:
            raise ValueError(f"Location payload missing hourly.{key}")

    df = pd.DataFrame({
        "time": hourly["time"],
        "wind_speed_10m": hourly["wind_speed_10m"],
        "wind_direction_10m": hourly["wind_direction_10m"],
        "wind_gusts_10m": hourly["wind_gusts_10m"],
    })
    if df.empty:
        raise ValueError("Hourly payload is empty")
    if not (len(df["time"]) == len(df["wind_speed_10m"]) ==
            len(df["wind_direction_10m"]) == len(df["wind_gusts_10m"])):
        raise ValueError("Hourly arrays have inconsistent lengths")

    ts_local = pd.to_datetime(df["time"], errors="coerce")
    if ts_local.dt.tz is None:
        ts_local = ts_local.dt.tz_localize(
            timezone, ambiguous="NaT", nonexistent="NaT")
    valid = ts_local.notna()
    df = df.loc[valid].copy()
    ts_local = ts_local.loc[valid]
    ts_utc = ts_local.dt.tz_convert("UTC").dt.tz_localize(None)
    df["time"] = ts_utc

    df["node"] = int(node_id)
    df["latitude"] = float(loc_payload.get("latitude", fallback_lat))
    df["longitude"] = float(loc_payload.get("longitude", fallback_lon))
    return df


def _coverage_stats(tidy: pd.DataFrame) -> dict[str, float]:
    total = max(len(tidy), 1)
    return {
        var: float(tidy[var].notna().sum()) / float(total)
        for var in WIND_VARS
    }


def _build_dataset_from_tidy(tidy: pd.DataFrame, timezone: str,
                             model: str | None) -> xr.Dataset:
    times = np.array(sorted(tidy["time"].unique()), dtype="datetime64[ns]")
    node_ids = np.array(sorted(tidy["node"].unique()), dtype=np.int32)

    node_meta = (
        tidy.loc[:, ["node", "latitude", "longitude"]]
            .drop_duplicates("node")
            .set_index("node")
            .sort_index()
    )

    data_vars = {}
    for var in WIND_VARS:
        grid = (tidy.pivot(index="time", columns="node", values=var)
                    .reindex(index=times, columns=node_ids))
        data_vars[var] = (("time", "node"), grid.to_numpy(dtype=np.float32))

    coverage = _coverage_stats(tidy)
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": times,
            "node": node_ids,
            "latitude": ("node", node_meta.loc[node_ids, "latitude"].to_numpy(dtype=np.float64)),
            "longitude": ("node", node_meta.loc[node_ids, "longitude"].to_numpy(dtype=np.float64)),
        },
        attrs={
            "source": "Open-Meteo",
            "model": _model_label(model),
            "timezone_requested": timezone,
            "time_axis": "UTC",
            "cell_selection": "nearest",
            "elevation": "nan",
            "coverage_wind_speed_10m": coverage["wind_speed_10m"],
            "coverage_wind_direction_10m": coverage["wind_direction_10m"],
            "coverage_wind_gusts_10m": coverage["wind_gusts_10m"],
        },
    )
    return ds


def _fetch_one_model_for_nodes(nodes: pd.DataFrame,
                               timezone: str,
                               chunk_size: int,
                               timeout_s: float,
                               start_date: str | None,
                               end_date: str | None,
                               forecast_days: int | None,
                               past_days: int | None,
                               model: str | None,
                               verbose: bool) -> tuple[pd.DataFrame, xr.Dataset]:
    all_frames = []
    with requests.Session() as sess:
        rows = list(nodes.loc[:, ["node", "latitude", "longitude"]].itertuples(index=False))
        for batch_idx, batch in enumerate(_chunked(rows, chunk_size), start=1):
            lats = [float(r.latitude) for r in batch]
            lons = [float(r.longitude) for r in batch]
            payloads = request_open_meteo_batch(
                lats, lons,
                timezone=timezone,
                timeout_s=timeout_s,
                start_date=start_date,
                end_date=end_date,
                forecast_days=forecast_days,
                past_days=past_days,
                model=model,
                session=sess,
            )
            if len(payloads) != len(batch):
                raise RuntimeError(
                    f"Open-Meteo returned {len(payloads)} locations for "
                    f"{len(batch)} requested nodes in batch {batch_idx}"
                )
            for rec, payload in zip(batch, payloads):
                frame = _rows_from_location(
                    node_id=int(rec.node),
                    fallback_lat=float(rec.latitude),
                    fallback_lon=float(rec.longitude),
                    loc_payload=payload,
                    timezone=timezone,
                )
                all_frames.append(frame)
            if verbose:
                print(f"Fetch batch {batch_idx}: nodes={len(batch)} model={_model_label(model)}")

    tidy = pd.concat(all_frames, ignore_index=True)
    tidy = tidy.sort_values(["time", "node"]).reset_index(drop=True)
    if tidy.duplicated(subset=["time", "node"]).any():
        tidy = tidy.drop_duplicates(subset=["time", "node"], keep="first")
        tidy = tidy.sort_values(["time", "node"]).reset_index(drop=True)

    ds = _build_dataset_from_tidy(tidy=tidy, timezone=timezone, model=model)
    return tidy, ds


def fetch_ecmwf_wind_for_nodes(nodes: pd.DataFrame,
                               timezone: str = DEFAULT_TIMEZONE,
                               chunk_size: int = OPEN_METEO_MAX_COORDS,
                               timeout_s: float = 60.0,
                               start_date: str | None = None,
                               end_date: str | None = None,
                               forecast_days: int | None = None,
                               past_days: int | None = None,
                               models: str | Sequence[str] | None = None,
                               min_non_null_coverage: float = DEFAULT_MIN_NON_NULL_COVERAGE,
                               output_csv: str | Path | None = None,
                               output_netcdf: str | Path | None = None,
                               verbose: bool = True) -> tuple[pd.DataFrame, xr.Dataset]:
    """Fetch hourly wind fields for cached nodes and return (DataFrame, Dataset)."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_size > OPEN_METEO_MAX_COORDS:
        raise ValueError(f"chunk_size cannot exceed {OPEN_METEO_MAX_COORDS}")
    if not (0.0 <= float(min_non_null_coverage) <= 1.0):
        raise ValueError("min_non_null_coverage must be between 0 and 1")
    if "node" not in nodes.columns:
        nodes = nodes.copy().reset_index(drop=True)
        nodes["node"] = np.arange(len(nodes), dtype=np.int32)
    candidates = _normalise_model_candidates(models)
    failures = []
    tidy = None
    ds = None
    selected_model = None
    coverage = None
    for model in candidates:
        label = _model_label(model)
        try:
            tidy_candidate, ds_candidate = _fetch_one_model_for_nodes(
                nodes=nodes,
                timezone=timezone,
                chunk_size=chunk_size,
                timeout_s=timeout_s,
                start_date=start_date,
                end_date=end_date,
                forecast_days=forecast_days,
                past_days=past_days,
                model=model,
                verbose=verbose,
            )
        except Exception as exc:
            failures.append(f"{label}: request failed ({exc})")
            continue

        coverage_candidate = _coverage_stats(tidy_candidate)
        min_coverage_found = min(coverage_candidate.values())
        if min_coverage_found < float(min_non_null_coverage):
            failures.append(
                f"{label}: low non-null coverage "
                f"(speed={coverage_candidate['wind_speed_10m']:.1%}, "
                f"dir={coverage_candidate['wind_direction_10m']:.1%}, "
                f"gusts={coverage_candidate['wind_gusts_10m']:.1%})"
            )
            if verbose:
                print(f"Rejecting wind model {label}: insufficient non-null coverage")
            continue

        tidy = tidy_candidate
        ds = ds_candidate
        selected_model = label
        coverage = coverage_candidate
        break

    if tidy is None or ds is None or coverage is None:
        tried = "; ".join(failures) if failures else "no model attempts were made"
        raise RuntimeError(
            "No usable Open-Meteo wind data found. "
            f"Tried: {tried}. "
            "Try a different model list or lower the coverage threshold."
        )
    if verbose:
        print(
            f"Selected wind model {selected_model}: "
            f"speed={coverage['wind_speed_10m']:.1%}, "
            f"dir={coverage['wind_direction_10m']:.1%}, "
            f"gusts={coverage['wind_gusts_10m']:.1%}"
        )

    if output_csv is not None:
        out_csv = Path(output_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        tidy.to_csv(out_csv, index=False)
        if verbose:
            print(f"Saved tidy wind table to {out_csv}")

    if output_netcdf is not None:
        out_nc = Path(output_netcdf)
        out_nc.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(out_nc)
        if verbose:
            print(f"Saved wind dataset to {out_nc}")

    return tidy, ds


def fetch_route_wind_dataset(waypoints_latlon: Sequence[tuple[float, float]],
                             nodes_csv: str | Path,
                             timezone: str = DEFAULT_TIMEZONE,
                             padding_deg: float = 0.25,
                             step_deg: float = DEFAULT_DISCOVERY_STEP_DEG,
                             chunk_size: int = OPEN_METEO_MAX_COORDS,
                             timeout_s: float = 60.0,
                             start_date: str | None = None,
                             end_date: str | None = None,
                             forecast_days: int | None = None,
                             past_days: int | None = None,
                             models: str | Sequence[str] | None = None,
                             min_non_null_coverage: float = DEFAULT_MIN_NON_NULL_COVERAGE,
                             output_csv: str | Path | None = None,
                             output_netcdf: str | Path | None = None,
                             use_cached_nodes: bool = True,
                             verbose: bool = True) -> tuple[pd.DataFrame, xr.Dataset, pd.DataFrame]:
    """Discover/reuse route-local nodes, then fetch wind dataset."""
    nodes_path = Path(nodes_csv)
    if use_cached_nodes and nodes_path.exists():
        if verbose:
            print(f"Using cached route nodes: {nodes_path}")
        nodes = load_nodes_csv(nodes_path)
    else:
        candidates = build_route_bbox_grid(
            waypoints_latlon=waypoints_latlon,
            padding_deg=padding_deg,
            step_deg=step_deg,
        )
        discovered = discover_ecmwf_nodes(
            candidate_points=candidates,
            output_csv=nodes_path,
            timezone=timezone,
            chunk_size=chunk_size,
            timeout_s=timeout_s,
            verbose=verbose,
        )
        nodes = discovered.copy()
        nodes["node"] = np.arange(len(nodes), dtype=np.int32)

    tidy, ds = fetch_ecmwf_wind_for_nodes(
        nodes=nodes,
        timezone=timezone,
        chunk_size=chunk_size,
        timeout_s=timeout_s,
        start_date=start_date,
        end_date=end_date,
        forecast_days=forecast_days,
        past_days=past_days,
        models=models,
        min_non_null_coverage=min_non_null_coverage,
        output_csv=output_csv,
        output_netcdf=output_netcdf,
        verbose=verbose,
    )
    return tidy, ds, nodes


def _parse_waypoints_arg(raw: str) -> list[tuple[float, float]]:
    """Parse 'lat,lon;lat,lon;...' into waypoint tuples."""
    wps = []
    for part in raw.split(";"):
        token = part.strip()
        if not token:
            continue
        lat_s, lon_s = token.split(",")
        wps.append((float(lat_s), float(lon_s)))
    if len(wps) < 2:
        raise ValueError("At least two waypoints are required")
    return wps


def main():
    parser = argparse.ArgumentParser(
        description="Open-Meteo ECMWF node discovery + wind fetch pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_disc = sub.add_parser("discover", help="Discover snapped ECMWF nodes")
    p_disc.add_argument("--output", type=Path, default=Path("ecmwf_seattle_nodes.csv"))
    p_disc.add_argument("--center-lat", type=float, default=DEFAULT_CENTER_LAT)
    p_disc.add_argument("--center-lon", type=float, default=DEFAULT_CENTER_LON)
    p_disc.add_argument("--radius-deg", type=float, default=DEFAULT_DISCOVERY_RADIUS_DEG)
    p_disc.add_argument("--step-deg", type=float, default=DEFAULT_DISCOVERY_STEP_DEG)
    p_disc.add_argument("--timezone", type=str, default=DEFAULT_TIMEZONE)
    p_disc.add_argument("--chunk-size", type=int, default=OPEN_METEO_MAX_COORDS)

    p_fetch = sub.add_parser("fetch", help="Fetch wind forecast for cached nodes")
    p_fetch.add_argument("--nodes", type=Path, required=True)
    p_fetch.add_argument("--output-csv", type=Path, default=None)
    p_fetch.add_argument("--output-netcdf", type=Path, default=Path("ecmwf_wind_forecast.nc"))
    p_fetch.add_argument("--timezone", type=str, default=DEFAULT_TIMEZONE)
    p_fetch.add_argument("--chunk-size", type=int, default=OPEN_METEO_MAX_COORDS)
    p_fetch.add_argument("--start-date", type=str, default=None)
    p_fetch.add_argument("--end-date", type=str, default=None)
    p_fetch.add_argument("--forecast-days", type=int, default=None)
    p_fetch.add_argument("--past-days", type=int, default=None)
    p_fetch.add_argument("--models", type=str, default=",".join(DEFAULT_WIND_MODELS))
    p_fetch.add_argument("--min-non-null-coverage", type=float,
                         default=DEFAULT_MIN_NON_NULL_COVERAGE)

    p_route = sub.add_parser("route-fetch", help="Discover nodes near a route and fetch wind")
    p_route.add_argument("--waypoints", type=str, required=True,
                         help='Format: "lat,lon;lat,lon;lat,lon"')
    p_route.add_argument("--nodes", type=Path, default=Path("ecmwf_route_nodes.csv"))
    p_route.add_argument("--padding-deg", type=float, default=0.25)
    p_route.add_argument("--step-deg", type=float, default=DEFAULT_DISCOVERY_STEP_DEG)
    p_route.add_argument("--timezone", type=str, default=DEFAULT_TIMEZONE)
    p_route.add_argument("--chunk-size", type=int, default=OPEN_METEO_MAX_COORDS)
    p_route.add_argument("--start-date", type=str, default=None)
    p_route.add_argument("--end-date", type=str, default=None)
    p_route.add_argument("--forecast-days", type=int, default=None)
    p_route.add_argument("--past-days", type=int, default=None)
    p_route.add_argument("--models", type=str, default=",".join(DEFAULT_WIND_MODELS))
    p_route.add_argument("--min-non-null-coverage", type=float,
                         default=DEFAULT_MIN_NON_NULL_COVERAGE)
    p_route.add_argument("--output-csv", type=Path, default=None)
    p_route.add_argument("--output-netcdf", type=Path, default=Path("ecmwf_route_wind.nc"))
    p_route.add_argument("--no-node-cache", action="store_true")

    args = parser.parse_args()

    if args.cmd == "discover":
        points = build_latlon_grid(
            center_lat=args.center_lat,
            center_lon=args.center_lon,
            radius_deg=args.radius_deg,
            step_deg=args.step_deg,
        )
        discover_ecmwf_nodes(
            candidate_points=points,
            output_csv=args.output,
            timezone=args.timezone,
            chunk_size=args.chunk_size,
            verbose=True,
        )
        return

    if args.cmd == "fetch":
        nodes = load_nodes_csv(args.nodes)
        fetch_ecmwf_wind_for_nodes(
            nodes=nodes,
            timezone=args.timezone,
            chunk_size=args.chunk_size,
            start_date=args.start_date,
            end_date=args.end_date,
            forecast_days=args.forecast_days,
            past_days=args.past_days,
            models=args.models,
            min_non_null_coverage=args.min_non_null_coverage,
            output_csv=args.output_csv,
            output_netcdf=args.output_netcdf,
            verbose=True,
        )
        return

    if args.cmd == "route-fetch":
        wps = _parse_waypoints_arg(args.waypoints)
        fetch_route_wind_dataset(
            waypoints_latlon=wps,
            nodes_csv=args.nodes,
            timezone=args.timezone,
            padding_deg=args.padding_deg,
            step_deg=args.step_deg,
            chunk_size=args.chunk_size,
            start_date=args.start_date,
            end_date=args.end_date,
            forecast_days=args.forecast_days,
            past_days=args.past_days,
            models=args.models,
            min_non_null_coverage=args.min_non_null_coverage,
            output_csv=args.output_csv,
            output_netcdf=args.output_netcdf,
            use_cached_nodes=not args.no_node_cache,
            verbose=True,
        )
        return

    raise RuntimeError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
