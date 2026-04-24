"""
run_route.py
------------

Run a multi-leg sailboat route defined in a YAML task file.

Usage:
    conda run -n anaconda python run_route.py routes/shilshole_alki_return.yaml
    python run_route.py routes/my_route.yaml --no-cache
    python run_route.py routes/my_route.yaml --no-plots

YAML schema:
    task:
      name: "Route name"
      description: "Optional description"

    departure:
      datetime: "YYYY-MM-DD HH:MM"   # local time
      tz: "America/Los_Angeles"       # IANA timezone (default: America/Los_Angeles)

    waypoints:
      - [lat, lon]                    # start
      - [lat, lon]                    # intermediate / turnaround
      - ...                           # back to start, or wherever

    boat:
      speed_kt: 6.0                   # fallback fixed speed
      polar: "path/to/polar.csv"      # optional; relative to this script's dir
      minimum_twa: 38                 # optional upwind no-go
      maximum_twa: 165                # optional downwind no-go

    wind:                             # optional; only used with a polar
      # Option A: constant wind
      source: "constant"
      speed_kt: 12.0
      direction_deg: 180.0            # meteorological "from" convention

      # Option B: dynamic Open-Meteo ECMWF wind near route
      # source: "open_meteo_ecmwf"
      # timezone: "America/Los_Angeles"
      # step_deg: 0.08
      # padding_deg: 0.25
      # duration_hours: 12
      # buffer_hours: 2
      # use_cache_even_if_stale: false # default; true reuses route wind NetCDF

      # Option C: manual time-varying schedule (spatially uniform, linearly
      # interpolated between anchors; queries outside the schedule clamp to
      # the nearest endpoint).
      # source: "schedule"
      # timezone: "America/Los_Angeles"  # tz for time strings (default: same as departure)
      # schedule:
      #   - { time: "09:15", speed_kt: 9.0,  from_deg: 340 }
      #   - { time: "11:00", speed_kt: 12.0, from_deg: 330 }
      #   - { time: "13:00", speed_kt: 16.0, from_deg: 320 }

    routing:
      router_type: "sector"           # required; only supported router
      tack_penalty_s: 90
      tack_threshold_deg: 50
      gybe_penalty_s: 90              # defaults to tack_penalty_s
      gybe_threshold_deg: 120         # TWA >= this is downwind/gybe territory
      duration_hours: 10              # forecast window to load per leg
      # Optional performance knobs (all have safe defaults):
      # polar_sweep_coarse_step: 5    # 1=exact, >1 faster approximate
      # k_candidates: 50              # KD candidates for sector graph fill
      # max_edge_m: 2000              # max sector edge length
      # min_edge_m: 50                # min sector edge length
      # los_samples: 15               # land-crossing samples per candidate edge
      # corridor_pad_factors: [0.85, 1.0, 1.35]
      # corridor_cache_max: 4
      # use_dense_polar: true
      # use_dot_filter: false         # experimental; disabled in production YAMLs

    output:
      save_plots: true
      save_hourly_frames: false       # optional PNG sequence; expensive
      plot_dir: "routes/output"       # relative to this script's dir
"""

import argparse
import sys
from pathlib import Path
import datetime as _dt
from datetime import timedelta

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from sail_routing import (
    PolarTable, WindField, BoatModel, SectorRouter,
    load_current_field, plot_route,
    KNOTS_TO_MS, MS_TO_KNOTS,
    CurrentField,
)
from shoreline_utils import draw_shoreline

HERE = Path(__file__).parent


# ---------------------------------------------------------------------------
# YAML loading and validation
# ---------------------------------------------------------------------------

_ROUTABLE_MARK_TYPES = {"close_round", "finish_seed", "waypoint"}
_CONSTRAINT_MARK_TYPES = {"constraint"}
_ALL_MARK_TYPES = _ROUTABLE_MARK_TYPES | _CONSTRAINT_MARK_TYPES


def _is_routable_wp(wp) -> bool:
    """True if *wp* is a waypoint the A* router should navigate TO."""
    if isinstance(wp, (list, tuple)):
        return True
    if isinstance(wp, dict):
        mt = wp.get("mark_type")
        if mt is None:
            return False          # old-format dict → land mark constraint
        return mt in _ROUTABLE_MARK_TYPES
    return False


def load_task(yaml_path: Path) -> dict:
    """Load and lightly validate a routing task YAML.

    Waypoints may be any of:
      - ``[lat, lon]``                   — real routing waypoint (old format)
      - dict with ``mark_type`` in {``close_round``, ``finish_seed``,
        ``waypoint``}                    — real routing waypoint (new format)
      - dict with ``mark_type: constraint`` and ``barrier_bearing_deg``
                                         — barrier constraint (new format)
      - dict with ``barrier_bearing_deg`` but no ``mark_type``
                                         — barrier constraint (old format)
    """
    with open(yaml_path) as f:
        doc = yaml.safe_load(f)

    raw_wps = doc.get("waypoints", [])
    if len(raw_wps) < 2:
        raise ValueError("YAML must contain at least 2 waypoints.")

    real_count = 0
    for i, wp in enumerate(raw_wps):
        if isinstance(wp, (list, tuple)):
            if len(wp) != 2:
                raise ValueError(f"Waypoint {i} must be [lat, lon], got {wp!r}.")
            real_count += 1

        elif isinstance(wp, dict):
            mt = wp.get("mark_type")

            if mt is not None and mt not in _ALL_MARK_TYPES:
                raise ValueError(
                    f"Waypoint {i}: unknown mark_type={mt!r}. "
                    f"Valid types: {sorted(_ALL_MARK_TYPES)}."
                )

            for key in ("lat", "lon"):
                if key not in wp:
                    raise ValueError(
                        f"Waypoint {i} (dict) is missing required key '{key}'."
                    )

            if mt in _ROUTABLE_MARK_TYPES:
                real_count += 1
            else:
                # constraint (new format) or old-format land mark (no mark_type)
                if "barrier_bearing_deg" not in wp:
                    raise ValueError(
                        f"Constraint mark at index {i} is missing 'barrier_bearing_deg'. "
                        "Constraint marks must have: lat, lon, rounding, barrier_bearing_deg."
                    )
                if "rounding" not in wp:
                    raise ValueError(
                        f"Constraint mark at index {i} is missing 'rounding'."
                    )
                if wp["rounding"] not in ("port", "starboard"):
                    raise ValueError(
                        f"Constraint mark at index {i}: rounding must be "
                        f"'port' or 'starboard', got {wp['rounding']!r}."
                    )
        else:
            raise ValueError(
                f"Waypoint {i} must be a [lat, lon] list or a dict, got {wp!r}."
            )

    if real_count < 2:
        raise ValueError(
            "YAML must contain at least 2 routable waypoints "
            "(constraint marks do not count)."
        )
    if not _is_routable_wp(raw_wps[0]):
        raise ValueError(
            "The first waypoint must be routable ([lat, lon] or "
            "mark_type close_round/finish_seed/waypoint), not a constraint."
        )
    if not _is_routable_wp(raw_wps[-1]):
        raise ValueError(
            "The last waypoint must be routable ([lat, lon] or "
            "mark_type close_round/finish_seed/waypoint), not a constraint."
        )

    dep = doc.get("departure", {})
    if "datetime" not in dep:
        raise ValueError("YAML must contain departure.datetime.")

    return doc


def partition_waypoints(raw_wps: list) -> tuple:
    """Split the raw waypoints list into real waypoints and per-leg constraints.

    Supports both the old format (``[lat, lon]`` lists + bare dicts) and the
    new ``mark_type``-based format.

    Parameters
    ----------
    raw_wps : list
        Each entry is either:
        - ``[lat, lon]``                       — routable waypoint (old format)
        - dict with ``mark_type`` in
          {``close_round``, ``finish_seed``, ``waypoint``}
                                               — routable waypoint (new format)
        - dict with ``mark_type: constraint``  — barrier constraint (new format)
        - dict with ``barrier_bearing_deg``
          but no ``mark_type``                 — barrier constraint (old format)

    Returns
    -------
    real_wps : list of (lat, lon)
        Routable waypoints the A* router navigates between.
    leg_marks : dict
        Maps leg index (0-based) to a list of constraint dicts for that leg.
        Leg *i* runs from ``real_wps[i]`` to ``real_wps[i+1]``.
    """
    real_wps = []
    leg_marks: dict = {}
    pending_marks: list = []

    for wp in raw_wps:
        if _is_routable_wp(wp):
            if pending_marks and real_wps:
                leg_idx = len(real_wps) - 1
                leg_marks.setdefault(leg_idx, []).extend(pending_marks)
                pending_marks = []
            if isinstance(wp, (list, tuple)):
                real_wps.append((float(wp[0]), float(wp[1])))
            else:
                real_wps.append((float(wp["lat"]), float(wp["lon"])))
        else:
            pending_marks.append(dict(wp))

    if pending_marks and len(real_wps) >= 2:
        leg_marks.setdefault(len(real_wps) - 2, []).extend(pending_marks)

    return real_wps, leg_marks


def compute_barriers(marks: list, transformer, ray_length: float = 25_000.0) -> np.ndarray:
    """Convert land mark constraints to barrier line segments in UTM coordinates.

    Each mark produces one ray segment starting at the mark and extending
    ``ray_length`` metres in the direction given by ``barrier_bearing_deg``
    (compass bearing: 0 = north, 90 = east, 180 = south, 270 = west).

    Parameters
    ----------
    marks : list of dict
        Each dict must contain ``lat``, ``lon``, ``barrier_bearing_deg``.
    transformer : pyproj.Transformer
        Lon→lat to UTM transformer (as returned by ``load_current_field``).
    ray_length : float
        Barrier ray length in metres (default 25 km — long enough to block
        any realistic wrong-side shortcut in Puget Sound).

    Returns
    -------
    barriers : np.ndarray, shape (N, 4), dtype float64
        Each row is ``[x1, y1, x2, y2]`` (UTM metres).
        Returns an empty ``(0, 4)`` array if *marks* is empty.
    """
    if not marks:
        return np.empty((0, 4), dtype=np.float64)

    rows = []
    for m in marks:
        lat = float(m["lat"])
        lon = float(m["lon"])
        bearing_deg = float(m["barrier_bearing_deg"])
        bearing_rad = np.radians(bearing_deg)
        # Compass bearing → UTM displacement: east = sin(b), north = cos(b)
        dx = np.sin(bearing_rad) * ray_length
        dy = np.cos(bearing_rad) * ray_length
        mx, my = transformer.transform(lon, lat)
        rows.append([mx, my, mx + dx, my + dy])

    return np.array(rows, dtype=np.float64)


# ---------------------------------------------------------------------------
# Build routing objects from YAML
# ---------------------------------------------------------------------------

def _build_schedule_wind(wind_cfg: dict, ctx: dict) -> WindField:
    """Build a spatially-uniform, time-varying wind from a YAML schedule.

    Each schedule entry is an anchor point ``(time, speed_kt, from_deg)``.
    The engine linearly interpolates between anchors and clamps queries
    outside the schedule to the nearest endpoint
    (``WindField.from_frames`` ``temporal_const`` mode).

    ``time`` may be either ``"HH:MM"`` (assumed to be on the same local date
    as departure, in the configured timezone) or ``"YYYY-MM-DD HH:MM"`` for
    multi-day schedules.
    """
    from zoneinfo import ZoneInfo
    from ecmwf_wind import DEFAULT_TIMEZONE

    sched = wind_cfg.get("schedule") or []
    if len(sched) < 1:
        raise ValueError("wind.schedule must have at least one entry")

    tz = ZoneInfo(str(wind_cfg.get("timezone", ctx.get("tz_str", DEFAULT_TIMEZONE))))
    depart_dt = ctx["depart_dt"]                        # local tz-aware
    depart_utc = ctx["depart_utc"]                      # UTC tz-aware
    start_time_s = float(ctx["start_time_s"])           # router time-base offset
    depart_local_date = depart_dt.astimezone(tz).date()

    anchors = []                                         # list of (elapsed_s, wu, wv)
    for entry in sched:
        t_raw = str(entry["time"]).strip()
        if len(t_raw) <= 5:                              # "HH:MM"
            hhmm = _dt.datetime.strptime(t_raw, "%H:%M").time()
            local_dt = _dt.datetime.combine(depart_local_date, hhmm, tzinfo=tz)
        else:                                            # "YYYY-MM-DD HH:MM"
            naive = _dt.datetime.strptime(t_raw, "%Y-%m-%d %H:%M")
            local_dt = naive.replace(tzinfo=tz)

        elapsed_s = start_time_s + (
            local_dt.astimezone(_dt.timezone.utc) - depart_utc
        ).total_seconds()

        speed_ms = float(entry["speed_kt"]) * KNOTS_TO_MS
        from_rad = np.radians(float(entry["from_deg"]))
        wu = -speed_ms * float(np.sin(from_rad))
        wv = -speed_ms * float(np.cos(from_rad))
        anchors.append((elapsed_s, wu, wv))

    anchors.sort(key=lambda a: a[0])
    if len(anchors) == 1:                                # degenerate single anchor → constant
        anchors.append((anchors[0][0] + 3600.0, anchors[0][1], anchors[0][2]))
    times_s = [a[0] for a in anchors]
    wus     = [a[1] for a in anchors]
    wvs     = [a[2] for a in anchors]

    print(f"Wind: schedule with {len(sched)} anchor(s):")
    for entry in sched:
        print(f"  {entry['time']}  {float(entry['speed_kt']):.1f} kt @ "
              f"{float(entry['from_deg']):.0f}°")

    return WindField.from_frames(
        xs=None, ys=None,
        wu_frames=wus, wv_frames=wvs,
        frame_times_s=times_s,
    )


def _build_openmeteo_route_wind(wind_cfg: dict, ctx: dict) -> WindField:
    """Build a dynamic wind field by fetching Open-Meteo ECMWF nodes."""
    from ecmwf_wind import fetch_route_wind_dataset, DEFAULT_TIMEZONE
    import xarray as xr

    tz_req = str(wind_cfg.get("timezone", ctx.get("tz_str", DEFAULT_TIMEZONE)))
    padding_deg = float(wind_cfg.get("padding_deg", 0.25))
    step_deg = float(wind_cfg.get("step_deg", 0.08))
    chunk_size = int(wind_cfg.get("chunk_size", 100))
    wind_hours = float(wind_cfg.get("duration_hours", ctx.get("duration_h", 10)))
    buffer_h = float(wind_cfg.get("buffer_hours", 2.0))

    depart_dt = ctx["depart_dt"]
    end_dt = depart_dt + timedelta(hours=wind_hours + buffer_h)
    start_date = wind_cfg.get("start_date", depart_dt.date().isoformat())
    end_date = wind_cfg.get("end_date", end_dt.date().isoformat())
    forecast_days = wind_cfg.get("forecast_days")
    past_days = wind_cfg.get("past_days")
    models = wind_cfg.get("models")
    min_non_null_coverage = float(wind_cfg.get("min_non_null_coverage", 0.05))

    cache_dir = HERE / ".wind_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    slug = ctx["task_slug"]

    nodes_csv = Path(wind_cfg.get("nodes_csv", cache_dir / f"{slug}_ecmwf_nodes.csv"))
    if not nodes_csv.is_absolute():
        nodes_csv = HERE / nodes_csv

    out_nc = wind_cfg.get("output_netcdf")
    if out_nc is None:
        out_nc_path = cache_dir / f"{slug}_ecmwf_wind.nc"
    else:
        out_nc_path = Path(out_nc)
        if not out_nc_path.is_absolute():
            out_nc_path = HERE / out_nc_path

    out_csv = wind_cfg.get("output_csv")
    out_csv_path = None
    if out_csv is not None:
        out_csv_path = Path(out_csv)
        if not out_csv_path.is_absolute():
            out_csv_path = HERE / out_csv_path

    use_cache_even_if_stale = bool(
        wind_cfg.get("use_cache_even_if_stale",
                     wind_cfg.get("use_cached_netcdf", False))
    )
    if use_cache_even_if_stale and (not ctx["no_cache"]) and out_nc_path.exists():
        print(f"Using cached wind dataset: {out_nc_path}")
        ds = xr.load_dataset(out_nc_path)
        nodes_count = int(ds.sizes.get("point", len(ds["latitude"].values)))
    else:
        if out_nc_path.exists() and not ctx["no_cache"]:
            print(
                "Fetching fresh route-local ECMWF wind "
                f"(ignoring existing cache: {out_nc_path.name})…"
            )
        else:
            print("Fetching route-local ECMWF wind nodes…")
        _, ds, nodes = fetch_route_wind_dataset(
            waypoints_latlon=ctx["waypoints"],
            nodes_csv=nodes_csv,
            timezone=tz_req,
            padding_deg=padding_deg,
            step_deg=step_deg,
            chunk_size=chunk_size,
            start_date=start_date,
            end_date=end_date,
            forecast_days=forecast_days,
            past_days=past_days,
            models=models,
            min_non_null_coverage=min_non_null_coverage,
            output_csv=out_csv_path,
            output_netcdf=out_nc_path,
            use_cached_nodes=(not ctx["no_cache"]),
            verbose=True,
        )
        nodes_count = len(nodes)

    time_vals = np.asarray(ds["time"].values, dtype="datetime64[s]")
    if time_vals.size == 0:
        raise RuntimeError("Open-Meteo wind dataset is empty")
    depart_utc_naive = np.datetime64(
        ctx["depart_utc"].astimezone(_dt.timezone.utc).replace(tzinfo=None),
        "s",
    )
    # Dataset times are already UTC (ecmwf_wind.py converts to UTC before
    # saving).  Compute offsets relative to the UTC departure time.
    frame_times_s = ctx["start_time_s"] + (
        (time_vals - depart_utc_naive) / np.timedelta64(1, "s")
    ).astype(np.float64)

    speed_ms = ds["wind_speed_10m"].values.astype(np.float64) * KNOTS_TO_MS
    from_rad = np.radians(ds["wind_direction_10m"].values.astype(np.float64))
    wu_frames = -speed_ms * np.sin(from_rad)
    wv_frames = -speed_ms * np.cos(from_rad)

    node_lons = ds["longitude"].values.astype(np.float64)
    node_lats = ds["latitude"].values.astype(np.float64)
    node_x, node_y = ctx["transformer"].transform(node_lons, node_lats)

    wind = WindField.from_node_frames(
        node_x=node_x,
        node_y=node_y,
        wu_frames=wu_frames,
        wv_frames=wv_frames,
        frame_times_s=frame_times_s,
    )
    print(f"Wind field ready: {nodes_count} nodes, {len(frame_times_s)} hourly frames, "
          f"model={ds.attrs.get('model', 'unknown')}")
    return wind


def build_boat_and_wind(doc: dict, context: dict | None = None):
    """Construct BoatModel and optional WindField from YAML config."""
    boat_cfg = doc.get("boat", {})
    speed_kt = float(boat_cfg.get("speed_kt", 6.0))

    polar = None
    polar_path_str = boat_cfg.get("polar")
    minimum_twa = float(boat_cfg.get("minimum_twa", 0.0))
    maximum_twa = float(boat_cfg.get("maximum_twa", 180.0))
    if polar_path_str:
        polar_path = Path(polar_path_str)
        if not polar_path.is_absolute():
            polar_path = HERE / polar_path
        if polar_path.exists():
            polar = PolarTable(polar_path, minimum_twa=minimum_twa,
                               maximum_twa=maximum_twa)
            zones = []
            if minimum_twa > 0:
                zones.append(f"TWA < {minimum_twa:.0f}°")
            if maximum_twa < 180:
                zones.append(f"TWA > {maximum_twa:.0f}°")
            nogo = f", no-go zone {' and '.join(zones)}" if zones else ""
            print(f"Polar loaded from {polar_path}: max {polar.max_speed_kt:.1f} kt{nogo}")
        else:
            print(f"Warning: polar file not found at {polar_path}, using fixed speed.")

    boat = BoatModel(base_speed_knots=speed_kt, polar=polar)

    wind = None
    wind_cfg = doc.get("wind")
    if wind_cfg and polar is not None:
        source = str(wind_cfg.get("source", "constant")).lower()
        if source in ("constant", "met", "manual"):
            ws = float(wind_cfg.get("speed_kt", 0.0))
            wd = float(wind_cfg.get("direction_deg", 0.0))
            wind = WindField.from_met(ws, wd)
            print(f"Wind: {ws:.1f} kt from {wd:.0f}°")
        elif source in ("schedule", "manual_schedule", "piecewise"):
            if context is None:
                raise ValueError("schedule wind source requires route context")
            wind = _build_schedule_wind(wind_cfg, context)
        elif source in ("open_meteo_ecmwf", "ecmwf_openmeteo", "ecmwf_route"):
            if context is None:
                raise ValueError("Open-Meteo wind source requires route context")
            allow_fallback = bool(wind_cfg.get("allow_fallback", True))
            try:
                wind = _build_openmeteo_route_wind(wind_cfg, context)
            except Exception as exc:
                if allow_fallback and "speed_kt" in wind_cfg and "direction_deg" in wind_cfg:
                    ws = float(wind_cfg.get("speed_kt", 0.0))
                    wd = float(wind_cfg.get("direction_deg", 0.0))
                    wind = WindField.from_met(ws, wd)
                    print("Warning: dynamic wind fetch failed "
                          f"({exc}). Falling back to constant wind "
                          f"{ws:.1f} kt from {wd:.0f}°")
                else:
                    raise
        else:
            print(f"Warning: unknown wind source '{source}', ignoring wind config.")
    elif wind_cfg and polar is None:
        print("Note: wind specified but no polar loaded — wind ignored.")

    return boat, wind


# ---------------------------------------------------------------------------
# Single-leg routing
# ---------------------------------------------------------------------------

def run_leg(router: SectorRouter, start_ll, end_ll, start_time_s: float,
            leg_num: int, leg_label: str,
            barriers: np.ndarray | None = None) -> tuple:
    """Run A* for one leg. Returns (route, xs, ys, water_mask)."""
    print(f"\n{'─'*60}")
    print(f"Leg {leg_num}: {leg_label}")
    print(f"  From: {start_ll[0]:.6f}, {start_ll[1]:.6f}")
    print(f"  To:   {end_ll[0]:.6f}, {end_ll[1]:.6f}")
    print(f"  Start time offset: {start_time_s:.0f}s ({start_time_s/3600:.2f}h)")
    if barriers is not None and len(barriers) > 0:
        print(f"  Land mark barriers: {len(barriers)}")
    route, xs, ys, wm = router.find_route(start_ll, end_ll,
                                           start_time_s=start_time_s,
                                           barriers=barriers)
    return route, xs, ys, wm


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_leg_summary(leg_num: int, label: str, route, depart_utc, tz):
    """Print a formatted summary for one leg."""
    import datetime as _dt
    arrive_utc = depart_utc + _dt.timedelta(seconds=route.total_time_s)
    arrive_local = arrive_utc.astimezone(tz)
    depart_local = depart_utc.astimezone(tz)

    print(f"\nLeg {leg_num} summary — {label}")
    print(f"  Depart:    {depart_local:%I:%M %p %Z}")
    print(f"  Arrive:    {arrive_local:%I:%M %p %Z}")
    print(f"  Distance:  {route.total_distance_m / 1852:.2f} nm")
    print(f"  Time:      {route.total_time_s / 60:.1f} min")
    print(f"  Avg SOG:   {route.avg_sog_knots:.2f} kt")
    print(f"  Boat STW:  {route.boat_speed_knots:.1f} kt")
    print(f"  SOG adv:   {route.avg_sog_knots - route.boat_speed_knots:+.2f} kt")
    if route.debug:
        tacks = int(route.debug.get("maneuver_tack_count", 0))
        gybes = int(route.debug.get("maneuver_gybe_count", 0))
        turns = int(route.debug.get("maneuver_turn_count", 0))
        print(f"  Maneuvers: {tacks} tacks, {gybes} gybes, {turns} other turns")


def print_trip_summary(routes, labels, total_start_utc, tz):
    """Print a combined summary for the whole trip."""
    import datetime as _dt
    total_time_s = sum(r.total_time_s for r in routes)
    total_dist_m = sum(r.total_distance_m for r in routes)
    arrive_utc = total_start_utc + _dt.timedelta(seconds=total_time_s)
    arrive_local = arrive_utc.astimezone(tz)
    depart_local = total_start_utc.astimezone(tz)

    print(f"\n{'═'*60}")
    print(f"TRIP TOTAL ({len(routes)} legs)")
    print(f"  Depart:    {depart_local:%Y-%m-%d %I:%M %p %Z}")
    print(f"  Arrive:    {arrive_local:%Y-%m-%d %I:%M %p %Z}")
    print(f"  Distance:  {total_dist_m / 1852:.2f} nm")
    print(f"  Time:      {total_time_s / 3600:.2f} hr  ({total_time_s / 60:.0f} min)")
    if total_time_s > 0:
        avg_sog = total_dist_m / total_time_s * MS_TO_KNOTS
        print(f"  Avg SOG:   {avg_sog:.2f} kt")

    print(f"\nLeg breakdown:")
    for i, (r, lbl) in enumerate(zip(routes, labels), 1):
        print(f"  {i}. {lbl:<35} {r.total_distance_m/1852:.2f} nm  "
              f"{r.total_time_s/60:.0f} min  SOG {r.avg_sog_knots:.2f} kt")
    print(f"{'═'*60}")


def _perf_value(route, key, default=0.0):
    """Return one route performance counter as a float."""
    perf = (route.debug or {}).get("perf", {})
    try:
        return float(perf.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def print_performance_summary(routes):
    """Print a compact aggregate benchmark table for route runs."""
    if not routes:
        return

    keys = ("setup", "graph_build", "astar", "diagnostics", "plot_grid", "total")
    totals = {k: sum(_perf_value(r, k) for r in routes) for k in keys}
    attempts = sum(int(_perf_value(r, "corridor_attempts", 0)) for r in routes)
    hits = sum(int(_perf_value(r, "graph_cache_hits", 0)) for r in routes)

    print(f"\n{'═'*60}")
    print("ROUTER PERFORMANCE")
    print("  Leg     setup   graph   astar   diag  grid   total   explored")
    for i, route in enumerate(routes, 1):
        print(
            f"  {i:>3}  "
            f"{_perf_value(route, 'setup'):7.3f} "
            f"{_perf_value(route, 'graph_build'):7.3f} "
            f"{_perf_value(route, 'astar'):7.3f} "
            f"{_perf_value(route, 'diagnostics'):6.3f} "
            f"{_perf_value(route, 'plot_grid'):5.3f} "
            f"{_perf_value(route, 'total'):7.3f} "
            f"{int(route.nodes_explored):>10,}"
        )
    print(
        "  ALL  "
        f"{totals['setup']:7.3f} "
        f"{totals['graph_build']:7.3f} "
        f"{totals['astar']:7.3f} "
        f"{totals['diagnostics']:6.3f} "
        f"{totals['plot_grid']:5.3f} "
        f"{totals['total']:7.3f}"
    )
    if attempts or hits:
        print(f"  Corridor attempts: {attempts} (cache hits {hits})")
    print(f"{'═'*60}")


# ---------------------------------------------------------------------------
# Hourly position frames
# ---------------------------------------------------------------------------

def build_combined_timeline(routes, depart_time_s):
    """Concatenate simulated tracks from all legs into one timeline.

    Returns
    -------
    xs : np.ndarray  shape (N,)  UTM easting of each track point
    ys : np.ndarray  shape (N,)  UTM northing
    ts : np.ndarray  shape (N,)  elapsed seconds (same reference as CurrentField)
    """
    all_x, all_y, all_t = [], [], []
    for route in routes:
        trk = route.simulated_track
        trk_t = route.simulated_track_times
        if not trk or not trk_t:
            continue
        # Avoid duplicating the junction point between legs
        start_idx = 1 if all_x else 0
        for (x, y), t in zip(trk[start_idx:], trk_t[start_idx:]):
            all_x.append(x)
            all_y.append(y)
            all_t.append(t)

    return np.array(all_x), np.array(all_y), np.array(all_t)


def position_at_time(xs, ys, ts, query_t):
    """Linear interpolation of (x, y) along the track at time query_t."""
    query_t = float(np.clip(query_t, ts[0], ts[-1]))
    idx = int(np.searchsorted(ts, query_t, side='right')) - 1
    idx = int(np.clip(idx, 0, len(ts) - 2))
    t0, t1 = ts[idx], ts[idx + 1]
    alpha = (query_t - t0) / (t1 - t0) if t1 > t0 else 0.0
    x = xs[idx] + alpha * (xs[idx + 1] - xs[idx])
    y = ys[idx] + alpha * (ys[idx + 1] - ys[idx])
    return float(x), float(y)


def generate_hourly_frames(routes, wps, cf, depart_utc, depart_time_s,
                           tz, task_name, plot_dir, slug, wind_field=None,
                           all_land_marks=None):
    """Generate one annotated PNG per hour of the trip.

    Each frame shows side-by-side panels for current and wind (if wind_field provided):
    - Current/wind speed heatmap at that exact hour (live forecast)
    - Current/wind direction arrows
    - Full planned route (all legs, light dashed)
    - Simulated ground track up to the current moment
    - Boat position marker
    - Start / end / waypoint markers
    - Time / SOG annotation
    """
    import datetime as _dt
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    import matplotlib.colors as mcolors
    from pyproj import Transformer

    transformer = cf.transformer
    inv_transformer = Transformer.from_crs(
        transformer.target_crs, transformer.source_crs, always_xy=True)

    # ── build a single bounding box covering all waypoints ────────────────
    all_utm = []
    for lat, lon in wps:
        x, y = transformer.transform(lon, lat)
        all_utm.append((x, y))

    pad = 4000.0
    res = 300.0  # display grid for binned SSCOFS vector averages
    x_min = min(p[0] for p in all_utm) - pad
    x_max = max(p[0] for p in all_utm) + pad
    y_min = min(p[1] for p in all_utm) - pad
    y_max = max(p[1] for p in all_utm) + pad
    xs = np.arange(x_min, x_max + res, res)
    ys_g = np.arange(y_min, y_max + res, res)

    # ── combined simulated track ──────────────────────────────────────────
    tx, ty, tt = build_combined_timeline(routes, depart_time_s)
    if len(tx) == 0:
        print("  No simulated track — skipping hourly frames.")
        return

    trip_duration_s = tt[-1] - tt[0]
    n_hours = int(np.ceil(trip_duration_s / 3600)) + 1

    # ── full simulated track for all legs (used as the "planned route" overlay) ──
    route_x = list(tx)
    route_y = list(ty)

    # ── maneuver overlays from route diagnostics ─────────────────────────
    maneuver_x, maneuver_y, maneuver_t, maneuver_type = [], [], [], []
    for route in routes:
        dbg = route.debug or {}
        mx = np.asarray(dbg.get("maneuver_x", []), dtype=np.float64)
        my = np.asarray(dbg.get("maneuver_y", []), dtype=np.float64)
        mt = np.asarray(dbg.get("maneuver_time_s", []), dtype=np.float64)
        mtype = np.asarray(dbg.get("maneuver_type", []), dtype=np.int8)
        maneuver_x.extend(mx.tolist())
        maneuver_y.extend(my.tolist())
        maneuver_t.extend(mt.tolist())
        maneuver_type.extend(mtype.tolist())
    maneuver_x = np.asarray(maneuver_x, dtype=np.float64)
    maneuver_y = np.asarray(maneuver_y, dtype=np.float64)
    maneuver_t = np.asarray(maneuver_t, dtype=np.float64)
    maneuver_type = np.asarray(maneuver_type, dtype=np.int8)

    # ── compute max current speed for consistent colorscale ──────────────
    speed_max = cf.max_current_speed * MS_TO_KNOTS

    # ── precompute wind speed max if wind field provided ──────────────────
    wind_max = 15.0  # default
    if wind_field is not None:
        try:
            wu_sample, wv_sample = wind_field.query(xs[len(xs)//2], ys_g[len(ys_g)//2], 0.0)
            wind_max = max(15.0, np.hypot(wu_sample, wv_sample) * MS_TO_KNOTS * 1.5)
        except Exception:
            pass

    print(f"\nGenerating {n_hours} hourly frames…")

    def _draw_route_overlay(ax, frame_local, track_x_past, track_y_past, bx, by, show_legend=True):
        """Draw route, track, waypoints and boat on an axis."""
        ax.plot(route_x, route_y,
                color="#4477aa", linewidth=1.5, linestyle="--",
                alpha=0.6, zorder=4, label="Planned route" if show_legend else None)

        if len(track_x_past) > 1:
            ax.plot(track_x_past, track_y_past,
                    color="#0055cc", linewidth=2.5, zorder=5,
                    label="Track so far" if show_legend else None,
                    path_effects=[pe.Stroke(linewidth=4.5,
                                            foreground="white",
                                            alpha=0.7),
                                  pe.Normal()])

        if maneuver_type.size:
            tack_mask = maneuver_type == 1
            if np.any(tack_mask):
                ax.plot(maneuver_x[tack_mask], maneuver_y[tack_mask], "o",
                        color="black", markersize=6, zorder=9,
                        markeredgecolor="white", markeredgewidth=1.0,
                        label="Tacks" if show_legend else None)
            gybe_mask = maneuver_type == 2
            if np.any(gybe_mask):
                ax.plot(maneuver_x[gybe_mask], maneuver_y[gybe_mask], "D",
                        color="black", markersize=6, zorder=9,
                        markeredgecolor="white", markeredgewidth=1.0,
                        label="Gybes" if show_legend else None)

        for k, (wx, wy) in enumerate(all_utm):
            if k == 0:
                ax.plot(wx, wy, "o", color="#2ecc71", markersize=11,
                        zorder=8, markeredgecolor="white",
                        markeredgewidth=2, label="Start" if show_legend else None)
            elif k == len(all_utm) - 1 and k != 0:
                ax.plot(wx, wy, "s", color="#e74c3c", markersize=11,
                        zorder=8, markeredgecolor="white",
                        markeredgewidth=2,
                        label="Finish" if show_legend else None)
            else:
                ax.plot(wx, wy, "D", color="#f39c12", markersize=8,
                        zorder=8, markeredgecolor="white",
                        markeredgewidth=1.5)

        for radius, alpha in [(18, 0.12), (13, 0.20), (9, 0.35)]:
            ax.plot(bx, by, "o", color="#ffdd00",
                    markersize=radius, alpha=alpha, zorder=9,
                    markeredgewidth=0)
        ax.plot(bx, by, "o", color="#ffdd00", markersize=8, zorder=10,
                markeredgecolor="white", markeredgewidth=2,
                label=f"Boat @ {frame_local:%I:%M %p}" if show_legend else None)

        # Constraint mark overlays
        if all_land_marks:
            for _mi, _m in enumerate(all_land_marks):
                _mx, _my = transformer.transform(float(_m["lon"]), float(_m["lat"]))
                _rnd = _m.get("rounding", "")
                _name = _m.get("name", "")
                _lbl = "Constraint mark" if show_legend and _mi == 0 else None
                ax.plot(_mx, _my, "D", color="#cc00cc", markersize=10,
                        zorder=11, markeredgecolor="white", markeredgewidth=1.5,
                        label=_lbl)
                _tag = _name if _name else ("P" if _rnd == "port" else "S")
                ax.text(_mx, _my + 250, _tag,
                        fontsize=6, ha="center", va="bottom", color="#cc00cc",
                        fontweight="bold", zorder=12)

    def _setup_ax(ax, title):
        """Common axis setup."""
        ax.set_facecolor("#f5f8fc")
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_xlabel("Easting (m)", fontsize=8)
        ax.set_ylabel("Northing (m)", fontsize=8)
        ax.tick_params(labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")
        ax.set_aspect("equal")
        ax.set_xlim(xs[0], xs[-1])
        ax.set_ylim(ys_g[0], ys_g[-1])
        ax.grid(True, alpha=0.25, color="#aaaaaa")
        draw_shoreline(ax, transformer, zorder=3)

    for h in range(n_hours):
        elapsed_s = depart_time_s + h * 3600.0
        elapsed_s = float(np.clip(elapsed_s, tt[0], tt[-1]))
        frame_utc  = depart_utc + _dt.timedelta(seconds=h * 3600)
        frame_local = frame_utc.astimezone(tz)

        # Current field at this hour
        u_grid, v_grid = cf.query_binned_grid(xs, ys_g, elapsed_s=elapsed_s)
        speed_grid = np.hypot(u_grid, v_grid) * MS_TO_KNOTS
        water_mask = ~np.isnan(u_grid)
        speed_grid[~water_mask] = np.nan
        frame_speed_max = (np.nanpercentile(speed_grid, 95)
                           if np.any(~np.isnan(speed_grid)) else 1.0)

        # Boat position at this hour
        bx, by = position_at_time(tx, ty, tt, elapsed_s)

        # Track so far (up to this moment)
        mask_past = tt <= elapsed_s + 1.0
        track_x_past = tx[mask_past]
        track_y_past = ty[mask_past]

        # Arrow subsampling
        step = max(1, int(round(900.0 / res)))
        xx, yy = np.meshgrid(xs, ys_g)

        if wind_field is None:
            fig, ax = plt.subplots(figsize=(10, 14))

            _setup_ax(ax, f"{task_name}  —  Hour {h}  —  {frame_local:%I:%M %p %Z}")

            ax.pcolormesh(xs, ys_g, np.where(water_mask, 0.0, np.nan),
                          cmap="Blues", vmin=0, vmax=1, alpha=0.08,
                          shading="auto", zorder=0)

            im = ax.pcolormesh(xs, ys_g, speed_grid,
                               cmap="plasma", alpha=0.5, shading="auto",
                               vmin=0, vmax=max(frame_speed_max, 1.0), zorder=1)
            cbar = fig.colorbar(im, ax=ax, label="Current (kt)",
                                pad=0.01, shrink=0.6, fraction=0.03)

            u_sub = u_grid[::step, ::step]
            v_sub = v_grid[::step, ::step]
            spd_sub = speed_grid[::step, ::step]
            mag = np.hypot(u_sub, v_sub)
            mag_safe = np.where(mag < 1e-8, 1.0, mag)
            u_arrow = u_sub / mag_safe
            v_arrow = v_sub / mag_safe
            visible = (mag > 0.04) & ~np.isnan(spd_sub)
            u_arrow[~visible] = np.nan
            v_arrow[~visible] = np.nan
            ax.quiver(xx[::step, ::step], yy[::step, ::step],
                      u_arrow, v_arrow,
                      color="black",
                      scale=1.0 / (step * res) * 1.0, scale_units="xy",
                      width=0.003, headwidth=4, headlength=5,
                      alpha=0.75, zorder=2.5)

            _draw_route_overlay(ax, frame_local, track_x_past, track_y_past, bx, by)
            ax.legend(loc="upper right", framealpha=0.9,
                      facecolor="white", edgecolor="#cccccc", fontsize=8)
        else:
            fig, (ax_cur, ax_wind) = plt.subplots(1, 2, figsize=(16, 10))

            _setup_ax(ax_cur, "Ocean Current")
            _setup_ax(ax_wind, "Wind")

            for ax in (ax_cur, ax_wind):
                ax.pcolormesh(xs, ys_g, np.where(water_mask, 0.0, np.nan),
                              cmap="Blues", vmin=0, vmax=1, alpha=0.08,
                              shading="auto", zorder=0)

            im_cur = ax_cur.pcolormesh(xs, ys_g, speed_grid,
                                       cmap="plasma", alpha=0.5, shading="auto",
                                       vmin=0, vmax=max(frame_speed_max, 1.0), zorder=1)
            cbar_cur = fig.colorbar(im_cur, ax=ax_cur, label="Current (kt)",
                                    pad=0.02, shrink=0.7)

            u_sub = u_grid[::step, ::step]
            v_sub = v_grid[::step, ::step]
            spd_sub = speed_grid[::step, ::step]
            mag = np.hypot(u_sub, v_sub)
            mag_safe = np.where(mag < 1e-8, 1.0, mag)
            u_arrow = u_sub / mag_safe
            v_arrow = v_sub / mag_safe
            visible = (mag > 0.04) & ~np.isnan(spd_sub)
            u_arrow[~visible] = np.nan
            v_arrow[~visible] = np.nan
            ax_cur.quiver(xx[::step, ::step], yy[::step, ::step],
                          u_arrow, v_arrow,
                          color="black",
                          scale=1.0 / (step * res) * 1.0, scale_units="xy",
                          width=0.003, headwidth=4, headlength=5,
                          alpha=0.75, zorder=2.5)

            wu_grid = np.zeros_like(u_grid)
            wv_grid = np.zeros_like(v_grid)
            for i, y in enumerate(ys_g):
                for j, x in enumerate(xs):
                    if water_mask[i, j]:
                        wu, wv = wind_field.query(x, y, elapsed_s=elapsed_s)
                        wu_grid[i, j] = wu
                        wv_grid[i, j] = wv
                    else:
                        wu_grid[i, j] = np.nan
                        wv_grid[i, j] = np.nan

            wind_speed_grid = np.hypot(wu_grid, wv_grid) * MS_TO_KNOTS

            im_wind = ax_wind.pcolormesh(xs, ys_g, wind_speed_grid,
                                         cmap="YlOrRd", alpha=0.5, shading="auto",
                                         vmin=0, vmax=wind_max, zorder=1)
            cbar_wind = fig.colorbar(im_wind, ax=ax_wind, label="Wind (kt)",
                                     pad=0.02, shrink=0.7)

            wu_sub = wu_grid[::step, ::step]
            wv_sub = wv_grid[::step, ::step]
            wspd_sub = wind_speed_grid[::step, ::step]
            wmag = np.hypot(wu_sub, wv_sub)
            wmag_safe = np.where(wmag < 1e-8, 1.0, wmag)
            wu_norm = wu_sub / wmag_safe
            wv_norm = wv_sub / wmag_safe
            wvisible = (wmag > 0.1) & ~np.isnan(wspd_sub)
            wu_norm[~wvisible] = np.nan
            wv_norm[~wvisible] = np.nan
            ax_wind.quiver(xx[::step, ::step], yy[::step, ::step],
                           wu_norm, wv_norm,
                           color="black",
                           scale=1.0 / (step * res) * 1.0, scale_units="xy",
                           width=0.003, headwidth=4, headlength=5,
                           alpha=0.75, zorder=2.5)

            _draw_route_overlay(ax_cur, frame_local, track_x_past, track_y_past, bx, by, show_legend=True)
            _draw_route_overlay(ax_wind, frame_local, track_x_past, track_y_past, bx, by, show_legend=False)
            ax_cur.legend(loc="upper right", framealpha=0.9,
                          facecolor="white", edgecolor="#cccccc", fontsize=7)

        # ── Annotations ───────────────────────────────────────────────────
        time_str = frame_local.strftime("%I:%M %p %Z")
        date_str = frame_local.strftime("%a %b %-d")
        frac = min((elapsed_s - tt[0]) / max(trip_duration_s, 1), 1.0)
        elapsed_h = (elapsed_s - depart_time_s) / 3600.0

        ann_text = (f"{date_str}\n{time_str}\n"
                    f"Hour {h} of trip\n"
                    f"Progress: {100*frac:.0f}%")

        ax_for_ann = ax if wind_field is None else ax_cur
        ax_for_ann.text(0.02, 0.98, ann_text,
                        transform=ax_for_ann.transAxes,
                        fontsize=9, va="top", ha="left",
                        color="#1a1a2a", fontfamily="monospace",
                        bbox=dict(boxstyle="round,pad=0.4",
                                  facecolor="white", alpha=0.85,
                                  edgecolor="#cccccc"))

        if len(track_x_past) > 1:
            seg_dists = np.hypot(np.diff(track_x_past), np.diff(track_y_past))
            cum_dist_nm = seg_dists.sum() / 1852.0
            avg_sog = cum_dist_nm / elapsed_h if elapsed_h > 0 else 0.0
            stats_text = (f"Dist: {cum_dist_nm:.1f} nm\n"
                          f"SOG:  {avg_sog:.2f} kt")
            ax_for_ann.text(0.02, 0.02, stats_text,
                            transform=ax_for_ann.transAxes,
                            fontsize=8, va="bottom", ha="left",
                            color="#1a1a2a", fontfamily="monospace",
                            bbox=dict(boxstyle="round,pad=0.3",
                                      facecolor="white", alpha=0.85,
                                      edgecolor="#cccccc"))

        if wind_field is not None:
            fig.suptitle(f"{task_name}  —  Hour {h}  —  {frame_local:%I:%M %p %Z}",
                         fontsize=12, y=0.98)

        plt.tight_layout()
        out_path = plot_dir / f"{slug}_hour{h:02d}.png"
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  Hour {h:02d}  {frame_local:%I:%M %p %Z}  "
              f"boat=({bx:.0f},{by:.0f})  → {out_path.name}")

    print(f"Hourly frames complete: {n_hours} PNGs in {plot_dir}/")


# ---------------------------------------------------------------------------
# Route data export
# ---------------------------------------------------------------------------

def save_route_json(routes, wps, depart_utc, depart_time_s, tz,
                    task_name, plot_dir, slug):
    """Save route data to a JSON file for later analysis.

    Structure
    ---------
    {
      "task": { name, departure_utc, departure_local },
      "legs": [
        { "from", "to", "distance_nm", "time_min", "avg_sog_kt",
          "waypoints": [[lat, lon], ...] }
      ],
      "trip": { "distance_nm", "time_hr", "avg_sog_kt" },
      "track": [
        { "elapsed_s", "lat", "lon", "leg" }, ...
      ]
    }
    """
    import json
    import datetime as _dt
    from pyproj import CRS, Transformer as _T

    ref_lat, ref_lon = wps[0]
    utm_crs = CRS.from_dict({
        "proj": "utm",
        "zone": int((ref_lon + 180) / 6) + 1,
        "north": ref_lat >= 0,
        "ellps": "WGS84",
    })
    inv_tf = _T.from_crs(utm_crs, CRS.from_epsg(4326), always_xy=True)

    dep_local = depart_utc.astimezone(tz)

    # ── per-leg summaries ─────────────────────────────────────────────────
    legs_out = []
    for li, r in enumerate(routes):
        wp_ll = [[float(lat), float(lon)] for lat, lon in r.waypoints_latlon]
        dbg = r.debug or {}
        tack_count = int(dbg.get("maneuver_tack_count", 0))
        gybe_count = int(dbg.get("maneuver_gybe_count", 0))
        turn_count = int(dbg.get("maneuver_turn_count", 0))
        maneuvers = []
        if "maneuver_type" in dbg:
            mtype_names = {1: "tack", 2: "gybe", 3: "turn"}
            mx = np.asarray(dbg.get("maneuver_x", []), dtype=np.float64)
            my = np.asarray(dbg.get("maneuver_y", []), dtype=np.float64)
            mt = np.asarray(dbg.get("maneuver_time_s", []), dtype=np.float64)
            mtypes = np.asarray(dbg.get("maneuver_type", []), dtype=np.int8)
            for x, y, t_s, mtype in zip(mx, my, mt, mtypes):
                lon, lat = inv_tf.transform(float(x), float(y))
                maneuvers.append({
                    "type": mtype_names.get(int(mtype), "unknown"),
                    "elapsed_s": round(float(t_s - depart_time_s), 1),
                    "lat": round(float(lat), 6),
                    "lon": round(float(lon), 6),
                })
        legs_out.append({
            "leg": li + 1,
            "from": list(wps[li]),
            "to":   list(wps[li + 1]),
            "distance_nm": round(r.total_distance_m / 1852.0, 3),
            "time_min":    round(r.total_time_s / 60.0, 1),
            "avg_sog_kt":  round(r.avg_sog_knots, 3),
            "tack_count": tack_count,
            "gybe_count": gybe_count,
            "turn_count": turn_count,
            "maneuvers": maneuvers,
            "waypoints_latlon": wp_ll,
        })

    total_dist = sum(r.total_distance_m for r in routes) / 1852.0
    total_time = sum(r.total_time_s for r in routes)

    # ── dense simulated track ─────────────────────────────────────────────
    track_out = []
    cursor = depart_time_s
    for li, r in enumerate(routes):
        trk   = r.simulated_track or []
        trk_t = r.simulated_track_times or []
        start_idx = 1 if li > 0 else 0
        for (x, y), t in zip(trk[start_idx:], trk_t[start_idx:]):
            lon, lat = inv_tf.transform(float(x), float(y))
            wall = (depart_utc + _dt.timedelta(
                        seconds=float(t - depart_time_s))).isoformat()
            track_out.append({
                "elapsed_s":  round(float(t - depart_time_s), 1),
                "wall_utc":   wall,
                "lat":        round(lat, 6),
                "lon":        round(lon, 6),
                "leg":        li + 1,
            })
        cursor += r.total_time_s

    out = {
        "task": {
            "name":            task_name,
            "departure_utc":   depart_utc.isoformat(),
            "departure_local": dep_local.isoformat(),
        },
        "legs":  legs_out,
        "trip": {
            "distance_nm": round(total_dist, 3),
            "time_hr":     round(total_time / 3600.0, 3),
            "avg_sog_kt":  round(total_dist / (total_time / 3600.0), 3),
        },
        "track": track_out,
    }

    out_path = plot_dir / f"{slug}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Route data saved  → {out_path.name}  "
          f"({len(track_out):,} track points)")
    return out_path


def save_route_report(routes, wps, depart_utc, depart_time_s, tz,
                      task_name, plot_dir, slug, doc):
    """Save a human-readable Markdown report next to the JSON output."""
    import datetime as _dt
    from pyproj import CRS, Transformer as _T

    ref_lat, ref_lon = wps[0]
    utm_crs = CRS.from_dict({
        "proj": "utm",
        "zone": int((ref_lon + 180) / 6) + 1,
        "north": ref_lat >= 0,
        "ellps": "WGS84",
    })
    inv_tf = _T.from_crs(utm_crs, CRS.from_epsg(4326), always_xy=True)

    total_time_s = sum(r.total_time_s for r in routes)
    total_dist_nm = sum(r.total_distance_m for r in routes) / 1852.0
    depart_local = depart_utc.astimezone(tz)
    arrive_local = (depart_utc + _dt.timedelta(seconds=total_time_s)).astimezone(tz)
    avg_sog = total_dist_nm / (total_time_s / 3600.0) if total_time_s > 0 else 0.0

    wind_cfg = doc.get("wind") or {}
    wind_source = str(wind_cfg.get("source", "none"))
    if wind_source.lower() in ("open_meteo_ecmwf", "ecmwf_openmeteo", "ecmwf_route"):
        wind_desc = (
            "Open-Meteo ECMWF `wind_speed_10m` and `wind_direction_10m` "
            "(10 m sustained/average wind). `wind_gusts_10m` is fetched into "
            "the cache for reference/coverage checks but is not used by the router."
        )
    elif wind_source.lower() in ("schedule", "manual_schedule", "piecewise"):
        wind_desc = "Manual schedule wind from YAML anchors."
    elif wind_source.lower() in ("constant", "met", "manual"):
        wind_desc = "Constant wind from YAML."
    else:
        wind_desc = wind_source

    lines = [
        f"# {task_name}",
        "",
        "## Summary",
        "",
        f"- Departure: {depart_local:%Y-%m-%d %I:%M %p %Z}",
        f"- Estimated finish: {arrive_local:%Y-%m-%d %I:%M %p %Z}",
        f"- Total time: {total_time_s / 3600.0:.2f} hr ({total_time_s / 60.0:.0f} min)",
        f"- Total distance: {total_dist_nm:.2f} nm",
        f"- Average SOG: {avg_sog:.2f} kt",
        "",
        "## Wind",
        "",
        f"- Source: `{wind_source}`",
        f"- Details: {wind_desc}",
        "",
        "## Routing Settings",
        "",
        f"- `polar_sweep_coarse_step`: {doc.get('routing', {}).get('polar_sweep_coarse_step', SectorRouter.POLAR_SWEEP_COARSE_STEP)}",
        f"- `use_dense_polar`: {doc.get('routing', {}).get('use_dense_polar', False)}",
        f"- `use_dot_filter`: {doc.get('routing', {}).get('use_dot_filter', False)}",
        f"- `k_candidates`: {doc.get('routing', {}).get('k_candidates', 50)}",
        f"- `min_edge_m`: {doc.get('routing', {}).get('min_edge_m', 50.0)}",
        f"- `max_edge_m`: {doc.get('routing', {}).get('max_edge_m', 2000.0)}",
        f"- `los_samples`: {doc.get('routing', {}).get('los_samples', 15)}",
        f"- `corridor_pad_factors`: {doc.get('routing', {}).get('corridor_pad_factors', list(SectorRouter.CORRIDOR_PAD_FACTORS))}",
        "",
        "## Leg Breakdown",
        "",
        "| Leg | From -> To | Depart | Arrive | Distance | Time | Avg SOG | Tacks | Gybes |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|",
    ]

    leg_depart = depart_utc
    for i, route in enumerate(routes):
        leg_arrive = leg_depart + _dt.timedelta(seconds=route.total_time_s)
        dbg = route.debug or {}
        tacks = int(dbg.get("maneuver_tack_count", 0))
        gybes = int(dbg.get("maneuver_gybe_count", 0))
        from_ll = f"{wps[i][0]:.4f},{wps[i][1]:.4f}"
        to_ll = f"{wps[i + 1][0]:.4f},{wps[i + 1][1]:.4f}"
        lines.append(
            f"| {i + 1} | {from_ll} -> {to_ll} | "
            f"{leg_depart.astimezone(tz):%I:%M %p} | "
            f"{leg_arrive.astimezone(tz):%I:%M %p} | "
            f"{route.total_distance_m / 1852.0:.2f} nm | "
            f"{route.total_time_s / 60.0:.1f} min | "
            f"{route.avg_sog_knots:.2f} kt | {tacks} | {gybes} |"
        )
        leg_depart = leg_arrive

    if any((r.debug or {}).get("perf") for r in routes):
        lines.extend([
            "",
            "## Router Performance",
            "",
            "| Leg | Setup | Graph | A* | Diagnostics | Plot Grid | Total | Explored |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        total_perf = {
            key: sum(_perf_value(r, key) for r in routes)
            for key in ("setup", "graph_build", "astar", "diagnostics", "plot_grid", "total")
        }
        for i, route in enumerate(routes, 1):
            lines.append(
                f"| {i} | "
                f"{_perf_value(route, 'setup'):.3f}s | "
                f"{_perf_value(route, 'graph_build'):.3f}s | "
                f"{_perf_value(route, 'astar'):.3f}s | "
                f"{_perf_value(route, 'diagnostics'):.3f}s | "
                f"{_perf_value(route, 'plot_grid'):.3f}s | "
                f"{_perf_value(route, 'total'):.3f}s | "
                f"{int(route.nodes_explored):,} |"
            )
        lines.append(
            f"| **All** | "
            f"{total_perf['setup']:.3f}s | "
            f"{total_perf['graph_build']:.3f}s | "
            f"{total_perf['astar']:.3f}s | "
            f"{total_perf['diagnostics']:.3f}s | "
            f"{total_perf['plot_grid']:.3f}s | "
            f"{total_perf['total']:.3f}s |  |"
        )

    lines.extend(["", "## Maneuvers", ""])
    any_maneuvers = False
    for i, route in enumerate(routes):
        dbg = route.debug or {}
        mtypes = np.asarray(dbg.get("maneuver_type", []), dtype=np.int8)
        mtimes = np.asarray(dbg.get("maneuver_time_s", []), dtype=np.float64)
        mx = np.asarray(dbg.get("maneuver_x", []), dtype=np.float64)
        my = np.asarray(dbg.get("maneuver_y", []), dtype=np.float64)
        if not len(mtypes):
            continue
        any_maneuvers = True
        lines.append(f"### Leg {i + 1}")
        lines.append("")
        lines.append("| Type | Local Time | Latitude | Longitude |")
        lines.append("|---|---|---:|---:|")
        for mtype, t_s, x, y in zip(mtypes, mtimes, mx, my):
            kind = {1: "tack", 2: "gybe", 3: "turn"}.get(int(mtype), "unknown")
            lon, lat = inv_tf.transform(float(x), float(y))
            local = (depart_utc + _dt.timedelta(seconds=float(t_s - depart_time_s))).astimezone(tz)
            lines.append(f"| {kind} | {local:%I:%M:%S %p} | {lat:.6f} | {lon:.6f} |")
        lines.append("")

    if not any_maneuvers:
        lines.append("No tacks, gybes, or large turns detected.")
        lines.append("")

    out_path = plot_dir / f"{slug}_report.md"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
        f.write("\n")
    print(f"Human-readable report saved → {out_path.name}")
    return out_path


# ---------------------------------------------------------------------------
# Position time-series plot
# ---------------------------------------------------------------------------

def plot_position_timeseries(routes, wps, depart_utc, depart_time_s, tz,
                             task_name, plot_dir, slug):
    """Two-panel time series: UTM northing and easting vs clock time.

    The N/S panel shows northing in metres and the E/W panel shows easting in
    metres. Using the native UTM coordinates keeps the plots consistent with
    the routing solver and avoids distortion from degree-based axes.
    """
    import datetime as _dt
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    tx, ty, tt = build_combined_timeline(routes, depart_time_s)
    if len(tx) == 0:
        return

    # Absolute wall-clock times: UTC-aware first, then converted to local tz for display
    t_utc = [depart_utc + _dt.timedelta(seconds=float(t - depart_time_s))
             for t in tt]
    t_local_tz = [t.astimezone(tz) for t in t_utc]
    t_hours = np.array([(t - t_local_tz[0]).total_seconds() / 3600
                        for t in t_local_tz])

    # Leg boundary times (relative hours from departure)
    leg_boundaries = []
    elapsed = 0.0
    for r in routes[:-1]:
        elapsed += r.total_time_s
        leg_boundaries.append(elapsed / 3600.0)

    # ── figure ────────────────────────────────────────────────────────────
    fig, (ax_ns, ax_ew) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    fig.patch.set_facecolor("#0f1923")
    for ax in (ax_ns, ax_ew):
        ax.set_facecolor("#111827")
        ax.tick_params(colors="#aaaaaa")
        ax.xaxis.label.set_color("#aaaaaa")
        ax.yaxis.label.set_color("#aaaaaa")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a3a4a")
        ax.grid(True, alpha=0.15, color="#334455")

    # Colour track by leg: each leg gets a distinct hue
    leg_colors = ["#00e5ff", "#ff9100", "#a0e060", "#ff4f81"]

    # Build per-leg index masks
    cursor = depart_time_s
    for li, r in enumerate(routes):
        leg_end = cursor + r.total_time_s
        mask = (tt >= cursor - 1) & (tt <= leg_end + 1)
        lbl = f"Leg {li+1}"
        col = leg_colors[li % len(leg_colors)]
        ax_ns.plot(t_hours[mask], ty[mask],
                   color=col, linewidth=1.8, label=lbl, alpha=0.9)
        ax_ew.plot(t_hours[mask], tx[mask],
                   color=col, linewidth=1.8, label=lbl, alpha=0.9)
        cursor = leg_end

    # Leg boundary vertical lines
    for lb in leg_boundaries:
        for ax in (ax_ns, ax_ew):
            ax.axvline(lb, color="#ffffff", linewidth=1.0,
                       linestyle="--", alpha=0.35)

    # Waypoint reference lines in the same UTM frame as the route.
    all_utm = [routes[0].waypoints_utm[0]]
    all_utm.extend(route.waypoints_utm[-1] for route in routes)
    for k, (wx, wy) in enumerate(all_utm):
        lbl = f"WP{k}" if 0 < k < len(wps) - 1 else ("Start" if k == 0 else "Finish")
        ax_ns.axhline(wy, color="#ffdd00", linewidth=0.8,
                      linestyle=":", alpha=0.4,
                      label=lbl if k in (0, len(wps)-1) else None)
        ax_ew.axhline(wx, color="#ffdd00", linewidth=0.8,
                      linestyle=":", alpha=0.4)

    # Hourly tick markers on the track
    trip_h = t_hours[-1]
    for h in range(int(np.floor(trip_h)) + 1):
        if h > trip_h:
            break
        for ax, coord in ((ax_ns, ty), (ax_ew, tx)):
            pos = float(np.interp(h, t_hours, coord))
            ax.plot(h, pos, "o", color="white", markersize=5,
                    zorder=5, markeredgewidth=0, alpha=0.6)
            ax.text(h, pos, f" {h}h", color="#aaaaaa", fontsize=7,
                    va="center", fontfamily="monospace")

    ax_ns.set_ylabel("Northing (m, UTM)", fontsize=10)
    ax_ew.set_ylabel("Easting (m, UTM)", fontsize=10)
    ax_ew.set_xlabel("Elapsed time (hours from departure)", fontsize=10)

    ax_ns.set_title(f"{task_name}  —  N/S position vs time", fontsize=12)
    ax_ew.set_title("E/W position vs time  (zigzag = tacking)", fontsize=12)

    # Secondary x-axis showing clock time at integer hours
    dep_local = depart_utc.astimezone(tz)
    xtick_h = np.arange(0, int(trip_h) + 2)
    xtick_labels = [(dep_local + _dt.timedelta(hours=float(h))).strftime("%-I%p")
                    for h in xtick_h]
    ax_ew.set_xticks(xtick_h)
    ax_ew.set_xticklabels(xtick_labels, color="#aaaaaa", fontsize=8)
    ax_ew.xaxis.set_minor_locator(mticker.MultipleLocator(0.25))

    # Leave latitude on the normal matplotlib axis orientation so larger
    # latitudes plot higher.  That keeps southbound motion trending downward
    # and northbound motion trending upward, which matches the geography.

    ax_ns.legend(loc="lower left", framealpha=0.6, facecolor="#0d1b2a",
                 edgecolor="#334455", labelcolor="white", fontsize=9)

    # Leg labels at midpoint of each leg
    cursor = 0.0
    for li, r in enumerate(routes):
        mid_h = (cursor + r.total_time_s / 2) / 3600.0
        ax_ns.text(mid_h, ax_ns.get_ylim()[0],
                   f"Leg {li+1}", color=leg_colors[li % len(leg_colors)],
                   fontsize=9, ha="center", va="bottom",
                   fontfamily="monospace", alpha=0.8)
        cursor += r.total_time_s

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = plot_dir / f"{slug}_timeseries.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Time-series plot saved → {out_path.name}")


def plot_clean_route_map(routes, wps, transformer, plot_dir, slug,
                         leg_marks=None, course_feature_centers=None):
    """Clean route-only overview: no wind/current layers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    import matplotlib.patheffects as pe

    if not routes:
        return None

    leg_marks = leg_marks or {}
    fig, ax = plt.subplots(figsize=(14, 13))
    fig.patch.set_facecolor("white")

    all_x, all_y = [], []
    leg_tracks = []
    for route in routes:
        if route.simulated_track and len(route.simulated_track) > 1:
            track = np.asarray(route.simulated_track, dtype=np.float64)
        else:
            track = np.asarray(route.waypoints_utm, dtype=np.float64)
        leg_tracks.append(track)
        all_x.extend(track[:, 0].tolist())
        all_y.extend(track[:, 1].tolist())

    route_waypoints = []
    route_waypoints.append(routes[0].waypoints_utm[0])
    route_waypoints.extend(route.waypoints_utm[-1] for route in routes)
    for wx, wy in route_waypoints:
        all_x.append(wx)
        all_y.append(wy)

    mark_entries = []
    for leg_idx in sorted(leg_marks):
        for m in leg_marks[leg_idx]:
            mx, my = transformer.transform(float(m["lon"]), float(m["lat"]))
            mark_entries.append((leg_idx, mx, my, m))
            all_x.append(mx)
            all_y.append(my)

    feature_entries = []
    for key, feature in (course_feature_centers or {}).items():
        if not isinstance(feature, dict):
            continue
        if "lat" not in feature or "lon" not in feature:
            continue
        fx, fy = transformer.transform(float(feature["lon"]), float(feature["lat"]))
        feature_entries.append((key, fx, fy, feature))
        all_x.append(fx)
        all_y.append(fy)

    for leg_idx, _, _, _ in mark_entries:
        barriers = compute_barriers(leg_marks.get(leg_idx, []), transformer)
        for bar in barriers:
            all_x.extend([bar[0], bar[2]])
            all_y.extend([bar[1], bar[3]])

    x_span = max(all_x) - min(all_x) if all_x else 1.0
    y_span = max(all_y) - min(all_y) if all_y else 1.0
    pad = max(1500.0, 0.08 * max(x_span, y_span))
    ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax.set_ylim(min(all_y) - pad, max(all_y) + pad)

    ax.set_facecolor("#f7fbff")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#d0d7de", alpha=0.45, linewidth=0.8)
    ax.tick_params(labelsize=9)
    ax.set_xlabel("Easting (m, UTM)", fontsize=9)
    ax.set_ylabel("Northing (m, UTM)", fontsize=9)
    draw_shoreline(ax, transformer, zorder=1)

    barrier_label_used = False
    for leg_idx in sorted(leg_marks):
        barriers = compute_barriers(leg_marks.get(leg_idx, []), transformer)
        for bar in barriers:
            ax.plot(
                [bar[0], bar[2]], [bar[1], bar[3]],
                "--", color="#c026d3", linewidth=1.2, alpha=0.45,
                zorder=2, label="Barrier" if not barrier_label_used else None,
            )
            barrier_label_used = True

    leg_colors = ["#dc2626", "#2563eb", "#f97316", "#16a34a"]
    for i, track in enumerate(leg_tracks):
        ax.plot(
            track[:, 0], track[:, 1],
            color=leg_colors[i % len(leg_colors)], linewidth=3.0, zorder=5,
            label=f"Leg {i + 1}",
            path_effects=[pe.Stroke(linewidth=5.0, foreground="white", alpha=0.9),
                          pe.Normal()],
        )

    key_lines = []
    for i, (wx, wy) in enumerate(route_waypoints):
        if i == 0:
            tag, color, marker, name = "S", "#16a34a", "o", "Start"
        elif i == len(route_waypoints) - 1:
            tag, color, marker, name = "F", "#f97316", "s", "Finish seed"
        else:
            tag, color, marker, name = f"R{i}", "#111827", "o", f"Race mark {i}"
        ax.plot(wx, wy, marker, color=color, markersize=10, zorder=8,
                markeredgecolor="white", markeredgewidth=1.4)
        ax.text(wx, wy, tag, ha="center", va="center", fontsize=7,
                color="white", fontweight="bold", zorder=9)
        key_lines.append(f"{tag}: {name}")

    for mi, (leg_idx, mx, my, m) in enumerate(mark_entries, 1):
        tag = f"C{mi}"
        ax.plot(mx, my, "D", color="#c026d3", markersize=9, zorder=8,
                markeredgecolor="white", markeredgewidth=1.2)
        ax.text(mx, my, tag, ha="center", va="center", fontsize=7,
                color="white", fontweight="bold", zorder=9)
        name = str(m.get("name", f"constraint {mi}"))
        name = name.replace(" (constraint point)", "")
        key_lines.append(f"{tag}: {name}")

    for fi, (key, fx, fy, feature) in enumerate(feature_entries, 1):
        tag = f"F{fi}"
        ax.plot(fx, fy, "o", color="#0ea5e9", markersize=8, zorder=8,
                markeredgecolor="white", markeredgewidth=1.3)
        ax.text(fx, fy, tag, ha="center", va="center", fontsize=6,
                color="white", fontweight="bold", zorder=9)
        label = str(feature.get("name", key.replace("_", " ").title()))
        key_lines.append(f"{tag}: {label}")

    if key_lines:
        split_at = (len(key_lines) + 1) // 2
        key_text = "\n".join(key_lines[:split_at])
        if split_at < len(key_lines):
            key_text += "\n\n" + "\n".join(key_lines[split_at:])
        ax.text(
            0.015, 0.985, key_text,
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            color="#111827",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#d1d5db", alpha=0.9),
            zorder=20,
        )

    total_nm = sum(r.total_distance_m for r in routes) / 1852.0
    total_min = sum(r.total_time_s for r in routes) / 60.0
    ax.set_title(
        f"{slug} race marks and constraints  |  {total_nm:.1f} nm, {total_min:.0f} min",
        fontsize=13,
    )

    handles = [
        mlines.Line2D([], [], color="#dc2626", linewidth=3, label="Route legs"),
        mlines.Line2D([], [], marker="o", color="none", markerfacecolor="#16a34a",
                      markeredgecolor="white", markersize=9, label="Start"),
        mlines.Line2D([], [], marker="s", color="none", markerfacecolor="#f97316",
                      markeredgecolor="white", markersize=9, label="End"),
        mlines.Line2D([], [], marker="D", color="none", markerfacecolor="#c026d3",
                      markeredgecolor="white", markersize=8, label="Constraint mark"),
        mlines.Line2D([], [], marker="o", color="none", markerfacecolor="#0ea5e9",
                      markeredgecolor="white", markersize=8, label="Feature center"),
        mlines.Line2D([], [], color="#c026d3", linestyle="--", linewidth=1.2,
                      label="Barrier"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.015))
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    out_path = plot_dir / f"{slug}_clean_map.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Clean route map saved → {out_path.name}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import datetime as _dt
    from zoneinfo import ZoneInfo

    parser = argparse.ArgumentParser(
        description="Run a multi-leg route from a YAML task file")
    parser.add_argument("yaml_file", help="Path to routing task YAML")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force fresh SSCOFS download (ignore local cache)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip all matplotlib output")
    # --- race-mode / web publishing flags (see plan: races to PWA) ---
    parser.add_argument("--race-mode", action="store_true",
                        help="Requires a race: block in the YAML. Implies "
                             "--no-plots, --geojson, --publish-s3. Overrides "
                             "departure from race.event_start_utc.")
    parser.add_argument("--geojson", action="store_true",
                        help="Also emit <slug>.geojson alongside the JSON.")
    parser.add_argument("--publish-s3", action="store_true",
                        help="Upload route.json + route.geojson + manifest.json "
                             "to s3://{race.s3_bucket}/{race.s3_prefix}/.")
    parser.add_argument("--only-if-within", type=float, default=None,
                        metavar="HOURS",
                        help="Exit 0 without running if the race start is "
                             "further than HOURS away. Used by scheduled GHA.")
    parser.add_argument("--min-lead-min", type=float, default=None,
                        metavar="MIN",
                        help="Exit 0 without running if the race start is "
                             "closer than MIN minutes away, or already "
                             "passed. Freezes the published nominal route "
                             "before start so live reroute takes over.")
    args = parser.parse_args()

    # --race-mode implies --no-plots / --geojson / --publish-s3
    if args.race_mode:
        args.no_plots = True
        args.geojson = True
        args.publish_s3 = True

    yaml_path = Path(args.yaml_file)
    if not yaml_path.is_absolute():
        yaml_path = HERE / yaml_path
    if not yaml_path.exists():
        sys.exit(f"YAML file not found: {yaml_path}")

    doc = load_task(yaml_path)

    task_name = doc.get("task", {}).get("name", yaml_path.stem)
    print(f"\n{'═'*60}")
    print(f"Route task: {task_name}")
    if desc := doc.get("task", {}).get("description"):
        print(f"  {desc.strip()}")
    print(f"{'═'*60}\n")

    # ---- Race block (optional; required for --race-mode) ----
    race_cfg = doc.get("race") or {}
    if args.race_mode and not race_cfg.get("slug"):
        sys.exit("--race-mode requires a race: block with a slug in the YAML")

    # ---- Departure time ----
    dep_cfg = doc["departure"]
    tz_str = dep_cfg.get("tz", "America/Los_Angeles")
    tz = ZoneInfo(tz_str)
    naive = _dt.datetime.strptime(dep_cfg["datetime"], "%Y-%m-%d %H:%M")
    depart_dt = naive.replace(tzinfo=tz)
    depart_utc = depart_dt.astimezone(_dt.timezone.utc)

    # Race mode anchors departure to event_start_utc, ignoring the
    # ad-hoc `departure.datetime` (which is useful for local dry runs only).
    if args.race_mode:
        ev_str = race_cfg["event_start_utc"]
        depart_utc = _dt.datetime.fromisoformat(ev_str.replace("Z", "+00:00"))
        depart_dt = depart_utc.astimezone(tz)
        print(f"Race mode: depart_utc = {depart_utc.isoformat()} "
              f"(slug={race_cfg['slug']})")

    # --only-if-within / --min-lead-min: bound the precompute window.
    # Together they say: "publish the route only while the race is within
    # WINDOW hours out AND at least LEAD minutes away from start."
    if args.only_if_within is not None or args.min_lead_min is not None:
        now_utc = _dt.datetime.now(_dt.timezone.utc)
        delta_s = (depart_utc - now_utc).total_seconds()
        hours_to_start = delta_s / 3600.0
        mins_to_start  = delta_s / 60.0
        if args.only_if_within is not None and hours_to_start > args.only_if_within:
            print(f"Race is {hours_to_start:.1f}h away "
                  f"(> --only-if-within={args.only_if_within}h). Skipping.")
            return
        if args.min_lead_min is not None and mins_to_start < args.min_lead_min:
            print(f"Race is {mins_to_start:.1f}min away "
                  f"(< --min-lead-min={args.min_lead_min}min). "
                  f"Freezing nominal route; use live reroute from here on.")
            return

    # ---- Waypoints ----
    wps, leg_marks = partition_waypoints(doc["waypoints"])
    leg_labels = [
        f"{wps[i][0]:.4f},{wps[i][1]:.4f} → {wps[i+1][0]:.4f},{wps[i+1][1]:.4f}"
        for i in range(len(wps) - 1)
    ]
    # Report any constraint marks found
    total_marks = sum(len(v) for v in leg_marks.values())
    if total_marks:
        print(f"Constraint marks: {total_marks} across {len(leg_marks)} leg(s)")
        for li, marks in sorted(leg_marks.items()):
            for m in marks:
                name = m.get("name", "")
                label = f" — {name}" if name else ""
                print(f"  Leg {li+1}: {m['rounding']} rounding of "
                      f"({m['lat']:.5f}, {m['lon']:.5f}), "
                      f"barrier bearing {m['barrier_bearing_deg']:.0f}°{label}")

    # ---- Routing config ----
    r_cfg = doc.get("routing", {})
    tack_penalty = float(r_cfg.get("tack_penalty_s", 90))
    tack_threshold_deg = float(r_cfg.get("tack_threshold_deg", 50))
    gybe_penalty = float(r_cfg.get("gybe_penalty_s", tack_penalty))
    gybe_threshold_deg = float(r_cfg.get("gybe_threshold_deg", 120))
    duration_h   = int(r_cfg.get("duration_hours", 10))
    k_candidates = int(r_cfg.get("k_candidates", 50))
    max_edge_m = float(r_cfg.get("max_edge_m", 2000.0))
    min_edge_m = float(r_cfg.get("min_edge_m", 50.0))
    los_samples = int(r_cfg.get("los_samples", 15))
    router_type  = str(r_cfg.get("router_type", "sector")).strip().lower()
    if router_type != "sector":
        raise ValueError(
            f"Unsupported routing.router_type={router_type!r}. "
            "Only 'sector' is supported."
        )
    polar_sweep_coarse_step = int(
        r_cfg.get("polar_sweep_coarse_step",
                  SectorRouter.POLAR_SWEEP_COARSE_STEP)
    )
    use_dense_polar = bool(r_cfg.get("use_dense_polar", False))
    use_dot_filter  = bool(r_cfg.get("use_dot_filter", False))
    corridor_pad_factors_raw = r_cfg.get("corridor_pad_factors")
    corridor_pad_factors = None
    if corridor_pad_factors_raw is not None:
        if isinstance(corridor_pad_factors_raw, (int, float)):
            corridor_pad_factors = (float(corridor_pad_factors_raw),)
        elif isinstance(corridor_pad_factors_raw, str):
            parts = [p.strip() for p in corridor_pad_factors_raw.split(",")]
            corridor_pad_factors = tuple(float(p) for p in parts if p)
        elif isinstance(corridor_pad_factors_raw, (list, tuple)):
            corridor_pad_factors = tuple(float(v) for v in corridor_pad_factors_raw)
        else:
            raise ValueError(
                "routing.corridor_pad_factors must be a number, comma-separated "
                "string, or list of numbers."
            )
    corridor_cache_max = r_cfg.get("corridor_cache_max")
    if corridor_cache_max is not None:
        corridor_cache_max = int(corridor_cache_max)

    print("Routing performance settings:")
    print(f"  polar_sweep_coarse_step={polar_sweep_coarse_step}, "
          f"use_dense_polar={use_dense_polar}, use_dot_filter={use_dot_filter}")
    print(f"  graph: k_candidates={k_candidates}, edge=[{min_edge_m:.0f}, "
          f"{max_edge_m:.0f}]m, los_samples={los_samples}")
    if corridor_pad_factors is not None:
        print(f"  corridor_pad_factors={corridor_pad_factors}")

    # ---- Output config ----
    out_cfg = doc.get("output", {})
    save_plots = out_cfg.get("save_plots", True) and not args.no_plots
    save_hourly_frames = bool(out_cfg.get("save_hourly_frames", False)) and save_plots
    # Default to a per-route subdirectory so outputs from many YAMLs don't
    # collide.  An explicit "plot_dir" in the YAML is taken as-is (no
    # auto-subdir) so users can override.
    if "plot_dir" in out_cfg:
        plot_dir = HERE / out_cfg["plot_dir"]
    else:
        plot_dir = HERE / "routes/output" / yaml_path.stem
    # JSON/report/NPZ/GeoJSON exports are written even when matplotlib output is off.
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load current data ----
    # load_current_field automatically picks historical vs. live cycle.
    print("Loading current field…")
    cf, transformer, start_time_s, _ = load_current_field(
        depart_dt=depart_dt,
        duration_hours=duration_h,
        use_cache=not args.no_cache,
    )

    # ---- Boat and wind ----
    boat, wind = build_boat_and_wind(
        doc,
        context={
            "waypoints": wps,
            "depart_dt": depart_dt,
            "depart_utc": depart_utc,
            "start_time_s": start_time_s,
            "duration_h": duration_h,
            "no_cache": args.no_cache,
            "transformer": transformer,
            "tz_str": tz_str,
            "task_slug": yaml_path.stem,
        },
    )

    # ---- Build router ----
    print(f"Using SectorRouter ({SectorRouter.N_SECTORS}-sector heading-binned connectivity)")
    router = SectorRouter(cf, boat, wind=wind,
                          tack_penalty_s=tack_penalty,
                          tack_threshold_deg=tack_threshold_deg,
                          gybe_penalty_s=gybe_penalty,
                          gybe_threshold_deg=gybe_threshold_deg,
                          max_edge_m=max_edge_m,
                          min_edge_m=min_edge_m,
                          k_candidates=k_candidates,
                          los_samples=los_samples,
                          polar_sweep_coarse_step=polar_sweep_coarse_step,
                          corridor_pad_factors=corridor_pad_factors,
                          corridor_cache_max=corridor_cache_max,
                          use_dense_polar=use_dense_polar,
                          use_dot_filter=use_dot_filter)

    # ---- Run each leg ----
    routes = []
    current_time_s = start_time_s
    current_depart_utc = depart_utc

    for i, (start_ll, end_ll) in enumerate(zip(wps[:-1], wps[1:]), 1):
        marks_this_leg = leg_marks.get(i - 1, [])
        barriers = compute_barriers(marks_this_leg, transformer)

        route, xs, ys, wm = run_leg(
            router, start_ll, end_ll,
            start_time_s=current_time_s,
            leg_num=i,
            leg_label=leg_labels[i - 1],
            barriers=barriers if len(barriers) > 0 else None,
        )
        routes.append(route)

        print_leg_summary(i, leg_labels[i - 1], route,
                          current_depart_utc, tz)

        if save_plots and not args.no_plots:
            import matplotlib
            matplotlib.use("Agg")
            slug = yaml_path.stem
            plot_path = plot_dir / f"{slug}_leg{i:02d}.png"
            sl_time, sl_dist = router.straight_line_time(
                start_ll, end_ll, start_time_s=current_time_s)
            plot_route(route, xs, ys, wm, cf,
                       start_ll, end_ll,
                       straight_time_s=sl_time, straight_dist_m=sl_dist,
                       save_path=plot_path, show=False,
                       wind_field=wind, elapsed_s=current_time_s,
                       land_marks=marks_this_leg,
                       barriers=barriers if len(barriers) > 0 else None)

        # Save rich NPZ diagnostics for this leg
        npz_path = plot_dir / f"{yaml_path.stem}_leg{i:02d}.npz"
        route.save_npz(str(npz_path))

        # Chain: departure of next leg = arrival of this leg
        current_time_s += route.total_time_s
        current_depart_utc = (depart_utc +
                               _dt.timedelta(seconds=current_time_s - start_time_s))

    # ---- Trip summary ----
    print_trip_summary(routes, leg_labels, depart_utc, tz)
    print_performance_summary(routes)

    # ---- Save route JSON ----
    route_json_path = save_route_json(
        routes=routes,
        wps=wps,
        depart_utc=depart_utc,
        depart_time_s=start_time_s,
        tz=tz,
        task_name=task_name,
        plot_dir=plot_dir,
        slug=yaml_path.stem,
    )
    save_route_report(
        routes=routes,
        wps=wps,
        depart_utc=depart_utc,
        depart_time_s=start_time_s,
        tz=tz,
        task_name=task_name,
        plot_dir=plot_dir,
        slug=yaml_path.stem,
        doc=doc,
    )

    # ---- GeoJSON + manifest for web publishing ----
    route_geojson_path = None
    manifest_path = None
    if args.geojson or args.publish_s3:
        import race_publish
        inputs_hash = race_publish.compute_inputs_hash()
        route_geojson_path = plot_dir / f"{yaml_path.stem}.geojson"
        race_publish.write_geojson(
            routes=routes, wps=wps,
            depart_utc=depart_utc,
            depart_time_s=start_time_s,
            task_name=task_name,
            race_cfg=race_cfg,
            inputs_hash=inputs_hash,
            out_path=route_geojson_path,
        )
        total_dist_nm = sum(r.total_distance_m for r in routes) / 1852.0
        total_time_hr = sum(r.total_time_s for r in routes) / 3600.0
        manifest_path = plot_dir / f"{yaml_path.stem}_manifest.json"
        race_publish.write_manifest(
            race_cfg=race_cfg,
            depart_utc=depart_utc,
            total_distance_nm=total_dist_nm,
            total_time_hr=total_time_hr,
            inputs_hash=inputs_hash,
            out_path=manifest_path,
        )

    # ---- S3 publish (canonical names: route.json, route.geojson, manifest.json) ----
    if args.publish_s3:
        import race_publish
        bucket = race_cfg.get("s3_bucket")
        prefix = race_cfg.get("s3_prefix")
        if not (bucket and prefix):
            sys.exit("--publish-s3 requires race.s3_bucket and race.s3_prefix in YAML")
        # Upload with canonical filenames regardless of local slug-based names.
        # boto3.upload_file lets us pick the S3 key independently.
        import boto3
        s3 = boto3.client("s3")
        prefix = prefix.strip("/")
        uploads = [
            (route_json_path,    "route.json",    "application/json"),
            (route_geojson_path, "route.geojson", "application/geo+json"),
            (manifest_path,      "manifest.json", "application/json"),
        ]
        print(f"\nUploading to s3://{bucket}/{prefix}/ ...")
        for local_path, remote_name, content_type in uploads:
            if local_path is None or not Path(local_path).exists():
                print(f"  SKIP (missing): {remote_name}")
                continue
            key = f"{prefix}/{remote_name}"
            s3.upload_file(
                str(local_path), bucket, key,
                ExtraArgs={"ContentType": content_type,
                           "CacheControl": "max-age=300"},
            )
            print(f"  -> s3://{bucket}/{key}")
        print("Upload complete.")

    # ---- Position time-series ----
    if save_plots and not args.no_plots:
        plot_clean_route_map(
            routes=routes,
            wps=wps,
            transformer=transformer,
            plot_dir=plot_dir,
            slug=yaml_path.stem,
            leg_marks=leg_marks,
            course_feature_centers=doc.get("rtc_course_feature_centers"),
        )
        plot_position_timeseries(
            routes=routes,
            wps=wps,
            depart_utc=depart_utc,
            depart_time_s=start_time_s,
            tz=tz,
            task_name=task_name,
            plot_dir=plot_dir,
            slug=yaml_path.stem,
        )

    # ---- Hourly position frames ----
    if save_hourly_frames:
        all_land_marks = [m for marks in leg_marks.values() for m in marks]
        generate_hourly_frames(
            routes=routes,
            wps=wps,
            cf=cf,
            depart_utc=depart_utc,
            depart_time_s=start_time_s,
            tz=tz,
            task_name=task_name,
            plot_dir=plot_dir,
            slug=yaml_path.stem,
            wind_field=wind,
            all_land_marks=all_land_marks if all_land_marks else None,
        )

    if save_plots or routes:
        print(f"\nAll output saved to {plot_dir}/")


if __name__ == "__main__":
    main()
