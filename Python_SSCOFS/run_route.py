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

    routing:
      router_type: "sector"           # "sector" (default), "mesh", or "grid" (legacy)
      grid_resolution_m: 300          # only used for grid router
      padding_m: 5000                 # only used for grid router
      tack_penalty_s: 60
      duration_hours: 10              # forecast window to load per leg

    output:
      save_plots: true
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
    PolarTable, WindField, BoatModel, Router, MeshRouter, SectorRouter,
    load_current_field, plot_route,
    KNOTS_TO_MS, MS_TO_KNOTS,
    CurrentField,
)

HERE = Path(__file__).parent


# ---------------------------------------------------------------------------
# YAML loading and validation
# ---------------------------------------------------------------------------

def load_task(yaml_path: Path) -> dict:
    """Load and lightly validate a routing task YAML."""
    with open(yaml_path) as f:
        doc = yaml.safe_load(f)

    wps = doc.get("waypoints", [])
    if len(wps) < 2:
        raise ValueError("YAML must contain at least 2 waypoints.")
    for i, wp in enumerate(wps):
        if len(wp) != 2:
            raise ValueError(f"Waypoint {i} must be [lat, lon], got {wp!r}.")

    dep = doc.get("departure", {})
    if "datetime" not in dep:
        raise ValueError("YAML must contain departure.datetime.")

    return doc


# ---------------------------------------------------------------------------
# Build routing objects from YAML
# ---------------------------------------------------------------------------

def _build_openmeteo_route_wind(wind_cfg: dict, ctx: dict) -> WindField:
    """Build a dynamic wind field by fetching Open-Meteo ECMWF nodes."""
    from ecmwf_wind import fetch_route_wind_dataset, DEFAULT_TIMEZONE

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

    time_vals = np.asarray(ds["time"].values, dtype="datetime64[s]")
    if time_vals.size == 0:
        raise RuntimeError("Open-Meteo wind dataset is empty")
    depart_utc_naive = np.datetime64(
        ctx["depart_utc"].astimezone(_dt.timezone.utc).replace(tzinfo=None),
        "s",
    )
    # Dataset times are in local timezone (from Open-Meteo); convert to
    # UTC before computing offsets relative to the UTC departure time.
    utc_offset_s = int(ctx["depart_dt"].utcoffset().total_seconds())
    time_vals_utc = time_vals - np.timedelta64(utc_offset_s, "s")
    frame_times_s = ctx["start_time_s"] + (
        (time_vals_utc - depart_utc_naive) / np.timedelta64(1, "s")
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
    print(f"Wind field ready: {len(nodes)} nodes, {len(frame_times_s)} hourly frames, "
          f"model={ds.attrs.get('model', 'unknown')}")
    return wind


def build_boat_and_wind(doc: dict, context: dict | None = None):
    """Construct BoatModel and optional WindField from YAML config."""
    boat_cfg = doc.get("boat", {})
    speed_kt = float(boat_cfg.get("speed_kt", 6.0))

    polar = None
    polar_path_str = boat_cfg.get("polar")
    minimum_twa = float(boat_cfg.get("minimum_twa", 0.0))
    if polar_path_str:
        polar_path = Path(polar_path_str)
        if not polar_path.is_absolute():
            polar_path = HERE / polar_path
        if polar_path.exists():
            polar = PolarTable(polar_path, minimum_twa=minimum_twa)
            nogo = f", no-go zone < {minimum_twa:.0f}°" if minimum_twa > 0 else ""
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

def run_leg(router: Router, start_ll, end_ll, start_time_s: float,
            leg_num: int, leg_label: str) -> tuple:
    """Run A* for one leg. Returns (route, xs, ys, water_mask)."""
    print(f"\n{'─'*60}")
    print(f"Leg {leg_num}: {leg_label}")
    print(f"  From: {start_ll[0]:.6f}, {start_ll[1]:.6f}")
    print(f"  To:   {end_ll[0]:.6f}, {end_ll[1]:.6f}")
    print(f"  Start time offset: {start_time_s:.0f}s ({start_time_s/3600:.2f}h)")
    route, xs, ys, wm = router.find_route(start_ll, end_ll,
                                           start_time_s=start_time_s)
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
                           tz, task_name, plot_dir, slug, wind_field=None):
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
    res = 400.0  # coarser grid for speed — just for background viz
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
                color="#88aacc", linewidth=1.5, linestyle="--",
                alpha=0.5, zorder=4, label="Planned route" if show_legend else None)

        if len(track_x_past) > 1:
            ax.plot(track_x_past, track_y_past,
                    color="#00e5ff", linewidth=2.5, zorder=5,
                    label="Track so far" if show_legend else None,
                    path_effects=[pe.Stroke(linewidth=4.5,
                                            foreground="#003344",
                                            alpha=0.6),
                                  pe.Normal()])

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

    def _setup_ax(ax, title):
        """Common axis setup."""
        ax.set_facecolor("#0f1923")
        ax.set_title(title, color="white", fontsize=11, pad=8)
        ax.set_xlabel("Easting (m)", color="#aaaaaa", fontsize=8)
        ax.set_ylabel("Northing (m)", color="#aaaaaa", fontsize=8)
        ax.tick_params(colors="#666666", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#334455")
        ax.set_aspect("equal")
        ax.set_xlim(xs[0], xs[-1])
        ax.set_ylim(ys_g[0], ys_g[-1])
        ax.grid(True, alpha=0.10, color="#445566")

    for h in range(n_hours):
        elapsed_s = depart_time_s + h * 3600.0
        elapsed_s = float(np.clip(elapsed_s, tt[0], tt[-1]))
        frame_utc  = depart_utc + _dt.timedelta(seconds=h * 3600)
        frame_local = frame_utc.astimezone(tz)

        # Current field at this hour
        u_grid, v_grid = cf.query_grid(xs, ys_g, elapsed_s=elapsed_s)
        speed_grid = np.hypot(u_grid, v_grid) * MS_TO_KNOTS
        water_mask = ~np.isnan(u_grid)
        speed_grid[~water_mask] = np.nan

        # Boat position at this hour
        bx, by = position_at_time(tx, ty, tt, elapsed_s)

        # Track so far (up to this moment)
        mask_past = tt <= elapsed_s + 1.0
        track_x_past = tx[mask_past]
        track_y_past = ty[mask_past]

        # Arrow subsampling
        step = max(1, len(xs) // 18)
        xx, yy = np.meshgrid(xs, ys_g)
        arrow_scale = 1.0 / (step * res) * 0.65

        if wind_field is None:
            fig, ax = plt.subplots(figsize=(10, 14))
            fig.patch.set_facecolor("#1a1a2e")

            _setup_ax(ax, f"{task_name}  —  Hour {h}")

            ax.pcolormesh(xs, ys_g, np.where(water_mask, 0.0, np.nan),
                          cmap="Blues", vmin=0, vmax=1, alpha=0.08,
                          shading="auto", zorder=0)

            im = ax.pcolormesh(xs, ys_g, speed_grid,
                               cmap="plasma", alpha=0.5, shading="auto",
                               vmin=0, vmax=max(speed_max, 1.0), zorder=1)
            cbar = fig.colorbar(im, ax=ax, label="Current (kt)",
                                pad=0.01, shrink=0.6, fraction=0.03)
            cbar.ax.yaxis.label.set_color("white")
            cbar.ax.tick_params(colors="white")

            u_sub = u_grid[::step, ::step]
            v_sub = v_grid[::step, ::step]
            spd_sub = speed_grid[::step, ::step]
            mag = np.hypot(u_sub, v_sub)
            mag_safe = np.where(mag < 1e-8, 1.0, mag)
            u_norm = u_sub / mag_safe
            v_norm = v_sub / mag_safe
            visible = (mag > 0.04) & ~np.isnan(spd_sub)
            u_norm[~visible] = np.nan
            v_norm[~visible] = np.nan
            ax.quiver(xx[::step, ::step], yy[::step, ::step],
                      u_norm, v_norm, spd_sub,
                      cmap="cool", clim=(0, max(speed_max, 1.0)),
                      scale=arrow_scale, scale_units="xy",
                      width=0.003, headwidth=4, headlength=5,
                      alpha=0.7, zorder=3)

            _draw_route_overlay(ax, frame_local, track_x_past, track_y_past, bx, by)
            ax.legend(loc="upper right", framealpha=0.7,
                      facecolor="#0d1b2a", edgecolor="#334455",
                      labelcolor="white", fontsize=8)
        else:
            fig, (ax_cur, ax_wind) = plt.subplots(1, 2, figsize=(16, 10))
            fig.patch.set_facecolor("#1a1a2e")

            _setup_ax(ax_cur, "Ocean Current")
            _setup_ax(ax_wind, "Wind")

            for ax in (ax_cur, ax_wind):
                ax.pcolormesh(xs, ys_g, np.where(water_mask, 0.0, np.nan),
                              cmap="Blues", vmin=0, vmax=1, alpha=0.08,
                              shading="auto", zorder=0)

            im_cur = ax_cur.pcolormesh(xs, ys_g, speed_grid,
                                       cmap="plasma", alpha=0.5, shading="auto",
                                       vmin=0, vmax=max(speed_max, 1.0), zorder=1)
            cbar_cur = fig.colorbar(im_cur, ax=ax_cur, label="Current (kt)",
                                    pad=0.02, shrink=0.7)
            cbar_cur.ax.yaxis.label.set_color("white")
            cbar_cur.ax.tick_params(colors="white")

            u_sub = u_grid[::step, ::step]
            v_sub = v_grid[::step, ::step]
            spd_sub = speed_grid[::step, ::step]
            mag = np.hypot(u_sub, v_sub)
            mag_safe = np.where(mag < 1e-8, 1.0, mag)
            u_norm = u_sub / mag_safe
            v_norm = v_sub / mag_safe
            visible = (mag > 0.04) & ~np.isnan(spd_sub)
            u_norm[~visible] = np.nan
            v_norm[~visible] = np.nan
            ax_cur.quiver(xx[::step, ::step], yy[::step, ::step],
                          u_norm, v_norm, spd_sub,
                          cmap="cool", clim=(0, max(speed_max, 1.0)),
                          scale=arrow_scale, scale_units="xy",
                          width=0.003, headwidth=4, headlength=5,
                          alpha=0.7, zorder=3)

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
            cbar_wind.ax.yaxis.label.set_color("white")
            cbar_wind.ax.tick_params(colors="white")

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
                           wu_norm, wv_norm, wspd_sub,
                           cmap="Reds", clim=(0, wind_max),
                           scale=arrow_scale, scale_units="xy",
                           width=0.003, headwidth=4, headlength=5,
                           alpha=0.7, zorder=3)

            _draw_route_overlay(ax_cur, frame_local, track_x_past, track_y_past, bx, by, show_legend=True)
            _draw_route_overlay(ax_wind, frame_local, track_x_past, track_y_past, bx, by, show_legend=False)
            ax_cur.legend(loc="upper right", framealpha=0.7,
                          facecolor="#0d1b2a", edgecolor="#334455",
                          labelcolor="white", fontsize=7)

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
                        color="white", fontfamily="monospace",
                        bbox=dict(boxstyle="round,pad=0.4",
                                  facecolor="#0d1b2a", alpha=0.85,
                                  edgecolor="#334455"))

        if len(track_x_past) > 1:
            seg_dists = np.hypot(np.diff(track_x_past), np.diff(track_y_past))
            cum_dist_nm = seg_dists.sum() / 1852.0
            avg_sog = cum_dist_nm / elapsed_h if elapsed_h > 0 else 0.0
            stats_text = (f"Dist: {cum_dist_nm:.1f} nm\n"
                          f"SOG:  {avg_sog:.2f} kt")
            ax_for_ann.text(0.02, 0.02, stats_text,
                            transform=ax_for_ann.transAxes,
                            fontsize=8, va="bottom", ha="left",
                            color="white", fontfamily="monospace",
                            bbox=dict(boxstyle="round,pad=0.3",
                                      facecolor="#0d1b2a", alpha=0.85,
                                      edgecolor="#334455"))

        if wind_field is not None:
            fig.suptitle(f"{task_name}  —  Hour {h}", color="white", fontsize=12, y=0.98)

        plt.tight_layout()
        out_path = plot_dir / f"{slug}_hour{h:02d}.png"
        fig.savefig(out_path, dpi=130, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
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
        legs_out.append({
            "leg": li + 1,
            "from": list(wps[li]),
            "to":   list(wps[li + 1]),
            "distance_nm": round(r.total_distance_m / 1852.0, 3),
            "time_min":    round(r.total_time_s / 60.0, 1),
            "avg_sog_kt":  round(r.avg_sog_knots, 3),
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

    # Absolute wall-clock times (local)
    t_local = [depart_utc + _dt.timedelta(seconds=float(t - depart_time_s))
               for t in tt]
    t_local_tz = [t.astimezone(tz) for t in t_local]
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
    parser.add_argument("--resolution", type=float, default=None,
                        help="Override grid resolution in metres")
    args = parser.parse_args()

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

    # ---- Departure time ----
    dep_cfg = doc["departure"]
    tz_str = dep_cfg.get("tz", "America/Los_Angeles")
    tz = ZoneInfo(tz_str)
    naive = _dt.datetime.strptime(dep_cfg["datetime"], "%Y-%m-%d %H:%M")
    depart_dt = naive.replace(tzinfo=tz)
    depart_utc = depart_dt.astimezone(_dt.timezone.utc)

    # ---- Waypoints ----
    wps = [(float(lat), float(lon)) for lat, lon in doc["waypoints"]]
    leg_labels = [
        f"{wps[i][0]:.4f},{wps[i][1]:.4f} → {wps[i+1][0]:.4f},{wps[i+1][1]:.4f}"
        for i in range(len(wps) - 1)
    ]

    # ---- Routing config ----
    r_cfg = doc.get("routing", {})
    resolution_m = args.resolution or float(r_cfg.get("grid_resolution_m", 300))
    padding_m    = float(r_cfg.get("padding_m", 5000))
    tack_penalty = float(r_cfg.get("tack_penalty_s", 60))
    duration_h   = int(r_cfg.get("duration_hours", 10))
    router_type  = r_cfg.get("router_type", "sector")  # "sector", "mesh", or "grid"

    # ---- Output config ----
    out_cfg = doc.get("output", {})
    save_plots = out_cfg.get("save_plots", True) and not args.no_plots
    plot_dir = HERE / out_cfg.get("plot_dir", "routes/output")
    if save_plots:
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
    if router_type == "sector":
        print("Using SectorRouter (16-sector heading-binned connectivity)")
        router = SectorRouter(cf, boat, wind=wind,
                              tack_penalty_s=tack_penalty)
    elif router_type == "mesh":
        print("Using MeshRouter (A* on SSCOFS Delaunay mesh)")
        router = MeshRouter(cf, boat, wind=wind,
                            tack_penalty_s=tack_penalty)
    else:
        print("Using Router (8-connected grid, legacy)")
        router = Router(cf, boat,
                        resolution_m=resolution_m,
                        padding_m=padding_m,
                        wind=wind,
                        tack_penalty_s=tack_penalty)

    # ---- Run each leg ----
    routes = []
    current_time_s = start_time_s
    current_depart_utc = depart_utc

    for i, (start_ll, end_ll) in enumerate(zip(wps[:-1], wps[1:]), 1):
        route, xs, ys, wm = run_leg(
            router, start_ll, end_ll,
            start_time_s=current_time_s,
            leg_num=i,
            leg_label=leg_labels[i - 1],
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
                       wind_field=wind, elapsed_s=current_time_s)

        # Chain: departure of next leg = arrival of this leg
        current_time_s += route.total_time_s
        current_depart_utc = (depart_utc +
                               _dt.timedelta(seconds=current_time_s - start_time_s))

    # ---- Trip summary ----
    print_trip_summary(routes, leg_labels, depart_utc, tz)

    # ---- Save route JSON ----
    save_route_json(
        routes=routes,
        wps=wps,
        depart_utc=depart_utc,
        depart_time_s=start_time_s,
        tz=tz,
        task_name=task_name,
        plot_dir=plot_dir,
        slug=yaml_path.stem,
    )

    # ---- Position time-series ----
    if save_plots and not args.no_plots:
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
    if save_plots and not args.no_plots:
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
        )

    if save_plots:
        print(f"\nAll output saved to {plot_dir}/")


if __name__ == "__main__":
    main()
