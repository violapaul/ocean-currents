#!/usr/bin/env python
"""
diagnostic_raw_path.py — Grid Router diagnostic
-------------------------------------------------
Shows the raw 8-connected A* path vs the smoothed path on the grid router.
Generates a zoomed-in view of the first part of the route so individual
grid cells and path steps are visible.
"""

import sys
from pathlib import Path
import datetime as _dt
from zoneinfo import ZoneInfo

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

sys.path.insert(0, str(Path(__file__).parent))

from sail_routing import (
    Router, BoatModel, load_current_field, KNOTS_TO_MS, MS_TO_KNOTS,
)
from run_route import build_boat_and_wind, load_task

HERE = Path(__file__).parent
YAML_PATH = HERE / "routes" / "shilshole_alki_return.yaml"


def run_diagnostic():
    doc = load_task(YAML_PATH)

    dep_cfg = doc["departure"]
    tz_str = dep_cfg.get("tz", "America/Los_Angeles")
    tz = ZoneInfo(tz_str)
    naive = _dt.datetime.strptime(dep_cfg["datetime"], "%Y-%m-%d %H:%M")
    depart_dt = naive.replace(tzinfo=tz)
    depart_utc = depart_dt.astimezone(_dt.timezone.utc)

    wps = [(float(lat), float(lon)) for lat, lon in doc["waypoints"]]
    start_ll, end_ll = wps[0], wps[1]

    r_cfg = doc.get("routing", {})
    resolution_m = float(r_cfg.get("grid_resolution_m", 300))
    padding_m = float(r_cfg.get("padding_m", 5000))
    tack_penalty = float(r_cfg.get("tack_penalty_s", 60))
    duration_h = int(r_cfg.get("duration_hours", 10))

    print("Loading current field...")
    cf, transformer, start_time_s, _ = load_current_field(
        depart_dt=depart_dt, duration_hours=duration_h, use_cache=True)

    print("Loading boat and wind...")
    boat, wind = build_boat_and_wind(
        doc,
        context={
            "waypoints": wps, "depart_dt": depart_dt, "depart_utc": depart_utc,
            "start_time_s": start_time_s, "duration_h": duration_h,
            "no_cache": False, "transformer": transformer, "tz_str": tz_str,
            "task_slug": "diagnostic_grid",
        },
    )

    print("\nBuilding grid Router...")
    router = Router(cf, boat, resolution_m=resolution_m, padding_m=padding_m,
                    wind=wind, tack_penalty_s=tack_penalty)

    print("Running A* on 8-connected grid...")
    result = router.find_route(start_ll, end_ll,
                               start_time_s=start_time_s, return_debug=True)
    route, xs, ys, water_mask, debug = result

    raw_rc = debug['raw_path_rc']
    smooth_rc = debug['smoothed_path_rc']

    raw_utm = [(xs[c], ys[r]) for r, c in raw_rc]
    smooth_utm = [(xs[c], ys[r]) for r, c in smooth_rc]

    # Verify subset property
    raw_set = set(raw_rc)
    smooth_set = set(smooth_rc)
    not_in_raw = [rc for rc in smooth_rc if rc not in raw_set]
    print(f"\nRaw path:      {len(raw_rc)} cells")
    print(f"Smoothed path: {len(smooth_rc)} waypoints")
    print(f"Smooth NOT in raw: {len(not_in_raw)}")
    for rc in not_in_raw:
        print(f"  MISSING: ({rc[0]}, {rc[1]}) -> UTM ({xs[rc[1]]:.0f}, {ys[rc[0]]:.0f})")

    # ── Full-route overview ──────────────────────────────────────────────
    def make_plot(xlim, ylim, suffix, title_extra=""):
        fig, ax = plt.subplots(figsize=(14, 14))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#0f1923")

        # Water mask background
        ax.pcolormesh(xs, ys, np.where(water_mask, 0.3, np.nan),
                      cmap="Blues", vmin=0, vmax=1, alpha=0.15,
                      shading="auto", zorder=0)

        # Current vectors
        elapsed_s = start_time_s
        u_grid, v_grid = cf.query_grid(xs, ys, elapsed_s=elapsed_s)
        speed_grid = np.hypot(u_grid, v_grid) * MS_TO_KNOTS
        wm = ~np.isnan(u_grid)
        speed_grid[~wm] = np.nan

        speed_max = np.nanmax(speed_grid) if np.any(~np.isnan(speed_grid)) else 1.0
        ax.pcolormesh(xs, ys, speed_grid, cmap="plasma", alpha=0.4,
                      shading="auto", vmin=0, vmax=max(speed_max, 1.0), zorder=1)

        # Raw 8-connected path (orange line + dots)
        rx = [p[0] for p in raw_utm]
        ry = [p[1] for p in raw_utm]
        ax.plot(rx, ry, '-', color="#ff8800", linewidth=1.0, alpha=0.7,
                zorder=5, label=f"Raw 8-connected ({len(raw_rc)} cells)")
        ax.scatter(rx, ry, s=8, c="#ffaa00", zorder=6, alpha=0.8,
                   edgecolors="none")

        # Smoothed path (green line + squares)
        sx_pts = [p[0] for p in smooth_utm]
        sy_pts = [p[1] for p in smooth_utm]
        ax.plot(sx_pts, sy_pts, '-', color="#00ff88", linewidth=2.5, alpha=0.9,
                zorder=7, label=f"Smoothed ({len(smooth_rc)} waypoints)",
                path_effects=[pe.Stroke(linewidth=4, foreground="#000000",
                                        alpha=0.4), pe.Normal()])
        ax.scatter(sx_pts, sy_pts, s=80, c="#00ff88", marker="s", zorder=8,
                   edgecolors="white", linewidths=1.5)

        # Label smoothed waypoints with their raw-path index
        for i, rc in enumerate(smooth_rc):
            if rc in raw_set:
                raw_idx = raw_rc.index(rc)
            else:
                raw_idx = -1
            x_pt, y_pt = xs[rc[1]], ys[rc[0]]
            ax.annotate(f"{i}(r{raw_idx})", (x_pt, y_pt),
                        textcoords="offset points", xytext=(8, 8),
                        fontsize=7, color="white", fontfamily="monospace",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor="#000000", alpha=0.7))

        # Start / end
        s_x, s_y = transformer.transform(start_ll[1], start_ll[0])
        e_x, e_y = transformer.transform(end_ll[1], end_ll[0])
        ax.plot(s_x, s_y, "o", color="#2ecc71", markersize=14, zorder=10,
                markeredgecolor="white", markeredgewidth=2.5, label="Start")
        ax.plot(e_x, e_y, "s", color="#e74c3c", markersize=14, zorder=10,
                markeredgecolor="white", markeredgewidth=2.5, label="End")

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.set_xlabel("Easting (m, UTM)", color="#aaaaaa", fontsize=9)
        ax.set_ylabel("Northing (m, UTM)", color="#aaaaaa", fontsize=9)
        ax.tick_params(colors="#666666", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#334455")
        ax.grid(True, alpha=0.1, color="#445566")

        title = f"Grid Router: Raw 8-connected vs Smoothed{title_extra}"
        ax.set_title(title, color="white", fontsize=13, pad=10)
        ax.legend(loc="upper left", framealpha=0.85,
                  facecolor="#0d1b2a", edgecolor="#334455",
                  labelcolor="white", fontsize=9)

        plt.tight_layout()
        out = HERE / "routes" / "output" / f"diagnostic_grid_{suffix}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Saved: {out}")

    # Full route view
    x_all = [p[0] for p in raw_utm]
    y_all = [p[1] for p in raw_utm]
    pad = 3000
    make_plot(
        xlim=(min(x_all) - pad, max(x_all) + pad),
        ylim=(min(y_all) - pad, max(y_all) + pad),
        suffix="full",
        title_extra=" — Full Route",
    )

    # Zoomed view: first ~50 raw path cells (near Shilshole)
    n_zoom = min(60, len(raw_utm))
    zx = [raw_utm[i][0] for i in range(n_zoom)]
    zy = [raw_utm[i][1] for i in range(n_zoom)]
    zpad = 800
    make_plot(
        xlim=(min(zx) - zpad, max(zx) + zpad),
        ylim=(min(zy) - zpad, max(zy) + zpad),
        suffix="zoom_start",
        title_extra=" — Zoomed: First 60 cells near Shilshole",
    )


if __name__ == "__main__":
    run_diagnostic()
