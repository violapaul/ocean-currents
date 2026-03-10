#!/usr/bin/env python
"""
diagnostic_sector_comparison.py
-------------------------------
Plot SectorRouter raw path vs smoothed waypoints on a short test leg.

Uses the shilshole_alki_return.yaml configuration (polar + real wind) but
overrides the endpoint to a shorter ~4nm test target so the whole run takes
~3 minutes instead of 15.

Run:
    conda run -n anaconda python diagnostic_sector_comparison.py
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from sail_routing import (
    SectorRouter, BoatModel,
    load_current_field, MS_TO_KNOTS,
)
from run_route import build_boat_and_wind, load_task
from shoreline_utils import draw_shoreline

HERE = Path(__file__).parent.parent
YAML_PATH = HERE / "routes" / "shilshole_alki_return.yaml"

# Shorter test endpoint (~4 nm south of Shilshole, confirmed in-water)
TEST_END = (47.61035199875063, -122.40093710207273)


def run_diagnostic():
    doc = load_task(YAML_PATH)

    dep_cfg = doc["departure"]
    tz_str  = dep_cfg.get("tz", "America/Los_Angeles")
    tz      = ZoneInfo(tz_str)
    naive   = _dt.datetime.strptime(dep_cfg["datetime"], "%Y-%m-%d %H:%M")
    depart_dt  = naive.replace(tzinfo=tz)
    depart_utc = depart_dt.astimezone(_dt.timezone.utc)

    wps = [(float(lat), float(lon)) for lat, lon in doc["waypoints"]]
    start_ll = wps[0]
    end_ll   = TEST_END      # ← shorter route for fast iteration

    r_cfg        = doc.get("routing", {})
    tack_penalty = float(r_cfg.get("tack_penalty_s", 60))
    duration_h   = 4           # 4-hour window is plenty for 4 nm

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
            "task_slug": "diagnostic_sector",
        },
    )

    print(f"\n{'='*60}")
    print("Running SectorRouter  (start → 4 nm test point)...")
    print(f"{'='*60}")
    router = SectorRouter(cf, boat, wind=wind, tack_penalty_s=tack_penalty)
    route, _, _, _, debug = router.find_route(
        start_ll, end_ll, start_time_s=start_time_s, return_debug=True)

    raw_path    = debug['raw_path']
    smooth_path = debug['smoothed_path']
    node_x      = router.node_x
    node_y      = router.node_y

    raw_utm    = [(node_x[n], node_y[n]) for n in raw_path]
    smooth_utm = [(node_x[n], node_y[n]) for n in smooth_path]

    print(f"\nRaw path:  {len(raw_path)} nodes")
    print(f"Smoothed:  {len(smooth_path)} waypoints  "
          f"({100*(1-len(smooth_path)/len(raw_path)):.0f}% reduction)")

    # Which smoothed waypoints are tacks (large turn at that raw node)?
    rpx = np.array([node_x[n] for n in raw_path])
    rpy = np.array([node_y[n] for n in raw_path])

    def _heading(i, j):
        return np.degrees(np.arctan2(rpx[j]-rpx[i], rpy[j]-rpy[i])) % 360.0
    def _hdiff(h1, h2):
        d = abs(h2 - h1); return d if d <= 180 else 360 - d

    raw_list = list(raw_path)
    tack_nodes = set()
    for i in range(1, len(raw_path) - 1):
        h_in  = _heading(i-1, i)
        h_out = _heading(i, i+1)
        if _hdiff(h_in, h_out) >= SectorRouter.SMOOTH_TACK_THRESHOLD_DEG:
            tack_nodes.add(raw_path[i])

    sx, sy = transformer.transform(start_ll[1], start_ll[0])
    ex, ey = transformer.transform(end_ll[1],   end_ll[0])

    def make_plot(xlim, ylim, suffix, title_extra="", annotate_raw=True):
        fig, ax = plt.subplots(figsize=(13, 13))
        ax.set_facecolor("#f5f8fc")

        # Raw path (orange)
        rx = [p[0] for p in raw_utm]
        ry = [p[1] for p in raw_utm]
        ax.plot(rx, ry, '-', color="#ff8800", linewidth=0.9, alpha=0.55,
                zorder=5, label=f"Raw A* path ({len(raw_path)} nodes)")
        ax.scatter(rx, ry, s=14, c="#ffaa00", zorder=6, alpha=0.85,
                   edgecolors="none")

        # Annotate raw node indices inside the view
        if annotate_raw:
            for idx, (px_pt, py_pt) in enumerate(raw_utm):
                if xlim[0] <= px_pt <= xlim[1] and ylim[0] <= py_pt <= ylim[1]:
                    ax.annotate(str(idx), (px_pt, py_pt),
                                textcoords="offset points", xytext=(-5, -9),
                                fontsize=5, color="#885500",
                                fontfamily="monospace", alpha=0.8)

        # Smoothed path (green)
        smx = [p[0] for p in smooth_utm]
        smy = [p[1] for p in smooth_utm]
        ax.plot(smx, smy, '-', color="#007744", linewidth=2.2, alpha=0.9,
                zorder=7, label=f"Smoothed ({len(smooth_path)} waypoints)",
                path_effects=[pe.Stroke(linewidth=3.5, foreground="white",
                                        alpha=0.6), pe.Normal()])

        # Mark smoothed waypoints — colour by type: tack (red) vs. curve (cyan)
        for wi, node in enumerate(smooth_path):
            px_pt, py_pt = node_x[node], node_y[node]
            ri = raw_list.index(node) if node in raw_list else -1
            is_tack = node in tack_nodes
            colour  = "#cc2244" if is_tack else "#0077cc"
            marker  = "^" if is_tack else "s"
            ax.scatter([px_pt], [py_pt], s=90, c=colour, marker=marker,
                       zorder=9, edgecolors="white", linewidths=1.3)
            lbl = f"{wi}(r{ri})" + (" T" if is_tack else "")
            ax.annotate(lbl, (px_pt, py_pt),
                        textcoords="offset points", xytext=(8, 7),
                        fontsize=7, color="#222222", fontfamily="monospace",
                        bbox=dict(boxstyle="round,pad=0.18",
                                  facecolor="white", alpha=0.8,
                                  edgecolor="#cccccc"))

        # Legend proxy for tack vs curve
        ax.scatter([], [], s=90, c="#cc2244", marker="^",
                   edgecolors="white", linewidths=1.3, label="Tack waypoint")
        ax.scatter([], [], s=90, c="#0077cc", marker="s",
                   edgecolors="white", linewidths=1.3, label="Curve waypoint (DP)")

        # Start / end
        ax.plot(sx, sy, "o", color="#2ecc71", markersize=14, zorder=10,
                markeredgecolor="white", markeredgewidth=2.5, label="Start")
        ax.plot(ex, ey, "s", color="#e74c3c", markersize=14, zorder=10,
                markeredgecolor="white", markeredgewidth=2.5, label="End")

        draw_shoreline(ax, transformer, zorder=4)

        ax.set_xlim(xlim);  ax.set_ylim(ylim);  ax.set_aspect("equal")
        ax.set_xlabel("Easting (m, UTM)", fontsize=9)
        ax.set_ylabel("Northing (m, UTM)", fontsize=9)
        ax.tick_params(labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")
        ax.grid(True, alpha=0.25, color="#aaaaaa")

        ax.set_title(f"SectorRouter — two-pass smoothing{title_extra}",
                     fontsize=13, pad=10)
        ax.legend(loc="upper left", framealpha=0.9,
                  facecolor="white", edgecolor="#cccccc", fontsize=8.5, ncol=2)

        info = (
            f"Raw: {len(raw_path)} nodes\n"
            f"Smoothed: {len(smooth_path)} wpts\n"
            f"Time: {route.total_time_s/60:.1f} min\n"
            f"Dist: {route.total_distance_m/1852:.2f} nm\n"
            f"SOG: {route.avg_sog_knots:.2f} kt"
        )
        ax.text(0.98, 0.02, info, transform=ax.transAxes,
                fontsize=8, va="bottom", ha="right", color="#222222",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          alpha=0.9, edgecolor="#cccccc"))

        plt.tight_layout()
        out = HERE / "routes" / "output" / f"diagnostic_sector_{suffix}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")

    # Full route view
    x_all = [p[0] for p in raw_utm]
    y_all = [p[1] for p in raw_utm]
    pad = 2500
    make_plot(
        xlim=(min(x_all) - pad, max(x_all) + pad),
        ylim=(min(y_all) - pad, max(y_all) + pad),
        suffix="full",
        title_extra=" — Full 4 nm leg",
        annotate_raw=False,
    )

    # Zoomed: first 40 raw nodes near Shilshole
    n_zoom = min(40, len(raw_utm))
    zx = [raw_utm[i][0] for i in range(n_zoom)]
    zy = [raw_utm[i][1] for i in range(n_zoom)]
    zpad = 600
    make_plot(
        xlim=(min(zx) - zpad, max(zx) + zpad),
        ylim=(min(zy) - zpad, max(zy) + zpad),
        suffix="zoom_start",
        title_extra=" — Zoomed: tacking near Shilshole",
        annotate_raw=True,
    )

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  Raw path:     {len(raw_path)} nodes")
    print(f"  Smoothed:     {len(smooth_path)} waypoints")
    print(f"  Tack wpts:    {sum(1 for n in smooth_path if n in tack_nodes)}")
    print(f"  Curve wpts:   {sum(1 for n in smooth_path if n not in tack_nodes)}")
    print(f"  Time:         {route.total_time_s/60:.1f} min")
    print(f"  Distance:     {route.total_distance_m/1852:.2f} nm")
    print(f"  Avg SOG:      {route.avg_sog_knots:.2f} kt")


if __name__ == "__main__":
    run_diagnostic()
