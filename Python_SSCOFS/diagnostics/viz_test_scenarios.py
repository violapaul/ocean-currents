"""
viz_test_scenarios.py
---------------------
Generate diagnostic plots for the key synthetic routing test scenarios.

Run:
    conda run -n currents python viz_test_scenarios.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from sail_routing import BoatModel, Router, MS_TO_KNOTS
from test_sail_routing import (
    make_synthetic_field, make_time_varying_field,
    _make_transformer, _latlon_for_offset,
    CENTER_LAT, CENTER_LON,
)

OUT_DIR = Path(__file__).parent


def plot_scenario(route, xs, ys, water_mask, cf, start_xy, end_xy,
                  title, filename, straight_xy=None):
    """Plot a synthetic routing scenario."""
    u_grid, v_grid = cf.query_grid(xs, ys, elapsed_s=0.0)
    speed_grid = np.sqrt(u_grid**2 + v_grid**2) * MS_TO_KNOTS
    speed_grid[~water_mask] = np.nan

    fig, ax = plt.subplots(figsize=(10, 8))

    xx, yy = np.meshgrid(xs, ys)
    im = ax.pcolormesh(xs, ys, speed_grid, cmap='Blues', alpha=0.6,
                       shading='auto', vmin=0)
    cbar = fig.colorbar(im, ax=ax, label='Current speed (knots)',
                        pad=0.02, shrink=0.8)

    # Land cells
    land = ~water_mask
    if land.any():
        ax.pcolormesh(xs, ys, np.where(land, 1.0, np.nan),
                      cmap='YlOrBr', alpha=0.4, shading='auto')

    # Current arrows (subsample)
    step = max(1, len(xs) // 20)
    valid = water_mask[::step, ::step]
    ax.quiver(
        xx[::step, ::step][valid], yy[::step, ::step][valid],
        u_grid[::step, ::step][valid], v_grid[::step, ::step][valid],
        color='steelblue', scale=8.0, scale_units='xy', alpha=0.7,
        width=0.003, headwidth=3, headlength=4,
    )

    # Route
    rx = [p[0] for p in route.waypoints_utm]
    ry = [p[1] for p in route.waypoints_utm]
    ax.plot(rx, ry, color='#e63946', linewidth=2.5, zorder=5,
            label='Optimal route',
            path_effects=[pe.Stroke(linewidth=4, foreground='white'),
                          pe.Normal()])

    # Straight line
    if straight_xy:
        ax.plot([straight_xy[0][0], straight_xy[1][0]],
                [straight_xy[0][1], straight_xy[1][1]],
                '--', color='#457b9d', linewidth=1.5, zorder=4,
                label='Straight line')

    # Markers
    ax.plot(start_xy[0], start_xy[1], 'o', color='#2a9d8f',
            markersize=12, zorder=6, markeredgecolor='white',
            markeredgewidth=2, label='Start')
    ax.plot(end_xy[0], end_xy[1], 's', color='#e76f51',
            markersize=12, zorder=6, markeredgecolor='white',
            markeredgewidth=2, label='End')

    # Stats
    stats = (
        f"Distance: {route.total_distance_m:.0f} m\n"
        f"Time: {route.total_time_s:.0f} s ({route.total_time_s/60:.1f} min)\n"
        f"Avg SOG: {route.avg_sog_knots:.2f} kt\n"
        f"Boat STW: {route.boat_speed_knots:.1f} kt"
    )
    ax.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    margin = 300
    ax.set_xlim(xs[0] - margin, xs[-1] + margin)
    ax.set_ylim(ys[0] - margin, ys[-1] + margin)
    plt.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {filename}")


def run_scenario(title, filename, cf, tr, boat, start_offset, end_offset,
                 resolution=200.0, padding=1000.0):
    """Helper: run routing and plot."""
    router = Router(cf, boat, resolution_m=resolution, padding_m=padding)
    start_ll = _latlon_for_offset(*start_offset, tr)
    end_ll = _latlon_for_offset(*end_offset, tr)

    route, xs, ys, wm = router.find_route(start_ll, end_ll)

    sx, sy = tr.transform(start_ll[1], start_ll[0])
    ex, ey = tr.transform(end_ll[1], end_ll[0])

    plot_scenario(route, xs, ys, wm, cf,
                  start_xy=(sx, sy), end_xy=(ex, ey),
                  straight_xy=((sx, sy), (ex, ey)),
                  title=title, filename=filename)
    return route


def main():
    boat = BoatModel(base_speed_knots=6.0)
    tr = _make_transformer()
    cx, cy = tr.transform(CENTER_LON, CENTER_LAT)

    # --- 1. No current ---
    print("1. No current (straight path)")
    cf, tr = make_synthetic_field()
    run_scenario("No Current -- Route Should Be Straight",
                 "test_viz_01_no_current.png",
                 cf, tr, boat, (0, -2500), (0, 2500))

    # --- 2. Favorable current ---
    print("2. Uniform favorable current (1 m/s northward)")
    cf, tr = make_synthetic_field(v_func=lambda x, y: 1.0)
    run_scenario("Uniform 1 m/s Northward Current -- Boosted SOG",
                 "test_viz_02_favorable.png",
                 cf, tr, boat, (0, -2500), (0, 2500))

    # --- 3. Opposing current ---
    print("3. Uniform opposing current (1 m/s southward)")
    cf, tr = make_synthetic_field(v_func=lambda x, y: -1.0)
    run_scenario("Uniform 1 m/s Opposing Current -- Reduced SOG",
                 "test_viz_03_opposing.png",
                 cf, tr, boat, (0, -2500), (0, 2500))

    # --- 4. Cross-current ---
    print("4. Uniform cross-current (1 m/s eastward)")
    cf, tr = make_synthetic_field(u_func=lambda x, y: 1.0)
    run_scenario("Uniform 1 m/s Cross-Current -- Crabbing Required",
                 "test_viz_04_cross_current.png",
                 cf, tr, boat, (0, -2500), (0, 2500))

    # --- 5. Detour into favorable band ---
    print("5. Favorable current band (route should detour east)")
    band_left = cx + 1000
    band_right = cx + 2000

    def v_band(x, y):
        return 2.0 if band_left <= x <= band_right else 0.0

    cf, tr = make_synthetic_field(v_func=v_band)
    run_scenario("2 m/s Current Band at +1.5 km East -- Route Should Detour",
                 "test_viz_05_detour_band.png",
                 cf, tr, boat, (0, -4000), (0, 4000),
                 padding=2000)

    # --- 6. Land wall avoidance ---
    print("6. Land wall avoidance")

    def land_wall(x, y):
        return (abs(x - cx) < 750) and (abs(y - cy) < 1000)

    cf, tr = make_synthetic_field(land_func=land_wall,
                                  land_threshold_m=250.0)
    run_scenario("Land Wall (1500 m wide) -- Route Goes Around",
                 "test_viz_06_land_wall.png",
                 cf, tr, boat, (-3000, 0), (3000, 0),
                 padding=2000)

    # =================================================================
    # Time-dependent scenarios: side-by-side static vs time-varying
    # =================================================================

    # --- 7. Disappearing band: static vs time-varying ---
    print("\n7. Time-dependent: disappearing current band")
    print("   7a. Static (band always present)")
    cf_static, tr = make_synthetic_field(v_func=v_band)
    route_static = run_scenario(
        "STATIC: Band Always Present -- Detours East",
        "test_viz_07a_band_static.png",
        cf_static, tr, boat, (0, -4000), (0, 4000), padding=2000)

    print("   7b. Time-varying (band disappears at t=800s)")
    u_zero = lambda x, y: 0.0
    v_zero = lambda x, y: 0.0
    cf_tv, tr = make_time_varying_field(
        frame_specs=[(u_zero, v_band), (u_zero, v_zero)],
        frame_times_s=[0.0, 800.0])
    route_tv = run_scenario(
        "TIME-VARYING: Band Disappears at 800s -- Stays Straighter",
        "test_viz_07b_band_disappears.png",
        cf_tv, tr, boat, (0, -4000), (0, 4000), padding=2000)

    print(f"   Static route:  {route_static.total_time_s:.0f}s, "
          f"deviation {max(p[0] for p in route_static.waypoints_utm) - route_static.waypoints_utm[0][0]:.0f} m east")
    print(f"   TV route:      {route_tv.total_time_s:.0f}s, "
          f"deviation {max(p[0] for p in route_tv.waypoints_utm) - route_tv.waypoints_utm[0][0]:.0f} m east")

    # --- 8. Current reversal ---
    print("\n8. Time-dependent: current reversal (west favorable -> opposing)")
    west_left = cx - 2000
    west_right = cx - 1000

    def v_west_fav(x, y):
        return 2.0 if west_left <= x <= west_right else 0.0
    def v_west_opp(x, y):
        return -2.0 if west_left <= x <= west_right else 0.0

    print("   8a. Static (west always favorable)")
    cf_fav, tr = make_synthetic_field(v_func=v_west_fav)
    route_fav = run_scenario(
        "STATIC: West Band Always Favorable -- Detours West",
        "test_viz_08a_west_static.png",
        cf_fav, tr, boat, (0, -4000), (0, 4000), padding=2500)

    print("   8b. Time-varying (west reverses at t=1000s)")
    cf_rev, tr = make_time_varying_field(
        frame_specs=[(u_zero, v_west_fav), (u_zero, v_west_opp)],
        frame_times_s=[0.0, 1000.0])
    route_rev = run_scenario(
        "TIME-VARYING: West Reverses at 1000s -- Avoids West",
        "test_viz_08b_west_reverses.png",
        cf_rev, tr, boat, (0, -4000), (0, 4000), padding=2500)

    west_dev_static = route_fav.waypoints_utm[0][0] - min(p[0] for p in route_fav.waypoints_utm)
    west_dev_tv = route_rev.waypoints_utm[0][0] - min(p[0] for p in route_rev.waypoints_utm)
    print(f"   Static route:  {route_fav.total_time_s:.0f}s, "
          f"deviation {west_dev_static:.0f} m west")
    print(f"   TV route:      {route_rev.total_time_s:.0f}s, "
          f"deviation {west_dev_tv:.0f} m west")

    print("\nAll scenario plots saved.")


if __name__ == "__main__":
    main()
