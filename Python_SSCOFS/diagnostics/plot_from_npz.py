"""
plot_from_npz.py
----------------
Render route diagnostics from NPZ files produced by SectorRouter.

Usage:
    python diagnostics/plot_from_npz.py routes/output/shilshole_alki_return_leg01.npz
    python diagnostics/plot_from_npz.py routes/output/*.npz          # all legs
    python diagnostics/plot_from_npz.py routes/output/leg01.npz --no-explored
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

OUTLINE = [pe.withStroke(linewidth=3, foreground='white')]
MS_TO_KNOTS = 1.0 / 0.514444


def load(path):
    return np.load(path, allow_pickle=True)


def plot_route_overview(d, ax, show_explored=True):
    """Main route plot: raw path, smoothed waypoints, explored cloud."""
    # Explored nodes as grey scatter
    if show_explored and 'explored_x' in d:
        ax.scatter(d['explored_x'], d['explored_y'],
                   s=0.3, c='#cccccc', alpha=0.4, zorder=1,
                   rasterized=True)

    # Raw A* path — thin blue with node dots
    if 'raw_x' in d:
        ax.plot(d['raw_x'], d['raw_y'],
                '-', color='royalblue', lw=1.0, alpha=0.8, zorder=2,
                label=f'Raw A* path ({len(d["raw_x"])} nodes)')
        ax.plot(d['raw_x'], d['raw_y'],
                '.', color='royalblue', ms=2.5, alpha=0.6, zorder=2)

    # Final smoothed route — bold orange with waypoint diamonds
    if 'wp_x' in d:
        ax.plot(d['wp_x'], d['wp_y'],
                '-', color='darkorange', lw=2.5, zorder=4,
                path_effects=OUTLINE,
                label=f'Smoothed route ({len(d["wp_x"])} wpts)')
        ax.plot(d['wp_x'], d['wp_y'],
                'D', color='darkorange', ms=6, zorder=5,
                markeredgecolor='black', markeredgewidth=0.6)

    # Start / end
    ax.plot(d['wp_x'][0], d['wp_y'][0], 'o', color='limegreen',
            ms=12, zorder=6, markeredgecolor='black', markeredgewidth=1.2,
            path_effects=OUTLINE, label='Start')
    ax.plot(d['wp_x'][-1], d['wp_y'][-1], 's', color='tomato',
            ms=12, zorder=6, markeredgecolor='black', markeredgewidth=1.2,
            path_effects=OUTLINE, label='End')


def plot_heading_profile(d, ax):
    """Heading along the raw path, with tack turns highlighted."""
    if 'raw_heading_deg' not in d:
        return
    hdg = d['raw_heading_deg']
    dist_cum = np.concatenate([[0], np.cumsum(d['raw_seg_dist_m'])])

    ax.plot(dist_cum[:-1], hdg, '.-', color='steelblue', ms=3, lw=0.8)
    ax.set_ylabel('Heading (°)')
    ax.set_xlabel('Distance along raw path (m)')
    ax.set_ylim(0, 360)
    ax.set_yticks([0, 90, 180, 270, 360])
    ax.grid(True, alpha=0.3)

    # Highlight big turns
    if 'raw_turn_deg' in d:
        turns = d['raw_turn_deg']
        big = turns >= 30
        big_idx = np.where(big)[0]
        for i in big_idx:
            ax.axvline(dist_cum[i + 1], color='red', alpha=0.3, lw=0.8)


def plot_sog_profile(d, ax):
    """SOG along the raw path."""
    if 'raw_sog_kt' not in d:
        return
    sog = d['raw_sog_kt']
    dist_cum = np.concatenate([[0], np.cumsum(d['raw_seg_dist_m'])])
    ax.fill_between(dist_cum[:-1], sog, alpha=0.3, color='teal')
    ax.plot(dist_cum[:-1], sog, '-', color='teal', lw=0.8)
    ax.set_ylabel('SOG (kt)')
    ax.set_xlabel('Distance along raw path (m)')
    ax.axhline(float(d['boat_speed_kt']), color='grey', ls='--',
               lw=0.8, label=f'STW {float(d["boat_speed_kt"]):.1f} kt')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_current_wind_profile(d, ax):
    """Current and wind speed along the raw path."""
    if 'raw_cu' not in d:
        return
    dist_cum = np.concatenate([[0], np.cumsum(d['raw_seg_dist_m'])])
    cur_spd = np.hypot(d['raw_cu'], d['raw_cv']) * MS_TO_KNOTS
    ax.plot(dist_cum, cur_spd, '-', color='navy', lw=1, label='Current')

    if 'raw_wu' in d:
        wind_spd = np.hypot(d['raw_wu'], d['raw_wv']) * MS_TO_KNOTS
        ax2 = ax.twinx()
        ax2.plot(dist_cum, wind_spd, '-', color='darkorange', lw=1,
                 label='Wind')
        ax2.set_ylabel('Wind (kt)', color='darkorange')
        ax2.tick_params(axis='y', labelcolor='darkorange')

    ax.set_ylabel('Current (kt)', color='navy')
    ax.tick_params(axis='y', labelcolor='navy')
    ax.set_xlabel('Distance along raw path (m)')
    ax.grid(True, alpha=0.3)


def plot_turn_histogram(d, ax):
    """Histogram of turn angles at raw-path nodes."""
    if 'raw_turn_deg' not in d:
        return
    turns = d['raw_turn_deg']
    bins = np.arange(0, 185, 5)
    ax.hist(turns, bins=bins, color='coral', edgecolor='white', lw=0.5)
    ax.axvline(30, color='red', ls='--', lw=1, label='30° threshold')
    ax.axvline(45, color='darkred', ls='--', lw=1, label='45° tack threshold')
    ax.set_xlabel('Turn angle (°)')
    ax.set_ylabel('Count')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def _route_bounds(d, pad_frac=0.15):
    """Bounding box around the raw path with fractional padding."""
    xs = np.concatenate([d['raw_x'], d['wp_x']])
    ys = np.concatenate([d['raw_y'], d['wp_y']])
    dx = xs.max() - xs.min() or 500
    dy = ys.max() - ys.min() or 500
    pad = max(dx, dy) * pad_frac
    return xs.min() - pad, xs.max() + pad, ys.min() - pad, ys.max() + pad


def make_figure(npz_path, show_explored=True):
    """Create the full diagnostic figure from one NPZ file."""
    d = load(npz_path)
    stem = Path(npz_path).stem

    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(3, 2, hspace=0.30, wspace=0.22,
                          height_ratios=[2.5, 1, 1])

    # ---- Top left: wide view with explored cloud ----
    ax_wide = fig.add_subplot(gs[0, 0])
    plot_route_overview(d, ax_wide, show_explored=show_explored)
    ax_wide.set_aspect('equal')
    ax_wide.legend(fontsize=7, loc='upper left')
    ax_wide.set_xlabel('Easting (m, UTM)')
    ax_wide.set_ylabel('Northing (m, UTM)')
    ax_wide.set_title('Search area + route', fontsize=10)

    # ---- Top right: zoomed to route ----
    ax_zoom = fig.add_subplot(gs[0, 1])
    plot_route_overview(d, ax_zoom, show_explored=False)
    x0, x1, y0, y1 = _route_bounds(d, pad_frac=0.08)
    ax_zoom.set_xlim(x0, x1)
    ax_zoom.set_ylim(y0, y1)
    ax_zoom.set_aspect('equal')
    ax_zoom.legend(fontsize=7, loc='upper left')
    ax_zoom.set_xlabel('Easting (m, UTM)')
    ax_zoom.set_ylabel('Northing (m, UTM)')
    ax_zoom.set_title('Route detail (raw vs smoothed)', fontsize=10)

    # Suptitle with stats
    mode = str(d['router_mode']) if 'router_mode' in d else '?'
    n_exp = int(d['nodes_explored']) if 'nodes_explored' in d else 0
    t_s = float(d['total_time_s']) if 'total_time_s' in d else 0
    dist_nm = float(d['total_distance_m']) / 1852 if 'total_distance_m' in d else 0
    sog = float(d['avg_sog_kt']) if 'avg_sog_kt' in d else 0
    n_raw = len(d['raw_node_ids']) if 'raw_node_ids' in d else 0
    n_wp = len(d['wp_x']) if 'wp_x' in d else 0
    perf_total = float(d['perf_total']) if 'perf_total' in d else 0
    perf_astar = float(d['perf_astar']) if 'perf_astar' in d else 0

    fig.suptitle(
        f'{stem}  [{mode}]\n'
        f'{dist_nm:.2f} nm · {t_s/60:.1f} min · SOG {sog:.2f} kt · '
        f'{n_exp:,} explored · {n_raw} raw → {n_wp} wpts · '
        f'A* {perf_astar:.2f}s / total {perf_total:.2f}s',
        fontsize=11, y=1.01)

    # ---- Middle left: heading profile ----
    ax_hdg = fig.add_subplot(gs[1, 0])
    plot_heading_profile(d, ax_hdg)
    ax_hdg.set_title('Heading along raw path', fontsize=10)

    # ---- Middle right: SOG profile ----
    ax_sog = fig.add_subplot(gs[1, 1])
    plot_sog_profile(d, ax_sog)
    ax_sog.set_title('Speed over ground', fontsize=10)

    # ---- Bottom left: current & wind ----
    ax_env = fig.add_subplot(gs[2, 0])
    plot_current_wind_profile(d, ax_env)
    ax_env.set_title('Current & wind at path nodes', fontsize=10)

    # ---- Bottom right: turn histogram ----
    ax_hist = fig.add_subplot(gs[2, 1])
    plot_turn_histogram(d, ax_hist)
    ax_hist.set_title('Turn angle distribution', fontsize=10)

    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Plot route diagnostics from NPZ files')
    parser.add_argument('npz_files', nargs='+',
                        help='One or more .npz files from run_route.py')
    parser.add_argument('--no-explored', action='store_true',
                        help='Skip explored-nodes scatter (faster rendering)')
    parser.add_argument('--out-dir', default=None,
                        help='Output directory (default: same as NPZ)')
    args = parser.parse_args()

    for npz_path in args.npz_files:
        npz_path = Path(npz_path)
        if not npz_path.exists():
            print(f'Skipping {npz_path} — not found')
            continue

        out_dir = Path(args.out_dir) if args.out_dir else npz_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f'{npz_path.stem}_diag.png'

        print(f'Plotting {npz_path.name} …', end=' ', flush=True)
        fig = make_figure(str(npz_path),
                          show_explored=not args.no_explored)
        fig.savefig(str(out_path), dpi=150, bbox_inches='tight',
                    facecolor='white')
        plt.close(fig)
        print(f'→ {out_path.name}')


if __name__ == '__main__':
    main()
