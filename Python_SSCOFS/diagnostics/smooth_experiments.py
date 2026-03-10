"""
smooth_experiments.py
---------------------
Load raw A* paths from NPZ files and experiment with smoothing
algorithms.  No routing needed — works purely from saved data.

Usage:
    python diagnostics/smooth_experiments.py routes/output/shilshole_ttp_return_leg01.npz
    python diagnostics/smooth_experiments.py routes/output/*.npz
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
DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi


# =====================================================================
# Sailability helpers (standalone, no CurrentField needed)
# =====================================================================

def wind_direction_at(wu, wv):
    """True wind direction (where wind comes FROM, degrees CW from N)."""
    return (np.degrees(np.arctan2(-wu, -wv)) + 360) % 360


def true_wind_angle(heading_deg, wind_from_deg):
    """Angle between sailing heading and true wind (0 = into the wind)."""
    d = abs(heading_deg - wind_from_deg)
    if d > 180:
        d = 360 - d
    return d


def segment_heading(x1, y1, x2, y2):
    """Heading in degrees CW from north."""
    return (np.degrees(np.arctan2(x2 - x1, y2 - y1)) + 360) % 360


def hdiff(h1, h2):
    """Absolute heading difference (0-180)."""
    d = abs(h2 - h1)
    return d if d <= 180 else 360 - d


def interpolate_wind(raw_x, raw_y, raw_wu, raw_wv, qx, qy):
    """Get wind at (qx,qy) by nearest raw-path node."""
    dists = np.hypot(raw_x - qx, raw_y - qy)
    idx = np.argmin(dists)
    return raw_wu[idx], raw_wv[idx]


def is_sailable(x1, y1, x2, y2, raw_x, raw_y, raw_wu, raw_wv,
                nogo_angle=32.0):
    """Check if a straight segment is sailable (heading outside no-go).

    Uses wind at the segment midpoint, interpolated from the nearest
    raw-path node.  The threshold is COG-based (not STW), so it must be
    lower than the polar's beat TWA to account for current offsetting
    COG from the boat's heading through water.
    """
    dist = np.hypot(x2 - x1, y2 - y1)
    if dist < 1.0:
        return True
    hdg = segment_heading(x1, y1, x2, y2)
    mx, my = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    wu, wv = interpolate_wind(raw_x, raw_y, raw_wu, raw_wv, mx, my)
    wind_from = wind_direction_at(wu, wv)
    twa = true_wind_angle(hdg, wind_from)
    return twa >= nogo_angle


def path_crosses_raw(raw_x, raw_y, idx_a, idx_b, max_dev_m=200.0):
    """Check if the straight line from raw[idx_a] to raw[idx_b] deviates
    more than max_dev_m from the raw path between those indices.
    Proxy for land avoidance.
    """
    x1, y1 = raw_x[idx_a], raw_y[idx_a]
    x2, y2 = raw_x[idx_b], raw_y[idx_b]
    seg_len = np.hypot(x2 - x1, y2 - y1)
    if seg_len < 1e-6:
        return False
    px = raw_x[idx_a:idx_b + 1]
    py = raw_y[idx_a:idx_b + 1]
    devs = np.abs((x2 - x1) * (y1 - py) - (x1 - px) * (y2 - y1)) / seg_len
    return devs.max() > max_dev_m


# =====================================================================
# Smoothing algorithms
# =====================================================================

def detect_tacks(headings, threshold_deg=45.0):
    """Find indices where heading changes >= threshold (tack points).
    Returns indices into the heading array (which is 1 shorter than nodes).
    Tack at index i means the turn happens at node i+1 in the path.
    """
    tack_indices = []
    for i in range(1, len(headings)):
        if hdiff(headings[i - 1], headings[i]) >= threshold_deg:
            tack_indices.append(i)  # node index = i (heading i is seg i→i+1)
    return tack_indices


def merge_nearby_tacks(tack_indices, cum_dist, headings, min_spacing_m=400.0):
    """Collapse tacks that are too close together."""
    if not tack_indices:
        return []
    merged = [tack_indices[0]]
    for ti in tack_indices[1:]:
        if cum_dist[ti] - cum_dist[merged[-1]] < min_spacing_m:
            prev = merged[-1]
            prev_turn = hdiff(headings[max(0, prev - 1)],
                              headings[min(prev, len(headings) - 1)])
            this_turn = hdiff(headings[max(0, ti - 1)],
                              headings[min(ti, len(headings) - 1)])
            if this_turn > prev_turn:
                merged[-1] = ti
        else:
            merged.append(ti)
    return merged


def sailable_dp(raw_x, raw_y, raw_wu, raw_wv,
                indices, shape_tol_m=120.0, nogo_angle=38.0):
    """DP-like simplification where every output segment is sailable.

    Parameters
    ----------
    indices : list of int
        Indices into raw_x/raw_y for this segment (contiguous slice).
    shape_tol_m : float
        Max allowed perpendicular deviation when segment is sailable.
    nogo_angle : float
        Minimum TWA for a segment to be considered sailable.

    Returns list of indices (into raw_x/raw_y) for the simplified segment.
    """
    if len(indices) <= 2:
        return list(indices)

    first, last = indices[0], indices[-1]
    x1, y1 = raw_x[first], raw_y[first]
    x2, y2 = raw_x[last], raw_y[last]

    # Check if direct connection is sailable
    sailable = is_sailable(x1, y1, x2, y2,
                           raw_x, raw_y, raw_wu, raw_wv, nogo_angle)

    if sailable:
        seg_len = np.hypot(x2 - x1, y2 - y1)
        if seg_len < 1e-6:
            return [first, last]
        # Check shape deviation
        px = raw_x[indices]
        py = raw_y[indices]
        devs = np.abs((x2 - x1) * (y1 - py) -
                       (x1 - px) * (y2 - y1)) / seg_len
        if devs.max() <= shape_tol_m:
            return [first, last]

    # Must split — find most-deviating point
    seg_len = np.hypot(x2 - x1, y2 - y1)
    if seg_len < 1e-6:
        return [first, last]

    px = raw_x[indices]
    py = raw_y[indices]
    devs = np.abs((x2 - x1) * (y1 - py) -
                   (x1 - px) * (y2 - y1)) / seg_len
    devs[0] = devs[-1] = 0.0
    local_split = int(np.argmax(devs))
    if local_split == 0:
        local_split = len(indices) // 2

    left = sailable_dp(raw_x, raw_y, raw_wu, raw_wv,
                        indices[:local_split + 1], shape_tol_m, nogo_angle)
    right = sailable_dp(raw_x, raw_y, raw_wu, raw_wv,
                         indices[local_split:], shape_tol_m, nogo_angle)
    return left + right[1:]


def smooth_sailable(d, tack_threshold=45.0, min_spacing_m=400.0,
                    shape_tol_m=120.0, nogo_angle=32.0):
    """Full sailable smoothing pipeline on NPZ data.

    Returns dict with 'indices', 'x', 'y', 'headings', 'tack_boundaries'.
    """
    raw_x = d['raw_x']
    raw_y = d['raw_y']
    raw_wu = d['raw_wu']
    raw_wv = d['raw_wv']
    headings = d['raw_heading_deg']
    seg_dists = d['raw_seg_dist_m']
    n = len(raw_x)

    cum_dist = np.concatenate([[0.0], np.cumsum(seg_dists)])

    # Detect and merge tacks
    raw_tacks = detect_tacks(headings, tack_threshold)
    merged_tacks = merge_nearby_tacks(raw_tacks, cum_dist, headings,
                                       min_spacing_m)

    boundaries = sorted(set([0] + merged_tacks + [n - 1]))

    # Simplify each segment
    result_indices = []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        seg_idx = list(range(s, e + 1))
        simplified = sailable_dp(raw_x, raw_y, raw_wu, raw_wv,
                                  seg_idx, shape_tol_m, nogo_angle)
        if result_indices and result_indices[-1] == simplified[0]:
            result_indices.extend(simplified[1:])
        else:
            result_indices.extend(simplified)

    result_indices = sorted(set(result_indices))
    sx = raw_x[result_indices]
    sy = raw_y[result_indices]

    # Compute heading and TWA for each simplified segment
    seg_hdg = []
    seg_twa = []
    for k in range(len(result_indices) - 1):
        i, j = result_indices[k], result_indices[k + 1]
        h = segment_heading(raw_x[i], raw_y[i], raw_x[j], raw_y[j])
        seg_hdg.append(h)
        mx = 0.5 * (raw_x[i] + raw_x[j])
        my = 0.5 * (raw_y[i] + raw_y[j])
        wu, wv = interpolate_wind(raw_x, raw_y, raw_wu, raw_wv, mx, my)
        wf = wind_direction_at(wu, wv)
        seg_twa.append(true_wind_angle(h, wf))

    n_unsailable = sum(1 for t in seg_twa if t < nogo_angle)

    return {
        'indices': result_indices,
        'x': sx,
        'y': sy,
        'headings': np.array(seg_hdg),
        'twa': np.array(seg_twa),
        'tack_boundaries': boundaries,
        'n_unsailable': n_unsailable,
    }


# =====================================================================
# Visualization
# =====================================================================

def plot_comparison(d, results, title_extra=''):
    """Create a comparison figure.

    Parameters
    ----------
    d : NPZ data dict
    results : dict  name -> smooth_sailable() output
    """
    n_methods = len(results)
    fig = plt.figure(figsize=(8 + 6 * n_methods, 14))
    gs = fig.add_gridspec(3, 1 + n_methods, hspace=0.30, wspace=0.25,
                          height_ratios=[2.5, 1, 1])

    raw_x, raw_y = d['raw_x'], d['raw_y']
    headings = d['raw_heading_deg']
    seg_dists = d['raw_seg_dist_m']
    cum_dist = np.concatenate([[0.0], np.cumsum(seg_dists)])

    # ── Bounding box ────────────────────────────────────────────────────
    all_x = np.concatenate([raw_x] + [r['x'] for r in results.values()])
    all_y = np.concatenate([raw_y] + [r['y'] for r in results.values()])
    dx = all_x.max() - all_x.min() or 500
    dy = all_y.max() - all_y.min() or 500
    pad = max(dx, dy) * 0.08
    xlim = (all_x.min() - pad, all_x.max() + pad)
    ylim = (all_y.min() - pad, all_y.max() + pad)

    # ── Column 0: original (from NPZ) ──────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(raw_x, raw_y, '-', color='royalblue', lw=0.8, alpha=0.6)
    ax0.plot(raw_x, raw_y, '.', color='royalblue', ms=2, alpha=0.5)
    if 'wp_x' in d:
        ax0.plot(d['wp_x'], d['wp_y'], '-', color='darkorange', lw=2.5,
                 path_effects=OUTLINE)
        ax0.plot(d['wp_x'], d['wp_y'], 'D', color='darkorange', ms=6,
                 markeredgecolor='black', markeredgewidth=0.5)
    ax0.plot(raw_x[0], raw_y[0], 'o', color='limegreen', ms=10,
             markeredgecolor='black', markeredgewidth=1, zorder=8)
    ax0.plot(raw_x[-1], raw_y[-1], 's', color='tomato', ms=10,
             markeredgecolor='black', markeredgewidth=1, zorder=8)
    ax0.set_xlim(xlim)
    ax0.set_ylim(ylim)
    ax0.set_aspect('equal')
    n_wp = len(d['wp_x']) if 'wp_x' in d else 0
    ax0.set_title(f'Current ({n_wp} wpts)', fontsize=10)
    ax0.grid(True, alpha=0.2)

    # ── Columns 1..N: each smoothing result ─────────────────────────────
    colors = ['#e63946', '#2a9d8f', '#6a4c93', '#f77f00']
    for col_idx, (name, res) in enumerate(results.items(), start=1):
        ax = fig.add_subplot(gs[0, col_idx])
        ax.plot(raw_x, raw_y, '-', color='royalblue', lw=0.8, alpha=0.4)
        ax.plot(raw_x, raw_y, '.', color='royalblue', ms=1.5, alpha=0.3)

        c = colors[(col_idx - 1) % len(colors)]
        ax.plot(res['x'], res['y'], '-', color=c, lw=2.5,
                path_effects=OUTLINE)
        ax.plot(res['x'], res['y'], 'D', color=c, ms=6,
                markeredgecolor='black', markeredgewidth=0.5)

        # Mark tack boundaries
        for ti in res['tack_boundaries']:
            if 0 < ti < len(raw_x) - 1:
                ax.plot(raw_x[ti], raw_y[ti], 'x', color='red',
                        ms=8, mew=2, zorder=9)

        ax.plot(raw_x[0], raw_y[0], 'o', color='limegreen', ms=10,
                markeredgecolor='black', markeredgewidth=1, zorder=8)
        ax.plot(raw_x[-1], raw_y[-1], 's', color='tomato', ms=10,
                markeredgecolor='black', markeredgewidth=1, zorder=8)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        n_pts = len(res['x'])
        n_bad = res['n_unsailable']
        ax.set_title(f'{name}\n{n_pts} wpts, {n_bad} unsailable segs',
                     fontsize=10)
        ax.grid(True, alpha=0.2)

    # ── Row 1: heading profile with tack markers ────────────────────────
    ax_hdg = fig.add_subplot(gs[1, :])
    ax_hdg.plot(cum_dist[:-1], headings, '.-', color='steelblue',
                ms=2, lw=0.6)
    ax_hdg.set_ylabel('Heading (°)')
    ax_hdg.set_xlabel('Distance along raw path (m)')
    ax_hdg.set_ylim(0, 360)
    ax_hdg.set_yticks([0, 90, 180, 270, 360])
    ax_hdg.grid(True, alpha=0.3)
    ax_hdg.set_title('Raw heading profile + tack detections', fontsize=10)

    # Overlay wind direction
    if 'raw_wu' in d:
        wf = wind_direction_at(d['raw_wu'], d['raw_wv'])
        ax_hdg.plot(cum_dist, wf, '-', color='orange', lw=1, alpha=0.6,
                    label='Wind from (°)')
        ax_hdg.legend(fontsize=8, loc='upper right')

    # Mark detected tacks for each method
    for col_idx, (name, res) in enumerate(results.items()):
        c = colors[col_idx % len(colors)]
        for ti in res['tack_boundaries']:
            if 0 < ti < len(raw_x) - 1:
                ax_hdg.axvline(cum_dist[ti], color=c, alpha=0.4, lw=1.2,
                               ls='--')

    # ── Row 2: TWA profile per method ───────────────────────────────────
    ax_twa = fig.add_subplot(gs[2, :])
    for col_idx, (name, res) in enumerate(results.items()):
        c = colors[col_idx % len(colors)]
        if len(res['twa']) > 0:
            # x-position at midpoint of each simplified segment
            seg_mids = []
            for k in range(len(res['indices']) - 1):
                i, j = res['indices'][k], res['indices'][k + 1]
                seg_mids.append(0.5 * (cum_dist[i] + cum_dist[j]))
            ax_twa.plot(seg_mids, res['twa'], '.-', color=c, lw=1.2,
                        ms=4, label=name)

    ax_twa.axhline(32, color='red', ls='--', lw=1.2,
                    label='No-go COG (32°)')
    ax_twa.axhline(39, color='orange', ls=':', lw=1,
                    label='Polar beat TWA (39°)')
    ax_twa.set_ylabel('TWA per segment (°)')
    ax_twa.set_xlabel('Distance along raw path (m)')
    ax_twa.set_ylim(0, 180)
    ax_twa.grid(True, alpha=0.3)
    ax_twa.legend(fontsize=8, loc='lower right')
    ax_twa.set_title('True Wind Angle of simplified segments '
                     '(below red = unsailable)', fontsize=10)

    stem = title_extra or 'smooth_comparison'
    fig.suptitle(stem, fontsize=12, y=1.01)
    return fig


# =====================================================================
# Main
# =====================================================================

def run_one(npz_path, out_dir):
    d = np.load(npz_path, allow_pickle=True)
    stem = Path(npz_path).stem

    results = {}

    # Method A: conservative (loose tolerance, standard tack threshold)
    results['Sailable DP\ntol=150m tack=45°'] = smooth_sailable(
        d, tack_threshold=45.0, min_spacing_m=400.0,
        shape_tol_m=150.0)

    # Method B: tighter shape tolerance
    results['Sailable DP\ntol=80m tack=45°'] = smooth_sailable(
        d, tack_threshold=45.0, min_spacing_m=400.0,
        shape_tol_m=80.0)

    # Method C: lower tack threshold (catch smaller heading changes)
    results['Sailable DP\ntol=120m tack=35°'] = smooth_sailable(
        d, tack_threshold=35.0, min_spacing_m=300.0,
        shape_tol_m=120.0)

    fig = plot_comparison(d, results, title_extra=stem)

    out_path = out_dir / f'{stem}_smooth_exp.png'
    fig.savefig(str(out_path), dpi=130, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    print(f'  → {out_path.name}')

    # Print summary
    for name, res in results.items():
        label = name.replace('\n', ' / ')
        print(f'    {label:40s}  '
              f'{len(res["x"]):3d} wpts  '
              f'{res["n_unsailable"]:2d} unsailable  '
              f'tacks: {len(res["tack_boundaries"])-2}')


def main():
    parser = argparse.ArgumentParser(
        description='Experiment with route smoothing on NPZ files')
    parser.add_argument('npz_files', nargs='+')
    parser.add_argument('--out-dir', default=None)
    args = parser.parse_args()

    for path_str in args.npz_files:
        p = Path(path_str)
        if not p.exists():
            print(f'Skipping {p} — not found')
            continue
        out_dir = Path(args.out_dir) if args.out_dir else p.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f'Processing {p.name} …')
        run_one(str(p), out_dir)


if __name__ == '__main__':
    main()
