#!/usr/bin/env python
"""Plot mesh router diagnostic from saved debug data."""

import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pathlib import Path

HERE = Path(__file__).parent

with open("/tmp/mesh_debug_data.pkl", "rb") as f:
    d = pickle.load(f)

raw_path = d['raw_path']
smooth_path = d['smooth_path']
node_x = d['node_x']
node_y = d['node_y']
transformer = d['transformer']
start_ll = d['start_ll']
end_ll = d['end_ll']

raw_utm = [(node_x[n], node_y[n]) for n in raw_path]
smooth_utm = [(node_x[n], node_y[n]) for n in smooth_path]

sx, sy = transformer.transform(start_ll[1], start_ll[0])
ex, ey = transformer.transform(end_ll[1], end_ll[0])


def make_plot(xlim, ylim, suffix, title_extra=""):
    fig, ax = plt.subplots(figsize=(14, 14))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#0f1923")

    # Raw mesh path
    rx = [p[0] for p in raw_utm]
    ry = [p[1] for p in raw_utm]
    ax.plot(rx, ry, '-', color="#ff8800", linewidth=1.0, alpha=0.7,
            zorder=5, label=f"Raw mesh A* ({len(raw_path)} nodes)")
    ax.scatter(rx, ry, s=12, c="#ffaa00", zorder=6, alpha=0.9,
               edgecolors="#885500", linewidths=0.5)

    # Number every raw node in zoomed views
    if "zoom" in suffix:
        for idx, (px, py) in enumerate(raw_utm):
            if xlim[0] <= px <= xlim[1] and ylim[0] <= py <= ylim[1]:
                ax.annotate(str(idx), (px, py),
                            textcoords="offset points", xytext=(-6, -10),
                            fontsize=5.5, color="#ffcc66",
                            fontfamily="monospace", alpha=0.8)

    # Smoothed path
    smx = [p[0] for p in smooth_utm]
    smy = [p[1] for p in smooth_utm]
    ax.plot(smx, smy, '-', color="#00ff88", linewidth=2.5, alpha=0.9,
            zorder=7, label=f"Smoothed ({len(smooth_path)} waypoints)",
            path_effects=[pe.Stroke(linewidth=4, foreground="#000000",
                                    alpha=0.4), pe.Normal()])
    ax.scatter(smx, smy, s=80, c="#00ff88", marker="s", zorder=8,
               edgecolors="white", linewidths=1.5)

    # Label smoothed waypoints
    raw_list = list(raw_path)
    for i, node in enumerate(smooth_path):
        px, py = node_x[node], node_y[node]
        raw_idx = raw_list.index(node)
        ax.annotate(f"{i}(r{raw_idx})", (px, py),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=7, color="white", fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor="#000000", alpha=0.7))

    # Start / end
    ax.plot(sx, sy, "o", color="#2ecc71", markersize=14, zorder=10,
            markeredgecolor="white", markeredgewidth=2.5, label="Start")
    ax.plot(ex, ey, "s", color="#e74c3c", markersize=14, zorder=10,
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

    title = f"Mesh Router: Raw A* vs Smoothed{title_extra}"
    ax.set_title(title, color="white", fontsize=13, pad=10)
    ax.legend(loc="upper left", framealpha=0.85,
              facecolor="#0d1b2a", edgecolor="#334455",
              labelcolor="white", fontsize=9)

    plt.tight_layout()
    out = HERE / "routes" / "output" / f"diagnostic_mesh_{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out}")


# Full route
x_all = [p[0] for p in raw_utm]
y_all = [p[1] for p in raw_utm]
pad = 3000
make_plot(
    xlim=(min(x_all) - pad, max(x_all) + pad),
    ylim=(min(y_all) - pad, max(y_all) + pad),
    suffix="full",
    title_extra=" — Full Route",
)

# Zoom: first ~60 nodes near Shilshole
n_zoom = min(60, len(raw_utm))
zx = [raw_utm[i][0] for i in range(n_zoom)]
zy = [raw_utm[i][1] for i in range(n_zoom)]
zpad = 800
make_plot(
    xlim=(min(zx) - zpad, max(zx) + zpad),
    ylim=(min(zy) - zpad, max(zy) + zpad),
    suffix="zoom_start",
    title_extra=" — Zoomed: First 60 nodes near Shilshole",
)

# Zoom: middle section (nodes 60-140)
mid_start = 60
mid_end = min(140, len(raw_utm))
mx = [raw_utm[i][0] for i in range(mid_start, mid_end)]
my = [raw_utm[i][1] for i in range(mid_start, mid_end)]
if mx:
    mpad = 800
    make_plot(
        xlim=(min(mx) - mpad, max(mx) + mpad),
        ylim=(min(my) - mpad, max(my) + mpad),
        suffix="zoom_mid",
        title_extra=f" — Zoomed: Nodes {mid_start}-{mid_end}",
    )
