#!/usr/bin/env python3
"""
Overlay two shoreline GeoJSON files for visual comparison.

Typical use:
- raw/base in dark gray or black
- processed/new in red
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def collect_segments(geojson_path: Path):
    obj = json.loads(geojson_path.read_text())
    segs = []
    for feature in obj.get("features", []):
        geom = feature.get("geometry") or {}
        gtype = geom.get("type")
        if gtype == "LineString":
            coords = geom.get("coordinates") or []
            if len(coords) >= 2:
                segs.append(coords)
        elif gtype == "MultiLineString":
            for line in geom.get("coordinates") or []:
                if len(line) >= 2:
                    segs.append(line)
    return segs


def bounds_from_segments(*segments_lists):
    min_x = float("inf")
    max_x = float("-inf")
    min_y = float("inf")
    max_y = float("-inf")
    found = False
    for segs in segments_lists:
        for line in segs:
            for x, y in line:
                found = True
                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y
    if not found:
        raise RuntimeError("No coordinates found in either dataset")
    return min_x, max_x, min_y, max_y


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot base vs overlay shoreline GeoJSON.")
    parser.add_argument("--base", type=Path, required=True, help="Base GeoJSON (e.g., raw)")
    parser.add_argument("--overlay", type=Path, required=True, help="Overlay GeoJSON (e.g., processed)")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path")
    parser.add_argument("--title", default="Shoreline Compare: Base vs Processed", help="Plot title")
    parser.add_argument("--dpi", type=int, default=220, help="Output DPI")
    parser.add_argument("--figsize", default="10,10", help="Figure size (w,h) in inches")
    parser.add_argument("--base-color", default="#222222", help="Base line color")
    parser.add_argument("--overlay-color", default="#d7191c", help="Overlay line color")
    parser.add_argument("--base-width", type=float, default=0.22, help="Base line width")
    parser.add_argument("--overlay-width", type=float, default=0.65, help="Overlay line width")
    parser.add_argument("--base-alpha", type=float, default=0.45, help="Base line alpha")
    parser.add_argument("--overlay-alpha", type=float, default=0.9, help="Overlay line alpha")
    args = parser.parse_args()

    base_segs = collect_segments(args.base)
    overlay_segs = collect_segments(args.overlay)
    if not base_segs:
        raise RuntimeError(f"No line segments found in base file: {args.base}")
    if not overlay_segs:
        raise RuntimeError(f"No line segments found in overlay file: {args.overlay}")

    min_x, max_x, min_y, max_y = bounds_from_segments(base_segs, overlay_segs)
    w, h = [float(v.strip()) for v in args.figsize.split(",")]

    fig, ax = plt.subplots(figsize=(w, h), dpi=args.dpi)
    base_lc = LineCollection(
        base_segs, colors=args.base_color, linewidths=args.base_width, alpha=args.base_alpha
    )
    overlay_lc = LineCollection(
        overlay_segs, colors=args.overlay_color, linewidths=args.overlay_width, alpha=args.overlay_alpha
    )
    ax.add_collection(base_lc)
    ax.add_collection(overlay_lc)

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(args.title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)

    print(f"saved: {args.output}")
    print(f"base_segments: {len(base_segs)}")
    print(f"overlay_segments: {len(overlay_segs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
