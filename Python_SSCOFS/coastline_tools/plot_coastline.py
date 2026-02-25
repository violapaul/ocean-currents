#!/usr/bin/env python3
"""
Plot shoreline GeoJSON as a quick matplotlib diagnostic image.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot coastline GeoJSON to PNG.")
    parser.add_argument("--input", type=Path, required=True, help="Input GeoJSON file")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path")
    parser.add_argument("--title", default="Coastline Diagnostic", help="Plot title")
    parser.add_argument("--line-width", type=float, default=0.25, help="Line width in px")
    parser.add_argument("--dpi", type=int, default=220, help="PNG DPI")
    parser.add_argument("--figsize", default="10,10", help="Figure size in inches, e.g. 10,10")
    args = parser.parse_args()

    obj = json.loads(args.input.read_text())
    segments = []
    min_x = float("inf")
    max_x = float("-inf")
    min_y = float("inf")
    max_y = float("-inf")

    for feature in obj.get("features", []):
        geom = feature.get("geometry") or {}
        gtype = geom.get("type")
        if gtype == "LineString":
            coords = geom.get("coordinates") or []
            if len(coords) >= 2:
                segments.append(coords)
        elif gtype == "MultiLineString":
            for line in geom.get("coordinates") or []:
                if len(line) >= 2:
                    segments.append(line)

    if not segments:
        raise RuntimeError("No line segments found in input GeoJSON")

    for line in segments:
        for x, y in line:
            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y

    w, h = [float(v.strip()) for v in args.figsize.split(",")]
    fig, ax = plt.subplots(figsize=(w, h), dpi=args.dpi)
    lc = LineCollection(segments, colors="black", linewidths=args.line_width, alpha=0.9)
    ax.add_collection(lc)
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
    print(f"segments: {len(segments)}")
    print(f"size_mb: {args.output.stat().st_size / (1024 * 1024):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
