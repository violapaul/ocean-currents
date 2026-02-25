#!/usr/bin/env python3
"""
Generate multiple simplified shoreline variants for quick visual/perf experiments.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from coastline_pipeline import (
    clip_lines_to_bbox,
    compute_basic_stats,
    iter_lines_from_geojson,
    lines_to_feature_collection,
    load_geojson,
    save_geojson,
    simplify_lines_meters,
)


def parse_bbox(text: str):
    parts = [float(x.strip()) for x in text.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be lon_min,lat_min,lon_max,lat_max")
    return parts[0], parts[1], parts[2], parts[3]


def parse_floats(text: str):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Create multiple shoreline simplification variants.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/shoreline_wa_ecology_raw.geojson"),
        help="Input raw shoreline GeoJSON",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/experiments"),
        help="Output directory for generated variants",
    )
    parser.add_argument(
        "--bbox",
        default="-123.5,46.9,-122.0,49.1",
        help="Clip bbox lon_min,lat_min,lon_max,lat_max",
    )
    parser.add_argument(
        "--tolerances-m",
        default="10,14,20",
        help="Comma-separated simplification tolerances in meters",
    )
    parser.add_argument(
        "--min-lengths-m",
        default="0,20,40",
        help="Comma-separated min segment lengths in meters",
    )
    parser.add_argument(
        "--epsg",
        type=int,
        default=32610,
        help="UTM EPSG code for metric operations (default: 32610)",
    )
    args = parser.parse_args()

    obj = load_geojson(args.input)
    lines = list(iter_lines_from_geojson(obj))
    lon_min, lat_min, lon_max, lat_max = parse_bbox(args.bbox)
    lines = clip_lines_to_bbox(lines, lon_min, lat_min, lon_max, lat_max)

    tolerances = parse_floats(args.tolerances_m)
    min_lengths = parse_floats(args.min_lengths_m)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Base input lines after clip: {len(lines)}")

    for tol in tolerances:
        for min_len in min_lengths:
            simplified = simplify_lines_meters(
                lines,
                tolerance_m=tol,
                min_len_m=min_len,
                epsg=args.epsg,
            )
            stats = compute_basic_stats(simplified)
            out_name = f"shoreline_puget_tolm{tol:.1f}_minm{min_len:.1f}.geojson".replace(".", "p")
            out_path = args.out_dir / out_name
            fc = lines_to_feature_collection(
                simplified,
                source_name=f"WA_Ecology_experiment_tolm={tol}_minm={min_len}",
            )
            save_geojson(out_path, fc)
            size_mb = out_path.stat().st_size / (1024 * 1024)
            print(
                f"{out_name}: segments={stats['segments']}, points={stats['points']}, "
                f"size={size_mb:.2f} MB"
            )

    print(f"Variants written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
