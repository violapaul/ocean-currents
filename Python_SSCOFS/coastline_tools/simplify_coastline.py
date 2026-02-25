#!/usr/bin/env python3
"""
Simplify raw shoreline GeoJSON into a compact viewer-ready MultiLineString.
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Simplify shoreline GeoJSON.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/shoreline_wa_ecology_raw.geojson"),
        help="Input raw shoreline GeoJSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/shoreline_puget.geojson"),
        help="Output simplified shoreline GeoJSON",
    )
    parser.add_argument(
        "--bbox",
        default="-123.5,46.9,-122.0,49.1",
        help="Clip bbox lon_min,lat_min,lon_max,lat_max",
    )
    parser.add_argument(
        "--tolerance-m",
        type=float,
        default=14.0,
        help="Point thinning tolerance in meters (default: 14.0)",
    )
    parser.add_argument(
        "--min-length-m",
        type=float,
        default=40.0,
        help="Drop simplified segments shorter than this meter length",
    )
    parser.add_argument(
        "--epsg",
        type=int,
        default=32610,
        help="UTM EPSG code for metric operations (default: 32610)",
    )
    parser.add_argument(
        "--source-name",
        default="WA_Ecology_CoastalAtlas_L13_simplified",
        help="Output source name in GeoJSON properties",
    )
    args = parser.parse_args()

    obj = load_geojson(args.input)
    lines = list(iter_lines_from_geojson(obj))
    print(f"Input lines: {len(lines)}")
    print(f"Input stats: {compute_basic_stats(lines)}")

    lon_min, lat_min, lon_max, lat_max = parse_bbox(args.bbox)
    lines = clip_lines_to_bbox(lines, lon_min, lat_min, lon_max, lat_max)
    print(f"After bbox clip: {len(lines)}")

    simplified = simplify_lines_meters(
        lines,
        tolerance_m=args.tolerance_m,
        min_len_m=args.min_length_m,
        epsg=args.epsg,
    )
    print(f"Simplified lines: {len(simplified)}")
    print(f"Simplified stats: {compute_basic_stats(simplified)}")

    output = lines_to_feature_collection(simplified, source_name=args.source_name)
    save_geojson(args.output, output)
    print(f"Wrote {args.output}")
    print(f"File size: {args.output.stat().st_size / (1024 * 1024):.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
