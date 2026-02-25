#!/usr/bin/env python3
"""
Build viewer-ready coastline data:
1) load lines
2) clip to bbox
3) simplify
4) stitch nearby endpoints
5) split very long chains
6) write one LineString feature per chunk with precomputed bbox
"""

from __future__ import annotations

import argparse
from pathlib import Path

from coastline_pipeline import (
    clip_lines_to_bbox,
    compute_basic_stats,
    filter_lines_min_length_meters,
    iter_lines_from_geojson,
    lines_to_feature_collection_with_bboxes,
    load_geojson,
    save_geojson,
    simplify_lines_meters,
    split_lines_max_length_meters,
)
from stitch_coastline import stitch_lines


def parse_bbox(text: str):
    parts = [float(x.strip()) for x in text.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be lon_min,lat_min,lon_max,lat_max")
    return parts[0], parts[1], parts[2], parts[3]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build viewer-ready coastline chunks with bboxes.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/shoreline_wa_ecology_raw.geojson"),
        help="Input raw or preprocessed shoreline GeoJSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/shoreline_puget.geojson"),
        help="Output viewer shoreline GeoJSON (LineString features + bbox props)",
    )
    parser.add_argument(
        "--bbox",
        default="-123.5,46.9,-122.0,49.1",
        help="lon_min,lat_min,lon_max,lat_max",
    )
    parser.add_argument(
        "--tolerance-m",
        type=float,
        default=14.0,
        help="Simplification tolerance in meters",
    )
    parser.add_argument(
        "--min-length-m",
        type=float,
        default=40.0,
        help="Drop simplified segments shorter than this meter length",
    )
    parser.add_argument(
        "--snap-tol-m",
        type=float,
        default=8.0,
        help="Endpoint stitching tolerance in meters",
    )
    parser.add_argument(
        "--max-chunk-len-m",
        type=float,
        default=900.0,
        help="Split stitched lines longer than this meter length",
    )
    parser.add_argument(
        "--epsg",
        type=int,
        default=32610,
        help="UTM EPSG code for metric operations (default: 32610)",
    )
    parser.add_argument(
        "--source-name",
        default="WA_Ecology_viewer_prepped",
        help="Source tag written to output properties",
    )
    args = parser.parse_args()

    lon_min, lat_min, lon_max, lat_max = parse_bbox(args.bbox)
    obj = load_geojson(args.input)
    lines = list(iter_lines_from_geojson(obj))
    print(f"Input lines: {len(lines)}")
    print(f"Input stats: {compute_basic_stats(lines)}")

    lines = clip_lines_to_bbox(lines, lon_min, lat_min, lon_max, lat_max)
    print(f"After bbox clip: {len(lines)}")

    lines = simplify_lines_meters(
        lines,
        tolerance_m=args.tolerance_m,
        min_len_m=0.0,
        epsg=args.epsg,
    )
    print(f"After simplify: {len(lines)}")
    print(f"Simplified stats: {compute_basic_stats(lines)}")

    lines = stitch_lines(lines, snap_tol_m=args.snap_tol_m, epsg=args.epsg)
    print(f"After stitch: {len(lines)}")
    print(f"Stitched stats: {compute_basic_stats(lines)}")

    lines = split_lines_max_length_meters(lines, max_len_m=args.max_chunk_len_m, epsg=args.epsg)
    print(f"After max-length split: {len(lines)}")
    print(f"Split stats: {compute_basic_stats(lines)}")

    lines = filter_lines_min_length_meters(lines, min_len_m=args.min_length_m, epsg=args.epsg)
    print(f"After final min-length filter: {len(lines)}")
    print(f"Final stats: {compute_basic_stats(lines)}")

    out = lines_to_feature_collection_with_bboxes(lines, source_name=args.source_name)
    save_geojson(args.output, out)
    print(f"Wrote {args.output}")
    print(f"File size: {args.output.stat().st_size / (1024 * 1024):.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
