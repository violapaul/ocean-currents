#!/usr/bin/env python3
"""
Shared utilities for shoreline data processing experiments.

All functions are standard-library only so these scripts run in
minimal environments.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

LonLat = Tuple[float, float]
Line = List[LonLat]


def _get_utm_transformers(epsg: int = 32610):
    """
    Return forward/inverse pyproj transformers for WGS84 <-> UTM.
    """
    try:
        from pyproj import Transformer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "pyproj is required for UTM-meter operations. "
            "Install with: conda install -c conda-forge pyproj"
        ) from exc
    fwd = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    inv = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    return fwd, inv


def load_geojson(path: Path) -> Dict:
    """Load a GeoJSON object from disk."""
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def save_geojson(path: Path, obj: Dict) -> None:
    """Write compact GeoJSON to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(obj, fp, separators=(",", ":"))


def iter_lines_from_geojson(obj: Dict) -> Iterable[Line]:
    """Yield LineString coordinate arrays from FeatureCollection data."""
    for feature in obj.get("features", []):
        geom = feature.get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates") or []
        if gtype == "LineString":
            if len(coords) >= 2:
                yield [(float(p[0]), float(p[1])) for p in coords]
        elif gtype == "MultiLineString":
            for line in coords:
                if len(line) >= 2:
                    yield [(float(p[0]), float(p[1])) for p in line]


def line_length_degrees(line: Sequence[LonLat]) -> float:
    """Approximate line length in degree-space."""
    total = 0.0
    for i in range(1, len(line)):
        dx = line[i][0] - line[i - 1][0]
        dy = line[i][1] - line[i - 1][1]
        total += math.hypot(dx, dy)
    return total


def line_length_meters(line: Sequence[LonLat], epsg: int = 32610) -> float:
    """Approximate line length in meters using UTM projection."""
    if len(line) < 2:
        return 0.0
    fwd, _inv = _get_utm_transformers(epsg=epsg)
    xs, ys = fwd.transform([p[0] for p in line], [p[1] for p in line])
    total = 0.0
    for i in range(1, len(xs)):
        dx = xs[i] - xs[i - 1]
        dy = ys[i] - ys[i - 1]
        total += math.hypot(dx, dy)
    return total


def line_bbox(line: Sequence[LonLat]) -> Tuple[float, float, float, float]:
    """Return (lon_min, lat_min, lon_max, lat_max) for a line."""
    xs = [p[0] for p in line]
    ys = [p[1] for p in line]
    return (min(xs), min(ys), max(xs), max(ys))


def clip_lines_to_bbox(
    lines: Iterable[Line], lon_min: float, lat_min: float, lon_max: float, lat_max: float
) -> List[Line]:
    """Keep lines whose bounding boxes intersect the provided bbox."""
    kept: List[Line] = []
    for line in lines:
        xs = [p[0] for p in line]
        ys = [p[1] for p in line]
        if not xs or not ys:
            continue
        if max(xs) < lon_min or min(xs) > lon_max or max(ys) < lat_min or min(ys) > lat_max:
            continue
        kept.append(line)
    return kept


def simplify_line(line: Sequence[LonLat], tolerance_deg: float) -> Line:
    """
    Very fast point thinning:
    keep a point only when it moves beyond tolerance from the last kept point.
    """
    if len(line) <= 2:
        return list(line)
    out: Line = [line[0]]
    last_x, last_y = line[0]
    tol2 = tolerance_deg * tolerance_deg
    for p in line[1:-1]:
        dx = p[0] - last_x
        dy = p[1] - last_y
        if (dx * dx + dy * dy) >= tol2:
            out.append(p)
            last_x, last_y = p
    out.append(line[-1])
    return out


def simplify_line_meters(line: Sequence[LonLat], tolerance_m: float, epsg: int = 32610) -> Line:
    """
    Point thinning in UTM meters:
    keep a point only when it moves beyond tolerance_m from last kept point.
    """
    if len(line) <= 2:
        return list(line)
    fwd, _inv = _get_utm_transformers(epsg=epsg)
    xs, ys = fwd.transform([p[0] for p in line], [p[1] for p in line])
    out: Line = [line[0]]
    last_x = xs[0]
    last_y = ys[0]
    tol2 = tolerance_m * tolerance_m
    for i in range(1, len(line) - 1):
        dx = xs[i] - last_x
        dy = ys[i] - last_y
        if (dx * dx + dy * dy) >= tol2:
            out.append(line[i])
            last_x = xs[i]
            last_y = ys[i]
    out.append(line[-1])
    return out


def simplify_lines(
    lines: Iterable[Line], tolerance_deg: float, min_len_deg: float
) -> List[Line]:
    """Simplify all lines and drop very short remnants."""
    out: List[Line] = []
    for line in lines:
        slim = simplify_line(line, tolerance_deg=tolerance_deg)
        if len(slim) < 2:
            continue
        if line_length_degrees(slim) < min_len_deg:
            continue
        out.append(slim)
    return out


def simplify_lines_meters(
    lines: Iterable[Line], tolerance_m: float, min_len_m: float, epsg: int = 32610
) -> List[Line]:
    """Simplify all lines with meter thresholds and drop short remnants."""
    out: List[Line] = []
    for line in lines:
        slim = simplify_line_meters(line, tolerance_m=tolerance_m, epsg=epsg)
        if len(slim) < 2:
            continue
        if line_length_meters(slim, epsg=epsg) < min_len_m:
            continue
        out.append(slim)
    return out


def filter_lines_min_length_meters(
    lines: Iterable[Line], min_len_m: float, epsg: int = 32610
) -> List[Line]:
    """Keep only lines whose UTM-meter length is >= min_len_m."""
    if min_len_m <= 0:
        return list(lines)
    out: List[Line] = []
    for line in lines:
        if len(line) < 2:
            continue
        if line_length_meters(line, epsg=epsg) >= min_len_m:
            out.append(line)
    return out


def split_line_max_length(line: Sequence[LonLat], max_len_deg: float) -> List[Line]:
    """
    Split a line into sub-lines whose approximate accumulated degree-length
    does not exceed max_len_deg.
    """
    if len(line) < 2:
        return []
    parts: List[Line] = []
    cur: Line = [line[0]]
    accum = 0.0
    for i in range(1, len(line)):
        p0 = line[i - 1]
        p1 = line[i]
        seg_len = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        if len(cur) > 1 and (accum + seg_len) > max_len_deg:
            parts.append(cur)
            cur = [p0, p1]
            accum = seg_len
        else:
            cur.append(p1)
            accum += seg_len
    if len(cur) >= 2:
        parts.append(cur)
    return parts


def split_line_max_length_meters(
    line: Sequence[LonLat], max_len_m: float, epsg: int = 32610
) -> List[Line]:
    """
    Split a line into sub-lines whose accumulated UTM-meter length
    does not exceed max_len_m.
    """
    if len(line) < 2:
        return []
    fwd, _inv = _get_utm_transformers(epsg=epsg)
    xs, ys = fwd.transform([p[0] for p in line], [p[1] for p in line])
    parts: List[Line] = []
    cur: Line = [line[0]]
    accum = 0.0
    for i in range(1, len(line)):
        seg_len = math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1])
        if len(cur) > 1 and (accum + seg_len) > max_len_m:
            parts.append(cur)
            cur = [line[i - 1], line[i]]
            accum = seg_len
        else:
            cur.append(line[i])
            accum += seg_len
    if len(cur) >= 2:
        parts.append(cur)
    return parts


def split_lines_max_length(lines: Iterable[Line], max_len_deg: float) -> List[Line]:
    """Apply max-length splitting to all lines."""
    out: List[Line] = []
    for line in lines:
        out.extend(split_line_max_length(line, max_len_deg=max_len_deg))
    return out


def split_lines_max_length_meters(
    lines: Iterable[Line], max_len_m: float, epsg: int = 32610
) -> List[Line]:
    """Apply max-length splitting in meters to all lines."""
    out: List[Line] = []
    for line in lines:
        out.extend(split_line_max_length_meters(line, max_len_m=max_len_m, epsg=epsg))
    return out


def lines_to_feature_collection(lines: Sequence[Line], source_name: str) -> Dict:
    """Create a compact FeatureCollection with a single MultiLineString feature."""
    return {
        "type": "FeatureCollection",
        "name": "shoreline_puget",
        "features": [
            {
                "type": "Feature",
                "properties": {"source": source_name},
                "geometry": {"type": "MultiLineString", "coordinates": lines},
            }
        ],
    }


def lines_to_feature_collection_with_bboxes(lines: Sequence[Line], source_name: str) -> Dict:
    """
    Create FeatureCollection with one LineString per segment and precomputed bbox
    in properties to speed runtime viewport pruning.
    """
    features = []
    for i, line in enumerate(lines):
        if len(line) < 2:
            continue
        bbox = line_bbox(line)
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "source": source_name,
                    "segment_id": i,
                    "bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],
                },
                "geometry": {"type": "LineString", "coordinates": line},
            }
        )
    return {
        "type": "FeatureCollection",
        "name": "shoreline_puget",
        "features": features,
    }


def compute_basic_stats(lines: Sequence[Line]) -> Dict[str, float]:
    """Return summary stats useful when tuning simplification."""
    if not lines:
        return {"segments": 0, "points": 0, "avg_points_per_segment": 0.0, "total_len_deg": 0.0}
    points = sum(len(line) for line in lines)
    total_len = sum(line_length_degrees(line) for line in lines)
    return {
        "segments": len(lines),
        "points": points,
        "avg_points_per_segment": points / len(lines),
        "total_len_deg": total_len,
    }
