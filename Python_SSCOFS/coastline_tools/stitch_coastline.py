#!/usr/bin/env python3
"""
Stitch nearby coastline segment endpoints into longer continuous lines.

This moves contiguity logic into offline preprocessing so the browser renderer
can remain simple and fast.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from coastline_pipeline import (
    _get_utm_transformers,
    compute_basic_stats,
    iter_lines_from_geojson,
    lines_to_feature_collection,
    load_geojson,
    save_geojson,
)

LonLat = Tuple[float, float]
Line = List[LonLat]


class EndpointClusterer:
    """Cluster endpoints in UTM meter space using a spatial hash."""

    def __init__(self, tolerance_m: float, epsg: int):
        self.tol = tolerance_m
        self.cell = tolerance_m
        self.centers: List[List[float]] = []  # [x_m, y_m, count]
        self.buckets: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        self.center_bucket: Dict[int, Tuple[int, int]] = {}
        self.fwd, self.inv = _get_utm_transformers(epsg=epsg)

    def _bucket_id(self, x: float, y: float) -> Tuple[int, int]:
        return (int(round(x / self.cell)), int(round(y / self.cell)))

    def assign(self, lon: float, lat: float) -> int:
        x, y = self.fwd.transform(lon, lat)
        bx, by = self._bucket_id(x, y)
        tol2 = self.tol * self.tol
        best_id = -1
        best_d2 = float("inf")

        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                for cid in self.buckets.get((bx + ox, by + oy), []):
                    cx, cy, _cnt = self.centers[cid]
                    dx = x - cx
                    dy = y - cy
                    d2 = dx * dx + dy * dy
                    if d2 <= tol2 and d2 < best_d2:
                        best_id = cid
                        best_d2 = d2

        if best_id < 0:
            cid = len(self.centers)
            self.centers.append([x, y, 1.0])
            self.buckets[(bx, by)].append(cid)
            self.center_bucket[cid] = (bx, by)
            return cid

        # Online centroid update
        cx, cy, cnt = self.centers[best_id]
        ncnt = cnt + 1.0
        self.centers[best_id][0] = (cx * cnt + x) / ncnt
        self.centers[best_id][1] = (cy * cnt + y) / ncnt
        self.centers[best_id][2] = ncnt
        nbx, nby = self._bucket_id(self.centers[best_id][0], self.centers[best_id][1])
        obx, oby = self.center_bucket[best_id]
        if (nbx, nby) != (obx, oby):
            # Keep spatial index consistent as centroid drifts.
            if best_id in self.buckets[(obx, oby)]:
                self.buckets[(obx, oby)].remove(best_id)
            self.buckets[(nbx, nby)].append(best_id)
            self.center_bucket[best_id] = (nbx, nby)
        return best_id

    def center(self, cid: int) -> LonLat:
        c = self.centers[cid]
        lon, lat = self.inv.transform(float(c[0]), float(c[1]))
        return (float(lon), float(lat))


def stitch_lines(lines: Sequence[Line], snap_tol_m: float, epsg: int = 32610) -> List[Line]:
    """Snap nearby endpoints in meters, then merge chains sharing endpoint IDs."""
    if not lines:
        return []

    clusterer = EndpointClusterer(snap_tol_m, epsg=epsg)
    segs = []
    for line in lines:
        if len(line) < 2:
            continue
        s_id = clusterer.assign(line[0][0], line[0][1])
        e_id = clusterer.assign(line[-1][0], line[-1][1])
        s_xy = clusterer.center(s_id)
        e_xy = clusterer.center(e_id)
        coords = list(line)
        coords[0] = s_xy
        coords[-1] = e_xy
        segs.append({"coords": coords, "start": s_id, "end": e_id, "used": False})

    starts: Dict[int, set] = defaultdict(set)
    ends: Dict[int, set] = defaultdict(set)
    for i, seg in enumerate(segs):
        starts[seg["start"]].add(i)
        ends[seg["end"]].add(i)

    def remove_idx(i: int) -> None:
        starts[segs[i]["start"]].discard(i)
        ends[segs[i]["end"]].discard(i)
        segs[i]["used"] = True

    def pick_next(cluster_id: int) -> Tuple[int, bool] | None:
        # bool indicates if segment should be used forward (start matches)
        for i in list(starts.get(cluster_id, ())):
            if not segs[i]["used"]:
                return (i, True)
        for i in list(ends.get(cluster_id, ())):
            if not segs[i]["used"]:
                return (i, False)
        return None

    stitched: List[Line] = []
    for i, seg in enumerate(segs):
        if seg["used"]:
            continue

        chain = list(seg["coords"])
        start_c = seg["start"]
        end_c = seg["end"]
        remove_idx(i)

        # Extend forward from chain end
        while True:
            nxt = pick_next(end_c)
            if not nxt:
                break
            j, forward = nxt
            if j == i:
                break
            jseg = segs[j]
            if forward:
                chain.extend(jseg["coords"][1:])
                end_c = jseg["end"]
            else:
                rev = list(reversed(jseg["coords"]))
                chain.extend(rev[1:])
                end_c = jseg["start"]
            remove_idx(j)

        # Extend backward from chain start
        while True:
            nxt = pick_next(start_c)
            if not nxt:
                break
            j, forward = nxt
            if j == i:
                break
            jseg = segs[j]
            if not forward:
                # j ends at chain start: prepend j (without duplicate endpoint)
                chain = jseg["coords"][:-1] + chain
                start_c = jseg["start"]
            else:
                # j starts at chain start: prepend reversed j
                rev = list(reversed(jseg["coords"]))
                chain = rev[:-1] + chain
                start_c = jseg["end"]
            remove_idx(j)

        stitched.append(chain)

    return stitched


def main() -> int:
    parser = argparse.ArgumentParser(description="Stitch nearby shoreline segment endpoints.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/shoreline_puget.geojson"),
        help="Input GeoJSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/shoreline_puget_stitched.geojson"),
        help="Output stitched GeoJSON",
    )
    parser.add_argument(
        "--snap-tol-m",
        type=float,
        default=8.0,
        help="Endpoint snap tolerance in meters (default: 8.0)",
    )
    parser.add_argument(
        "--epsg",
        type=int,
        default=32610,
        help="UTM EPSG code for metric operations (default: 32610)",
    )
    parser.add_argument(
        "--source-name",
        default="shoreline_stitched",
        help="Output source name",
    )
    args = parser.parse_args()

    obj = load_geojson(args.input)
    lines = list(iter_lines_from_geojson(obj))
    print(f"Input lines: {len(lines)}")
    print(f"Input stats: {compute_basic_stats(lines)}")

    stitched = stitch_lines(lines, snap_tol_m=args.snap_tol_m, epsg=args.epsg)
    print(f"Stitched lines: {len(stitched)}")
    print(f"Stitched stats: {compute_basic_stats(stitched)}")

    out = lines_to_feature_collection(stitched, source_name=args.source_name)
    save_geojson(args.output, out)
    print(f"Wrote {args.output}")
    print(f"File size: {args.output.stat().st_size / (1024 * 1024):.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
