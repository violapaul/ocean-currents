#!/usr/bin/env python3
"""
Fetch high-detail shoreline polylines from WA Ecology Coastal Atlas (Layer 13).

Outputs a GeoJSON FeatureCollection with raw line features.
Use simplify_coastline.py to convert this into a viewer-ready compact file.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlencode
from urllib.request import urlopen


WA_ECOLOGY_QUERY_URL = (
    "https://gis.ecology.wa.gov/serverext/rest/services/GIS/CoastalAtlas/MapServer/13/query"
)


def fetch_json(url: str, timeout_s: int = 120) -> Dict:
    with urlopen(url, timeout=timeout_s) as response:
        return json.load(response)


def get_object_ids(bbox: str) -> List[int]:
    params = {
        "where": "1=1",
        "geometry": bbox,
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "returnIdsOnly": "true",
        "f": "pjson",
    }
    payload = fetch_json(f"{WA_ECOLOGY_QUERY_URL}?{urlencode(params)}")
    return sorted(payload.get("objectIds", []))


def fetch_features_chunk(object_ids: List[int]) -> List[Dict]:
    params = {
        "objectIds": ",".join(str(x) for x in object_ids),
        "outFields": "OBJECTID,Shoretype,DataSource",
        "outSR": "4326",
        "f": "geojson",
    }
    payload = fetch_json(f"{WA_ECOLOGY_QUERY_URL}?{urlencode(params)}")
    return payload.get("features", [])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch WA Ecology shoreline features for a bbox.",
    )
    parser.add_argument(
        "--bbox",
        default="-123.5,46.9,-122.0,49.1",
        help="lon_min,lat_min,lon_max,lat_max in WGS84",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="ObjectID chunk size per query (default: 500)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/shoreline_wa_ecology_raw.geojson"),
        help="Output raw GeoJSON path",
    )
    args = parser.parse_args()

    object_ids = get_object_ids(args.bbox)
    if not object_ids:
        print("No object IDs returned for bbox; nothing to write.")
        return 1

    print(f"Found {len(object_ids)} object IDs in bbox {args.bbox}")

    all_features: List[Dict] = []
    total_chunks = (len(object_ids) + args.chunk_size - 1) // args.chunk_size

    for start in range(0, len(object_ids), args.chunk_size):
        chunk = object_ids[start : start + args.chunk_size]
        chunk_idx = start // args.chunk_size + 1
        retries = 0
        while True:
            try:
                feats = fetch_features_chunk(chunk)
                all_features.extend(feats)
                print(f"Chunk {chunk_idx}/{total_chunks}: +{len(feats)} features")
                break
            except Exception as exc:  # noqa: BLE001
                retries += 1
                if retries > 3:
                    raise RuntimeError(f"Chunk {chunk_idx} failed after retries: {exc}") from exc
                wait_s = 1.5 * retries
                print(f"Chunk {chunk_idx} retry {retries} after error: {exc}")
                time.sleep(wait_s)

    # De-duplicate by OBJECTID.
    dedup = {}
    for feature in all_features:
        props = feature.get("properties") or {}
        oid = props.get("OBJECTID")
        if oid is None:
            continue
        dedup[oid] = feature

    output_fc = {
        "type": "FeatureCollection",
        "name": "shoreline_wa_ecology_raw",
        "features": list(dedup.values()),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fp:
        json.dump(output_fc, fp, separators=(",", ":"))

    print(f"Wrote {args.output} with {len(output_fc['features'])} features")
    print(f"File size: {args.output.stat().st_size / (1024 * 1024):.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
