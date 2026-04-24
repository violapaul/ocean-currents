"""Race-route publication helpers: GeoJSON, manifest, S3 upload.

Called from run_route.py when --race-mode is active. Reuses the route data
already computed by SectorRouter + save_route_json; produces a web-app-ready
GeoJSON sidecar and a tiny manifest for cache-busting in the PWA.
"""
import datetime as _dt
import json
import subprocess
from pathlib import Path

from pyproj import CRS, Transformer as _T


# ---------------------------------------------------------------------------
# GeoJSON writer
# ---------------------------------------------------------------------------

def write_geojson(routes, wps, depart_utc, depart_time_s, task_name,
                  race_cfg, inputs_hash, out_path):
    """Write a MapLibre-ready FeatureCollection for one race.

    Schema (see plan §1.3):
      Top-level properties: slug, name, depart_utc, total_distance_nm,
        total_time_hr, generated_at, inputs_hash.
      LineString per leg (id="leg-N"): coords from simulated_track;
        properties leg, distance_nm, time_min, avg_sog_kt,
        start_elapsed_s, end_elapsed_s.
      Point per input waypoint (id="mark-N"): kind = start|turn|finish,
        order, name.
    """
    ref_lat, ref_lon = wps[0]
    utm_crs = CRS.from_dict({
        "proj": "utm",
        "zone": int((ref_lon + 180) / 6) + 1,
        "north": ref_lat >= 0,
        "ellps": "WGS84",
    })
    inv_tf = _T.from_crs(utm_crs, CRS.from_epsg(4326), always_xy=True)

    features = []
    elapsed_cursor = 0.0
    total_dist_nm = 0.0
    total_time_s = 0.0

    for li, r in enumerate(routes):
        trk   = r.simulated_track or []
        trk_t = r.simulated_track_times or []

        coords = []
        for (x, y) in trk:
            lon, lat = inv_tf.transform(float(x), float(y))
            coords.append([round(lon, 6), round(lat, 6)])

        leg_time_s = float(r.total_time_s)
        leg_dist_nm = r.total_distance_m / 1852.0
        start_elapsed = elapsed_cursor
        end_elapsed = elapsed_cursor + leg_time_s

        features.append({
            "type": "Feature",
            "id": f"leg-{li + 1}",
            "geometry": {
                "type": "LineString",
                "coordinates": coords,
            },
            "properties": {
                "kind": "leg",
                "leg": li + 1,
                "distance_nm": round(leg_dist_nm, 3),
                "time_min": round(leg_time_s / 60.0, 1),
                "avg_sog_kt": round(float(r.avg_sog_knots), 3),
                "start_elapsed_s": round(start_elapsed, 1),
                "end_elapsed_s": round(end_elapsed, 1),
                "from": [round(wps[li][0], 6), round(wps[li][1], 6)],
                "to":   [round(wps[li + 1][0], 6), round(wps[li + 1][1], 6)],
            },
        })

        elapsed_cursor = end_elapsed
        total_dist_nm += leg_dist_nm
        total_time_s  += leg_time_s

    # One Point feature per input waypoint (start, turn(s), finish).
    n_wps = len(wps)
    for i, (lat, lon) in enumerate(wps):
        if i == 0:
            kind = "start"
        elif i == n_wps - 1:
            kind = "finish"
        else:
            kind = "turn"
        features.append({
            "type": "Feature",
            "id": f"mark-{i + 1}",
            "geometry": {
                "type": "Point",
                "coordinates": [round(float(lon), 6), round(float(lat), 6)],
            },
            "properties": {
                "kind": "mark",
                "mark_kind": kind,
                "order": i + 1,
                "name": _mark_label(kind, i + 1),
            },
        })

    fc = {
        "type": "FeatureCollection",
        "properties": {
            "slug": race_cfg.get("slug"),
            "name": task_name,
            "depart_utc": depart_utc.isoformat(),
            "total_distance_nm": round(total_dist_nm, 3),
            "total_time_hr": round(total_time_s / 3600.0, 3),
            "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "inputs_hash": inputs_hash,
        },
        "features": features,
    }

    out_path = Path(out_path)
    with open(out_path, "w") as f:
        json.dump(fc, f, separators=(",", ":"))
    print(f"GeoJSON saved     → {out_path.name}  "
          f"({len(features)} features, "
          f"{out_path.stat().st_size / 1024:.1f} KB)")
    return out_path


def _mark_label(kind, order):
    if kind == "start":
        return "Start"
    if kind == "finish":
        return "Finish"
    return f"Mark {order - 1}"


# ---------------------------------------------------------------------------
# Manifest writer
# ---------------------------------------------------------------------------

def write_manifest(race_cfg, depart_utc, total_distance_nm, total_time_hr,
                   inputs_hash, out_path,
                   route_json_name="route.json",
                   route_geojson_name="route.geojson"):
    """Small JSON blob for cache-busting in the PWA."""
    manifest = {
        "slug": race_cfg.get("slug"),
        "event_start_utc": race_cfg.get("event_start_utc"),
        "depart_utc": depart_utc.isoformat(),
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "inputs_hash": inputs_hash,
        "total_distance_nm": round(total_distance_nm, 3),
        "total_time_hr": round(total_time_hr, 3),
        "route_json": route_json_name,
        "route_geojson": route_geojson_name,
        "version": 1,
    }
    out_path = Path(out_path)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved    → {out_path.name}")
    return out_path


# ---------------------------------------------------------------------------
# S3 upload
# ---------------------------------------------------------------------------

# Matches the existing convention in generate_current_data.upload_to_s3:
# bucket is assumed to have a bucket-level public-read policy; no per-object
# ACL is needed.

_CONTENT_TYPES = {
    ".json":    "application/json",
    ".geojson": "application/geo+json",
}


def upload_files_to_s3(bucket, prefix, files):
    """Upload a list of Path objects to s3://{bucket}/{prefix}/{name}.

    Small objects (JSON / GeoJSON) get short cache TTLs so the PWA picks up
    fresh precomputes quickly once the manifest updates.
    """
    try:
        import boto3
    except ImportError:
        print("ERROR: boto3 not installed. Run: pip install boto3")
        return

    print(f"\nUploading to s3://{bucket}/{prefix}/ ...")
    s3 = boto3.client("s3")
    prefix = prefix.strip("/")

    for path in files:
        path = Path(path)
        if not path.exists():
            print(f"  SKIP (missing): {path}")
            continue
        key = f"{prefix}/{path.name}"
        content_type = _CONTENT_TYPES.get(path.suffix, "application/octet-stream")
        extra = {
            "ContentType": content_type,
            # Short TTL so PWA can see a new precompute within ~5 min.
            "CacheControl": "max-age=300",
        }
        s3.upload_file(str(path), bucket, key, ExtraArgs=extra)
        print(f"  -> s3://{bucket}/{key}")

    print("Upload complete.")


# ---------------------------------------------------------------------------
# Inputs hash (best-effort)
# ---------------------------------------------------------------------------

def compute_inputs_hash(cycle_id=None, ecmwf_init=None):
    """Fingerprint of the inputs used to compute this route.

    Callers pass whatever they know; missing pieces are omitted. The git SHA
    of the code is appended when we can get it cheaply.
    """
    parts = []
    if cycle_id:
        parts.append(f"sscofs={cycle_id}")
    if ecmwf_init:
        parts.append(f"ecmwf={ecmwf_init}")
    git_sha = _git_short_sha()
    if git_sha:
        parts.append(f"git={git_sha}")
    return ";".join(parts) if parts else "unknown"


def _git_short_sha():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=2,
            cwd=Path(__file__).parent,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None
