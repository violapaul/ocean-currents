"""AWS Lambda handler for on-demand race route recompute.

Wired to a Function URL (not API Gateway) — handles AWS_PROXY-style events.

Request body (POST, application/json):
  {
    "race_slug": "shilshole_double_bluff_return",
    "start": {"lat": 47.85, "lon": -122.48},
    "depart_utc": "2026-04-25T17:30:00Z",   // optional; default = now()
    "next_waypoint_index": 1                  // optional; default = 1
                                              //   (0 = include start; 1 = skip start)
  }

Response (200):
  {
    "slug": "...",
    "reroute": true,
    "generated_at": "...",
    "summary": {
      "remaining_distance_nm": ..., "remaining_time_hr": ..., "legs": ...
    },
    "geojson": { ...FeatureCollection from race_publish.write_geojson... }
  }

Errors return 4xx/5xx with {"error": "..."} body. CORS preflights (OPTIONS)
return 204 with permissive headers so the GitHub-Pages PWA can call us
cross-origin.

Cold start budget: ~30s. We import the heavy router stack at module load
so it's amortized into init_duration; warm invocations skip it.
Per-invocation budget: ~25-45s for a sub-12 nm reroute.
"""

import datetime as dt
import json
import os
import sys
from pathlib import Path

# Lambda's /var/task is read-only. numba's @njit(cache=True) tries to write
# pycache next to the source and aborts with "no locator available" if it
# can't. Point its cache at /tmp BEFORE any of the heavy modules below get
# imported. (setdefault so a future move to a writable layout, e.g. EFS,
# Just Works.) Matplotlib is NOT installed in the Lambda image — all our
# matplotlib imports are inside plot-only function bodies and Lambda runs
# with --no-plots.
os.environ.setdefault("NUMBA_CACHE_DIR",        "/tmp/numba")
os.environ.setdefault("SSCOFS_CACHE_DIR",       "/tmp/sscofs_cache")
os.environ.setdefault("SSCOFS_SURFACE_CACHE_DIR","/tmp/sscofs_surface_cache")
os.environ.setdefault("WIND_CACHE_DIR",         "/tmp/wind_cache")
for _k in ("NUMBA_CACHE_DIR", "SSCOFS_CACHE_DIR",
           "SSCOFS_SURFACE_CACHE_DIR", "WIND_CACHE_DIR"):
    Path(os.environ[_k]).mkdir(parents=True, exist_ok=True)

# On Lambda the Dockerfile lays everything (handler.py + the router
# modules + routes/races/ + j105_new_polars.csv) directly under
# /var/task, not in the Python_SSCOFS subdir we have locally. So HERE
# is also where the imports + race YAMLs live.
HERE = Path(__file__).parent
PYTHON_SSCOFS = HERE
sys.path.insert(0, str(PYTHON_SSCOFS))

# /tmp is the only writable path in the Lambda runtime. Pre-create the
# subdirectory we hand to run_route via --output-dir.
LAMBDA_OUTPUT_DIR = Path(os.environ.get("REROUTE_OUTPUT_DIR", "/tmp/reroute"))
LAMBDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Heavy imports at module load so cold-start eats the cost once. Each of
# these pulls in numpy/scipy/xarray/pyproj/numba/matplotlib transitively.
import yaml as _yaml  # noqa: F401
import sail_routing  # noqa: F401  -- triggers numba JIT lazily on first solve
import run_route as rr
import race_publish  # noqa: F401  -- imported for symmetry; also pulled by rr


RACES_DIR = PYTHON_SSCOFS / "routes" / "races"
POLAR_PATH = PYTHON_SSCOFS / "j105_new_polars.csv"  # bundled by Dockerfile

# CORS headers: permissive because the PWA lives at violapaul.github.io
# while the Function URL is on amazonaws.com.
_CORS = {
    "Access-Control-Allow-Origin":  "*",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
}


def _ok(body):
    return {"statusCode": 200,
            "headers": {"Content-Type": "application/json", **_CORS},
            "body": json.dumps(body, separators=(",", ":"))}


def _err(status, msg):
    print(f"[reroute] ERROR {status}: {msg}")
    return {"statusCode": status,
            "headers": {"Content-Type": "application/json", **_CORS},
            "body": json.dumps({"error": msg})}


def _http_method(event):
    # Function URL events use requestContext.http.method; classic API GW
    # uses httpMethod. Support both.
    rc = event.get("requestContext") or {}
    if isinstance(rc, dict) and "http" in rc:
        return rc["http"].get("method", "POST")
    return event.get("httpMethod", "POST")


def _parse_body(event):
    body = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        import base64
        body = base64.b64decode(body).decode("utf-8")
    elif isinstance(body, bytes):
        body = body.decode("utf-8")
    return json.loads(body)


def handler(event, context):
    method = _http_method(event)
    if method == "OPTIONS":
        return {"statusCode": 204, "headers": _CORS}
    if method != "POST":
        return _err(405, f"method {method} not allowed; POST only")

    try:
        req = _parse_body(event)
    except Exception as e:
        return _err(400, f"bad request body: {e}")

    slug = req.get("race_slug")
    start = req.get("start") or {}
    if not slug:
        return _err(400, "race_slug is required")
    if "lat" not in start or "lon" not in start:
        return _err(400, "start.lat and start.lon are required")
    try:
        slat = float(start["lat"]); slon = float(start["lon"])
    except (TypeError, ValueError):
        return _err(400, "start.lat / start.lon must be numbers")

    yaml_path = RACES_DIR / f"{slug}.yaml"
    if not yaml_path.exists():
        return _err(404, f"unknown race_slug: {slug}")

    next_idx = int(req.get("next_waypoint_index", 1))
    depart_str = req.get("depart_utc")
    if depart_str:
        try:
            d = dt.datetime.fromisoformat(depart_str.replace("Z", "+00:00"))
            if d.tzinfo is None:
                d = d.replace(tzinfo=dt.timezone.utc)
        except ValueError as e:
            return _err(400, f"bad depart_utc: {e}")
        depart_utc = d
    else:
        depart_utc = dt.datetime.now(dt.timezone.utc)

    # Build an effective YAML in /tmp with two adjustments:
    #   1. boat.polar rewritten to the bundled absolute path. The original
    #      uses ../../j105_new_polars.csv which resolves outside the
    #      Lambda root (/var/task/../../) and breaks.
    #   2. waypoints trimmed to drop already-rounded marks when
    #      next_waypoint_index > 1.
    with open(yaml_path) as f:
        doc = _yaml.safe_load(f)
    boat_cfg = doc.setdefault("boat", {})
    boat_cfg["polar"] = str(POLAR_PATH)

    if next_idx > 1:
        wps = doc.get("waypoints", [])
        if next_idx >= len(wps):
            return _err(400, f"next_waypoint_index {next_idx} >= waypoints "
                             f"{len(wps)}")
        doc["waypoints"] = [wps[0]] + wps[next_idx:]

    effective_yaml = LAMBDA_OUTPUT_DIR / f"{slug}_eff.yaml"
    with open(effective_yaml, "w") as f:
        _yaml.safe_dump(doc, f)

    out_dir = LAMBDA_OUTPUT_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    geojson_path = out_dir / f"{Path(effective_yaml).stem}.geojson"
    if geojson_path.exists():
        geojson_path.unlink()  # ensure we read THIS run's output

    saved_argv = sys.argv
    sys.argv = [
        "run_route.py",
        str(effective_yaml),
        "--race-mode",
        "--no-plots",
        "--geojson",
        "--ignore-window",                  # mid-race: gates would skip
        f"--start-latlon={slat},{slon}",
        f"--departure-utc={depart_utc.isoformat()}",
        f"--output-dir={out_dir}",
    ]
    print(f"[reroute] argv = {sys.argv[1:]}")

    try:
        rr.main()
    except SystemExit as e:
        # argparse / sys.exit on validation errors. Treat 0 as success.
        if e.code not in (0, None):
            return _err(500, f"run_route.main() exited with {e.code}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        return _err(500, f"run_route raised: {e!r}")
    finally:
        sys.argv = saved_argv

    if not geojson_path.exists():
        return _err(500, f"run_route did not produce {geojson_path}")

    with open(geojson_path) as f:
        fc = json.load(f)

    # Compose the response. The PWA only needs `geojson`; summary is for
    # debugging and future UI hooks (ETA card, etc.).
    props = fc.get("properties", {})
    return _ok({
        "slug": slug,
        "reroute": True,
        "generated_at": props.get("generated_at"),
        "summary": {
            "remaining_distance_nm": props.get("total_distance_nm"),
            "remaining_time_hr":     props.get("total_time_hr"),
            "legs": sum(1 for f in fc.get("features", [])
                        if f.get("geometry", {}).get("type") == "LineString"),
            "next_waypoint_index": next_idx,
        },
        "geojson": fc,
    })
