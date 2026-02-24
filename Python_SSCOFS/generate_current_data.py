"""
generate_current_data.py
------------------------
Fetches SSCOFS model data from NOAA S3 and exports the native unstructured
mesh (cropped to a region around Seattle) as compact binary files suitable
for client-side rendering in the browser.

Output files:
  geometry.bin  - Float32 array of [lon0, lat0, lon1, lat1, ...] for element centers
  manifest.json - Metadata (model run, element count, bounds, available hours)
  f000.bin      - Float16 array of [u0, v0, u1, v1, ...] for forecast hour 0
  f001.bin      - ... etc through f072.bin

Usage:
  python generate_current_data.py                    # latest cycle, default output
  python generate_current_data.py --output ./data    # custom output directory
  python generate_current_data.py --hours 0-24       # only first 24 hours
  python generate_current_data.py --upload            # upload to S3 after generation
"""

import argparse
import gzip
import json
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import s3fs
import xarray as xr

# Add parent directory for local imports
sys.path.insert(0, str(Path(__file__).parent))
from latest_cycle import find_latest_cycle
from fetch_sscofs import build_sscofs_url
from sscofs_cache import load_sscofs_data, bulk_download_forecasts


# Seattle coordinates and default region
SEATTLE_LAT = 47.6062
SEATTLE_LON = -122.3321
DEFAULT_RADIUS_MI = 100
MI_TO_KM = 1.60934
KM_PER_DEG_LAT = 111.0


def compute_region_mask(lonc, latc, center_lat, center_lon, radius_mi):
    """Create a boolean mask for elements within radius_mi of center point."""
    radius_km = radius_mi * MI_TO_KM
    r_lat = radius_km / KM_PER_DEG_LAT
    km_per_deg_lon = KM_PER_DEG_LAT * np.cos(np.radians(center_lat))
    r_lon = radius_km / km_per_deg_lon

    mask = ((np.abs(latc - center_lat) < r_lat) &
            (np.abs(lonc - center_lon) < r_lon))
    return mask


def fix_longitude(lonc):
    """Convert 0-360 longitude to -180/180 if needed."""
    if lonc.max() > 180:
        return np.where(lonc > 180, lonc - 360, lonc)
    return lonc


# Shared s3fs filesystem with optimized settings for byte-range reads
_global_fs = None

def _get_s3fs():
    """Get a shared s3fs filesystem instance with optimized settings."""
    global _global_fs
    if _global_fs is None:
        _global_fs = s3fs.S3FileSystem(
            anon=True,
            default_block_size=8 * 1024 * 1024,  # 8MB read blocks
            default_fill_cache=True,
            config_kwargs={'max_pool_connections': 50},
        )
    return _global_fs


def load_geometry_direct(run_date, cycle, forecast_hour=0):
    """
    Load only lonc/latc from S3 using byte-range reads (no full file download).
    
    Returns (lonc, latc) numpy arrays.
    """
    url = build_sscofs_url(run_date.isoformat(), cycle, forecast_hour)
    s3_key = url.replace("https://noaa-nos-ofs-pds.s3.amazonaws.com/", "noaa-nos-ofs-pds/")
    
    fs = _get_s3fs()
    with fs.open(s3_key, 'rb', block_size=8*1024*1024) as f:
        ds = xr.open_dataset(f, engine='h5netcdf')
        lonc = ds["lonc"].values
        latc = ds["latc"].values
        ds.close()
    
    return lonc, latc


def load_velocity_direct(run_date, cycle, forecast_hour):
    """
    Load only u,v at surface (siglay=0) from S3 using byte-range reads.
    
    This fetches only ~3.4MB instead of the full ~200MB file.
    Returns (u, v) numpy arrays at siglay=0, time=0.
    """
    url = build_sscofs_url(run_date.isoformat(), cycle, forecast_hour)
    s3_key = url.replace("https://noaa-nos-ofs-pds.s3.amazonaws.com/", "noaa-nos-ofs-pds/")
    
    fs = _get_s3fs()
    with fs.open(s3_key, 'rb', block_size=8*1024*1024) as f:
        ds = xr.open_dataset(f, engine='h5netcdf')
        u = ds["u"].isel(time=0, siglay=0).values
        v = ds["v"].isel(time=0, siglay=0).values
        ds.close()
    
    return u, v


def export_velocity_from_arrays(u, v, mask, forecast_hour, output_dir):
    """Export masked surface u,v (from arrays) as a gzipped Float16 binary file."""
    u_masked = u[mask]
    v_masked = v[mask]

    u_masked = np.nan_to_num(u_masked, nan=0.0).astype(np.float16)
    v_masked = np.nan_to_num(v_masked, nan=0.0).astype(np.float16)

    interleaved = np.empty(len(u_masked) * 2, dtype=np.float16)
    interleaved[0::2] = u_masked
    interleaved[1::2] = v_masked

    fname = f"f{forecast_hour:03d}.bin"
    out_path = output_dir / fname
    with gzip.open(out_path, "wb") as f:
        f.write(interleaved.tobytes())

    return out_path.stat().st_size


def process_hour_worker(args):
    """
    Worker function for parallel hour processing.
    Returns (hour, gz_size) on success or (hour, None) on failure.
    """
    run_date, cycle, hour, mask, output_dir = args
    try:
        u, v = load_velocity_direct(run_date, cycle, hour)
        gz_size = export_velocity_from_arrays(u, v, mask, hour, output_dir)
        return (hour, gz_size)
    except Exception as e:
        return (hour, None, str(e))


def export_geometry(lonc, latc, mask, output_dir):
    """Export masked element coordinates as a gzipped Float32 binary file."""
    lons = lonc[mask].astype(np.float32)
    lats = latc[mask].astype(np.float32)
    interleaved = np.empty(len(lons) * 2, dtype=np.float32)
    interleaved[0::2] = lons
    interleaved[1::2] = lats

    out_path = output_dir / "geometry.bin"
    with gzip.open(out_path, "wb") as f:
        f.write(interleaved.tobytes())

    raw_size = interleaved.nbytes
    gz_size = out_path.stat().st_size
    print(f"  geometry.bin: {len(lons):,} elements, "
          f"{raw_size/1e6:.1f}MB raw -> {gz_size/1e6:.1f}MB gzipped")
    return len(lons)


def export_velocity(ds, mask, forecast_hour, output_dir):
    """Export masked surface u,v as a gzipped Float16 binary file."""
    u = ds["u"].isel(time=0, siglay=0).values[mask]
    v = ds["v"].isel(time=0, siglay=0).values[mask]

    # Replace NaN with 0 for clean binary output
    u = np.nan_to_num(u, nan=0.0).astype(np.float16)
    v = np.nan_to_num(v, nan=0.0).astype(np.float16)

    interleaved = np.empty(len(u) * 2, dtype=np.float16)
    interleaved[0::2] = u
    interleaved[1::2] = v

    fname = f"f{forecast_hour:03d}.bin"
    out_path = output_dir / fname
    with gzip.open(out_path, "wb") as f:
        f.write(interleaved.tobytes())

    gz_size = out_path.stat().st_size
    return gz_size


def write_manifest(output_dir, model_run, num_elements, bounds, hours):
    """Write manifest.json with metadata."""
    manifest = {
        "model_run": model_run,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "num_elements": num_elements,
        "bounds": bounds,
        "forecast_hours": hours,
        "format": {
            "geometry": "gzipped Float32 [lon0,lat0,lon1,lat1,...] little-endian",
            "velocity": "gzipped Float16 [u0,v0,u1,v1,...] little-endian, m/s",
        },
    }
    out_path = output_dir / "manifest.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  manifest.json written")
    return manifest


def generate_fast(output_dir, hour_range=(0, 72), radius_mi=DEFAULT_RADIUS_MI,
                  center_lat=SEATTLE_LAT, center_lon=SEATTLE_LON, upload=False,
                  s3_bucket=None, s3_prefix="ocean-currents", max_workers=10):
    """
    Fast generation pipeline using byte-range S3 reads and parallel processing.
    
    Instead of downloading full 200MB files, this fetches only the variables
    we need (~3.4MB per hour) using s3fs lazy loading.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SSCOFS Current Data Generator (FAST MODE)")
    print(f"  Parallel workers: {max_workers}")
    print(f"  Using byte-range S3 reads (~3MB/file vs 200MB)")
    print("=" * 60)
    
    start_time = time.time()

    # Step 1: Find latest model cycle
    print("\n[1/4] Finding latest model cycle on NOAA S3...")
    run_date, cycle, keys = find_latest_cycle(max_days_back=3)
    model_run_str = f"{run_date.isoformat()}T{cycle:02d}:00:00Z"
    run_tag = f"{run_date:%Y%m%d}_{cycle:02d}z"
    print(f"  Latest cycle: {run_date} {cycle:02d}z")
    print(f"  Model run: {model_run_str}")

    run_dir = output_dir / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    # Step 2: Load geometry using byte-range reads
    print(f"\n[2/4] Loading mesh geometry (byte-range read)...")
    t0 = time.time()
    lonc, latc = load_geometry_direct(run_date, cycle, 0)
    lonc = fix_longitude(lonc)
    print(f"  Loaded in {time.time() - t0:.1f}s")
    
    mask = compute_region_mask(lonc, latc, center_lat, center_lon, radius_mi)
    print(f"  Region: {radius_mi}mi from ({center_lat}, {center_lon})")
    print(f"  Elements in region: {mask.sum():,} / {len(lonc):,}")

    num_elements = export_geometry(lonc, latc, mask, run_dir)

    masked_lon = lonc[mask]
    masked_lat = latc[mask]
    bounds = {
        "lat_min": float(masked_lat.min()),
        "lat_max": float(masked_lat.max()),
        "lon_min": float(masked_lon.min()),
        "lon_max": float(masked_lon.max()),
    }

    # Step 3: Process all hours in parallel
    hours = list(range(hour_range[0], hour_range[1] + 1))
    print(f"\n[3/4] Processing {len(hours)} hours with {max_workers} parallel workers...")
    
    completed_hours = []
    failed_hours = []
    total_vel_size = 0
    
    work_items = [(run_date, cycle, h, mask, run_dir) for h in hours]
    
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_hour_worker, item): item[2] for item in work_items}
        
        for future in as_completed(futures):
            hour = futures[future]
            result = future.result()
            
            if len(result) == 2:
                h, gz_size = result
                completed_hours.append(h)
                total_vel_size += gz_size
                print(f".", end="", flush=True)
            else:
                h, _, err = result
                failed_hours.append((h, err))
                print(f"x", end="", flush=True)
    
    elapsed = time.time() - t0
    print(f"\n  Completed {len(completed_hours)}/{len(hours)} hours in {elapsed:.1f}s")
    print(f"  Speed: {len(completed_hours)/elapsed:.1f} hours/sec")
    print(f"  Total velocity: {total_vel_size/1e6:.1f}MB gzipped")
    
    if failed_hours:
        print(f"  Failed hours: {[h for h, _ in failed_hours]}")

    # Step 4: Write manifest
    print(f"\n[4/4] Writing manifest...")
    manifest = write_manifest(run_dir, model_run_str, num_elements,
                              bounds, sorted(completed_hours))

    latest = {"run": run_tag, "model_run": model_run_str}
    with open(output_dir / "latest.json", "w") as f:
        json.dump(latest, f)
    print(f"  latest.json -> {run_tag}")

    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Done! Output in: {run_dir}")
    print(f"  {num_elements:,} elements, {len(completed_hours)} hours")
    geom_size = (run_dir / "geometry.bin").stat().st_size
    print(f"  Geometry: {geom_size/1e6:.1f}MB")
    print(f"  Velocity: {total_vel_size/1e6:.1f}MB total")
    print(f"  TOTAL TIME: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'=' * 60}")

    if upload:
        upload_to_s3(output_dir, run_tag, s3_bucket, s3_prefix)

    return run_dir


def generate(output_dir, hour_range=(0, 72), radius_mi=DEFAULT_RADIUS_MI,
             center_lat=SEATTLE_LAT, center_lon=SEATTLE_LON, upload=False,
             s3_bucket=None, s3_prefix="ocean-currents"):
    """Legacy generation pipeline using full-file downloads with caching."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SSCOFS Current Data Generator (CACHE MODE)")
    print("=" * 60)

    # Step 1: Find latest model cycle
    print("\n[1/4] Finding latest model cycle on NOAA S3...")
    run_date, cycle, keys = find_latest_cycle(max_days_back=3)
    model_run_str = f"{run_date.isoformat()}T{cycle:02d}:00:00Z"
    run_tag = f"{run_date:%Y%m%d}_{cycle:02d}z"
    print(f"  Latest cycle: {run_date} {cycle:02d}z")
    print(f"  Model run: {model_run_str}")

    # Create run-specific output directory
    run_dir = output_dir / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    # Step 2: Download hour 0 to get geometry + build mask
    print(f"\n[2/4] Downloading F000 for mesh geometry...")
    run_info_0 = {
        "run_date_utc": run_date.isoformat(),
        "cycle_utc": f"{cycle:02d}z",
        "forecast_hour_index": 0,
        "url": build_sscofs_url(run_date.isoformat(), cycle, 0),
    }
    ds0 = load_sscofs_data(run_info_0, use_cache=True, verbose=True)

    lonc = fix_longitude(ds0["lonc"].values)
    latc = ds0["latc"].values
    mask = compute_region_mask(lonc, latc, center_lat, center_lon, radius_mi)
    print(f"  Region: {radius_mi}mi from ({center_lat}, {center_lon})")
    print(f"  Elements in region: {mask.sum():,} / {len(lonc):,}")

    # Export geometry
    num_elements = export_geometry(lonc, latc, mask, run_dir)

    # Compute bounds
    masked_lon = lonc[mask]
    masked_lat = latc[mask]
    bounds = {
        "lat_min": float(masked_lat.min()),
        "lat_max": float(masked_lat.max()),
        "lon_min": float(masked_lon.min()),
        "lon_max": float(masked_lon.max()),
    }

    # Export velocity for hour 0 (already loaded)
    print(f"\n[3/4] Exporting velocity data for F{hour_range[0]:03d}-F{hour_range[1]:03d}...")
    completed_hours = []
    total_vel_size = 0

    gz_size = export_velocity(ds0, mask, 0, run_dir)
    completed_hours.append(0)
    total_vel_size += gz_size
    ds0.close()
    print(f"  F000: {gz_size/1e3:.0f}KB", end="", flush=True)

    # Download and export remaining hours
    for hour in range(max(1, hour_range[0]), hour_range[1] + 1):
        run_info = {
            "run_date_utc": run_date.isoformat(),
            "cycle_utc": f"{cycle:02d}z",
            "forecast_hour_index": hour,
            "url": build_sscofs_url(run_date.isoformat(), cycle, hour),
        }
        try:
            ds = load_sscofs_data(run_info, use_cache=True, verbose=False)
            gz_size = export_velocity(ds, mask, hour, run_dir)
            completed_hours.append(hour)
            total_vel_size += gz_size
            ds.close()

            if hour % 10 == 0 or hour == hour_range[1]:
                print(f"\n  F{hour:03d}: {gz_size/1e3:.0f}KB", end="", flush=True)
            else:
                print(".", end="", flush=True)
        except Exception as e:
            print(f"\n  F{hour:03d}: FAILED ({e})", flush=True)

    print(f"\n  Total velocity: {total_vel_size/1e6:.1f}MB gzipped "
          f"({len(completed_hours)} hours)")

    # Step 4: Write manifest
    print(f"\n[4/4] Writing manifest...")
    manifest = write_manifest(run_dir, model_run_str, num_elements,
                              bounds, sorted(completed_hours))

    # Write latest.json pointer
    latest = {"run": run_tag, "model_run": model_run_str}
    with open(output_dir / "latest.json", "w") as f:
        json.dump(latest, f)
    print(f"  latest.json -> {run_tag}")

    print(f"\n{'=' * 60}")
    print(f"Done! Output in: {run_dir}")
    print(f"  {num_elements:,} elements, {len(completed_hours)} hours")
    geom_size = (run_dir / "geometry.bin").stat().st_size
    print(f"  Geometry: {geom_size/1e6:.1f}MB")
    print(f"  Velocity: {total_vel_size/1e6:.1f}MB total")
    print(f"{'=' * 60}")

    # Optional S3 upload
    if upload:
        upload_to_s3(output_dir, run_tag, s3_bucket, s3_prefix)

    return run_dir


def upload_to_s3(output_dir, run_tag, bucket, prefix):
    """Upload generated files to S3."""
    try:
        import boto3
    except ImportError:
        print("ERROR: boto3 not installed. Run: pip install boto3")
        return

    print(f"\nUploading to s3://{bucket}/{prefix}/...")
    s3 = boto3.client("s3")
    run_dir = output_dir / run_tag

    for path in sorted(run_dir.iterdir()):
        key = f"{prefix}/{run_tag}/{path.name}"
        content_type = "application/json" if path.suffix == ".json" else "application/octet-stream"
        extra = {"ContentType": content_type}
        if path.suffix == ".bin":
            extra["ContentEncoding"] = "gzip"
        s3.upload_file(str(path), bucket, key, ExtraArgs=extra)
        print(f"  -> {key}")

    # Upload latest.json
    latest_path = output_dir / "latest.json"
    s3.upload_file(str(latest_path), bucket, f"{prefix}/latest.json",
                   ExtraArgs={"ContentType": "application/json",
                              "CacheControl": "max-age=300"})
    print(f"  -> {prefix}/latest.json")
    print("Upload complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate binary current data files from SSCOFS model")
    parser.add_argument("--output", type=str, default="./current_data",
                        help="Output directory (default: ./current_data)")
    parser.add_argument("--hours", type=str, default="0-72",
                        help="Forecast hour range, e.g. '0-72' or '0-24' (default: 0-72)")
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS_MI,
                        help=f"Radius in miles from Seattle (default: {DEFAULT_RADIUS_MI})")
    parser.add_argument("--upload", action="store_true",
                        help="Upload to S3 after generation")
    parser.add_argument("--s3-bucket", type=str, default=None,
                        help="S3 bucket name for upload")
    parser.add_argument("--s3-prefix", type=str, default="ocean-currents",
                        help="S3 key prefix (default: ocean-currents)")
    parser.add_argument("--mode", type=str, choices=["fast", "cache"], default="fast",
                        help="Download mode: 'fast' (byte-range S3 reads, parallel) or 'cache' (full file downloads)")
    parser.add_argument("--workers", type=int, default=10,
                        help="Number of parallel workers for fast mode (default: 10)")
    args = parser.parse_args()

    # Parse hour range
    parts = args.hours.split("-")
    hour_range = (int(parts[0]), int(parts[1]))

    if args.mode == "fast":
        generate_fast(
            output_dir=args.output,
            hour_range=hour_range,
            radius_mi=args.radius,
            upload=args.upload,
            s3_bucket=args.s3_bucket,
            s3_prefix=args.s3_prefix,
            max_workers=args.workers,
        )
    else:
        generate(
            output_dir=args.output,
            hour_range=hour_range,
            radius_mi=args.radius,
            upload=args.upload,
            s3_bucket=args.s3_bucket,
            s3_prefix=args.s3_prefix,
        )


if __name__ == "__main__":
    main()
