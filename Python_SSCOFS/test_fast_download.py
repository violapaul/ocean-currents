"""
test_fast_download.py
---------------------

Benchmark and correctness test: serial full-file download via sscofs_cache
vs parallel byte-range reads via s3fs for loading SSCOFS current frames.

Usage:
    conda run -n anaconda python test_fast_download.py
"""

import sys
import time as _time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import s3fs
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from fetch_sscofs import build_sscofs_url
from sscofs_cache import load_sscofs_data, get_cached_filename, DEFAULT_CACHE_DIR


# ── Config ────────────────────────────────────────────────────────────────────
RUN_DATE = "2026-03-08"
CYCLE = 3
BASE_FH = 3
TEST_FH_LIST = [2, 3, 4, 5, 6]          # 5 frames for a quick test
MAX_WORKERS = 5

# ── Shared s3fs instance ─────────────────────────────────────────────────────
_fs = None
def _get_fs():
    global _fs
    if _fs is None:
        _fs = s3fs.S3FileSystem(
            anon=True,
            default_block_size=8 * 1024 * 1024,
            default_fill_cache=True,
        )
    return _fs


def _s3_key_for(run_date: str, cycle: int, fh: int) -> str:
    url = build_sscofs_url(run_date, cycle, fh)
    return url.replace(
        "https://noaa-nos-ofs-pds.s3.amazonaws.com/", "noaa-nos-ofs-pds/"
    )


# ── Method A: serial full-file download (current code path) ──────────────────

def download_serial_full(run_date, cycle, fh_list):
    """Download full ~200 MB files one at a time, return (lonc, latc, u_frames, v_frames)."""
    u_frames, v_frames = [], []
    lonc = latc = None
    for fh in fh_list:
        info = {
            "run_date_utc": run_date,
            "cycle_utc": f"{cycle:02d}z",
            "forecast_hour_index": fh,
            "url": build_sscofs_url(run_date, cycle, fh),
        }
        ds = load_sscofs_data(info, use_cache=True, verbose=False)
        if lonc is None:
            lonc = ds["lonc"].values
            latc = ds["latc"].values
        u = ds["u"].isel(time=0, siglay=0).values
        v = ds["v"].isel(time=0, siglay=0).values
        u_frames.append(np.nan_to_num(u, nan=0.0))
        v_frames.append(np.nan_to_num(v, nan=0.0))
        ds.close()
    return lonc, latc, u_frames, v_frames


# ── Method B: parallel byte-range reads ───────────────────────────────────────

def _load_one_frame_byterange(run_date, cycle, fh):
    """Fetch u, v at siglay=0 via byte-range read (~3 MB)."""
    fs = _get_fs()
    key = _s3_key_for(run_date, cycle, fh)
    with fs.open(key, "rb") as fobj:
        ds = xr.open_dataset(fobj, engine="h5netcdf",
                             drop_variables=["siglay", "siglev"])
        u = ds["u"].isel(time=0, siglay=0).values
        v = ds["v"].isel(time=0, siglay=0).values
        ds.close()
    return fh, np.nan_to_num(u, nan=0.0), np.nan_to_num(v, nan=0.0)


def _load_geometry_byterange(run_date, cycle, fh):
    fs = _get_fs()
    key = _s3_key_for(run_date, cycle, fh)
    with fs.open(key, "rb") as fobj:
        ds = xr.open_dataset(fobj, engine="h5netcdf",
                             drop_variables=["siglay", "siglev"])
        lonc = ds["lonc"].values
        latc = ds["latc"].values
        ds.close()
    return lonc, latc


def download_parallel_byterange(run_date, cycle, fh_list, max_workers=MAX_WORKERS):
    """Parallel byte-range reads, return (lonc, latc, u_frames, v_frames)."""
    lonc, latc = _load_geometry_byterange(run_date, cycle, fh_list[0])

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_load_one_frame_byterange, run_date, cycle, fh): fh
            for fh in fh_list
        }
        for fut in as_completed(futures):
            fh, u, v = fut.result()
            results[fh] = (u, v)

    u_frames = [results[fh][0] for fh in fh_list]
    v_frames = [results[fh][1] for fh in fh_list]
    return lonc, latc, u_frames, v_frames


# ── Method C: parallel full-file download to cache, then local read ───────────

def download_parallel_cached(run_date, cycle, fh_list, max_workers=MAX_WORKERS):
    """Parallel full-file download via bulk_download_forecasts, then local read."""
    from sscofs_cache import bulk_download_forecasts, load_sscofs_data

    run_infos = []
    for fh in fh_list:
        run_infos.append({
            'run_date_utc': run_date,
            'cycle_utc': f"{cycle:02d}z",
            'forecast_hour_index': fh,
            'url': build_sscofs_url(run_date, cycle, fh),
        })

    bulk_download_forecasts(run_infos, use_cache=False,
                            max_workers=max_workers, verbose=True)

    u_frames, v_frames = [], []
    lonc = latc = None
    for info in run_infos:
        ds = load_sscofs_data(info, use_cache=True, verbose=False)
        if lonc is None:
            lonc = ds["lonc"].values
            latc = ds["latc"].values
        u = ds["u"].isel(time=0, siglay=0).values
        v = ds["v"].isel(time=0, siglay=0).values
        u_frames.append(np.nan_to_num(u, nan=0.0))
        v_frames.append(np.nan_to_num(v, nan=0.0))
        ds.close()
    return lonc, latc, u_frames, v_frames


# ── Main benchmark ────────────────────────────────────────────────────────────

def main():
    from sscofs_cache import clear_cache

    print("=" * 60)
    print("SSCOFS Download Benchmark")
    print(f"  Run: {RUN_DATE} {CYCLE:02d}z")
    print(f"  Frames: {TEST_FH_LIST} ({len(TEST_FH_LIST)} files)")
    print("=" * 60)

    # --- Method A: serial full-file download (cold cache) ---
    clear_cache()
    print("\n[A] Serial full-file download + load (cold cache)...")
    t0 = _time.monotonic()
    lonc_a, latc_a, u_a, v_a = download_serial_full(RUN_DATE, CYCLE, TEST_FH_LIST)
    t_serial_cold = _time.monotonic() - t0
    print(f"    Time: {t_serial_cold:.2f}s")

    # --- Method A2: serial from warm cache ---
    print("\n[A2] Serial load from warm cache...")
    t0 = _time.monotonic()
    lonc_a2, _, u_a2, v_a2 = download_serial_full(RUN_DATE, CYCLE, TEST_FH_LIST)
    t_serial_warm = _time.monotonic() - t0
    print(f"    Time: {t_serial_warm:.2f}s")

    # --- Method B: parallel byte-range reads (always from S3) ---
    print(f"\n[B] Parallel byte-range reads ({MAX_WORKERS} workers, from S3)...")
    t0 = _time.monotonic()
    lonc_b, latc_b, u_b, v_b = download_parallel_byterange(
        RUN_DATE, CYCLE, TEST_FH_LIST, max_workers=MAX_WORKERS
    )
    t_byterange = _time.monotonic() - t0
    print(f"    Time: {t_byterange:.2f}s")

    # --- Method C: parallel full-file download + local read ---
    clear_cache()
    print(f"\n[C] Parallel full-file download + local read ({MAX_WORKERS} workers)...")
    t0 = _time.monotonic()
    lonc_c, latc_c, u_c, v_c = download_parallel_cached(
        RUN_DATE, CYCLE, TEST_FH_LIST, max_workers=MAX_WORKERS
    )
    t_parallel_cached = _time.monotonic() - t0
    print(f"    Time: {t_parallel_cached:.2f}s")

    # --- Correctness check ---
    print("\n[D] Correctness check...")
    lonc_a_fixed = np.where(lonc_a > 180, lonc_a - 360, lonc_a)
    lonc_b_fixed = np.where(lonc_b > 180, lonc_b - 360, lonc_b)
    lonc_c_fixed = np.where(lonc_c > 180, lonc_c - 360, lonc_c)
    assert np.allclose(lonc_a_fixed, lonc_b_fixed, atol=1e-6), "lonc A/B mismatch!"
    assert np.allclose(lonc_a_fixed, lonc_c_fixed, atol=1e-6), "lonc A/C mismatch!"
    for i, fh in enumerate(TEST_FH_LIST):
        assert np.allclose(u_a[i], u_b[i], atol=1e-6), f"u A/B mismatch at fh={fh}"
        assert np.allclose(v_a[i], v_b[i], atol=1e-6), f"v A/B mismatch at fh={fh}"
        assert np.allclose(u_a[i], u_c[i], atol=1e-6), f"u A/C mismatch at fh={fh}"
        assert np.allclose(v_a[i], v_c[i], atol=1e-6), f"v A/C mismatch at fh={fh}"
    print("    All methods match exactly.")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print(f"[A]  Serial download (cold):        {t_serial_cold:6.1f}s")
    print(f"[A2] Serial load (warm cache):      {t_serial_warm:6.1f}s")
    print(f"[B]  Parallel byte-range (S3):      {t_byterange:6.1f}s")
    print(f"[C]  Parallel download + cache read: {t_parallel_cached:6.1f}s  ← best cold")
    print(f"{'=' * 60}")
    print(f"\nCold speedup (C vs A): {t_serial_cold / t_parallel_cached:.1f}x")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
