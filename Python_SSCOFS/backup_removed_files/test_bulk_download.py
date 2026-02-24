#!/usr/bin/env python3
"""
test_bulk_download.py
---------------------

Test script to benchmark parallel downloading of SSCOFS forecast files.
This script will download multiple forecast hours and compare sequential
vs parallel download speeds.
"""

import time
from datetime import datetime, timezone
from latest_cycle import find_latest_cycle, build_url
from sscofs_cache import bulk_download_forecasts, load_sscofs_data, list_cache, clear_cache


def test_parallel_download(forecast_hours, max_workers=5, use_cache=False):
    """
    Test parallel download speed.
    
    Parameters:
    -----------
    forecast_hours : list of int
        List of forecast hour indices to download (e.g., [0, 6, 12, 18, 24])
    max_workers : int
        Number of parallel workers
    use_cache : bool
        If True, may use cached files (not a fair speed test)
        If False, always download fresh
    """
    print("=" * 70)
    print("PARALLEL DOWNLOAD TEST")
    print("=" * 70)
    
    # Find latest cycle
    print("\nFinding latest available cycle...")
    run_date, cycle, keys = find_latest_cycle()
    print(f"  Run date: {run_date}")
    print(f"  Cycle: {cycle:02d}z")
    print(f"  Forecast hours to download: {forecast_hours}")
    print(f"  Workers: {max_workers}")
    print(f"  Use cache: {use_cache}")
    
    # Build run_info list
    run_infos = []
    for fh in forecast_hours:
        run_infos.append({
            'run_date_utc': f'{run_date:%Y-%m-%d}',
            'cycle_utc': f'{cycle:02d}z',
            'forecast_hour_index': fh,
            'url': build_url(run_date, cycle, True, fh)
        })
    
    # Time the parallel download
    print(f"\n{'─' * 70}")
    print("Starting parallel download...")
    print(f"{'─' * 70}")
    start_time = time.time()
    
    cache_paths = bulk_download_forecasts(
        run_infos, 
        use_cache=use_cache, 
        max_workers=max_workers,
        verbose=True
    )
    
    elapsed = time.time() - start_time
    
    # Calculate total size
    total_size_mb = sum(
        p.stat().st_size / (1024 * 1024) 
        for p in cache_paths if p is not None
    )
    
    # Summary
    print(f"\n{'═' * 70}")
    print("RESULTS")
    print(f"{'═' * 70}")
    print(f"  Total files: {len(forecast_hours)}")
    print(f"  Successful downloads: {sum(1 for p in cache_paths if p is not None)}")
    print(f"  Failed downloads: {sum(1 for p in cache_paths if p is None)}")
    print(f"  Total size: {total_size_mb:.1f} MB")
    print(f"  Total time: {elapsed:.1f} seconds")
    print(f"  Average time per file: {elapsed/len(forecast_hours):.1f} seconds")
    if not use_cache:
        print(f"  Download speed: {total_size_mb/elapsed:.1f} MB/s")
    print(f"  Workers used: {max_workers}")
    
    return elapsed, cache_paths


def test_sequential_download(forecast_hours, use_cache=False):
    """
    Test sequential (single-threaded) download speed for comparison.
    
    Parameters:
    -----------
    forecast_hours : list of int
        List of forecast hour indices to download
    use_cache : bool
        If True, may use cached files (not a fair speed test)
    """
    print("\n" + "=" * 70)
    print("SEQUENTIAL DOWNLOAD TEST (for comparison)")
    print("=" * 70)
    
    # Find latest cycle
    print("\nFinding latest available cycle...")
    run_date, cycle, keys = find_latest_cycle()
    print(f"  Run date: {run_date}")
    print(f"  Cycle: {cycle:02d}z")
    print(f"  Forecast hours to download: {forecast_hours}")
    print(f"  Use cache: {use_cache}")
    
    # Build run_info list
    run_infos = []
    for fh in forecast_hours:
        run_infos.append({
            'run_date_utc': f'{run_date:%Y-%m-%d}',
            'cycle_utc': f'{cycle:02d}z',
            'forecast_hour_index': fh,
            'url': build_url(run_date, cycle, True, fh)
        })
    
    # Time the sequential download
    print(f"\n{'─' * 70}")
    print("Starting sequential download...")
    print(f"{'─' * 70}")
    start_time = time.time()
    
    from sscofs_cache import download_to_cache, DEFAULT_CACHE_DIR
    
    cache_paths = []
    for idx, info in enumerate(run_infos):
        fh = info['forecast_hour_index']
        print(f"  [{idx+1}/{len(run_infos)}] Downloading forecast hour {fh:03d}...")
        try:
            cache_path = download_to_cache(info, verbose=False)
            cache_paths.append(cache_path)
            size_mb = cache_path.stat().st_size / (1024 * 1024)
            print(f"    ✓ Success - {size_mb:.1f} MB")
        except Exception as e:
            print(f"    ✗ Error: {e}")
            cache_paths.append(None)
    
    elapsed = time.time() - start_time
    
    # Calculate total size
    total_size_mb = sum(
        p.stat().st_size / (1024 * 1024) 
        for p in cache_paths if p is not None
    )
    
    # Summary
    print(f"\n{'═' * 70}")
    print("RESULTS")
    print(f"{'═' * 70}")
    print(f"  Total files: {len(forecast_hours)}")
    print(f"  Successful downloads: {sum(1 for p in cache_paths if p is not None)}")
    print(f"  Failed downloads: {sum(1 for p in cache_paths if p is None)}")
    print(f"  Total size: {total_size_mb:.1f} MB")
    print(f"  Total time: {elapsed:.1f} seconds")
    print(f"  Average time per file: {elapsed/len(forecast_hours):.1f} seconds")
    if not use_cache:
        print(f"  Download speed: {total_size_mb/elapsed:.1f} MB/s")
    
    return elapsed, cache_paths


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test bulk download speed for SSCOFS forecast files"
    )
    parser.add_argument(
        '--forecast-hours',
        type=int,
        nargs='+',
        default=[0, 6, 12, 18, 24],
        help='Forecast hours to download (default: 0 6 12 18 24)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=5,
        help='Number of parallel workers (default: 5)'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help="Don't use cached data, always download fresh (for accurate speed test)"
    )
    parser.add_argument(
        '--list-cache',
        action='store_true',
        help='List cached files and exit'
    )
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear all cached files and exit'
    )
    parser.add_argument(
        '--compare-sequential',
        action='store_true',
        help='Also run sequential download for comparison (WARNING: takes much longer!)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test with just 3 files (forecast hours 0, 12, 24)'
    )
    
    args = parser.parse_args()
    
    # Handle cache management commands first
    if args.list_cache:
        list_cache()
        return
    
    if args.clear_cache:
        clear_cache()
        return
    
    # Override forecast hours if quick test
    if args.quick:
        forecast_hours = [0, 12, 24]
    else:
        forecast_hours = args.forecast_hours
    
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "SSCOFS BULK DOWNLOAD SPEED TEST" + " " * 21 + "║")
    print("╚" + "═" * 68 + "╝")
    print(f"\nTest started at: {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S UTC}")
    
    # Run parallel test
    parallel_time, parallel_paths = test_parallel_download(
        forecast_hours, 
        max_workers=args.workers,
        use_cache=not args.no_cache
    )
    
    # Optionally run sequential test
    sequential_time = None
    if args.compare_sequential:
        print("\n" + "⚠" * 35)
        print("WARNING: Sequential test will take much longer!")
        print("Press Ctrl+C to skip...")
        print("⚠" * 35)
        time.sleep(3)
        
        sequential_time, sequential_paths = test_sequential_download(
            forecast_hours,
            use_cache=not args.no_cache
        )
    
    # Final comparison
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 25 + "FINAL COMPARISON" + " " * 27 + "║")
    print("╚" + "═" * 68 + "╝")
    print(f"\n  Files downloaded: {len(forecast_hours)}")
    print(f"  Parallel time ({args.workers} workers): {parallel_time:.1f} seconds")
    
    if sequential_time:
        print(f"  Sequential time (1 worker): {sequential_time:.1f} seconds")
        speedup = sequential_time / parallel_time
        print(f"\n  ⚡ SPEEDUP: {speedup:.1f}x faster with parallel download!")
        print(f"  Time saved: {sequential_time - parallel_time:.1f} seconds")
    
    if args.no_cache:
        print("\n  💡 Tip: Downloaded files are now cached.")
        print("     Re-running without --no-cache will be nearly instant!")
    
    print("\n" + "═" * 70)


if __name__ == "__main__":
    main()

