"""
sscofs_cache.py
---------------

Shared caching functionality for SSCOFS NetCDF data files.
This module provides a consistent caching mechanism for all scripts
that download SSCOFS data.
"""

import xarray as xr
import s3fs
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Default cache directory - can be overridden
DEFAULT_CACHE_DIR = Path(__file__).parent / ".sscofs_cache"


def get_cached_filename(run_info: Dict) -> str:
    """
    Generate a standardized cache filename from model run info.
    
    Parameters:
    -----------
    run_info : dict
        Dictionary with keys: 'run_date_utc', 'cycle_utc', 'forecast_hour_index'
        
    Returns:
    --------
    str : Filename for the cached file
    """
    run_date = run_info['run_date_utc'].replace('-', '')
    cycle = run_info['cycle_utc'].replace('z', '')
    fhour = run_info['forecast_hour_index']
    return f"sscofs_{run_date}_t{cycle}z_f{fhour:03d}.nc"


def download_to_cache(run_info: Dict,
                      cache_dir: Optional[Path] = None,
                      verbose: bool = True) -> Path:
    """
    Download SSCOFS file from S3 to cache as raw bytes (no parsing).
    
    This is much faster than load_sscofs_data() because it just copies
    bytes without parsing the NetCDF structure.
    
    Parameters:
    -----------
    run_info : dict
        Dictionary with keys: 'run_date_utc', 'cycle_utc', 'forecast_hour_index', 'url'
    cache_dir : Path, optional
        Directory to store cached files. If None, uses DEFAULT_CACHE_DIR.
    verbose : bool
        If True, print status messages.
        
    Returns:
    --------
    Path : Path to the cached file
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    
    # Create cache directory if it doesn't exist
    cache_dir.mkdir(exist_ok=True)
    
    # Generate cache filename
    cache_file = cache_dir / get_cached_filename(run_info)
    
    # Download raw bytes from S3
    if verbose:
        fh = run_info['forecast_hour_index']
        print(f"Downloading forecast hour {fh:03d}...")
    
    fs = s3fs.S3FileSystem(anon=True)
    
    # Extract the S3 key from the URL
    url = run_info['url']
    key = url.split("https://noaa-nos-ofs-pds.s3.amazonaws.com/")[1]
    
    # Just copy bytes - no parsing!
    with fs.open(f"noaa-nos-ofs-pds/{key}", 'rb') as f_in:
        with open(cache_file, 'wb') as f_out:
            # Copy in chunks for memory efficiency
            chunk_size = 8 * 1024 * 1024  # 8 MB chunks
            while True:
                chunk = f_in.read(chunk_size)
                if not chunk:
                    break
                # Ensure chunk is bytes
                if isinstance(chunk, str):
                    chunk = chunk.encode()
                f_out.write(chunk)
    
    if verbose:
        size_mb = cache_file.stat().st_size / (1024 * 1024)
        print(f"  ✓ Saved {size_mb:.1f} MB to cache")
    
    return cache_file


def load_sscofs_data(run_info: Dict, 
                     use_cache: bool = True,
                     cache_dir: Optional[Path] = None,
                     verbose: bool = True) -> xr.Dataset:
    """
    Load SSCOFS data from cache or download from S3.
    
    Parameters:
    -----------
    run_info : dict
        Dictionary with keys: 'run_date_utc', 'cycle_utc', 'forecast_hour_index', 'url'
    use_cache : bool
        If True, check for cached files and use them if available.
        If False, always download from S3 (but still save to cache).
    cache_dir : Path, optional
        Directory to store cached files. If None, uses DEFAULT_CACHE_DIR.
    verbose : bool
        If True, print status messages.
        
    Returns:
    --------
    xr.Dataset : The loaded SSCOFS dataset
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    
    # Create cache directory if it doesn't exist
    cache_dir.mkdir(exist_ok=True)
    
    # Generate cache filename
    cache_file = cache_dir / get_cached_filename(run_info)
    
    # Check if cached file exists
    if not (use_cache and cache_file.exists()):
        # Need to download it first
        download_to_cache(run_info, cache_dir=cache_dir, verbose=verbose)
    elif verbose:
        print(f"Using cached file: {cache_file.name}")
    
    # Load from the cached file
    ds = xr.open_dataset(cache_file, engine='h5netcdf')
    return ds


def list_cache(cache_dir: Optional[Path] = None) -> None:
    """
    List all cached files and their sizes.
    
    Parameters:
    -----------
    cache_dir : Path, optional
        Directory containing cached files. If None, uses DEFAULT_CACHE_DIR.
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
        
    if not cache_dir.exists():
        print("No cache directory found.")
        return
    
    cache_files = sorted(cache_dir.glob("*.nc"))
    if not cache_files:
        print("Cache is empty.")
        return
    
    print(f"\nCached files in {cache_dir}:")
    print("-" * 70)
    total_size = 0
    for f in cache_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        total_size += size_mb
        print(f"  {f.name:50s} {size_mb:7.2f} MB")
    print("-" * 70)
    print(f"Total: {len(cache_files)} files, {total_size:.2f} MB")


def clear_cache(cache_dir: Optional[Path] = None) -> int:
    """
    Delete all cached files.
    
    Parameters:
    -----------
    cache_dir : Path, optional
        Directory containing cached files. If None, uses DEFAULT_CACHE_DIR.
        
    Returns:
    --------
    int : Number of files deleted
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
        
    if not cache_dir.exists():
        print("No cache directory found.")
        return 0
    
    cache_files = list(cache_dir.glob("*.nc"))
    if not cache_files:
        print("Cache is already empty.")
        return 0
    
    count = 0
    for f in cache_files:
        f.unlink()
        count += 1
    print(f"Deleted {count} cached files from {cache_dir}")
    return count


def get_cache_info(cache_dir: Optional[Path] = None) -> Dict:
    """
    Get information about the cache.
    
    Parameters:
    -----------
    cache_dir : Path, optional
        Directory containing cached files. If None, uses DEFAULT_CACHE_DIR.
        
    Returns:
    --------
    dict : Dictionary with keys: 'num_files', 'total_size_mb', 'files'
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
        
    if not cache_dir.exists():
        return {'num_files': 0, 'total_size_mb': 0.0, 'files': []}
    
    cache_files = sorted(cache_dir.glob("*.nc"))
    total_size = sum(f.stat().st_size for f in cache_files) / (1024 * 1024)
    
    files_info = [
        {
            'name': f.name,
            'size_mb': f.stat().st_size / (1024 * 1024),
            'path': str(f)
        }
        for f in cache_files
    ]
    
    return {
        'num_files': len(cache_files),
        'total_size_mb': total_size,
        'files': files_info
    }


def bulk_download_forecasts(
    run_infos: List[Dict[str, Any]],
    use_cache: bool = True,
    cache_dir: Optional[Path] = None,
    max_workers: int = 5,
    verbose: bool = True
) -> List[Optional[Path]]:
    """
    Download multiple SSCOFS forecast files in parallel to cache.
    
    This function downloads raw NetCDF files without parsing them,
    which is much faster than loading into memory. Files are saved
    to cache and can be loaded later with xarray as needed.
    
    Parameters:
    -----------
    run_infos : list of dict
        List of run_info dictionaries, each with keys:
        'run_date_utc', 'cycle_utc', 'forecast_hour_index', 'url'
    use_cache : bool
        If True, skip downloading files that are already cached.
    cache_dir : Path, optional
        Directory to store cached files. If None, uses DEFAULT_CACHE_DIR.
    max_workers : int
        Maximum number of parallel downloads (default: 5).
        Adjust based on your network bandwidth.
    verbose : bool
        If True, print progress messages.
        
    Returns:
    --------
    list of Path : List of paths to cached files (in same order as run_infos)
    
    Example:
    --------
    >>> from latest_cycle import find_latest_cycle, build_url
    >>> run_date, cycle, _ = find_latest_cycle()
    >>> run_infos = []
    >>> for fh in [0, 6, 12, 18, 24]:
    ...     run_infos.append({
    ...         'run_date_utc': f'{run_date:%Y-%m-%d}',
    ...         'cycle_utc': f'{cycle:02d}z',
    ...         'forecast_hour_index': fh,
    ...         'url': build_url(run_date, cycle, True, fh)
    ...     })
    >>> cache_paths = bulk_download_forecasts(run_infos, max_workers=5)
    >>> # Later, load a specific file:
    >>> import xarray as xr
    >>> ds = xr.open_dataset(cache_paths[0], engine='h5netcdf')
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    
    cache_dir.mkdir(exist_ok=True)
    
    if verbose:
        print(f"\nBulk downloading {len(run_infos)} forecast files with {max_workers} workers...")
        print(f"Skip cached files: {use_cache}")
    
    # Store results with their original index to maintain order
    results: List[Optional[Path]] = [None] * len(run_infos)
    
    # Filter out files that are already cached if use_cache=True
    to_download = []
    for idx, info in enumerate(run_infos):
        cache_file = cache_dir / get_cached_filename(info)
        if use_cache and cache_file.exists():
            results[idx] = cache_file
            if verbose:
                fh = info['forecast_hour_index']
                print(f"  [{idx+1}/{len(run_infos)}] Forecast hour {fh:03d} - already cached")
        else:
            to_download.append((idx, info))
    
    if not to_download:
        if verbose:
            print("\n✓ All files already cached!")
        return results
    
    if verbose:
        print(f"\nDownloading {len(to_download)} files...")
    
    def download_one(idx_and_info: Tuple[int, Dict[str, Any]]) -> Tuple[int, Path]:
        idx, info = idx_and_info
        cache_file = download_to_cache(info, cache_dir=cache_dir, verbose=False)
        return idx, cache_file
    
    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        futures = {
            executor.submit(download_one, item): item[0] 
            for item in to_download
        }
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(futures):
            try:
                idx, cache_file = future.result()
                results[idx] = cache_file
                completed += 1
                if verbose:
                    fh = run_infos[idx]['forecast_hour_index']
                    size_mb = cache_file.stat().st_size / (1024 * 1024)
                    print(f"  ✓ [{completed}/{len(to_download)}] Forecast hour {fh:03d} - {size_mb:.1f} MB", flush=True)
            except Exception as e:
                idx = futures[future]
                if verbose:
                    fh = run_infos[idx]['forecast_hour_index']
                    print(f"  ✗ Error downloading forecast hour {fh:03d}: {e}", flush=True)
                # Keep None in results to maintain order
    
    if verbose:
        successful = sum(1 for r in results if r is not None)
        print(f"\n{'═' * 70}")
        print(f"Downloaded {successful}/{len(run_infos)} files successfully")
        print(f"All files cached in: {cache_dir}")
    
    return results


if __name__ == "__main__":
    # Command-line interface for cache management
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Manage SSCOFS data cache"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List cached files"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all cached files"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show cache information"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_cache()
    elif args.clear:
        clear_cache()
    elif args.info:
        info = get_cache_info()
        print(f"\nCache information:")
        print(f"  Files: {info['num_files']}")
        print(f"  Total size: {info['total_size_mb']:.2f} MB")
        print(f"  Location: {DEFAULT_CACHE_DIR}")
    else:
        parser.print_help()

