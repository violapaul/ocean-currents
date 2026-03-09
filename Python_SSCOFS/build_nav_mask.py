#!/usr/bin/env python3
"""
build_nav_mask.py
-----------------
Downloads the NOAA NCEI Puget Sound 1/3 arc-second Coastal DEM, thresholds
by depth to identify navigable water, resamples to ~50m resolution, and
saves a compact boolean mask for use in routing and display.

The mask is used to:
1. Prune Delaunay triangles whose centroids fall on land (display)
2. Set water_mask cells in A* routing grid (routing)

Usage:
    python build_nav_mask.py                     # default -3.0m threshold
    python build_nav_mask.py --depth -2.5        # shallower threshold
    python build_nav_mask.py --depth -4.0        # more conservative
    python build_nav_mask.py --output custom.npz # custom output path
"""

import argparse
import os
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import xarray as xr

# NOAA NCEI Puget Sound 1/3 arc-second Coastal DEM (NAVD88)
# Direct download URL (file is ~400MB)
DEM_DOWNLOAD_URL = "https://www.ngdc.noaa.gov/thredds/fileServer/regional/puget_sound_13_navd88_2014.nc"

# Default output path
DEFAULT_OUTPUT = Path(__file__).parent / "data" / "nav_mask.npz"

# Resample factor: 5x reduces ~10m to ~50m resolution
# This keeps the mask small (~600KB) while still being denser than SSCOFS mesh
RESAMPLE_FACTOR = 5

# Cache directory for downloaded DEM
CACHE_DIR = Path(__file__).parent / "data" / "cache"


def _download_with_progress(url: str, dest: Path, verbose: bool = True):
    """Download file with progress indicator."""
    def reporthook(count, block_size, total_size):
        if verbose and total_size > 0:
            pct = count * block_size * 100 / total_size
            print(f"\r  Downloading: {pct:.1f}%", end="", flush=True)
    
    urlretrieve(url, dest, reporthook)
    if verbose:
        print()


def build_nav_mask(
    depth_threshold: float = -3.0,
    output_path: Path = DEFAULT_OUTPUT,
    resample_factor: int = RESAMPLE_FACTOR,
    verbose: bool = True,
    use_cache: bool = True,
) -> dict:
    """Download DEM, threshold, resample, and save navigable water mask.
    
    Parameters
    ----------
    depth_threshold : float
        Elevation threshold in meters (NAVD88). Cells below this value
        are considered navigable water. Default -3.0m gives ~1m clearance
        for a J105 (draft ~2.1m).
    output_path : Path
        Where to save the .npz mask file.
    resample_factor : int
        Downsample factor (5 = ~10m -> ~50m).
    verbose : bool
        Print progress messages.
    use_cache : bool
        If True, cache downloaded DEM file for reuse.
        
    Returns
    -------
    dict with keys: water, lon_min, lon_max, lat_min, lat_max, res_deg, depth_threshold
    """
    # Check for cached DEM file
    cache_file = CACHE_DIR / "puget_sound_13_navd88_2014.nc"
    
    if use_cache and cache_file.exists():
        if verbose:
            print(f"Using cached DEM: {cache_file}")
        dem_path = cache_file
    else:
        if verbose:
            print(f"Downloading DEM (~400MB): {DEM_DOWNLOAD_URL}")
        
        # Download to cache or temp file
        if use_cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            dem_path = cache_file
            _download_with_progress(DEM_DOWNLOAD_URL, dem_path, verbose)
        else:
            fd, dem_path = tempfile.mkstemp(suffix=".nc")
            os.close(fd)
            dem_path = Path(dem_path)
            _download_with_progress(DEM_DOWNLOAD_URL, dem_path, verbose)
    
    if verbose:
        print(f"Opening DEM: {dem_path}")
    
    # Open the DEM
    ds = xr.open_dataset(dem_path)
    
    if verbose:
        print(f"  Grid shape: {ds['Band1'].shape}")
        print(f"  Lon range: {float(ds['lon'].min()):.4f} to {float(ds['lon'].max()):.4f}")
        print(f"  Lat range: {float(ds['lat'].min()):.4f} to {float(ds['lat'].max()):.4f}")
    
    # Extract coordinates
    lon = ds['lon'].values
    lat = ds['lat'].values
    
    if verbose:
        print(f"Loading elevation data (this may take a minute)...")
    
    # Load the elevation data - this triggers the actual download
    # Band1 contains elevation in meters (NAVD88)
    elev = ds['Band1'].values  # shape: (lat, lon)
    
    if verbose:
        print(f"  Elevation range: {np.nanmin(elev):.1f}m to {np.nanmax(elev):.1f}m")
    
    # Threshold: water where elevation < depth_threshold
    # NaN values (no data) are treated as land
    water_full = elev < depth_threshold
    water_full = np.where(np.isnan(elev), False, water_full)
    
    if verbose:
        water_pct = 100.0 * water_full.sum() / water_full.size
        print(f"  Water cells (< {depth_threshold}m): {water_full.sum():,} / {water_full.size:,} ({water_pct:.1f}%)")
    
    # Resample by taking every Nth cell
    # Use a logical OR over the NxN block to be conservative (any water = water)
    # Actually, let's use majority vote for cleaner edges
    ny_full, nx_full = water_full.shape
    ny_new = ny_full // resample_factor
    nx_new = nx_full // resample_factor
    
    # Trim to exact multiple
    water_trimmed = water_full[:ny_new * resample_factor, :nx_new * resample_factor]
    
    # Reshape and compute majority vote per block
    water_blocks = water_trimmed.reshape(ny_new, resample_factor, nx_new, resample_factor)
    water_resampled = water_blocks.sum(axis=(1, 3)) > (resample_factor * resample_factor // 2)
    
    # Compute new coordinate bounds
    lon_resampled = lon[:nx_new * resample_factor:resample_factor]
    lat_resampled = lat[:ny_new * resample_factor:resample_factor]
    
    lon_min = float(lon_resampled[0])
    lon_max = float(lon_resampled[-1])
    lat_min = float(lat_resampled[-1])  # lat is typically descending in rasters
    lat_max = float(lat_resampled[0])
    
    # Handle case where lat might be ascending
    if lat_min > lat_max:
        lat_min, lat_max = lat_max, lat_min
    
    # Resolution in degrees (approximate, assumes regular grid)
    res_deg = float(abs(lon[resample_factor] - lon[0]))
    
    if verbose:
        print(f"Resampled to {nx_new} x {ny_new} ({resample_factor}x downsample)")
        print(f"  Resolution: ~{res_deg * 111000:.0f}m")
        water_pct = 100.0 * water_resampled.sum() / water_resampled.size
        print(f"  Water cells: {water_resampled.sum():,} / {water_resampled.size:,} ({water_pct:.1f}%)")
    
    # Save as compressed numpy archive
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    result = {
        'water': water_resampled.astype(np.uint8),  # uint8 for smaller file
        'lon_min': lon_min,
        'lon_max': lon_max,
        'lat_min': lat_min,
        'lat_max': lat_max,
        'res_deg': res_deg,
        'depth_threshold': depth_threshold,
    }
    
    np.savez_compressed(output_path, **result)
    
    size_kb = output_path.stat().st_size / 1024
    if verbose:
        print(f"Saved to {output_path} ({size_kb:.0f} KB)")
    
    ds.close()
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Build navigable water mask from NOAA bathymetric DEM"
    )
    parser.add_argument(
        "--depth", type=float, default=-3.0,
        help="Depth threshold in meters NAVD88 (default: -3.0, gives ~1m clearance for J105)"
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output .npz file (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--resample", type=int, default=RESAMPLE_FACTOR,
        help=f"Downsample factor (default: {RESAMPLE_FACTOR}, ~10m -> ~50m)"
    )
    args = parser.parse_args()
    
    build_nav_mask(
        depth_threshold=args.depth,
        output_path=args.output,
        resample_factor=args.resample,
    )


if __name__ == "__main__":
    main()
