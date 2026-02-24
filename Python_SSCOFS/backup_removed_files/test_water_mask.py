"""
test_water_mask.py
------------------

Quick test to check if SSCOFS data contains water/land mask variables.

Many coastal models ship one or more of these (names vary):
- wet, wet_nodes, wetdry_elem
- mask_rho, land_mask
- h/depth (<=0 = land)

If present, these masks should be used to exclude land points before
any interpolation or derivative calculations.
"""

import xarray as xr
import numpy as np
from latest_cycle import latest_cycle_and_url_for_local_hour
from sscofs_cache import load_sscofs_data
import datetime as dt
from zoneinfo import ZoneInfo


def check_for_water_mask(ds: xr.Dataset, verbose: bool = True):
    """
    Search for candidate water/land mask variables in the dataset.
    
    Parameters:
    -----------
    ds : xr.Dataset
        The SSCOFS dataset to check
    verbose : bool
        If True, print detailed information
        
    Returns:
    --------
    list : List of candidate mask variable names found
    """
    # Common mask-related keywords
    mask_keywords = ["wet", "wetdry", "mask", "land", "rho", "depth", "h"]
    
    # Search all variables for candidates
    candidates = [k for k in ds.variables if any(s in k.lower() for s in mask_keywords)]
    
    if verbose:
        print("=" * 70)
        print("SEARCHING FOR WATER/LAND MASK VARIABLES")
        print("=" * 70)
        print(f"\nCandidate mask variables found: {len(candidates)}")
        if candidates:
            for var in candidates:
                print(f"  - {var}")
        else:
            print("  (none found)")
        print()
    
    return candidates


def examine_variable(ds: xr.Dataset, var_name: str):
    """
    Print detailed information about a variable.
    
    Parameters:
    -----------
    ds : xr.Dataset
        The dataset
    var_name : str
        Name of the variable to examine
    """
    if var_name not in ds:
        print(f"Variable '{var_name}' not found in dataset")
        return
    
    var = ds[var_name]
    print(f"\n{'='*70}")
    print(f"VARIABLE: {var_name}")
    print(f"{'='*70}")
    print(f"Shape: {var.shape}")
    print(f"Dimensions: {var.dims}")
    print(f"Data type: {var.dtype}")
    
    # Print attributes
    if var.attrs:
        print("\nAttributes:")
        for attr, value in var.attrs.items():
            print(f"  {attr}: {value}")
    
    # For small variables or time-independent masks, show unique values
    if var.size < 1000000:  # Only for reasonably-sized variables
        try:
            unique_vals = np.unique(var.values[~np.isnan(var.values)])
            print(f"\nUnique values (excluding NaN): {unique_vals[:20]}")  # First 20
            if len(unique_vals) > 20:
                print(f"  ... and {len(unique_vals) - 20} more")
        except Exception as e:
            print(f"\nCould not compute unique values: {e}")
    
    # Show min/max for numeric data
    try:
        print(f"\nValue range: [{float(var.min()):.3f}, {float(var.max()):.3f}]")
    except Exception:
        pass


def demonstrate_masking(ds: xr.Dataset, mask_var: str):
    """
    Demonstrate how to apply a mask to u/v velocity data.
    
    Parameters:
    -----------
    ds : xr.Dataset
        The dataset
    mask_var : str
        Name of the mask variable to use
    """
    print(f"\n{'='*70}")
    print(f"DEMONSTRATION: Applying mask '{mask_var}' to velocity data")
    print(f"{'='*70}")
    
    # Check if u and v exist
    if 'u' not in ds or 'v' not in ds:
        print("Variables 'u' and 'v' not found - cannot demonstrate masking")
        return
    
    mask = ds[mask_var]
    
    print("\nExample code to apply mask:")
    print("-" * 70)
    
    # Determine how to apply the mask based on its dimensions
    if 'time' in mask.dims:
        print(f"""
# Time-varying mask (e.g., for tidal flats)
time_idx = 0  # or use .sel(time=...) 
mask_at_time = ds["{mask_var}"].isel(time=time_idx)

# Apply mask (assuming 1=water, 0=land)
u_masked = ds["u"].isel(time=time_idx).where(mask_at_time == 1)
v_masked = ds["v"].isel(time=time_idx).where(mask_at_time == 1)

# Or apply to all times
u_masked = ds["u"].where(ds["{mask_var}"] == 1)
v_masked = ds["v"].where(ds["{mask_var}"] == 1)
""")
    else:
        print(f"""
# Static mask (time-independent)
u_masked = ds["u"].where(ds["{mask_var}"] == 1)
v_masked = ds["v"].where(ds["{mask_var}"] == 1)
""")
    
    print("-" * 70)
    
    # Try to actually apply the mask and show results
    try:
        print("\nAttempting to apply mask to first time step...")
        u = ds["u"]
        v = ds["v"]
        
        # Get first time step
        if 'time' in u.dims:
            u_t0 = u.isel(time=0)
            v_t0 = v.isel(time=0)
            
            # Get mask at first time or use static mask
            if 'time' in mask.dims:
                mask_t0 = mask.isel(time=0)
            else:
                mask_t0 = mask
            
            # Apply mask
            u_masked = u_t0.where(mask_t0 == 1)
            v_masked = v_t0.where(mask_t0 == 1)
            
            # Count valid points
            u_valid_orig = np.sum(~np.isnan(u_t0.values))
            u_valid_masked = np.sum(~np.isnan(u_masked.values))
            
            print(f"\nResults:")
            print(f"  Original valid points: {u_valid_orig}")
            print(f"  After masking: {u_valid_masked}")
            print(f"  Points masked out: {u_valid_orig - u_valid_masked}")
            
    except Exception as e:
        print(f"\nCould not apply mask: {e}")


def main():
    print("SSCOFS Water Mask Detection Test")
    print("=" * 70)
    
    # Get latest cycle for current local time
    current_time_utc = dt.datetime.now(dt.timezone.utc)
    current_time_local = current_time_utc.astimezone(ZoneInfo("America/Los_Angeles"))
    local_hour = current_time_local.hour * 100 + current_time_local.minute
    
    print(f"\nCurrent time (local): {current_time_local:%Y-%m-%d %H:%M %Z}")
    
    # Get run info
    print("\nFetching latest SSCOFS cycle...")
    run_info = latest_cycle_and_url_for_local_hour(local_hour, "America/Los_Angeles")
    
    print(f"Run date: {run_info['run_date_utc']}")
    print(f"Cycle: {run_info['cycle_utc']}")
    print(f"Forecast hour: {run_info['forecast_hour_index']:03d}")
    
    # Load data
    print("\nLoading data (may download if not cached)...")
    ds = load_sscofs_data(run_info, use_cache=True, verbose=True)
    
    print(f"\nDataset loaded successfully!")
    print(f"Dimensions: {dict(ds.dims)}")
    print(f"Number of variables: {len(ds.variables)}")
    
    # Search for mask candidates
    candidates = check_for_water_mask(ds)
    
    # Examine each candidate in detail
    if candidates:
        print("\n" + "=" * 70)
        print("DETAILED EXAMINATION OF CANDIDATE VARIABLES")
        print("=" * 70)
        
        for var in candidates:
            examine_variable(ds, var)
        
        # Demonstrate masking with the first candidate that looks promising
        # (typically would be 'wet', 'mask_rho', or similar)
        print("\n" + "=" * 70)
        print("MASKING DEMONSTRATION")
        print("=" * 70)
        
        # Try to find the most likely mask variable
        likely_masks = [v for v in candidates if any(
            kw in v.lower() for kw in ['wet', 'mask']
        )]
        
        if likely_masks:
            demonstrate_masking(ds, likely_masks[0])
        else:
            print("\nNo obvious mask variable found to demonstrate with.")
            print("You may need to manually inspect the candidates above.")
    
    else:
        print("\n" + "=" * 70)
        print("NO MASK VARIABLES FOUND")
        print("=" * 70)
        print("\nThe dataset does not appear to contain standard mask variables.")
        print("\nAlternative approaches:")
        print("  1. Use depth/bathymetry (if h <= 0, it's land)")
        print("  2. Check for NaN patterns in u/v (NaN often indicates land)")
        print("  3. Look for mesh connectivity info (dry elements)")
        
        # Check for depth/bathymetry
        depth_vars = [v for v in ds.variables if any(
            kw in v.lower() for kw in ['depth', 'h', 'bath']
        )]
        if depth_vars:
            print(f"\nFound potential depth variables: {depth_vars}")
            print("You can create a mask from depth: mask = (depth > 0)")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

