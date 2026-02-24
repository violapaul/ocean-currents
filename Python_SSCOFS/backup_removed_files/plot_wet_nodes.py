"""
plot_wet_nodes.py
-----------------

Quick visualization of the wet_nodes mask from SSCOFS data.
Shows which nodes are wet (water) vs dry (land) within a radius of a location.

Usage:
    python plot_wet_nodes.py --lat 47.67181822458632 --lon -122.4583957143628 --radius 5
"""

import argparse
import datetime as dt
from datetime import timezone, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.patches import Circle
from pathlib import Path
from pyproj import Transformer

# Import helper functions from existing modules
from latest_cycle import latest_cycle_and_url_for_local_hour
from sscofs_cache import load_sscofs_data

def get_utm_zone(lon):
    """Get UTM zone number from longitude."""
    return int((lon + 180) / 6) + 1

def create_utm_transformer(center_lat, center_lon):
    """Create a transformer for converting lat/lon to UTM coordinates."""
    utm_zone = get_utm_zone(center_lon)
    hemisphere = 'north' if center_lat >= 0 else 'south'
    utm_crs = f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84 +units=m +no_defs"
    transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    return transformer, utm_zone, hemisphere

def get_latest_current_data(use_cache=True):
    """Get the latest SSCOFS current data using the current time."""
    current_time_utc = dt.datetime.now(timezone.utc)
    try:
        from zoneinfo import ZoneInfo
        current_time_local = current_time_utc.astimezone(ZoneInfo("America/Los_Angeles"))
    except ImportError:
        current_time_local = current_time_utc - timedelta(hours=8)
    
    local_hour = current_time_local.hour * 100 + current_time_local.minute
    
    print(f"Current time (local): {current_time_local:%Y-%m-%d %H:%M:%S}")
    
    # Get the latest cycle info
    info = latest_cycle_and_url_for_local_hour(local_hour, "America/Los_Angeles")
    
    print(f"Using SSCOFS: {info['run_date_utc']} {info['cycle_utc']} f{info['forecast_hour_index']:03d}")
    
    # Use shared caching module
    ds = load_sscofs_data(info, use_cache=use_cache, verbose=True)
    
    return ds, info

def plot_wet_nodes(ds, center_lat, center_lon, radius_miles=5, 
                   time_index=0, save_file=None, wet_vars=None):
    """
    Plot wet/dry nodes/cells within a radius of a center point.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        SSCOFS dataset
    center_lat, center_lon : float
        Center coordinates in decimal degrees
    radius_miles : float
        Radius in miles to plot around center
    time_index : int
        Which time step to plot (default 0)
    save_file : str, optional
        If provided, save the plot to this file
    wet_vars : list of str, optional
        Names of wet variables to plot (default: ["wet_nodes"])
        Common options: "wet_nodes", "wet_cells"
        Can specify multiple to compare, e.g., ["wet_nodes", "wet_cells"]
    """
    
    if wet_vars is None:
        wet_vars = ["wet_nodes"]
    
    # Ensure wet_vars is a list
    if isinstance(wet_vars, str):
        wet_vars = [wet_vars]
    
    # Convert radius from miles to meters
    radius_meters = radius_miles * 1609.34
    
    # Create UTM transformer
    transformer, utm_zone, hemisphere = create_utm_transformer(center_lat, center_lon)
    
    print(f"\nUsing UTM Zone {utm_zone}{hemisphere[0].upper()}")
    
    # Define colors for different variables (distinct colors)
    colors = ['red', 'green', 'blue', 'magenta', 'orange', 'purple']
    dry_color = 'brown'
    
    # Data structure to hold results for each variable
    plot_data = []
    
    # Process each wet variable
    for var_idx, wet_var in enumerate(wet_vars):
        print(f"\nProcessing variable: {wet_var}")
        
        # Check if the requested variable exists
        if wet_var not in ds:
            print(f"  WARNING: '{wet_var}' not found in dataset - skipping")
            print(f"  Available variables with 'wet' or 'mask':")
            candidates = [v for v in ds.variables if 'wet' in v.lower() or 'mask' in v.lower()]
            if candidates:
                for v in candidates:
                    print(f"    - {v}")
            continue
        
        # Extract wet variable at the time step
        wet = ds[wet_var].isel(time=time_index)
        
        # Get coordinates - determine if it's on nodes or cells
        if "nodes" in wet_var.lower():
            lons = ds["lon"].values
            lats = ds["lat"].values
            coord_type = "nodes"
        elif "cells" in wet_var.lower() or "elem" in wet_var.lower():
            lons = ds["lonc"].values
            lats = ds["latc"].values
            coord_type = "cells"
        else:
            # Try to infer from dimensions
            if 'node' in wet.dims:
                lons = ds["lon"].values
                lats = ds["lat"].values
                coord_type = "nodes"
            else:
                lons = ds["lonc"].values
                lats = ds["latc"].values
                coord_type = "cells"
        
        print(f"  Coordinate type: {coord_type}")
        
        # Convert longitudes from 0-360 to -180 to 180 format if needed
        if lons.max() > 180:
            lons = np.where(lons > 180, lons - 360, lons)
        
        # Transform all points to UTM
        x_utm, y_utm = transformer.transform(lons, lats)
        
        # Transform center point to UTM
        center_x, center_y = transformer.transform(center_lon, center_lat)
        
        # Calculate distances from center in meters
        distances = np.sqrt((x_utm - center_x)**2 + (y_utm - center_y)**2)
        
        # Create mask for points within radius
        radius_mask = distances <= radius_meters
        
        # Apply radius mask
        x_in_radius = x_utm[radius_mask]
        y_in_radius = y_utm[radius_mask]
        wet_in_radius = wet.values[radius_mask]
        
        # Separate wet and dry points
        wet_points_mask = wet_in_radius == 1
        dry_points_mask = wet_in_radius == 0
        
        x_wet = x_in_radius[wet_points_mask]
        y_wet = y_in_radius[wet_points_mask]
        x_dry = x_in_radius[dry_points_mask]
        y_dry = y_in_radius[dry_points_mask]
        
        print(f"  Total points in radius: {len(x_in_radius)}")
        print(f"  Wet: {len(x_wet)} ({100*len(x_wet)/len(x_in_radius):.1f}%)")
        print(f"  Dry: {len(x_dry)} ({100*len(x_dry)/len(x_in_radius):.1f}%)")
        
        # Store data for plotting
        plot_data.append({
            'var_name': wet_var,
            'coord_type': coord_type,
            'x_wet': x_wet,
            'y_wet': y_wet,
            'x_dry': x_dry,
            'y_dry': y_dry,
            'color': colors[var_idx % len(colors)]
        })
    
    if not plot_data:
        print("\nERROR: No valid variables to plot!")
        return None
    
    # Get time information
    time_val = ds["time"].isel(time=time_index).values
    time_utc = pd.Timestamp(time_val).to_pydatetime()
    try:
        from zoneinfo import ZoneInfo
        time_local = time_utc.replace(tzinfo=timezone.utc).astimezone(ZoneInfo("America/Los_Angeles"))
        time_str = time_local.strftime("%Y-%m-%d %H:%M:%S %Z")
    except ImportError:
        time_local = time_utc - timedelta(hours=8)
        time_str = time_local.strftime("%Y-%m-%d %H:%M:%S PST")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Plot each variable with its own color
    for data in plot_data:
        # Plot dry nodes first (if any) with consistent brown color
        if len(data['x_dry']) > 0:
            ax.scatter(data['x_dry'], data['y_dry'], 
                      c=dry_color, s=10, alpha=0.3, 
                      marker='x',
                      label=f"{data['var_name']}: Dry ({len(data['x_dry'])})", 
                      edgecolors='none')
        
        # Plot wet nodes with variable-specific color
        if len(data['x_wet']) > 0:
            ax.scatter(data['x_wet'], data['y_wet'], 
                      c=data['color'], s=10, alpha=0.4, 
                      label=f"{data['var_name']}: Wet ({len(data['x_wet'])})", 
                      edgecolors='none')
    
    # Mark center point
    ax.plot(center_x, center_y, 'r*', markersize=20, 
            label='Center Point', zorder=10, markeredgecolor='white', markeredgewidth=1)
    
    # Add circle to show radius
    circle = Circle((center_x, center_y), radius_meters,
                   fill=False, edgecolor='red', linewidth=2, 
                   linestyle='--', label=f'{radius_miles} mile radius')
    ax.add_patch(circle)
    
    ax.set_xlabel('Easting (m, UTM)', fontsize=12)
    ax.set_ylabel('Northing (m, UTM)', fontsize=12)
    
    # Create title based on number of variables
    if len(wet_vars) == 1:
        title = f'SSCOFS {wet_vars[0]} Mask'
    else:
        title = f'SSCOFS Wet Masks: {", ".join(wet_vars)}'
    
    ax.set_title(f'{title}\nTime: {time_str}\nUTM Zone {utm_zone}{hemisphere[0].upper()}', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_aspect('equal')
    
    # Set axis limits with margin
    margin = radius_meters * 0.05
    ax.set_xlim(center_x - radius_meters - margin, center_x + radius_meters + margin)
    ax.set_ylim(center_y - radius_meters - margin, center_y + radius_meters + margin)
    
    # Add location info as text
    location_text = f'Location: {center_lat:.4f}°N, {abs(center_lon):.4f}°W'
    ax.text(0.02, 0.02, location_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_file:
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_file}")
    
    plt.show()
    
    return fig

def main():
    parser = argparse.ArgumentParser(
        description="Plot SSCOFS wet mask (wet_nodes or wet_cells) within a radius of a GPS coordinate"
    )
    parser.add_argument(
        "--lat",
        type=float,
        default=47.67181822458632,
        help="Latitude of center point (default: 47.6718)"
    )
    parser.add_argument(
        "--lon", 
        type=float,
        default=-122.4583957143628,
        help="Longitude of center point (default: -122.4584)"
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=5.0,
        help="Radius in miles (default: 5)"
    )
    parser.add_argument(
        "--time-index",
        type=int,
        default=0,
        help="Time index to plot (default: 0, first time step)"
    )
    parser.add_argument(
        "--save",
        type=str,
        help="Save plot to file (e.g., 'wet_nodes.png')"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Don't use cached data, always download fresh"
    )
    parser.add_argument(
        "--wet-var",
        type=str,
        nargs='+',
        default=["wet_nodes"],
        help="Wet variable(s) to plot. Can specify multiple: --wet-var wet_nodes wet_cells (default: wet_nodes)"
    )
    
    args = parser.parse_args()
    
    var_list = ", ".join(args.wet_var) if isinstance(args.wet_var, list) else args.wet_var
    print(f"Plotting {var_list} within {args.radius} miles of ({args.lat}, {args.lon})")
    
    try:
        # Get the latest data (use cache unless --no-cache is specified)
        ds, info = get_latest_current_data(use_cache=not args.no_cache)
        
        # Plot the wet nodes/cells
        fig = plot_wet_nodes(
            ds, 
            args.lat, 
            args.lon, 
            radius_miles=args.radius,
            time_index=args.time_index,
            save_file=args.save,
            wet_vars=args.wet_var
        )
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

