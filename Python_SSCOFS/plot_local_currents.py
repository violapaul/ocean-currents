"""
plot_local_currents.py
----------------------

Download SSCOFS current model data and plot currents within a specified radius
of a GPS coordinate in the Puget Sound area.

Usage:
    python plot_local_currents.py --lat 47.67181822458632 --lon -122.4583957143628 --radius 5
    
Or use defaults (your example coordinates):
    python plot_local_currents.py
"""

import argparse
import datetime as dt
from datetime import timezone, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.patches import Circle, Ellipse
from matplotlib import cm
import matplotlib.patches as mpatches
from pathlib import Path
from pyproj import Transformer

# Import helper functions from existing modules
from latest_cycle import latest_cycle_and_url_for_local_hour
from fetch_sscofs import build_sscofs_url, compute_file_for_datetime
from sscofs_cache import load_sscofs_data, list_cache, clear_cache

def get_utm_zone(lon):
    """Get UTM zone number from longitude."""
    return int((lon + 180) / 6) + 1

def create_utm_transformer(center_lat, center_lon):
    """
    Create a transformer for converting lat/lon to UTM coordinates.
    Returns a transformer and the UTM zone info.
    """
    utm_zone = get_utm_zone(center_lon)
    # Determine hemisphere (north or south)
    hemisphere = 'north' if center_lat >= 0 else 'south'
    
    # Create transformer from WGS84 (EPSG:4326) to UTM
    utm_crs = f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84 +units=m +no_defs"
    transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    
    return transformer, utm_zone, hemisphere


def get_latest_current_data(use_cache=True, target_datetime=None, tz_str="America/Los_Angeles", 
                            forecast_hour_offset=None):
    """
    Get SSCOFS current data. If target_datetime is provided, get data for that time.
    Otherwise, get the latest available data.
    Returns the dataset and metadata about the run.
    
    Parameters:
    -----------
    use_cache : bool
        If True, check for cached files and use them if available.
        If False, always download from S3.
    target_datetime : datetime, optional
        Target date and time to get data for. Can be timezone-aware or naive.
        If naive, will be interpreted as being in the timezone specified by tz_str.
        If None, gets the latest available data.
    tz_str : str
        IANA timezone name (default: "America/Los_Angeles")
    forecast_hour_offset : int, optional
        If specified along with target_datetime, use this forecast hour offset
        from the model run closest to target_datetime. This allows you to get
        an older model run and then look at its forecast hours.
    """
    if target_datetime is not None:
        # Use the new function to find data for specific datetime
        if forecast_hour_offset is not None:
            print(f"Finding model run closest to: {target_datetime}")
            print(f"Then using forecast hour: {forecast_hour_offset}")
        else:
            print(f"Finding model data for target time: {target_datetime}")
        
        info = compute_file_for_datetime(target_datetime, tz_str=tz_str, 
                                        forecast_hour_override=forecast_hour_offset)
        
        print(f"\nTarget time (local): {info['target_datetime_local']:%Y-%m-%d %H:%M:%S %Z}")
        print(f"Target time (UTC):   {info['target_datetime_utc']:%Y-%m-%d %H:%M:%S %Z}")
        print(f"\nUsing SSCOFS model run:")
        print(f"  Run date: {info['run_date_utc']}")
        print(f"  Cycle: {info['cycle_utc']}")
        print(f"  Cycle start: {info['cycle_start_utc']:%Y-%m-%d %H:%M:%S %Z}")
        print(f"  Forecast hour: {info['forecast_hour_index']:03d}")
        print(f"  URL: {info['url']}")
        print()
    else:
        # Get current time in local timezone
        current_time_utc = dt.datetime.now(timezone.utc)
        try:
            from zoneinfo import ZoneInfo
            current_time_local = current_time_utc.astimezone(ZoneInfo(tz_str))
        except ImportError:
            # Fallback to PST/PDT offset
            current_time_local = current_time_utc - timedelta(hours=8)  # Approximate PST
        
        local_hour = current_time_local.hour * 100 + current_time_local.minute
        
        print(f"Current time (UTC):   {current_time_utc:%Y-%m-%d %H:%M:%S}")
        print(f"Current time (local): {current_time_local:%Y-%m-%d %H:%M:%S}")
        
        # Get the latest cycle info
        info = latest_cycle_and_url_for_local_hour(local_hour, tz_str)
        
        print(f"\nUsing SSCOFS model run:")
        print(f"  Run date: {info['run_date_utc']}")
        print(f"  Cycle: {info['cycle_utc']}")
        print(f"  Forecast hour: {info['forecast_hour_index']:03d}")
        print(f"  URL: {info['url']}")
        print()
    
    # Use shared caching module
    ds = load_sscofs_data(info, use_cache=use_cache, verbose=True)
    
    return ds, info

def plot_currents_at_location(ds, center_lat, center_lon, radius_miles=5, 
                             time_index=0, save_file=None, subsample_n=3,
                             vector_scale_multiplier=10.0):
    """
    Plot current vectors and speed within a radius of a center point.
    Uses UTM projection for accurate scaling in meters.
    
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
    subsample_n : int
        Subsample every nth point for arrows (default 3)
    vector_scale_multiplier : float
        Multiplier for vector lengths (default 10.0). Higher = longer arrows.
    """
    
    # Convert radius from miles to meters
    radius_meters = radius_miles * 1609.34
    
    # Create UTM transformer centered on the location
    transformer, utm_zone, hemisphere = create_utm_transformer(center_lat, center_lon)
    
    print(f"\nUsing UTM Zone {utm_zone}{hemisphere[0].upper()} for plotting")
    
    # Extract surface currents (first sigma layer)
    u = ds["u"].isel(time=time_index, siglay=0)
    v = ds["v"].isel(time=time_index, siglay=0)
    
    # Get coordinates - u,v are on elements (nele), so use lonc, latc
    lons = ds["lonc"].values
    lats = ds["latc"].values
    
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
    mask = distances <= radius_meters
    
    # Apply mask
    x_masked = x_utm[mask]
    y_masked = y_utm[mask]
    u_masked = u.values[mask]
    v_masked = v.values[mask]
    
    # Calculate current speed
    speed_masked = np.sqrt(u_masked**2 + v_masked**2)
    
    # Convert m/s to knots for display
    speed_knots = speed_masked * 1.94384
    
    print(f"\nStatistics for currents within {radius_miles} miles of ({center_lat:.4f}, {center_lon:.4f}):")
    print(f"  Number of points: {len(x_masked)}")
    if len(x_masked) > 0:
        print(f"  Max current speed: {np.nanmax(speed_knots):.2f} knots ({np.nanmax(speed_masked):.3f} m/s)")
        print(f"  Mean current speed: {np.nanmean(speed_knots):.2f} knots ({np.nanmean(speed_masked):.3f} m/s)")
        print(f"  Min current speed: {np.nanmin(speed_knots):.2f} knots ({np.nanmin(speed_masked):.3f} m/s)")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Get time information and convert to local time (Pacific)
    time_val = ds["time"].isel(time=time_index).values
    # Convert numpy datetime64 to datetime
    time_utc = pd.Timestamp(time_val).to_pydatetime()
    try:
        from zoneinfo import ZoneInfo
        time_local = time_utc.replace(tzinfo=timezone.utc).astimezone(ZoneInfo("America/Los_Angeles"))
        time_str = time_local.strftime("%Y-%m-%d %H:%M:%S %Z")
    except ImportError:
        # Fallback for older Python
        time_local = time_utc - timedelta(hours=8)  # Approximate PST
        time_str = time_local.strftime("%Y-%m-%d %H:%M:%S PST")
    
    # ---- Subplot 1: Current vectors ----
    if len(x_masked) > 0:
        # Subsample for arrows
        subsample = slice(None, None, subsample_n)
        x_sub = x_masked[subsample]
        y_sub = y_masked[subsample]
        u_sub = u_masked[subsample]
        v_sub = v_masked[subsample]
        speed_sub = speed_masked[subsample]
        speed_sub_knots = speed_sub * 1.94384  # Convert to knots for display
        
        if len(x_sub) > 1:
            # Calculate typical spacing between adjacent points in meters
            # Use a simple approach: compute distances to nearest neighbors
            from scipy.spatial import cKDTree
            tree = cKDTree(np.column_stack([x_sub, y_sub]))
            # Query 2 nearest neighbors (1st is the point itself, 2nd is nearest neighbor)
            distances_nn, _ = tree.query(np.column_stack([x_sub, y_sub]), k=2)
            typical_spacing = np.median(distances_nn[:, 1])  # Median nearest neighbor distance
            
            print(f"\nVector scaling:")
            print(f"  Typical spacing between arrows: {typical_spacing:.1f} m")
            print(f"  Max current magnitude: {np.max(speed_sub):.3f} m/s")
            print(f"  Vector scale multiplier: {vector_scale_multiplier}x")
            
            # Scale arrows so that a typical current (1 m/s) is about 40% of the spacing
            # This prevents overlap while making them visible
            # For quiver: scale parameter means "data units per arrow length units"
            # We want: arrow_length = velocity / scale
            # Target: 1.0 m/s should produce arrow_length = 0.4 * typical_spacing * multiplier
            # So: 0.4 * typical_spacing * multiplier = 1.0 / scale
            # Therefore: scale = 1.0 / (0.4 * typical_spacing * multiplier) = 2.5 / (typical_spacing * multiplier)
            base_scale = 2.5 / typical_spacing
            scale = base_scale / vector_scale_multiplier
            
            arrow_length_for_1ms = typical_spacing * 0.4 * vector_scale_multiplier
            print(f"  Arrow scale factor: {scale:.6f} (1 m/s = {arrow_length_for_1ms:.1f} m arrow)")
        else:
            scale = 1.0 / 200.0  # Fallback
        
        # Create quiver plot with arrows colored by speed in knots
        q = ax1.quiver(x_sub, y_sub, 
                      u_sub, v_sub,
                      speed_sub_knots,  # Color by speed in knots
                      cmap='viridis',
                      scale=scale,
                      scale_units='xy',
                      angles='xy',
                      width=0.003,
                      headwidth=3,
                      headlength=4,
                      headaxislength=3,
                      alpha=0.9,
                      zorder=5)
        
        # Add colorbar for arrows
        cbar1 = plt.colorbar(q, ax=ax1, label='Current Speed (knots)', pad=0.02)
        
        # Add text showing number of arrows
        ax1.text(0.02, 0.98, f'{len(x_sub)} arrows (every {subsample_n} points)', 
                transform=ax1.transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='white', alpha=0.8))
        
        # Add a reference arrow to show scale
        # Position it in axis coordinates, showing 1 m/s
        ref_speed = 1.0  # m/s
        ref_arrow_length = ref_speed / scale
        ax1.annotate('', xy=(0.85, 0.08), xytext=(0.85 - ref_arrow_length/radius_meters/2, 0.08),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', lw=2, color='red'),
                    fontsize=10)
        ax1.text(0.85, 0.03, f'{ref_speed:.1f} m/s ({ref_speed*1.94384:.1f} knots)', 
                transform=ax1.transAxes, fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Mark center point
        ax1.plot(center_x, center_y, 'r*', markersize=15, label='Center Point', zorder=10)
        
        # Add circle to show radius
        circle = Circle((center_x, center_y), radius_meters,
                       fill=False, edgecolor='red', linewidth=2, 
                       linestyle='--', label=f'{radius_miles} mile radius')
        ax1.add_patch(circle)
    
    ax1.set_xlabel('Easting (m, UTM)')
    ax1.set_ylabel('Northing (m, UTM)')
    ax1.set_title(f'Current Vectors\nTime: {time_str}')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_aspect('equal')
    
    # Set axis limits with margin
    margin = radius_meters * 0.05
    ax1.set_xlim(center_x - radius_meters - margin, center_x + radius_meters + margin)
    ax1.set_ylim(center_y - radius_meters - margin, center_y + radius_meters + margin)
    
    # ---- Subplot 2: Current speed heatmap ----
    if len(x_masked) > 0:
        # Create scatter plot for speed (show in knots for mariners)
        sc = ax2.scatter(x_masked, y_masked, c=speed_knots, 
                        cmap='plasma', s=30, alpha=0.7, edgecolors='k', linewidth=0.3)
        
        # Add colorbar
        cbar2 = plt.colorbar(sc, ax=ax2, label='Current Speed (knots)', pad=0.02)
        
        # Mark center point
        ax2.plot(center_x, center_y, 'r*', markersize=15, label='Center Point', zorder=10)
        
        # Add circle
        circle2 = Circle((center_x, center_y), radius_meters,
                        fill=False, edgecolor='red', linewidth=2, 
                        linestyle='--', label=f'{radius_miles} mile radius')
        ax2.add_patch(circle2)
    
    ax2.set_xlabel('Easting (m, UTM)')
    ax2.set_ylabel('Northing (m, UTM)')
    ax2.set_title(f'Current Speed Distribution\nTime: {time_str}')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_aspect('equal')
    
    # Set same axis limits
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(ax1.get_ylim())
    
    plt.suptitle(f'SSCOFS Surface Currents - Puget Sound\nLocation: ({center_lat:.4f}°N, {abs(center_lon):.4f}°W) - UTM Zone {utm_zone}{hemisphere[0].upper()}', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_file:
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_file}")
    
    plt.show()
    
    return fig

def main():
    parser = argparse.ArgumentParser(
        description="Download and plot SSCOFS currents within a radius of a GPS coordinate"
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
        help="Save plot to file (e.g., 'currents.png')"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Don't use cached data, always download fresh"
    )
    parser.add_argument(
        "--list-cache",
        action="store_true",
        help="List cached files and exit"
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear all cached files and exit"
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=3,
        help="Subsample every nth point for arrows (default: 3)"
    )
    parser.add_argument(
        "--vector-scale",
        type=float,
        default=10.0,
        help="Multiplier for vector lengths (default: 10.0). Higher = longer arrows."
    )
    parser.add_argument(
        "--datetime",
        type=str,
        help="Target date and time in format 'YYYY-MM-DD HH:MM' in local timezone. "
             "If not provided, uses current time. Example: '2025-10-15 14:30'"
    )
    parser.add_argument(
        "--forecast-hour-offset",
        type=int,
        help="When used with --datetime, get the model run closest to that datetime, "
             "then use this forecast hour offset (0-72). Example: --datetime '2025-10-15 20:00' "
             "--forecast-hour-offset 26 gets the model run at 20:00 and its 26-hour forecast."
    )
    parser.add_argument(
        "--timezone",
        type=str,
        default="America/Los_Angeles",
        help="IANA timezone name for --datetime (default: America/Los_Angeles)"
    )
    
    args = parser.parse_args()
    
    # Handle cache management commands
    if args.list_cache:
        list_cache()
        return 0
    
    if args.clear_cache:
        clear_cache()
        return 0
    
    print(f"Plotting currents within {args.radius} miles of ({args.lat}, {args.lon})")
    
    # Parse target datetime if provided
    target_datetime = None
    if args.datetime:
        try:
            # Parse the datetime string
            target_datetime = dt.datetime.strptime(args.datetime, "%Y-%m-%d %H:%M")
            # Make it timezone-aware
            try:
                from zoneinfo import ZoneInfo
                target_datetime = target_datetime.replace(tzinfo=ZoneInfo(args.timezone))
            except ImportError:
                # Fallback: leave as naive, will be handled by compute_file_for_datetime
                pass
            print(f"Parsed target datetime: {target_datetime}")
        except ValueError as e:
            print(f"Error parsing datetime: {e}")
            print("Expected format: 'YYYY-MM-DD HH:MM' (e.g., '2025-10-15 14:30')")
            return 1
    
    try:
        # Get the data (use cache unless --no-cache is specified)
        ds, info = get_latest_current_data(
            use_cache=not args.no_cache, 
            target_datetime=target_datetime,
            tz_str=args.timezone,
            forecast_hour_offset=args.forecast_hour_offset
        )
        
        # Plot the currents
        fig = plot_currents_at_location(
            ds, 
            args.lat, 
            args.lon, 
            radius_miles=args.radius,
            time_index=args.time_index,
            save_file=args.save,
            subsample_n=args.subsample,
            vector_scale_multiplier=args.vector_scale
        )
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
