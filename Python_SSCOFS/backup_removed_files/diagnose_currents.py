"""
Diagnostic script to examine current data in detail
"""
import argparse
import xarray as xr
import numpy as np
from latest_cycle import latest_cycle_and_url_for_local_hour
from sscofs_cache import load_sscofs_data, list_cache, clear_cache
import datetime as dt
from datetime import timezone
from zoneinfo import ZoneInfo

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in miles"""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 3959.0  # miles
    return c * r

def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic script to examine SSCOFS current data in detail"
    )
    parser.add_argument(
        "--lat", type=float, default=47.67181822458632,
        help="Latitude of center point (default: 47.6718)"
    )
    parser.add_argument(
        "--lon", type=float, default=-122.4583957143628,
        help="Longitude of center point (default: -122.4584)"
    )
    parser.add_argument(
        "--radius", type=float, default=5.0,
        help="Radius in miles (default: 5)"
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Don't use cached data, always download fresh"
    )
    parser.add_argument(
        "--list-cache", action="store_true",
        help="List cached files and exit"
    )
    parser.add_argument(
        "--clear-cache", action="store_true",
        help="Clear all cached files and exit"
    )
    
    args = parser.parse_args()
    
    # Handle cache management commands first
    if args.list_cache:
        list_cache()
        return 0
    
    if args.clear_cache:
        clear_cache()
        return 0
    
    # Get parameters from args
    center_lat = args.lat
    center_lon = args.lon
    radius_miles = args.radius
    
    # Get current time
    current_time_utc = dt.datetime.now(timezone.utc)
    current_time_local = current_time_utc.astimezone(ZoneInfo("America/Los_Angeles"))
    local_hour = current_time_local.hour * 100 + current_time_local.minute
    
    # Get the latest cycle info
    info = latest_cycle_and_url_for_local_hour(local_hour, "America/Los_Angeles")
    
    print(f"Current time (UTC):   {current_time_utc:%Y-%m-%d %H:%M:%S}")
    print(f"Current time (local): {current_time_local:%Y-%m-%d %H:%M:%S}")
    print(f"\nUsing SSCOFS model run:")
    print(f"  Run date: {info['run_date_utc']}")
    print(f"  Cycle: {info['cycle_utc']}")
    print(f"  Forecast hour: {info['forecast_hour_index']:03d}")
    print(f"  URL: {info['url']}")
    
    # Load data using shared cache
    print()
    ds = load_sscofs_data(info, use_cache=not args.no_cache, verbose=True)
    
    # Now examine the dataset
    with ds:
        
        print("\n" + "="*70)
        print("DATASET STRUCTURE")
        print("="*70)
        print(f"Dimensions: {dict(ds.dims)}")
        print(f"\nTime values: {ds.time.values}")
        
        # Extract currents
        print("\n" + "="*70)
        print("EXAMINING U and V VARIABLES")
        print("="*70)
        u = ds["u"].isel(time=0, siglay=0)
        v = ds["v"].isel(time=0, siglay=0)
        
        print(f"u shape: {u.shape}")
        print(f"v shape: {v.shape}")
        print(f"u dimensions: {u.dims}")
        print(f"v dimensions: {v.dims}")
        
        # Get coordinates
        lons = ds["lonc"].values
        lats = ds["latc"].values
        
        print(f"\nCoordinate arrays shape: lon={lons.shape}, lat={lats.shape}")
        print(f"Longitude range: {lons.min():.4f} to {lons.max():.4f}")
        print(f"Latitude range: {lats.min():.4f} to {lats.max():.4f}")
        
        # Statistics on full domain
        u_vals = u.values
        v_vals = v.values
        speed = np.sqrt(u_vals**2 + v_vals**2)
        speed_knots = speed * 1.94384
        
        print("\n" + "="*70)
        print("FULL DOMAIN STATISTICS")
        print("="*70)
        print(f"Total grid points: {len(u_vals)}")
        print(f"u - min: {np.nanmin(u_vals):.6f}, max: {np.nanmax(u_vals):.6f}, mean: {np.nanmean(u_vals):.6f} m/s")
        print(f"v - min: {np.nanmin(v_vals):.6f}, max: {np.nanmax(v_vals):.6f}, mean: {np.nanmean(v_vals):.6f} m/s")
        print(f"speed - min: {np.nanmin(speed_knots):.6f}, max: {np.nanmax(speed_knots):.6f}, mean: {np.nanmean(speed_knots):.6f} knots")
        print(f"Non-zero currents: {np.count_nonzero(speed > 0.001)} ({100*np.count_nonzero(speed > 0.001)/len(speed):.1f}%)")
        
        # Now filter by location
        print("\n" + "="*70)
        print(f"FILTERING TO {radius_miles} MILES OF TARGET LOCATION")
        print("="*70)
        print(f"Center: ({center_lat:.4f}, {center_lon:.4f})")
        
        distances = haversine_distance(center_lat, center_lon, lats, lons)
        mask = distances <= radius_miles
        
        print(f"Points within radius: {np.sum(mask)}")
        
        if np.sum(mask) > 0:
            lons_masked = lons[mask]
            lats_masked = lats[mask]
            u_masked = u_vals[mask]
            v_masked = v_vals[mask]
            speed_masked = speed[mask]
            speed_knots_masked = speed_knots[mask]
            distances_masked = distances[mask]
            
            print(f"\nLon range in area: {lons_masked.min():.4f} to {lons_masked.max():.4f}")
            print(f"Lat range in area: {lats_masked.min():.4f} to {lats_masked.max():.4f}")
            print(f"Distance range: {distances_masked.min():.2f} to {distances_masked.max():.2f} miles")
            
            print("\n" + "="*70)
            print("LOCAL AREA STATISTICS")
            print("="*70)
            print(f"u - min: {np.nanmin(u_masked):.6f}, max: {np.nanmax(u_masked):.6f}, mean: {np.nanmean(u_masked):.6f} m/s")
            print(f"v - min: {np.nanmin(v_masked):.6f}, max: {np.nanmax(v_masked):.6f}, mean: {np.nanmean(v_masked):.6f} m/s")
            print(f"speed - min: {np.nanmin(speed_knots_masked):.6f}, max: {np.nanmax(speed_knots_masked):.6f}, mean: {np.nanmean(speed_knots_masked):.6f} knots")
            print(f"Non-zero currents: {np.count_nonzero(speed_masked > 0.001)} ({100*np.count_nonzero(speed_masked > 0.001)/len(speed_masked):.1f}%)")
            
            # Show some sample values
            print("\n" + "="*70)
            print("SAMPLE DATA POINTS (first 10 in area)")
            print("="*70)
            print(f"{'Lon':>10s} {'Lat':>10s} {'Dist(mi)':>10s} {'u(m/s)':>10s} {'v(m/s)':>10s} {'Speed(kt)':>10s}")
            print("-"*70)
            for i in range(min(10, len(lons_masked))):
                print(f"{lons_masked[i]:10.4f} {lats_masked[i]:10.4f} {distances_masked[i]:10.2f} "
                      f"{u_masked[i]:10.4f} {v_masked[i]:10.4f} {speed_knots_masked[i]:10.4f}")
            
            # Show highest speed points
            if np.nanmax(speed_knots_masked) > 0:
                print("\n" + "="*70)
                print("TOP 10 HIGHEST SPEED POINTS")
                print("="*70)
                print(f"{'Lon':>10s} {'Lat':>10s} {'Dist(mi)':>10s} {'u(m/s)':>10s} {'v(m/s)':>10s} {'Speed(kt)':>10s}")
                print("-"*70)
                sorted_idx = np.argsort(speed_knots_masked)[::-1][:10]
                for i in sorted_idx:
                    print(f"{lons_masked[i]:10.4f} {lats_masked[i]:10.4f} {distances_masked[i]:10.2f} "
                          f"{u_masked[i]:10.4f} {v_masked[i]:10.4f} {speed_knots_masked[i]:10.4f}")
        else:
            print("ERROR: No points found in the specified area!")
            print("\nClosest point to target:")
            closest_idx = np.argmin(distances)
            print(f"  Distance: {distances[closest_idx]:.2f} miles")
            print(f"  Location: ({lats[closest_idx]:.4f}, {lons[closest_idx]:.4f})")
            print(f"  Speed: {speed_knots[closest_idx]:.4f} knots")
    
    return 0


if __name__ == "__main__":
    exit(main())

