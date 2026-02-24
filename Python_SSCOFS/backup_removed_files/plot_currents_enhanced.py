"""
plot_currents_enhanced.py
-------------------------

Enhanced current visualization with tactical racing features:
- Speed contours and streamlines for flow structure
- Adaptive quiver placement (denser at fronts/shears)
- Vorticity, divergence, and Okubo-Weiss diagnostics
- Optional basemap support (Natural Earth or contextily)

Based on suggestions from UPDATES.md for tactical sailing applications.

Usage:
    python plot_currents_enhanced.py --lat 47.67181822458632 --lon -122.4583957143628 --radius 5
    python plot_currents_enhanced.py --style streamline  # streamlines only
    python plot_currents_enhanced.py --style adaptive    # adaptive quiver
    python plot_currents_enhanced.py --style diagnostic  # with vorticity/OW overlays
    python plot_currents_enhanced.py --basemap contextily  # add web tile basemap
"""

import argparse
import datetime as dt
from datetime import timezone, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import xarray as xr
from matplotlib.patches import Circle
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
from pathlib import Path
from pyproj import Transformer

# Optional basemap dependencies
try:
    import contextily as ctx
    HAS_CONTEXTILY = True
except ImportError:
    HAS_CONTEXTILY = False

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

# Import helper functions from existing modules
from latest_cycle import latest_cycle_and_url_for_local_hour
from fetch_sscofs import build_sscofs_url
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
    hemisphere = 'north' if center_lat >= 0 else 'south'
    
    # For Puget Sound, we know it's UTM Zone 10N (EPSG:32610)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True)
    
    return transformer, utm_zone, hemisphere

def add_basemap(ax, basemap_type='contextily', zoom='auto', alpha=0.5):
    """
    Add a basemap to the axis.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis to add the basemap to (must be in UTM Zone 10N coords)
    basemap_type : str
        'contextily' - web tiles (requires reprojection to EPSG:3857)
        'natural_earth' - coastline shapefile (requires geopandas and data file)
        'none' - no basemap
    zoom : str or int
        Zoom level for contextily ('auto' or 1-18)
    alpha : float
        Transparency of basemap (0=transparent, 1=opaque)
    
    Returns:
    --------
    bool : True if basemap was added successfully
    """
    if not basemap_type or basemap_type == 'none':
        return False
    
    if basemap_type == 'contextily':
        if not HAS_CONTEXTILY:
            print("Warning: contextily not installed. Install with: pip install contextily")
            return False
        
        try:
            # Get current axis limits in UTM (EPSG:32610)
            xlim_utm = ax.get_xlim()
            ylim_utm = ax.get_ylim()
            
            # Transform to Web Mercator (EPSG:3857) for tiles
            transformer_to_3857 = Transformer.from_crs(
                "EPSG:32610", "EPSG:3857", always_xy=True
            )
            
            # Transform plot data to Web Mercator temporarily
            # contextily will handle the coordinate transformation
            # Using OpenStreetMap for better land/water contrast
            ctx.add_basemap(
                ax, 
                crs="EPSG:32610",  # Tell contextily our current CRS
                source=ctx.providers.OpenStreetMap.Mapnik,  # Better land/water contrast
                zoom=zoom,
                alpha=alpha,
                attribution=""  # Clean plot for racing
            )
            
            print(f"  Added basemap: OpenStreetMap (zoom={zoom}, alpha={alpha})")
            return True
            
        except Exception as e:
            print(f"Warning: Could not add contextily basemap: {e}")
            return False
    
    elif basemap_type == 'natural_earth':
        if not HAS_GEOPANDAS:
            print("Warning: geopandas not installed. Install with: conda install geopandas")
            return False
        
        # Look for Natural Earth or Puget Sound shoreline data
        data_dir = Path(__file__).parent / "data"
        possible_files = [
            data_dir / "shoreline_puget.geojson",
            data_dir / "ne_10m_coastline.shp",
            data_dir / "coastline.geojson",
        ]
        
        shoreline_file = None
        for f in possible_files:
            if f.exists():
                shoreline_file = f
                break
        
        if shoreline_file is None:
            print(f"Warning: No shoreline data found in {data_dir}")
            print("Download Natural Earth coastlines or create data/shoreline_puget.geojson")
            return False
        
        try:
            # Load and reproject to UTM Zone 10N
            shore = gpd.read_file(shoreline_file).to_crs("EPSG:32610")
            shore.boundary.plot(ax=ax, linewidth=0.8, color='#555555', zorder=1)
            print(f"  Added basemap: {shoreline_file.name}")
            return True
            
        except Exception as e:
            print(f"Warning: Could not load shoreline data: {e}")
            return False
    
    else:
        print(f"Warning: Unknown basemap type '{basemap_type}'")
        return False

def get_latest_current_data(use_cache=True):
    """
    Get the latest SSCOFS current data using the current time.
    Returns the dataset and metadata about the run.
    """
    current_time_utc = dt.datetime.now(timezone.utc)
    try:
        from zoneinfo import ZoneInfo
        current_time_local = current_time_utc.astimezone(ZoneInfo("America/Los_Angeles"))
    except ImportError:
        current_time_local = current_time_utc - timedelta(hours=8)
    
    local_hour = current_time_local.hour * 100 + current_time_local.minute
    
    print(f"Current time (UTC):   {current_time_utc:%Y-%m-%d %H:%M:%S}")
    print(f"Current time (local): {current_time_local:%Y-%m-%d %H:%M:%S}")
    
    info = latest_cycle_and_url_for_local_hour(local_hour, "America/Los_Angeles")
    
    print(f"\nUsing SSCOFS model run:")
    print(f"  Run date: {info['run_date_utc']}")
    print(f"  Cycle: {info['cycle_utc']}")
    print(f"  Forecast hour: {info['forecast_hour_index']:03d}")
    print(f"  URL: {info['url']}")
    print()
    
    ds = load_sscofs_data(info, use_cache=use_cache, verbose=True)
    
    return ds, info

def interpolate_to_grid(x_utm, y_utm, u, v, x0, y0, R_utm, nx=300, ny=300, pad_factor=0.1):
    """
    Interpolate unstructured u,v data to a regular grid for visualization.
    
    Parameters:
    -----------
    x_utm, y_utm : array
        UTM coordinates of data points
    u, v : array
        Velocity components (m/s)
    x0, y0 : float
        Center point in UTM
    R_utm : float
        Radius in meters
    nx, ny : int
        Grid resolution
    pad_factor : float
        Padding around the circle as fraction of radius
    
    Returns:
    --------
    xg, yg : 1D arrays
        Grid coordinates
    Xg, Yg : 2D arrays
        Meshgrid
    Ug, Vg, Sg : 2D arrays
        Interpolated u, v velocities and speed (with NaN mask applied)
    """
    pad = R_utm * pad_factor
    xmin, xmax = x0 - R_utm - pad, x0 + R_utm + pad
    ymin, ymax = y0 - R_utm - pad, y0 + R_utm + pad
    
    xg = np.linspace(xmin, xmax, nx)
    yg = np.linspace(ymin, ymax, ny)
    Xg, Yg = np.meshgrid(xg, yg)
    
    # Triangulate and interpolate
    tri = mtri.Triangulation(x_utm, y_utm)
    interp_u = mtri.LinearTriInterpolator(tri, u)
    interp_v = mtri.LinearTriInterpolator(tri, v)
    
    Ug = interp_u(Xg, Yg).filled(np.nan)
    Vg = interp_v(Xg, Yg).filled(np.nan)
    Sg = np.hypot(Ug, Vg)
    
    # Mask land/NaN
    mask = np.isnan(Sg)
    Ug[mask] = np.nan
    Vg[mask] = np.nan
    Sg[mask] = np.nan
    
    return xg, yg, Xg, Yg, Ug, Vg, Sg

def compute_flow_diagnostics(Ug, Vg, xg, yg):
    """
    Compute flow diagnostics: vorticity, divergence, and Okubo-Weiss parameter.
    
    Parameters:
    -----------
    Ug, Vg : 2D arrays
        Velocity components on regular grid
    xg, yg : 1D arrays
        Grid coordinates
    
    Returns:
    --------
    zeta : 2D array
        Vorticity (s^-1)
    div : 2D array
        Divergence (s^-1)
    OW : 2D array
        Okubo-Weiss parameter (s^-2)
    """
    # Compute derivatives (note: np.gradient order is rows->y, cols->x)
    Vy, Vx = np.gradient(Vg, yg, xg)
    Uy, Ux = np.gradient(Ug, yg, xg)
    
    # Vorticity: ζ = ∂v/∂x - ∂u/∂y
    zeta = Vx - Uy
    
    # Divergence: δ = ∂u/∂x + ∂v/∂y
    div = Ux + Vy
    
    # Strain components
    sn = Ux - Vy  # normal strain
    ss = Vx + Uy  # shear strain
    
    # Okubo-Weiss: W = s_n² + s_s² - ζ²
    # Negative W → eddy-dominated (vorticity wins)
    # Positive W → strain-dominated (fronts/shear)
    OW = sn**2 + ss**2 - zeta**2
    
    return zeta, div, OW

def adaptive_quiver_points(Xg, Yg, Ug, Vg, Sg, xg, yg, percentile=70, min_pix=5):
    """
    Select quiver arrow positions adaptively based on speed gradient magnitude.
    Places more arrows where the flow structure changes rapidly.
    
    Parameters:
    -----------
    Xg, Yg, Ug, Vg, Sg : 2D arrays
        Grid and velocity fields
    xg, yg : 1D arrays
        Grid coordinates
    percentile : float
        Percentile threshold for gradient magnitude (default 70 = top 30%)
    min_pix : int
        Minimum pixel spacing between arrows
    
    Returns:
    --------
    qx, qy, qu, qv, qs : 1D arrays
        Arrow positions, velocities, and speeds
    """
    # Compute gradient magnitude of speed
    Sy, Sx = np.gradient(Sg, yg, xg)
    W = np.hypot(Sx, Sy)
    
    mask = np.isnan(Sg)
    
    # Build sampling mask using gradient threshold
    thr = np.nanpercentile(W, percentile)
    candidates = np.argwhere((~mask) & (W >= thr))
    
    # Greedy min-distance thinning
    keep = []
    for iy, ix in candidates:
        if all((abs(iy - ky) > min_pix) or (abs(ix - kx) > min_pix) for ky, kx in keep):
            keep.append((iy, ix))
    
    if len(keep) == 0:
        # Fallback: just take every min_pix points
        iy_idx = np.arange(0, len(yg), min_pix*2)
        ix_idx = np.arange(0, len(xg), min_pix*2)
        for iy in iy_idx:
            for ix in ix_idx:
                if iy < Sg.shape[0] and ix < Sg.shape[1] and not mask[iy, ix]:
                    keep.append((iy, ix))
    
    qy = np.array([yg[iy] for iy, ix in keep])
    qx = np.array([xg[ix] for iy, ix in keep])
    qu = np.array([Ug[iy, ix] for iy, ix in keep])
    qv = np.array([Vg[iy, ix] for iy, ix in keep])
    qs = np.hypot(qu, qv)
    
    return qx, qy, qu, qv, qs

def plot_currents_enhanced(ds, center_lat, center_lon, radius_miles=5, 
                          time_index=0, save_file=None, 
                          style='streamline', show_diagnostics=False,
                          basemap=None):
    """
    Enhanced current plotting with tactical racing features.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        SSCOFS dataset
    center_lat, center_lon : float
        Center coordinates in decimal degrees
    radius_miles : float
        Radius in miles to plot around center
    time_index : int
        Which time step to plot
    save_file : str, optional
        If provided, save the plot to this file
    style : str
        'streamline' - contours + streamlines
        'adaptive' - contours + adaptive quiver
        'both' - streamlines + adaptive quiver
        'diagnostic' - include vorticity/OW overlays
    show_diagnostics : bool
        If True, add vorticity and Okubo-Weiss overlays
    basemap : str, optional
        'contextily' - use web tiles (requires projection to EPSG:3857)
        None - no basemap (default)
    """
    
    # Convert radius from miles to meters
    radius_meters = radius_miles * 1609.34
    
    # Create UTM transformer
    transformer, utm_zone, hemisphere = create_utm_transformer(center_lat, center_lon)
    print(f"\nUsing UTM Zone {utm_zone}{hemisphere[0].upper()} (EPSG:32610) for plotting")
    
    # Extract surface currents (first sigma layer)
    u = ds["u"].isel(time=time_index, siglay=0)
    v = ds["v"].isel(time=time_index, siglay=0)
    
    # Get coordinates (u,v are on elements, so use lonc, latc)
    lons = ds["lonc"].values
    lats = ds["latc"].values
    
    # Convert longitudes from 0-360 to -180 to 180 if needed
    if lons.max() > 180:
        lons = np.where(lons > 180, lons - 360, lons)
    
    # Transform to UTM
    x_utm, y_utm = transformer.transform(lons, lats)
    center_x, center_y = transformer.transform(center_lon, center_lat)
    
    # Mask to radius
    distances = np.sqrt((x_utm - center_x)**2 + (y_utm - center_y)**2)
    mask = distances <= radius_meters * 1.2  # Slightly larger for interpolation
    
    x_masked = x_utm[mask]
    y_masked = y_utm[mask]
    u_masked = u.values[mask]
    v_masked = v.values[mask]
    
    print(f"\nStatistics for currents within {radius_miles} miles:")
    print(f"  Number of data points: {len(x_masked)}")
    
    if len(x_masked) < 10:
        print("Warning: Too few points for interpolation")
        return None
    
    # Interpolate to regular grid
    print("\nInterpolating to regular grid...")
    xg, yg, Xg, Yg, Ug, Vg, Sg = interpolate_to_grid(
        x_masked, y_masked, u_masked, v_masked,
        center_x, center_y, radius_meters,
        nx=300, ny=300
    )
    
    # Convert speed to knots
    Sg_knots = Sg * 1.94384
    
    print(f"  Max current speed: {np.nanmax(Sg_knots):.2f} knots ({np.nanmax(Sg):.3f} m/s)")
    print(f"  Mean current speed: {np.nanmean(Sg_knots):.2f} knots ({np.nanmean(Sg):.3f} m/s)")
    
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
    
    # Determine number of subplots
    if show_diagnostics or style == 'diagnostic':
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        ax1, ax2 = axes
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 10))
        ax2 = None
    
    # Choose colormap - try cmocean if available, else viridis
    try:
        import cmocean
        cmap = cmocean.cm.speed
    except ImportError:
        cmap = get_cmap('viridis')
    
    # Normalize speed for consistent coloring
    vmin, vmax = np.nanpercentile(Sg_knots, [5, 97])
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # ---- Main plot: Speed contours + streamlines/quiver ----
    
    # 0) Add basemap first (background layer)
    if basemap:
        # Set axis limits first so basemap knows what to fetch
        margin = radius_meters * 0.05
        ax1.set_xlim(center_x - radius_meters - margin, center_x + radius_meters + margin)
        ax1.set_ylim(center_y - radius_meters - margin, center_y + radius_meters + margin)
        ax1.set_aspect('equal')
        
        add_basemap(ax1, basemap_type=basemap, zoom='auto', alpha=0.5)
    
    # 1) Filled contours for speed bands
    levels = np.linspace(vmin, vmax, 20)
    cs = ax1.contourf(Xg, Yg, Sg_knots, levels=levels, cmap=cmap, norm=norm, alpha=0.7)
    
    # 2) Add flow visualization based on style
    if style in ['streamline', 'both', 'diagnostic']:
        print("Adding streamlines...")
        # Variable linewidth based on speed
        lw_base = 0.5
        lw_var = 2.0 * (Sg - np.nanmin(Sg)) / (np.nanmax(Sg) - np.nanmin(Sg) + 1e-10)
        lw = lw_base + lw_var
        
        # Streamplot
        strm = ax1.streamplot(
            xg, yg, Ug, Vg,
            color=Sg_knots, cmap=cmap, norm=norm,
            density=1.5, minlength=0.2, arrowsize=1.0, 
            linewidth=lw, broken_streamlines=False,
            zorder=3
        )
    
    if style in ['adaptive', 'both', 'diagnostic']:
        print("Computing adaptive quiver placement...")
        qx, qy, qu, qv, qs = adaptive_quiver_points(Xg, Yg, Ug, Vg, Sg, xg, yg,
                                                     percentile=70, min_pix=8)
        qs_knots = qs * 1.94384
        
        print(f"  Placing {len(qx)} arrows at high-gradient locations")
        
        # Adaptive quiver
        q = ax1.quiver(qx, qy, qu, qv, qs_knots,
                      cmap=cmap, norm=norm,
                      scale=25, width=0.003,
                      headwidth=3, headlength=4, headaxislength=3,
                      alpha=0.8, zorder=5)
    
    # 3) Tactical speed contours (0.5, 1.0, 1.5 knots)
    tactical_levels = [0.5, 1.0, 1.5, 2.0]
    ax1.contour(Xg, Yg, Sg_knots, levels=tactical_levels, 
               colors='white', linewidths=1.2, alpha=0.6, linestyles='-')
    
    # Label the tactical contours
    ax1.text(0.02, 0.98, 'White contours: 0.5, 1.0, 1.5, 2.0 knots',
            transform=ax1.transAxes, fontsize=9, color='white',
            verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))
    
    # 4) Add center and radius
    ax1.plot(center_x, center_y, 'r*', markersize=15, label='Center', zorder=10)
    circle = Circle((center_x, center_y), radius_meters,
                   fill=False, edgecolor='red', linewidth=2, 
                   linestyle='--', label=f'{radius_miles} mi radius', zorder=10)
    ax1.add_patch(circle)
    
    # 5) Colorbar
    cbar = plt.colorbar(cs, ax=ax1, label='Current Speed (knots)', pad=0.02)
    
    # 6) Formatting
    ax1.set_xlabel('Easting (m, UTM Zone 10N)', fontsize=11)
    ax1.set_ylabel('Northing (m, UTM Zone 10N)', fontsize=11)
    ax1.set_title(f'Surface Currents - {style.capitalize()} View\n{time_str}', 
                 fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_aspect('equal')
    
    # Set axis limits (if not already set by basemap)
    if not basemap:
        margin = radius_meters * 0.05
        ax1.set_xlim(center_x - radius_meters - margin, center_x + radius_meters + margin)
        ax1.set_ylim(center_y - radius_meters - margin, center_y + radius_meters + margin)
    
    # ---- Diagnostic plot: Vorticity + Okubo-Weiss ----
    if ax2 is not None:
        print("Computing flow diagnostics (vorticity, divergence, Okubo-Weiss)...")
        zeta, div, OW = compute_flow_diagnostics(Ug, Vg, xg, yg)
        
        # Add basemap for diagnostic panel too
        if basemap:
            ax2.set_xlim(ax1.get_xlim())
            ax2.set_ylim(ax1.get_ylim())
            ax2.set_aspect('equal')
            add_basemap(ax2, basemap_type=basemap, zoom='auto', alpha=0.5)
        
        # Background: speed contours (lighter)
        ax2.contourf(Xg, Yg, Sg_knots, levels=levels, cmap=cmap, norm=norm, alpha=0.4)
        
        # Vorticity contours (diverging colormap)
        zeta_levels = np.linspace(np.nanpercentile(zeta, 5), 
                                 np.nanpercentile(zeta, 95), 9)
        vort_cs = ax2.contour(Xg, Yg, zeta, levels=zeta_levels,
                             cmap='PuOr', alpha=0.7, linewidths=1.5)
        ax2.clabel(vort_cs, inline=True, fontsize=8, fmt='%0.1e')
        
        # Okubo-Weiss zero contour (boundary between strain/eddy regimes)
        ax2.contour(Xg, Yg, OW, levels=[0], colors='black', 
                   linewidths=2, linestyles='--', 
                   label='OW=0 (eddy/strain boundary)')
        
        # Mark eddy cores (OW < 0, strong vorticity)
        eddy_mask = OW < np.nanpercentile(OW, 20)
        if np.any(eddy_mask):
            ax2.contourf(Xg, Yg, eddy_mask.astype(float), 
                        levels=[0.5, 1.5], colors=['cyan'], alpha=0.2)
        
        # Add annotations
        ax2.text(0.02, 0.98, 
                'Purple/Orange: Vorticity contours\n'
                'Black dashed: OW=0 (eddy/strain)\n'
                'Cyan regions: Eddy cores (OW<0)',
                transform=ax2.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Center and radius
        ax2.plot(center_x, center_y, 'r*', markersize=15, zorder=10)
        circle2 = Circle((center_x, center_y), radius_meters,
                        fill=False, edgecolor='red', linewidth=2, 
                        linestyle='--', zorder=10)
        ax2.add_patch(circle2)
        
        # Formatting
        ax2.set_xlabel('Easting (m, UTM Zone 10N)', fontsize=11)
        ax2.set_ylabel('Northing (m, UTM Zone 10N)', fontsize=11)
        ax2.set_title(f'Flow Diagnostics (Vorticity & Okubo-Weiss)\n{time_str}',
                     fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax2.set_aspect('equal')
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_ylim(ax1.get_ylim())
    
    # Overall title
    plt.suptitle(
        f'SSCOFS Surface Currents - Enhanced Tactical View\n'
        f'Location: ({center_lat:.4f}°N, {abs(center_lon):.4f}°W)',
        fontsize=14, fontweight='bold', y=0.995
    )
    
    plt.tight_layout()
    
    if save_file:
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_file}")
    
    plt.show()
    
    return fig

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced SSCOFS current visualization with tactical racing features"
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
        "--time-index", type=int, default=0,
        help="Time index to plot (default: 0)"
    )
    parser.add_argument(
        "--style", type=str, default='streamline',
        choices=['streamline', 'adaptive', 'both', 'diagnostic'],
        help="Visualization style (default: streamline)"
    )
    parser.add_argument(
        "--diagnostics", action="store_true",
        help="Show vorticity and Okubo-Weiss diagnostics"
    )
    parser.add_argument(
        "--basemap", type=str, default=None,
        choices=['contextily', 'natural_earth', 'none', None],
        help="Add basemap: 'contextily' (web tiles, needs pip install contextily), "
             "'natural_earth' (local shapefile), or 'none' (default: none)"
    )
    parser.add_argument(
        "--save", type=str,
        help="Save plot to file (e.g., 'currents_enhanced.png')"
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Don't use cached data"
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
    
    # Handle cache management
    if args.list_cache:
        list_cache()
        return 0
    
    if args.clear_cache:
        clear_cache()
        return 0
    
    print(f"Enhanced current plot within {args.radius} miles of ({args.lat}, {args.lon})")
    print(f"Style: {args.style}")
    if args.basemap:
        print(f"Basemap: {args.basemap}")
    
    try:
        # Get latest data
        ds, info = get_latest_current_data(use_cache=not args.no_cache)
        
        # Create enhanced plot
        fig = plot_currents_enhanced(
            ds, 
            args.lat, 
            args.lon, 
            radius_miles=args.radius,
            time_index=args.time_index,
            save_file=args.save,
            style=args.style,
            show_diagnostics=args.diagnostics or (args.style == 'diagnostic'),
            basemap=args.basemap
        )
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

