"""
plot_currents_simple.py
-----------------------

Simple current visualization starting with interpolation basics.
Display u and v components side by side to verify the interpolation works correctly.

Key features:
- Area-weighted interpolation from cells to nodes (respects wet/dry boundaries)
- Masked triangulation that excludes dry cells
- Land/water mask for proper rendering of coastlines
- Only interpolates over water areas (land = NaN)
- Quiver arrows showing current direction and magnitude
- Optional basemap support (contextily web tiles or local shapefiles)

The 'inside' mask identifies valid water grid points.
The land_mask (~inside) can be used to overlay basemaps, coastlines, or land features.

Usage:
    python plot_currents_simple.py
    python plot_currents_simple.py --lat 47.67 --lon -122.46 --radius 5
    python plot_currents_simple.py --basemap contextily --arrow-scale 2.0 --quiver-spacing 50
"""

import argparse
import datetime as dt
from datetime import timezone, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import xarray as xr
from pathlib import Path
from pyproj import Transformer

# Import helper functions from existing modules
from latest_cycle import latest_cycle_and_url_for_local_hour
from sscofs_cache import load_sscofs_data, list_cache, clear_cache
from basemap_utils import add_basemap

def create_utm_transformer(center_lat, center_lon):
    """
    Create a transformer for converting lat/lon to UTM coordinates.
    For Puget Sound, we use UTM Zone 10N (EPSG:32610).
    """
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True)
    return transformer

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

def triangle_areas(x, y, tri_idx):
    """Compute areas of triangles given vertices."""
    p0, p1, p2 = tri_idx[:,0], tri_idx[:,1], tri_idx[:,2]
    x0, y0 = x[p0], y[p0]
    x1, y1 = x[p1], y[p1]
    x2, y2 = x[p2], y[p2]
    return 0.5 * np.abs((x1-x0)*(y2-y0) - (x2-x0)*(y1-y0))

def cells_to_nodes_area_weighted(x, y, tri_idx, u_cell, wet_cells):
    """
    Move cell-centered values to nodes by area-weighted averaging of *wet* neighboring cells.
    Returns node arrays with NaN for nodes with no wet neighbors.
    """
    nele = tri_idx.shape[0]
    nn = x.size
    A = triangle_areas(x, y, tri_idx)  # (nele,)
    
    # Only use wet cells: weight = area if wet else 0
    wA = (A * wet_cells).astype(float)
    
    accU = np.zeros(nn, dtype=float)
    wsum = np.zeros(nn, dtype=float)
    
    for k in range(3):  # scatter-add to each vertex
        i = tri_idx[:, k]
        np.add.at(accU, i, u_cell * wA)
        np.add.at(wsum, i, wA)
    
    u_node = accU / np.where(wsum > 0, wsum, np.nan)
    return u_node

def interpolate_to_grid(x_nodes, y_nodes, tri_idx, u_cells, v_cells, wet_cells, 
                       x0, y0, R_utm, nx=300, ny=300):
    """
    Interpolate unstructured u,v data to a regular grid, respecting wet/dry boundaries.
    
    Parameters:
    -----------
    x_nodes, y_nodes : array
        UTM coordinates of nodes
    tri_idx : array (nele, 3)
        Triangle connectivity (0-based indexing)
    u_cells, v_cells : array (nele,)
        Velocity components at cell centers (m/s)
    wet_cells : array (nele,)
        Boolean mask for wet cells
    x0, y0 : float
        Center point in UTM
    R_utm : float
        Radius in meters
    nx, ny : int
        Grid resolution
    
    Returns:
    --------
    xg, yg : 1D arrays
        Grid coordinates
    Xg, Yg : 2D arrays
        Meshgrid
    Ug, Vg : 2D arrays
        Interpolated u, v velocities (NaN over land)
    tri : Triangulation
        The triangulation object
    inside : 2D boolean array
        Mask indicating valid water grid points
    """
    # Create grid bounds with some padding
    pad = R_utm * 0.1
    xmin, xmax = x0 - R_utm - pad, x0 + R_utm + pad
    ymin, ymax = y0 - R_utm - pad, y0 + R_utm + pad
    
    xg = np.linspace(xmin, xmax, nx)
    yg = np.linspace(ymin, ymax, ny)
    Xg, Yg = np.meshgrid(xg, yg)
    
    print(f"\nGrid setup:")
    print(f"  X range: {xmin:.0f} to {xmax:.0f} m")
    print(f"  Y range: {ymin:.0f} to {ymax:.0f} m")
    print(f"  Grid size: {nx} x {ny}")
    
    # Create triangulation and mask out dry triangles
    print(f"  Creating triangulation with {len(tri_idx)} triangles...")
    tri = mtri.Triangulation(x_nodes, y_nodes, triangles=tri_idx)
    tri.set_mask(~wet_cells)  # Mask out dry triangles
    
    num_wet = np.sum(wet_cells)
    print(f"  Wet triangles: {num_wet} / {len(wet_cells)} ({100*num_wet/len(wet_cells):.1f}%)")
    
    # Move cell values to nodes using area-weighted averaging over wet neighbors
    print(f"  Converting cell values to nodes (area-weighted)...")
    u_node = cells_to_nodes_area_weighted(x_nodes, y_nodes, tri_idx, u_cells, wet_cells)
    v_node = cells_to_nodes_area_weighted(x_nodes, y_nodes, tri_idx, v_cells, wet_cells)
    
    # Create masked interpolators (respect tri.mask)
    print(f"  Creating interpolators...")
    Ui = mtri.LinearTriInterpolator(tri, u_node)
    Vi = mtri.LinearTriInterpolator(tri, v_node)
    
    # Grid and wet-only mask via TriFinder
    print(f"  Finding wet grid points...")
    tind = tri.get_trifinder()(Xg, Yg)
    inside = tind >= 0  # Inside triangulation
    inside &= wet_cells[np.where(inside, tind, 0)]  # Inside AND wet triangle
    
    # Initialize with NaN, then fill wet points
    Ug = np.full_like(Xg, np.nan, dtype=float)
    Vg = np.full_like(Yg, np.nan, dtype=float)
    Ug[inside] = Ui(Xg, Yg)[inside]
    Vg[inside] = Vi(Xg, Yg)[inside]
    
    valid_points = np.sum(inside)
    print(f"  Valid water grid points: {valid_points} / {nx*ny} ({100*valid_points/(nx*ny):.1f}%)")
    
    return xg, yg, Xg, Yg, Ug, Vg, tri, inside, u_node, v_node

def create_land_overlay(Xg, Yg, land_mask, color='tan', alpha=0.5):
    """
    Create a masked array for rendering land areas.
    
    Parameters:
    -----------
    Xg, Yg : 2D arrays
        Grid coordinates
    land_mask : 2D boolean array
        True for land points, False for water
    color : str
        Color to use for land (will be mapped through colormap)
    alpha : float
        Transparency
    
    Returns:
    --------
    masked_array : numpy masked array
        Values are 1.0 where land_mask is True, masked elsewhere
    """
    return np.ma.masked_where(~land_mask, np.ones_like(Xg))

def create_quiver_grid(x0, y0, R_utm, spacing_m=100):
    """
    Create a regular grid for quiver arrows.
    
    Parameters:
    -----------
    x0, y0 : float
        Center point in UTM
    R_utm : float
        Radius in meters
    spacing_m : float
        Spacing between arrows in meters (default: 100m)
    
    Returns:
    --------
    qx, qy : 1D arrays
        X and Y coordinates of quiver points
    """
    # Create grid bounds
    xmin, xmax = x0 - R_utm, x0 + R_utm
    ymin, ymax = y0 - R_utm, y0 + R_utm
    
    # Create regular grid with specified spacing
    x_points = np.arange(xmin, xmax + spacing_m, spacing_m)
    y_points = np.arange(ymin, ymax + spacing_m, spacing_m)
    
    # Create meshgrid and flatten
    qx_grid, qy_grid = np.meshgrid(x_points, y_points)
    qx = qx_grid.flatten()
    qy = qy_grid.flatten()
    
    return qx, qy

def interpolate_at_points(tri, u_node, v_node, qx, qy, inside_mask_grid, xg, yg):
    """
    Interpolate u,v velocities at specific points, excluding land.
    
    Parameters:
    -----------
    tri : matplotlib.tri.Triangulation
        Triangulation object (with mask set for dry cells)
    u_node, v_node : 1D arrays
        Velocity components at nodes
    qx, qy : 1D arrays
        Points where to interpolate
    inside_mask_grid : 2D boolean array
        Water mask on the regular grid
    xg, yg : 1D arrays
        Grid coordinates (for inside_mask_grid)
    
    Returns:
    --------
    qx_water, qy_water : 1D arrays
        Coordinates of points in water
    qu, qv : 1D arrays
        Interpolated velocities at those points
    """
    # Create interpolators
    Ui = mtri.LinearTriInterpolator(tri, u_node)
    Vi = mtri.LinearTriInterpolator(tri, v_node)
    
    # Find which quiver points are in water
    # Use TriFinder to check if points are in wet triangles
    trifinder = tri.get_trifinder()
    tri_indices = trifinder(qx, qy)
    
    # Keep only points inside a wet triangle
    in_water = tri_indices >= 0
    
    qx_water = qx[in_water]
    qy_water = qy[in_water]
    
    # Interpolate at water points
    qu = Ui(qx_water, qy_water)
    qv = Vi(qx_water, qy_water)
    
    # Handle any NaN values
    valid = ~(np.isnan(qu) | np.isnan(qv))
    qx_water = qx_water[valid]
    qy_water = qy_water[valid]
    qu = qu[valid]
    qv = qv[valid]
    
    return qx_water, qy_water, qu, qv

def plot_uv_components(ds, center_lat, center_lon, radius_miles=5, 
                       time_index=0, save_file=None, basemap=None, arrow_scale=1.0,
                       quiver_spacing_m=100):
    """
    Plot u and v components side by side to verify interpolation.
    
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
    basemap : str, optional
        'contextily' - use web tiles
        None - no basemap (default)
    arrow_scale : float
        Multiplier for arrow lengths (default: 1.0)
        Higher values = longer arrows, lower values = shorter arrows
    quiver_spacing_m : float
        Spacing between quiver arrows in meters (default: 100)
    """
    
    # Convert radius from miles to meters
    radius_meters = radius_miles * 1609.34
    
    print(f"\nPlotting region:")
    print(f"  Center: {center_lat:.5f}°N, {center_lon:.5f}°W")
    print(f"  Radius: {radius_miles} miles ({radius_meters:.0f} meters)")
    
    # Create UTM transformer
    transformer = create_utm_transformer(center_lat, center_lon)
    
    # Extract data needed for interpolation
    # u, v are on cells (elements)
    u_cells = ds["u"].isel(time=time_index, siglay=0).values
    v_cells = ds["v"].isel(time=time_index, siglay=0).values
    wet_cells = ds["wet_cells"].isel(time=time_index).values.astype(bool)
    
    # Node coordinates
    lons_nodes = ds["lon"].values
    lats_nodes = ds["lat"].values
    
    # Triangle connectivity (nv is 1-based, convert to 0-based)
    tri_idx = ds["nv"].values.T - 1  # (nele, 3)
    
    # Convert longitudes from 0-360 to -180 to 180 if needed
    if lons_nodes.max() > 180:
        lons_nodes = np.where(lons_nodes > 180, lons_nodes - 360, lons_nodes)
    
    print(f"\nDataset info:")
    print(f"  Nodes: {len(lons_nodes)}")
    print(f"  Cells: {len(u_cells)}")
    print(f"  Triangles: {len(tri_idx)}")
    print(f"  Wet cells: {np.sum(wet_cells)} ({100*np.sum(wet_cells)/len(wet_cells):.1f}%)")
    
    # Transform all nodes to UTM
    x_nodes, y_nodes = transformer.transform(lons_nodes, lats_nodes)
    center_x, center_y = transformer.transform(center_lon, center_lat)
    
    print(f"\nCenter in UTM: ({center_x:.0f}, {center_y:.0f})")
    
    # Find cells within radius (use cell centers)
    lons_cells = ds["lonc"].values
    lats_cells = ds["latc"].values
    if lons_cells.max() > 180:
        lons_cells = np.where(lons_cells > 180, lons_cells - 360, lons_cells)
    
    x_cells, y_cells = transformer.transform(lons_cells, lats_cells)
    distances = np.sqrt((x_cells - center_x)**2 + (y_cells - center_y)**2)
    cells_in_radius = distances <= radius_meters * 1.2
    
    print(f"\nCells within radius:")
    print(f"  Total: {np.sum(cells_in_radius)}")
    print(f"  Wet: {np.sum(cells_in_radius & wet_cells)}")
    
    # Get stats on velocities
    u_in_radius = u_cells[cells_in_radius & wet_cells]
    v_in_radius = v_cells[cells_in_radius & wet_cells]
    
    if len(u_in_radius) < 10:
        print("\nError: Too few wet cells for interpolation!")
        return None
    
    print(f"  U range: {u_in_radius.min():.3f} to {u_in_radius.max():.3f} m/s")
    print(f"  V range: {v_in_radius.min():.3f} to {v_in_radius.max():.3f} m/s")
    
    # Get wet_nodes for overlay
    print("\nExtracting wet_nodes mask...")
    wet_nodes = ds["wet_nodes"].isel(time=time_index).values.astype(bool)
    
    # Mask nodes to radius
    distances_nodes = np.sqrt((x_nodes - center_x)**2 + (y_nodes - center_y)**2)
    node_mask = distances_nodes <= radius_meters * 1.2
    
    x_nodes_in = x_nodes[node_mask]
    y_nodes_in = y_nodes[node_mask]
    wet_in = wet_nodes[node_mask]
    
    # Separate wet and dry nodes
    x_wet = x_nodes_in[wet_in]
    y_wet = y_nodes_in[wet_in]
    x_dry = x_nodes_in[~wet_in]
    y_dry = y_nodes_in[~wet_in]
    
    print(f"  Wet nodes: {len(x_wet)}")
    print(f"  Dry nodes: {len(x_dry)}")
    
    # Interpolate to regular grid with proper wet/dry masking
    print("\n" + "="*60)
    print("INTERPOLATION (WET-ONLY)")
    print("="*60)
    xg, yg, Xg, Yg, Ug, Vg, tri, inside, u_node, v_node = interpolate_to_grid(
        x_nodes, y_nodes, tri_idx, u_cells, v_cells, wet_cells,
        center_x, center_y, radius_meters,
        nx=300, ny=300
    )
    
    # Statistics on interpolated grid
    print(f"\nInterpolated field statistics:")
    print(f"  U component: {np.nanmin(Ug):.3f} to {np.nanmax(Ug):.3f} m/s (mean: {np.nanmean(Ug):.3f})")
    print(f"  V component: {np.nanmin(Vg):.3f} to {np.nanmax(Vg):.3f} m/s (mean: {np.nanmean(Vg):.3f})")
    
    # Land mask statistics
    land_points = np.sum(~inside)
    water_points = np.sum(inside)
    print(f"\nGrid mask statistics:")
    print(f"  Water points: {water_points} ({100*water_points/(water_points+land_points):.1f}%)")
    print(f"  Land points: {land_points} ({100*land_points/(water_points+land_points):.1f}%)")
    
    # Create quiver grid
    print("\n" + "="*60)
    print(f"QUIVER GRID ({quiver_spacing_m}m spacing)")
    print("="*60)
    qx, qy = create_quiver_grid(center_x, center_y, radius_meters, spacing_m=quiver_spacing_m)
    print(f"  Total quiver points: {len(qx)}")
    
    # Interpolate velocities at quiver points
    qx_water, qy_water, qu, qv = interpolate_at_points(
        tri, u_node, v_node, qx, qy, inside, xg, yg
    )
    print(f"  Quiver points in water: {len(qx_water)}")
    
    # Compute speed at quiver points for coloring
    qs = np.hypot(qu, qv)
    qs_knots = qs * 1.94384  # Convert to knots
    print(f"  Speed range: {qs_knots.min():.2f} to {qs_knots.max():.2f} knots")
    
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
    
    # Create side-by-side plot
    print("\n" + "="*60)
    print("CREATING PLOT")
    print("="*60)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Determine shared color limits for consistency
    u_vmin, u_vmax = np.nanpercentile(Ug, [2, 98])
    v_vmin, v_vmax = np.nanpercentile(Vg, [2, 98])
    
    # Make limits symmetric for diverging colormap
    u_lim = max(abs(u_vmin), abs(u_vmax))
    v_lim = max(abs(v_vmin), abs(v_vmax))
    
    # Create land mask (inverse of water mask) and overlay
    land_mask = ~inside
    land_overlay = create_land_overlay(Xg, Yg, land_mask)
    
    # Set axis limits first (needed for basemap)
    margin = radius_meters * 0.05
    ax1.set_xlim(center_x - radius_meters - margin, center_x + radius_meters + margin)
    ax1.set_ylim(center_y - radius_meters - margin, center_y + radius_meters + margin)
    ax1.set_aspect('equal')
    
    # Add basemap if requested (behind everything)
    if basemap:
        print(f"\nAdding basemap: {basemap}")
        add_basemap(ax1, basemap_type=basemap, zoom='auto', alpha=0.7)
    
    # Fill land areas with a neutral color (before plotting data)
    # Only if no basemap, otherwise basemap shows land
    if not basemap:
        ax1.pcolormesh(Xg, Yg, land_overlay, cmap='Greys', vmin=0, vmax=1, 
                       alpha=0.5, shading='auto', zorder=1)
    
    # Plot U component (only over water)
    im1 = ax1.pcolormesh(Xg, Yg, Ug, cmap='RdBu_r', vmin=-u_lim, vmax=u_lim, 
                         shading='auto', zorder=2)
    
    # Add quiver plot (colored by speed)
    try:
        import cmocean
        quiver_cmap = cmocean.cm.speed
    except ImportError:
        quiver_cmap = 'viridis'
    
    # Calculate effective scale (smaller = longer arrows)
    base_scale = 10.0
    effective_scale = base_scale / arrow_scale
    
    q1 = ax1.quiver(qx_water, qy_water, qu, qv, qs_knots,
                    cmap=quiver_cmap, 
                    scale=effective_scale, scale_units='inches',
                    width=0.003, headwidth=3, headlength=4, headaxislength=3,
                    alpha=0.8, zorder=10, 
                    clim=[0, np.nanpercentile(qs_knots, 95)])
    
    # Overlay wet/dry nodes (optional - can be removed for cleaner plot)
    # if len(x_wet) > 0:
    #     ax1.scatter(x_wet, y_wet, c='green', s=1, alpha=0.1, 
    #                label=f'Wet nodes ({len(x_wet)})', zorder=5)
    # if len(x_dry) > 0:
    #     ax1.scatter(x_dry, y_dry, c='brown', s=1, alpha=0.1, marker='x',
    #                label=f'Dry nodes ({len(x_dry)})', zorder=5)
    
    ax1.plot(center_x, center_y, 'k*', markersize=20, markeredgecolor='yellow', 
             markeredgewidth=2, label='Center', zorder=10)
    
    # Add circle for reference
    from matplotlib.patches import Circle
    circle1 = Circle((center_x, center_y), radius_meters,
                     fill=False, edgecolor='yellow', linewidth=2, 
                     linestyle='--', label=f'{radius_miles} mi radius', zorder=10)
    ax1.add_patch(circle1)
    
    ax1.set_xlabel('Easting (m, UTM Zone 10N)', fontsize=11)
    ax1.set_ylabel('Northing (m, UTM Zone 10N)', fontsize=11)
    ax1.set_title(f'U Component (Eastward) + Quiver\n{time_str}', fontsize=12, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax1.legend(loc='upper right', fontsize=8)
    
    # Add quiver reference
    ax1.text(0.02, 0.02, f'Arrows: {quiver_spacing_m:.0f}m spacing\n{len(qx_water)} vectors\nScale: {arrow_scale:.1f}x',
            transform=ax1.transAxes, fontsize=8, color='white',
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))
    
    cbar1 = plt.colorbar(im1, ax=ax1, label='U velocity (m/s)', pad=0.02)
    
    # Set axis limits for V component panel
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_aspect('equal')
    
    # Add basemap if requested
    if basemap:
        add_basemap(ax2, basemap_type=basemap, zoom='auto', alpha=0.7)
    
    # Fill land areas with a neutral color (before plotting data)
    # Only if no basemap, otherwise basemap shows land
    if not basemap:
        ax2.pcolormesh(Xg, Yg, land_overlay, cmap='Greys', vmin=0, vmax=1, 
                       alpha=0.5, shading='auto', zorder=1)
    
    # Plot V component (only over water)
    im2 = ax2.pcolormesh(Xg, Yg, Vg, cmap='RdBu_r', vmin=-v_lim, vmax=v_lim, 
                         shading='auto', zorder=2)
    
    # Add quiver plot (colored by speed)
    q2 = ax2.quiver(qx_water, qy_water, qu, qv, qs_knots,
                    cmap=quiver_cmap, 
                    scale=effective_scale, scale_units='inches',
                    width=0.003, headwidth=3, headlength=4, headaxislength=3,
                    alpha=0.8, zorder=10,
                    clim=[0, np.nanpercentile(qs_knots, 95)])
    
    # Overlay wet/dry nodes (optional)
    # if len(x_wet) > 0:
    #     ax2.scatter(x_wet, y_wet, c='green', s=1, alpha=0.1, 
    #                label=f'Wet nodes ({len(x_wet)})', zorder=5)
    # if len(x_dry) > 0:
    #     ax2.scatter(x_dry, y_dry, c='brown', s=1, alpha=0.1, marker='x',
    #                label=f'Dry nodes ({len(x_dry)})', zorder=5)
    
    ax2.plot(center_x, center_y, 'k*', markersize=20, markeredgecolor='yellow',
             markeredgewidth=2, label='Center', zorder=10)
    
    circle2 = Circle((center_x, center_y), radius_meters,
                     fill=False, edgecolor='yellow', linewidth=2, 
                     linestyle='--', label=f'{radius_miles} mi radius', zorder=10)
    ax2.add_patch(circle2)
    
    ax2.set_xlabel('Easting (m, UTM Zone 10N)', fontsize=11)
    ax2.set_ylabel('Northing (m, UTM Zone 10N)', fontsize=11)
    ax2.set_title(f'V Component (Northward) + Quiver\n{time_str}', fontsize=12, fontweight='bold')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax2.legend(loc='upper right', fontsize=8)
    
    # Add quiver reference
    ax2.text(0.02, 0.02, f'Arrows: {quiver_spacing_m:.0f}m spacing\n{len(qx_water)} vectors\nScale: {arrow_scale:.1f}x',
            transform=ax2.transAxes, fontsize=8, color='white',
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))
    
    cbar2 = plt.colorbar(im2, ax=ax2, label='V velocity (m/s)', pad=0.02)
    
    # Overall title
    plt.suptitle(
        f'SSCOFS Surface Current Components\n'
        f'Location: ({center_lat:.4f}°N, {abs(center_lon):.4f}°W)',
        fontsize=14, fontweight='bold'
    )
    
    plt.tight_layout()
    
    if save_file:
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_file}")
    
    print("\nDisplaying plot...")
    plt.show()
    
    return fig

def main():
    parser = argparse.ArgumentParser(
        description="Simple SSCOFS current visualization - U and V components"
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
        "--save", type=str,
        help="Save plot to file (e.g., 'currents_uv.png')"
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
    parser.add_argument(
        "--basemap", type=str, default=None,
        choices=['contextily', 'natural_earth', 'none', None],
        help="Add basemap: 'contextily' (web tiles) or 'natural_earth' (local shapefile)"
    )
    parser.add_argument(
        "--arrow-scale", type=float, default=1.0,
        help="Arrow length multiplier (default: 1.0, higher=longer arrows, lower=shorter)"
    )
    parser.add_argument(
        "--quiver-spacing", type=float, default=100.0,
        help="Spacing between quiver arrows in meters (default: 100)"
    )
    
    args = parser.parse_args()
    
    # Handle cache management commands first
    if args.list_cache:
        list_cache()
        return 0
    
    if args.clear_cache:
        clear_cache()
        return 0
    
    print("="*60)
    print("SIMPLE CURRENT PLOT - U AND V COMPONENTS")
    print("="*60)
    print(f"Location: {args.lat:.5f}°N, {abs(args.lon):.5f}°W")
    print(f"Radius: {args.radius} miles")
    print("="*60)
    
    try:
        # Get latest data
        ds, info = get_latest_current_data(use_cache=not args.no_cache)
        
        # Create plot
        fig = plot_uv_components(
            ds, 
            args.lat, 
            args.lon, 
            radius_miles=args.radius,
            time_index=args.time_index,
            save_file=args.save,
            basemap=args.basemap,
            arrow_scale=args.arrow_scale,
            quiver_spacing_m=args.quiver_spacing
        )
        
        if fig is None:
            return 1
            
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

