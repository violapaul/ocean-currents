#!/usr/bin/env python3
"""
test_basemaps.py
----------------

Test basemap functionality independently without requiring SSCOFS data.

This script creates simple test plots to verify each basemap option works correctly.
Uses synthetic data so it doesn't need to download ocean model data.

Usage:
    python test_basemaps.py --all           # Test all available options
    python test_basemaps.py --contextily    # Test web tiles only
    python test_basemaps.py --natural-earth # Test shapefiles only
    python test_basemaps.py --simple        # Test without any basemap
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import Normalize
from pathlib import Path
from pyproj import Transformer

# Try to import basemap dependencies
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

def create_test_data(center_x, center_y, radius_m, resolution=50):
    """
    Create synthetic current data for testing.
    Returns a simple vortex pattern.
    """
    # Create grid
    x = np.linspace(center_x - radius_m, center_x + radius_m, resolution)
    y = np.linspace(center_y - radius_m, center_y + radius_m, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Create a vortex pattern
    dx = X - center_x
    dy = Y - center_y
    r = np.sqrt(dx**2 + dy**2)
    
    # Avoid division by zero
    r = np.where(r < 100, 100, r)
    
    # Tangential velocity (vortex)
    theta = np.arctan2(dy, dx)
    speed = 0.5 * (1 - np.exp(-r / (radius_m * 0.3)))  # m/s
    
    U = -speed * np.sin(theta)
    V = speed * np.cos(theta)
    
    # Add some asymmetry
    U += 0.2 * np.sin(Y / (radius_m * 0.5))
    V += 0.1 * np.cos(X / (radius_m * 0.5))
    
    S = np.hypot(U, V)
    
    return x, y, X, Y, U, V, S

def add_basemap_contextily(ax, zoom='auto', alpha=1.0):
    """Add contextily basemap to axis."""
    if not HAS_CONTEXTILY:
        print("✗ Contextily not available")
        return False
    
    try:
        # Use OpenStreetMap for better land/water contrast
        # Land shows as tan/beige, water as blue - much clearer distinction
        ctx.add_basemap(
            ax,
            crs="EPSG:32610",  # UTM Zone 10N
            source=ctx.providers.OpenStreetMap.Mapnik,  # Better land/water contrast than WorldGrayCanvas
            zoom=zoom,
            alpha=alpha,
            attribution=""
        )
        print(f"✓ Contextily basemap added (OpenStreetMap, alpha={alpha})")
        return True
    except Exception as e:
        print(f"✗ Failed to add contextily basemap: {e}")
        return False

def add_basemap_natural_earth(ax, data_dir='data'):
    """Add Natural Earth basemap to axis."""
    if not HAS_GEOPANDAS:
        print("✗ Geopandas not available")
        return False
    
    data_dir = Path(data_dir)
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
    
    if not shoreline_file:
        print(f"✗ No coastline data found in {data_dir}")
        print(f"  Looked for: {[f.name for f in possible_files]}")
        return False
    
    try:
        shore = gpd.read_file(shoreline_file).to_crs("EPSG:32610")
        shore.boundary.plot(ax=ax, linewidth=0.8, color='#555555', zorder=1)
        print(f"✓ Natural Earth basemap added: {shoreline_file.name}")
        return True
    except Exception as e:
        print(f"✗ Failed to load shapefile: {e}")
        return False

def test_basemap(basemap_type='none', save_file=None, data_dir='data', basemap_only=False):
    """
    Create a test plot with synthetic current data and specified basemap.
    
    Parameters:
    -----------
    basemap_type : str
        'contextily', 'natural_earth', or 'none'
    save_file : str, optional
        Path to save the plot
    data_dir : str
        Directory with coastline data
    basemap_only : bool
        If True, show only basemap without current overlays (clearer view)
    """
    print("\n" + "="*60)
    print(f"TESTING: {basemap_type.upper()}")
    print("="*60)
    
    # Test location: User's point in Puget Sound
    center_lat = 47.67181822458632
    center_lon = -122.4583957143628
    radius_miles = 5
    radius_m = radius_miles * 1609.34
    
    # Transform to UTM
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True)
    center_x, center_y = transformer.transform(center_lon, center_lat)
    
    print(f"Test location: {center_lat:.4f}°N, {abs(center_lon):.4f}°W")
    print(f"UTM coordinates: {center_x:.0f}E, {center_y:.0f}N")
    print(f"Radius: {radius_miles} miles ({radius_m:.0f} m)")
    
    # Create synthetic test data
    print("\nGenerating synthetic vortex pattern...")
    xg, yg, Xg, Yg, Ug, Vg, Sg = create_test_data(center_x, center_y, radius_m)
    Sg_knots = Sg * 1.94384  # Convert to knots
    
    print(f"  Grid: {len(xg)}x{len(yg)}")
    print(f"  Speed range: {np.min(Sg_knots):.2f} - {np.max(Sg_knots):.2f} knots")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Set limits first (for basemap)
    margin = radius_m * 0.05
    ax.set_xlim(center_x - radius_m - margin, center_x + radius_m + margin)
    ax.set_ylim(center_y - radius_m - margin, center_y + radius_m + margin)
    ax.set_aspect('equal')
    
    # Add basemap
    basemap_added = False
    if basemap_type == 'contextily':
        print("\nAdding contextily basemap...")
        basemap_added = add_basemap_contextily(ax, zoom='auto', alpha=0.5)
    elif basemap_type == 'natural_earth':
        print("\nAdding Natural Earth basemap...")
        basemap_added = add_basemap_natural_earth(ax, data_dir=data_dir)
    else:
        print("\nNo basemap (testing basic plot)")
    
    # Speed contours and overlays (skip if basemap_only)
    if not basemap_only:
        print("\nAdding current visualization...")
        try:
            import cmocean
            cmap = cmocean.cm.speed
            print("  Using cmocean colormap")
        except ImportError:
            cmap = 'viridis'
            print("  Using viridis colormap")
        
        vmin, vmax = np.percentile(Sg_knots, [5, 95])
        norm = Normalize(vmin=vmin, vmax=vmax)
        
        # Filled contours
        levels = np.linspace(vmin, vmax, 20)
        cs = ax.contourf(Xg, Yg, Sg_knots, levels=levels, cmap=cmap, norm=norm, alpha=0.7)
        
        # Streamlines
        lw = 0.5 + 2.0 * (Sg - np.min(Sg)) / (np.max(Sg) - np.min(Sg))
        strm = ax.streamplot(
            xg, yg, Ug, Vg,
            color=Sg_knots, cmap=cmap, norm=norm,
            density=1.5, linewidth=lw, arrowsize=1.0, zorder=3
        )
        
        # Tactical contours
        tactical_levels = [0.5, 1.0, 1.5]
        ax.contour(Xg, Yg, Sg_knots, levels=tactical_levels,
                  colors='white', linewidths=1.0, alpha=0.6)
    else:
        print("\nBasemap-only mode (no current overlays)")
    
    # Center point and radius
    ax.plot(center_x, center_y, 'r*', markersize=15, label='Test Center', zorder=10)
    circle = Circle((center_x, center_y), radius_m,
                   fill=False, edgecolor='red', linewidth=2,
                   linestyle='--', label=f'{radius_miles} mi radius', zorder=10)
    ax.add_patch(circle)
    
    # Colorbar (only if we have current data)
    if not basemap_only:
        cbar = plt.colorbar(cs, ax=ax, label='Current Speed (knots)', pad=0.02)
    
    # Labels and title
    ax.set_xlabel('Easting (m, UTM Zone 10N)', fontsize=11)
    ax.set_ylabel('Northing (m, UTM Zone 10N)', fontsize=11)
    
    if basemap_only:
        title = f'Basemap Display: {basemap_type.upper()}'
        subtitle = f'Test Area: Puget Sound ({center_lat:.4f}°N, {abs(center_lon):.4f}°W)'
    else:
        title = f'Basemap Test: {basemap_type.upper()}'
        if basemap_type != 'none' and not basemap_added:
            title += ' (FAILED - showing without basemap)'
        subtitle = f'Synthetic Vortex Pattern - Puget Sound ({center_lat:.4f}°N, {abs(center_lon):.4f}°W)'
    
    ax.set_title(
        f'{title}\n{subtitle}',
        fontsize=12, fontweight='bold'
    )
    
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    # Save
    if save_file:
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved to: {save_file}")
        size_kb = Path(save_file).stat().st_size / 1024
        print(f"  File size: {size_kb:.1f} KB")
    
    # Show
    print("\nDisplaying plot...")
    plt.show()
    
    # Result
    print("\n" + "-"*60)
    if basemap_type == 'none':
        print("✓ Basic plot test completed")
        return True
    elif basemap_added:
        print(f"✓ {basemap_type.upper()} test PASSED")
        return True
    else:
        print(f"✗ {basemap_type.upper()} test FAILED")
        print(f"  See error messages above")
        return False

def check_availability():
    """Check which basemap options are available."""
    print("\n" + "="*60)
    print("BASEMAP AVAILABILITY CHECK")
    print("="*60)
    
    available = []
    
    # Check contextily
    print("\n1. Contextily (web tiles):")
    if HAS_CONTEXTILY:
        print("   ✓ Installed and ready")
        available.append('contextily')
    else:
        print("   ✗ Not installed")
        print("   → Install with: pip install contextily")
    
    # Check geopandas
    print("\n2. Geopandas (shapefile support):")
    if HAS_GEOPANDAS:
        print("   ✓ Installed")
        
        # Check for data files
        data_dir = Path('data')
        data_files = [
            data_dir / "shoreline_puget.geojson",
            data_dir / "ne_10m_coastline.shp",
        ]
        
        found = [f for f in data_files if f.exists()]
        if found:
            print("   ✓ Coastline data found:")
            for f in found:
                size_kb = f.stat().st_size / 1024
                print(f"     - {f.name} ({size_kb:.1f} KB)")
            available.append('natural_earth')
        else:
            print("   ✗ No coastline data files found")
            print(f"   → Download with: python setup_basemaps.py --natural-earth")
    else:
        print("   ✗ Not installed")
        print("   → Install with: conda install -c conda-forge geopandas")
    
    print("\n" + "="*60)
    print("AVAILABLE OPTIONS")
    print("="*60)
    
    if available:
        print(f"✓ Can test: {', '.join(available)}")
        print("\nRun tests:")
        for opt in available:
            print(f"  python test_basemaps.py --{opt}")
        print(f"  python test_basemaps.py --all")
    else:
        print("✗ No basemap options available")
        print("\nSetup:")
        print("  python setup_basemaps.py --all")
    
    print()
    
    return available

def main():
    parser = argparse.ArgumentParser(
        description="Test basemap functionality with synthetic data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_basemaps.py --check          # Check what's available
  python test_basemaps.py --all            # Test all available options
  python test_basemaps.py --contextily     # Test web tiles
  python test_basemaps.py --natural-earth  # Test shapefiles
  python test_basemaps.py --simple         # Test without basemap
        """
    )
    
    parser.add_argument(
        '--all', action='store_true',
        help='Test all available basemap options'
    )
    parser.add_argument(
        '--contextily', action='store_true',
        help='Test contextily web tiles'
    )
    parser.add_argument(
        '--natural-earth', action='store_true',
        help='Test Natural Earth shapefiles'
    )
    parser.add_argument(
        '--simple', action='store_true',
        help='Test basic plot without basemap'
    )
    parser.add_argument(
        '--check', action='store_true',
        help='Check which options are available'
    )
    parser.add_argument(
        '--save', action='store_true',
        help='Save plots to files instead of just displaying'
    )
    parser.add_argument(
        '--data-dir', type=str, default='data',
        help='Directory with coastline data (default: data/)'
    )
    parser.add_argument(
        '--basemap-only', action='store_true',
        help='Show only basemap without current overlays (clearer view of basemap)'
    )
    
    args = parser.parse_args()
    
    # If no flags, show availability
    if not any([args.all, args.contextily, args.natural_earth, args.simple, args.check]):
        available = check_availability()
        if not available:
            print("Run 'python setup_basemaps.py --all' to setup basemaps")
            return 1
        return 0
    
    # Just check
    if args.check:
        check_availability()
        return 0
    
    # Determine what to test
    tests = []
    if args.all:
        available = []
        if HAS_CONTEXTILY:
            available.append('contextily')
        if HAS_GEOPANDAS:
            # Check if data exists
            data_dir = Path(args.data_dir)
            if any((data_dir / f).exists() for f in ['shoreline_puget.geojson', 'ne_10m_coastline.shp']):
                available.append('natural_earth')
        tests = available
    else:
        if args.contextily:
            tests.append('contextily')
        if args.natural_earth:
            tests.append('natural_earth')
        if args.simple:
            tests.append('none')
    
    if not tests:
        print("No tests selected or no basemaps available")
        check_availability()
        return 1
    
    # Run tests
    print("="*60)
    print("BASEMAP TESTS")
    print("="*60)
    print(f"Will test: {', '.join(tests)}")
    
    results = {}
    for basemap_type in tests:
        save_file = None
        if args.save:
            save_file = f"test_basemap_{basemap_type}.png"
        
        try:
            success = test_basemap(basemap_type, save_file=save_file, 
                                  data_dir=args.data_dir, basemap_only=args.basemap_only)
            results[basemap_type] = success
        except Exception as e:
            print(f"\n✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results[basemap_type] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for basemap_type, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {basemap_type}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! Basemaps are ready for use.")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
        return 1

if __name__ == '__main__':
    exit(main())

