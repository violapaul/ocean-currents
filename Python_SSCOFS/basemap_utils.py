"""
basemap_utils.py
----------------

Utilities for adding basemaps to matplotlib plots.
Supports contextily (web tiles) and local shapefiles (Natural Earth).

Usage:
    from basemap_utils import add_basemap
    
    fig, ax = plt.subplots()
    # ... plot data in UTM Zone 10N coordinates ...
    add_basemap(ax, basemap_type='contextily', zoom='auto', alpha=0.5)
"""

from pathlib import Path

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

try:
    from pyproj import Transformer
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False


def add_basemap(ax, basemap_type='contextily', zoom='auto', alpha=0.5, 
                source=None, crs="EPSG:32610"):
    """
    Add a basemap to the axis.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis to add the basemap to (must be in projected coords)
    basemap_type : str
        'contextily' - web tiles (requires contextily package)
        'natural_earth' - coastline shapefile (requires geopandas and data file)
        'none' - no basemap
    zoom : str or int
        Zoom level for contextily ('auto' or 1-18)
    alpha : float
        Transparency of basemap (0=transparent, 1=opaque)
    source : contextily tile provider, optional
        Custom tile source (default: OpenStreetMap.Mapnik)
    crs : str
        Coordinate reference system of the axis (default: EPSG:32610 - UTM Zone 10N)
    
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
            # Use default source if not provided
            if source is None:
                source = ctx.providers.OpenStreetMap.Mapnik
            
            # contextily will handle coordinate transformation
            ctx.add_basemap(
                ax, 
                crs=crs,  # Tell contextily our current CRS
                source=source,
                zoom=zoom,
                alpha=alpha,
                attribution=""  # Clean plot
            )
            
            print(f"  Added basemap: {source.get('name', 'OpenStreetMap')} (zoom={zoom}, alpha={alpha})")
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
            # Load and reproject to target CRS
            shore = gpd.read_file(shoreline_file).to_crs(crs)
            shore.boundary.plot(ax=ax, linewidth=0.8, color='#555555', zorder=1, alpha=alpha)
            print(f"  Added basemap: {shoreline_file.name}")
            return True
            
        except Exception as e:
            print(f"Warning: Could not load shoreline data: {e}")
            return False
    
    else:
        print(f"Warning: Unknown basemap type '{basemap_type}'")
        return False


def get_available_basemap_sources():
    """
    Return a dictionary of available contextily basemap sources.
    
    Returns:
    --------
    dict : Dictionary of source names to contextily providers
    """
    if not HAS_CONTEXTILY:
        return {}
    
    return {
        'osm': ctx.providers.OpenStreetMap.Mapnik,
        'esri_worldimagery': ctx.providers.Esri.WorldImagery,
        'esri_natgeo': ctx.providers.Esri.NatGeoWorldMap,
        'cartodb_positron': ctx.providers.CartoDB.Positron,
        'cartodb_darkmatter': ctx.providers.CartoDB.DarkMatter,
        'stamen_terrain': ctx.providers.Stamen.Terrain,
        'stamen_toner': ctx.providers.Stamen.Toner,
    }


def check_basemap_dependencies():
    """
    Check which basemap dependencies are available.
    
    Returns:
    --------
    dict : Status of each dependency
    """
    return {
        'contextily': HAS_CONTEXTILY,
        'geopandas': HAS_GEOPANDAS,
        'pyproj': HAS_PYPROJ,
    }


if __name__ == "__main__":
    # Test/demo code
    print("Basemap utilities")
    print("=" * 60)
    
    deps = check_basemap_dependencies()
    print("\nDependencies:")
    for name, available in deps.items():
        status = "✓ Available" if available else "✗ Not installed"
        print(f"  {name}: {status}")
    
    if HAS_CONTEXTILY:
        print("\nAvailable basemap sources:")
        for name in get_available_basemap_sources():
            print(f"  - {name}")

