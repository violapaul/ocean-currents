# Current Visualization Enhancements

This document summarizes the new enhanced plotting capabilities added to the SSCOFS current visualization tools.

## What's New

A new script `plot_currents_enhanced.py` has been created based on oceanographic best practices from `UPDATES.md`. This script provides advanced flow visualization techniques specifically designed for tactical sailing and racing applications.

## Key Improvements Over Basic Plotting

### 1. **Basemap Support** (NEW!)
Optional background maps for geographic context:
- **Contextily (Web Tiles)** - Beautiful Esri WorldGrayCanvas tiles
  - Requires `pip install contextily`
  - Automatically handles coordinate transformation from UTM to Web Mercator
  - Auto-zoom to appropriate level
  - Semi-transparent (50% alpha) to not overwhelm current data
- **Natural Earth (Shapefiles)** - Offline coastline data
  - Requires `conda install geopandas` + data file in `data/` directory
  - Stays in UTM Zone 10N (no reprojection overhead)
  - Lightweight and fast for offline use

Usage: `--basemap contextily` or `--basemap natural_earth`

### 2. **Streamline Visualization**
Instead of uniform arrow grids, streamlines show the actual flow pathways:
- **Variable width** - thicker where currents are faster
- **Continuous paths** - easier to see flow direction and structure
- **No arrow clutter** - cleaner visualization of overall pattern

### 2. **Speed Contour Analysis**
Filled contours reveal speed bands and gradients:
- **Jets and fronts** become immediately visible as color bands
- **Tactical threshold contours** (0.5, 1.0, 1.5, 2.0 knots) overlay in white
- Helps identify where to cross a band vs. where to ride it

### 3. **Adaptive Quiver Placement**
Arrows are placed intelligently based on flow structure:
- **Gradient-weighted sampling** - more arrows where speed changes rapidly
- **Highlights fronts, shears, and eddies** automatically
- **Cleaner than uniform subsampling** - arrows appear where they matter most
- Uses greedy min-distance algorithm to maintain visibility

### 4. **Flow Diagnostics** (Oceanographic Analysis)

#### Vorticity (ζ = ∂v/∂x - ∂u/∂y)
- Shows **rotation and circulation** in the flow
- Positive values: counterclockwise rotation
- Negative values: clockwise rotation
- Strong vorticity indicates eddies or turning flows

#### Divergence (δ = ∂u/∂x + ∂v/∂y)
- Shows where flow is **converging or diverging**
- Positive: flow spreading out
- Negative: flow concentrating (upwelling/downwelling signals)

#### Okubo-Weiss Parameter (OW = s_n² + s_s² - ζ²)
Critical for tactical decisions:
- **OW > 0**: Strain-dominated regions (fronts, sharp bands)
  - Expect **rapid speed changes** across these zones
  - Good tacking/jibing opportunities if you cross at optimal angle
- **OW < 0**: Eddy-dominated regions (vortex cores)
  - Watch for **flow "hooking"** or curved paths
  - Laylines may bend significantly near headlands/eddies
- **OW ≈ 0**: Boundary between regimes (marked with black dashed line)

## Visualization Styles

### `--style streamline` (Recommended for quick overview)
- Speed contours + streamlines
- Best for understanding overall flow pattern
- Tactical threshold contours highlight key speeds

### `--style adaptive` (Best for detailed structure)
- Speed contours + adaptive quiver arrows
- Arrows concentrate at fronts and shear zones
- Good for identifying specific features

### `--style both` (Comprehensive)
- Combines streamlines + adaptive quiver
- Most information-rich single panel
- Can be busy, but shows everything

### `--style diagnostic` (For race planning)
- Two panels: flow view + diagnostics view
- Left: Speed contours + streamlines + adaptive arrows
- Right: Vorticity contours + Okubo-Weiss boundaries + eddy cores
- Use this for detailed race course analysis

## Technical Implementation

### Grid Interpolation
The script interpolates the unstructured FVCOM grid to a regular 300×300 grid:
- Uses Delaunay triangulation (`matplotlib.tri`)
- Linear interpolation preserves flow features
- NaN masking for land/invalid regions
- Padding around region of interest for edge effects

### Coordinate System
- **UTM Zone 10N (EPSG:32610)** for Puget Sound
- Meters for accurate distance representation
- Consistent with the basic plotting script

### Performance
- Interpolation takes ~2-5 seconds for typical regions
- Adaptive quiver selection is fast (greedy algorithm)
- Diagnostic computations use numpy gradient (vectorized)

## Usage Examples

```bash
# Quick streamline view
python plot_currents_enhanced.py --lat 47.6718 --lon -122.4584 --radius 5

# Adaptive arrows for detailed analysis
python plot_currents_enhanced.py --style adaptive --save fronts.png

# Full diagnostic for race planning
python plot_currents_enhanced.py --style diagnostic --radius 3 --save race_analysis.png

# Add web tile basemap for geographic context
python plot_currents_enhanced.py --basemap contextily --style streamline --save with_map.png

# Different time steps
python plot_currents_enhanced.py --time-index 12 --style both  # 12 hours into forecast

# Complete race planning map with all features
python plot_currents_enhanced.py --basemap contextily --style diagnostic --radius 3 --save complete_analysis.png
```

## Interpreting the Diagnostics for Racing

### Before the Race
1. **Run diagnostic mode** to identify:
   - Eddy cores (avoid or use strategically)
   - Strain fronts (where to expect rapid changes)
   - Jet boundaries (fast bands to ride)

2. **Look for OW = 0 contours**:
   - These mark transitions between different flow regimes
   - Crossing these may require tactical adjustments

3. **Check vorticity patterns**:
   - Strong vorticity near headlands → curved flow
   - Laylines may not be straight near these features

### During the Race
1. **Use streamline or adaptive views** for quick checks
2. **Watch tactical threshold contours**:
   - White lines at 0.5, 1.0, 1.5, 2.0 kt make it easy to gauge advantage
3. **Look for adaptive arrow clusters**:
   - These mark rapidly changing conditions
   - Approach with caution or exploit if timed well

### After the Race
1. **Compare diagnostic view** with actual GPS track
2. **Identify where model matched/diverged** from reality
3. **Learn which features are reliable** in your area

## Future Enhancements (Possible)

Based on UPDATES.md suggestions not yet fully implemented:

1. **Time series visualization**
   - Multi-panel showing T-2h, T-1h, T, T+1h, T+2h, T+3h
   - Delta speed panel showing changes

2. **GPX/KML overlay**
   - Race marks and course boundaries
   - Planned routes
   - Actual GPS tracks for comparison

3. **Interactive features**
   - Click to probe speed/direction at point
   - Toggle overlays on/off
   - Zoom/pan while preserving aspect ratio

4. **Alternative basemap providers**
   - NOAA ENC/RNC chart tiles
   - OpenStreetMap variants
   - Custom tile providers

## Dependencies

All dependencies are in the existing `currents_env.yml`:
- `matplotlib` (streamplot, quiver, contour)
- `numpy` (gradients, array operations)
- `scipy` (spatial KDTree for adaptive sampling)
- `xarray`, `netcdf4` (data loading)
- `pyproj` (coordinate transformations)

Optional (for basemaps):
- `contextily` - web tile basemaps (install via `pip install contextily`)
- `geopandas` - shapefile support (install via `conda install geopandas`)
- `cmocean` - better colormaps for oceanography (install via `pip install cmocean`)

## References

The techniques implemented are based on standard oceanographic analysis methods:

1. **Okubo-Weiss parameter**: Okubo (1970), Weiss (1991)
   - Standard tool for identifying eddies vs. strain regions
   - Widely used in ocean circulation analysis

2. **Adaptive sampling**: Based on gradient magnitude
   - Common in flow visualization (Turk & Banks, 1996)
   - Optimizes information density

3. **Streamline visualization**: Standard in fluid dynamics
   - Matplotlib implementation based on Jobard & Lefer (1997)
   - Variable width encoding from Turk et al. (1996)

## Credits

Implementation based on suggestions in `.sscofs_cache/UPDATES.md`, incorporating:
- Clean basemap approaches
- Structure-emphasizing visualization
- Tactical overlays for racing
- Oceanographic diagnostic techniques

---

**For questions or suggestions, see the main README.md**

