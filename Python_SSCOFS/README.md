# Python Tools for SSCOFS Current Visualization

Command-line tools for downloading, analyzing, and visualizing ocean current data from NOAA's Salish Sea Coastal Ocean Forecast System (SSCOFS).

## Quick Start

```bash
# Setup
conda env create -f currents_env.yml
conda activate currents

# Basic plot
python plot_local_currents.py --lat 47.6718 --lon -122.4584 --radius 5 --save currents.png

# Setup basemaps (one-time).  Optional.  
python setup_basemaps.py --all
python test_basemaps.py --all
```

## Core Modules

### sscofs_cache.py
Centralized caching for SSCOFS NetCDF data. All plotting scripts use this module.

**Features:**
- Downloads from NOAA S3 bucket (`noaa-nos-ofs-pds`)
- Caches files in `.sscofs_cache/` (~200 MB each)
- Provides cache management commands

**Usage:**
```bash
python sscofs_cache.py --list      # List cached files
python sscofs_cache.py --info      # Cache statistics
python sscofs_cache.py --clear     # Clear all cache
```

**API:**
```python
from sscofs_cache import get_sscofs_data

ds = get_sscofs_data(cycle_date='2025-10-18', cycle_hour=9, forecast_hour=12)
```

### latest_cycle.py
Determines which SSCOFS model cycle to use for a given time.

**Features:**
- Finds latest available cycle (00z, 03z, 09z, 15z, 21z)
- Calculates appropriate forecast hour (F000-F072)
- Handles timezone conversions (UTC ↔ Pacific)

**Usage:**
```python
from latest_cycle import find_latest_cycle, pick_forecast_for_local_hour

cycle_date, cycle_hour = find_latest_cycle()
forecast_hour = pick_forecast_for_local_hour(local_hour=14, target='hour_after')
```

### fetch_sscofs.py
Low-level URL construction and S3 access.

**Features:**
- Builds S3 URLs for specific model runs
- Example plotting functionality
- Direct NetCDF file access

### basemap_utils.py
Shared utilities for basemap support across plotting scripts.

**Features:**
- Contextily (web tiles) integration
- Natural Earth (shapefiles) support
- Coordinate transformation helpers
- Consistent styling across scripts

## Plotting Scripts

### plot_local_currents.py - Basic Visualization

Simple quiver plots showing current vectors and speeds.

**Best For:** Quick checks, simple visualization needs

**Features:**
- Uniform arrow grid (subsampled)
- Two-panel layout: vectors + speed scatter
- UTM coordinates for accuracy
- Reference arrow showing 1 m/s scale

**Example:**
```bash
python plot_local_currents.py --lat 47.6718 --lon -122.4584 --radius 5 \
    --subsample 3 --vector-scale 10 --save basic.png
```

**Options:**
- `--lat`, `--lon` - Center coordinates
- `--radius` - Search radius in miles
- `--time-index` - Forecast hour to plot (0-72)
- `--subsample` - Arrow density (every Nth point)
- `--vector-scale` - Arrow length multiplier
- `--save` - Save to file
- `--no-cache` - Force fresh download

### [Archived Scripts]

*Several specialized and debugging scripts have been moved to `backup_removed_files/` to streamline the codebase:*
- `plot_currents_simple.py` - Simplified plotting (functionality duplicated in main scripts)
- `diagnose_currents.py` - Data diagnostics and statistics
- `extract_sscofs_metadata.py` - Metadata extraction (output saved in metadata.txt)
- `plot_wet_nodes.py` - FVCOM grid visualization
- `test_bulk_download.py` - Bulk download testing
- `test_water_mask.py` - Land/water masking tests

## Analysis Scripts

*The core functionality remains in the main plotting scripts. Additional analysis tools are archived in `backup_removed_files/`.*

## Setup & Testing Scripts

### setup_basemaps.py - Automated Basemap Setup

One-time setup for basemap support in enhanced plotting.

**Features:**
- Installs contextily for web tiles
- Downloads Natural Earth coastline data (~5 MB)
- Clips to Puget Sound region (optional)
- Verifies installation

**Usage:**
```bash
# Setup everything (recommended)
python setup_basemaps.py --all

# Just web tiles
python setup_basemaps.py --contextily

# Just shapefiles
python setup_basemaps.py --natural-earth

# Check what's installed
python setup_basemaps.py --verify
```

**Output:**
```
data/
├── ne_10m_coastline.shp          # Full Natural Earth
├── ne_10m_coastline.{shx,dbf,prj}
└── shoreline_puget.geojson       # Clipped (faster)
```

See **BASEMAP_SETUP.md** for detailed manual installation.

### test_basemaps.py - Verify Basemap Setup

Test basemap functionality without needing real SSCOFS data.

**Features:**
- Creates synthetic current data (vortex pattern)
- Tests each basemap option
- Saves test plots
- Reports success/failure

**Usage:**
```bash
# Test all basemaps
python test_basemaps.py --all

# Test specific one
python test_basemaps.py --contextily
python test_basemaps.py --natural-earth

# Check availability
python test_basemaps.py --check

# Save test plots
python test_basemaps.py --all --save
```

### Other Test Scripts

*Additional test scripts have been archived in `backup_removed_files/` directory.*

## Basemap Support

Enhanced plotting supports two basemap types:

### 1. Contextily (Web Tiles) - Recommended

**Pros:**
- ✅ Beautiful grayscale Esri WorldGrayCanvas
- ✅ Auto-zoom to appropriate level
- ✅ Easy setup: `pip install contextily`
- ✅ Cached tiles for speed

**Cons:**
- ⚠️ Requires internet on first use
- ⚠️ Coordinate transformation overhead (UTM ↔ Web Mercator)

**Setup:**
```bash
python setup_basemaps.py --contextily
# or manually:
pip install contextily
```

### 2. Natural Earth (Shapefiles) - Offline

**Pros:**
- ✅ Works completely offline
- ✅ Stays in UTM (no reprojection)
- ✅ Fast and lightweight

**Cons:**
- ⚠️ More setup steps
- ⚠️ Less visually detailed

**Setup:**
```bash
python setup_basemaps.py --natural-earth
# or manually:
conda install geopandas
# Download from naturalearthdata.com
```

**See BASEMAP_SETUP.md and BASEMAP_SCRIPTS_README.md for details.**

## Data Management

### SSCOFS Model
- **Runs**: 00z, 03z, 09z, 15z, 21z UTC (5 times daily)
- **Forecasts**: Up to 72 hours (F000-F072)
- **Domain**: Salish Sea (Puget Sound, Strait of Juan de Fuca)
- **Resolution**: ~500m unstructured FVCOM grid
- **Availability**: 3-5 hours after cycle time

### Cache Directory

**Location**: `.sscofs_cache/` (auto-created in project root)

**Filename Format**: `sscofs_YYYYMMDD_tCCz_fHHH.nc`
- `YYYYMMDD`: Run date
- `CC`: Cycle hour (00, 03, 09, 15, 21)
- `HHH`: Forecast hour (000-072)

**Size**: ~200 MB per file

**Management:**
```bash
python sscofs_cache.py --list    # List files
python sscofs_cache.py --info    # Statistics
python sscofs_cache.py --clear   # Delete all
```

**Or through plotting scripts:**
```bash
python plot_local_currents.py --list-cache
python plot_local_currents.py --clear-cache
python plot_local_currents.py --no-cache  # Bypass cache
```

## Output Formats

### Basic Plotting (plot_local_currents.py)

Two-panel figure:
1. **Left**: Current vectors (quiver plot)
   - Arrows colored by speed (m/s)
   - Uniform subsampling
   - Reference arrow (1 m/s scale)
   - Red circle showing search radius
   
2. **Right**: Current speed distribution (scatter)
   - All points colored by speed (knots)
   - UTM coordinates (meters)

**Coordinates**: UTM Zone 10N (EPSG:32610)
**Units**: m/s for arrows, knots for scatter

### Enhanced Plotting (plot_currents_enhanced.py)

**Single-panel modes** (streamline, adaptive, both):
- Filled speed contours (background)
- Tactical threshold contours (white lines: 0.5, 1.0, 1.5, 2.0 kt)
- Streamlines or adaptive arrows (or both)
- Optional basemap
- Red center marker and radius

**Diagnostic mode** (two panels):
- **Left**: Speed + streamlines + adaptive arrows
- **Right**: 
  - Vorticity contours (rotation/circulation)
  - Okubo-Weiss = 0 contour (regime boundary)
  - Eddy cores shaded (cyan)
  - Faded speed contours (background)

**Coordinates**: UTM Zone 10N (EPSG:32610)
**Units**: Knots (for mariners)

## Environment

### Dependencies (currents_env.yml)

**Core:**
- Python 3.11
- numpy, matplotlib, scipy
- xarray, netcdf4, h5netcdf
- pandas, pyproj

**Data access:**
- s3fs (S3 bucket access)
- requests

**Optional (for basemaps):**
- contextily (`pip install contextily`)
- geopandas (`conda install geopandas`)
- cmocean (`pip install cmocean`)

### Installation

```bash
conda env create -f currents_env.yml
conda activate currents

# Optional basemap dependencies
pip install contextily
conda install geopandas
```

## Examples

### Quick Current Check
```bash
python plot_local_currents.py --lat 47.6 --lon -122.4 --radius 5
```

### Race Planning
```bash
# Full diagnostic for race area
python plot_currents_enhanced.py \
    --lat 47.65 --lon -122.35 --radius 3 \
    --style diagnostic --basemap contextily \
    --save race_2025-10-18.png

# Compare two time steps
python plot_currents_enhanced.py --time-index 0 --save t0.png
python plot_currents_enhanced.py --time-index 6 --save t6.png
```

### Different Regions
```bash
# Strait of Juan de Fuca
python plot_currents_enhanced.py --lat 48.5 --lon -123.0 --radius 10

# Central Puget Sound
python plot_currents_enhanced.py --lat 47.6 --lon -122.3 --radius 5

# North Sound
python plot_currents_enhanced.py --lat 48.0 --lon -122.5 --radius 8
```

### Cache Management
```bash
# See what's cached
python sscofs_cache.py --list

# Cache statistics
python sscofs_cache.py --info

# Clear old data
python sscofs_cache.py --clear
```

## Troubleshooting

### No data showing in plot
```bash
# Currents may be very weak (<0.2 knots)
# Try different location or time
python plot_local_currents.py --time-index 12
```

### Download fails
```bash
# Model may not be available yet (wait 3-5 hours after cycle)
# Try different time
python plot_local_currents.py --time-index 0  # nowcast

# Check internet connection
# Verify S3 access
```

### Cache issues
```bash
# Clear and re-download
python sscofs_cache.py --clear
python plot_local_currents.py --no-cache
```

### Basemap not showing
```bash
# Verify setup
python test_basemaps.py --check

# Test specific basemap
python test_basemaps.py --contextily --save

# Re-run setup
python setup_basemaps.py --all
```

### Import errors
```bash
# Verify environment
conda activate currents
conda list | grep xarray

# Reinstall if needed
conda env remove -n currents
conda env create -f currents_env.yml
```

## Documentation

- **BASEMAP_SETUP.md** - Detailed basemap installation guide (manual steps)
- **BASEMAP_SCRIPTS_README.md** - Quick reference for setup_basemaps.py and test_basemaps.py
- **ENHANCEMENTS.md** - Technical details on enhanced plotting features and algorithms

## Coordinate Systems

All Python scripts use **UTM Zone 10N (EPSG:32610)**:
- Origin: Center of zone at equator
- Units: Meters
- Suitable for Puget Sound region (122°W - 126°W)
- Accurate distances up to ~50 miles from zone center

**Why UTM?**
- ✅ Accurate distance representation
- ✅ No distortion for local areas
- ✅ Meters are intuitive for marine work
- ✅ Compatible with nautical charts

## Performance

**First download**: 5-10 seconds (200 MB NetCDF from S3)
**Cached plot**: 2-5 seconds
**Enhanced diagnostic**: 5-10 seconds (includes interpolation)

**Optimization tips:**
- Use cache (`--no-cache` only when needed)
- Smaller radius = faster (fewer points)
- Basic plotting faster than enhanced
- Contextily basemap adds 1-2 seconds

## Notes

- Currents in Puget Sound typically weak (0.1-0.6 knots) vs tidal areas
- FVCOM grid is unstructured with higher resolution near shore
- Model runs available ~3-5 hours after cycle time
- Longitude in dataset uses 0-360° convention (subtract 360 for negative)
- Coordinates are node-centered (not cell-centered)

## Related

- **Parent README**: `../README.md` - Project overview
- **Web Tools**: `../Web/README.md` - Interactive map viewer
- **SSCOFS Info**: https://tidesandcurrents.noaa.gov/ofs/sscofs/sscofs.html
- **NOAA Data**: S3 bucket `s3://noaa-nos-ofs-pds/sscofs/`

---

**For web-based visualization, see `../Web/README.md`**

