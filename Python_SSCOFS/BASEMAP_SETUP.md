# Basemap Setup Guide

The enhanced plotting script `plot_currents_enhanced.py` supports optional basemaps for geographic context. This guide shows how to set them up.

## Quick Setup (Recommended)

Use the automated setup script:

```bash
# Setup everything automatically
python setup_basemaps.py --all

# Then test it works
python test_basemaps.py --all
```

The setup script will:
- ✅ Install contextily (if you want web tiles)
- ✅ Download Natural Earth coastline data
- ✅ Clip data to Puget Sound region (faster rendering)
- ✅ Verify everything is working

---

## Manual Setup (if you prefer)

Follow the detailed instructions below for manual installation.

## Option 1: Web Tiles (Contextily) - Recommended

**Best for:** Quick setup, beautiful maps, minimal configuration

### Installation

```bash
conda activate currents
pip install contextily
```

### Usage

```bash
python plot_currents_enhanced.py --basemap contextily --style streamline
```

### Features
- ✅ Automatic tile fetching from Esri WorldGrayCanvas
- ✅ Auto-zoom to appropriate level
- ✅ Handles coordinate transformation automatically (UTM ↔ Web Mercator)
- ✅ Cached tiles for faster repeated use
- ✅ Semi-transparent (50%) to not overwhelm current data
- ⚠️ Requires internet connection on first use

### What It Looks Like
- Clean grayscale background showing land, water, and major features
- Coastlines, islands, and major landmarks visible
- Professional appearance suitable for presentations

---

## Option 2: Natural Earth Shapefiles - For Offline Use

**Best for:** Offline use, minimal data, full control

### Installation

1. **Install geopandas:**
```bash
conda activate currents
conda install geopandas
```

2. **Download coastline data:**

Go to [Natural Earth Data](https://www.naturalearthdata.com/) and download:
- **10m Coastline** (recommended): https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-coastline/
  - File: `ne_10m_coastline.zip`
- OR **10m Land** (alternative): https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-land/

3. **Create data directory and extract:**
```bash
cd /path/to/WaysWaterMoves
mkdir -p data
cd data
# Extract your downloaded zip file here
unzip ~/Downloads/ne_10m_coastline.zip
```

Your directory structure should look like:
```
WaysWaterMoves/
├── data/
│   ├── ne_10m_coastline.shp
│   ├── ne_10m_coastline.shx
│   ├── ne_10m_coastline.dbf
│   └── ne_10m_coastline.prj
├── plot_currents_enhanced.py
└── ...
```

### Alternative: Puget Sound Specific Data

For a smaller, focused shapefile of just Puget Sound:

1. Export from [OpenStreetMap](https://www.openstreetmap.org/) or use [QGIS](https://qgis.org/)
2. Save as `data/shoreline_puget.geojson`

The script will automatically look for:
- `data/shoreline_puget.geojson` (first priority)
- `data/ne_10m_coastline.shp` (second priority)
- `data/coastline.geojson` (third priority)

### Usage

```bash
python plot_currents_enhanced.py --basemap natural_earth --style streamline
```

### Features
- ✅ Works completely offline
- ✅ Stays in UTM projection (no reprojection needed = faster)
- ✅ Lightweight and fast
- ✅ Full control over what's displayed
- ⚠️ Requires one-time setup
- ⚠️ Less visually detailed than web tiles

---

## Comparison: Which Should I Use?

| Feature | Contextily | Natural Earth |
|---------|-----------|---------------|
| **Setup time** | 30 seconds | 5-10 minutes |
| **Internet required** | First use only | No |
| **Visual quality** | Excellent | Good |
| **Speed** | Fast | Very fast |
| **File size** | ~0 (cached tiles) | ~20-50 MB |
| **Best for** | Day-to-day use | Offline/field use |

### Recommendation

1. **Start with Contextily** - easiest to get going
2. **Add Natural Earth later** if you need offline capability

---

## Troubleshooting

### Contextily Issues

**Error: "contextily not installed"**
```bash
pip install contextily
```

**Error: "Could not add basemap" or connection issues**
- Check internet connection
- Tiles may be temporarily unavailable
- Try again in a few minutes
- As fallback, use `--basemap none` or try Natural Earth

**Slow tile fetching**
- First fetch for a new area downloads tiles (1-2 seconds)
- Subsequent uses are cached and much faster
- Zoom level auto-adjusts; larger areas = slower initial fetch

### Natural Earth Issues

**Error: "geopandas not installed"**
```bash
conda install geopandas
```

**Error: "No shoreline data found"**
- Check that shapefiles are in `data/` directory
- Verify filenames match what the script expects
- See directory structure above

**Coastlines don't show up**
- Ensure shapefile has `.shp`, `.shx`, `.dbf`, `.prj` files
- Try loading in QGIS to verify it's valid
- Check CRS is specified (should work with any CRS, will be reprojected to UTM 10N)

---

## Advanced: Custom Basemaps

### Using Different Contextily Providers

Edit `plot_currents_enhanced.py`, line ~114:
```python
# Change this line:
source=ctx.providers.Esri.WorldGrayCanvas,

# To one of these:
source=ctx.providers.OpenStreetMap.Mapnik,           # OSM standard
source=ctx.providers.Stamen.Terrain,                 # Terrain view
source=ctx.providers.CartoDB.Positron,               # Light clean
source=ctx.providers.Esri.WorldImagery,              # Satellite
```

See all providers: https://contextily.readthedocs.io/en/latest/providers_deepdive.html

### Adjusting Basemap Transparency

Edit `plot_currents_enhanced.py`, line ~467:
```python
# Change alpha value (0.0 = invisible, 1.0 = opaque):
add_basemap(ax1, basemap_type=basemap, zoom='auto', alpha=0.3)  # lighter
add_basemap(ax1, basemap_type=basemap, zoom='auto', alpha=0.7)  # darker
```

---

## Testing Your Setup

Quick test to verify basemaps work:

```bash
# Test contextily (if installed)
python plot_currents_enhanced.py --basemap contextily --radius 3 --save test_contextily.png

# Test natural_earth (if installed)
python plot_currents_enhanced.py --basemap natural_earth --radius 3 --save test_naturalearth.png
```

Check the saved PNG files - you should see current overlays on top of coastline/land features.

---

## No Basemap? No Problem!

The script works perfectly without any basemap:

```bash
python plot_currents_enhanced.py  # defaults to no basemap
```

The current visualizations (streamlines, contours, diagnostics) are the primary feature. Basemaps just add geographic context.

