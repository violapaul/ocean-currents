# Coastline Tools

Reusable scripts for building and tuning shoreline GeoJSON overlays used by the mobile viewer.

## What These Scripts Do

- `fetch_wa_ecology_coastline.py`
  - Fetches high-detail shoreline lines from WA Ecology Coastal Atlas layer 13
  - Handles ArcGIS object-id pagination and writes a raw GeoJSON

- `simplify_coastline.py`
  - Converts raw shoreline lines into a compact MultiLineString GeoJSON
  - Applies bbox clipping + point thinning + short-segment filtering

- `experiment_simplification.py`
  - Generates multiple simplified variants for quick quality/performance comparisons
- `build_viewer_coastline.py`
  - End-to-end prep for the web viewer:
    clip + simplify + stitch + split-long-lines + precompute bbox per chunk

## Typical Workflow

From `OceanCurrents/Python_SSCOFS`:

```bash
# 1) Fetch raw high-detail shoreline data
python coastline_tools/fetch_wa_ecology_coastline.py \
  --output data/shoreline_wa_ecology_raw.geojson

# 2) Build viewer-ready shoreline file
python coastline_tools/simplify_coastline.py \
  --input data/shoreline_wa_ecology_raw.geojson \
  --output data/shoreline_puget.geojson \
  --tolerance-m 14 \
  --min-length-m 40

# 3) (Optional) Stitch nearby segment endpoints to improve continuity
python coastline_tools/stitch_coastline.py \
  --input data/shoreline_puget.geojson \
  --output data/shoreline_puget_stitched.geojson \
  --snap-tol-m 8

# 4) (Optional) Generate variants for tuning
python coastline_tools/experiment_simplification.py \
  --input data/shoreline_wa_ecology_raw.geojson \
  --out-dir data/experiments

# 5) Build final viewer file with precomputed bboxes
python coastline_tools/build_viewer_coastline.py \
  --input data/shoreline_wa_ecology_raw.geojson \
  --output data/shoreline_puget.geojson \
  --tolerance-m 14 \
  --min-length-m 40 \
  --snap-tol-m 8 \
  --max-chunk-len-m 900
```

## Notes

- Coordinates are expected in WGS84 lon/lat.
- Processing thresholds are in **meters** using UTM (default `EPSG:32610`).
- Requires `pyproj` for UTM transforms:
  - `conda install -c conda-forge pyproj`
- Start with conservative settings, then increase tolerance until visual quality starts to degrade.
- Stitching is best done after simplification to keep runtime files compact.
- The viewer performs fastest when the final file has medium-length chunks
  with precomputed bboxes (produced by `build_viewer_coastline.py`).
