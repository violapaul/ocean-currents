# Ocean Currents Mobile Viewer

A mobile-optimized Progressive Web App (PWA) for viewing real-time ocean current forecasts from NOAA's Salish Sea Coastal Ocean Forecast System (SSCOFS) with tide data integration.

Live app: https://violapaul.github.io/ocean-currents/map-viewer-mobile.html

## 🌊 Features

- **Self-hosted data pipeline** - No dependency on unreliable third-party tile servers
- **Automatic updates** - GitHub Actions refreshes data every 6 hours
- **Canvas-based rendering** - Fast, smooth vector visualization with 310K+ mesh elements
- **Speed heatmap** - Color-coded water showing current speed
- **Tide integration** - Seattle tide chart with current time indicator
- **Mobile-first PWA** - Installable on iOS and Android
- **No proxy needed** - Direct fetch from S3 and NOAA APIs (both support CORS)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  NOAA S3 Bucket (noaa-nos-ofs-pds)                             │
│  └── SSCOFS NetCDF files (~200MB each, 73 hours/cycle)         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Byte-range reads (~3MB/file)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  GitHub Actions (every 6 hours)                                 │
│  └── generate_current_data.py --mode fast                       │
│      - Extracts u,v,lonc,latc at surface layer                 │
│      - 8 parallel workers in GitHub Actions                     │
│      - ~2 minutes total runtime                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Upload (boto3)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  AWS S3 (viola-ocean-currents) - public read, CORS enabled     │
│  └── ocean-currents/                                            │
│      ├── latest.json                                            │
│      ├── {run_tag}/manifest.json                                │
│      ├── {run_tag}/geometry.bin   (1.5MB, gzipped)             │
│      ├── {run_tag}/water_boundary.geojson (~600KB)             │
│      └── {run_tag}/f000-f072.bin  (~1.1MB each, gzipped)       │
└─────────────────────────────────────────────────────────────────┘
          │                                           │
          │ Direct fetch (CORS)                       │
          ▼                                           │
┌─────────────────────────────────────────────────────────────────┐
│  Browser (map-viewer-mobile.html)                               │
│  └── Canvas renderer draws 310K vectors                         │
│  └── Speed heatmap + direction arrows                           │
│  └── Spatial indexing for fast queries                          │
└─────────────────────────────────────────────────────────────────┘
          │
          │ Direct fetch (CORS)
          ▼
┌─────────────────────────────────────────────────────────────────┐
│  NOAA Tides API (api.tidesandcurrents.noaa.gov)                │
│  └── Seattle tide predictions (Station 9447130)                 │
└─────────────────────────────────────────────────────────────────┘
```

## Data Pipeline

### Automatic Updates (No Mac Needed!)

GitHub Actions runs every 6 hours to fetch fresh NOAA data:

| Schedule (UTC) | Catches Model Cycle |
|----------------|---------------------|
| 01:30          | 21z (previous day)  |
| 07:30          | 03z                 |
| 13:30          | 09z                 |
| 19:30          | 15z                 |

**Monitor**: https://github.com/violapaul/WaysWaterMoves/actions

### Manual Run

The data-pipeline source lives in the private `violapaul/WaysWaterMoves` repo
under `race_routing/` and is not mirrored here.

### Performance

| Metric | Old (full download) | New (byte-range) |
|--------|---------------------|------------------|
| Data per file | ~200 MB | ~3.4 MB |
| Total (73 hours) | ~14.6 GB | ~250 MB |
| Time | ~22 min | ~2 min |

## Files

This repo contains **only the viewer**. The data-pipeline and sailboat-
routing source live in the private `violapaul/WaysWaterMoves` repo under
`race_routing/`; the viewer consumes their published artifacts via S3.

```
ocean-currents/
├── map-viewer-mobile.html    # Main PWA with Canvas renderer
├── manifest.json             # PWA manifest
├── service-worker.js         # Offline caching
├── index.html                # Splash + install prompt
├── update-helper.html        # PWA cache-flush utility
├── app-icon.svg              # App icon
├── app-icon-192.png          # PWA icon
├── app-icon-512.png          # PWA icon
└── apple-touch-icon.png      # iOS install icon
```

## Development

### Local Testing

```bash
# From the repo root
python3 -m http.server 8080

# Open http://localhost:8080/map-viewer-mobile.html
```

### Debug Mode

Add `?debug` to URL for verbose logging:
```
http://localhost:8080/map-viewer-mobile.html?debug
```

## URLs

| Component | URL |
|-----------|-----|
| **Live App** | https://violapaul.github.io/ocean-currents/map-viewer-mobile.html |
| **S3 Data** | https://viola-ocean-currents.s3.us-west-2.amazonaws.com/ocean-currents/ |
| **NOAA Tides** | https://api.tidesandcurrents.noaa.gov/api/prod/datagetter |
| **Actions** | https://github.com/violapaul/WaysWaterMoves/actions |

## Deployment

Pushing to `main` here triggers a GitHub Pages rebuild (~30–60s) at
https://violapaul.github.io/ocean-currents/. AWS/IAM/GHA setup for the data
pipeline lives in the private `violapaul/WaysWaterMoves` repo's
`DEPLOYMENT.md`.

## Data Format

### geometry.bin (gzipped Float32)
```
[lon0, lat0, lon1, lat1, ...] // 310K elements × 2 coords = 1.5MB compressed
```

### f{NNN}.bin (gzipped Float16)
```
[u0, v0, u1, v1, ...] // velocity in m/s, ~1.1MB compressed per hour
```

### water_boundary.geojson
GeoJSON MultiPolygon defining the water domain boundary, derived from Delaunay
triangulation of SSCOFS element centers. Used for:
- **Land detection** — rasterized to bitmap for O(1) `isWater(lon, lat)` lookups
- **Coastline rendering** — polygon edges drawn as shoreline overlay

Triangles with edges longer than 3.5× local mesh density are classified as
"land-spanning" and excluded from the water domain.

### manifest.json
```json
{
  "model_run": "2026-02-24T15:00:00Z",
  "num_elements": 310778,
  "bounds": { "lat_min": 46.2, "lat_max": 49.0, "lon_min": -124.5, "lon_max": -121.5 },
  "forecast_hours": [0, 1, 2, ..., 72]
}
```

## Credits

- **Ocean Data**: NOAA/NOS/CO-OPS SSCOFS
- **Tides**: NOAA CO-OPS (Station 9447130 - Seattle)
- **Map Library**: MapLibre GL JS
- **Storage**: AWS S3
