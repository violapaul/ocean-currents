# SSCOFS Data Pipeline

Automated pipeline for fetching NOAA SSCOFS ocean current data and exporting it as compact binary files for web visualization.

## Quick Start

```bash
# Setup environment
conda env create -f currents_env.yml
conda activate currents

# Generate data for Puget Sound region (fast mode, ~2 minutes)
python generate_current_data.py --mode fast --workers 12

# Generate and upload to S3
python generate_current_data.py --mode fast --workers 12 --upload --s3-bucket viola-ocean-currents
```

## Pipeline Modes

### Fast Mode (Recommended)

Uses **byte-range S3 reads** via `s3fs` to fetch only the variables needed (~3.4MB per file instead of ~200MB).

```bash
python generate_current_data.py --mode fast --workers 12 --hours 0-72
```

**Performance:**
- 73 hours in ~2 minutes
- ~250MB total data transfer (vs 14.6GB for full files)
- 12 parallel workers for maximum throughput

### Cache Mode (Offline/Debug)

Downloads full NetCDF files to local cache. Useful for offline development or when byte-range reads fail.

```bash
python generate_current_data.py --mode cache --hours 0-72
```

**Performance:**
- 73 hours in ~20+ minutes
- Downloads full ~200MB files
- Files cached in `.sscofs_cache/`

## Output Files

```
current_data/
├── latest.json                    # Points to current model run
└── {YYYYMMDD}_{HH}z/             # e.g., 20260224_15z/
    ├── manifest.json              # Metadata (bounds, element count, hours)
    ├── geometry.bin               # Float32 [lon,lat,...] gzipped (~1.5MB)
    ├── f000.bin                   # Float16 [u,v,...] for hour 0 (~1.1MB)
    ├── f001.bin                   # Float16 [u,v,...] for hour 1
    └── ...through f072.bin
```

### Data Format

**geometry.bin** (gzipped Float32, little-endian):
```
[lon0, lat0, lon1, lat1, lon2, lat2, ...]
```

**f{NNN}.bin** (gzipped Float16, little-endian):
```
[u0, v0, u1, v1, u2, v2, ...]  // velocity in m/s
```

**manifest.json**:
```json
{
  "model_run": "2026-02-24T15:00:00Z",
  "generated_at": "2026-02-24T21:44:48Z",
  "num_elements": 310778,
  "bounds": {
    "lat_min": 46.21,
    "lat_max": 49.02,
    "lon_min": -124.51,
    "lon_max": -121.48
  },
  "forecast_hours": [0, 1, 2, ..., 72],
  "format": {
    "geometry": "gzipped Float32 [lon0,lat0,lon1,lat1,...] little-endian",
    "velocity": "gzipped Float16 [u0,v0,u1,v1,...] little-endian, m/s"
  }
}
```

## Command Line Options

```bash
python generate_current_data.py [OPTIONS]

Options:
  --output DIR       Output directory (default: ./current_data)
  --hours RANGE      Forecast hour range, e.g., "0-72" or "0-24" (default: 0-72)
  --radius MILES     Radius from Seattle in miles (default: 100)
  --mode {fast,cache} Download mode (default: fast)
  --workers N        Parallel workers for fast mode (default: 10)
  --upload           Upload to S3 after generation
  --s3-bucket NAME   S3 bucket name for upload
  --s3-prefix PREFIX S3 key prefix (default: ocean-currents)
```

## Core Modules

### generate_current_data.py
Main pipeline script. Orchestrates data fetching, processing, and export.

### latest_cycle.py
Finds the latest available SSCOFS model cycle on NOAA S3.

```python
from latest_cycle import find_latest_cycle

run_date, cycle, keys = find_latest_cycle(max_days_back=3)
# run_date: datetime.date
# cycle: int (0, 3, 9, 15, or 21)
# keys: list of available S3 keys
```

### fetch_sscofs.py
Constructs URLs for SSCOFS NetCDF files.

```python
from fetch_sscofs import build_sscofs_url

url = build_sscofs_url("2026-02-24", 15, 1)
# https://noaa-nos-ofs-pds.s3.amazonaws.com/sscofs/netcdf/2026/02/24/sscofs.t15z.20260224.fields.f001.nc
```

### sscofs_cache.py
Manages local cache of full NetCDF files (for cache mode).

```bash
python sscofs_cache.py --list   # List cached files
python sscofs_cache.py --info   # Cache statistics
python sscofs_cache.py --clear  # Clear all cache
```

## SSCOFS Model Info

| Property | Value |
|----------|-------|
| **Provider** | NOAA/NOS/CO-OPS |
| **S3 Bucket** | `noaa-nos-ofs-pds` |
| **Model Cycles** | 00z, 03z, 09z, 15z, 21z UTC |
| **Forecast Range** | 0-72 hours |
| **Grid Type** | Unstructured FVCOM |
| **Total Elements** | 433,410 |
| **Puget Sound Region** | ~310,778 elements (100mi from Seattle) |
| **Variables Used** | u, v (velocity), lonc, latc (element centers) |
| **Surface Layer** | siglay=0 |

## GitHub Actions Integration

The pipeline is designed to run automatically via GitHub Actions:

```yaml
# .github/workflows/update-currents.yml
- name: Generate current data
  run: |
    python generate_current_data.py \
      --mode fast \
      --hours 0-72 \
      --workers 8 \
      --upload \
      --s3-bucket viola-ocean-currents
  env:
    AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
    AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
```

## Dependencies

```
numpy
xarray
h5netcdf
h5py
s3fs
boto3
requests
aiohttp
```

Install via:
```bash
pip install -r requirements.txt
# or
conda env create -f currents_env.yml
```

## Troubleshooting

### "Failed hours" in output

Some forecast hours may fail if NOAA hasn't published them yet. This is normal for recent model runs - the pipeline still succeeds with available hours.

### Slow byte-range reads

The s3fs connection can be slow initially. The pipeline uses optimized settings:
- 8MB block size
- Connection pooling (50 connections)
- Fill cache enabled

### Cache mode for offline work

If byte-range reads fail or you need offline access:
```bash
python generate_current_data.py --mode cache
```

This downloads full files but takes ~10x longer.
