# Weather Download Attempts and Failures

Date: 2026-03-08 (America/Los_Angeles)
Project path: `/Users/viola/Resilio Sync/Dropbox/Sailboat/Code/WaysWaterMoves/OceanCurrents/Python_SSCOFS`

## Goal
Run route planning end-to-end with dynamic Open-Meteo ECMWF wind download and use that wind in routing.

## What Was Tried

1. End-to-end route run with fresh downloads:

```bash
'.venv/bin/python' run_route.py routes/shilshole_alki_return.yaml --no-plots --no-cache
```

Observed behavior:
- SSCOFS current files downloaded successfully.
- Open-Meteo node discovery succeeded (`9` unique nodes).
- Open-Meteo wind fetch succeeded at HTTP/API level and wrote:
  - `.wind_cache/shilshole_alki_return_ecmwf_nodes.csv`
  - `.wind_cache/shilshole_alki_return_ecmwf_wind.nc`
- Routing failed with:
  - `RuntimeError: No route found -- end point is unreachable.`
- Warnings during wind field creation:
  - `RuntimeWarning: Mean of empty slice` in `sail_routing.py`.

2. Inspected generated NetCDF wind output:

```bash
'.venv/bin/python' - <<'PY'
import xarray as xr, numpy as np
p='.wind_cache/shilshole_alki_return_ecmwf_wind.nc'
ds=xr.open_dataset(p)
for v in ['wind_speed_10m','wind_direction_10m','wind_gusts_10m']:
    a=ds[v].values
    print(v, np.isnan(a).sum(), a.size)
PY
```

Result:
- `wind_speed_10m`: `423 / 423` NaN
- `wind_direction_10m`: `423 / 423` NaN
- `wind_gusts_10m`: `423 / 423` NaN

3. Direct Open-Meteo API probe (single coordinate) with required params:

```text
models=ecmwf_ifs04
cell_selection=nearest
elevation=nan
hourly=wind_speed_10m,wind_direction_10m,wind_gusts_10m
wind_speed_unit=kn
timezone=America/Los_Angeles
```

Result:
- HTTP status `200`
- `hourly` arrays returned, but values were all `None`.

4. Model comparison probe at same coordinate:

- `ecmwf_ifs04`: all `None` (`0` non-null out of `168`)
- `ecmwf_ifs025`: valid values (`168` non-null out of `168`)
- `gfs_seamless`: valid values (`168` non-null out of `168`)
- auto (no model): valid values (`168` non-null out of `168`)

## What Failed
- Weather download did not fail at transport level; it failed at data quality level for `ecmwf_ifs04`.
- The pipeline saved structurally valid files containing all-missing wind data.
- Downstream routing then failed because wind field data was unusable.

## Notes on Code-Level Fixes Already Applied
- Multi-coordinate elevation formatting was corrected (`elevation=nan` per coordinate) to avoid Open-Meteo 400 errors.
- Time parsing/cleanup was hardened in `ecmwf_wind.py` (coercion + dropping invalid timestamps + duplicate `(time,node)` handling).

## Current Working Hypothesis
- `ecmwf_ifs04` is currently returning null wind values in this environment/API access mode.
- Likely causes include model availability/entitlement changes or API-side behavior change.

## Recommended Next Steps
1. Add a non-null coverage check after fetch; fail fast if coverage is below threshold.
2. Add model fallback order in config (example: `ecmwf_ifs04 -> ecmwf_ifs025 -> gfs_seamless`).
3. If strict 9km is mandatory, keep `ecmwf_ifs04` only and surface explicit failure with actionable message instead of writing all-NaN datasets.
