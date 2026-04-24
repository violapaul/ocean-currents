# Polar Sweep Optimizations in `sail_routing.py`

**Date:** March 2026  
**Related transcript:** [Polar sweep optimization session](04c6d497-d610-446e-a193-efe1584a462b)

---

## Background

The A* router (`_sector_astar_jit`) must evaluate the optimal boat heading for every
candidate edge it considers. This is done by a **polar sweep**: iterate over a set of
candidate headings, compute boat speed from the polar table, add current vector, measure
SOG along the desired edge track, and keep the best one. This sweep is the innermost loop
of the routing kernel and dominates runtime.

---

## Original Sweep (baseline)

```
_SWEEP_DEGS  – 180 headings covering 0°–358° at 2° resolution (indices 0–179)
_SWEEP_RADS  – same in radians
_SWEEP_HX/HY – unit vector components for each heading
```

For each heading the original code performed:
1. Compute TWA (angle between heading and wind direction)
2. **Binary search** on the sparse TWA axis of the polar table (typically 14 points)
3. **Binary search** on the sparse TWS axis (typically 7–8 points)
4. **4-point bilinear interpolation** from the four surrounding table cells

This is repeated ~180 times per edge evaluation (or ~36 times with `coarse_step=5`).
The two binary searches were identified as the dominant cost.

---

## Optimization 1 — Pre-baked Dense Polar Table (`use_dense_polar`)

### Idea

Pre-interpolate the sparse polar table (14 TWA × 8 TWS points) into a dense array of
shape `(182, max_tws+2)` at **1° TWA × 1 kt TWS** resolution, once at startup.  During
the sweep, replace the two binary searches + bilinear interpolation with direct integer
indexing + a single bilinear step (no search at all):

```python
i_d  = int(twa_deg)          # integer TWA degree (0–181)
j_d  = int(tws_kt)           # integer TWS knot bucket
alpha = twa_deg - i_d        # sub-degree fraction
beta  = tws_kt  - j_d        # sub-knot fraction
V_kt  = bilinear(polar_dense, i_d, j_d, alpha, beta)
```

### Implementation

- `PolarTable._build_dense_lookup()` — called from `__init__`, builds `_polar_dense`
  (numpy float32 array) and `_polar_dense_max_tws`.
- `_sector_astar_jit` — new parameters `use_dense_polar`, `polar_dense`,
  `polar_dense_max_tws`; the dense path is taken when `use_dense_polar=True`.
- `SectorRouter.__init__` — new kwarg `use_dense_polar`; assembles the dense arrays
  and passes them to the JIT kernel.
- `run_route.py` — reads `use_dense_polar` from YAML config.

### YAML flag

```yaml
routing:
  use_dense_polar: true
```

### Benchmark results (TTP route, `coarse_step=5`)

| Config | Headings/edge | A* search time |
|---|---|---|
| Baseline (sparse + `coarse_step=5`) | ~36 | ~8.5 s |
| Dense + `coarse_step=5` | ~36 | ~5.2 s |
| **Speedup** | | **~1.6×** |

Route quality (distance, ETA) was identical to baseline.

---

## Optimization 2 — Forward-Hemisphere Filter (`use_dot_filter`, Experimental)

### Original mathematical basis

For any edge, define:
- `d_hat` = unit vector from current node to neighbour node (desired ground track)
- `c_par = cu·d_hat_x + cv·d_hat_y` — component of current **along** the track

The SOG along the track for heading `h` is:

```
sog(h) = V_ms(h) × dot(h, d_hat) + c_par
```

`c_par` is **constant across all headings** — it is a property of the current at the
edge, not of boat heading.  Therefore:

- Any **backward heading** (dot < 0):  `V_ms × dot ≤ 0`  →  `sog ≤ c_par`
- Any **forward heading** (dot ≥ 0):   `V_ms × dot ≥ 0`  →  `sog ≥ c_par`

The perpendicular heading (dot = 0) gives exactly `sog = c_par`, so no backward
heading can beat an otherwise identical forward heading on raw along-track SOG.

> **Key insight:** the question is not "can a strong current carry me to the mark while
> pointing away from it?" (yes it can, but only as well as pointing perpendicular). The
> question is "can pointing backward ever beat pointing forward?" — it cannot, because
> the current's c_par contribution is heading-independent; only the polar term changes,
> and that is always maximised by being in the forward hemisphere.

In practice the edge accept/reject logic also includes cross-track drift,
upwind/downwind TWA no-go bands, and current-induced ground-track constraints.
Those extra constraints mean the filter is no longer treated as production-safe
without route-specific validation. Production YAMLs keep `use_dot_filter: false`.

### Implementation

- `_sector_astar_jit` — `use_dot_filter` parameter. At the start of the coarse
  sweep loop (both dense and sparse paths):

```python
if use_dot_filter and (sweep_hx[h] * d_hat_x + sweep_hy[h] * d_hat_y) < 0.0:
    continue
```

- The dot filter works with **both** the dense and sparse polar paths.
  It is a simple multiply-add that is independent of the polar lookup strategy.
- The refine pass (Pass 2) does **not** apply the filter — the refine window is small
  and centred on the coarse winner which already passed.
- `SectorRouter.__init__` — `use_dot_filter` kwarg; stored as `_use_dot_filter`.
- `run_route.py` — reads `use_dot_filter` from YAML config.

### YAML flag

```yaml
routing:
  use_dot_filter: false
  # works with or without use_dense_polar
  polar_sweep_coarse_step: 1    # exact sweep, no coarse approximation
```

### Benchmark results (TTP route, Leg 1 A* search time)

| Config | Headings/edge | A* search | Route quality |
|---|---|---|---|
| dense + `coarse_step=5` | ~36 | ~5.2 s | baseline |
| dot+dense + `coarse_step=1` (exact) | ~90 | ~12.4 s | 29.15 nm / 234 min |
| dot+dense + `coarse_step=2` | ~45 | ~6.3 s | 29.15 nm / 234 min |
| dot+dense + `coarse_step=3` | ~30 | ~4.7 s | 29.14 nm / 234 min |

`coarse_step=2` halves A* time vs the exact sweep with no measurable quality loss.
`coarse_step=3` is the fastest config tested, and route quality is still within 0.01 nm
of exact. `coarse_step=5` (without dot filter) can miss the optimal heading when it falls
between coarse samples, producing marginally worse routes on some legs.

---

## Cross-track drift constraint

In addition to the polar sweep optimizations, the heading evaluator applies a
**cross-track drift filter**: a heading is only accepted if its cross-track ground
velocity (drift) does not exceed **10%** of its along-track progress:

```python
drift_h = abs(gx * (-d_hat_y) + gy * d_hat_x)
prog    = max(sog_h, 1e-6)
if sog_h > 0.01 and drift_h <= 0.10 * prog:
    # heading accepted
```

This corresponds to a maximum crab angle of ~5.7° (`arctan(0.10)`).

### Why this matters

The drift constraint rejects headings **even in the forward hemisphere**. A loose
threshold can allow pseudo-physical routes where the boat reaches quickly while
the ground track points into the no-go zone. The current production setting is
the stricter `0.10` value, matching `_solve_heading()` and the SectorRouter
Numba kernel.

---

## Summary of flags and their interactions

| Flag | Effect | Requires |
|---|---|---|
| `use_dense_polar: true` | Replace binary-search polar lookup with direct array index | — |
| `use_dot_filter: true` | Experimental: skip ~90 backward headings per edge | route-specific validation |
| `polar_sweep_coarse_step: N` | Only sample every Nth heading in Pass 1, refine ±N around winner | — |

Estimated headings evaluated per edge:

| Dot filter | coarse_step | Headings/edge |
|---|---|---|
| off | 1 | ~180 |
| off | 5 | ~36 |
| on | 1 | ~90 |
| on | 2 | ~45 |
| on | 3 | ~30 |
| on | 5 | ~18 |

Recommended production settings:
- **Recommended:** `use_dense_polar: true`, `use_dot_filter: false`, `coarse_step: 3`
- **Exact comparison:** `use_dense_polar: true`, `use_dot_filter: false`, `coarse_step: 1`

On the TTP benchmark route, `coarse_step=3` is the fastest config and matches or beats
`coarse_step=1` and `coarse_step=2` on route quality — the 6° coarse pass + 2° refine
window is sufficient to find the optimal heading.

> **Note:** the dense polar lookup is exact. The dot filter remains available as an
> experimental optimization, but production YAMLs keep it disabled until tests cover
> the drift/no-go edge cases that can make backward boat headings feasible.

---

## Files changed

| File | Change |
|---|---|
| `sail_routing.py` | `PolarTable._build_dense_lookup()`; `_sector_astar_jit` dense-polar params; production drift threshold `0.10`; optional experimental dot filter |
| `run_route.py` | Parses `use_dense_polar`, `use_dot_filter`, and route YAML performance controls |
| `routes/*.yaml` | Production route configs use dense polar, coarse step 3, and `use_dot_filter: false` |

---

## Possible future work

- **Variable sweep resolution near the beat angle:** the most useful headings are
  clustered near ±38° from the wind and near beam reach; a non-uniform sweep could
  concentrate samples there.
- **Edge-level caching:** if two adjacent edges share the same wind/current cell,
  the best heading could be reused. Correctness is subtle with time-varying wind.
- **Pre-rotate sweep table per wind direction:** compute `sweep_twa_deg[]` once per
  wind cell instead of doing modular arithmetic per heading per edge.
- **Pass 1/Pass 2 code deduplication:** the dense coarse sweep and refine pass share
  ~20 lines of evaluation code. A Numba `@njit(inline='always')` helper could reduce
  duplication, but the shared SOG/drift evaluation is only ~5 lines and requires 7+
  locals as parameters, so the benefit is marginal in practice.
