# Sailboat Routing — Design and Algorithms

`sail_routing.py` finds the time-optimal path for a sailboat through
SSCOFS ocean current forecasts, with optional polar-based sail performance
and wind fields.

---

## Contents

1. [Goals and Constraints](#1-goals-and-constraints)
2. [Architecture Overview](#2-architecture-overview)
3. [Data Layer: CurrentField](#3-data-layer-currentfield)
4. [Polar Performance: PolarTable](#4-polar-performance-polartable)
5. [Wind: WindField](#5-wind-windfield)
6. [Boat Model: BoatModel](#6-boat-model-boatmodel)
7. [Edge Cost Physics](#7-edge-cost-physics)
8. [A* Router](#8-a-router)
9. [Path Smoothing and Stub Removal](#9-path-smoothing-and-stub-removal)
10. [Ground-Track Simulation](#10-ground-track-simulation)
11. [CLI Usage](#11-cli-usage)
12. [Testing](#12-testing)
13. [Known Limitations and Future Work](#13-known-limitations-and-future-work)

---

## 1. Goals and Constraints

**Goal:** Given a start and end position (lat/lon), find the route that
minimises total elapsed time for a sailboat, accounting for:

- **Ocean currents** — SSCOFS unstructured-mesh FVCOM data, up to 72 hourly
  forecast frames.
- **Sail performance** — optionally, a J105 (or any) polar table that maps
  True Wind Angle (TWA) × True Wind Speed (TWS) → boat speed through water.
- **Wind field** — constant, spatially varying, or time-varying.
- **Land avoidance** — any point outside the SSCOFS mesh (i.e., over land)
  is impassable.
- **Tacking cost** — abrupt course changes incur a configurable time penalty
  to produce realistically smooth routes.

**Scope:** Puget Sound and surrounding waters (~300 m grid cells, typical
runs take 1–10 seconds on a laptop).

---

## 2. Architecture Overview

```
                 ┌─────────────────────────────────────────────────────┐
                 │                   sail_routing.py                    │
                 │                                                       │
  SSCOFS NetCDF  │  CurrentField ──────────────────────────────────┐    │
  (hourly u,v)──►│  (KD-tree IDW,                                  │    │
                 │   time interpolation)                            ▼    │
                 │                                               Router  │
  Polar CSV ────►│  PolarTable ──► BoatModel ──► edge cost ───►  (A*)   │──► Route
  (TWA,TWS,Bsp)  │  (bilinear interp)            physics          │    │
                 │                                                  │    │
  Wind params ──►│  WindField ─────────────────────────────────────┘    │
                 │  (constant / grid / temporal)                        │
                 └─────────────────────────────────────────────────────┘
```

**Key classes:**

| Class | Role |
|-------|------|
| `CurrentField` | Stores SSCOFS velocity data; answers `(u, v)` at any UTM point and time via KD-tree IDW + temporal linear interpolation |
| `PolarTable` | Stores boat speed vs TWA/TWS from CSV; bilinear interpolation |
| `WindField` | Stores wind velocity; constant, spatial-grid, or temporal modes |
| `BoatModel` | Wraps polar + fallback fixed speed; single `speed()` method |
| `Router` | Builds a UTM grid, runs time-dependent A\*, smooths, simulates |
| `Route` | Result container with waypoints, leg times, SOG stats |

---

## 3. Data Layer: CurrentField

### Input

SSCOFS is an unstructured FVCOM mesh — element centers are scattered points
(not a regular grid), stored as 1-D arrays `lonc`, `latc`, `u`, `v`.
Multiple time frames (one per forecast hour) are loaded as `u_frames`,
`v_frames`.

### KD-tree construction

Element centers are projected to UTM and indexed in a `scipy.spatial.cKDTree`
for fast nearest-neighbour lookup.

### Spatial interpolation: IDW

To query velocity at an arbitrary UTM point, the `k=6` nearest SSCOFS
elements are found.  Velocity is the inverse-distance weighted average of
those elements' velocities:

```
w_i = 1 / dist_i          (w_i = 1e12 if exactly on an element centre)
(u, v) = Σ w_i * (u_i, v_i) / Σ w_i
```

**Land detection:** If the nearest element is farther than `land_threshold_m`
(default 750 m), the point is treated as land and `(NaN, NaN)` is returned.
This is tuned so that query points just offshore remain water-connected even
at coarse grid resolutions.

### Temporal interpolation

Between two adjacent time frames at `t0` and `t1`:

```
α = (t - t0) / (t1 - t0)
(u, v) = (1 - α) * (u0, v0) + α * (u1, v1)
```

Queries outside the frame range are clamped to the first/last frame (no
extrapolation).

---

## 4. Polar Performance: PolarTable

The polar CSV has three columns: `TWA_deg`, `TWS_kt`, `BoatSpeed_kt`.
The J105 table covers TWA 0–180° and TWS 0–60 kt.

### Bilinear interpolation

Speed at any `(twa, tws)` is bilinear-interpolated over the four surrounding
grid points `(i0,j0)`, `(i0,j1)`, `(i1,j0)`, `(i1,j1)`:

```
α = (twa - twa[i0]) / (twa[i1] - twa[i0])    # fraction along TWA axis
β = (tws - tws[j0]) / (tws[j1] - tws[j0])    # fraction along TWS axis

speed = (1-α)(1-β)*s[i0,j0] + α(1-β)*s[i1,j0]
       + (1-α) β  *s[i0,j1] + α  β  *s[i1,j1]
```

Values outside the table range are clamped (no extrapolation).

### True Wind Angle (TWA)

TWA is the angle between the boat's heading and the direction the wind is
coming **from**, expressed as [0°, 180°].  The math frame used throughout
is CCW from east (i.e., `atan2(north, east)`), matching UTM coordinates:

```python
wind_from_rad = atan2(-wind_v, -wind_u)   # reverse the "blows TO" vector
delta = heading_rad - wind_from_rad
delta = (delta + π) % (2π) - π           # normalise to [-π, π]
TWA = degrees(|delta|)                    # symmetric: port = starboard
```

---

## 5. Wind: WindField

Three modes, selected by the constructor used:

| Mode | Constructor | Description |
|------|-------------|-------------|
| Constant | `WindField.from_met(speed_kt, from_deg)` | Same wind everywhere at all times |
| Spatial grid | `WindField.from_grid(xs, ys, wu_grid, wv_grid)` | Spatially varying, constant in time; `scipy.interpolate.RegularGridInterpolator` |
| Temporal | `WindField.from_frames(xs, ys, wu_frames, wv_frames, times_s)` | Per-frame (optionally also spatially varying); linearly interpolated in time |

Wind components `(wu, wv)` are in m/s and follow the oceanographic "blows
**TO**" convention, matching `CurrentField`.  The CLI `--wind-direction`
flag uses the meteorological "from" convention and is converted internally.

---

## 6. Boat Model: BoatModel

```
Without polar:   speed through water = base_speed_ms  (constant)
With polar:      speed through water = PolarTable.speed_ms(TWA, TWS)
```

The `speed(heading_rad, wind_u, wind_v)` method computes TWA from the
heading and wind vector, then looks up the polar.  If any argument is
missing it falls back to the base speed.

---

## 7. Edge Cost Physics

Each A\* edge connects two adjacent grid cells.  Computing its cost
(travel time in seconds) requires knowing the speed-over-ground (SOG)
along that edge.

### No polar (fixed speed through water)

The boat aims to cancel the cross-track component of the current while
maintaining heading toward the next cell ("crabbing"):

```
c_par    = current · d̂          # along-track current component
c_perp   = current - c_par * d̂  # cross-track component (must be cancelled)
|c_perp| ≥ boat_speed → impassable (np.inf)

v_water_along = √(boat_speed² - |c_perp|²)  # along-track water speed
SOG           = v_water_along + c_par
SOG ≤ 0.01 → impassable (boat makes no progress)
```

This logic lives in the module-level `_fixed_speed_sog()` helper and is
shared by `_edge_cost`, `_segment_travel_time`, and `straight_line_time`.

### With polar (heading sweep)

The polar case replaces the single "best heading" geometry with a **sweep
of 180 candidate headings** at 2° resolution.  For each heading `θ`:

1. Compute TWA from `θ` and wind direction.
2. Look up boat speed `V(θ)` from the polar (vectorised bilinear lookup
   over all 180 headings at once).
3. Compute ground velocity: `(gx, gy) = V(θ) * (cos θ, sin θ) + (cu, cv)`.
4. Compute along-track SOG = `gx * d̂x + gy * d̂y`.
5. Compute cross-track drift = `|gx * (-d̂y) + gy * d̂x|`.

Accept a heading if: `SOG > 0.01 m/s` **and** `drift ≤ 0.10 * SOG`.
The drift tolerance (10%) allows realistic crabbing angles while rejecting
headings that push the boat far off course.

The **best SOG** across all accepted headings is used as the edge cost
divisor: `edge_time = dist / max_SOG`.

This is implemented in `_polar_boat_speeds()` (the vectorised polar lookup)
and `_solve_heading_full()` (the full sweep returning SOG + ground velocity
+ best heading angle).

---

## 8. A\* Router

### Grid construction

A regular UTM grid is laid over the bounding box of start and end, with
`padding_m` (default 5 km) on each side.  `resolution_m` (default 300 m)
controls cell size.  All grid cells are pre-screened for water/land using
`CurrentField.query_grid` at `t=0`.

Typical Puget Sound crossing: ~40 × 60 cells = 2,400 water cells.

### State space

Standard A\* uses state `(row, col)`.  To implement tacking penalties,
state is extended to `(row, col, dir_idx)` where `dir_idx` is the index
of the incoming movement direction (0–7 for the 8 cardinal/diagonal
neighbors, 8 = start sentinel).  This means each grid cell has 9 states,
so the search arrays are `(ny, nx, 9)`.

**Trade-off:** This triples memory and computation, but enables the router
to distinguish a northward approach from a southward approach to the same
cell, which is essential for tack-penalty accounting.

### Time-dependent edge costs

Each node carries its **arrival time** (seconds elapsed since simulation
start) stored in `arrival_time[r, c, d_in]`.  When evaluating an edge,
the current (and wind) are sampled at the **physical arrival time** at the
source cell — not at `t=0`.  This is what makes the search truly
time-dependent: nodes explored later in the search see later forecasts.

```
arr_t = arrival_time[r, c, d_in]
dt    = _edge_cost(r, c, nr, nc, xs, ys, arr_t)  ← uses current at arr_t
new_arrival = arr_t + dt                          ← physical clock only
new_cost    = cost  + dt + penalty                ← optimisation cost
```

The **tacking penalty** is added to `best_cost` but **not** to
`arrival_time`.  This is critical: the penalty is an optimisation bias to
discourage tacks; it is not real time on the water and must not corrupt the
current-lookup timestamps of subsequent cells.

### Heuristic

```
h(r, c) = dist_to_goal / max_speed
max_speed = max_boat_speed + max_current_speed  (admissible upper bound)
```

This is admissible (never over-estimates), so A\* returns the optimal path.

### Tacking penalty

If the incoming direction `d_in` and outgoing direction `d_out` differ by
more than `tack_threshold_deg` (default 90°), a penalty of `tack_penalty_s`
seconds (default 60 s) is added to `best_cost`.

The 8 grid directions' angles are pre-computed in `_DIR_ANGLES` and their
pairwise differences in `_ANGLE_DIFF` (built once via `_build_angle_diff()`).

### Path reconstruction

The lowest-cost direction at the goal is found via `argmin(best_cost[er, ec, :])`.
The path is traced backwards through `came_from[r, c, d, :]` which stores
`(prev_r, prev_c, prev_dir)` for each state.  The sentinel value `-1` for
`prev_r` marks the start node, terminating the trace.

---

## 9. Path Smoothing and Stub Removal

The raw A\* path is a staircase of 8-connected grid cells, which
over-estimates distance by up to ~8% on diagonal routes and produces
unnecessary intermediate waypoints.  Two post-processing passes clean it up.

### Pass 1: Greedy string-pulling (`_smooth_path`)

Classic line-of-sight shortcut algorithm:

```
i = 0
while i < n-1:
    for j = n-1 down to i+2:
        if line_of_sight(path[i], path[j], water_mask)
           and travel_time(path[i]→path[j]) ≤ 1.005 × sum(raw_times[i..j]):
            accept shortcut; best_j = j; break
    append path[best_j]; i = best_j
```

The **1.005 tolerance** allows shortcuts that are trivially slower than
the original (grid discretisation noise) but rejects beneficial detours
from being collapsed into a slower straight line — e.g., a route that
deliberately goes into a favorable current band should not be straightened
away from it.

`travel_time` is computed by `_segment_travel_time`, which sub-samples
the segment at ~1 resolution-length intervals and accounts for currents
(and polar/wind) along the way.

### Pass 2: Stub removal (`_remove_stubs`)

After smoothing, Y-junction artifacts sometimes remain: a short stub leg
branching off at a large angle.  `_remove_stubs` scans for waypoints where:

- One adjacent leg is > 3× shorter than the other, **and**
- The turn angle at that waypoint exceeds 45°.

Such points are dropped.  The scan repeats (`while changed`) until no more
stubs are found.  The "previous" point used for leg-angle calculation is
always the last **kept** point (not the raw index), so consecutive stubs
are handled correctly.

---

## 10. Ground-Track Simulation

`simulate_track()` now **densifies the solved route polyline** rather than
free-integrating a separate steering simulation.  This is still purely for
**visualisation/export**, but it keeps the JSON track and time-series plots
continuous and consistent with the solver's own segment travel times.

For each waypoint-to-waypoint segment:

1. Compute `seg_time` with `_segment_travel_time(...)`, using the same current /
   wind / polar physics as the router.
2. Choose a sample count based on both elapsed time and segment length.
3. Linearly interpolate position from the segment start to end, and linearly
   interpolate timestamps from `elapsed` to `elapsed + seg_time`.

This produces a dense ground-track polyline that:

- lands exactly on every solved waypoint,
- ends at the same total elapsed time as `route.total_time_s`, and
- avoids discontinuities or "teleports" in exported track products.

---

## 11. CLI Usage

```
python sail_routing.py \
    --start-lat 47.63 --start-lon -122.40 \
    --end-lat   47.75 --end-lon   -122.42 \
    --boat-speed 6 \
    --grid-resolution 300 \
    --depart "2026-03-10 09:00" \
    --save route.png
```

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--boat-speed` | 6 kt | Fallback fixed speed (used without polar) |
| `--grid-resolution` | 300 m | Routing grid cell size |
| `--padding` | 5000 m | Grid margin beyond start/end |
| `--polar` | — | Path to `TWA_deg, TWS_kt, BoatSpeed_kt` CSV |
| `--wind-speed` | — | Constant TWS in knots (requires `--wind-direction`) |
| `--wind-direction` | — | Met convention "from" bearing, CW from north |
| `--tack-penalty` | 60 s | Time penalty per tack; 0 to disable |
| `--depart` | now | Local departure time (`YYYY-MM-DD HH:MM`); auto-selects forecast hours |
| `--tz` | America/Los_Angeles | IANA timezone for `--depart` |
| `--no-cache` | — | Force fresh SSCOFS download |
| `--no-plot` | — | Skip matplotlib; still saves if `--save` given |

When `--depart` is provided, `load_current_field` automatically computes
which SSCOFS forecast hours to load (departure ± 1 hour through departure
+ 8 hours by default) and returns the correct `start_time_s` offset into
the loaded frames.

---

## 12. Testing

`test_sail_routing.py` has 82 pytest tests, all using **synthetic current
fields** (no SSCOFS download required):

| Test class | Coverage |
|------------|----------|
| `TestEdgeCostPhysics` | Unit tests for travel-time physics (favorable, opposing, cross, diagonal, impassable) |
| `TestRoutingNoCurrents` | Straight routes match `dist / speed` |
| `TestRoutingWithUniformCurrent` | Uniform currents speed up / slow down SOG correctly |
| `TestRoutingDetour` | Router detours into favorable current bands when worth it |
| `TestRoutingLandAvoidance` | Route goes around land obstacles |
| `TestRoutingStraightLineComparison` | A\* ≤ straight-line time (within grid noise) |
| `TestCurrentFieldInterpolation` | IDW accuracy, land detection, time interpolation |
| `TestPathSmoothing` | Diagonal distance near Euclidean, land preserved, stubs removed |
| `TestTimeDependentRouting` | Disappearing bands, reversing currents, late-appearing bands |
| `TestPolarTable` | Bilinear interpolation, clamping, exact lookups |
| `TestComputeTwa` | TWA geometry for head-to-wind, beam, downwind |
| `TestBoatModelPolar` | Polar vs fixed-speed modes |
| `TestSolveHeading` | Heading sweep SOG, current bonus, zero-wind fallback |
| `TestRoutingWithPolar` | Full polar routing integration (beam reach, upwind, downwind) |
| `TestWindFieldSpatial/Temporal` | Grid and temporal wind interpolation |
| `TestTackingPenalty` | Penalty logic, angle precomputation |
| `TestRouteQuality` | No Y-junctions, no backward progress, no stubs |
| `TestSimulatedTrack` | Exported track continuity, timing, no teleports |

Run with:

```bash
conda activate anaconda
cd OceanCurrents/Python_SSCOFS
python -m pytest test_sail_routing.py -v
```

`viz_test_scenarios.py` generates diagnostic plots of the key synthetic
scenarios (no SSCOFS download required):

```bash
python viz_test_scenarios.py   # saves test_viz_*.png in current directory
```

---

## 13. Known Limitations and Future Work

### Routing grid is fixed

The grid resolution is uniform.  A coarser grid is faster but may miss
narrow channels (e.g., Admiralty Inlet).  Adaptive grid refinement near
coastlines would improve accuracy without blowing up computation time.

### No layline or gybe strategy

The heading sweep finds the best instantaneous VMG toward the next grid
cell.  For upwind sailing, this naturally produces tacking when the
wind-axis route is forced to zig-zag across the grid.  However, the
algorithm doesn't explicitly compute laylines or optimal tack angles —
those emerge implicitly from the polar lookup and tack penalty.

### Single-layer currents

SSCOFS provides multiple sigma layers (`siglay`).  Only the surface layer
(`siglay=0`) is used.  Tidal mixing can make sub-surface currents
significantly different, especially in shallow water.

### Polar is port/starboard symmetric

`compute_twa` returns a value in [0°, 180°], treating port and starboard
tacks as equivalent.  Real polars are slightly asymmetric (and downwind
gybing has a gybe penalty just like tacking).  The current model applies
the same tacking penalty to all direction changes ≥ 90°.

### Wind field is not forecast

The wind field is user-supplied (constant, or a manually constructed grid/
temporal field).  There is no automatic ingestion of NWS or GFS wind
forecast data.  Integration with a gridded wind product (e.g., HRRR or
NAM) would make the routing significantly more realistic.

### Exported track follows the solved polyline

The dense exported track is sampled from the already-solved waypoint polyline,
not produced by an independent helm/autopilot simulation.  That makes the
JSON export and UTM time-series plots stable and continuous, but it also means
the exported track is best interpreted as a time-parameterised version of the
chosen route rather than a separate dynamic boat-motion model.
