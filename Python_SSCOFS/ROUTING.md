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
8. [A* Router (Grid-Based, Legacy)](#8-a-router-grid-based-legacy)
9. [MeshRouter (Delaunay-Based, Legacy)](#9-meshrouter-delaunay-based-legacy)
10. [SectorRouter (Default)](#10-sectorrouter-default)
11. [SectorRouter Path Smoothing](#11-sectorrouter-path-smoothing)
12. [Legacy Router Smoothing and Stub Removal](#12-legacy-router-smoothing-and-stub-removal)
13. [Ground-Track Simulation](#13-ground-track-simulation)
14. [CLI Usage](#14-cli-usage)
15. [Testing](#15-testing)
16. [Known Limitations and Future Work](#16-known-limitations-and-future-work)

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
| `SectorRouter` | **Default.** 32-sector heading-binned connectivity on SSCOFS nodes |
| `Router` | Legacy 8-connected UTM grid router |
| `MeshRouter` | Legacy Delaunay mesh edge router |
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

**Land detection:** Uses Delaunay triangulation of SSCOFS element centers.
Triangles with edges longer than 3.5× the local mesh density are classified
as "land-spanning" (i.e., they bridge a gap where land exists). A query point
is water if it falls inside a valid (non-land-spanning) Delaunay triangle;
otherwise it is treated as land and `(NaN, NaN)` is returned.

This approach automatically adapts to the SSCOFS mesh density gradient — tight
thresholds (~300 m) near the coast where elements are 50-100 m apart, loose
thresholds (~1500 m) in open ocean where elements are 300-500 m apart. Falls
back to distance-threshold for very small point sets (synthetic tests).

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

Two CSV formats are supported:

**Simple format** — columns `TWA_deg`, `TWS_kt`, `BoatSpeed_kt`.  Must form
a complete rectangular grid.  Missing grid points raise an error.

**Sail-config format** — columns include `sail`, `TWS_kt`, `TWA_deg`,
`BTV_kt`.  Only the "Best Performance" rows are used (the envelope of all
sail configurations).  Beat/run-target TWAs vary by TWS, so the grid is
built from the union of all TWA values; gaps are filled by linear
interpolation within each TWS column, and angles below the minimum
data TWA for each column are forced to zero (no-go zone).  A 0° row at
zero speed is prepended automatically.

The optional `minimum_twa` parameter forces boat speed to zero for any TWA
below the given angle, regardless of what the polar table says.  The optional
`maximum_twa` parameter does the same for too-deep downwind angles.  The
production J/105 race configs use `38 <= abs(TWA) <= 165`, which rejects both
dead-upwind and dead-downwind/deep-running shortcuts while treating port and
starboard symmetrically.  For sail-config polars, `minimum_twa` is auto-set to
the global minimum beat-target angle if not specified.

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

Four modes, selected by the constructor used:

| Mode | Constructor | Description |
|------|-------------|-------------|
| Constant | `WindField.from_met(speed_kt, from_deg)` | Same wind everywhere at all times |
| Spatial grid | `WindField.from_grid(xs, ys, wu_grid, wv_grid)` | Spatially varying, constant in time; `RegularGridInterpolator` |
| Temporal grid | `WindField.from_frames(xs, ys, wu_frames, wv_frames, times_s)` | Per-frame on a regular grid; `RegularGridInterpolator` per frame, linearly interpolated in time |
| Temporal nodes | `WindField.from_node_frames(node_x, node_y, wu_frames, wv_frames, times_s)` | Per-frame on irregular spatial nodes; nearest-node (KD-tree) spatial lookup, linearly interpolated in time |

The **temporal nodes** mode is used by the ECMWF wind pipeline: Open-Meteo
returns forecasts at snapped ECMWF native grid nodes, which are irregularly
spaced in UTM.  A `cKDTree` on `(node_x, node_y)` provides O(log n) nearest
lookup for each spatial query.

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
The drift tolerance (10%, ~5.7° max crab angle) rejects pseudo-physical
shortcuts where the boat sails fast on a reach while the ground track points
inside the no-go zone.  See [POLAR_SWEEP_OPTIMIZATIONS.md](POLAR_SWEEP_OPTIMIZATIONS.md)
for the history of this threshold.

The **best SOG** across all accepted headings is used as the edge cost
divisor: `edge_time = dist / max_SOG`.

This is implemented in `_polar_boat_speeds()` (the vectorised polar lookup)
and `_solve_heading_full()` (the full sweep returning SOG + ground velocity
+ best heading angle).

---

## 8. A\* Router (Grid-Based, Legacy)

> **Note:** This 8-connected grid router is retained for compatibility but is
> not recommended. Use `SectorRouter` (the default) for better path quality.

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

The maneuver penalty is added to both `best_cost` and the stored arrival time.
It represents practical maneuver loss and keeps subsequent current/wind lookups
aligned with the elapsed time implied by the chosen route.

### Heuristic

```
h(r, c) = dist_to_goal / max_speed
max_speed = max_boat_speed + max_current_speed  (admissible upper bound)
```

This is admissible (never over-estimates), so A\* returns the optimal path.

### Tacking penalty

If the incoming direction `d_in` and outgoing direction `d_out` differ by
more than `tack_threshold_deg` (default 50°), a penalty of `tack_penalty_s`
seconds (default 90 s) is added to `best_cost`.

The 8 grid directions' angles are pre-computed in `_DIR_ANGLES` and their
pairwise differences in `_ANGLE_DIFF` (built once via `_build_angle_diff()`).

### Path reconstruction

The lowest-cost direction at the goal is found via `argmin(best_cost[er, ec, :])`.
The path is traced backwards through `came_from[r, c, d, :]` which stores
`(prev_r, prev_c, prev_dir)` for each state.  The sentinel value `-1` for
`prev_r` marks the start node, terminating the trace.

---

## 9. MeshRouter (Delaunay-Based, Legacy)

> **Note:** MeshRouter is retained for comparison but is not recommended.
> Use `SectorRouter` (the default) for better heading freedom and path quality.

`MeshRouter` runs A\* directly on the SSCOFS Delaunay mesh edges.  While this
provides exact node velocities and adaptive resolution, the mesh topology was
designed for hydrodynamics (FVCOM), not navigation.  The arbitrary edge angles
create path wobble that requires aggressive smoothing.

### Mesh graph construction

The search graph is built by `build_mesh_adjacency(delaunay, valid_mask)`:

```
for each valid triangle:
    add edges (v0,v1), (v1,v2), (v2,v0) to adjacency
```

An edge exists between nodes `a` and `b` if they share a valid water triangle.
Edge distances are precomputed in UTM metres.

### State space

State is `(node_id, bearing_bucket)` where `bearing_bucket` quantizes the
incoming direction into 16 bins (22.5° each).  This enables tacking penalties
without a fixed grid direction set.

- 16 bearing buckets + 1 start sentinel = 17 states per node
- Sparse storage: Python dicts for `best_cost`, `came_from`, `arrival_time`

### Edge cost

Uses the same physics as the grid router (`_fixed_speed_sog`, `_solve_heading`)
but with direct node velocity lookup instead of IDW:

```
cu = 0.5 * (u[node_a] + u[node_b])  # averaged, time-interpolated
cv = 0.5 * (v[node_a] + v[node_b])
```

### Tacking penalty

Angle difference between bearing buckets:

```
diff = |bucket_in - bucket_out|
if diff > N_BUCKETS/2: diff = N_BUCKETS - diff
angle_diff = diff * 22.5°
```

Penalty applies when `angle_diff > tack_threshold_deg`.

### Path smoothing

The same greedy string-pulling approach as the grid router, but:

- **Line-of-sight** is checked by sampling points along the shortcut and
  calling `CurrentField.query()` — any NaN (land) rejects the shortcut.
- **Stub removal** uses the same geometric heuristic (leg ratio + turn angle).

### Status in current entrypoints

`MeshRouter` remains in the codebase for reference and diagnostics, but the
current `sail_routing.py` CLI and `run_route.py` YAML entrypoint are configured
to use `SectorRouter`.

---

## 10. SectorRouter (Default)

`SectorRouter` is the recommended router.  It uses SSCOFS element centers as
routing nodes (inheriting their adaptive density: ~100 m near shore, ~500 m
offshore) but replaces Delaunay edge adjacency with **32-sector heading-binned
connectivity**.

### Why SectorRouter?

The grid router (8 directions) forces zigzag paths that must be heavily
smoothed.  The mesh router (Delaunay edges, 3–6 directions) has arbitrary angles
tied to the PDE mesh, causing path wobble.

SectorRouter gives the A\* **32 angular choices** (11.25° sectors) at every node:

```
For each node, connect to the nearest reachable neighbor in each sector:
    0 = North, 1 = N by E, 2 = NNE, ..., 31
```

This decouples the navigation action lattice from both grid geometry and mesh
topology, producing paths that already represent reasonable sailing headings.

### Key properties

| Aspect | SectorRouter | MeshRouter | Grid Router |
|--------|--------------|------------|-------------|
| Directions per node | 32 | 3–6 | 8 |
| Edge cost eval | `cf.query` at midpoint | Averaged node velocities | `cf.query` at midpoint |
| Graph build | Adaptive corridor + cached sector graph | Eager (all edges) | Eager (full grid) |
| Smoothing | Two-pass (tack-aware DP, ~74% reduction) | Heavy greedy string-pull (217→17 typical) | Heavy greedy string-pull (120→29 typical) |

### Lazy neighbor discovery

Neighbors are computed on-demand during A\*:

```python
def _find_sector_neighbors(self, node_id):
    1. Query KD-tree for k_candidates nearest nodes (default 50)
    2. Filter by distance: min_edge_m < dist < max_edge_m
    3. Assign each candidate to its heading sector (0–31)
    4. Try nearest candidate in each sector
    5. If LOS fails, retry next-nearest candidate in that sector
    6. Cache and return list of (neighbor_id, sector, distance)
```

Only explored nodes pay the neighbor-discovery cost.  Typical runs cache
~10–20k nodes out of 400k+ SSCOFS elements.

### Edge cost

Same physics as other routers (`_fixed_speed_sog`, `_solve_heading`) but queries
`CurrentField.query(midpoint)` rather than averaging node velocities:

```python
mx, my = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
cu, cv = cf.query(mx, my, elapsed_s=arrival_time)
# then _solve_heading or _fixed_speed_sog
```

### Maneuver penalties

SectorRouter still uses the COG-sector bucket system as a fallback:

```
angle_diff = |sector_in - sector_out| * 11.25°
if angle_diff > tack_threshold_deg: add tack_penalty_s
```

When polar + wind routing is active, the router also tracks the best
boat-heading selected for each edge.  A wind-relative port/starboard side flip
near upwind applies `tack_penalty_s`; a side flip with both TWAs at or above
`gybe_threshold_deg` applies `gybe_penalty_s`.  This catches downwind gybe
shortcuts even when the COG change is split across multiple small sector steps.

### Path quality

Because the raw A\* path already has 32 heading options per step, the output
is cleaner than grid or mesh routing.  A two-pass smoothing step (§11) further
reduces the waypoint count by ~70–80% on typical Puget Sound routes while
correctly preserving tacks and handling land avoidance.
Example: a 4 nm upwind beat produces ~57 raw nodes → 15 smoothed waypoints.

### Performance

SectorRouter performance is dominated by A\* state expansion. The current
implementation reduces this with:

- adaptive corridor retries (`corridor_pad_factors`, tight-to-wide),
- in-memory corridor graph caching (`corridor_cache_max`),
- pre-baked dense polar table (`use_dense_polar`) replacing binary search,
- optional forward-hemisphere dot filter (`use_dot_filter`, experimental),
- coarse polar heading sweep (`polar_sweep_coarse_step`), and
- Numba kernels for expansion and polar/wind cost evaluation.

See [POLAR_SWEEP_OPTIMIZATIONS.md](POLAR_SWEEP_OPTIMIZATIONS.md) for benchmarks.

### CLI usage

```bash
# SectorRouter is the active router (no selection flag needed)
python sail_routing.py \
    --start-lat 47.63 --start-lon -122.40 \
    --end-lat 47.75 --end-lon -122.42
```

### YAML usage (`run_route.py`)

```yaml
routing:
  router_type: "sector"
  tack_penalty_s: 90
  tack_threshold_deg: 50
  gybe_penalty_s: 90
  gybe_threshold_deg: 120
  use_dense_polar: true
  use_dot_filter: false
  polar_sweep_coarse_step: 3
  corridor_pad_factors: [0.85, 1.0, 1.35]
  corridor_cache_max: 4
```

---

## 11. SectorRouter Path Smoothing

The raw SectorRouter A\* path contains every SSCOFS node visited during the
search.  While already more angular than grid/mesh routes, it still has many
redundant intermediate waypoints — especially in tacking sequences where the
boat alternates headings every 100–200 m.  `_smooth_path` reduces these with
a two-pass algorithm that distinguishes **genuine tacks** from **gradual
curves**.

### Why the distinction matters

A sailing route has two very different kinds of heading changes:

- **Tacks** — the boat physically changes tack (port ↔ starboard), producing a
  sharp 45–90° heading reversal that is a true navigational event.
- **Gradual curves** — the heading drifts slowly as the boat follows a current
  band, avoids a shore, or sails a slightly curved VMG course.  These can be
  approximated by fewer waypoints without losing navigational meaning.

### Pass 1a — Tack detection

A waypoint is a **tack** if the turn angle (incoming vs outgoing heading) is
≥ `SMOOTH_TACK_THRESHOLD_DEG` (default 45°).  Tack points are always kept.

### Pass 1b — Tack merging

In a short upwind beat the A\* may produce tack legs only 75–200 m long,
leaving every other node as a "tack" — which would produce no simplifiable
inter-tack segments.  Nearby tacks are merged: any raw tack within
`SMOOTH_MIN_TACK_SPACING_M` (default 400 m) of the previous kept tack is
either skipped or replaces it if it has a larger turn angle.

```
tack_boundaries = [start]
for each raw tack candidate t:
    if dist(last_boundary, t) >= 400 m:
        tack_boundaries.append(t)
    elif t.turn_angle > last_boundary.turn_angle:
        tack_boundaries[-1] = t     # sharper tack wins
tack_boundaries.append(end)
```

### Pass 2 — Douglas-Peucker within each inter-tack segment

Between consecutive tack boundaries the path follows a gradual curve.
Douglas-Peucker with `SMOOTH_DP_EPSILON_M` (default 120 m) collapses
intermediate nodes whose perpendicular distance from the straight chord is
below the threshold:

```
dp_segment(start, end):
    find node k in (start..end) with max perpendicular distance d to chord
    if d > 120 m:
        return dp_segment(start, k) + dp_segment(k, end)
    else:
        return [start]          # chord is good enough
```

All `dp_segment` results are merged and `end` is appended once.

### Per-leg land fallback

A simplified chord (tack boundary → tack boundary) may cross land even though
all individual raw edges were over water.  Rather than reverting the **entire**
smoothed route, `_compute_leg_times` detects non-finite travel times per leg
and re-inserts the original raw sub-path for just that leg.

### Tuning knobs

| Constant | Default | Effect |
|----------|---------|--------|
| `SMOOTH_TACK_THRESHOLD_DEG` | 45° | Lower → more waypoints treated as tacks |
| `SMOOTH_MIN_TACK_SPACING_M` | 400 m | Higher → fewer tack waypoints in dense beats |
| `SMOOTH_DP_EPSILON_M` | 120 m | Higher → more aggressive curve simplification |

### Typical results

| Scenario | Raw nodes | Smoothed | Reduction |
|----------|-----------|----------|-----------|
| 4 nm upwind beat (polar + ECMWF wind) | 57–82 | 15–22 | ~73–74% |
| Downwind / reaching run | lower | lower | ~70%+ |

---

## 12. Legacy Router Smoothing and Stub Removal

The raw A\* path from the grid or mesh router is a staircase of connected
cells that over-estimates distance by up to ~8% on diagonal routes and
produces unnecessary intermediate waypoints.  Two post-processing passes
clean it up.

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
the original (grid discretisation noise) but rejects beneficial detours —
a route that deliberately enters a favorable current band should not be
straightened away from it.

`travel_time` is computed by `_segment_travel_time`, which sub-samples the
segment at ~1 resolution-length intervals and accounts for currents (and
polar/wind) along the way.

### Pass 2: Stub removal (`_remove_stubs`)

After smoothing, Y-junction artifacts sometimes remain: a short stub leg
branching off at a large angle.  `_remove_stubs` scans for waypoints where:

- One adjacent leg is > 3× shorter than the other, **and**
- The turn angle at that waypoint exceeds 45°.

Such points are dropped.  The scan repeats until no more stubs are found.
The "previous" point used for leg-angle calculation is always the last **kept**
point (not the raw index), so consecutive stubs are handled correctly.

---

## 13. Ground-Track Simulation

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

## 14. CLI Usage

```
python sail_routing.py \
    --start-lat 47.63 --start-lon -122.40 \
    --end-lat   47.75 --end-lon   -122.42 \
    --boat-speed 6 \
    --depart "2026-03-10 09:00" \
    --save route.png
```

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--boat-speed` | 6 kt | Fallback fixed speed (used without polar) |
| `--polar` | — | Path to polar CSV (simple or sail-config format) |
| `--minimum-twa` | 0 | No-go zone: zero boat speed below this TWA (degrees) |
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

## 15. Testing

`test_sail_routing.py` has 86 pytest tests across 19 test classes, all
using **synthetic current fields** (no SSCOFS download required):

| Test class | Coverage |
|------------|----------|
| `TestEdgeCostPhysics` | Unit tests for travel-time physics (favorable, opposing, cross, diagonal, impassable) |
| `TestRoutingNoCurrents` | Straight routes match `dist / speed` |
| `TestRoutingWithUniformCurrent` | Uniform currents speed up / slow down SOG correctly |
| `TestRoutingDetour` | Router detours into favorable current bands when worth it |
| `TestRoutingLandAvoidance` | Route goes around land obstacles |
| `TestRoutingStraightLineComparison` | A\* ≤ straight-line time (within grid noise) |
| `TestCurrentFieldInterpolation` | IDW accuracy, land detection, time interpolation |
| `TestPathSmoothing` | Diagonal distance near Euclidean, land preserved, stubs removed, land-crossing shortcuts rejected |
| `TestTimeDependentRouting` | Disappearing bands, reversing currents, late-appearing bands |
| `TestPolarTable` | Bilinear interpolation, clamping, exact lookups, missing-grid-point rejection |
| `TestComputeTwa` | TWA geometry for head-to-wind, beam, downwind |
| `TestBoatModelPolar` | Polar vs fixed-speed modes |
| `TestSolveHeading` | Heading sweep SOG, current bonus, zero-wind fallback |
| `TestRoutingWithPolar` | Full polar routing integration (beam reach, upwind, downwind) |
| `TestWindFieldSpatial` | Spatial grid wind interpolation |
| `TestWindFieldTemporal` | Temporal constant, temporal grid, and temporal node wind interpolation |
| `TestTackingPenalty` | Penalty logic, angle precomputation |
| `TestRouteQuality` | No Y-junctions, no backward progress, no stubs |
| `TestSimulatedTrack` | Exported track continuity, timing, no teleports |
| `TestSectorRouterBasic` | Straight north/diagonal routes on synthetic field |
| `TestSectorRouterWithCurrent` | Favorable/opposing current affects SOG correctly |
| `TestSectorRouterLandAvoidance` | Routes around land obstacles |
| `TestSectorRouterTackingPenalty` | Tack penalty raises cost vs no-penalty baseline |
| `TestSectorRouterHeadingDiversity` | All 16 sectors get at least one neighbor |
| `TestSectorRouterRouteQuality` | Waypoints progress toward goal; simulated track timing |

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

## 16. Known Limitations and Future Work

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
tacks/gybes as equivalent for boat speed.  The router still tracks the signed
wind-relative heading selected for each edge so maneuver diagnostics and
penalties can distinguish tacks from gybes.  Real polars are slightly
asymmetric; that asymmetry is not modeled yet.

### Wind forecast integration

`run_route.py` now supports automatic ECMWF wind ingestion via
`source: "open_meteo_ecmwf"` in the YAML config, using `ecmwf_wind.py` to
fetch 9 km IFS HRES hourly forecasts from Open-Meteo.  The wind field is
built as `temporal_nodes` (nearest-node spatial, linearly interpolated in
time).  For the CLI (`sail_routing.py`), wind is still user-supplied
(constant speed/direction).  Higher-resolution wind products (HRRR, NAM)
are not yet integrated.

### Exported track follows the solved polyline

The dense exported track is sampled from the already-solved waypoint polyline,
not produced by an independent helm/autopilot simulation.  That makes the
JSON export and UTM time-series plots stable and continuous, but it also means
the exported track is best interpreted as a time-parameterised version of the
chosen route rather than a separate dynamic boat-motion model.
