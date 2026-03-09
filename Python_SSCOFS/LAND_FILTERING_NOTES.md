# Land Filtering Architecture

## Pipeline

```
SSCOFS element centers  →  Delaunay triangulation
                                    ↓
                           DEM edge refinement (cut triangles that cross land)
                                    ↓
                           Velocity coherence refinement
                                    ↓
                           Cleaned water domain boundary
                                    ↓
                    ┌───────────────┴───────────────┐
                    ↓                               ↓
           water_boundary.geojson            Valid triangle mask
           (shipped to browser)              (used in routing)
                    ↓
            ┌───────┴───────┐
            ↓               ↓
     Coastline drawing   Bitmap rasterization
     (Canvas stroke)     (Canvas fill → Uint8Array)
                                ↓
                         isWater(lon, lat)
                         O(1) per grid point
```

## Design Principles

1. **SSCOFS is the source of truth.** Element centers only exist at water locations. This defines the water domain.

2. **DEM refines the graph, not the elements.** `refine_with_nav_mask()` samples along Delaunay triangle edges and rejects triangles that bridge across land (e.g. across peninsulas). It does not mark individual SSCOFS elements as land — the elements are always valid water points.

3. **One boundary serves two purposes.** The `water_boundary.geojson` derived from the cleaned Delaunay is used for both coastline drawing and arrow filtering. This guarantees alignment — arrows never appear outside the drawn coastline.

4. **Filtering happens at grid-point level, not element level.** IDW interpolation from valid coastal elements can extend beyond the water domain. The fix is filtering where arrows are *placed* (grid points), not which elements contribute velocity. The browser bitmap handles this.

## Files

| File | Role |
|------|------|
| `generate_current_data.py` | Runs pipeline, exports geometry + velocity + boundary |
| `water_boundary.py` | Delaunay triangulation, DEM refinement, boundary extraction |
| `build_nav_mask.py` | Creates `nav_mask.npz` from NOAA bathymetric DEM |
| `sail_routing.py` | A* routing (uses DEM for land avoidance with safety margin) |
| `map-viewer-mobile.html` | Loads boundary, rasterizes to bitmap, draws coastline + filtered arrows |
| `data/nav_mask.npz` | DEM-derived water grid (~10m resolution) |

## Tuning: `max_edge_m`

`build_water_mask()` has three geometric filters for classifying Delaunay triangles:

1. **Local scale** (3.5x) — adaptive to mesh density, handles coast/open-water transitions
2. **Aspect ratio** (>6) — catches elongated slivers bridging narrow land gaps
3. **Absolute edge cap** (`max_edge_m`) — catches convex-hull boundary artifacts

The absolute cap must be larger than the coarsest valid mesh spacing. The SSCOFS mesh
is ~50–200m near coast but ~400–1000m in the open Strait of Juan de Fuca (P95 ≈ 1039m).
A cap of 800m incorrectly rejected ~9,000 valid open-water triangles in the strait,
creating holes in the boundary polygon that the browser bitmap treated as land.
Raised to 2000m — convex-hull artifacts (up to 63km) are still caught while all valid
open-water mesh passes through.

## Known Issue: Jagged Coastline

The Delaunay boundary follows triangle edges, producing angular coastline segments
rather than smooth curves. This is cosmetic — the filtering is correct. Potential
improvements:

- Chaikin or Catmull-Rom smoothing on the extracted boundary rings
- Increase `minRingVerts` threshold to suppress tiny isolated triangles
- Higher-density SSCOFS nodes near coast naturally improve resolution
