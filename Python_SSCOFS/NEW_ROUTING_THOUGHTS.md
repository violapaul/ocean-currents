Yes. With SSCOFS, I would use the current model to define the **physics mesh** for routing, but I would not search on a uniform raster. SSCOFS is already an **unstructured triangular FVCOM mesh**, with resolution around **100 m near shoreline**, **500 m in deeper Puget Sound / Georgia Basin**, **100–200 m in the Columbia**, and **hourly 3-D fields** in NetCDF. That is a strong substrate for a near-shore, high-resolution router. ([NOAA Tides and Currents][1])

The clean design is to separate three things:

1. **Physics mesh:** native SSCOFS triangles/elements for interpolating current.
2. **Geometry constraints:** ENC shoreline, drying areas, depth/UKC buffers, exclusion zones.
3. **Routing graph:** a derived graph built on top of the physics mesh.

I would not use the raw model mesh as the final navigation lattice. Ocean-model meshes are built for hydrodynamics, not for giving a good set of navigation actions. NOAA’s own SSCOFS assessment notes a case where the mesh could not resolve a complicated coastline well enough at Garibaldi, which is exactly the kind of failure mode that matters for narrow near-shore routing. ([NOAA Tides and Currents][2])

What I would do instead is this.

**Use SSCOFS as the field representation, not as the only move set.**

Start from the native wet triangles. For routing nodes, use either triangle centroids or a filtered subset of wet mesh nodes. Then build a separate neighbor graph whose spacing is controlled by a target path scale `ΔP(x)`, not by raw triangle adjacency. VISIR-2 is a good reference here: it varies graph resolution/connectivity while keeping a fixed **path-resolution** idea, and it uses a **KD-tree** to screen coast intersections efficiently. ([GMD][3])

A practical rule is:

[
\Delta P(x)=\mathrm{clamp}!\left(
\min!\left[
\alpha h_{\text{mesh}}(x),
\beta d_{\text{shore}}(x),
\gamma w_{\text{channel}}(x),
\frac{\delta}{\varepsilon+|\nabla \mathbf{c}(x)|},
\frac{\eta}{\varepsilon+|\partial_t \mathbf{c}(x)|}
\right],
\Delta P_{\min}, \Delta P_{\max}
\right)
]

where:

* `h_mesh(x)` is local SSCOFS triangle size,
* `d_shore(x)` is distance to shoreline,
* `w_channel(x)` is local channel width,
* `c(x,t)` is current,
* `∇c` and `∂t c` capture current shear and time variation.

In plain terms: **refine where the coast is tight, where the channel is narrow, and where the current field changes fast**.

For connectivity, do **not** just follow triangle edges. That bakes mesh orientation into the route and tends to create stair-stepping and artificial extra tacks. Instead, for each routing node, divide heading space into bins, such as 16 or 24 sectors, and connect to the nearest reachable node in each sector at distance about `ΔP(x)`. Validate each candidate edge against shoreline/depth constraints with a KD-tree or R-tree over chart geometry. That gives you a much better action lattice for sailing.

For near shore, use a **hierarchical solve**. A recent 2025 coastal-routing paper for small ships uses exactly that pattern: a **quadtree** chart, a **high-level layer** for the coarse corridor, then a **low-level layer** for detailed routing near entrances and obstacles. That is the right shape for your problem too. ([ScienceDirect][4])

For the sailing part, let **current define node density**, but let **wind/polar define edge cost and feasibility**. Recent inshore sailing work does the same at the algorithm level: it builds an **uneven graph** from a local wind field and solves minimum-time routing with **dynamic programming**, **multiple heading choices**, and **tacking penalties**. ([ScienceDirect][5])

So the search state I would actually use is:

[
s = (\text{routing_node}, \text{time_index}, \text{tack})
]

and maybe also `sail_mode` or `reef_state` if you care about that.

For an edge from node `i` to node `j`:

1. Walk the segment in short substeps.
2. Interpolate current from SSCOFS at each substep and time.
3. Use a draft-weighted current, not blindly the surface layer, if your keel is deep enough for shear to matter. SSCOFS gives you 3-D hourly fields, so you can do this cleanly. ([NOAA Tides and Currents][1])
4. Solve for the boat heading that makes the **ground-track** align with the edge, since in current the heading is not the same as track-over-ground.
5. Evaluate polar/VPP speed and leeway.
6. Integrate travel time and add tack/gybe penalties, shallow-water penalties, and clearance penalties.

That gives you a time-dependent edge cost:

[
J_e = \Delta t_e

* \lambda_{\text{tack}}\mathbf{1}[q' \neq q]
* \lambda_{\text{clear}}\phi(\text{clearance})
* \lambda_{\text{shallow}}\phi(\text{UKC})
  ]

Then run **time-dependent A*** on that graph. The admissible heuristic can be as simple as straight-line distance divided by an upper bound on achievable SOG.

The shortest useful implementation is:

* native SSCOFS mesh for interpolation,
* ENC shoreline/depth as hard constraints,
* derived centroid graph with heading-binned neighbors,
* time-dependent A* over `(node, time, tack)`,
* edge propagation with current substepping and polar evaluation,
* coarse-to-fine refinement inside a corridor.

Two things I would avoid:

* **Do not regrid SSCOFS to a regular lat/lon raster** unless it is only for visualization. You will lose the native shoreline-aware resolution that SSCOFS already gives you. ([NOAA Tides and Currents][1])
* **Do not use raw FVCOM adjacency as the only graph topology.** It is too tied to the PDE mesh and usually too poor as a sailing action lattice.

So the answer to your idea is: **yes, use SSCOFS to define the search grid, but use it to define an adaptive routing graph, not a uniform raster and not a raw mesh-edge graph**.

The next design decision is whether your wind field is at roughly the same coastal resolution as SSCOFS, or much coarser.

[1]: https://tidesandcurrents.noaa.gov/ofs/dev/sscofs/sscofs_info.html "Salish Sea and Columbia River Operational Forecast System (SSCOFS) (Developmental)"
[2]: https://tidesandcurrents.noaa.gov/ofs/publications/noaa_70292_DS1.pdf "Implementation of the Salish Sea and Columbia River Operational Forecast System and the Semi-Operational Nowcast/Forecast Skill Assessment"
[3]: https://gmd.copernicus.org/articles/17/4355/2024/gmd-17-4355-2024.html "GMD - VISIR-2: ship weather routing in Python"
[4]: https://www.sciencedirect.com/science/article/pii/S2092678225000056 "A route planning method for small ships in coastal areas based on quadtree - ScienceDirect"
[5]: https://www.sciencedirect.com/science/article/abs/pii/S0029801825005621 "Inshore sailing route optimization integrating terrain-influenced wind field - ScienceDirect"
