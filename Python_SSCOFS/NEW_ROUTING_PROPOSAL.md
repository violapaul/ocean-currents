# Near-Shore Sailing Router Design and Implementation Specification

*Recommended architecture for high-resolution current routing with coarse wind forcing using SSCOFS, chart constraints, and uncertainty-aware tack management*

Version 1.0 | Prepared March 10, 2026

> Core decision. Build the navigation graph from chart geometry and SSCOFS-derived local scale, not from a uniform raster and not from raw FVCOM mesh adjacency. Evaluate edge costs with native-current interpolation, a separate locally corrected wind field, and explicit tack hysteresis.

*Figure 1. Recommended system architecture. The graph is driven by shoreline geometry and SSCOFS current structure; wind is a separate field that affects edge cost, not graph density.*

## 1. Purpose and scope

This document specifies a single recommended design for a sailing-router intended for near-shore, narrow-channel, and high-resolution operation. It assumes the current field is available from SSCOFS at native model resolution, while the wind forecast is available from ECMWF at approximately 9 km resolution.

The design is aimed at a practical implementation that can be built incrementally, validated with race or passage logs, and operated under forecast uncertainty. The objective is not merely to find a shortest path in a flow field; it is to produce routes that are credible to a human sailor, avoid overfitting to noisy wind forecasts, and strongly discourage unnecessary tacks on long races.

The recommended end state is a time-dependent graph search over state (node, time, tack, age) with: (1) an adaptive graph derived from chart geometry and SSCOFS local scale, (2) edge costs computed by substep propagation through native currents plus locally corrected wind, (3) tack hysteresis and forecast-uncertainty penalties, and (4) receding-horizon replanning.

## 2. Executive summary

- Do use SSCOFS to define routing resolution. SSCOFS provides native unstructured-mesh currents with hourly 3-D fields; its mesh is already refined into the near-shore domain and gives the right signal for where the graph should be denser.

- Do not route on a uniform raster. A regular 100 m raster is wasteful offshore and still awkward in narrow channels unless combined with heavy chart processing.

- Do not use raw FVCOM mesh adjacency as the move set. The hydrodynamic mesh is good for interpolation, not for navigation actions.

- Let chart geometry control legality. Shoreline, depth, drying areas, and user exclusions must define the navigable domain. The ocean-model wet mask alone is not enough in complex inlets and marinas.

- Treat ECMWF wind as a separate, coarse field. It should not define graph resolution. Apply local sector-based correction, WRF/CFD downscaling, or another auditable near-shore wind model before it drives route choice.

- Represent tack explicitly in the state. Tack cost is not a constant offset; it must include physical loss, persistence, and uncertainty.

- Use a robust objective. Optimize not only mean ETA, but also tack count, minimum leg persistence, and ETA risk under wind uncertainty.

- Replan tactically. The long-horizon planner should choose corridor and tack gates, while short-horizon replanning chooses the exact moment of a tack.

> Recommended algorithmic core. Adaptive graph construction + time-dependent A* (or Dijkstra if the heuristic is weak) + corridor refinement + ensemble-aware edge cost aggregation.

## 3. Design requirements

| Requirement | Design response |
| --- | --- |
| Near-shore passages | Graph density tied to shoreline distance, channel width, and current shear; legality enforced against charts. |
| Narrow areas | Explicit centerline / medial-axis preservation and local refinement inside channels and harbor mouths. |
| High-resolution current data | Native SSCOFS interpolation on the unstructured mesh; no mandatory rasterization. |
| Coarse wind input | Separate local-wind correction layer; wind affects edge cost, not node spacing. |
| Avoid too many tacks | State includes tack and age; cost includes physical maneuver loss, hysteresis, and uncertainty. |
| Forecast uncertainty | Ensemble or perturbed-scenario evaluation; objective includes ETA risk and tack aversion growth with lead time. |
| Operational speed | Hierarchical corridor search, cached interpolation indices, and vectorized edge propagation. |

## 4. Recommended architecture

The system should be organized into six modules with narrow interfaces. Keep geometry, environmental data, vessel performance, and search logic separate. That separation makes validation easier and allows later replacement of the wind model without rebuilding the router.

| Module | Responsibility | Key outputs |
| --- | --- | --- |
| Domain builder | Load ENC shoreline, depth constraints, exclusions, and construct the legal navigation polygon. | Wet navigable domain; shoreline distance; channel metrics |
| Field loader | Read SSCOFS currents and ECMWF wind; resample in time; cache interpolation structures. | Current sampler; wind sampler; uncertainty metadata |
| Wind corrector | Convert coarse synoptic wind into local near-shore wind using sector maps or higher-resolution downscaling. | Local wind field U_loc(x,t) |
| Graph builder | Create adaptive nodes and heading-binned neighbor edges; reject edges crossing land or violating depth. | Coarse graph; local refined graph; corridor subgraphs |
| Edge evaluator | Propagate motion along an edge under current, wind, polar, leeway, and tack logic. | Feasible/infeasible edge; travel time; maneuver annotations |
| Search and replan | Run time-dependent search on (node,time,tack,age); aggregate uncertainty; produce route and tactical updates. | Route geometry; ETA band; tack gates; diagnostic traces |

## 5. Data sources and assumptions

The design assumes the following baseline data products and interfaces.

| Input | Minimum requirement | Notes |
| --- | --- | --- |
| SSCOFS | Hourly 3-D currents on native unstructured mesh | Use native node/element geometry for interpolation and local resolution estimates. |
| Charts | ENC/S-57 or equivalent shoreline, depth, rock, drying, restricted-area objects | Charts define legal navigation space and under-keel constraints. |
| Wind | ECMWF HRES and, ideally, ENS or perturbed alternatives | At ~9 km, this is too coarse for graph density; correct it locally before routing. |
| Vessel model | Polars or VPP table by TWS/TWA; leeway estimate; tack recovery model | If available, calibrate from race or instrument logs. |
| Bathymetry / safety | Draft, required under-keel clearance, user-defined hazard buffers | Can vary by vessel mode and tide stage. |
| Logs for calibration | Boat speed, heading, course over ground, wind, current estimate if available | Used to fit maneuver loss and route realism. |

Assumption set used in this document:

- The routing horizon can extend far enough that wind uncertainty is operationally meaningful.

- The vessel is sailing, not motoring as a primary mode, so upwind no-go regions and tacking decisions are central.

- The target use case prefers a credible and stable route over a marginally faster but noisy or twitchy solution.

- The current model is materially better resolved than the wind model in the near-shore domain.

## 6. Navigation domain model

The legal routing domain must be derived from chart geometry first and only then filtered by the model's wet cells. The chart domain should include shoreline, drying areas, rocks, restricted areas, and a tide- and draft-aware under-keel-clearance rule.

Construct the navigable polygon by subtracting all forbidden areas from the chart water polygon, then applying a safety buffer. The safety buffer should be larger than zero even if the chart says the water is legally open; this creates routing clearance around shoreline complexity and allows the graph to remain numerically stable.

- Compute distance-to-shore on the navigable domain.

- Estimate local channel width from medial-axis or nearest-opposite-boundary calculations.

- Mark must-pass corridors in narrow entrances and cuts so the graph preserves them even if generic sampling would miss them.

- Use SSCOFS wetness only as a supporting mask and interpolation domain, not as the primary legal shoreline definition.

> Reason for this choice. Ocean-model meshes are built to resolve hydrodynamics, not to represent every legal navigational nuance. Inlets, harbor mouths, drying flats, and man-made constraints must come from chart geometry.

## 7. Graph construction

This is the central structural decision. The router should use SSCOFS as the physics mesh and build a derived navigation graph on top of it.

For each candidate routing region, define a local target path scale ΔP(x) such as:

```text
ΔP(x) = clamp(
    min(
        2 * h_mesh(x),
        0.25 * w_channel(x),
        k1 / (eps + |∇c(x)|),
        k2 / (eps + |∂c/∂t(x)|)
    ),
    ΔP_min,
    ΔP_max
)
```

Where h_mesh(x) is local SSCOFS triangle size, w_channel(x) is local channel width, |∇c| is current shear magnitude, and |∂c/∂t| is local current variability. This rule does three useful things at once: it refines near shore, refines where the current field changes fast, and stays coarse where the water is open and quiet.

Recommended node-generation sequence:

1. Generate seed nodes from legal-water polygon plus must-pass centerlines.

1. Augment seed density where distance-to-shore is small, where channel width is small, and where current shear is large.

1. Snap or associate each routing node to the nearest valid SSCOFS interpolation support, such as containing element and barycentric weights.

1. For each routing node, build candidate neighbors by heading bins, not raw mesh adjacency. Use 16 to 24 angular sectors and keep the nearest valid edge in each sector.

1. Reject edges that cross shoreline, violate depth or clearance rules, or exceed local curvature expectations in narrow passages.

1. Optionally apply quasi-collinear edge pruning so multiple edges with almost identical headings collapse to the shortest useful one.

Use a hierarchical search graph. First solve on a coarse graph for a corridor. Then re-solve inside a corridor-expanded local graph at higher density. The result is a high-resolution route without paying full fine-grid search cost across the whole domain.

## 8. Environmental field handling

### 8.1 Current field

Use native SSCOFS geometry for interpolation. Do not regrid currents to a uniform raster except possibly for visualization or debugging.

- Precompute, for each routing node and each edge substep sample, the containing element and barycentric weights where possible.

- If a vessel draft makes vertical shear relevant, use a depth-weighted or draft-relevant current rather than blindly taking the surface layer.

- Resample time only as needed. For most implementations, linear interpolation in time between hourly fields is adequate.

Recommended edge substep size is the minimum of: 50 to 100 m, 20 percent of edge length, and a CFL-like distance implied by local current variability. Shorter substeps are justified in very narrow cuts or strong shear.

### 8.2 Wind field

ECMWF at approximately 9 km should be treated as a synoptic-scale forcing field, not as a truthful 100 m coastal wind map. Near shore, the unresolved effects are terrain shelter, channeling, shoreline turning, and localized accelerations.

Use one of the following correction models, listed in preferred order.

1. Offline local wind atlas: WRF to CFD or LES, or sector-based CFD/RANS over the local terrain and shoreline.

1. Sector-based empirical map: for each inflow sector and speed bin, store a speed-up multiplier and turning angle field.

1. Data-driven surrogate of a validated local model, if you can verify it against mast, station, or race-course observations.

A practical online representation is:

```text
U_loc(x, t) = m_k(x) * R(δ_k(x)) * U_ecmwf(t) + b_k(x)

where:
  k       = inflow sector / speed / stability bin
  m_k(x)  = local shelter or speed-up multiplier
  δ_k(x)  = local turning angle
  b_k(x)  = optional bias term
```

This local correction feeds the vessel polar and no-go logic. It does not redefine graph spacing.

## 9. Vessel model and edge propagation

Edge evaluation should solve a constrained sailing problem, not simply add vector current to a fixed boat-speed scalar.

For a substep along a route tangent t̂, solve for heading ψ such that the boat's ground-track aligns with the segment:

```text
x_dot = c(x, t) + V_polar(TWS, TWA) * h_hat(ψ)

Choose ψ so that normalize(x_dot) ≈ t_hat

subject to:
  TWA outside the no-go zone
  heel / mode / reef limits if modeled
  optional leeway model
```

If no feasible heading satisfies the segment direction because the edge would force the vessel too deep into the no-go zone, mark the edge infeasible for that state and time.

Leeway can be handled either from a separate table or as a low-order function of apparent wind angle, wind speed, and boat mode. For route quality, getting the sign and approximate magnitude correct is usually more important than building a very high-order leeway model.

At the edge level, compute:

- Travel time through the segment under current and local wind

- Whether the move preserves or changes tack

- Whether the segment enters a low-clearance or high-risk zone

- Whether the segment depends on a highly uncertain wind angle or timing window

## 10. State space and cost function

The minimum useful search state is:

```text
state s = (node_id, time_bin, tack, age_bin)
```

Where tack is port or starboard, and age_bin is the time or distance since the last tack. age_bin is important because it allows the search to represent hysteresis: a tack immediately after a previous tack is more expensive than a tack after a long committed leg.

A practical edge cost under scenario m is:

```text
c_e^(m) =
    Δt_e^(m)
  + 1[tack changes] * (τ_phys + τ_persist + τ_unc)
  + λ_clear * φ(clearance)
  + λ_shallow * φ(UKC)
  + λ_wait * 1[edge is wait]
```

Where Δt_e^(m) is the segment travel time under scenario m, τ_phys is physical tack loss, τ_persist is the anti-chatter term, and τ_unc grows with forecast uncertainty or lead time.

## 11. Tack management under uncertainty

This section is the policy layer that makes the route look sane on a long race.

A tack should occur only when the predicted gain clearly exceeds a switching threshold:

```text
Tack only if predicted_gain > τ_phys + τ_persist + τ_unc
```

Use the following decomposition.

### 11.1 Physical tack loss

Estimate maneuver loss as an equivalent time penalty using instrument or race logs. The cleanest calibration variable is loss of over-ground VMG on the relevant course axis, not just temporary drop in through-water speed.

```text
τ_phys =
  (1 / VMG_ref) * ∫_0^Trec max(VMG_ref - VMG_obs(t), 0) dt
```

Fit τ_phys at least as a function of true wind speed and whether the boat is sailing upwind or reaching. If you have sea-state or chop labels, include them. For routing realism, a coarse but data-backed function is better than a sophisticated uncalibrated one.

### 11.2 Persistence penalty

The route should prefer long, committed legs. Use a persistence term such as:

```text
τ_persist = λ0 + λ1 * exp(-age / a0)
```

This gives every tack a baseline cost and makes a second tack soon after the last one substantially more expensive. It prevents chatter and forecast-noise zig-zagging.

Also add a hard dwell constraint unless safety or legality requires an exception:

```text
No opposite tack until age >= age_min
```

A good initial implementation is to measure age in minutes. A better production implementation tracks both time and sailed distance, then uses the stricter of the two thresholds.

*Figure 2. Example hysteresis logic. Soon after a tack, the route must show a much larger predicted gain before another tack is allowed. The threshold should rise further when forecast uncertainty is higher.*

### 11.3 Uncertainty penalty

Because your wind forecast is much coarser and less trustworthy than your current model, tack aversion should grow with uncertainty. One practical form is:

```text
τ_unc(h) = β0 + β1 * σ_dir(h) + β2 * σ_speed(h)
```

Where σ_dir and σ_speed are uncertainty metrics at forecast horizon h, either from ECMWF ENS spread or from synthetic perturbations if ENS is not in the first release.

This does what you want operationally: a small far-horizon wind shift no longer triggers a speculative strategic tack unless the expected gain is decisively larger than the uncertainty margin.

### 11.4 Robust route objective

Do not optimize only deterministic ETA. Use a route score such as:

```text
J(route) =
    E[ETA]
  + λN * N_tack
  + λS * Σ exp(-leg_length_k / L0)
  + λR * Risk(ETA)
```

Here Risk(ETA) can be standard deviation, a high-percentile ETA such as P90, or conditional value at risk if you want stronger control of bad tails. The exp(-leg_length_k / L0) term penalizes short legs more strongly than long ones and works well with the hysteresis rule.

> Practical interpretation. The router is allowed to tack. It is simply required to prove that the tack still makes sense after accounting for maneuver loss, the desire for persistence, and the possibility that the wind shift is wrong.

## 12. Search algorithm

Use time-dependent A* on the state (node, time, tack, age). If the heuristic is weak or difficult to certify, fall back to time-dependent Dijkstra for correctness in the first release.

Recommended heuristic:

```text
h(n) = great_circle_distance(n, goal) / Vmax_ground

where Vmax_ground is a conservative upper bound on attainable SOG
```

The heuristic should ignore favorable current and favorable wind shifts unless you can prove admissibility. Underestimate rather than overestimate.

Support wait edges only at explicitly allowed holding nodes or safe loiter areas. Waiting can be necessary when a future tide gate or wind change makes a later departure faster overall.

Use a coarse-to-fine solve:

1. Coarse graph solve to get a corridor and rough ETA envelope.

1. Refine graph density inside a corridor around the coarse path.

1. Run fine solve inside the corridor.

1. As new forecasts arrive, re-run only the rolling horizon ahead of the vessel.

## 13. Uncertainty handling

The first release does not need a full probabilistic control framework, but it should not ignore uncertainty.

Use one of two implementations.

### 13.1 Preferred

Evaluate each edge across a small set of wind scenarios, either from ECMWF ENS or from synthetic perturbations around the deterministic forecast. Aggregate by mean plus risk penalty:

```text
c̄_e = mean_m(c_e^(m)) + λR * risk_m(c_e^(m))
```

Because the current model is assumed strong, you can keep current deterministic in the first version and perturb wind timing, direction, and speed only.

### 13.2 Minimal viable

If ENS is not wired in yet, use deterministic wind plus an uncertainty proxy derived from forecast lead time and local terrain sensitivity. Then increase tack aversion, no-go margin, or route-side commitment as lead time grows.

This is weaker than ensemble routing, but materially better than pretending the 9 km wind is exact.

## 14. Software design

A Python-first implementation is reasonable, especially if heavy geometry and interpolation kernels are vectorized or moved to NumPy, Numba, Cython, or Rust later. Keep the data model explicit. Avoid baking routing assumptions into ad hoc arrays.

| Package / component | Suggested contents |
| --- | --- |
| data/ | Readers for SSCOFS NetCDF, ECMWF files, ENC/geometry loaders, cache builders |
| domain/ | Water polygon construction, depth filters, shoreline distance, channel width, corridor expansion |
| graph/ | Node sampling, heading-bin neighbor generation, edge legality checks, pruning |
| env/ | Current sampler, local wind corrector, uncertainty provider |
| vessel/ | Polar interpolation, no-go logic, leeway, tack-loss calibration |
| search/ | State encoding, time-dependent A*, wait edges, robust objective aggregation |
| sim/ | Edge propagator, replay tools, scenario evaluation |
| validation/ | Regression routes, calibration notebooks, metrics, visualization |
| ui_or_api/ | CLI or service layer for requests, route export, and diagnostics |

## 15. Implementation plan

Build this in four phases. Do not start with ensemble logic or CFD if the base geometry and edge evaluator are not yet correct.

| Phase | Deliverable | Exit criteria |
| --- | --- | --- |
| 1. Deterministic core | Chart-constrained adaptive graph; SSCOFS current interpolation; deterministic ECMWF wind; time-dependent search on (node,time,tack,age) | Valid routes through narrow areas; sensible tack count on historical cases |
| 2. Local wind correction | Sector-based near-shore wind correction layer | Route choice changes plausibly near terrain shelter and channeling zones |
| 3. Robust routing | Scenario bundle or ENS support; ETA-risk objective; uncertainty-aware tack penalty | Routes become less twitchy and more stable under perturbed winds |
| 4. Operational hardening | Rolling replanning, caching, performance tuning, diagnostic tooling | Fast reroute, reproducible logs, and route explanations for why a tack was or was not chosen |

## 16. Default parameters for the first working version

These are starting values, not final tuning targets. They should be revised after replaying real tracks.

| Parameter | Initial value | Comment |
| --- | --- | --- |
| ΔP_min | 75-100 m | Use lower end only where chart complexity and current shear justify it. |
| ΔP_max | 400-800 m | Open-water coarse scale inside the same domain. |
| Heading bins | 16 | Increase to 24 if upwind geometry is too coarse. |
| Edge substep | min(75 m, 0.2 * edge length) | Shorten in strong shear or in harbor entrances. |
| Time bin | 5 min | Use 10 min if runtime dominates and route quality holds. |
| age_min | 15-30 min upwind | Also consider 0.5-1.0 NM minimum leg distance. |
| τ_phys | From logs; seed 60-180 s | Calibrate by TWS and point of sail. |
| λ0 | 2-5 min | Baseline tack aversion beyond physical loss. |
| λ1 | 5-10 min | Extra anti-chatter penalty immediately after a tack. |
| L0 | 0.75-1.5 NM | Scale for short-leg penalty in the robust objective. |
| Risk metric | P90 ETA or ETA stdev | Pick one for initial simplicity. |
| Corridor half-width | 1-3 local node spacings | Increase if coarse solve misses alternate sides. |

## 17. Main algorithm sketch

```text
for each forecast cycle:
    load charts and legal-water domain
    load SSCOFS current fields
    load ECMWF wind and uncertainty inputs
    build / update local wind correction layer

    if no coarse graph exists or domain changed:
        build coarse adaptive graph from chart geometry + SSCOFS scale

    route0 = TD_A_star(coarse_graph, start_state, goal, robust_cost)

    corridor = expand(route0, width = corridor_half_width)
    fine_graph = build_refined_graph(corridor)

    route1 = TD_A_star(fine_graph, start_state, goal, robust_cost)

    publish:
        polyline
        ETA mean / risk band
        tack gates
        diagnostics (why tack chosen, why edges rejected)

during execution:
    replan on rolling horizon
    decide exact tack timing only in the short horizon
    preserve strategic corridor unless new data strongly invalidates it
```

## 18. Validation and acceptance tests

Validation must include both numerical correctness and sailor credibility. A route that is mathematically optimal but looks like forecast-noise chatter is not acceptable.

### 18.1 Unit tests

- Current interpolation on known SSCOFS elements and time interpolation across hourly fields.

- Edge legality against shoreline and depth polygons, including narrow-channel corner cases.

- No-go logic and tack-state transitions.

- Equivalent time-loss calibration from maneuver logs.

### 18.2 Regression tests

- Historical routes through narrow areas with known safe passages.

- Cases with strong tidal currents where the current should dominate route geometry.

- Cases with plausible wind-shift ambiguity where the robust route should tack less than the deterministic route.

### 18.3 Operational metrics

| Metric | Target behavior |
| --- | --- |
| Route legality | Zero land intersections; zero depth violations under the chosen UKC rule |
| Tack count | Consistent with human expectation for similar conditions |
| ETA stability | Small changes under modest wind perturbations; major route changes only when the scenario meaningfully changes |
| Replan stability | Rolling forecasts should not cause repeated corridor flips unless conditions genuinely reverse |
| Runtime | Fast enough for frequent replans on the target hardware |

## 19. Risks and mitigation

| Risk | Mitigation |
| --- | --- |
| Wind too coarse for harbor-scale routing | Use a local correction atlas and restrict far-horizon tactical precision. |
| Raw graph misses very narrow passes | Preserve must-pass centerlines and add local refinement around entrances. |
| Too many speculative tacks | Increase age-based hysteresis, tack budget, or ETA-risk weight. |
| Routes depend too strongly on one forecast cycle | Use ENS or synthetic perturbations and penalize ETA tail risk. |
| Runtime too high | Use hierarchical corridor search, cached interpolation, vectorized edge propagation, and edge pruning. |
| Human trust is low | Emit diagnostics: why tack chosen, which constraint blocked alternatives, and ETA uncertainty band. |

## 20. Final recommendation

For this problem, the best single design is:

- An adaptive chart-constrained navigation graph whose density is informed by SSCOFS mesh scale, shoreline distance, channel width, and current shear.

- Native-current interpolation from SSCOFS, with edge propagation performed by substeps.

- A separate local wind layer derived from ECMWF plus sector-based correction or higher-resolution downscaling.

- A time-dependent search state (node, time, tack, age) with explicit no-go feasibility, leeway, and tack hysteresis.

- A robust objective that minimizes expected ETA while penalizing tack count, short legs, and ETA risk under wind uncertainty.

- A coarse-to-fine corridor workflow with rolling replanning so strategic side choice is stable and exact tack timing remains tactical.

That design is technically defensible, consistent with recent routing literature, and realistic for an implementation that starts with the data you already have: high-resolution SSCOFS currents and coarser ECMWF wind.

## References

[1] NOAA CO-OPS. Salish Sea and Columbia River Operational Forecast System (SSCOFS).

[2] NOAA/NWS. Service Change Notice 24-77 Updated: SSCOFS implementation details and mesh resolution, 2024.

[3] ECMWF. Medium-range forecasts. HRES and ENS model configuration overview.

[4] Mannarini G, Orović J, et al. VISIR-2: ship weather routing in Python. Geosci Model Dev. 2024;17:4355-4392.

[5] Hou J, et al. Inshore sailing route optimization integrating terrain-influenced wind field. Ocean Engineering. 2025.

[6] Jeong D-G, Roh M-I, Yeo I-C, et al. A route planning method for small ships in coastal areas based on quadtree. Int J Nav Archit Ocean Eng. 2025.

[7] Zhou Y, et al. Efficient Weather Routing Method in Coastal and Island-Rich Waters Guided by Ship Trajectory Big Data. J Mar Sci Eng. 2025;13(9):1801.

[8] Marjanović M, Prpić-Oršić J, Turk A, Valčić M. Anomalous Behavior in Weather Forecast Uncertainty: Implications for Ship Weather Routing. J Mar Sci Eng. 2025;13(6):1185.

[9] Wang W, et al. Wind Field Modeling over Hilly Terrain: A Review of Methods, Challenges, Limitations, and Future Directions. Applied Sciences. 2025;15(18):10186.
