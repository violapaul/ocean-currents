"""
water_boundary.py
-----------------
Delaunay-based water domain detection and boundary extraction.

Uses Delaunay triangulation of SSCOFS element centers to:
1. Classify triangles as water (valid) or land-spanning (invalid)
2. Extract the boundary polygon(s) of the water domain
3. Export as compact GeoJSON for use in the web app

The key insight: SSCOFS element centers are dense in water (50-200m near coast)
and absent over land. Delaunay triangles that span land gaps have anomalously
long edges compared to the local mesh density.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional, Union

import numpy as np
from scipy.spatial import Delaunay


def compute_local_scale(delaunay: Delaunay, points: np.ndarray) -> np.ndarray:
    """Compute the local mesh scale at each vertex.
    
    For each vertex, finds its Delaunay neighbors and returns a robust
    estimate of the local element spacing. Uses the 25th percentile of
    neighbor distances to avoid being skewed by long edges across gaps.
    
    Parameters
    ----------
    delaunay : Delaunay
        Delaunay triangulation.
    points : ndarray, shape (n, 2)
        Vertex coordinates.
        
    Returns
    -------
    local_scale : ndarray, shape (n,)
        25th percentile neighbor distance for each vertex.
    """
    n_points = len(points)
    
    # Build vertex-to-neighbor adjacency from triangles
    neighbors = defaultdict(set)
    for tri in delaunay.simplices:
        for i in range(3):
            for j in range(3):
                if i != j:
                    neighbors[tri[i]].add(tri[j])
    
    # Compute 25th percentile neighbor distance for each vertex
    # This is more robust than median at gap edges where long edges
    # would otherwise skew the local scale high.
    local_scale = np.zeros(n_points)
    for v in range(n_points):
        if v in neighbors and len(neighbors[v]) > 0:
            nbr_list = list(neighbors[v])
            dists = np.linalg.norm(points[nbr_list] - points[v], axis=1)
            local_scale[v] = np.percentile(dists, 25) if len(dists) >= 4 else np.min(dists)
        else:
            local_scale[v] = np.inf
    
    return local_scale


def build_water_mask(
    lon: np.ndarray,
    lat: np.ndarray,
    threshold_factor: float = 3.5,
    max_aspect_ratio: float = 6.0,
    max_edge_m: float = 2000.0,
    use_utm: bool = False,
    utm_zone: int = 10,
) -> Tuple[Delaunay, np.ndarray]:
    """Build Delaunay triangulation and classify water vs land-spanning triangles.
    
    Three independent filters (any one triggers invalidation):
    1. Local scale: max edge > threshold_factor * local mesh density
    2. Aspect ratio: max_edge / min_edge > max_aspect_ratio (catches
       elongated slivers that bridge narrow peninsulas)
    3. Absolute edge length: max edge > max_edge_m (catches convex-hull
       boundary artifacts; set high enough to allow coarse open-water mesh)
    
    Parameters
    ----------
    lon, lat : ndarray
        Element center coordinates in degrees.
    threshold_factor : float
        A triangle is land-spanning if its max edge exceeds this factor times
        the local mesh scale (min of the three vertices' local scales).
    max_aspect_ratio : float
        Reject triangles where max_edge / min_edge exceeds this value.
    max_edge_m : float
        Absolute maximum edge length in meters. Catches convex-hull boundary
        artifacts. Must be larger than the coarsest valid mesh spacing (~1km
        in the open Strait of Juan de Fuca).
    use_utm : bool
        If True, convert to UTM for more accurate distance calculations.
    utm_zone : int
        UTM zone number (default 10 for Puget Sound).
        
    Returns
    -------
    delaunay : Delaunay
        The triangulation object.
    valid_mask : ndarray, shape (n_triangles,), dtype=bool
        True for water triangles, False for land-spanning.
    """
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    
    if use_utm:
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:326{utm_zone:02d}", always_xy=True)
        x, y = transformer.transform(lon, lat)
        points = np.column_stack([x, y])
    else:
        lat_scale = 111_000  # meters per degree latitude
        lon_scale = 111_000 * np.cos(np.radians(np.mean(lat)))
        points = np.column_stack([lon * lon_scale, lat * lat_scale])
    
    delaunay = Delaunay(points)
    local_scale = compute_local_scale(delaunay, points)
    
    # Classify each triangle
    n_tri = len(delaunay.simplices)
    valid_mask = np.ones(n_tri, dtype=bool)
    
    for t in range(n_tri):
        v0, v1, v2 = delaunay.simplices[t]
        p0, p1, p2 = points[v0], points[v1], points[v2]
        
        e01 = np.linalg.norm(p1 - p0)
        e12 = np.linalg.norm(p2 - p1)
        e20 = np.linalg.norm(p0 - p2)
        edges = sorted([e01, e12, e20])
        max_edge = edges[2]
        min_edge = edges[0]
        
        # Filter 1: local scale threshold
        local_thresh = threshold_factor * min(local_scale[v0], local_scale[v1], local_scale[v2])
        if max_edge > local_thresh:
            valid_mask[t] = False
            continue
        
        # Filter 2: absolute max edge length
        if max_edge > max_edge_m:
            valid_mask[t] = False
            continue
        
        # Filter 3: aspect ratio (elongated slivers bridging peninsulas)
        if min_edge > 0 and max_edge / min_edge > max_aspect_ratio:
            valid_mask[t] = False
    
    # Store points on the delaunay object for later use
    delaunay.input_points = points
    
    return delaunay, valid_mask


def refine_with_velocity(
    delaunay: Delaunay,
    valid_mask: np.ndarray,
    velocity_frames: List[np.ndarray],
    min_edge_m: float = 400.0,
    angle_threshold: float = 90.0,
    min_consistent_hours: int = 3,
) -> np.ndarray:
    """Refine water mask using velocity coherence across multiple hours.
    
    At real coastlines, adjacent elements have similar current directions
    (no-normal-flow boundary condition). Across land peninsulas, elements
    on opposite sides often have consistently opposing currents.
    
    Only examines triangles with edges longer than min_edge_m to avoid
    rejecting small tidal-reversal triangles near the coast.
    
    Parameters
    ----------
    delaunay : Delaunay
        Triangulation from build_water_mask().
    valid_mask : ndarray
        Boolean mask from build_water_mask() (modified in place).
    velocity_frames : list of ndarray
        Each entry is shape (n_elements, 2) with [u, v] in m/s.
        Multiple hours provide robustness against tidal reversals.
    min_edge_m : float
        Only check triangles with max_edge above this length.
    angle_threshold : float
        Maximum allowed angle (degrees) between vertex velocity pairs.
    min_consistent_hours : int
        Number of hours the angle must exceed threshold to reject.
        
    Returns
    -------
    valid_mask : ndarray
        Updated mask with velocity-inconsistent triangles removed.
    """
    points = delaunay.input_points
    simplices = delaunay.simplices
    n_frames = len(velocity_frames)
    rejected = 0
    
    for t in range(len(simplices)):
        if not valid_mask[t]:
            continue
        
        v0, v1, v2 = simplices[t]
        p0, p1, p2 = points[v0], points[v1], points[v2]
        
        e01 = np.linalg.norm(p1 - p0)
        e12 = np.linalg.norm(p2 - p1)
        e20 = np.linalg.norm(p0 - p2)
        if max(e01, e12, e20) < min_edge_m:
            continue
        
        # Count hours where max pairwise velocity angle exceeds threshold
        bad_hours = 0
        for uv in velocity_frames:
            max_angle = 0.0
            for vi, vj in [(v0, v1), (v1, v2), (v2, v0)]:
                ui, uj = uv[vi], uv[vj]
                si = np.sqrt(ui[0]**2 + ui[1]**2)
                sj = np.sqrt(uj[0]**2 + uj[1]**2)
                if si < 0.005 or sj < 0.005:
                    continue  # near-zero speed, skip
                dot = (ui[0]*uj[0] + ui[1]*uj[1]) / (si * sj)
                angle = np.degrees(np.arccos(np.clip(dot, -1, 1)))
                if angle > max_angle:
                    max_angle = angle
            if max_angle > angle_threshold:
                bad_hours += 1
        
        if bad_hours >= min_consistent_hours:
            valid_mask[t] = False
            rejected += 1
    
    return valid_mask, rejected


def load_nav_mask(path: Union[str, Path]) -> Tuple[np.ndarray, Tuple[float, float, float, float], float]:
    """Load navigable water mask from .npz file.
    
    Parameters
    ----------
    path : str or Path
        Path to nav_mask.npz created by build_nav_mask.py.
        
    Returns
    -------
    water_grid : ndarray, shape (ny, nx), dtype bool
        True where water is navigable.
    bounds : tuple (lon_min, lon_max, lat_min, lat_max)
        Geographic extent of the grid.
    res_deg : float
        Grid resolution in degrees.
    """
    data = np.load(path)
    water_grid = data['water'].astype(bool)
    bounds = (
        float(data['lon_min']),
        float(data['lon_max']),
        float(data['lat_min']),
        float(data['lat_max']),
    )
    res_deg = float(data['res_deg'])
    return water_grid, bounds, res_deg


def _is_water_at(lon_pts, lat_pts, water_grid, lon_min, lat_min, res_deg, ny, nx):
    """Look up water status for arrays of lon/lat points.
    
    Points outside the DEM extent are treated as water (open ocean).
    """
    col = ((lon_pts - lon_min) / res_deg).astype(int)
    row = ((lat_pts - lat_min) / res_deg).astype(int)
    in_bounds = (col >= 0) & (col < nx) & (row >= 0) & (row < ny)
    result = np.ones(len(lon_pts), dtype=bool)  # default: water
    result[in_bounds] = water_grid[row[in_bounds], col[in_bounds]]
    return result


def refine_with_nav_mask(
    delaunay: Delaunay,
    valid_mask: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    water_grid: np.ndarray,
    bounds: Tuple[float, float, float, float],
    res_deg: float,
) -> Tuple[np.ndarray, int]:
    """Refine water mask by rejecting triangles whose edges cross land.
    
    Samples points along each triangle edge at DEM resolution and checks
    against the DEM-derived water mask.  This gives a clean coastline that
    follows the DEM depth contour while keeping the Delaunay mesh topology
    smooth.
    
    Parameters
    ----------
    delaunay : Delaunay
        Triangulation from build_water_mask().
    valid_mask : ndarray
        Boolean mask (modified in place).
    lon, lat : ndarray
        Element coordinates in degrees.
    water_grid : ndarray, shape (ny, nx)
        Boolean grid where True = navigable water.
    bounds : tuple (lon_min, lon_max, lat_min, lat_max)
        Geographic extent of water_grid.
    res_deg : float
        Grid resolution in degrees.
        
    Returns
    -------
    valid_mask : ndarray
        Updated mask with land-crossing triangles removed.
    rejected : int
        Number of additionally rejected triangles.
    """
    lon_min, lon_max, lat_min, lat_max = bounds
    ny, nx = water_grid.shape
    
    simplices = delaunay.simplices
    valid_idx = np.where(valid_mask)[0]
    
    if len(valid_idx) == 0:
        return valid_mask, 0
    
    # Sample spacing: half a DEM pixel to catch every land pixel an edge crosses
    step = res_deg * 0.5
    
    rejected = 0
    for i in range(len(valid_idx)):
        t = valid_idx[i]
        v0, v1, v2 = simplices[t]
        
        hit_land = False
        for va, vb in [(v0, v1), (v1, v2), (v2, v0)]:
            dlon = lon[vb] - lon[va]
            dlat = lat[vb] - lat[va]
            edge_len = max(abs(dlon), abs(dlat))
            n_samples = max(int(edge_len / step) + 1, 2)
            
            ts = np.linspace(0, 1, n_samples)
            sample_lon = lon[va] + ts * dlon
            sample_lat = lat[va] + ts * dlat
            
            water = _is_water_at(sample_lon, sample_lat,
                                 water_grid, lon_min, lat_min, res_deg, ny, nx)
            if not water.all():
                hit_land = True
                break
        
        if hit_land:
            valid_mask[t] = False
            rejected += 1
    
    return valid_mask, rejected


def extract_boundary_edges(delaunay: Delaunay, valid_mask: np.ndarray) -> List[Tuple[int, int]]:
    """Extract boundary edges of the valid water region.
    
    A boundary edge is one where:
    - The triangle is valid AND
    - The neighbor on that edge is either invalid or doesn't exist (convex hull)
    
    Parameters
    ----------
    delaunay : Delaunay
        The triangulation.
    valid_mask : ndarray
        Boolean mask of valid triangles.
        
    Returns
    -------
    edges : list of (int, int)
        List of boundary edges as vertex index pairs.
    """
    edges = []
    simplices = delaunay.simplices
    neighbors = delaunay.neighbors
    
    for t in range(len(simplices)):
        if not valid_mask[t]:
            continue
        
        tri = simplices[t]
        nbrs = neighbors[t]
        
        # For each edge (opposite to vertex i)
        for i in range(3):
            nbr = nbrs[i]
            # Boundary if neighbor is -1 (hull) or invalid
            if nbr == -1 or not valid_mask[nbr]:
                # Edge is between vertices (i+1)%3 and (i+2)%3
                v_a = tri[(i + 1) % 3]
                v_b = tri[(i + 2) % 3]
                edges.append((v_a, v_b))
    
    return edges


def chain_edges_to_rings(edges: List[Tuple[int, int]]) -> List[List[int]]:
    """Chain boundary edges into closed rings.
    
    Parameters
    ----------
    edges : list of (int, int)
        Unordered boundary edges.
        
    Returns
    -------
    rings : list of list of int
        Each ring is a list of vertex indices forming a closed loop.
    """
    # Build adjacency: each vertex connects to others via boundary edges
    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    
    # Track which edges we've used
    used_edges = set()
    rings = []
    
    for start_v in adj:
        if not adj[start_v]:
            continue
        
        for next_v in adj[start_v]:
            edge_key = (min(start_v, next_v), max(start_v, next_v))
            if edge_key in used_edges:
                continue
            
            # Start a new ring
            ring = [start_v]
            prev_v = start_v
            curr_v = next_v
            used_edges.add(edge_key)
            
            while curr_v != start_v:
                ring.append(curr_v)
                # Find next vertex (not the one we came from)
                candidates = [v for v in adj[curr_v] if v != prev_v]
                if not candidates:
                    break
                
                # Pick an unused edge if possible
                found = False
                for cand in candidates:
                    edge_key = (min(curr_v, cand), max(curr_v, cand))
                    if edge_key not in used_edges:
                        used_edges.add(edge_key)
                        prev_v = curr_v
                        curr_v = cand
                        found = True
                        break
                
                if not found:
                    break
            
            if len(ring) >= 3 and curr_v == start_v:
                rings.append(ring)
    
    return rings


def ring_signed_area(coords: np.ndarray) -> float:
    """Compute signed area of a ring (shoelace formula).
    
    Positive = counter-clockwise, Negative = clockwise.
    """
    n = len(coords)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += coords[i, 0] * coords[j, 1]
        area -= coords[j, 0] * coords[i, 1]
    return area / 2.0


def extract_boundary(
    delaunay: Delaunay,
    valid_mask: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
) -> Dict[str, Any]:
    """Extract water domain boundary as GeoJSON-ready structure.
    
    Parameters
    ----------
    delaunay : Delaunay
        The triangulation.
    valid_mask : ndarray
        Boolean mask of valid triangles.
    lon, lat : ndarray
        Original coordinates in degrees.
        
    Returns
    -------
    geojson : dict
        GeoJSON FeatureCollection with a single MultiPolygon feature.
    """
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    
    edges = extract_boundary_edges(delaunay, valid_mask)
    if not edges:
        return {"type": "FeatureCollection", "features": []}
    
    rings = chain_edges_to_rings(edges)
    if not rings:
        return {"type": "FeatureCollection", "features": []}
    
    # Convert vertex indices to lon/lat coordinates
    ring_coords = []
    for ring in rings:
        coords = np.column_stack([lon[ring], lat[ring]])
        # Close the ring
        coords = np.vstack([coords, coords[0]])
        ring_coords.append(coords)
    
    # Sort rings by absolute area (largest first = outer rings)
    ring_areas = [(i, ring_signed_area(rc[:-1])) for i, rc in enumerate(ring_coords)]
    ring_areas.sort(key=lambda x: -abs(x[1]))
    
    # Group into polygons: outer rings (CCW, positive area) with their holes (CW, negative)
    # For GeoJSON MultiPolygon: [[[outer], [hole1], [hole2]], [[outer2], ...]]
    polygons = []
    used = set()
    
    for idx, area in ring_areas:
        if idx in used:
            continue
        
        coords = ring_coords[idx]
        
        # Ensure outer ring is CCW (positive area)
        if area < 0:
            coords = coords[::-1]
            area = -area
        
        polygon = [coords.tolist()]
        used.add(idx)
        
        # Find holes: rings fully contained within this one
        # (simplified: just check if CW and not yet used)
        # A proper implementation would do point-in-polygon tests
        for hole_idx, hole_area in ring_areas:
            if hole_idx in used:
                continue
            if hole_area > 0:
                continue  # Not a hole (CCW)
            
            hole_coords = ring_coords[hole_idx]
            # Ensure hole is CW (negative area in original)
            if ring_signed_area(hole_coords[:-1]) > 0:
                hole_coords = hole_coords[::-1]
            
            polygon.append(hole_coords.tolist())
            used.add(hole_idx)
        
        polygons.append(polygon)
    
    feature = {
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "MultiPolygon",
            "coordinates": polygons,
        },
    }
    
    return {"type": "FeatureCollection", "features": [feature]}


def export_geojson(
    boundary: Dict[str, Any],
    filepath: Union[str, Path],
    precision: int = 6,
) -> int:
    """Export boundary to GeoJSON file.
    
    Parameters
    ----------
    boundary : dict
        GeoJSON structure from extract_boundary().
    filepath : str or Path
        Output file path.
    precision : int
        Decimal places for coordinates (6 = ~0.1m precision).
        
    Returns
    -------
    size_bytes : int
        Size of the written file.
    """
    filepath = Path(filepath)
    
    # Round coordinates to reduce file size
    def round_coords(obj):
        if isinstance(obj, list):
            if obj and isinstance(obj[0], (int, float)):
                return [round(x, precision) for x in obj]
            return [round_coords(item) for item in obj]
        return obj
    
    if boundary.get("features"):
        for feature in boundary["features"]:
            if "geometry" in feature and "coordinates" in feature["geometry"]:
                feature["geometry"]["coordinates"] = round_coords(
                    feature["geometry"]["coordinates"]
                )
    
    with open(filepath, "w") as f:
        json.dump(boundary, f, separators=(",", ":"))
    
    return filepath.stat().st_size


def build_water_mask_utm(
    x_utm: np.ndarray,
    y_utm: np.ndarray,
    threshold_factor: float = 3.5,
    max_aspect_ratio: float = 6.0,
    max_edge_m: float = 2000.0,
) -> Tuple[Delaunay, np.ndarray]:
    """Build Delaunay triangulation from UTM coordinates directly.
    
    This is for use in sail_routing.py where coordinates are already in UTM.
    
    Parameters
    ----------
    x_utm, y_utm : ndarray
        UTM coordinates in meters.
    threshold_factor : float
        Triangle classification threshold.
    max_aspect_ratio : float
        Reject triangles where max_edge / min_edge exceeds this.
    max_edge_m : float
        Absolute maximum edge length in meters.
        
    Returns
    -------
    delaunay : Delaunay
        The triangulation.
    valid_mask : ndarray
        Boolean mask of valid triangles.
    """
    points = np.column_stack([x_utm, y_utm])
    delaunay = Delaunay(points)
    local_scale = compute_local_scale(delaunay, points)
    
    n_tri = len(delaunay.simplices)
    valid_mask = np.ones(n_tri, dtype=bool)
    
    for t in range(n_tri):
        v0, v1, v2 = delaunay.simplices[t]
        p0, p1, p2 = points[v0], points[v1], points[v2]
        
        e01 = np.linalg.norm(p1 - p0)
        e12 = np.linalg.norm(p2 - p1)
        e20 = np.linalg.norm(p0 - p2)
        edges = sorted([e01, e12, e20])
        max_edge = edges[2]
        min_edge = edges[0]
        
        local_thresh = threshold_factor * min(local_scale[v0], local_scale[v1], local_scale[v2])
        if max_edge > local_thresh:
            valid_mask[t] = False
            continue
        
        if max_edge > max_edge_m:
            valid_mask[t] = False
            continue
        
        if min_edge > 0 and max_edge / min_edge > max_aspect_ratio:
            valid_mask[t] = False
    
    return delaunay, valid_mask


# CLI for testing
if __name__ == "__main__":
    import argparse
    import gzip
    
    parser = argparse.ArgumentParser(description="Extract water boundary from geometry.bin")
    parser.add_argument("geometry_bin", help="Path to geometry.bin (gzipped)")
    parser.add_argument("-o", "--output", default="water_boundary.geojson",
                        help="Output GeoJSON file")
    parser.add_argument("-t", "--threshold", type=float, default=3.0,
                        help="Edge length threshold factor (default: 3.0)")
    args = parser.parse_args()
    
    print(f"Loading {args.geometry_bin}...")
    with gzip.open(args.geometry_bin, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.float32)
    
    n = len(data) // 2
    lon = data[0::2]
    lat = data[1::2]
    print(f"  {n:,} elements")
    
    print(f"Building Delaunay triangulation...")
    tri, valid = build_water_mask(lon, lat, threshold_factor=args.threshold)
    n_valid = valid.sum()
    n_invalid = len(valid) - n_valid
    print(f"  {n_valid:,} water triangles, {n_invalid:,} land-spanning")
    
    print(f"Extracting boundary...")
    boundary = extract_boundary(tri, valid, lon, lat)
    n_features = len(boundary.get("features", []))
    if n_features > 0:
        geom = boundary["features"][0]["geometry"]
        n_polygons = len(geom.get("coordinates", []))
        print(f"  {n_polygons} polygon(s)")
    
    print(f"Writing {args.output}...")
    size = export_geojson(boundary, args.output)
    print(f"  {size / 1024:.1f} KB")
