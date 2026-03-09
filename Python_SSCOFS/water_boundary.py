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
    use_utm: bool = False,
    utm_zone: int = 10,
) -> Tuple[Delaunay, np.ndarray]:
    """Build Delaunay triangulation and classify water vs land-spanning triangles.
    
    Parameters
    ----------
    lon, lat : ndarray
        Element center coordinates in degrees.
    threshold_factor : float
        A triangle is land-spanning if its max edge exceeds this factor times
        the local mesh scale (max of the three vertices' local scales).
    use_utm : bool
        If True, convert to UTM for more accurate distance calculations.
        Recommended for routing; optional for boundary extraction.
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
        # Use scaled lon/lat (approximate meters at mid-latitude)
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
        
        # Compute edge lengths
        e01 = np.linalg.norm(p1 - p0)
        e12 = np.linalg.norm(p2 - p1)
        e20 = np.linalg.norm(p0 - p2)
        max_edge = max(e01, e12, e20)
        
        # Local scale threshold: use min of the three vertices' scales.
        # If any vertex is in a dense part of the mesh, a long edge from
        # that vertex is anomalous (likely spanning a land gap).
        local_thresh = threshold_factor * min(local_scale[v0], local_scale[v1], local_scale[v2])
        
        if max_edge > local_thresh:
            valid_mask[t] = False
    
    # Store points on the delaunay object for later use
    delaunay.input_points = points
    
    return delaunay, valid_mask


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
) -> Tuple[Delaunay, np.ndarray]:
    """Build Delaunay triangulation from UTM coordinates directly.
    
    This is for use in sail_routing.py where coordinates are already in UTM.
    
    Parameters
    ----------
    x_utm, y_utm : ndarray
        UTM coordinates in meters.
    threshold_factor : float
        Triangle classification threshold.
        
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
        max_edge = max(e01, e12, e20)
        
        # Use min of local scales - if any vertex is in dense mesh, long edge is anomalous
        local_thresh = threshold_factor * min(local_scale[v0], local_scale[v1], local_scale[v2])
        
        if max_edge > local_thresh:
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
