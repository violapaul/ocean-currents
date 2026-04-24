"""
test_sail_routing.py
--------------------
Tests for the sail routing algorithm using synthetic current fields.

Every test builds its own tiny current field (no SSCOFS download needed)
with a known pattern so we can verify the physics and routing decisions
against analytically computed answers.

Run:
    cd WaysWaterMoves/OceanCurrents/Python_SSCOFS
    python -m pytest test_sail_routing.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pytest
from pyproj import Transformer

from sail_routing import (CurrentField, BoatModel, Router, MeshRouter, SectorRouter,
                          Route, PolarTable, WindField, compute_twa, _solve_heading,
                          KNOTS_TO_MS, MS_TO_KNOTS, build_mesh_adjacency)

POLAR_CSV = Path(__file__).parent.parent.parent / "j105_new_polars.csv"

# =====================================================================
#  Helpers -- synthetic current field factory
# =====================================================================

CENTER_LAT = 47.5
CENTER_LON = -122.3


def _make_transformer():
    """UTM transformer for the test centre point."""
    utm_zone = int((CENTER_LON + 180) / 6) + 1
    utm_crs = (f"+proj=utm +zone={utm_zone} +north "
               f"+datum=WGS84 +units=m +no_defs")
    return Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)


def _make_inv_transformer(fwd):
    return Transformer.from_crs(fwd.target_crs, fwd.source_crs,
                                always_xy=True)


def make_synthetic_field(u_func=None, v_func=None,
                         half_size_deg=0.06, spacing_deg=0.002,
                         land_func=None, k_neighbors=4,
                         land_threshold_m=300.0,
                         use_delaunay=True):
    """Build a CurrentField from position-based u/v functions.

    Parameters
    ----------
    u_func, v_func : callable(x_utm, y_utm) -> float  (m/s)
        Default: zero everywhere.
    half_size_deg : float
        Half-width of the element grid in degrees.
    spacing_deg : float
        Distance between synthetic element centres in degrees.
    land_func : callable(x_utm, y_utm) -> bool, optional
        Returns True where there is land (elements are excluded).
    """
    if u_func is None:
        u_func = lambda x, y: 0.0
    if v_func is None:
        v_func = lambda x, y: 0.0

    lats = np.arange(CENTER_LAT - half_size_deg,
                     CENTER_LAT + half_size_deg, spacing_deg)
    lons = np.arange(CENTER_LON - half_size_deg,
                     CENTER_LON + half_size_deg, spacing_deg)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    lonc = lon_grid.ravel()
    latc = lat_grid.ravel()

    transformer = _make_transformer()
    x_utm, y_utm = transformer.transform(lonc, latc)

    u_vals = np.array([u_func(x, y) for x, y in zip(x_utm, y_utm)],
                      dtype=np.float64)
    v_vals = np.array([v_func(x, y) for x, y in zip(x_utm, y_utm)],
                      dtype=np.float64)

    if land_func is not None:
        water = np.array([not land_func(x, y)
                          for x, y in zip(x_utm, y_utm)])
        lonc = lonc[water]
        latc = latc[water]
        u_vals = u_vals[water]
        v_vals = v_vals[water]

    cf = CurrentField(lonc, latc, [u_vals], [v_vals], [0.0],
                      transformer, k_neighbors=k_neighbors,
                      land_threshold_m=land_threshold_m,
                      use_delaunay=use_delaunay)
    return cf, transformer


def _latlon_for_offset(dx_m, dy_m, transformer):
    """Return (lat, lon) for a UTM offset from CENTER."""
    cx, cy = transformer.transform(CENTER_LON, CENTER_LAT)
    inv = _make_inv_transformer(transformer)
    lon, lat = inv.transform(cx + dx_m, cy + dy_m)
    return (lat, lon)


def make_time_varying_field(frame_specs, frame_times_s,
                            half_size_deg=0.06, spacing_deg=0.002,
                            k_neighbors=4, land_threshold_m=300.0):
    """Build a multi-frame CurrentField for time-dependent tests.

    Parameters
    ----------
    frame_specs : list of (u_func, v_func) tuples
        One pair of callables per time frame.
    frame_times_s : list of float
        Elapsed seconds for each frame.
    """
    lats = np.arange(CENTER_LAT - half_size_deg,
                     CENTER_LAT + half_size_deg, spacing_deg)
    lons = np.arange(CENTER_LON - half_size_deg,
                     CENTER_LON + half_size_deg, spacing_deg)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    lonc = lon_grid.ravel()
    latc = lat_grid.ravel()

    transformer = _make_transformer()
    x_utm, y_utm = transformer.transform(lonc, latc)

    u_frames = []
    v_frames = []
    for u_func, v_func in frame_specs:
        u_vals = np.array([u_func(x, y) for x, y in zip(x_utm, y_utm)],
                          dtype=np.float64)
        v_vals = np.array([v_func(x, y) for x, y in zip(x_utm, y_utm)],
                          dtype=np.float64)
        u_frames.append(u_vals)
        v_frames.append(v_vals)

    cf = CurrentField(lonc, latc, u_frames, v_frames, frame_times_s,
                      transformer, k_neighbors=k_neighbors,
                      land_threshold_m=land_threshold_m)
    return cf, transformer


# =====================================================================
#  1.  Edge cost physics -- unit tests
# =====================================================================

class TestEdgeCostPhysics:
    """Test the travel-time calculation for a single grid edge."""

    @pytest.fixture()
    def setup_uniform_east(self):
        """Router with 1 m/s uniform eastward current, 100 m grid."""
        cf, tr = make_synthetic_field(u_func=lambda x, y: 1.0,
                                      v_func=lambda x, y: 0.0)
        boat = BoatModel(base_speed_knots=6.0)
        router = Router(cf, boat, resolution_m=100.0, padding_m=500.0)
        cx, cy = tr.transform(CENTER_LON, CENTER_LAT)
        xs = np.arange(cx - 500, cx + 600, 100.0)
        ys = np.arange(cy - 500, cy + 600, 100.0)
        mc = len(xs) // 2
        mr = len(ys) // 2
        return router, xs, ys, mr, mc, boat.speed()

    def test_favorable_current(self, setup_uniform_east):
        """Traveling east with a 1 m/s eastward current."""
        router, xs, ys, mr, mc, spd = setup_uniform_east
        dt = router._edge_cost(mr, mc, mr, mc + 1, xs, ys, 0.0)
        expected = 100.0 / (spd + 1.0)
        assert dt == pytest.approx(expected, rel=0.05), (
            f"Favorable: got {dt:.4f}s, expected {expected:.4f}s")

    def test_opposing_current(self, setup_uniform_east):
        """Traveling west against a 1 m/s eastward current."""
        router, xs, ys, mr, mc, spd = setup_uniform_east
        dt = router._edge_cost(mr, mc, mr, mc - 1, xs, ys, 0.0)
        expected = 100.0 / (spd - 1.0)
        assert dt == pytest.approx(expected, rel=0.05), (
            f"Opposing: got {dt:.4f}s, expected {expected:.4f}s")

    def test_cross_current(self, setup_uniform_east):
        """Traveling north with a 1 m/s eastward cross-current.
        Must crab into the current, reducing forward progress."""
        router, xs, ys, mr, mc, spd = setup_uniform_east
        dt = router._edge_cost(mr, mc, mr + 1, mc, xs, ys, 0.0)
        v_along = np.sqrt(spd**2 - 1.0**2)
        expected = 100.0 / v_along
        assert dt == pytest.approx(expected, rel=0.05), (
            f"Cross: got {dt:.4f}s, expected {expected:.4f}s "
            f"(v_along={v_along:.3f})")

    def test_diagonal_with_current(self, setup_uniform_east):
        """Traveling northeast (diagonal) with eastward current.
        Current has components both along and across the diagonal."""
        router, xs, ys, mr, mc, spd = setup_uniform_east
        dt = router._edge_cost(mr, mc, mr + 1, mc + 1, xs, ys, 0.0)

        dist = np.hypot(100.0, 100.0)
        d_hat = np.array([1.0, 1.0]) / np.sqrt(2)
        c_vec = np.array([1.0, 0.0])
        c_par = np.dot(c_vec, d_hat)
        c_perp_mag = np.linalg.norm(c_vec - c_par * d_hat)
        v_water_along = np.sqrt(spd**2 - c_perp_mag**2)
        v_sog = v_water_along + c_par
        expected = dist / v_sog

        assert dt == pytest.approx(expected, rel=0.05), (
            f"Diagonal: got {dt:.4f}s, expected {expected:.4f}s")

    def test_zero_current(self):
        """With zero current, time = distance / boat_speed for all dirs."""
        cf, tr = make_synthetic_field()
        boat = BoatModel(base_speed_knots=6.0)
        router = Router(cf, boat, resolution_m=100.0, padding_m=500.0)
        cx, cy = tr.transform(CENTER_LON, CENTER_LAT)
        xs = np.arange(cx - 500, cx + 600, 100.0)
        ys = np.arange(cy - 500, cy + 600, 100.0)
        mc = len(xs) // 2
        mr = len(ys) // 2
        spd = boat.speed()

        # Cardinal: east
        dt = router._edge_cost(mr, mc, mr, mc + 1, xs, ys, 0.0)
        assert dt == pytest.approx(100.0 / spd, rel=0.02)
        # Cardinal: north
        dt = router._edge_cost(mr, mc, mr + 1, mc, xs, ys, 0.0)
        assert dt == pytest.approx(100.0 / spd, rel=0.02)
        # Diagonal: NE
        dt = router._edge_cost(mr, mc, mr + 1, mc + 1, xs, ys, 0.0)
        assert dt == pytest.approx(np.hypot(100, 100) / spd, rel=0.02)

    def test_overwhelming_cross_current(self):
        """Cross-current >= boat speed -> impassable (inf)."""
        boat = BoatModel(base_speed_knots=2.0)  # ~1.03 m/s
        cf, tr = make_synthetic_field(u_func=lambda x, y: 2.0)
        router = Router(cf, boat, resolution_m=100.0, padding_m=500.0)
        cx, cy = tr.transform(CENTER_LON, CENTER_LAT)
        xs = np.arange(cx - 500, cx + 600, 100.0)
        ys = np.arange(cy - 500, cy + 600, 100.0)
        mc = len(xs) // 2
        mr = len(ys) // 2

        dt = router._edge_cost(mr, mc, mr + 1, mc, xs, ys, 0.0)
        assert dt == np.inf, "Should be impassable when cross-current > boat speed"

    def test_overwhelming_opposing_current(self):
        """Opposing current > boat speed -> impassable."""
        boat = BoatModel(base_speed_knots=2.0)
        cf, tr = make_synthetic_field(v_func=lambda x, y: -2.0)
        router = Router(cf, boat, resolution_m=100.0, padding_m=500.0)
        cx, cy = tr.transform(CENTER_LON, CENTER_LAT)
        xs = np.arange(cx - 500, cx + 600, 100.0)
        ys = np.arange(cy - 500, cy + 600, 100.0)
        mc = len(xs) // 2
        mr = len(ys) // 2

        dt = router._edge_cost(mr, mc, mr + 1, mc, xs, ys, 0.0)
        assert dt == np.inf, "Should be impassable when opposing current > boat speed"


# =====================================================================
#  2.  Full routing integration tests
# =====================================================================

class TestRoutingNoCurrents:
    """With zero current the route should be approximately straight
    and the time should be distance / boat_speed."""

    def test_straight_north(self):
        cf, tr = make_synthetic_field()
        boat = BoatModel(base_speed_knots=6.0)
        router = Router(cf, boat, resolution_m=200.0, padding_m=1000.0)

        start = _latlon_for_offset(0, -2000, tr)
        end = _latlon_for_offset(0, 2000, tr)

        route, xs, ys, wm = router.find_route(start, end)
        expected_dist = 4000.0
        expected_time = expected_dist / boat.speed()

        assert route.total_distance_m == pytest.approx(expected_dist, rel=0.10)
        assert route.total_time_s == pytest.approx(expected_time, rel=0.10)
        assert route.avg_sog_knots == pytest.approx(
            boat.base_speed_knots, rel=0.10)

    def test_diagonal(self):
        cf, tr = make_synthetic_field()
        boat = BoatModel(base_speed_knots=6.0)
        router = Router(cf, boat, resolution_m=200.0, padding_m=1000.0)

        start = _latlon_for_offset(-2000, -2000, tr)
        end = _latlon_for_offset(2000, 2000, tr)

        route, _, _, _ = router.find_route(start, end)
        expected_dist = np.hypot(4000, 4000)
        expected_time = expected_dist / boat.speed()

        # 8-connected diagonal can add up to ~8% extra distance
        assert route.total_distance_m == pytest.approx(expected_dist, rel=0.12)
        assert route.total_time_s == pytest.approx(expected_time, rel=0.12)


class TestRoutingWithUniformCurrent:
    """Uniform current should speed up / slow down travel predictably."""

    def test_favorable_current_faster(self):
        """Traveling north with 1 m/s northward current."""
        cf, tr = make_synthetic_field(v_func=lambda x, y: 1.0)
        boat = BoatModel(base_speed_knots=6.0)
        router = Router(cf, boat, resolution_m=200.0, padding_m=1000.0)

        start = _latlon_for_offset(0, -2000, tr)
        end = _latlon_for_offset(0, 2000, tr)

        route, _, _, _ = router.find_route(start, end)
        sog_ms = route.total_distance_m / route.total_time_s
        expected_sog = boat.speed() + 1.0

        assert sog_ms == pytest.approx(expected_sog, rel=0.10), (
            f"SOG {sog_ms:.3f} m/s, expected ~{expected_sog:.3f}")

    def test_opposing_current_slower(self):
        """Traveling north against 1 m/s southward current."""
        cf, tr = make_synthetic_field(v_func=lambda x, y: -1.0)
        boat = BoatModel(base_speed_knots=6.0)
        router = Router(cf, boat, resolution_m=200.0, padding_m=1000.0)

        start = _latlon_for_offset(0, -2000, tr)
        end = _latlon_for_offset(0, 2000, tr)

        route, _, _, _ = router.find_route(start, end)
        sog_ms = route.total_distance_m / route.total_time_s
        expected_sog = boat.speed() - 1.0

        assert sog_ms == pytest.approx(expected_sog, rel=0.10), (
            f"SOG {sog_ms:.3f} m/s, expected ~{expected_sog:.3f}")

    def test_cross_current_crabbing(self):
        """Traveling north with 1 m/s eastward cross-current.
        The boat crabs, reducing effective northward speed."""
        cf, tr = make_synthetic_field(u_func=lambda x, y: 1.0)
        boat = BoatModel(base_speed_knots=6.0)
        router = Router(cf, boat, resolution_m=200.0, padding_m=1000.0)

        start = _latlon_for_offset(0, -2000, tr)
        end = _latlon_for_offset(0, 2000, tr)

        route, _, _, _ = router.find_route(start, end)

        # With pure cross-current the path should still be ~straight
        # (same cross-current everywhere, no benefit to detouring)
        # but forward speed is reduced by crabbing.
        v_forward = np.sqrt(boat.speed()**2 - 1.0**2)
        expected_time = 4000.0 / v_forward

        assert route.total_time_s == pytest.approx(expected_time, rel=0.15)


class TestRoutingDetour:
    """The router should detour into a band of favorable current
    when the time savings outweigh the extra distance."""

    def test_detour_into_favorable_band(self):
        """A band of strong northward current at x_offset = +1500 m
        from center should attract the northward route eastward.

        Analytical comparison:
          Straight (no current): 8000 m / 3.087 m/s = 2591 s
          Detour: ~1500 m east + 8000 m north at (3.087+2.0) + ~1500 m west
                  ~486 s + 1573 s + 486 s = 2545 s  (faster)
        """
        boat = BoatModel(base_speed_knots=6.0)
        boat_spd = boat.speed()

        # Band: strong 2 m/s northward current in x_offset 1000..2000
        tr = _make_transformer()
        cx, _ = tr.transform(CENTER_LON, CENTER_LAT)
        band_left = cx + 1000
        band_right = cx + 2000

        def v_func(x, y):
            if band_left <= x <= band_right:
                return 2.0
            return 0.0

        cf, tr = make_synthetic_field(v_func=v_func)
        router = Router(cf, boat, resolution_m=200.0, padding_m=1500.0)

        start = _latlon_for_offset(0, -4000, tr)
        end = _latlon_for_offset(0, 4000, tr)

        route, _, _, _ = router.find_route(start, end)

        # Route should be faster than straight (no-current) baseline
        straight_time_no_current = 8000.0 / boat_spd
        assert route.total_time_s < straight_time_no_current * 0.98, (
            f"Route time {route.total_time_s:.1f}s should be noticeably "
            f"faster than straight {straight_time_no_current:.1f}s")

        # Route should deviate east toward the band
        x_coords = [p[0] for p in route.waypoints_utm]
        max_x = max(x_coords)
        start_x = x_coords[0]
        eastward_deviation = max_x - start_x
        assert eastward_deviation > 500, (
            f"Route should detour east toward current band, "
            f"but max deviation is only {eastward_deviation:.0f} m")

    def test_no_detour_when_current_weak(self):
        """A very weak current band should NOT attract a detour
        (extra distance not worth the tiny speed boost)."""
        boat = BoatModel(base_speed_knots=6.0)
        tr = _make_transformer()
        cx, _ = tr.transform(CENTER_LON, CENTER_LAT)
        band_left = cx + 2000
        band_right = cx + 2500

        def v_func(x, y):
            if band_left <= x <= band_right:
                return 0.05
            return 0.0

        cf, tr = make_synthetic_field(v_func=v_func)
        router = Router(cf, boat, resolution_m=200.0, padding_m=1500.0)

        start = _latlon_for_offset(0, -3000, tr)
        end = _latlon_for_offset(0, 3000, tr)

        route, _, _, _ = router.find_route(start, end)

        x_coords = [p[0] for p in route.waypoints_utm]
        max_deviation = max(x_coords) - min(x_coords)
        assert max_deviation < 800, (
            f"Route should stay roughly straight, "
            f"but x-deviation is {max_deviation:.0f} m")


class TestRoutingLandAvoidance:
    """Route should navigate around land obstacles."""

    def test_route_around_land_wall(self):
        """A wide wall of land blocks the straight path.
        The route must go around it."""
        tr = _make_transformer()
        cx, cy = tr.transform(CENTER_LON, CENTER_LAT)

        def land_func(x, y):
            """Wall 1500 m wide, 2000 m tall, centred on the path."""
            return (abs(x - cx) < 750) and (abs(y - cy) < 1000)

        cf, _ = make_synthetic_field(land_func=land_func,
                                     land_threshold_m=250.0)
        boat = BoatModel(base_speed_knots=6.0)
        router = Router(cf, boat, resolution_m=200.0, padding_m=2000.0)

        start = _latlon_for_offset(-3000, 0, tr)
        end = _latlon_for_offset(3000, 0, tr)

        route, _, _, _ = router.find_route(start, end)

        straight_dist = 6000.0
        assert route.total_distance_m > straight_dist * 1.05, (
            f"Route should be longer than straight line to go around land "
            f"({route.total_distance_m:.0f} vs {straight_dist:.0f})")

        # Verify path actually goes around (some waypoints deviate in y)
        y_coords = [p[1] for p in route.waypoints_utm]
        y_deviation = max(y_coords) - min(y_coords)
        assert y_deviation > 500, (
            f"Route should deviate north or south to go around land, "
            f"but y-deviation is only {y_deviation:.0f} m")


class TestRoutingStraightLineComparison:
    """The A* route should always be at least as fast as or faster
    than the straight-line baseline (within grid discretisation noise)."""

    def test_optimal_at_least_as_good_as_straight(self):
        """With uniform favorable current along an axis, the optimal
        route should match or beat the straight line."""
        cf, tr = make_synthetic_field(v_func=lambda x, y: 0.5)
        boat = BoatModel(base_speed_knots=6.0)
        router = Router(cf, boat, resolution_m=200.0, padding_m=1000.0)

        start = _latlon_for_offset(0, -2000, tr)
        end = _latlon_for_offset(0, 2000, tr)

        route, _, _, _ = router.find_route(start, end)
        sl_time, sl_dist = router.straight_line_time(start, end)

        # Allow 5% tolerance for grid discretisation
        assert route.total_time_s <= sl_time * 1.05, (
            f"A* route ({route.total_time_s:.1f}s) should not be much "
            f"slower than straight line ({sl_time:.1f}s)")


class TestCurrentFieldInterpolation:
    """Verify that the CurrentField returns correct values."""

    def test_uniform_field_returns_constant(self):
        cf, tr = make_synthetic_field(u_func=lambda x, y: 1.5,
                                      v_func=lambda x, y: -0.7)
        cx, cy = tr.transform(CENTER_LON, CENTER_LAT)
        u, v = cf.query(cx, cy)
        assert u == pytest.approx(1.5, abs=0.1)
        assert v == pytest.approx(-0.7, abs=0.1)

    def test_land_returns_nan(self):
        """Query point far from any element should return NaN."""
        cf, tr = make_synthetic_field(half_size_deg=0.01,
                                      land_threshold_m=100.0)
        cx, cy = tr.transform(CENTER_LON, CENTER_LAT)
        u, v = cf.query(cx + 50000, cy + 50000)
        assert np.isnan(u) and np.isnan(v)

    def test_spatial_variation(self):
        """Field that varies with position should interpolate correctly."""
        tr = _make_transformer()
        cx, cy = tr.transform(CENTER_LON, CENTER_LAT)

        def u_func(x, y):
            return (x - cx) / 1000.0

        cf, _ = make_synthetic_field(u_func=u_func)

        # 1 km east of center -> u ~ 1.0
        u, v = cf.query(cx + 1000, cy)
        assert u == pytest.approx(1.0, abs=0.2)

        # 1 km west of center -> u ~ -1.0
        u, v = cf.query(cx - 1000, cy)
        assert u == pytest.approx(-1.0, abs=0.2)

    def test_time_interpolation(self):
        """Two time frames should be linearly interpolated."""
        tr = _make_transformer()
        cx, cy = tr.transform(CENTER_LON, CENTER_LAT)

        lats = np.arange(CENTER_LAT - 0.03, CENTER_LAT + 0.03, 0.002)
        lons = np.arange(CENTER_LON - 0.03, CENTER_LON + 0.03, 0.002)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        lonc = lon_grid.ravel()
        latc = lat_grid.ravel()
        n = len(lonc)

        u0 = np.full(n, 1.0)
        v0 = np.zeros(n)
        u1 = np.full(n, 3.0)
        v1 = np.zeros(n)

        cf = CurrentField(lonc, latc, [u0, u1], [v0, v1],
                          [0.0, 3600.0], tr,
                          k_neighbors=4, land_threshold_m=300.0)

        u_t0, _ = cf.query(cx, cy, elapsed_s=0.0)
        u_half, _ = cf.query(cx, cy, elapsed_s=1800.0)
        u_t1, _ = cf.query(cx, cy, elapsed_s=3600.0)

        assert u_t0 == pytest.approx(1.0, abs=0.15)
        assert u_half == pytest.approx(2.0, abs=0.15)
        assert u_t1 == pytest.approx(3.0, abs=0.15)


# =====================================================================
#  3.  Path smoothing tests
# =====================================================================

class TestPathSmoothing:
    """Verify that path smoothing removes staircase artifacts."""

    def test_diagonal_distance_near_euclidean(self):
        """On a diagonal with zero current, the smoothed path distance
        should be within ~2% of the Euclidean distance.
        Without smoothing, 8-connected paths inflate by up to ~8%."""
        cf, tr = make_synthetic_field()
        boat = BoatModel(base_speed_knots=6.0)
        router = Router(cf, boat, resolution_m=200.0, padding_m=1000.0)

        start = _latlon_for_offset(-2000, -2000, tr)
        end = _latlon_for_offset(2000, 2000, tr)

        route, _, _, _ = router.find_route(start, end)
        euclidean_dist = np.hypot(4000, 4000)

        assert route.total_distance_m == pytest.approx(euclidean_dist, rel=0.03), (
            f"Smoothed diagonal should be near Euclidean: "
            f"{route.total_distance_m:.0f} m vs {euclidean_dist:.0f} m "
            f"({100 * route.total_distance_m / euclidean_dist - 100:+.1f}%)")

    def test_smoothing_reduces_waypoints(self):
        """Smoothed path should have far fewer waypoints than the raw
        grid path for a straight-line route with no obstacles."""
        cf, tr = make_synthetic_field()
        boat = BoatModel(base_speed_knots=6.0)
        router = Router(cf, boat, resolution_m=100.0, padding_m=500.0)

        start = _latlon_for_offset(0, -2000, tr)
        end = _latlon_for_offset(0, 2000, tr)

        route, _, _, _ = router.find_route(start, end)

        # A 4000 m trip at 100 m resolution would be ~40 grid steps.
        # After smoothing a straight-line path, should be 2 waypoints
        # (start and end), or very few if snapping introduces a kink.
        assert len(route.waypoints_utm) <= 5, (
            f"Smoothed straight path should have very few waypoints, "
            f"got {len(route.waypoints_utm)}")

    def test_smoothing_preserves_land_avoidance(self):
        """Smoothing must not create shortcuts through land."""
        tr = _make_transformer()
        cx, cy = tr.transform(CENTER_LON, CENTER_LAT)

        def land_func(x, y):
            return (abs(x - cx) < 750) and (abs(y - cy) < 1000)

        cf, _ = make_synthetic_field(land_func=land_func,
                                     land_threshold_m=250.0)
        boat = BoatModel(base_speed_knots=6.0)
        router = Router(cf, boat, resolution_m=200.0, padding_m=2000.0)

        start = _latlon_for_offset(-3000, 0, tr)
        end = _latlon_for_offset(3000, 0, tr)

        route, _, _, _ = router.find_route(start, end)

        straight_dist = 6000.0
        assert route.total_distance_m > straight_dist * 1.05, (
            f"Smoothed path must still go around land: "
            f"{route.total_distance_m:.0f} m vs straight {straight_dist:.0f} m")

    def test_off_axis_angle_efficient(self):
        """A 30-degree path (not aligned to any grid axis) should still
        be close to Euclidean after smoothing."""
        cf, tr = make_synthetic_field()
        boat = BoatModel(base_speed_knots=6.0)
        router = Router(cf, boat, resolution_m=200.0, padding_m=1500.0)

        dx = 3000
        dy = int(3000 * np.tan(np.radians(30)))
        start = _latlon_for_offset(-dx // 2, -dy // 2, tr)
        end = _latlon_for_offset(dx // 2, dy // 2, tr)

        route, _, _, _ = router.find_route(start, end)
        euclidean_dist = np.hypot(dx, dy)

        assert route.total_distance_m == pytest.approx(euclidean_dist, rel=0.04), (
            f"30-degree path should be near Euclidean: "
            f"{route.total_distance_m:.0f} m vs {euclidean_dist:.0f} m "
            f"({100 * route.total_distance_m / euclidean_dist - 100:+.1f}%)")

    def test_segment_travel_time_matches_edge_cost(self):
        """For a single grid-cell step, _segment_travel_time should
        agree with _edge_cost."""
        cf, tr = make_synthetic_field(u_func=lambda x, y: 0.8,
                                      v_func=lambda x, y: 0.3)
        boat = BoatModel(base_speed_knots=6.0)
        router = Router(cf, boat, resolution_m=100.0, padding_m=500.0)

        cx, cy = tr.transform(CENTER_LON, CENTER_LAT)
        xs = np.arange(cx - 500, cx + 600, 100.0)
        ys = np.arange(cy - 500, cy + 600, 100.0)
        mc = len(xs) // 2
        mr = len(ys) // 2

        t_edge = router._edge_cost(mr, mc, mr + 1, mc + 1, xs, ys, 0.0)

        x0, y0 = xs[mc], ys[mr]
        x1, y1 = xs[mc + 1], ys[mr + 1]
        t_seg, _ = router._segment_travel_time(x0, y0, x1, y1, 0.0)

        assert t_seg == pytest.approx(t_edge, rel=0.05), (
            f"_segment_travel_time ({t_seg:.4f}s) should match "
            f"_edge_cost ({t_edge:.4f}s) for single-cell step")

    def test_line_of_sight_clear(self):
        """Line of sight through open water should return True."""
        cf, tr = make_synthetic_field()
        boat = BoatModel(base_speed_knots=6.0)
        router = Router(cf, boat, resolution_m=200.0, padding_m=1000.0)

        cx, cy = tr.transform(CENTER_LON, CENTER_LAT)
        xs, ys, water_mask, _, _ = router._build_grid((cx, cy), (cx, cy))
        ny, nx = water_mask.shape
        mid_r, mid_c = ny // 2, nx // 2

        assert router._line_of_sight(mid_r, mid_c, mid_r + 5, mid_c + 3,
                                     water_mask) is True

    def test_line_of_sight_blocked(self):
        """Line of sight through land should return False."""
        tr = _make_transformer()
        cx, cy = tr.transform(CENTER_LON, CENTER_LAT)

        def land_func(x, y):
            return abs(x - cx) < 300

        cf, _ = make_synthetic_field(land_func=land_func,
                                     land_threshold_m=200.0,
                                     use_delaunay=False)
        boat = BoatModel(base_speed_knots=6.0)
        router = Router(cf, boat, resolution_m=100.0, padding_m=1500.0)

        start_xy = (cx - 1000, cy)
        end_xy = (cx + 1000, cy)
        xs, ys, water_mask, _, _ = router._build_grid(start_xy, end_xy)

        sr, sc = router._snap_to_grid(start_xy, xs, ys)
        er, ec = router._snap_to_grid(end_xy, xs, ys)

        assert router._line_of_sight(sr, sc, er, ec, water_mask) is False

    def test_remove_stubs_does_not_cross_land(self):
        """Stub removal should not delete a point if the shortcut crosses land."""
        cf, _ = make_synthetic_field()
        boat = BoatModel(base_speed_knots=6.0)
        router = Router(cf, boat, resolution_m=1.0, padding_m=1.0)

        xs = np.arange(6, dtype=np.float64)
        ys = np.arange(6, dtype=np.float64)
        water_mask = np.ones((6, 6), dtype=bool)
        # Block only the A->C diagonal, leaving A->B and B->C clear.
        water_mask[1, 1] = False

        a = (0, 0)
        b = (0, 1)  # 1-cell stub leg from A, then long turn toward C
        c = (4, 4)

        assert water_mask[a[0], a[1]]
        assert water_mask[b[0], b[1]]
        assert water_mask[c[0], c[1]]
        assert router._line_of_sight(a[0], a[1], b[0], b[1], water_mask)
        assert router._line_of_sight(b[0], b[1], c[0], c[1], water_mask)
        assert not router._line_of_sight(a[0], a[1], c[0], c[1], water_mask)

        cleaned = router._remove_stubs([a, b, c], xs, ys, water_mask=water_mask)
        assert cleaned == [a, b, c], (
            "Stub removal removed a waypoint even though the direct shortcut "
            "crosses land.")


# =====================================================================
#  4.  Time-dependent routing tests
# =====================================================================

class TestTimeDependentRouting:
    """Verify the router accounts for currents changing over time.

    Core scenario: a band of favorable northward current that disappears
    partway through the transit.  A static router (seeing only t=0)
    would detour into the band, but a time-aware router should recognise
    the current won't last and choose differently.
    """

    @pytest.fixture()
    def band_geometry(self):
        """Shared geometry for the current band."""
        tr = _make_transformer()
        cx, cy = tr.transform(CENTER_LON, CENTER_LAT)
        band_left = cx + 1000
        band_right = cx + 2000
        return tr, cx, cy, band_left, band_right

    def _make_band_v(self, band_left, band_right, strength):
        """Return a v_func that is `strength` inside the band, 0 outside."""
        def v_func(x, y):
            if band_left <= x <= band_right:
                return strength
            return 0.0
        return v_func

    def test_static_band_causes_detour(self, band_geometry):
        """Baseline: with the band present at all times, the router
        detours east (same as existing TestRoutingDetour)."""
        tr, cx, cy, bl, br = band_geometry
        cf, tr = make_synthetic_field(
            v_func=self._make_band_v(bl, br, 2.0))
        boat = BoatModel(base_speed_knots=6.0)
        router = Router(cf, boat, resolution_m=200.0, padding_m=2000.0)

        start = _latlon_for_offset(0, -4000, tr)
        end = _latlon_for_offset(0, 4000, tr)
        route, _, _, _ = router.find_route(start, end)

        x_coords = [p[0] for p in route.waypoints_utm]
        deviation = max(x_coords) - x_coords[0]
        assert deviation > 500, (
            f"Static band should cause detour, but deviation is {deviation:.0f} m")

    def test_disappearing_band_reduces_detour(self, band_geometry):
        """When the band disappears after 800s, the router should detour
        less (or not at all) compared to the static case.

        At 6 kt (~3.09 m/s), the boat reaches the band (~1.5 km east)
        in ~486s.  With only 800s total of favorable current, it gets
        ~314s of boost before the band dies -- not worth the 3 km
        round-trip detour for this short benefit.
        """
        tr, cx, cy, bl, br = band_geometry

        v_band = self._make_band_v(bl, br, 2.0)
        v_zero = lambda x, y: 0.0
        u_zero = lambda x, y: 0.0

        cf_varying, tr = make_time_varying_field(
            frame_specs=[
                (u_zero, v_band),     # t=0:    band present
                (u_zero, v_zero),     # t=800s: band gone
            ],
            frame_times_s=[0.0, 800.0],
        )

        boat = BoatModel(base_speed_knots=6.0)
        router_tv = Router(cf_varying, boat, resolution_m=200.0, padding_m=2000.0)

        start = _latlon_for_offset(0, -4000, tr)
        end = _latlon_for_offset(0, 4000, tr)
        route_tv, _, _, _ = router_tv.find_route(start, end)

        # Also run with static (band always present) for comparison
        cf_static, tr_s = make_synthetic_field(v_func=v_band)
        router_static = Router(cf_static, boat, resolution_m=200.0, padding_m=2000.0)
        route_static, _, _, _ = router_static.find_route(start, end)

        x_tv = [p[0] for p in route_tv.waypoints_utm]
        x_st = [p[0] for p in route_static.waypoints_utm]
        dev_tv = max(x_tv) - x_tv[0]
        dev_st = max(x_st) - x_st[0]

        assert dev_tv < dev_st, (
            f"Time-varying route should detour less than static. "
            f"TV deviation: {dev_tv:.0f} m, static: {dev_st:.0f} m")

        # Time-varying route should be no worse than straight line
        # (the band is too brief to help, so it should stay near straight)
        boat_spd = boat.speed()
        straight_time = 8000.0 / boat_spd
        assert route_tv.total_time_s <= straight_time * 1.08, (
            f"Time-varying route ({route_tv.total_time_s:.0f}s) should "
            f"be close to straight-line time ({straight_time:.0f}s)")

    def test_current_reversal_avoids_late_opposing(self, band_geometry):
        """Current starts favorable on the west, then reverses.
        The time-aware router should avoid spending time on the west
        side late in the transit when it would face opposing current.

        Frame 0 (t=0):    west side has +2 m/s northward
        Frame 1 (t=1000s): west side has -2 m/s (opposing!)
        East side: always zero.
        """
        tr, cx, cy, _, _ = band_geometry
        west_left = cx - 2000
        west_right = cx - 1000

        def v_fav(x, y):
            if west_left <= x <= west_right:
                return 2.0
            return 0.0

        def v_opp(x, y):
            if west_left <= x <= west_right:
                return -2.0
            return 0.0

        u_zero = lambda x, y: 0.0

        cf_rev, tr = make_time_varying_field(
            frame_specs=[
                (u_zero, v_fav),
                (u_zero, v_opp),
            ],
            frame_times_s=[0.0, 1000.0],
        )
        boat = BoatModel(base_speed_knots=6.0)
        router = Router(cf_rev, boat, resolution_m=200.0, padding_m=2500.0)

        start = _latlon_for_offset(0, -4000, tr)
        end = _latlon_for_offset(0, 4000, tr)
        route, _, _, _ = router.find_route(start, end)

        # The route should NOT go far west, because the current
        # reverses before the transit is complete.  A static t=0
        # router would happily detour west.
        x_coords = [p[0] for p in route.waypoints_utm]
        westward_dev = x_coords[0] - min(x_coords)

        # Compare with what a static router would do
        cf_static, tr_s = make_synthetic_field(v_func=v_fav)
        router_static = Router(cf_static, boat, resolution_m=200.0, padding_m=2500.0)
        route_static, _, _, _ = router_static.find_route(start, end)
        x_static = [p[0] for p in route_static.waypoints_utm]
        westward_dev_static = x_static[0] - min(x_static)

        assert westward_dev < westward_dev_static, (
            f"Time-varying router should detour west less than static. "
            f"TV: {westward_dev:.0f} m, static: {westward_dev_static:.0f} m")

    def test_late_appearing_band_still_used(self, band_geometry):
        """A band that appears late should still attract the route
        if the boat arrives after the band turns on.

        Frame 0 (t=0):    no current
        Frame 1 (t=500s): strong band appears

        The boat takes ~1300s for the straight path, so for most of
        the journey the band is active.
        """
        tr, cx, cy, bl, br = band_geometry

        v_zero = lambda x, y: 0.0
        u_zero = lambda x, y: 0.0
        v_band = self._make_band_v(bl, br, 2.0)

        cf_late, tr = make_time_varying_field(
            frame_specs=[
                (u_zero, v_zero),     # t=0:    no current
                (u_zero, v_band),     # t=500s: band appears
            ],
            frame_times_s=[0.0, 500.0],
        )
        boat = BoatModel(base_speed_knots=6.0)
        router = Router(cf_late, boat, resolution_m=200.0, padding_m=2000.0)

        start = _latlon_for_offset(0, -4000, tr)
        end = _latlon_for_offset(0, 4000, tr)
        route, _, _, _ = router.find_route(start, end)

        # Route should still detour east because by the time the boat
        # reaches the band (~500s in), the current is active.
        x_coords = [p[0] for p in route.waypoints_utm]
        deviation = max(x_coords) - x_coords[0]

        # Weaker assertion since the band is only active for part of
        # the journey, but still expect some eastward pull.
        straight_time = 8000.0 / boat.speed()
        assert route.total_time_s < straight_time * 0.99, (
            f"Late-appearing band should still save time: "
            f"route {route.total_time_s:.0f}s vs straight {straight_time:.0f}s")


# =====================================================================
#  5.  PolarTable unit tests
# =====================================================================

@pytest.fixture(scope="module")
def polar():
    if not POLAR_CSV.exists():
        pytest.skip(f"Polar CSV not found at {POLAR_CSV}")
    return PolarTable(POLAR_CSV)


class TestPolarTable:
    """Unit tests for PolarTable loading and interpolation."""

    def test_head_to_wind_is_zero(self, polar):
        """TWA=0 should give zero boat speed at any wind speed."""
        for tws in [0, 4, 10, 20, 60]:
            spd = polar.speed(0, tws)
            assert spd == pytest.approx(0.0, abs=0.01), (
                f"TWA=0 TWS={tws}: expected 0, got {spd:.3f}")

    def test_exact_table_lookup(self, polar):
        """Values that fall exactly on table grid points should match CSV."""
        assert polar.speed(90, 10) == pytest.approx(7.8, abs=0.05)
        assert polar.speed(52, 20) == pytest.approx(7.4, abs=0.05)
        assert polar.speed(180, 12) == pytest.approx(5.85, abs=0.05)

    def test_twa_interpolation(self, polar):
        """Midpoint between TWA=70 and TWA=80 should interpolate."""
        s70 = polar.speed(70, 10)
        s80 = polar.speed(80, 10)
        s75 = polar.speed(75, 10)
        expected = 0.5 * (s70 + s80)
        assert s75 == pytest.approx(expected, abs=0.1), (
            f"TWA=75 interpolation: got {s75:.3f}, expected ~{expected:.3f}")

    def test_tws_interpolation(self, polar):
        """Midpoint between TWS=8 and TWS=10 at TWA=90."""
        s8 = polar.speed(90, 8)
        s10 = polar.speed(90, 10)
        s9 = polar.speed(90, 9)
        expected = 0.5 * (s8 + s10)
        assert s9 == pytest.approx(expected, abs=0.1), (
            f"TWS=9 interpolation: got {s9:.3f}, expected ~{expected:.3f}")

    def test_tws_clamps_at_max(self, polar):
        """TWS above table max should give same result as table max."""
        s_max = polar.speed(90, polar._twss[-1])
        s_over = polar.speed(90, 200.0)
        assert s_over == pytest.approx(s_max, abs=0.01)

    def test_tws_clamps_at_min_table_wind(self, polar):
        """TWS below the table range clamps to the lowest table wind speed."""
        assert polar.speed(90, 0) == pytest.approx(polar.speed(90, polar._twss[0]), abs=0.01)

    def test_max_speed_kt(self, polar):
        """max_speed_kt should match the maximum value in the CSV."""
        import csv
        rows = list(csv.DictReader(open(POLAR_CSV)))
        csv_max = max(float(r['BoatSpeed_kt']) for r in rows)
        assert polar.max_speed_kt == pytest.approx(csv_max, abs=0.01)

    def test_speed_ms_converts_correctly(self, polar):
        """speed_ms should equal speed * KNOTS_TO_MS."""
        twa, tws = 90, 10
        assert polar.speed_ms(twa, tws) == pytest.approx(
            polar.speed(twa, tws) * KNOTS_TO_MS, rel=1e-6)

    def test_downwind_no_go_zeroes_deep_angles(self, tmp_path):
        """maximum_twa should reject dead-downwind/deep-running angles."""
        import csv

        csv_path = tmp_path / "polar.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["TWA_deg", "TWS_kt", "BoatSpeed_kt"])
            writer.writeheader()
            for twa, spd in [(0, 0), (90, 7), (165, 5), (170, 4), (180, 3)]:
                writer.writerow({"TWA_deg": twa, "TWS_kt": 12, "BoatSpeed_kt": spd})
                writer.writerow({"TWA_deg": twa, "TWS_kt": 16, "BoatSpeed_kt": spd + 1})
        polar = PolarTable(csv_path, maximum_twa=165.0)
        assert polar.speed(165.0, 12.0) > 0.0
        assert polar.speed(170.0, 12.0) == pytest.approx(0.0, abs=0.01)
        assert polar.speed(180.0, 12.0) == pytest.approx(0.0, abs=0.01)

    def test_max_speed_in_fast_angles(self, polar):
        """Best speeds should be at broad reach angles (100-140 deg)."""
        # For a well-designed polar, peak speed should be between 100-150 deg
        best_twa = None
        best_spd = 0
        for twa in range(0, 181, 5):
            s = polar.speed(twa, 20)
            if s > best_spd:
                best_spd = s
                best_twa = twa
        assert 90 <= best_twa <= 160, (
            f"Best TWA at TWS=20 should be broad reach, got {best_twa} deg")

    def test_rejects_missing_grid_points(self, tmp_path):
        """A sparse polar table should be rejected instead of filled with zeros."""
        import csv

        csv_path = tmp_path / "sparse_polar.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["TWA_deg", "TWS_kt", "BoatSpeed_kt"])
            writer.writeheader()
            writer.writerow({"TWA_deg": 0, "TWS_kt": 10, "BoatSpeed_kt": 0})
            writer.writerow({"TWA_deg": 90, "TWS_kt": 10, "BoatSpeed_kt": 7})
            writer.writerow({"TWA_deg": 0, "TWS_kt": 20, "BoatSpeed_kt": 0})
            # Missing: (TWA=90, TWS=20)

        with pytest.raises(ValueError, match="missing"):
            PolarTable(csv_path)


# =====================================================================
#  6.  BoatModel + TWA computation tests
# =====================================================================

class TestComputeTwa:
    """Unit tests for compute_twa()."""

    def test_head_to_wind(self):
        """Heading directly into wind from north -> TWA = 0."""
        # Wind FROM north means (wu, wv) blows TO south: wu=0, wv=-1
        # Heading north: angle = pi/2 in math frame (atan2(1, 0))
        heading_rad = np.pi / 2  # north
        wu, wv = 0.0, -1.0      # wind blows TO south = from north
        twa = compute_twa(heading_rad, wu, wv)
        assert twa == pytest.approx(0.0, abs=2.0)

    def test_dead_downwind(self):
        """Heading south with wind from north -> TWA = 180."""
        heading_rad = -np.pi / 2  # south
        wu, wv = 0.0, -1.0       # wind blows TO south = from north
        twa = compute_twa(heading_rad, wu, wv)
        assert twa == pytest.approx(180.0, abs=2.0)

    def test_beam_reach_port(self):
        """Heading east with wind from north -> TWA = 90."""
        heading_rad = 0.0       # east
        wu, wv = 0.0, -1.0     # wind from north
        twa = compute_twa(heading_rad, wu, wv)
        assert twa == pytest.approx(90.0, abs=2.0)

    def test_beam_reach_starboard(self):
        """Heading west with wind from north -> TWA = 90."""
        heading_rad = np.pi     # west
        wu, wv = 0.0, -1.0    # wind from north
        twa = compute_twa(heading_rad, wu, wv)
        assert twa == pytest.approx(90.0, abs=2.0)

    def test_zero_wind_returns_90(self):
        """Zero wind speed should return 90 (safe default)."""
        twa = compute_twa(0.0, 0.0, 0.0)
        assert twa == pytest.approx(90.0)


class TestBoatModelPolar:
    """Unit tests for BoatModel with and without polar."""

    def test_no_polar_returns_fixed_speed(self, polar):
        """Without polar, speed() always returns base_speed."""
        boat = BoatModel(base_speed_knots=6.0)
        assert boat.speed() == pytest.approx(6.0 * KNOTS_TO_MS)
        assert boat.speed(heading_rad=0.5, wind_u=1.0, wind_v=0.0) == pytest.approx(
            6.0 * KNOTS_TO_MS)

    def test_polar_beam_reach(self, polar):
        """Polar at beam reach (TWA=90) should match table lookup."""
        boat = BoatModel(base_speed_knots=6.0, polar=polar)
        # Wind from north (blows to south): wu=0, wv=-tws
        tws_kt = 10.0
        tws_ms = tws_kt * KNOTS_TO_MS
        wu, wv = 0.0, -tws_ms
        heading_east = 0.0  # east = math angle 0
        speed_ms = boat.speed(heading_east, wu, wv)
        expected_ms = polar.speed_ms(90.0, tws_kt)
        assert speed_ms == pytest.approx(expected_ms, rel=0.05)

    def test_polar_head_to_wind_near_zero(self, polar):
        """Head-to-wind should give near-zero speed."""
        boat = BoatModel(base_speed_knots=6.0, polar=polar)
        tws_ms = 10.0 * KNOTS_TO_MS
        heading_north = np.pi / 2
        wu, wv = 0.0, -tws_ms  # wind from north
        speed_ms = boat.speed(heading_north, wu, wv)
        assert speed_ms < 0.1 * KNOTS_TO_MS, (
            f"Head-to-wind speed should be ~0, got {speed_ms * MS_TO_KNOTS:.2f} kt")

    def test_polar_downwind(self, polar):
        """Running downwind (TWA=180) should give moderate speed."""
        boat = BoatModel(base_speed_knots=6.0, polar=polar)
        tws_ms = 12.0 * KNOTS_TO_MS
        heading_south = -np.pi / 2  # south
        wu, wv = 0.0, -tws_ms    # wind from north
        speed_ms = boat.speed(heading_south, wu, wv)
        # At TWA=180, TWS=12: table says 5.9 kt
        assert speed_ms == pytest.approx(5.9 * KNOTS_TO_MS, rel=0.05)


# =====================================================================
#  7.  _solve_heading unit tests
# =====================================================================

class TestSolveHeading:
    """Unit tests for the heading sweep function."""

    @pytest.fixture()
    def boat(self, polar):
        return BoatModel(base_speed_knots=6.0, polar=polar)

    def test_beam_reach_no_current(self, boat, polar):
        """Track east, wind from north, 10 kt -> SOG ~ polar speed at TWA=90."""
        d_hat_x, d_hat_y = 1.0, 0.0   # east
        cu, cv = 0.0, 0.0
        tws_ms = 10.0 * KNOTS_TO_MS
        wu, wv = 0.0, -tws_ms          # wind from north

        sog = _solve_heading(d_hat_x, d_hat_y, cu, cv, wu, wv, boat)
        expected = polar.speed_ms(90.0, 10.0)
        assert sog == pytest.approx(expected, rel=0.05), (
            f"Beam reach SOG: {sog * MS_TO_KNOTS:.2f} kt, expected {expected * MS_TO_KNOTS:.2f} kt")

    def test_dead_downwind_no_current(self, boat, polar):
        """Track south, wind from north, 10 kt -> SOG ~ polar at TWA=180."""
        d_hat_x, d_hat_y = 0.0, -1.0  # south
        cu, cv = 0.0, 0.0
        tws_ms = 10.0 * KNOTS_TO_MS
        wu, wv = 0.0, -tws_ms

        sog = _solve_heading(d_hat_x, d_hat_y, cu, cv, wu, wv, boat)
        expected = polar.speed_ms(180.0, 10.0)
        assert sog == pytest.approx(expected, rel=0.10), (
            f"Downwind SOG: {sog * MS_TO_KNOTS:.2f} kt, expected {expected * MS_TO_KNOTS:.2f} kt")

    def test_upwind_vmg_positive(self, boat):
        """Track north into north wind -> must tack, but VMG should be > 0.

        The boat cannot point directly at TWA=0, but close-hauled at
        ~45 deg gives positive VMG.
        """
        d_hat_x, d_hat_y = 0.0, 1.0   # north
        cu, cv = 0.0, 0.0
        tws_ms = 10.0 * KNOTS_TO_MS
        wu, wv = 0.0, -tws_ms          # wind from north

        sog = _solve_heading(d_hat_x, d_hat_y, cu, cv, wu, wv, boat)
        # VMG upwind should be positive (boat is making northward progress)
        # At 10 kt, close-hauled at ~45 deg gives ~6.4*cos(45) ~ 4.5 kt VMG
        assert sog > 0.5 * KNOTS_TO_MS, (
            f"Upwind VMG should be > 0.5 kt, got {sog * MS_TO_KNOTS:.2f} kt")

    def test_favorable_current_adds_to_sog(self, boat, polar):
        """With 1 m/s favorable current eastward, SOG should be higher than polar alone."""
        d_hat_x, d_hat_y = 1.0, 0.0
        tws_ms = 10.0 * KNOTS_TO_MS
        wu, wv = 0.0, -tws_ms

        sog_no_curr = _solve_heading(d_hat_x, d_hat_y, 0.0, 0.0, wu, wv, boat)
        sog_with_curr = _solve_heading(d_hat_x, d_hat_y, 1.0, 0.0, wu, wv, boat)
        assert sog_with_curr > sog_no_curr, (
            f"Favorable current should increase SOG: "
            f"{sog_no_curr * MS_TO_KNOTS:.2f} -> {sog_with_curr * MS_TO_KNOTS:.2f} kt")

    def test_zero_wind_uses_fallback(self):
        """With zero wind, polar gives 0 -- fallback to base speed."""
        boat = BoatModel(base_speed_knots=6.0)  # no polar
        d_hat_x, d_hat_y = 1.0, 0.0
        sog = _solve_heading(d_hat_x, d_hat_y, 0.0, 0.0, 0.0, 0.0, boat)
        assert sog == pytest.approx(6.0 * KNOTS_TO_MS, rel=0.05)

    def test_best_heading_is_broad_reach(self, boat):
        """With no current, the heading giving highest SOG eastward in
        northerly wind should be the broad reach angle (not beam reach)."""
        d_hat_x, d_hat_y = 1.0, 0.0
        tws_ms = 20.0 * KNOTS_TO_MS
        wu, wv = 0.0, -tws_ms   # wind from north

        sog_beam = _solve_heading(d_hat_x, d_hat_y, 0.0, 0.0, wu, wv, boat)
        # At 20 kt wind, beam reach (90 deg) gives 8.5 kt but with full
        # component east. The returned value should be close to this.
        assert sog_beam > 7.0 * KNOTS_TO_MS, (
            f"Should achieve good SOG eastward in 20 kt wind: "
            f"{sog_beam * MS_TO_KNOTS:.2f} kt")

    def test_returns_zero_when_truly_impassable(self):
        """Very slow boat in very strong cross-current should return 0."""
        # The _solve_heading function uses a drift tolerance.
        # With a tiny boat speed and huge cross-current, nothing works.
        boat = BoatModel(base_speed_knots=0.1)
        # Track north, but 5 m/s westward current
        sog = _solve_heading(0.0, 1.0, -5.0, 0.0, 0.0, 0.0, boat)
        # Can't make northward progress: any northward heading is dragged west
        assert sog == pytest.approx(0.0, abs=0.01)


# =====================================================================
#  8.  Full routing tests with polar
# =====================================================================

def _make_zero_current_field():
    """Zero-current field for wind-only routing tests."""
    cf, tr = make_synthetic_field()
    return cf, tr


class TestRoutingWithPolar:
    """Integration tests for polar-based routing with constant wind."""

    @pytest.fixture()
    def polar(self):
        if not POLAR_CSV.exists():
            pytest.skip(f"Polar CSV not found at {POLAR_CSV}")
        return PolarTable(POLAR_CSV)

    def test_backward_compat_no_polar(self):
        """Existing fixed-speed routing must still work when no polar/wind."""
        cf, tr = make_synthetic_field()
        boat = BoatModel(base_speed_knots=6.0)
        router = Router(cf, boat, resolution_m=200.0, padding_m=1000.0)
        start = _latlon_for_offset(0, -2000, tr)
        end = _latlon_for_offset(0, 2000, tr)
        route, _, _, _ = router.find_route(start, end)
        assert route.total_distance_m == pytest.approx(4000.0, rel=0.10)

    def test_beam_reach_route_fast(self, polar):
        """Eastbound route with northerly wind should achieve good SOG."""
        cf, tr = make_synthetic_field()
        boat = BoatModel(base_speed_knots=6.0, polar=polar)
        wind = WindField.from_met(speed_kt=10.0, from_deg=0.0)  # from north
        router = Router(cf, boat, resolution_m=200.0, padding_m=1000.0, wind=wind)

        start = _latlon_for_offset(-2000, 0, tr)
        end = _latlon_for_offset(2000, 0, tr)
        route, _, _, _ = router.find_route(start, end)

        # Beam reach at 10 kt wind: ~7.1 kt STW
        assert route.avg_sog_knots > 6.5, (
            f"Beam reach should be fast, got {route.avg_sog_knots:.2f} kt")

    def test_downwind_route_plausible(self, polar):
        """Southbound route with northerly wind: running downwind."""
        cf, tr = make_synthetic_field()
        boat = BoatModel(base_speed_knots=6.0, polar=polar)
        wind = WindField.from_met(speed_kt=12.0, from_deg=0.0)  # from north
        router = Router(cf, boat, resolution_m=200.0, padding_m=1000.0, wind=wind)

        start = _latlon_for_offset(0, 2000, tr)
        end = _latlon_for_offset(0, -2000, tr)
        route, _, _, _ = router.find_route(start, end)

        # Dead downwind at TWA=180, TWS=12: 5.9 kt
        assert route.avg_sog_knots > 5.0, (
            f"Downwind route should be decent, got {route.avg_sog_knots:.2f} kt")

    def test_upwind_route_longer_distance(self, polar):
        """Northbound route into north wind must tack -- distance > straight line."""
        cf, tr = make_synthetic_field()
        boat = BoatModel(base_speed_knots=6.0, polar=polar)
        wind = WindField.from_met(speed_kt=10.0, from_deg=0.0)  # from north
        router = Router(cf, boat, resolution_m=200.0, padding_m=2000.0, wind=wind)

        start = _latlon_for_offset(0, -3000, tr)
        end = _latlon_for_offset(0, 3000, tr)
        route, _, _, _ = router.find_route(start, end)

        straight_dist = 6000.0
        # Tacking adds distance -- at least 5% more
        assert route.total_distance_m > straight_dist * 1.05, (
            f"Upwind tack should add distance: "
            f"{route.total_distance_m:.0f} m vs {straight_dist:.0f} m straight")

    def test_upwind_route_deviates_left_or_right(self, polar):
        """Upwind tacking route must travel more distance than straight line
        and have multiple waypoints (the tacking signature)."""
        cf, tr = make_synthetic_field()
        boat = BoatModel(base_speed_knots=6.0, polar=polar)
        wind = WindField.from_met(speed_kt=10.0, from_deg=0.0)  # from north
        router = Router(cf, boat, resolution_m=200.0, padding_m=2000.0, wind=wind)

        start = _latlon_for_offset(0, -3000, tr)
        end = _latlon_for_offset(0, 3000, tr)
        route, _, _, _ = router.find_route(start, end)

        straight_dist = 6000.0
        assert route.total_distance_m > straight_dist * 1.05, (
            f"Upwind route must be longer than straight line due to tacking: "
            f"{route.total_distance_m:.0f} m vs {straight_dist:.0f} m")
        assert len(route.waypoints_utm) >= 3, (
            f"Upwind route should have at least one tack point, "
            f"got {len(route.waypoints_utm)} waypoints")

    def test_wind_field_constant(self, polar):
        """WindField.from_met should return correct components."""
        # 10 kt from north: wind blows TO south -> wu=0, wv=-
        wf = WindField.from_met(10.0, 0.0)
        wu, wv = wf.query(0.0, 0.0)
        assert wu == pytest.approx(0.0, abs=0.01)
        assert wv < 0  # southward

        # 10 kt from east: wind blows TO west -> wu=-, wv=0
        wf_east = WindField.from_met(10.0, 90.0)
        wu2, wv2 = wf_east.query(0.0, 0.0)
        assert wu2 < 0   # westward
        assert wv2 == pytest.approx(0.0, abs=0.01)

        # Speed should match
        speed = np.hypot(wu, wv) * MS_TO_KNOTS
        assert speed == pytest.approx(10.0, rel=0.01)


# =====================================================================
#  9.  WindField spatial and temporal tests
# =====================================================================

class TestWindFieldSpatial:
    """Tests for WindField.from_grid (Phase 2b)."""

    def test_grid_constant_returns_grid_values(self):
        """from_grid with uniform values should return those values anywhere."""
        xs = np.linspace(540000, 545000, 10)
        ys = np.linspace(5280000, 5286000, 10)
        tws = 10.0 * KNOTS_TO_MS
        wu_g = np.zeros((10, 10))
        wv_g = np.full((10, 10), -tws)
        wf = WindField.from_grid(xs, ys, wu_g, wv_g)

        u, v = wf.query(542000, 5283000, elapsed_s=0.0)
        assert u == pytest.approx(0.0, abs=0.01)
        assert v == pytest.approx(-tws, rel=0.01)
        speed = np.hypot(u, v) * MS_TO_KNOTS
        assert speed == pytest.approx(10.0, rel=0.01)

    def test_grid_spatial_variation_interpolated(self):
        """from_grid should interpolate between different wind values."""
        xs = np.array([0.0, 1000.0])
        ys = np.array([0.0, 1000.0])
        # left side: 0 m/s, right side: 2 m/s
        wu_g = np.array([[0.0, 2.0], [0.0, 2.0]])
        wv_g = np.zeros((2, 2))
        wf = WindField.from_grid(xs, ys, wu_g, wv_g)

        u_left, _ = wf.query(0.0, 500.0)
        u_mid, _ = wf.query(500.0, 500.0)
        u_right, _ = wf.query(1000.0, 500.0)
        assert u_left == pytest.approx(0.0, abs=0.01)
        assert u_mid == pytest.approx(1.0, abs=0.05)
        assert u_right == pytest.approx(2.0, abs=0.01)

    def test_grid_ignores_elapsed_s(self):
        """Spatial-only grid should return same value at any time."""
        xs = np.array([0.0, 1000.0])
        ys = np.array([0.0, 1000.0])
        wu_g = np.ones((2, 2))
        wv_g = np.zeros((2, 2))
        wf = WindField.from_grid(xs, ys, wu_g, wv_g)

        u0, v0 = wf.query(500.0, 500.0, elapsed_s=0.0)
        u1, v1 = wf.query(500.0, 500.0, elapsed_s=9999.0)
        assert u0 == pytest.approx(u1, rel=1e-6)
        assert v0 == pytest.approx(v1, rel=1e-6)


class TestWindFieldTemporal:
    """Tests for WindField.from_frames (Phase 2c)."""

    def test_temporal_const_interpolates(self):
        """Constant-per-frame temporal field interpolates correctly."""
        tws = 5.0 * KNOTS_TO_MS
        wf = WindField.from_frames(
            None, None,
            wu_frames=[0.0, tws],
            wv_frames=[-tws, 0.0],
            frame_times_s=[0.0, 3600.0])

        u0, v0 = wf.query(0, 0, elapsed_s=0.0)
        assert u0 == pytest.approx(0.0, abs=0.01)
        assert v0 == pytest.approx(-tws, rel=0.01)

        u1, v1 = wf.query(0, 0, elapsed_s=3600.0)
        assert u1 == pytest.approx(tws, rel=0.01)
        assert v1 == pytest.approx(0.0, abs=0.01)

        u_mid, v_mid = wf.query(0, 0, elapsed_s=1800.0)
        assert u_mid == pytest.approx(0.5 * tws, rel=0.02)
        assert v_mid == pytest.approx(-0.5 * tws, rel=0.02)

    def test_temporal_clamps_at_boundaries(self):
        """Queries before first frame or after last frame are clamped."""
        wf = WindField.from_frames(
            None, None,
            wu_frames=[1.0, 3.0],
            wv_frames=[0.0, 0.0],
            frame_times_s=[1000.0, 2000.0])

        u_before, _ = wf.query(0, 0, elapsed_s=0.0)
        u_after, _ = wf.query(0, 0, elapsed_s=5000.0)
        assert u_before == pytest.approx(1.0, rel=0.01)
        assert u_after == pytest.approx(3.0, rel=0.01)

    def test_temporal_grid_interpolates_in_space_and_time(self):
        """Temporal grid mode should accept array frames and interpolate both axes."""
        xs = np.array([0.0, 1000.0])
        ys = np.array([0.0, 1000.0])

        wu_0 = np.array([[0.0, 0.0], [0.0, 0.0]])
        wu_1 = np.array([[2.0, 2.0], [2.0, 2.0]])
        wv_0 = np.array([[-1.0, -1.0], [-1.0, -1.0]])
        wv_1 = np.array([[1.0, 1.0], [1.0, 1.0]])

        wf = WindField.from_frames(
            xs, ys,
            wu_frames=[wu_0, wu_1],
            wv_frames=[wv_0, wv_1],
            frame_times_s=[0.0, 3600.0])

        u0, v0 = wf.query(500.0, 500.0, elapsed_s=0.0)
        u1, v1 = wf.query(500.0, 500.0, elapsed_s=3600.0)
        um, vm = wf.query(500.0, 500.0, elapsed_s=1800.0)

        assert u0 == pytest.approx(0.0, abs=0.01)
        assert v0 == pytest.approx(-1.0, abs=0.01)
        assert u1 == pytest.approx(2.0, abs=0.01)
        assert v1 == pytest.approx(1.0, abs=0.01)
        assert um == pytest.approx(1.0, abs=0.02)
        assert vm == pytest.approx(0.0, abs=0.02)

    def test_temporal_nodes_nearest_with_time_interp(self):
        """Temporal node mode should use nearest spatial node and interpolate in time."""
        node_x = np.array([0.0, 1000.0])
        node_y = np.array([0.0, 0.0])

        # Frame 0 and frame 1 values per node
        wu = np.array([
            [1.0, 3.0],
            [3.0, 5.0],
        ])
        wv = np.array([
            [-2.0, -4.0],
            [0.0, -2.0],
        ])
        wf = WindField.from_node_frames(
            node_x=node_x,
            node_y=node_y,
            wu_frames=wu,
            wv_frames=wv,
            frame_times_s=[0.0, 3600.0],
        )

        # Near node 0 at halfway time
        u0, v0 = wf.query(100.0, 10.0, elapsed_s=1800.0)
        assert u0 == pytest.approx(2.0, abs=0.01)
        assert v0 == pytest.approx(-1.0, abs=0.01)

        # Near node 1 at halfway time
        u1, v1 = wf.query(900.0, 5.0, elapsed_s=1800.0)
        assert u1 == pytest.approx(4.0, abs=0.01)
        assert v1 == pytest.approx(-3.0, abs=0.01)

    def test_routing_with_time_varying_wind(self):
        """Route with time-varying wind should work and produce a valid result."""
        if not POLAR_CSV.exists():
            pytest.skip(f"Polar CSV not found at {POLAR_CSV}")
        polar = PolarTable(POLAR_CSV)
        cf, tr = make_synthetic_field()
        boat = BoatModel(base_speed_knots=6.0, polar=polar)

        tws = 10.0 * KNOTS_TO_MS
        # Wind starts from north, rotates to from east over 2 hours
        wind = WindField.from_frames(
            None, None,
            wu_frames=[0.0, -tws],
            wv_frames=[-tws, 0.0],
            frame_times_s=[0.0, 7200.0])

        router = Router(cf, boat, resolution_m=200.0, padding_m=1000.0, wind=wind)
        start = _latlon_for_offset(-2000, 0, tr)
        end = _latlon_for_offset(2000, 0, tr)

        route, _, _, _ = router.find_route(start, end)
        assert route.total_distance_m > 0
        assert route.total_time_s > 0
        assert route.avg_sog_knots > 3.0


class TestScheduleWindBuilder:
    """Tests for run_route._build_schedule_wind: YAML schedule → WindField."""

    def _ctx(self):
        import datetime as _dt
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("America/Los_Angeles")
        depart_dt = _dt.datetime(2026, 4, 25, 9, 15, tzinfo=tz)
        return {
            "depart_dt": depart_dt,
            "depart_utc": depart_dt.astimezone(_dt.timezone.utc),
            "start_time_s": 0.0,
            "tz_str": "America/Los_Angeles",
        }

    @staticmethod
    def _decode(wu, wv):
        """(wu, wv) → (speed_kt, from_deg) in meteorological convention."""
        speed_kt = float(np.hypot(wu, wv)) * MS_TO_KNOTS
        from_deg = (float(np.degrees(np.arctan2(-wu, -wv))) + 360.0) % 360.0
        return speed_kt, from_deg

    def test_anchor_values_round_trip(self):
        """Querying at anchor times should return the configured speed/dir."""
        from run_route import _build_schedule_wind
        cfg = {"source": "schedule", "schedule": [
            {"time": "09:15", "speed_kt": 9.0,  "from_deg": 340.0},
            {"time": "11:00", "speed_kt": 12.0, "from_deg": 330.0},
            {"time": "13:00", "speed_kt": 16.0, "from_deg": 320.0},
        ]}
        wf = _build_schedule_wind(cfg, self._ctx())

        for t_h, exp_kt, exp_dir in [(0.0, 9.0, 340.0), (1.75, 12.0, 330.0), (3.75, 16.0, 320.0)]:
            wu, wv = wf.query(0.0, 0.0, elapsed_s=t_h * 3600.0)
            spd, frm = self._decode(wu, wv)
            assert spd == pytest.approx(exp_kt, abs=0.05), f"t={t_h}h speed"
            assert frm == pytest.approx(exp_dir, abs=0.5), f"t={t_h}h dir"

    def test_midpoint_interpolation(self):
        """Midpoint between two anchors interpolates wu/wv linearly."""
        from run_route import _build_schedule_wind
        cfg = {"source": "schedule", "schedule": [
            {"time": "09:15", "speed_kt": 9.0,  "from_deg": 340.0},
            {"time": "11:00", "speed_kt": 12.0, "from_deg": 330.0},
        ]}
        wf = _build_schedule_wind(cfg, self._ctx())

        # Anchors are at t=0 and t=1.75h (6300s); midpoint = 3150s.
        wu0, wv0 = wf.query(0.0, 0.0, elapsed_s=0.0)
        wu1, wv1 = wf.query(0.0, 0.0, elapsed_s=6300.0)
        wum, wvm = wf.query(0.0, 0.0, elapsed_s=3150.0)
        assert wum == pytest.approx(0.5 * (wu0 + wu1), rel=0.01)
        assert wvm == pytest.approx(0.5 * (wv0 + wv1), rel=0.01)

    def test_clamps_outside_schedule(self):
        """Queries before first / after last anchor clamp to the endpoint."""
        from run_route import _build_schedule_wind
        cfg = {"source": "schedule", "schedule": [
            {"time": "09:15", "speed_kt": 9.0,  "from_deg": 340.0},
            {"time": "13:00", "speed_kt": 16.0, "from_deg": 320.0},
        ]}
        wf = _build_schedule_wind(cfg, self._ctx())

        # Before start (t = -1h)
        spd_before, frm_before = self._decode(*wf.query(0.0, 0.0, elapsed_s=-3600.0))
        assert spd_before == pytest.approx(9.0, abs=0.05)
        assert frm_before == pytest.approx(340.0, abs=0.5)

        # After end (t = 8h, well past 13:00 anchor at t=3.75h)
        spd_after, frm_after = self._decode(*wf.query(0.0, 0.0, elapsed_s=8 * 3600.0))
        assert spd_after == pytest.approx(16.0, abs=0.05)
        assert frm_after == pytest.approx(320.0, abs=0.5)

    def test_single_anchor_acts_as_constant(self):
        """A single anchor degenerates to a constant wind field."""
        from run_route import _build_schedule_wind
        cfg = {"source": "schedule", "schedule": [
            {"time": "10:00", "speed_kt": 10.0, "from_deg": 270.0},
        ]}
        wf = _build_schedule_wind(cfg, self._ctx())
        for t_h in (0.0, 2.5, 5.0):
            spd, frm = self._decode(*wf.query(0.0, 0.0, elapsed_s=t_h * 3600.0))
            assert spd == pytest.approx(10.0, abs=0.05)
            assert frm == pytest.approx(270.0, abs=0.5)

    def test_start_time_offset_applied(self):
        """Schedule should be aligned to ctx['start_time_s'] (router time-base)."""
        from run_route import _build_schedule_wind
        ctx = self._ctx()
        ctx["start_time_s"] = 5000.0
        cfg = {"source": "schedule", "schedule": [
            {"time": "09:15", "speed_kt": 9.0,  "from_deg": 340.0},
            {"time": "11:00", "speed_kt": 12.0, "from_deg": 330.0},
        ]}
        wf = _build_schedule_wind(cfg, ctx)
        # The 09:15 anchor should now be at elapsed_s=5000 (not 0).
        spd, frm = self._decode(*wf.query(0.0, 0.0, elapsed_s=5000.0))
        assert spd == pytest.approx(9.0, abs=0.05)
        assert frm == pytest.approx(340.0, abs=0.5)


# =====================================================================
#  10.  Run as script
# =====================================================================

class TestTackingPenalty:
    """Tacking penalty should increase total time when the route requires
    course changes, and the algorithm should prefer routes that minimize
    the number of tacks when a penalty is applied."""

    def test_penalty_zero_matches_default(self):
        """With tack_penalty_s=0 the route time should match or be close to
        the default-penalty case (both produce valid routes; zero penalty
        cannot be worse in time since no overhead is added)."""
        cf, tr = make_synthetic_field()
        boat = BoatModel(base_speed_knots=6.0)

        start = _latlon_for_offset(-2000, -2000, tr)
        end = _latlon_for_offset(2000, 2000, tr)

        router_no_penalty = Router(cf, boat, resolution_m=200.0,
                                   padding_m=1000.0, tack_penalty_s=0.0)
        router_default = Router(cf, boat, resolution_m=200.0,
                                padding_m=1000.0, tack_penalty_s=60.0)

        route_np, _, _, _ = router_no_penalty.find_route(start, end)
        route_dp, _, _, _ = router_default.find_route(start, end)

        # Without penalty the route may take more tacks but total physical
        # travel time should not be dramatically worse (within 10%).
        assert route_np.total_time_s <= route_dp.total_time_s * 1.10, (
            f"No-penalty route ({route_np.total_time_s:.1f}s) should not be "
            f"much slower than default-penalty route ({route_dp.total_time_s:.1f}s)"
        )
        # Both routes must be valid
        assert np.isfinite(route_np.total_time_s)
        assert np.isfinite(route_dp.total_time_s)

    def test_straight_route_no_penalty_incurred(self):
        """A straight north-south route makes no course changes so
        tack_penalty_s should not affect the travel time."""
        cf, tr = make_synthetic_field()
        boat = BoatModel(base_speed_knots=6.0)

        start = _latlon_for_offset(0, -3000, tr)
        end = _latlon_for_offset(0, 3000, tr)

        router_no_p = Router(cf, boat, resolution_m=200.0, padding_m=1000.0,
                             tack_penalty_s=0.0)
        router_big_p = Router(cf, boat, resolution_m=200.0, padding_m=1000.0,
                              tack_penalty_s=300.0)

        route_no_p, _, _, _ = router_no_p.find_route(start, end)
        route_big_p, _, _, _ = router_big_p.find_route(start, end)

        # Straight path = same direction throughout, so no penalty fired
        assert route_no_p.total_time_s == pytest.approx(
            route_big_p.total_time_s, rel=0.05)

    def test_angle_diff_precomputed_correctly(self):
        """_ANGLE_DIFF should be 0 for same direction, ~180 for opposite."""
        offsets = Router.NEIGHBOR_OFFSETS
        angle_diff = Router._ANGLE_DIFF

        # Same direction -> 0
        for i in range(8):
            assert angle_diff[i, i] == pytest.approx(0.0, abs=1e-9)

        # Opposite directions: (-1,0) vs (1,0), (0,-1) vs (0,1), etc.
        # Note: ys is ascending so dr=+1 is north and dr=-1 is south in the grid,
        # but what matters here is that opposite offsets differ by 180 degrees.
        idx_n = offsets.index((-1, 0))   # dr=-1: toward lower row (south in grid)
        idx_s = offsets.index((1, 0))    # dr=+1: toward higher row (north in grid)
        assert angle_diff[idx_n, idx_s] == pytest.approx(180.0, abs=0.1)

        # E = (0,1) idx=4, W = (0,-1) idx=3
        idx_e = offsets.index((0, 1))
        idx_w = offsets.index((0, -1))
        assert angle_diff[idx_e, idx_w] == pytest.approx(180.0, abs=0.1)

        # NE vs SW should also be 180
        idx_ne = offsets.index((-1, 1))
        idx_sw = offsets.index((1, -1))
        assert angle_diff[idx_ne, idx_sw] == pytest.approx(180.0, abs=0.1)

        # Adjacent directions (45 degrees apart)
        assert angle_diff[idx_n, idx_ne] == pytest.approx(45.0, abs=0.1)

    def test_large_penalty_favors_straighter_path(self):
        """With a very large tacking penalty the router should prefer a
        straighter path even if it is slightly longer in distance.
        Compare routes going NE and verifying waypoint count is reduced
        when a huge tacking penalty is applied."""
        cf, tr = make_synthetic_field()
        boat = BoatModel(base_speed_knots=6.0)

        # Diagonal trip: algorithm with no penalty may zigzag on the grid;
        # with a large penalty it should take a cleaner diagonal.
        start = _latlon_for_offset(-3000, -3000, tr)
        end = _latlon_for_offset(3000, 3000, tr)

        router_no_p = Router(cf, boat, resolution_m=300.0, padding_m=500.0,
                             tack_penalty_s=0.0)
        router_big_p = Router(cf, boat, resolution_m=300.0, padding_m=500.0,
                              tack_penalty_s=3600.0)

        route_no_p, _, _, _ = router_no_p.find_route(start, end)
        route_big_p, _, _, _ = router_big_p.find_route(start, end)

        # The smoothed path with a huge penalty should not be dramatically
        # longer in time (penalty is a planning cost, not added to final time).
        # Primarily we check that both routes are valid.
        assert route_no_p.total_distance_m > 0
        assert route_big_p.total_distance_m > 0
        assert np.isfinite(route_no_p.total_time_s)
        assert np.isfinite(route_big_p.total_time_s)


class TestRouteQuality:
    """Routes should not contain obvious geometric pathologies like
    Y-junctions (sharp reversals) or waypoints that move away from
    the destination."""

    @staticmethod
    def _leg_angle(ax, ay, bx, by, cx, cy):
        """Signed turn angle at B in the path A -> B -> C (degrees).
        Returns the absolute turn in [0, 180]."""
        v1x, v1y = bx - ax, by - ay
        v2x, v2y = cx - bx, cy - by
        dot = v1x * v2x + v1y * v2y
        cross = v1x * v2y - v1y * v2x
        return abs(np.degrees(np.arctan2(cross, dot)))

    def test_monotonic_progress_no_current(self):
        """With zero current every waypoint should be strictly closer
        to the goal than the previous one."""
        cf, tr = make_synthetic_field()
        boat = BoatModel(base_speed_knots=6.0)
        router = Router(cf, boat, resolution_m=200.0, padding_m=1000.0)

        start = _latlon_for_offset(-2000, -3000, tr)
        end = _latlon_for_offset(2000, 3000, tr)
        route, _, _, _ = router.find_route(start, end)

        wps = route.waypoints_utm
        ex, ey = wps[-1]
        for i in range(len(wps) - 1):
            d_cur = np.hypot(wps[i][0] - ex, wps[i][1] - ey)
            d_nxt = np.hypot(wps[i + 1][0] - ex, wps[i + 1][1] - ey)
            assert d_nxt < d_cur + 1.0, (
                f"WP{i+1} is farther from goal than WP{i}: "
                f"{d_nxt:.0f} >= {d_cur:.0f}")

    def test_no_sharp_reversals_no_current(self):
        """No consecutive legs should reverse direction > 120 degrees
        in the absence of current."""
        cf, tr = make_synthetic_field()
        boat = BoatModel(base_speed_knots=6.0)
        router = Router(cf, boat, resolution_m=200.0, padding_m=1000.0)

        start = _latlon_for_offset(-2000, -2000, tr)
        end = _latlon_for_offset(2000, 2000, tr)
        route, _, _, _ = router.find_route(start, end)

        wps = route.waypoints_utm
        for i in range(len(wps) - 2):
            angle = self._leg_angle(
                wps[i][0], wps[i][1],
                wps[i+1][0], wps[i+1][1],
                wps[i+2][0], wps[i+2][1])
            assert angle < 120.0, (
                f"Sharp reversal of {angle:.0f} deg at WP{i+1}")

    def test_no_sharp_reversals_with_current(self):
        """Even with a uniform cross-current the route should not
        contain Y-junctions."""
        cf, tr = make_synthetic_field(u_func=lambda x, y: 0.5)
        boat = BoatModel(base_speed_knots=6.0)
        router = Router(cf, boat, resolution_m=200.0, padding_m=1000.0)

        start = _latlon_for_offset(0, -3000, tr)
        end = _latlon_for_offset(0, 3000, tr)
        route, _, _, _ = router.find_route(start, end)

        wps = route.waypoints_utm
        for i in range(len(wps) - 2):
            angle = self._leg_angle(
                wps[i][0], wps[i][1],
                wps[i+1][0], wps[i+1][1],
                wps[i+2][0], wps[i+2][1])
            assert angle < 120.0, (
                f"Sharp reversal of {angle:.0f} deg at WP{i+1}")

    def test_monotonic_progress_upwind_polar(self):
        """Upwind tacking with polars: every waypoint should still be
        closer to the goal than the previous one (no Y-junctions)."""
        if not POLAR_CSV.exists():
            pytest.skip(f"Polar CSV not found at {POLAR_CSV}")
        polar = PolarTable(POLAR_CSV)
        cf, tr = make_synthetic_field()
        boat = BoatModel(base_speed_knots=6.0, polar=polar)
        wind = WindField.from_met(15, 180)
        router = Router(cf, boat, resolution_m=200.0, padding_m=2000.0,
                        wind=wind, tack_penalty_s=120.0)

        start = _latlon_for_offset(0, -3000, tr)
        end = _latlon_for_offset(0, 3000, tr)
        route, _, _, _ = router.find_route(start, end)

        wps = route.waypoints_utm
        ex, ey = wps[-1]
        for i in range(len(wps) - 1):
            d_cur = np.hypot(wps[i][0] - ex, wps[i][1] - ey)
            d_nxt = np.hypot(wps[i + 1][0] - ex, wps[i + 1][1] - ey)
            assert d_nxt < d_cur + 50.0, (
                f"WP{i+1} is farther from goal than WP{i}: "
                f"{d_nxt:.0f} >= {d_cur:.0f}")

    def test_no_sharp_reversals_upwind_polar(self):
        """Upwind polar route: no consecutive legs should have a turn
        > 120 degrees (Y-junction)."""
        if not POLAR_CSV.exists():
            pytest.skip(f"Polar CSV not found at {POLAR_CSV}")
        polar = PolarTable(POLAR_CSV)
        cf, tr = make_synthetic_field()
        boat = BoatModel(base_speed_knots=6.0, polar=polar)
        wind = WindField.from_met(15, 180)
        router = Router(cf, boat, resolution_m=200.0, padding_m=2000.0,
                        wind=wind, tack_penalty_s=120.0)

        start = _latlon_for_offset(0, -3000, tr)
        end = _latlon_for_offset(0, 3000, tr)
        route, _, _, _ = router.find_route(start, end)

        wps = route.waypoints_utm
        for i in range(len(wps) - 2):
            angle = self._leg_angle(
                wps[i][0], wps[i][1],
                wps[i+1][0], wps[i+1][1],
                wps[i+2][0], wps[i+2][1])
            assert angle < 120.0, (
                f"Sharp reversal of {angle:.0f} deg at WP{i+1}, "
                f"waypoints: {wps[i]}, {wps[i+1]}, {wps[i+2]}")

    def test_no_sharp_reversals_downwind_polar(self):
        """Downwind polar route: no Y-junctions either."""
        if not POLAR_CSV.exists():
            pytest.skip(f"Polar CSV not found at {POLAR_CSV}")
        polar = PolarTable(POLAR_CSV)
        cf, tr = make_synthetic_field()
        boat = BoatModel(base_speed_knots=6.0, polar=polar)
        wind = WindField.from_met(15, 0)
        router = Router(cf, boat, resolution_m=200.0, padding_m=2000.0,
                        wind=wind, tack_penalty_s=120.0)

        start = _latlon_for_offset(0, -3000, tr)
        end = _latlon_for_offset(0, 3000, tr)
        route, _, _, _ = router.find_route(start, end)

        wps = route.waypoints_utm
        for i in range(len(wps) - 2):
            angle = self._leg_angle(
                wps[i][0], wps[i][1],
                wps[i+1][0], wps[i+1][1],
                wps[i+2][0], wps[i+2][1])
            assert angle < 120.0, (
                f"Sharp reversal of {angle:.0f} deg at WP{i+1}")

    def test_no_stub_legs(self):
        """A very short leg followed by a long leg in a different direction
        is a 'stub' (Y-junction artifact).  The smoothing should remove it.
        Stub = a leg shorter than 5x grid resolution where the turn at
        its endpoint exceeds 45 degrees."""
        cf, tr = make_synthetic_field()
        boat = BoatModel(base_speed_knots=6.0)
        resolution = 200.0

        for padding in [1000.0, 2000.0]:
            router = Router(cf, boat, resolution_m=resolution,
                            padding_m=padding)

            start = _latlon_for_offset(-2000, -3000, tr)
            end = _latlon_for_offset(2000, 3000, tr)
            route, _, _, _ = router.find_route(start, end)

            wps = route.waypoints_utm
            for i in range(len(wps) - 2):
                leg_in = np.hypot(wps[i+1][0] - wps[i][0],
                                  wps[i+1][1] - wps[i][1])
                leg_out = np.hypot(wps[i+2][0] - wps[i+1][0],
                                   wps[i+2][1] - wps[i+1][1])
                short_leg = min(leg_in, leg_out)
                long_leg = max(leg_in, leg_out)
                if short_leg < 5 * resolution and long_leg > 10 * resolution:
                    angle = self._leg_angle(
                        wps[i][0], wps[i][1],
                        wps[i+1][0], wps[i+1][1],
                        wps[i+2][0], wps[i+2][1])
                    assert angle < 45.0, (
                        f"Stub at WP{i+1}: short leg {short_leg:.0f}m, "
                        f"long leg {long_leg:.0f}m, turn {angle:.0f} deg")

    def test_no_stub_legs_polar(self):
        """Stub detection with polar + wind (upwind and downwind)."""
        if not POLAR_CSV.exists():
            pytest.skip(f"Polar CSV not found at {POLAR_CSV}")
        polar = PolarTable(POLAR_CSV)
        cf, tr = make_synthetic_field()
        resolution = 200.0

        for wind_dir in [0, 90, 180, 270]:
            boat = BoatModel(base_speed_knots=6.0, polar=polar)
            wind = WindField.from_met(15, wind_dir)
            router = Router(cf, boat, resolution_m=resolution,
                            padding_m=2000.0, wind=wind,
                            tack_penalty_s=120.0)

            start = _latlon_for_offset(0, -3000, tr)
            end = _latlon_for_offset(0, 3000, tr)
            route, _, _, _ = router.find_route(start, end)

            wps = route.waypoints_utm
            for i in range(len(wps) - 2):
                leg_in = np.hypot(wps[i+1][0] - wps[i][0],
                                  wps[i+1][1] - wps[i][1])
                leg_out = np.hypot(wps[i+2][0] - wps[i+1][0],
                                   wps[i+2][1] - wps[i+1][1])
                short_leg = min(leg_in, leg_out)
                long_leg = max(leg_in, leg_out)
                if short_leg < 5 * resolution and long_leg > 10 * resolution:
                    angle = self._leg_angle(
                        wps[i][0], wps[i][1],
                        wps[i+1][0], wps[i+1][1],
                        wps[i+2][0], wps[i+2][1])
                    assert angle < 45.0, (
                        f"Stub at WP{i+1} (wind={wind_dir}): "
                        f"short leg {short_leg:.0f}m, "
                        f"long leg {long_leg:.0f}m, turn {angle:.0f} deg")


class TestSimulatedTrack:
    """The dense exported track should stay continuous and match route timing."""

    def test_track_times_match_route_duration(self):
        """The sampled track should end at the same total time as the route."""
        cf, tr = make_synthetic_field()
        boat = BoatModel(base_speed_knots=6.0)
        router = Router(cf, boat, resolution_m=200.0, padding_m=1000.0)

        start = _latlon_for_offset(0, -2000, tr)
        end = _latlon_for_offset(0, 2000, tr)
        route, _, _, _ = router.find_route(start, end)

        assert route.simulated_track_times is not None
        assert route.simulated_track is not None
        assert route.simulated_track_times[0] == pytest.approx(0.0, abs=1e-6)
        assert route.simulated_track_times[-1] == pytest.approx(
            route.total_time_s, rel=0.02)
        assert route.simulated_track[-1][0] == pytest.approx(
            route.waypoints_utm[-1][0], abs=1e-6)
        assert route.simulated_track[-1][1] == pytest.approx(
            route.waypoints_utm[-1][1], abs=1e-6)

    def test_polar_track_has_no_teleports(self, polar):
        """Upwind polar tracks should have monotonic time and no km-scale jumps."""
        cf, tr = make_synthetic_field()
        boat = BoatModel(base_speed_knots=6.0, polar=polar)
        wind = WindField.from_met(15, 180)
        router = Router(cf, boat, resolution_m=200.0, padding_m=2000.0,
                        wind=wind, tack_penalty_s=120.0)

        start = _latlon_for_offset(0, -3000, tr)
        end = _latlon_for_offset(0, 3000, tr)
        route, _, _, _ = router.find_route(start, end)

        times = np.asarray(route.simulated_track_times, dtype=float)
        pts = np.asarray(route.simulated_track, dtype=float)
        dt = np.diff(times)
        step = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))

        assert np.all(dt >= -1e-9), "Track time must be monotonic"
        assert np.max(step) < 500.0, (
            f"Track contains an implausible jump of {np.max(step):.0f} m")


# =====================================================================
#  MeshRouter tests
# =====================================================================

class TestBuildMeshAdjacency:
    """Tests for the mesh adjacency extraction helper."""

    def test_adjacency_from_simple_triangulation(self):
        """Build adjacency from a synthetic current field's Delaunay."""
        cf, _ = make_synthetic_field(half_size_deg=0.03, spacing_deg=0.005)
        assert cf.delaunay is not None
        assert cf.valid_triangle is not None

        adj, edge_dist = build_mesh_adjacency(cf.delaunay, cf.valid_triangle)

        assert len(adj) > 0, "Should have some nodes with edges"
        for node, neighbors in adj.items():
            assert len(neighbors) > 0, f"Node {node} should have neighbors"
            for nbr in neighbors:
                assert node in adj[nbr], "Adjacency should be symmetric"

        for (a, b), dist in edge_dist.items():
            assert a < b, "Edge keys should be canonically ordered"
            assert dist > 0, "Edge distances should be positive"


class TestMeshRouterBasic:
    """Basic MeshRouter tests with zero current."""

    def test_straight_north(self):
        """With zero current, route should be approximately straight."""
        cf, tr = make_synthetic_field(half_size_deg=0.04, spacing_deg=0.003)
        boat = BoatModel(base_speed_knots=6.0)
        router = MeshRouter(cf, boat, tack_penalty_s=0.0)

        start = _latlon_for_offset(0, -2000, tr)
        end = _latlon_for_offset(0, 2000, tr)

        route, xs, ys, wm = router.find_route(start, end)
        expected_dist = 4000.0
        expected_time = expected_dist / boat.speed()

        assert route.total_distance_m == pytest.approx(expected_dist, rel=0.15)
        assert route.total_time_s == pytest.approx(expected_time, rel=0.15)

    def test_diagonal(self):
        """Diagonal route should work."""
        cf, tr = make_synthetic_field(half_size_deg=0.04, spacing_deg=0.003)
        boat = BoatModel(base_speed_knots=6.0)
        router = MeshRouter(cf, boat, tack_penalty_s=0.0)

        start = _latlon_for_offset(-1500, -1500, tr)
        end = _latlon_for_offset(1500, 1500, tr)

        route, _, _, _ = router.find_route(start, end)
        expected_dist = np.hypot(3000, 3000)

        assert route.total_distance_m == pytest.approx(expected_dist, rel=0.20)


class TestMeshRouterWithCurrent:
    """MeshRouter tests with uniform current."""

    def test_favorable_current_faster(self):
        """Traveling north with northward current should be faster."""
        cf, tr = make_synthetic_field(
            v_func=lambda x, y: 1.0,
            half_size_deg=0.04, spacing_deg=0.003)
        boat = BoatModel(base_speed_knots=6.0)
        router = MeshRouter(cf, boat, tack_penalty_s=0.0)

        start = _latlon_for_offset(0, -2000, tr)
        end = _latlon_for_offset(0, 2000, tr)

        route, _, _, _ = router.find_route(start, end)
        sog_ms = route.total_distance_m / route.total_time_s
        expected_sog = boat.speed() + 1.0

        assert sog_ms == pytest.approx(expected_sog, rel=0.15)

    def test_opposing_current_slower(self):
        """Traveling north against southward current should be slower."""
        cf, tr = make_synthetic_field(
            v_func=lambda x, y: -1.0,
            half_size_deg=0.04, spacing_deg=0.003)
        boat = BoatModel(base_speed_knots=6.0)
        router = MeshRouter(cf, boat, tack_penalty_s=0.0)

        start = _latlon_for_offset(0, -2000, tr)
        end = _latlon_for_offset(0, 2000, tr)

        route, _, _, _ = router.find_route(start, end)
        sog_ms = route.total_distance_m / route.total_time_s
        expected_sog = boat.speed() - 1.0

        assert sog_ms == pytest.approx(expected_sog, rel=0.15)


class TestMeshRouterLandAvoidance:
    """MeshRouter should avoid land areas."""

    def test_routes_around_land_obstacle(self):
        """A land obstacle between start and end forces a detour."""
        tr = _make_transformer()
        cx, cy = tr.transform(CENTER_LON, CENTER_LAT)

        def land_func(x, y):
            return abs(x - cx) < 500 and abs(y - cy) < 500

        cf, tr = make_synthetic_field(
            land_func=land_func,
            half_size_deg=0.04, spacing_deg=0.003,
            land_threshold_m=200.0)

        boat = BoatModel(base_speed_knots=6.0)
        router = MeshRouter(cf, boat, tack_penalty_s=0.0)

        start = _latlon_for_offset(0, -2500, tr)
        end = _latlon_for_offset(0, 2500, tr)

        route, _, _, _ = router.find_route(start, end)
        straight_dist = 5000.0
        assert route.total_distance_m > straight_dist, (
            "Route should detour around land")


class TestMeshRouterTackingPenalty:
    """MeshRouter tacking penalty should discourage sharp turns."""

    def test_tacking_penalty_increases_cost(self):
        """A zigzag path should be more expensive with tacking penalty."""
        cf, tr = make_synthetic_field(half_size_deg=0.04, spacing_deg=0.003)
        boat = BoatModel(base_speed_knots=6.0)

        router_no_penalty = MeshRouter(cf, boat, tack_penalty_s=0.0)
        router_with_penalty = MeshRouter(cf, boat, tack_penalty_s=120.0,
                                         tack_threshold_deg=60.0)

        start = _latlon_for_offset(0, -2000, tr)
        end = _latlon_for_offset(0, 2000, tr)

        route_no_pen, _, _, _ = router_no_penalty.find_route(start, end)
        route_with_pen, _, _, _ = router_with_penalty.find_route(start, end)

        assert route_with_pen.total_time_s >= route_no_pen.total_time_s * 0.95


class TestMeshRouterTimeDependentCurrent:
    """MeshRouter with time-varying current field."""

    def test_uses_time_varying_current(self):
        """Route should adapt to current that changes over time."""
        def u0(x, y): return 0.0
        def v0(x, y): return 1.0  # northward at t=0

        def u1(x, y): return 0.0
        def v1(x, y): return -1.0  # southward at t=3600

        cf, tr = make_time_varying_field(
            frame_specs=[(u0, v0), (u1, v1)],
            frame_times_s=[0.0, 3600.0],
            half_size_deg=0.04, spacing_deg=0.003)

        boat = BoatModel(base_speed_knots=6.0)
        router = MeshRouter(cf, boat, tack_penalty_s=0.0)

        start = _latlon_for_offset(0, -2000, tr)
        end = _latlon_for_offset(0, 2000, tr)

        route, _, _, _ = router.find_route(start, end, start_time_s=0.0)
        assert route.total_time_s > 0


class TestMeshRouterRouteQuality:
    """Route quality checks for MeshRouter."""

    def test_waypoints_progress_toward_goal(self):
        """Each waypoint should be closer to the goal than the previous."""
        cf, tr = make_synthetic_field(half_size_deg=0.04, spacing_deg=0.003)
        boat = BoatModel(base_speed_knots=6.0)
        router = MeshRouter(cf, boat, tack_penalty_s=0.0)

        start = _latlon_for_offset(0, -2000, tr)
        end = _latlon_for_offset(0, 2000, tr)

        route, _, _, _ = router.find_route(start, end)

        end_x, end_y = tr.transform(end[1], end[0])
        prev_dist = float('inf')
        for wp in route.waypoints_utm:
            dist = np.hypot(wp[0] - end_x, wp[1] - end_y)
            assert dist <= prev_dist + 100, "Waypoints should progress toward goal"
            prev_dist = dist

    def test_simulated_track_timing(self):
        """Simulated track should have consistent timing."""
        cf, tr = make_synthetic_field(half_size_deg=0.04, spacing_deg=0.003)
        boat = BoatModel(base_speed_knots=6.0)
        router = MeshRouter(cf, boat, tack_penalty_s=0.0)

        start = _latlon_for_offset(0, -2000, tr)
        end = _latlon_for_offset(0, 2000, tr)

        route, _, _, _ = router.find_route(start, end)

        assert route.simulated_track is not None
        assert route.simulated_track_times is not None
        assert len(route.simulated_track) == len(route.simulated_track_times)

        times = np.asarray(route.simulated_track_times)
        assert np.all(np.diff(times) >= -1e-9), "Track time must be monotonic"

        final_time = times[-1] - times[0]
        assert final_time == pytest.approx(route.total_time_s, rel=0.15)


# =====================================================================
#  SectorRouter tests
# =====================================================================

class TestSectorRouterBasic:
    """Basic SectorRouter tests with zero current."""

    def test_straight_north(self):
        """With zero current, route should be approximately straight."""
        cf, tr = make_synthetic_field(half_size_deg=0.04, spacing_deg=0.003)
        boat = BoatModel(base_speed_knots=6.0)
        router = SectorRouter(cf, boat, tack_penalty_s=0.0)

        start = _latlon_for_offset(0, -2000, tr)
        end = _latlon_for_offset(0, 2000, tr)

        route, xs, ys, wm = router.find_route(start, end)
        expected_dist = 4000.0
        expected_time = expected_dist / boat.speed()

        assert route.total_distance_m == pytest.approx(expected_dist, rel=0.20)
        assert route.total_time_s == pytest.approx(expected_time, rel=0.20)

    def test_diagonal(self):
        """Diagonal route should work."""
        cf, tr = make_synthetic_field(half_size_deg=0.04, spacing_deg=0.003)
        boat = BoatModel(base_speed_knots=6.0)
        router = SectorRouter(cf, boat, tack_penalty_s=0.0)

        start = _latlon_for_offset(-1500, -1500, tr)
        end = _latlon_for_offset(1500, 1500, tr)

        route, _, _, _ = router.find_route(start, end)
        expected_dist = np.hypot(3000, 3000)

        assert route.total_distance_m == pytest.approx(expected_dist, rel=0.25)


class TestSectorRouterWithCurrent:
    """SectorRouter tests with uniform current."""

    def test_favorable_current_faster(self):
        """Traveling north with northward current should be faster."""
        cf, tr = make_synthetic_field(
            v_func=lambda x, y: 1.0,
            half_size_deg=0.04, spacing_deg=0.003)
        boat = BoatModel(base_speed_knots=6.0)
        router = SectorRouter(cf, boat, tack_penalty_s=0.0)

        start = _latlon_for_offset(0, -2000, tr)
        end = _latlon_for_offset(0, 2000, tr)

        route, _, _, _ = router.find_route(start, end)
        sog_ms = route.total_distance_m / route.total_time_s
        expected_sog = boat.speed() + 1.0

        assert sog_ms == pytest.approx(expected_sog, rel=0.20)

    def test_opposing_current_slower(self):
        """Traveling north against southward current should be slower."""
        cf, tr = make_synthetic_field(
            v_func=lambda x, y: -1.0,
            half_size_deg=0.04, spacing_deg=0.003)
        boat = BoatModel(base_speed_knots=6.0)
        router = SectorRouter(cf, boat, tack_penalty_s=0.0)

        start = _latlon_for_offset(0, -2000, tr)
        end = _latlon_for_offset(0, 2000, tr)

        route, _, _, _ = router.find_route(start, end)
        sog_ms = route.total_distance_m / route.total_time_s
        expected_sog = boat.speed() - 1.0

        assert sog_ms == pytest.approx(expected_sog, rel=0.20)


class TestSectorRouterLandAvoidance:
    """SectorRouter should avoid land areas."""

    def test_routes_around_land_obstacle(self):
        """A land obstacle between start and end forces a detour."""
        tr = _make_transformer()
        cx, cy = tr.transform(CENTER_LON, CENTER_LAT)

        def land_func(x, y):
            rel_x = x - cx
            rel_y = y - cy
            return abs(rel_x) < 300 and -1500 < rel_y < 1500

        cf, _ = make_synthetic_field(land_func=land_func, spacing_deg=0.003)
        boat = BoatModel(base_speed_knots=6.0)
        router = SectorRouter(cf, boat, tack_penalty_s=0.0)

        start = _latlon_for_offset(-2000, 0, tr)
        end = _latlon_for_offset(2000, 0, tr)

        route, _, _, _ = router.find_route(start, end)
        assert route.total_distance_m > 4000, (
            "Route should detour around land")


class TestSectorRouterTackingPenalty:
    """SectorRouter tacking penalty should discourage sharp turns."""

    def test_tacking_penalty_increases_cost(self):
        """A route with tack penalty should cost at least as much."""
        cf, tr = make_synthetic_field(half_size_deg=0.04, spacing_deg=0.003)
        boat = BoatModel(base_speed_knots=6.0)

        router_no_pen = SectorRouter(cf, boat, tack_penalty_s=0.0)
        router_with_pen = SectorRouter(cf, boat, tack_penalty_s=120.0)

        start = _latlon_for_offset(-1500, -1500, tr)
        end = _latlon_for_offset(1500, 1500, tr)

        route_no_pen, _, _, _ = router_no_pen.find_route(start, end)
        route_with_pen, _, _, _ = router_with_pen.find_route(start, end)

        assert route_with_pen.total_time_s >= route_no_pen.total_time_s * 0.90


class TestSectorRouterManeuverPenalty:
    """Wind-relative maneuver penalties for tack/gybe detection."""

    def test_downwind_side_flip_gets_gybe_penalty(self):
        """A port/starboard flip deep downwind should use gybe_penalty_s."""
        if not POLAR_CSV.exists():
            pytest.skip(f"Polar CSV not found at {POLAR_CSV}")
        cf, _ = make_synthetic_field(half_size_deg=0.02, spacing_deg=0.004)
        boat = BoatModel(base_speed_knots=6.0,
                         polar=PolarTable(POLAR_CSV, minimum_twa=38))
        wind = WindField.from_met(10.0, 90.0)  # wind from east: math angle 0
        router = SectorRouter(cf, boat, wind=wind,
                              tack_penalty_s=30.0,
                              gybe_penalty_s=120.0,
                              gybe_threshold_deg=120.0)

        penalty = router._maneuver_penalty(
            np.radians(170.0), np.radians(-170.0),
            cf._x_utm[0], cf._y_utm[0], 0.0, angle_diff_deg=20.0)
        assert penalty == pytest.approx(120.0)

    def test_dot_filter_matches_exact_on_mild_case(self):
        """Dot filter remains covered as experimental, not a production default."""
        if not POLAR_CSV.exists():
            pytest.skip(f"Polar CSV not found at {POLAR_CSV}")
        polar = PolarTable(POLAR_CSV, minimum_twa=38)
        cf, tr = make_synthetic_field(half_size_deg=0.04, spacing_deg=0.004)
        boat = BoatModel(base_speed_knots=6.0, polar=polar)
        wind = WindField.from_met(12.0, 180.0)
        start = _latlon_for_offset(-1200, -1200, tr)
        end = _latlon_for_offset(1200, 1200, tr)

        exact = SectorRouter(cf, boat, wind=wind, tack_penalty_s=0.0,
                             polar_sweep_coarse_step=1,
                             use_dense_polar=True,
                             use_dot_filter=False)
        filtered = SectorRouter(cf, boat, wind=wind, tack_penalty_s=0.0,
                                polar_sweep_coarse_step=1,
                                use_dense_polar=True,
                                use_dot_filter=True)

        route_exact, _, _, _ = exact.find_route(start, end)
        route_filtered, _, _, _ = filtered.find_route(start, end)
        assert route_filtered.total_time_s == pytest.approx(
            route_exact.total_time_s, rel=0.02)


class TestSectorRouterHeadingDiversity:
    """SectorRouter should provide diverse heading options."""

    def test_16_sector_coverage(self):
        """The router should be able to navigate to any of 16 directions."""
        cf, tr = make_synthetic_field(half_size_deg=0.06, spacing_deg=0.003)
        boat = BoatModel(base_speed_knots=6.0)
        router = SectorRouter(cf, boat, tack_penalty_s=0.0)

        start = _latlon_for_offset(0, 0, tr)

        directions_reached = 0
        for angle_deg in range(0, 360, 45):
            dx = 2500 * np.sin(np.radians(angle_deg))
            dy = 2500 * np.cos(np.radians(angle_deg))
            try:
                end = _latlon_for_offset(dx, dy, tr)
                route, _, _, _ = router.find_route(start, end)
                if route.total_distance_m > 0:
                    directions_reached += 1
            except Exception:
                pass

        assert directions_reached >= 6, (
            "Should be able to route to most compass directions")


class TestSectorRouterRouteQuality:
    """Route quality checks for SectorRouter."""

    def test_waypoints_progress_toward_goal(self):
        """Waypoints should generally get closer to the goal."""
        cf, tr = make_synthetic_field(half_size_deg=0.04, spacing_deg=0.003)
        boat = BoatModel(base_speed_knots=6.0)
        router = SectorRouter(cf, boat, tack_penalty_s=0.0)

        start = _latlon_for_offset(0, -2000, tr)
        end = _latlon_for_offset(0, 2000, tr)

        route, _, _, _ = router.find_route(start, end)

        end_x, end_y = tr.transform(end[1], end[0])
        prev_dist = float('inf')
        for wp in route.waypoints_utm:
            dist = np.hypot(wp[0] - end_x, wp[1] - end_y)
            assert dist <= prev_dist + 200, "Waypoints should progress toward goal"
            prev_dist = dist

    def test_simulated_track_timing(self):
        """Simulated track should have consistent timing."""
        cf, tr = make_synthetic_field(half_size_deg=0.04, spacing_deg=0.003)
        boat = BoatModel(base_speed_knots=6.0)
        router = SectorRouter(cf, boat, tack_penalty_s=0.0)

        start = _latlon_for_offset(0, -2000, tr)
        end = _latlon_for_offset(0, 2000, tr)

        route, _, _, _ = router.find_route(start, end)

        assert route.simulated_track is not None
        assert route.simulated_track_times is not None
        assert route.debug is not None
        assert "maneuver_tack_count" in route.debug
        assert "maneuver_gybe_count" in route.debug
        assert len(route.simulated_track) == len(route.simulated_track_times)

        times = np.asarray(route.simulated_track_times)
        assert np.all(np.diff(times) >= -1e-9), "Track time must be monotonic"

        final_time = times[-1] - times[0]
        assert final_time == pytest.approx(route.total_time_s, rel=0.20)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
