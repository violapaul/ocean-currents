"""
sail_routing.py
---------------

Sailboat routing algorithm that finds the time-optimal path through
SSCOFS ocean current data.

Phase 1: fixed boat speed through water (current only).
Phase 2: polar-based speed dependent on true wind angle and speed,
         with optional spatial/time-varying wind field.

The router uses time-dependent A* on a regular grid.  Each node carries
an arrival time so that edge costs reflect the currents *at that moment*
(SSCOFS provides hourly forecasts).  With a single time snapshot the
algorithm degenerates to standard static A*.

Usage:
    python sail_routing.py \\
        --start-lat 47.63 --start-lon -122.40 \\
        --end-lat 47.75 --end-lon -122.42 \\
        --boat-speed 6 \\
        --grid-resolution 300 \\
        --save route.png

    # With polar and constant wind:
    python sail_routing.py \\
        --start-lat 47.63 --start-lon -122.40 \\
        --end-lat 47.75 --end-lon -122.42 \\
        --polar /path/to/j105_polar_data_long.csv \\
        --wind-speed 12 --wind-direction 180 \\
        --grid-resolution 300 \\
        --save route_polar.png
"""

import argparse
import heapq
import sys
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Local imports from the existing SSCOFS pipeline
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from plot_local_currents import get_latest_current_data, create_utm_transformer

KNOTS_TO_MS = 0.514444
MS_TO_KNOTS = 1.0 / KNOTS_TO_MS


# ===================================================================
#  PolarTable -- boat performance as function of TWA and TWS
# ===================================================================

class PolarTable:
    """Sailboat polar performance table.

    Stores boat speed (knots) as a function of True Wind Angle (TWA,
    degrees 0-180) and True Wind Speed (TWS, knots).  Bilinear
    interpolation is used for intermediate values.

    Parameters
    ----------
    csv_path : str or Path
        CSV with columns TWA_deg, TWS_kt, BoatSpeed_kt.
    """

    def __init__(self, csv_path):
        import csv
        csv_path = Path(csv_path)
        rows = []
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append((float(row['TWA_deg']),
                             float(row['TWS_kt']),
                             float(row['BoatSpeed_kt'])))

        twa_vals = sorted(set(r[0] for r in rows))
        tws_vals = sorted(set(r[1] for r in rows))
        self._twas = np.array(twa_vals, dtype=np.float64)
        self._twss = np.array(tws_vals, dtype=np.float64)

        n_twa = len(twa_vals)
        n_tws = len(tws_vals)
        twa_idx = {v: i for i, v in enumerate(twa_vals)}
        tws_idx = {v: i for i, v in enumerate(tws_vals)}

        self._speeds = np.zeros((n_twa, n_tws), dtype=np.float64)
        for twa, tws, spd in rows:
            self._speeds[twa_idx[twa], tws_idx[tws]] = spd

        self.max_speed_kt = float(np.max(self._speeds))
        self.max_speed_ms = self.max_speed_kt * KNOTS_TO_MS

    def speed(self, twa_deg, tws_kt):
        """Interpolated boat speed in knots for given TWA and TWS.

        Parameters
        ----------
        twa_deg : float
            True Wind Angle in degrees [0, 180].
        tws_kt : float
            True Wind Speed in knots.

        Returns
        -------
        float  (knots)
        """
        twa_c = float(np.clip(twa_deg, self._twas[0], self._twas[-1]))
        tws_c = float(np.clip(tws_kt, self._twss[0], self._twss[-1]))

        i0 = int(np.searchsorted(self._twas, twa_c, side='right')) - 1
        i0 = int(np.clip(i0, 0, len(self._twas) - 2))
        i1 = i0 + 1

        j0 = int(np.searchsorted(self._twss, tws_c, side='right')) - 1
        j0 = int(np.clip(j0, 0, len(self._twss) - 2))
        j1 = j0 + 1

        ta0, ta1 = self._twas[i0], self._twas[i1]
        ts0, ts1 = self._twss[j0], self._twss[j1]

        alpha = (twa_c - ta0) / (ta1 - ta0) if ta1 > ta0 else 0.0
        beta = (tws_c - ts0) / (ts1 - ts0) if ts1 > ts0 else 0.0

        s00 = self._speeds[i0, j0]
        s10 = self._speeds[i1, j0]
        s01 = self._speeds[i0, j1]
        s11 = self._speeds[i1, j1]

        return float((1 - alpha) * (1 - beta) * s00 +
                     alpha       * (1 - beta) * s10 +
                     (1 - alpha) * beta       * s01 +
                     alpha       * beta       * s11)

    def speed_ms(self, twa_deg, tws_kt):
        """Interpolated boat speed in m/s."""
        return self.speed(twa_deg, tws_kt) * KNOTS_TO_MS


# ===================================================================
#  WindField -- wind vector field (constant, spatial, or time-varying)
# ===================================================================

class WindField:
    """Wind velocity field.

    Stores wind as (wu, wv) in m/s, where wu is the eastward component
    and wv the northward component of where the wind is blowing TO
    (matching the UTM convention of CurrentField).

    Phase 2a: constant everywhere and in time.
    Phase 2b: spatially varying (regular grid).
    Phase 2c: time-varying (multiple frames).

    Parameters
    ----------
    wu : float or ndarray
        Eastward wind component (m/s), wind blowing TO east.
    wv : float or ndarray
        Northward wind component (m/s), wind blowing TO north.
    """

    def __init__(self, wu, wv):
        self._wu = float(wu)
        self._wv = float(wv)
        self._mode = 'constant'
        # Spatial/temporal extensions filled by from_grid / from_frames
        self._xs = None
        self._ys = None
        self._wu_grid = None
        self._wv_grid = None
        self._wu_frames = None
        self._wv_frames = None
        self._frame_times = None

    @classmethod
    def from_met(cls, speed_kt, from_deg):
        """Construct from meteorological convention.

        Parameters
        ----------
        speed_kt : float
            Wind speed in knots.
        from_deg : float
            Direction wind is coming FROM, degrees clockwise from north.
        """
        speed_ms = speed_kt * KNOTS_TO_MS
        from_rad = np.radians(from_deg)
        wu = -speed_ms * np.sin(from_rad)
        wv = -speed_ms * np.cos(from_rad)
        return cls(wu, wv)

    @classmethod
    def from_grid(cls, xs, ys, wu_grid, wv_grid):
        """Construct a spatially varying (constant in time) wind field.

        Parameters
        ----------
        xs, ys : 1-D arrays
            UTM easting and northing coordinate arrays.
        wu_grid, wv_grid : 2-D arrays (ny, nx)
            Wind components at each grid point.
        """
        from scipy.interpolate import RegularGridInterpolator
        inst = cls(0.0, 0.0)
        inst._mode = 'grid'
        inst._xs = np.asarray(xs, dtype=np.float64)
        inst._ys = np.asarray(ys, dtype=np.float64)
        inst._wu_grid = np.asarray(wu_grid, dtype=np.float64)
        inst._wv_grid = np.asarray(wv_grid, dtype=np.float64)
        inst._interp_u = RegularGridInterpolator(
            (ys, xs), wu_grid, method='linear', bounds_error=False,
            fill_value=None)
        inst._interp_v = RegularGridInterpolator(
            (ys, xs), wv_grid, method='linear', bounds_error=False,
            fill_value=None)
        return inst

    @classmethod
    def from_frames(cls, xs, ys, wu_frames, wv_frames, frame_times_s):
        """Construct a time-varying (and optionally spatially varying) wind field.

        Parameters
        ----------
        xs, ys : 1-D arrays or None
            UTM coordinate arrays.  If None, wind is spatially constant
            per frame (just one value per frame).
        wu_frames, wv_frames : list of floats or 2-D arrays
        frame_times_s : list of float
        """
        if xs is None:
            inst = cls(wu_frames[0], wv_frames[0])
            inst._mode = 'temporal_const'
        else:
            inst = cls(0.0, 0.0)
            inst._mode = 'temporal_grid'
            inst._xs = np.asarray(xs, dtype=np.float64)
            inst._ys = np.asarray(ys, dtype=np.float64)
        inst._wu_frames = wu_frames
        inst._wv_frames = wv_frames
        inst._frame_times = np.asarray(frame_times_s, dtype=np.float64)
        return inst

    def query(self, x_utm, y_utm, elapsed_s=0.0):
        """Return (wu, wv) in m/s at given UTM position and time."""
        if self._mode == 'constant':
            return self._wu, self._wv

        if self._mode in ('temporal_const', 'temporal_grid'):
            t = np.clip(elapsed_s, self._frame_times[0],
                        self._frame_times[-1])
            idx = int(np.searchsorted(self._frame_times, t, side='right')) - 1
            idx = int(np.clip(idx, 0, len(self._frame_times) - 2))
            t0 = self._frame_times[idx]
            t1 = self._frame_times[idx + 1]
            alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0.0

            if self._mode == 'temporal_const':
                wu = (self._wu_frames[idx] * (1 - alpha) +
                      self._wu_frames[idx + 1] * alpha)
                wv = (self._wv_frames[idx] * (1 - alpha) +
                      self._wv_frames[idx + 1] * alpha)
                return float(wu), float(wv)
            else:
                pt = np.array([[y_utm, x_utm]])
                wu = float((self._wu_frames[idx](pt) * (1 - alpha) +
                            self._wu_frames[idx + 1](pt) * alpha)[0])
                wv = float((self._wv_frames[idx](pt) * (1 - alpha) +
                            self._wv_frames[idx + 1](pt) * alpha)[0])
                return wu, wv

        if self._mode == 'grid':
            pt = np.array([[y_utm, x_utm]])
            return float(self._interp_u(pt)[0]), float(self._interp_v(pt)[0])

        return self._wu, self._wv

    @property
    def wind_speed_ms(self):
        """Characteristic wind speed (m/s) for the constant/initial frame."""
        return float(np.hypot(self._wu, self._wv))

    @property
    def wind_speed_kt(self):
        return self.wind_speed_ms * MS_TO_KNOTS


# ===================================================================
#  CurrentField -- interpolated, optionally time-varying current data
# ===================================================================

class CurrentField:
    """Spatially interpolated ocean current field with optional time axis.

    Builds a KD-tree of SSCOFS element centers in UTM coordinates.
    Queries return (u, v) in m/s at any (x_utm, y_utm, time) via
    inverse-distance weighted interpolation of the K nearest elements.
    Points farther than ``land_threshold_m`` from any element center
    are treated as land (returns NaN).
    """

    def __init__(self, lonc, latc, u_frames, v_frames, frame_times_s,
                 transformer, k_neighbors=6, land_threshold_m=750.0):
        """
        Parameters
        ----------
        lonc, latc : 1-d arrays
            Element center coordinates (degrees).
        u_frames, v_frames : list of 1-d arrays
            Surface velocity components for each time frame (m/s).
        frame_times_s : 1-d array
            Elapsed seconds from simulation start for each frame.
        transformer : pyproj.Transformer
            Lon/lat -> UTM transformer.
        k_neighbors : int
            Number of neighbours for IDW interpolation.
        land_threshold_m : float
            Max distance (m) to nearest element; beyond this -> land.
        """
        self.transformer = transformer
        self.k = k_neighbors
        self.land_threshold = land_threshold_m

        x_utm, y_utm = transformer.transform(lonc, latc)
        self.tree = cKDTree(np.column_stack([x_utm, y_utm]))

        self.u_frames = [np.asarray(u, dtype=np.float64) for u in u_frames]
        self.v_frames = [np.asarray(v, dtype=np.float64) for v in v_frames]
        self.frame_times = np.asarray(frame_times_s, dtype=np.float64)
        self.n_frames = len(self.u_frames)

        speeds = [np.sqrt(u**2 + v**2) for u, v in zip(self.u_frames, self.v_frames)]
        self.max_current_speed = float(max(np.nanmax(s) for s in speeds))

    # ------------------------------------------------------------------

    def _idw_at_points(self, x, y, values):
        """IDW interpolation for an array of query points."""
        dists, idxs = self.tree.query(np.column_stack([x, y]), k=self.k)
        if self.k == 1:
            dists = dists[:, np.newaxis]
            idxs = idxs[:, np.newaxis]

        land_mask = dists[:, 0] > self.land_threshold
        weights = np.where(dists < 1e-12, 1e12, 1.0 / dists)
        w_sum = weights.sum(axis=1, keepdims=True)
        weights /= w_sum

        result = np.sum(weights * values[idxs], axis=1)
        result[land_mask] = np.nan
        return result

    def query(self, x_utm, y_utm, elapsed_s=0.0):
        """Return (u, v) in m/s at given UTM position(s) and time.

        Parameters
        ----------
        x_utm, y_utm : float or array
            UTM coordinates (m).
        elapsed_s : float
            Seconds elapsed since the simulation start time.

        Returns
        -------
        u, v : float or array  (m/s)
            NaN where the point is over land.
        """
        x = np.atleast_1d(np.asarray(x_utm, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y_utm, dtype=np.float64))

        if self.n_frames == 1:
            u = self._idw_at_points(x, y, self.u_frames[0])
            v = self._idw_at_points(x, y, self.v_frames[0])
        else:
            t = np.clip(elapsed_s, self.frame_times[0], self.frame_times[-1])
            idx = np.searchsorted(self.frame_times, t, side='right') - 1
            idx = int(np.clip(idx, 0, self.n_frames - 2))
            t0 = self.frame_times[idx]
            t1 = self.frame_times[idx + 1]
            alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0.0

            u0 = self._idw_at_points(x, y, self.u_frames[idx])
            v0 = self._idw_at_points(x, y, self.v_frames[idx])
            u1 = self._idw_at_points(x, y, self.u_frames[idx + 1])
            v1 = self._idw_at_points(x, y, self.v_frames[idx + 1])
            u = u0 * (1 - alpha) + u1 * alpha
            v = v0 * (1 - alpha) + v1 * alpha

        if x.size == 1:
            return float(u[0]), float(v[0])
        return u, v

    def query_grid(self, xs, ys, elapsed_s=0.0):
        """Query on a meshgrid and return 2-d (u, v) arrays."""
        xx, yy = np.meshgrid(xs, ys)
        xf = xx.ravel()
        yf = yy.ravel()
        u, v = self.query(xf, yf, elapsed_s)
        return u.reshape(xx.shape), v.reshape(xx.shape)


# ===================================================================
#  BoatModel -- boat speed through water
# ===================================================================

class BoatModel:
    """Boat performance model.

    Without a polar: constant speed regardless of heading or wind.
    With a polar: speed depends on True Wind Angle and True Wind Speed.

    Parameters
    ----------
    base_speed_knots : float
        Fallback speed used when no polar is provided.
    polar : PolarTable, optional
        If given, speed is determined by polar lookup.
    """

    def __init__(self, base_speed_knots=6.0, polar=None):
        self.base_speed_knots = base_speed_knots
        self.base_speed_ms = base_speed_knots * KNOTS_TO_MS
        self.polar = polar

    def speed(self, heading_rad=None, wind_u=None, wind_v=None):
        """Return boat speed through water in m/s.

        Parameters
        ----------
        heading_rad : float, optional
            Boat heading in radians, measured CCW from east (math convention,
            matching UTM atan2 frame).  Required when using a polar.
        wind_u, wind_v : float, optional
            Wind velocity components in m/s (where wind blows TO).
            Required when using a polar.

        Returns
        -------
        float (m/s)
        """
        if self.polar is None or heading_rad is None or wind_u is None:
            return self.base_speed_ms

        twa = compute_twa(heading_rad, wind_u, wind_v)
        tws_kt = np.hypot(wind_u, wind_v) * MS_TO_KNOTS
        return self.polar.speed_ms(twa, tws_kt)


def compute_twa(heading_rad, wind_u, wind_v):
    """Compute True Wind Angle in degrees [0, 180].

    Parameters
    ----------
    heading_rad : float
        Boat heading in radians, CCW from east (math/UTM convention).
    wind_u, wind_v : float
        Wind velocity in m/s, where wind blows TO (eastward, northward).

    Returns
    -------
    float : TWA in degrees [0, 180].
    """
    wind_spd = np.hypot(wind_u, wind_v)
    if wind_spd < 1e-6:
        return 90.0

    # Direction wind blows FROM (opposite of (wu, wv) vector)
    wind_from_rad = np.arctan2(-wind_v, -wind_u)

    # Difference between heading and wind-from direction
    delta = heading_rad - wind_from_rad
    # Normalise to [-pi, pi]
    delta = (delta + np.pi) % (2 * np.pi) - np.pi
    # TWA is the absolute angle [0, 180]
    return float(np.degrees(np.abs(delta)))


# ===================================================================
#  Route -- result container
# ===================================================================

@dataclass
class Route:
    waypoints_utm: list          # [(x, y), ...] in UTM metres
    waypoints_latlon: list       # [(lat, lon), ...]
    leg_times_s: list            # seconds for each leg
    leg_distances_m: list        # metres for each leg
    total_time_s: float = 0.0
    total_distance_m: float = 0.0
    avg_sog_knots: float = 0.0
    boat_speed_knots: float = 0.0
    nodes_explored: int = 0
    simulated_track: list = None  # [(x, y), ...] actual ground track

    def summary(self):
        lines = [
            "Route Summary",
            "=============",
            f"  Legs:             {len(self.leg_times_s)}",
            f"  Total distance:   {self.total_distance_m / 1852:.2f} nm "
            f"({self.total_distance_m / 1000:.2f} km)",
            f"  Total time:       {self.total_time_s / 3600:.2f} hr "
            f"({self.total_time_s / 60:.1f} min)",
            f"  Avg SOG:          {self.avg_sog_knots:.2f} kt",
            f"  Boat STW:         {self.boat_speed_knots:.2f} kt",
            f"  SOG advantage:    {self.avg_sog_knots - self.boat_speed_knots:+.2f} kt",
            f"  Nodes explored:   {self.nodes_explored}",
        ]
        return "\n".join(lines)


# ===================================================================
#  Heading sweep -- polar + current physics
# ===================================================================

# Pre-computed heading unit vectors at 2-degree resolution.
_SWEEP_DEGS = np.arange(0, 360, 2, dtype=np.float64)
_SWEEP_RADS = np.radians(_SWEEP_DEGS)
_SWEEP_HX = np.cos(_SWEEP_RADS)   # east component (math frame)
_SWEEP_HY = np.sin(_SWEEP_RADS)   # north component


def _solve_heading(d_hat_x, d_hat_y, cu, cv, wind_u, wind_v,
                   boat_model, drift_tol=0.10):
    """Find the best speed-over-ground along a desired track.

    Sweeps candidate headings at 2-degree resolution and returns the
    maximum SOG achievable along the track direction, accounting for
    current and polar-based boat speed.

    Parameters
    ----------
    d_hat_x, d_hat_y : float
        Unit vector of desired ground track (east, north).
    cu, cv : float
        Current velocity (m/s, east/north).
    wind_u, wind_v : float
        Wind velocity (m/s, where wind blows TO).
    boat_model : BoatModel
        Provides speed(heading_rad, wu, wv) -> m/s.
    drift_tol : float
        Maximum allowed |cross-track / along-track| ratio.

    Returns
    -------
    float : best SOG in m/s along d_hat.  Returns 0.0 if impassable.
    """
    tws_kt = np.hypot(wind_u, wind_v) * MS_TO_KNOTS

    # Vectorised sweep over all candidate headings
    if boat_model.polar is not None and tws_kt > 1e-3:
        # Direction wind blows FROM
        wind_from_rad = np.arctan2(-wind_v, -wind_u)
        delta = _SWEEP_RADS - wind_from_rad
        delta = (delta + np.pi) % (2 * np.pi) - np.pi
        twa_arr = np.degrees(np.abs(delta))

        # Vectorised bilinear lookup
        twa_c = np.clip(twa_arr, boat_model.polar._twas[0],
                        boat_model.polar._twas[-1])
        tws_c = float(np.clip(tws_kt, boat_model.polar._twss[0],
                               boat_model.polar._twss[-1]))

        twas = boat_model.polar._twas
        twss = boat_model.polar._twss
        speeds = boat_model.polar._speeds

        i0 = np.searchsorted(twas, twa_c, side='right') - 1
        i0 = np.clip(i0, 0, len(twas) - 2)
        i1 = i0 + 1

        j0 = int(np.searchsorted(twss, tws_c, side='right')) - 1
        j0 = int(np.clip(j0, 0, len(twss) - 2))
        j1 = j0 + 1

        ta0 = twas[i0]
        ta1 = twas[i1]
        ts0 = twss[j0]
        ts1 = twss[j1]

        alpha = np.where(ta1 > ta0, (twa_c - ta0) / (ta1 - ta0), 0.0)
        beta = float((tws_c - ts0) / (ts1 - ts0)) if ts1 > ts0 else 0.0

        s00 = speeds[i0, j0]
        s10 = speeds[i1, j0]
        s01 = speeds[i0, j1]
        s11 = speeds[i1, j1]

        V_kt = ((1 - alpha) * (1 - beta) * s00 +
                alpha       * (1 - beta) * s10 +
                (1 - alpha) * beta       * s01 +
                alpha       * beta       * s11)
        V = V_kt * KNOTS_TO_MS
    else:
        V = np.full(len(_SWEEP_HX), boat_model.base_speed_ms)

    # Ground velocity = boat velocity + current
    gx = V * _SWEEP_HX + cu
    gy = V * _SWEEP_HY + cv

    # Along-track and cross-track components
    sog = gx * d_hat_x + gy * d_hat_y
    drift = np.abs(gx * (-d_hat_y) + gy * d_hat_x)

    # Heading must make positive progress and cross-track drift must be
    # within tolerance relative to along-track speed
    progress = np.maximum(sog, 1e-6)
    accept = (sog > 0.01) & (drift <= drift_tol * progress)

    if not np.any(accept):
        return 0.0

    return float(np.max(sog[accept]))


def _solve_heading_full(d_hat_x, d_hat_y, cu, cv, wind_u, wind_v,
                        boat_model, drift_tol=0.10):
    """Like _solve_heading but returns (sog, gvx, gvy, heading_rad).

    gvx/gvy are the *actual* ground velocity of the best heading --
    including cross-track components from current and sailing angle.
    This is the velocity that determines the true ground track.

    Returns (0, 0, 0, 0) if impassable.
    """
    tws_kt = np.hypot(wind_u, wind_v) * MS_TO_KNOTS

    if boat_model.polar is not None and tws_kt > 1e-3:
        wind_from_rad = np.arctan2(-wind_v, -wind_u)
        delta = _SWEEP_RADS - wind_from_rad
        delta = (delta + np.pi) % (2 * np.pi) - np.pi
        twa_arr = np.degrees(np.abs(delta))

        twa_c = np.clip(twa_arr, boat_model.polar._twas[0],
                        boat_model.polar._twas[-1])
        tws_c = float(np.clip(tws_kt, boat_model.polar._twss[0],
                               boat_model.polar._twss[-1]))

        twas = boat_model.polar._twas
        twss = boat_model.polar._twss
        speeds = boat_model.polar._speeds

        i0 = np.searchsorted(twas, twa_c, side='right') - 1
        i0 = np.clip(i0, 0, len(twas) - 2)
        i1 = i0 + 1

        j0 = int(np.searchsorted(twss, tws_c, side='right')) - 1
        j0 = int(np.clip(j0, 0, len(twss) - 2))
        j1 = j0 + 1

        ta0 = twas[i0]
        ta1 = twas[i1]
        ts0 = twss[j0]
        ts1 = twss[j1]

        alpha = np.where(ta1 > ta0, (twa_c - ta0) / (ta1 - ta0), 0.0)
        beta = float((tws_c - ts0) / (ts1 - ts0)) if ts1 > ts0 else 0.0

        s00 = speeds[i0, j0]
        s10 = speeds[i1, j0]
        s01 = speeds[i0, j1]
        s11 = speeds[i1, j1]

        V_kt = ((1 - alpha) * (1 - beta) * s00 +
                alpha       * (1 - beta) * s10 +
                (1 - alpha) * beta       * s01 +
                alpha       * beta       * s11)
        V = V_kt * KNOTS_TO_MS
    else:
        V = np.full(len(_SWEEP_HX), boat_model.base_speed_ms)

    gx = V * _SWEEP_HX + cu
    gy = V * _SWEEP_HY + cv

    sog = gx * d_hat_x + gy * d_hat_y
    drift = np.abs(gx * (-d_hat_y) + gy * d_hat_x)

    progress = np.maximum(sog, 1e-6)
    accept = (sog > 0.01) & (drift <= drift_tol * progress)

    if not np.any(accept):
        return 0.0, 0.0, 0.0, 0.0

    best_idx = int(np.argmax(np.where(accept, sog, -np.inf)))
    return (float(sog[best_idx]),
            float(gx[best_idx]),
            float(gy[best_idx]),
            float(_SWEEP_RADS[best_idx]))


# ===================================================================
#  Router -- time-dependent A* on a regular grid
# ===================================================================

class Router:
    """Find the time-optimal route through a current field.

    Overlays a regular UTM grid on the bounding box of the start/end
    points (with padding), pre-screens for land, and runs A* with
    time-dependent edge costs.
    """

    NEIGHBOR_OFFSETS = [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),          ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1),
    ]

    # Precomputed angle between every pair of the 8 grid directions (degrees).
    # Index 8 is a sentinel for "no incoming direction" (start node).
    # For direction i: vector is (dc_i, dr_i) in (east, north) UTM coords.
    _N_DIRS = 9  # 0-7 grid directions + 8 start sentinel
    _DIR_ANGLES = np.array([
        np.degrees(np.arctan2(dc, dr))
        for dr, dc in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    ])
    _ANGLE_DIFF = np.zeros((8, 8))
    for _i in range(8):
        for _j in range(8):
            _d = abs(_DIR_ANGLES[_i] - _DIR_ANGLES[_j]) % 360.0
            _ANGLE_DIFF[_i, _j] = min(_d, 360.0 - _d)

    def __init__(self, current_field: CurrentField, boat: BoatModel,
                 resolution_m: float = 300.0, padding_m: float = 5000.0,
                 wind: 'WindField | None' = None,
                 tack_penalty_s: float = 60.0,
                 tack_threshold_deg: float = 90.0):
        self.cf = current_field
        self.boat = boat
        self.resolution = resolution_m
        self.padding = padding_m
        self.wind = wind
        self.tack_penalty_s = tack_penalty_s
        self.tack_threshold_deg = tack_threshold_deg

    @property
    def _use_polar(self):
        return self.wind is not None and self.boat.polar is not None

    # ------------------------------------------------------------------
    #  Grid construction
    # ------------------------------------------------------------------

    def _build_grid(self, start_xy, end_xy):
        """Build a regular grid covering start/end with padding.

        Returns (xs, ys, water_mask, u_grid, v_grid) where xs/ys are
        1-d coordinate arrays and the grids are (ny, nx).
        """
        x0 = min(start_xy[0], end_xy[0]) - self.padding
        x1 = max(start_xy[0], end_xy[0]) + self.padding
        y0 = min(start_xy[1], end_xy[1]) - self.padding
        y1 = max(start_xy[1], end_xy[1]) + self.padding

        xs = np.arange(x0, x1 + self.resolution, self.resolution)
        ys = np.arange(y0, y1 + self.resolution, self.resolution)

        u_grid, v_grid = self.cf.query_grid(xs, ys, elapsed_s=0.0)
        water_mask = ~np.isnan(u_grid)

        return xs, ys, water_mask, u_grid, v_grid

    def _snap_to_grid(self, xy, xs, ys):
        """Return the (row, col) index of the nearest grid cell."""
        col = int(np.argmin(np.abs(xs - xy[0])))
        row = int(np.argmin(np.abs(ys - xy[1])))
        return row, col

    # ------------------------------------------------------------------
    #  Line-of-sight and path smoothing
    # ------------------------------------------------------------------

    @staticmethod
    def _cells_along_line(r0, c0, r1, c1):
        """Yield every grid cell (r, c) that a line from (r0,c0) to (r1,c1)
        passes through, using a Bresenham-like walk."""
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r1 > r0 else -1
        sc = 1 if c1 > c0 else -1
        err = dr - dc
        r, c = r0, c0
        while True:
            yield r, c
            if r == r1 and c == c1:
                break
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dr
                c += sc

    def _line_of_sight(self, r0, c0, r1, c1, water_mask):
        """Return True if every grid cell on the straight line between
        (r0,c0) and (r1,c1) is water."""
        for r, c in self._cells_along_line(r0, c0, r1, c1):
            if not water_mask[r, c]:
                return False
        return True

    def _segment_travel_time(self, x0, y0, x1, y1, start_time_s,
                             n_samples=None):
        """Compute travel time (s) along an arbitrary straight segment
        by sampling currents (and wind if available) at multiple points.

        Returns (time_s, dist_m).  time_s is np.inf if impassable.
        """
        dx = x1 - x0
        dy = y1 - y0
        dist = np.hypot(dx, dy)
        if dist < 1e-6:
            return 0.0, 0.0

        d_hat_x = dx / dist
        d_hat_y = dy / dist

        if n_samples is None:
            n_samples = max(2, int(np.ceil(dist / self.resolution)))

        seg_len = dist / n_samples
        elapsed = start_time_s
        total_time = 0.0

        for i in range(n_samples):
            frac0 = i / n_samples
            frac1 = (i + 1) / n_samples
            mx = x0 + 0.5 * (frac0 + frac1) * dx
            my = y0 + 0.5 * (frac0 + frac1) * dy

            cu, cv = self.cf.query(mx, my, elapsed_s=elapsed)
            if np.isnan(cu) or np.isnan(cv):
                return np.inf, dist

            if self._use_polar:
                wu, wv = self.wind.query(mx, my, elapsed_s=elapsed)
                v_sog = _solve_heading(d_hat_x, d_hat_y, cu, cv,
                                       wu, wv, self.boat)
                if v_sog <= 0.01:
                    return np.inf, dist
            else:
                boat_speed = self.boat.speed()
                c_par = cu * d_hat_x + cv * d_hat_y
                c_perp_x = cu - c_par * d_hat_x
                c_perp_y = cv - c_par * d_hat_y
                c_perp_mag = np.hypot(c_perp_x, c_perp_y)
                if c_perp_mag >= boat_speed:
                    return np.inf, dist
                v_water_along = np.sqrt(boat_speed**2 - c_perp_mag**2)
                v_sog = v_water_along + c_par
                if v_sog <= 0.01:
                    return np.inf, dist

            dt = seg_len / v_sog
            total_time += dt
            elapsed += dt

        return total_time, dist

    def _smooth_path(self, path_rc, water_mask, xs, ys, arrival_time):
        """Remove unnecessary waypoints via greedy string-pulling.

        A shortcut from waypoint i to waypoint j is accepted only when:
        1. All grid cells on the straight line are water (line-of-sight).
        2. The smoothed segment travel time is no worse than the sum of
           the original grid-path segment times it replaces.  This
           prevents collapsing beneficial detours into slower straight lines.

        Returns a new (shorter) list of (row, col) tuples.
        """
        if len(path_rc) <= 2:
            return path_rc

        # Pre-compute cumulative travel time along the raw grid path
        # so we can quickly get the cost of any sub-path [i..j].
        n = len(path_rc)
        cum_time = np.zeros(n)
        for k in range(1, n):
            rk0, ck0 = path_rc[k - 1]
            rk1, ck1 = path_rc[k]
            x0, y0 = xs[ck0], ys[rk0]
            x1, y1 = xs[ck1], ys[rk1]
            t_k, _ = self._segment_travel_time(
                x0, y0, x1, y1, arrival_time[rk0, ck0])
            cum_time[k] = cum_time[k - 1] + t_k

        smoothed = [path_rc[0]]
        i = 0
        while i < n - 1:
            best_j = i + 1
            for j in range(n - 1, i + 1, -1):
                if not self._line_of_sight(
                    path_rc[i][0], path_rc[i][1],
                    path_rc[j][0], path_rc[j][1],
                    water_mask,
                ):
                    continue

                r0, c0 = path_rc[i]
                rj, cj = path_rc[j]
                x0, y0 = xs[c0], ys[r0]
                xj, yj = xs[cj], ys[rj]
                t_shortcut, _ = self._segment_travel_time(
                    x0, y0, xj, yj, arrival_time[r0, c0])

                if t_shortcut >= np.inf:
                    continue

                t_grid = cum_time[j] - cum_time[i]
                if t_shortcut <= t_grid * 1.005:
                    best_j = j
                    break

            smoothed.append(path_rc[best_j])
            i = best_j

        return smoothed

    @staticmethod
    def _remove_stubs(path_rc, xs, ys):
        """Remove stub waypoints that create Y-junctions.

        A stub is an intermediate waypoint where one adjacent leg is
        much shorter than the other and the turn angle is large.
        Removing the stub lets the route go directly from the previous
        to the next waypoint, eliminating the visual and navigational
        artifact.  The check is purely geometric (no time comparison)
        so it runs fast.
        """
        if len(path_rc) <= 2:
            return path_rc

        changed = True
        while changed:
            changed = False
            new_path = [path_rc[0]]
            i = 1
            while i < len(path_rc) - 1:
                ax, ay = xs[path_rc[i-1][1]], ys[path_rc[i-1][0]]
                bx, by = xs[path_rc[i][1]],   ys[path_rc[i][0]]
                cx, cy = xs[path_rc[i+1][1]], ys[path_rc[i+1][0]]

                leg_in = np.hypot(bx - ax, by - ay)
                leg_out = np.hypot(cx - bx, cy - by)
                short_leg = min(leg_in, leg_out)
                long_leg = max(leg_in, leg_out)

                # Only consider stubs where one leg is much shorter
                if short_leg > 0 and long_leg / short_leg > 3.0:
                    v1x, v1y = bx - ax, by - ay
                    v2x, v2y = cx - bx, cy - by
                    dot = v1x * v2x + v1y * v2y
                    cross = v1x * v2y - v1y * v2x
                    turn = abs(np.degrees(np.arctan2(cross, dot)))
                    if turn > 45.0:
                        changed = True
                        i += 1
                        continue

                new_path.append(path_rc[i])
                i += 1
            new_path.append(path_rc[-1])
            path_rc = new_path

        return path_rc

    # ------------------------------------------------------------------
    #  Edge cost
    # ------------------------------------------------------------------

    def _edge_cost(self, r1, c1, r2, c2, xs, ys, arrival_time_s):
        """Compute travel time (seconds) from grid cell (r1,c1) to (r2,c2).

        Returns np.inf if the leg is impassable.
        """
        x1, y1 = xs[c1], ys[r1]
        x2, y2 = xs[c2], ys[r2]

        dx = x2 - x1
        dy = y2 - y1
        dist = np.hypot(dx, dy)
        if dist < 1e-6:
            return 0.0

        d_hat_x = dx / dist
        d_hat_y = dy / dist

        mx = 0.5 * (x1 + x2)
        my = 0.5 * (y1 + y2)
        cu, cv = self.cf.query(mx, my, elapsed_s=arrival_time_s)
        if np.isnan(cu) or np.isnan(cv):
            return np.inf

        if self._use_polar:
            wu, wv = self.wind.query(mx, my, elapsed_s=arrival_time_s)
            v_sog = _solve_heading(d_hat_x, d_hat_y, cu, cv,
                                   wu, wv, self.boat)
            if v_sog <= 0.01:
                return np.inf
            return dist / v_sog

        boat_speed = self.boat.speed()
        c_par = cu * d_hat_x + cv * d_hat_y
        c_perp_x = cu - c_par * d_hat_x
        c_perp_y = cv - c_par * d_hat_y
        c_perp_mag = np.hypot(c_perp_x, c_perp_y)

        if c_perp_mag >= boat_speed:
            return np.inf

        v_water_along = np.sqrt(boat_speed**2 - c_perp_mag**2)
        v_sog = v_water_along + c_par

        if v_sog <= 0.01:
            return np.inf

        return dist / v_sog

    # ------------------------------------------------------------------
    #  A* search
    # ------------------------------------------------------------------

    def find_route(self, start_latlon, end_latlon, start_time_s=0.0):
        """Find the time-optimal route between two lat/lon points.

        Parameters
        ----------
        start_latlon : (lat, lon)
        end_latlon : (lat, lon)
        start_time_s : float
            Elapsed seconds from the forecast reference time at departure.

        Returns
        -------
        Route
        """
        t_wall_0 = _time.monotonic()

        transformer = self.cf.transformer
        sx, sy = transformer.transform(start_latlon[1], start_latlon[0])
        ex, ey = transformer.transform(end_latlon[1], end_latlon[0])
        start_xy = (sx, sy)
        end_xy = (ex, ey)

        xs, ys, water_mask, _, _ = self._build_grid(start_xy, end_xy)
        ny, nx = water_mask.shape
        print(f"Grid: {nx} x {ny} = {nx * ny} cells, "
              f"resolution {self.resolution:.0f} m")
        print(f"Water cells: {water_mask.sum()} "
              f"({100 * water_mask.sum() / water_mask.size:.1f}%)")

        sr, sc = self._snap_to_grid(start_xy, xs, ys)
        er, ec = self._snap_to_grid(end_xy, xs, ys)

        if not water_mask[sr, sc]:
            raise ValueError(
                f"Start point snapped to land at grid ({sr},{sc}).  "
                "Try a location farther from shore.")
        if not water_mask[er, ec]:
            raise ValueError(
                f"End point snapped to land at grid ({er},{ec}).  "
                "Try a location farther from shore.")

        print(f"Start grid cell: ({sr}, {sc})  ->  "
              f"End grid cell: ({er}, {ec})")

        goal_x, goal_y = xs[ec], ys[er]
        boat_speed = self.boat.speed()
        if self._use_polar:
            max_boat_speed = self.boat.polar.max_speed_ms
        else:
            max_boat_speed = boat_speed
        max_speed = max_boat_speed + self.cf.max_current_speed

        def heuristic(r, c):
            dx = xs[c] - goal_x
            dy = ys[r] - goal_y
            return np.hypot(dx, dy) / max_speed

        INF = float('inf')
        # State: (row, col, dir_idx) where dir_idx 0-7 = NEIGHBOR_OFFSETS index,
        # and dir_idx=8 is the start sentinel (no incoming direction).
        N_DIRS = self._N_DIRS  # 9
        best_cost = np.full((ny, nx, N_DIRS), INF)
        best_cost[sr, sc, 8] = 0.0
        # came_from stores (prev_r, prev_c, prev_dir) for each state
        came_from = np.full((ny, nx, N_DIRS, 3), -1, dtype=np.int32)
        arrival_time = np.full((ny, nx, N_DIRS), INF)
        arrival_time[sr, sc, 8] = start_time_s

        # priority queue: (estimated_total, cost_so_far, row, col, dir_idx)
        open_set = [(heuristic(sr, sc), 0.0, sr, sc, 8)]
        explored = 0

        while open_set:
            est_total, cost, r, c, d_in = heapq.heappop(open_set)

            if r == er and c == ec:
                break

            if cost > best_cost[r, c, d_in]:
                continue

            explored += 1
            arr_t = arrival_time[r, c, d_in]

            for d_out, (dr, dc) in enumerate(self.NEIGHBOR_OFFSETS):
                nr, nc_ = r + dr, c + dc
                if nr < 0 or nr >= ny or nc_ < 0 or nc_ >= nx:
                    continue
                if not water_mask[nr, nc_]:
                    continue

                dt = self._edge_cost(r, c, nr, nc_, xs, ys, arr_t)
                if dt == INF:
                    continue

                # Tacking penalty: course change beyond threshold costs time
                penalty = 0.0
                if (d_in < 8 and self.tack_penalty_s > 0
                        and self._ANGLE_DIFF[d_in, d_out]
                        > self.tack_threshold_deg):
                    penalty = self.tack_penalty_s

                new_cost = cost + dt + penalty
                if new_cost < best_cost[nr, nc_, d_out]:
                    best_cost[nr, nc_, d_out] = new_cost
                    arrival_time[nr, nc_, d_out] = arr_t + dt + penalty
                    came_from[nr, nc_, d_out, 0] = r
                    came_from[nr, nc_, d_out, 1] = c
                    came_from[nr, nc_, d_out, 2] = d_in
                    est = new_cost + heuristic(nr, nc_)
                    heapq.heappush(open_set, (est, new_cost, nr, nc_, d_out))

        if np.min(best_cost[er, ec, :]) == INF:
            raise RuntimeError("No route found -- end point is unreachable.")

        # Reconstruct raw grid path by following came_from backwards
        final_dir = int(np.argmin(best_cost[er, ec, :]))
        raw_path_rc = []
        r, c, d = er, ec, final_dir
        while r != -1:
            raw_path_rc.append((r, c))
            pr = int(came_from[r, c, d, 0])
            pc = int(came_from[r, c, d, 1])
            pd = int(came_from[r, c, d, 2])
            r, c, d = pr, pc, pd
        raw_path_rc.reverse()

        # Collapse arrival_time to 2D (minimum over directions) for smooth_path
        arrival_time_2d = np.min(arrival_time, axis=2)

        # Smooth: remove unnecessary intermediate waypoints
        path_rc = self._smooth_path(
            raw_path_rc, water_mask, xs, ys, arrival_time_2d)

        # Remove short stub legs that create Y-junctions
        path_rc = self._remove_stubs(path_rc, xs, ys)

        inv_transformer = Transformer.from_crs(
            transformer.target_crs, transformer.source_crs, always_xy=True)

        waypoints_utm = [(xs[c], ys[r]) for r, c in path_rc]
        waypoints_latlon = []
        for x, y in waypoints_utm:
            lon, lat = inv_transformer.transform(x, y)
            waypoints_latlon.append((lat, lon))

        leg_times = []
        leg_dists = []
        cum_time = start_time_s
        for i in range(len(waypoints_utm) - 1):
            x1, y1 = waypoints_utm[i]
            x2, y2 = waypoints_utm[i + 1]
            t_seg, d_seg = self._segment_travel_time(
                x1, y1, x2, y2, cum_time)
            leg_times.append(t_seg)
            leg_dists.append(d_seg)
            cum_time += t_seg

        total_time = sum(leg_times)
        total_dist = sum(leg_dists)
        avg_sog = (total_dist / total_time * MS_TO_KNOTS) if total_time > 0 else 0.0

        n_raw = len(raw_path_rc)
        n_smooth = len(path_rc)

        sim_track = self.simulate_track(waypoints_utm, start_time_s)

        route = Route(
            waypoints_utm=waypoints_utm,
            waypoints_latlon=waypoints_latlon,
            leg_times_s=leg_times,
            leg_distances_m=leg_dists,
            total_time_s=total_time,
            total_distance_m=total_dist,
            avg_sog_knots=avg_sog,
            boat_speed_knots=self.boat.base_speed_knots,
            nodes_explored=explored,
            simulated_track=sim_track,
        )

        wall_s = _time.monotonic() - t_wall_0
        print(f"\nA* completed in {wall_s:.2f}s, explored {explored} nodes")
        print(f"Path smoothing: {n_raw} -> {n_smooth} waypoints")
        print(route.summary())
        return route, xs, ys, water_mask

    # ------------------------------------------------------------------
    #  Straight-line baseline
    # ------------------------------------------------------------------

    def straight_line_time(self, start_latlon, end_latlon, start_time_s=0.0,
                           n_samples=50):
        """Estimate travel time along a straight rhumb line, accounting
        for currents (and wind if available) sampled along the way."""
        transformer = self.cf.transformer
        sx, sy = transformer.transform(start_latlon[1], start_latlon[0])
        ex, ey = transformer.transform(end_latlon[1], end_latlon[0])

        total_dist = np.hypot(ex - sx, ey - sy)
        if total_dist < 1e-3:
            return 0.0, total_dist

        fracs = np.linspace(0, 1, n_samples + 1)
        xs_line = sx + fracs * (ex - sx)
        ys_line = sy + fracs * (ey - sy)

        dx = ex - sx
        dy = ey - sy
        d_hat_x = dx / total_dist
        d_hat_y = dy / total_dist

        seg_len = total_dist / n_samples
        elapsed = start_time_s
        total_time = 0.0

        for i in range(n_samples):
            mx = 0.5 * (xs_line[i] + xs_line[i + 1])
            my = 0.5 * (ys_line[i] + ys_line[i + 1])
            cu, cv = self.cf.query(mx, my, elapsed_s=elapsed)
            if np.isnan(cu):
                return np.inf, total_dist

            if self._use_polar:
                wu, wv = self.wind.query(mx, my, elapsed_s=elapsed)
                v_sog = _solve_heading(d_hat_x, d_hat_y, cu, cv,
                                       wu, wv, self.boat)
                if v_sog <= 0.01:
                    return np.inf, total_dist
            else:
                boat_speed = self.boat.speed()
                c_par = cu * d_hat_x + cv * d_hat_y
                c_perp_x = cu - c_par * d_hat_x
                c_perp_y = cv - c_par * d_hat_y
                c_perp_mag = np.hypot(c_perp_x, c_perp_y)
                if c_perp_mag >= boat_speed:
                    return np.inf, total_dist
                v_water_along = np.sqrt(boat_speed**2 - c_perp_mag**2)
                v_sog = v_water_along + c_par
                if v_sog <= 0.01:
                    return np.inf, total_dist

            dt = seg_len / v_sog
            total_time += dt
            elapsed += dt

        return total_time, total_dist

    # ------------------------------------------------------------------
    #  Ground-track simulation
    # ------------------------------------------------------------------

    def simulate_track(self, waypoints_utm, start_time_s=0.0, dt_s=10.0):
        """Forward-integrate the actual ground track along the route.

        On each leg the boat aims toward the next waypoint, picks the
        heading that maximises along-track SOG (via _solve_heading_full),
        and then advances by the *full* ground velocity vector (including
        cross-track drift from current).  The result is a dense polyline
        that curves realistically with the current.

        Parameters
        ----------
        waypoints_utm : list[(x,y)]
            Waypoints in UTM metres (from Route.waypoints_utm).
        start_time_s : float
            Elapsed seconds at departure for current look-up.
        dt_s : float
            Integration time step in seconds (smaller = smoother).

        Returns
        -------
        list[(x, y)] : simulated ground track positions.
        """
        if len(waypoints_utm) < 2:
            return list(waypoints_utm)

        track = []
        elapsed = start_time_s
        px, py = waypoints_utm[0]
        track.append((px, py))

        for wp_idx in range(1, len(waypoints_utm)):
            tx, ty = waypoints_utm[wp_idx]

            max_iters = 200_000
            for _ in range(max_iters):
                dx = tx - px
                dy = ty - py
                remaining = np.hypot(dx, dy)

                if remaining < 5.0:
                    px, py = tx, ty
                    track.append((px, py))
                    break

                d_hat_x = dx / remaining
                d_hat_y = dy / remaining

                cu, cv = self.cf.query(px, py, elapsed_s=elapsed)
                if np.isnan(cu) or np.isnan(cv):
                    cu, cv = 0.0, 0.0

                if self._use_polar:
                    wu, wv = self.wind.query(px, py, elapsed_s=elapsed)
                    _, gvx, gvy, _ = _solve_heading_full(
                        d_hat_x, d_hat_y, cu, cv, wu, wv, self.boat)
                else:
                    boat_speed = self.boat.speed()
                    c_perp_x = cu - (cu * d_hat_x + cv * d_hat_y) * d_hat_x
                    c_perp_y = cv - (cu * d_hat_x + cv * d_hat_y) * d_hat_y
                    c_perp_mag = np.hypot(c_perp_x, c_perp_y)
                    if c_perp_mag >= boat_speed:
                        gvx, gvy = cu, cv
                    else:
                        v_along = np.sqrt(boat_speed**2 - c_perp_mag**2)
                        head_x = v_along * d_hat_x - c_perp_x
                        head_y = v_along * d_hat_y - c_perp_y
                        h_mag = np.hypot(head_x, head_y)
                        if h_mag > 1e-9:
                            head_x = head_x / h_mag * boat_speed
                            head_y = head_y / h_mag * boat_speed
                        gvx = head_x + cu
                        gvy = head_y + cv

                spd = np.hypot(gvx, gvy)
                if spd < 0.01:
                    track.append((tx, ty))
                    break

                step = min(dt_s, remaining / spd)
                px += gvx * step
                py += gvy * step
                elapsed += step
                track.append((px, py))
            else:
                track.append((tx, ty))

        return track


# ===================================================================
#  Visualization
# ===================================================================

def plot_route(route, xs, ys, water_mask, current_field,
               start_latlon, end_latlon,
               straight_time_s=None, straight_dist_m=None,
               save_path=None, subsample_arrows=3, show=True):
    """Plot the optimised route over the current field."""
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    import matplotlib.colors as mcolors

    transformer = current_field.transformer

    u_grid, v_grid = current_field.query_grid(xs, ys, elapsed_s=0.0)
    speed_grid = np.sqrt(u_grid**2 + v_grid**2) * MS_TO_KNOTS
    speed_grid[~water_mask] = np.nan

    fig, ax = plt.subplots(figsize=(14, 10))

    xx, yy = np.meshgrid(xs, ys)

    # Land / water background
    land_color = '#f0e6d3'
    ax.set_facecolor(land_color)
    water_bg = np.where(water_mask, 0.0, np.nan)
    ax.pcolormesh(xs, ys, water_bg, cmap='Greys', vmin=0, vmax=1,
                  alpha=0.05, shading='auto', zorder=0)

    # Current speed heatmap (water only)
    speed_max = np.nanmax(speed_grid) if np.any(~np.isnan(speed_grid)) else 1.0
    im = ax.pcolormesh(xs, ys, speed_grid,
                       cmap='cividis', alpha=0.35, shading='auto',
                       vmin=0, vmax=max(speed_max, 0.5), zorder=1)
    cbar = fig.colorbar(im, ax=ax, label='Current speed (knots)',
                        pad=0.02, shrink=0.8)

    # Current direction arrows -- normalised to uniform length,
    # colored by speed for magnitude, black outlines for contrast.
    step = subsample_arrows
    u_sub = u_grid[::step, ::step].copy()
    v_sub = v_grid[::step, ::step].copy()
    spd_sub = speed_grid[::step, ::step].copy()
    xx_sub = xx[::step, ::step]
    yy_sub = yy[::step, ::step]

    mag = np.sqrt(u_sub**2 + v_sub**2)
    mag_safe = np.where(mag < 1e-8, 1.0, mag)
    u_norm = u_sub / mag_safe
    v_norm = v_sub / mag_safe

    # Hide arrows where speed is negligible or over land
    visible = (mag > 0.02) & ~np.isnan(spd_sub)
    u_norm[~visible] = np.nan
    v_norm[~visible] = np.nan

    arrow_scale = 1.0 / (step * (xs[1] - xs[0])) * 0.7
    ax.quiver(
        xx_sub, yy_sub, u_norm, v_norm,
        spd_sub,
        cmap='plasma', clim=(0, max(speed_max, 0.5)),
        scale=arrow_scale, scale_units='xy',
        width=0.004, headwidth=4, headlength=5, headaxislength=4,
        alpha=0.85, zorder=3,
        edgecolors='#333333', linewidth=0.3,
    )

    # Shoreline overlay
    shoreline_path = Path(__file__).parent / "data" / "shoreline_puget.geojson"
    if shoreline_path.exists():
        try:
            import geopandas as gpd
            shore = gpd.read_file(shoreline_path)
            for geom in shore.geometry:
                if geom is None:
                    continue
                lines = []
                if geom.geom_type == 'LineString':
                    lines = [geom]
                elif geom.geom_type == 'MultiLineString':
                    lines = list(geom.geoms)
                for line in lines:
                    coords = np.array(line.coords)
                    sx_arr, sy_arr = transformer.transform(
                        coords[:, 0], coords[:, 1])
                    ax.plot(sx_arr, sy_arr, color='#3d3d3d',
                            linewidth=0.7, zorder=2)
        except Exception:
            pass

    # Route: simulated ground track (curved) if available, else waypoints
    if route.simulated_track and len(route.simulated_track) > 1:
        stx = [p[0] for p in route.simulated_track]
        sty = [p[1] for p in route.simulated_track]
        ax.plot(stx, sty, color='#e63946', linewidth=3, zorder=5,
                label='Ground track',
                path_effects=[pe.Stroke(linewidth=5, foreground='white'),
                              pe.Normal()])
        # Tack/waypoint markers (small dots, no connecting lines)
        rx = [p[0] for p in route.waypoints_utm[1:-1]]
        ry = [p[1] for p in route.waypoints_utm[1:-1]]
        if rx:
            ax.plot(rx, ry, 'o', color='#e63946', markersize=5, zorder=6,
                    markeredgecolor='white', markeredgewidth=1,
                    label='Tack points')
    else:
        rx = [p[0] for p in route.waypoints_utm]
        ry = [p[1] for p in route.waypoints_utm]
        ax.plot(rx, ry, color='#e63946', linewidth=3, zorder=5,
                label='Optimal route',
                path_effects=[pe.Stroke(linewidth=5, foreground='white'),
                              pe.Normal()])

    # Straight line
    sx, sy = transformer.transform(start_latlon[1], start_latlon[0])
    ex, ey = transformer.transform(end_latlon[1], end_latlon[0])
    ax.plot([sx, ex], [sy, ey], '--', color='#457b9d', linewidth=2,
            zorder=4, label='Straight line',
            path_effects=[pe.Stroke(linewidth=3.5, foreground='white',
                                    alpha=0.6),
                          pe.Normal()])

    # Markers
    ax.plot(sx, sy, 'o', color='#2a9d8f', markersize=14, zorder=6,
            markeredgecolor='white', markeredgewidth=2.5, label='Start')
    ax.plot(ex, ey, 's', color='#e76f51', markersize=14, zorder=6,
            markeredgecolor='white', markeredgewidth=2.5, label='End')

    # Stats box
    stats_lines = [
        f"Optimal:  {route.total_time_s / 60:.1f} min, "
        f"{route.total_distance_m / 1852:.2f} nm, "
        f"SOG {route.avg_sog_knots:.2f} kt",
    ]
    if straight_time_s is not None and np.isfinite(straight_time_s):
        sl_sog = (straight_dist_m / straight_time_s * MS_TO_KNOTS
                  if straight_time_s > 0 else 0)
        stats_lines.append(
            f"Straight: {straight_time_s / 60:.1f} min, "
            f"{straight_dist_m / 1852:.2f} nm, "
            f"SOG {sl_sog:.2f} kt")
        saved = straight_time_s - route.total_time_s
        stats_lines.append(f"Time saved: {saved / 60:.1f} min")
    stats_lines.append(f"Boat STW: {route.boat_speed_knots:.1f} kt")
    stats_text = "\n".join(stats_lines)
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                      edgecolor='#cccccc'))

    ax.set_xlabel('Easting (m, UTM)')
    ax.set_ylabel('Northing (m, UTM)')
    ax.set_title('Sail Routing -- Time-Optimal Path Through Currents')
    ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15, color='#999999')

    margin = 500
    ax.set_xlim(xs[0] - margin, xs[-1] + margin)
    ax.set_ylim(ys[0] - margin, ys[-1] + margin)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ===================================================================
#  Data loading helpers
# ===================================================================

def load_current_field(depart_dt=None, forecast_hours=None,
                       duration_hours=8, use_cache=True):
    """Load SSCOFS data and build a CurrentField.

    Parameters
    ----------
    depart_dt : datetime, optional
        Departure time (timezone-aware or naive local).  When given the
        loader picks the right model cycle, loads hourly frames covering
        ``depart_dt`` through ``depart_dt + duration_hours``, and returns
        the correct ``start_time_s`` offset.  When *None* the latest
        available frame is loaded and ``start_time_s`` is 0.
    forecast_hours : list[int], optional
        Explicit forecast hour indices to load.  Overrides the automatic
        window computed from *depart_dt*.
    duration_hours : int
        Hours of forecast data to load after *depart_dt* (default 8).
    use_cache : bool
        Use cached NetCDF files when available.

    Returns
    -------
    (CurrentField, transformer, start_time_s, depart_utc)
        start_time_s : seconds from the base (first-loaded) frame to the
            departure time.  Pass this to ``Router.find_route()``.
        depart_utc : datetime or None -- the departure time in UTC.
    """
    import datetime as _dt
    from fetch_sscofs import build_sscofs_url
    from sscofs_cache import load_sscofs_data as _load

    depart_utc = None
    if depart_dt is not None:
        if depart_dt.tzinfo is None:
            from zoneinfo import ZoneInfo
            depart_dt = depart_dt.replace(tzinfo=ZoneInfo("America/Los_Angeles"))
        depart_utc = depart_dt.astimezone(_dt.timezone.utc)

    # Always use get_latest_current_data (which picks the latest
    # *actually-available* cycle) rather than compute_file_for_datetime
    # which may pick a cycle that hasn't been produced yet.
    ds, info = get_latest_current_data(use_cache=use_cache)

    lonc = ds["lonc"].values
    latc = ds["latc"].values
    if lonc.max() > 180:
        lonc = np.where(lonc > 180, lonc - 360, lonc)

    u0 = ds["u"].isel(time=0, siglay=0).values
    v0 = ds["v"].isel(time=0, siglay=0).values
    u0 = np.nan_to_num(u0, nan=0.0)
    v0 = np.nan_to_num(v0, nan=0.0)
    ds.close()

    transformer, _, _ = create_utm_transformer(
        float(np.mean(latc)), float(np.mean(lonc)))

    base_hour = info.get('forecast_hour_index', 0)
    run_date = info['run_date_utc']
    cycle = info['cycle_utc']
    cycle_hour_int = int(cycle.replace('z', ''))

    cycle_start = _dt.datetime.combine(
        _dt.date.fromisoformat(run_date),
        _dt.time(hour=cycle_hour_int),
        tzinfo=_dt.timezone.utc)

    print(f"Model cycle:        {run_date} {cycle}")
    print(f"Base forecast hour: f{base_hour:03d} "
          f"({cycle_start + _dt.timedelta(hours=base_hour):%Y-%m-%d %H:%M UTC})")

    # Auto-compute forecast hours if depart_dt given and no explicit override
    if depart_utc is not None and forecast_hours is None:
        depart_fh = (depart_utc - cycle_start).total_seconds() / 3600.0
        fh_start = max(0, int(depart_fh) - 1)
        fh_end = int(depart_fh) + duration_hours + 1
        forecast_hours = list(range(fh_start, fh_end + 1))
        print(f"Auto forecast hrs:  f{fh_start:03d} .. f{fh_end:03d} "
              f"({len(forecast_hours)} frames)")

    if depart_utc is not None:
        depart_local = depart_utc.astimezone(
            _dt.timezone(_dt.timedelta(
                seconds=depart_dt.utcoffset().total_seconds())))
        print(f"Departure (local):  {depart_local:%Y-%m-%d %H:%M}")
        print(f"Departure (UTC):    {depart_utc:%Y-%m-%d %H:%M UTC}")

    u_frames = [u0]
    v_frames = [v0]
    frame_times = [0.0]

    if forecast_hours and len(forecast_hours) > 1:
        for fh in forecast_hours:
            if fh == base_hour:
                continue
            extra_info = {
                'run_date_utc': run_date,
                'cycle_utc': cycle,
                'forecast_hour_index': fh,
                'url': build_sscofs_url(run_date, int(cycle.replace('z', '')), fh),
            }
            try:
                ds_extra = _load(extra_info, use_cache=use_cache, verbose=True)
                ue = ds_extra["u"].isel(time=0, siglay=0).values
                ve = ds_extra["v"].isel(time=0, siglay=0).values
                ue = np.nan_to_num(ue, nan=0.0)
                ve = np.nan_to_num(ve, nan=0.0)
                ds_extra.close()

                u_frames.append(ue)
                v_frames.append(ve)
                frame_times.append(float((fh - base_hour) * 3600))
            except Exception as exc:
                print(f"Warning: could not load forecast hour {fh}: {exc}")

        order = np.argsort(frame_times)
        u_frames = [u_frames[i] for i in order]
        v_frames = [v_frames[i] for i in order]
        frame_times = [frame_times[i] for i in order]

    cf = CurrentField(lonc, latc, u_frames, v_frames, frame_times,
                      transformer)

    # start_time_s: offset from the first loaded frame to the departure.
    start_time_s = 0.0
    if depart_utc is not None:
        base_frame_utc = cycle_start + _dt.timedelta(hours=base_hour)
        start_time_s = (depart_utc - base_frame_utc).total_seconds()
        print(f"Departure offset:   {start_time_s:.0f}s "
              f"({start_time_s / 3600:.2f} hr after base frame)")

    print(f"CurrentField ready: {len(u_frames)} time frame(s), "
          f"{len(lonc)} elements, "
          f"max current {cf.max_current_speed * MS_TO_KNOTS:.2f} kt")
    return cf, transformer, start_time_s, depart_utc


# ===================================================================
#  CLI
# ===================================================================

def main():
    import datetime as _dt
    from zoneinfo import ZoneInfo

    parser = argparse.ArgumentParser(
        description="Sailboat routing through ocean currents")
    parser.add_argument("--start-lat", type=float, required=True,
                        help="Start latitude")
    parser.add_argument("--start-lon", type=float, required=True,
                        help="Start longitude")
    parser.add_argument("--end-lat", type=float, required=True,
                        help="End latitude")
    parser.add_argument("--end-lon", type=float, required=True,
                        help="End longitude")
    parser.add_argument("--boat-speed", type=float, default=6.0,
                        help="Boat speed through water in knots (default: 6)")
    parser.add_argument("--grid-resolution", type=float, default=300.0,
                        help="Routing grid cell size in metres (default: 300)")
    parser.add_argument("--padding", type=float, default=5000.0,
                        help="Grid padding around start/end in metres "
                             "(default: 5000)")
    parser.add_argument("--depart", type=str, default=None,
                        help="Departure time in local time, e.g. "
                             "'2026-03-07 10:30'.  The system auto-selects "
                             "the best model cycle and loads the right "
                             "forecast hours.  Omit to use 'now'.")
    parser.add_argument("--forecast-hours", type=str, default=None,
                        help="Override: comma-separated forecast hours "
                             "(advanced use, normally auto-computed).")
    parser.add_argument("--tz", type=str, default="America/Los_Angeles",
                        help="IANA timezone for --depart (default: "
                             "America/Los_Angeles)")
    parser.add_argument("--polar", type=str, default=None,
                        help="Path to polar CSV (TWA_deg, TWS_kt, BoatSpeed_kt)")
    parser.add_argument("--wind-speed", type=float, default=None,
                        help="Constant true wind speed in knots")
    parser.add_argument("--wind-direction", type=float, default=None,
                        help="Constant wind direction in degrees, "
                             "meteorological 'from' convention (CW from north)")
    parser.add_argument("--tack-penalty", type=float, default=60.0,
                        help="Time penalty in seconds added whenever the boat "
                             "makes a course change exceeding 90 degrees "
                             "(default: 60).  Set 0 to disable.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Skip SSCOFS cache, always download fresh")
    parser.add_argument("--save", type=str, default=None,
                        help="Save plot to file (e.g. route.png)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plotting")
    args = parser.parse_args()

    # ---- Parse departure time ----
    depart_dt = None
    if args.depart:
        naive = _dt.datetime.strptime(args.depart, "%Y-%m-%d %H:%M")
        depart_dt = naive.replace(tzinfo=ZoneInfo(args.tz))

    # ---- Forecast hours override ----
    forecast_hours = None
    if args.forecast_hours:
        forecast_hours = [int(h) for h in args.forecast_hours.split(",")]

    # ---- Load current data ----
    cf, transformer, start_time_s, depart_utc = load_current_field(
        depart_dt=depart_dt,
        forecast_hours=forecast_hours,
        use_cache=not args.no_cache)

    # ---- Polar / wind ----
    polar = None
    if args.polar:
        polar = PolarTable(args.polar)
        print(f"Polar loaded: max speed {polar.max_speed_kt:.1f} kt")

    wind = None
    if args.wind_speed is not None and args.wind_direction is not None:
        wind = WindField.from_met(args.wind_speed, args.wind_direction)
        print(f"Wind: {args.wind_speed:.1f} kt from {args.wind_direction:.0f} deg")
    elif (args.wind_speed is not None) != (args.wind_direction is not None):
        print("Warning: both --wind-speed and --wind-direction are required "
              "together; ignoring wind.")

    boat = BoatModel(base_speed_knots=args.boat_speed, polar=polar)
    router = Router(cf, boat, resolution_m=args.grid_resolution,
                    padding_m=args.padding, wind=wind,
                    tack_penalty_s=args.tack_penalty)

    start = (args.start_lat, args.start_lon)
    end = (args.end_lat, args.end_lon)

    route, xs, ys, water_mask = router.find_route(
        start, end, start_time_s=start_time_s)

    # ---- Print arrival time ----
    if depart_utc is not None:
        tz = ZoneInfo(args.tz)
        arrive_utc = depart_utc + _dt.timedelta(seconds=route.total_time_s)
        arrive_local = arrive_utc.astimezone(tz)
        depart_local = depart_utc.astimezone(tz)
        print(f"\nDepart:  {depart_local:%I:%M %p %Z}")
        print(f"Arrive:  {arrive_local:%I:%M %p %Z}")

    sl_time, sl_dist = router.straight_line_time(
        start, end, start_time_s=start_time_s)
    if np.isfinite(sl_time):
        print(f"\nStraight-line time: {sl_time / 60:.1f} min "
              f"({sl_dist / 1852:.2f} nm)")
    else:
        print("\nStraight line crosses land -- no direct comparison.")

    if not args.no_plot or args.save:
        if args.no_plot:
            import matplotlib
            matplotlib.use('Agg')
        plot_route(route, xs, ys, water_mask, cf,
                   start, end,
                   straight_time_s=sl_time, straight_dist_m=sl_dist,
                   save_path=args.save,
                   show=(not args.no_plot))

    return route


if __name__ == "__main__":
    main()
