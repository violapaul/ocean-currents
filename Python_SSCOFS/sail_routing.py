"""
sail_routing.py
---------------

Sailboat routing algorithm that finds the time-optimal path through
SSCOFS ocean current data.

Phase 1: fixed boat speed through water (current only).
Phase 2: polar-based speed dependent on true wind angle and speed,
         with optional spatial/time-varying wind field.

The active router uses time-dependent A* on SSCOFS nodes with heading-binned
connectivity (SectorRouter). Each node carries an arrival time so edge costs
reflect currents and wind at that moment (SSCOFS provides hourly forecasts).
With a single time snapshot the algorithm degenerates to standard static A*.

Usage:
    python sail_routing.py \\
        --start-lat 47.63 --start-lon -122.40 \\
        --end-lat 47.75 --end-lon -122.42 \\
        --boat-speed 6 \\
        --save route.png

    # With polar and constant wind:
    python sail_routing.py \\
        --start-lat 47.63 --start-lon -122.40 \\
        --end-lat 47.75 --end-lon -122.42 \\
        --polar /path/to/j105_polar_data_long.csv \\
        --wind-speed 12 --wind-direction 180 \\
        --save route_polar.png
"""

import argparse
import csv
import datetime as _dt
import heapq
import sys
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree, Delaunay

try:
    import numba as _numba_mod
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    _numba_mod = None

# ---------------------------------------------------------------------------
# Local imports from the existing SSCOFS pipeline
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from plot_local_currents import get_latest_current_data, create_utm_transformer
from shoreline_utils import draw_shoreline
from water_boundary import build_water_mask_utm, load_nav_mask

NAV_MASK_PATH = Path(__file__).parent / "data" / "nav_mask.npz"

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

    Supports two CSV formats:

    **Simple format** — columns ``TWA_deg, TWS_kt, BoatSpeed_kt``.
    Must form a complete rectangular grid.

    **Sail-config format** — columns include ``sail, TWS_kt, TWA_deg,
    BTV_kt``.  Only the *Best Performance* rows are used (the envelope
    of all sail configurations).  Beat/run-target TWAs vary by TWS, so
    the grid is built from the union of all TWA values and any gaps are
    filled by linear interpolation within each TWS column.  A 0° row
    at zero speed is prepended automatically.

    Parameters
    ----------
    csv_path : str or Path
    minimum_twa : float, optional
        No-go zone: TWAs below this angle are forced to zero speed.
        Default 0 (use raw polar values).
    sail_filter : str, optional
        Which sail / row group to use from a sail-config polar.
        Default ``"Best Performance"``.
    """

    def __init__(self, csv_path, minimum_twa=0.0, sail_filter="Best Performance"):
        csv_path = Path(csv_path)
        self.minimum_twa = float(minimum_twa)

        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            all_rows = list(reader)

        if not all_rows:
            raise ValueError(f"Polar CSV is empty: {csv_path}")

        if 'sail' in all_rows[0] and 'BTV_kt' in all_rows[0]:
            self._load_sail_config(all_rows, sail_filter)
        else:
            self._load_simple(all_rows)

        if self.minimum_twa > 0:
            self._speeds[self._twas < self.minimum_twa, :] = 0.0

        self.max_speed_kt = float(np.nanmax(self._speeds))
        self.max_speed_ms = self.max_speed_kt * KNOTS_TO_MS

        self._build_dense_lookup()

    # ------------------------------------------------------------------

    def _build_dense_lookup(self, max_tws_kt: int = 60):
        """Pre-bake a dense (1° TWA × 1 kt TWS) lookup array.

        Evaluates the full bilinear interpolation from the sparse polar grid
        once at startup, producing ``self._polar_dense`` of shape
        ``(182, max_tws_kt + 2)``.  During A* the Numba kernel can then look
        up boat speed with direct integer indexing — no binary search needed.

        Index semantics
        ---------------
        TWA axis  : row index = int(twa_deg), valid 0..181 (clamped at 181)
        TWS axis  : col index = int(tws_kt),  valid 0..max_tws_kt (clamped)
        Fractional bilinear blending still uses the sub-integer remainders
        so accuracy matches the original sparse-grid interpolation.
        """
        n_twa = 182                      # rows: TWA 0 .. 181°
        n_tws = max_tws_kt + 2           # cols: TWS 0 .. max_tws_kt+1 kt
        dense = np.zeros((n_twa, n_tws), dtype=np.float64)

        twa_pts = np.arange(n_twa, dtype=np.float64)
        twa_c = np.clip(twa_pts, self._twas[0], self._twas[-1])

        # TWA bracket for every row — vectorised
        i0 = np.clip(
            np.searchsorted(self._twas, twa_c, side='right') - 1,
            0, len(self._twas) - 2,
        ).astype(int)
        i1 = i0 + 1
        denom_twa = self._twas[i1] - self._twas[i0]
        alpha = np.where(denom_twa > 0,
                         (twa_c - self._twas[i0]) / denom_twa,
                         0.0)

        for col in range(n_tws):
            tws = float(col)
            tws_c = float(np.clip(tws, self._twss[0], self._twss[-1]))
            j0 = int(np.clip(
                np.searchsorted(self._twss, tws_c, side='right') - 1,
                0, len(self._twss) - 2,
            ))
            j1 = j0 + 1
            denom_tws = self._twss[j1] - self._twss[j0]
            beta = (tws_c - self._twss[j0]) / denom_tws if denom_tws > 0 else 0.0
            oma, omb = 1.0 - alpha, 1.0 - beta
            dense[:, col] = (
                oma * omb * self._speeds[i0, j0] +
                alpha * omb * self._speeds[i1, j0] +
                oma * beta * self._speeds[i0, j1] +
                alpha * beta * self._speeds[i1, j1]
            )

        # Enforce the no-go zone already applied to _speeds.
        if self.minimum_twa > 0:
            cutoff = min(int(np.ceil(self.minimum_twa)), n_twa)
            dense[:cutoff, :] = 0.0

        self._polar_dense = dense
        self._polar_dense_max_tws = max_tws_kt

    # ------------------------------------------------------------------

    def _load_simple(self, all_rows):
        """Load a simple TWA_deg / TWS_kt / BoatSpeed_kt grid."""
        rows = [(float(r['TWA_deg']), float(r['TWS_kt']),
                 float(r['BoatSpeed_kt'])) for r in all_rows]

        twa_vals = sorted(set(r[0] for r in rows))
        tws_vals = sorted(set(r[1] for r in rows))
        self._twas = np.array(twa_vals, dtype=np.float64)
        self._twss = np.array(tws_vals, dtype=np.float64)

        twa_idx = {v: i for i, v in enumerate(twa_vals)}
        tws_idx = {v: i for i, v in enumerate(tws_vals)}

        self._speeds = np.full((len(twa_vals), len(tws_vals)),
                               np.nan, dtype=np.float64)
        for twa, tws, spd in rows:
            self._speeds[twa_idx[twa], tws_idx[tws]] = spd

        missing = np.argwhere(np.isnan(self._speeds))
        if missing.size:
            n_missing = int(missing.shape[0])
            examples = ", ".join(
                f"(TWA={self._twas[i]:g},TWS={self._twss[j]:g})"
                for i, j in missing[:5]
            )
            raise ValueError(
                f"Polar table missing {n_missing} grid point(s): {examples}"
            )

    def _load_sail_config(self, all_rows, sail_filter):
        """Load a sail-configuration polar (sail / TWS / TWA / BTV).

        Speeds below the minimum data TWA for each TWS column are set
        to zero (no-go zone) rather than interpolated.  The minimum_twa
        is auto-set to the global minimum beat-target angle.
        """
        triples = []
        for r in all_rows:
            if r['sail'].strip() != sail_filter:
                continue
            triples.append((float(r['TWA_deg']),
                            float(r['TWS_kt']),
                            float(r['BTV_kt'])))

        if not triples:
            available = sorted(set(r['sail'].strip() for r in all_rows))
            raise ValueError(
                f"No rows matching sail='{sail_filter}'. "
                f"Available: {available}"
            )

        twa_set = sorted(set(t[0] for t in triples))
        tws_vals = sorted(set(t[1] for t in triples))

        if twa_set[0] > 0.5:
            twa_set = [0.0] + twa_set

        self._twas = np.array(twa_set, dtype=np.float64)
        self._twss = np.array(tws_vals, dtype=np.float64)

        lookup = {}
        for twa, tws, spd in triples:
            lookup[(twa, tws)] = spd

        n_twa = len(self._twas)
        n_tws = len(self._twss)
        self._speeds = np.full((n_twa, n_tws), np.nan, dtype=np.float64)

        if self._twas[0] == 0.0:
            self._speeds[0, :] = 0.0

        for (twa, tws), spd in lookup.items():
            i = int(np.searchsorted(self._twas, twa))
            j = int(np.searchsorted(self._twss, tws))
            if i < n_twa and j < n_tws:
                self._speeds[i, j] = spd

        for j in range(n_tws):
            col = self._speeds[:, j]
            known = np.where(~np.isnan(col))[0]
            if len(known) < 2:
                continue
            min_data_twa = self._twas[known[known > 0].min() if np.any(known > 0) else 0]
            for i in range(n_twa):
                if np.isnan(col[i]):
                    if self._twas[i] < min_data_twa:
                        col[i] = 0.0
                    else:
                        col[i] = float(np.interp(
                            self._twas[i],
                            self._twas[known],
                            col[known],
                        ))
            self._speeds[:, j] = col

        still_nan = np.argwhere(np.isnan(self._speeds))
        if still_nan.size:
            for i, j in still_nan:
                self._speeds[i, j] = 0.0

        data_min_twa = min(t[0] for t in triples)
        if self.minimum_twa < data_min_twa:
            self.minimum_twa = data_min_twa

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
        if self.minimum_twa > 0 and twa_deg < self.minimum_twa:
            return 0.0
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
        self._wu_frame_means = None
        self._wv_frame_means = None
        self._frame_times = None
        self._node_x = None
        self._node_y = None
        self._node_tree = None

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
        if len(wu_frames) != len(wv_frames):
            raise ValueError("wu_frames and wv_frames must have the same length")
        if len(wu_frames) == 0:
            raise ValueError("WindField.from_frames requires at least one frame")
        frame_times = np.asarray(frame_times_s, dtype=np.float64)
        if frame_times.size != len(wu_frames):
            raise ValueError("frame_times_s length must match frame count")
        if np.any(np.diff(frame_times) < 0):
            raise ValueError("frame_times_s must be non-decreasing")

        if xs is None:
            wu_vals = [float(wu) for wu in wu_frames]
            wv_vals = [float(wv) for wv in wv_frames]
            inst = cls(wu_vals[0], wv_vals[0])
            inst._mode = 'temporal_const'
            inst._wu_frames = wu_vals
            inst._wv_frames = wv_vals
            inst._wu_frame_means = wu_vals
            inst._wv_frame_means = wv_vals
        else:
            from scipy.interpolate import RegularGridInterpolator
            inst = cls(0.0, 0.0)
            inst._mode = 'temporal_grid'
            inst._xs = np.asarray(xs, dtype=np.float64)
            inst._ys = np.asarray(ys, dtype=np.float64)
            ny = inst._ys.size
            nx = inst._xs.size

            wu_interps = []
            wv_interps = []
            wu_means = []
            wv_means = []
            for wu, wv in zip(wu_frames, wv_frames):
                wu_arr = np.asarray(wu, dtype=np.float64)
                wv_arr = np.asarray(wv, dtype=np.float64)
                expected_shape = (ny, nx)
                if wu_arr.shape != expected_shape or wv_arr.shape != expected_shape:
                    raise ValueError(
                        f"Temporal grid frame shape must be {expected_shape}, "
                        f"got wu={wu_arr.shape}, wv={wv_arr.shape}"
                    )
                wu_interps.append(RegularGridInterpolator(
                    (inst._ys, inst._xs), wu_arr, method='linear',
                    bounds_error=False, fill_value=None))
                wv_interps.append(RegularGridInterpolator(
                    (inst._ys, inst._xs), wv_arr, method='linear',
                    bounds_error=False, fill_value=None))
                wu_means.append(float(np.nanmean(wu_arr)))
                wv_means.append(float(np.nanmean(wv_arr)))

            inst._wu_frames = wu_interps
            inst._wv_frames = wv_interps
            inst._wu_frame_means = wu_means
            inst._wv_frame_means = wv_means

        inst._frame_times = frame_times
        return inst

    @classmethod
    def from_node_frames(cls, node_x, node_y, wu_frames, wv_frames, frame_times_s):
        """Construct a time-varying wind field on irregular spatial nodes.

        Spatial query is nearest-node (no interpolation), preserving the raw
        model node values. Time is linearly interpolated between frames.
        """
        if len(wu_frames) != len(wv_frames):
            raise ValueError("wu_frames and wv_frames must have the same length")
        if len(wu_frames) == 0:
            raise ValueError("from_node_frames requires at least one frame")

        frame_times = np.asarray(frame_times_s, dtype=np.float64)
        if frame_times.size != len(wu_frames):
            raise ValueError("frame_times_s length must match frame count")
        if np.any(np.diff(frame_times) < 0):
            raise ValueError("frame_times_s must be non-decreasing")

        node_x = np.asarray(node_x, dtype=np.float64).ravel()
        node_y = np.asarray(node_y, dtype=np.float64).ravel()
        if node_x.size == 0 or node_x.size != node_y.size:
            raise ValueError("node_x/node_y must be non-empty and same length")

        wu_arr = np.asarray(wu_frames, dtype=np.float64)
        wv_arr = np.asarray(wv_frames, dtype=np.float64)
        expected = (len(wu_frames), node_x.size)
        if wu_arr.shape != expected or wv_arr.shape != expected:
            raise ValueError(
                f"Node frame shape must be {expected}, got "
                f"wu={wu_arr.shape}, wv={wv_arr.shape}"
            )

        inst = cls(0.0, 0.0)
        inst._mode = 'temporal_nodes'
        inst._node_x = node_x
        inst._node_y = node_y
        inst._node_tree = cKDTree(np.column_stack([node_x, node_y]))
        inst._wu_frames = wu_arr
        inst._wv_frames = wv_arr
        inst._wu_frame_means = np.nanmean(wu_arr, axis=1)
        inst._wv_frame_means = np.nanmean(wv_arr, axis=1)
        inst._frame_times = frame_times
        return inst

    def query(self, x_utm, y_utm, elapsed_s=0.0):
        """Return (wu, wv) in m/s at given UTM position and time."""
        if self._mode == 'constant':
            return self._wu, self._wv

        if self._mode in ('temporal_const', 'temporal_grid'):
            if len(self._frame_times) == 1:
                if self._mode == 'temporal_const':
                    return float(self._wu_frames[0]), float(self._wv_frames[0])
                pt = np.array([[y_utm, x_utm]])
                return (float(self._wu_frames[0](pt)[0]),
                        float(self._wv_frames[0](pt)[0]))

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

        if self._mode == 'temporal_nodes':
            x = np.atleast_1d(np.asarray(x_utm, dtype=np.float64))
            y = np.atleast_1d(np.asarray(y_utm, dtype=np.float64))
            _, node_idx = self._node_tree.query(np.column_stack([x, y]), k=1)
            node_idx = np.asarray(node_idx, dtype=np.int64)

            if len(self._frame_times) == 1:
                wu = self._wu_frames[0, node_idx]
                wv = self._wv_frames[0, node_idx]
            else:
                t = np.clip(elapsed_s, self._frame_times[0], self._frame_times[-1])
                idx = int(np.searchsorted(self._frame_times, t, side='right')) - 1
                idx = int(np.clip(idx, 0, len(self._frame_times) - 2))
                t0 = self._frame_times[idx]
                t1 = self._frame_times[idx + 1]
                alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                wu = self._wu_frames[idx, node_idx] * (1 - alpha) + self._wu_frames[idx + 1, node_idx] * alpha
                wv = self._wv_frames[idx, node_idx] * (1 - alpha) + self._wv_frames[idx + 1, node_idx] * alpha

            if x.size == 1:
                return float(wu[0]), float(wv[0])
            return wu, wv

        if self._mode == 'grid':
            pt = np.array([[y_utm, x_utm]])
            return float(self._interp_u(pt)[0]), float(self._interp_v(pt)[0])

        return self._wu, self._wv

    @property
    def wind_speed_ms(self):
        """Characteristic wind speed (m/s).

        For constant mode: the actual wind speed.
        For temporal modes: speed of the first frame.
        For spatial grid mode: not well-defined; returns 0.0.
        Use ``query()`` for position/time-specific values.
        """
        if self._mode == 'temporal_const':
            return float(np.hypot(self._wu_frames[0], self._wv_frames[0]))
        if self._mode == 'temporal_grid':
            if self._wu_frame_means is not None and self._wv_frame_means is not None:
                return float(np.hypot(self._wu_frame_means[0],
                                      self._wv_frame_means[0]))
        if self._mode == 'temporal_nodes':
            if self._wu_frame_means is not None and self._wv_frame_means is not None:
                return float(np.hypot(self._wu_frame_means[0],
                                      self._wv_frame_means[0]))
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

    Land detection uses Delaunay triangulation: points inside valid water
    triangles are water; points inside land-spanning triangles (with
    anomalously long edges) or outside the convex hull are land. Falls
    back to a distance threshold (``land_threshold_m``) if Delaunay fails.
    """

    def __init__(self, lonc, latc, u_frames, v_frames, frame_times_s,
                 transformer, k_neighbors=6, land_threshold_m=750.0,
                 use_delaunay=True, nav_mask_path=None):
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
            Fallback max distance (m) to nearest element; used when Delaunay
            is disabled or fails to build.
        use_delaunay : bool
            If True, use Delaunay triangulation for land detection (more
            accurate). Falls back to distance threshold if Delaunay fails.
        nav_mask_path : str or Path, optional
            Path to nav_mask.npz for authoritative land detection. If provided,
            supplements Delaunay-based detection.
        """
        self.transformer = transformer
        self.k = k_neighbors
        self.land_threshold = land_threshold_m

        x_utm, y_utm = transformer.transform(lonc, latc)
        self._x_utm = np.asarray(x_utm, dtype=np.float64)
        self._y_utm = np.asarray(y_utm, dtype=np.float64)
        self.tree = cKDTree(np.column_stack([self._x_utm, self._y_utm]))

        # Build Delaunay triangulation for land detection
        self.delaunay = None
        self.valid_triangle = None
        self.delaunay_error = None
        if use_delaunay and len(lonc) >= 4:
            try:
                self.delaunay, self.valid_triangle = build_water_mask_utm(
                    self._x_utm, self._y_utm
                )
            except Exception as exc:
                self.delaunay_error = str(exc)
                # Fall back to distance-based detection.

        # Load nav mask for authoritative land detection (DEM-based)
        self._nav_mask = None
        self._nav_mask_bounds = None
        self._nav_mask_res = None
        self._inv_transformer = None
        if nav_mask_path is not None and Path(nav_mask_path).exists():
            try:
                self._nav_mask, self._nav_mask_bounds, self._nav_mask_res = load_nav_mask(nav_mask_path)
                # Create inverse transformer (UTM -> lon/lat) for nav mask lookups
                self._inv_transformer = Transformer.from_crs(
                    transformer.target_crs, transformer.source_crs, always_xy=True
                )
            except Exception:
                pass  # Nav mask is optional, fall back to Delaunay only

        self.u_frames = [np.asarray(u, dtype=np.float64) for u in u_frames]
        self.v_frames = [np.asarray(v, dtype=np.float64) for v in v_frames]
        self.frame_times = np.asarray(frame_times_s, dtype=np.float64)
        self.n_frames = len(self.u_frames)

        speeds = [np.sqrt(u**2 + v**2) for u, v in zip(self.u_frames, self.v_frames)]
        self.max_current_speed = float(max(np.nanmax(s) for s in speeds))

    # ------------------------------------------------------------------

    def _idw_prepare(self, x, y):
        """Precompute neighbour indices/weights for IDW interpolation."""
        dists, idxs = self.tree.query(np.column_stack([x, y]), k=self.k)
        if self.k == 1:
            dists = dists[:, np.newaxis]
            idxs = idxs[:, np.newaxis]

        # Land detection: use Delaunay triangulation if available
        if self.delaunay is not None and self.valid_triangle is not None:
            # find_simplex returns triangle index, or -1 if outside convex hull
            simplex_idx = self.delaunay.find_simplex(np.column_stack([x, y]))
            land_mask = np.ones(len(x), dtype=bool)
            in_hull = simplex_idx >= 0
            land_mask[in_hull] = ~self.valid_triangle[simplex_idx[in_hull]]
        else:
            # Fallback to distance-based detection
            land_mask = dists[:, 0] > self.land_threshold

        # Supplement with nav mask (DEM-based) if available
        if self._nav_mask is not None and self._inv_transformer is not None:
            lon, lat = self._inv_transformer.transform(x, y)
            lon_min, lon_max, lat_min, lat_max = self._nav_mask_bounds
            ny, nx = self._nav_mask.shape
            col = ((np.asarray(lon) - lon_min) / self._nav_mask_res).astype(int)
            row = ((np.asarray(lat) - lat_min) / self._nav_mask_res).astype(int)
            # Out-of-DEM-bounds points are treated as water (open ocean)
            in_bounds = (col >= 0) & (col < nx) & (row >= 0) & (row < ny)
            nav_water = np.ones(len(x), dtype=bool)
            nav_water[in_bounds] = self._nav_mask[row[in_bounds], col[in_bounds]]
            land_mask = land_mask | ~nav_water

        weights = np.where(dists < 1e-12, 1e12, 1.0 / dists)
        w_sum = weights.sum(axis=1, keepdims=True)
        weights /= w_sum
        return idxs, weights, land_mask

    @staticmethod
    def _idw_apply(values, idxs, weights, land_mask):
        """Apply precomputed IDW weights to one value field."""
        result = np.sum(weights * values[idxs], axis=1)
        result[land_mask] = np.nan
        return result

    def _idw_at_points(self, x, y, values):
        """IDW interpolation for an array of query points."""
        idxs, weights, land_mask = self._idw_prepare(x, y)
        return self._idw_apply(values, idxs, weights, land_mask)

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
        idxs, weights, land_mask = self._idw_prepare(x, y)

        if self.n_frames == 1:
            u = self._idw_apply(self.u_frames[0], idxs, weights, land_mask)
            v = self._idw_apply(self.v_frames[0], idxs, weights, land_mask)
        else:
            t = np.clip(elapsed_s, self.frame_times[0], self.frame_times[-1])
            idx = np.searchsorted(self.frame_times, t, side='right') - 1
            idx = int(np.clip(idx, 0, self.n_frames - 2))
            t0 = self.frame_times[idx]
            t1 = self.frame_times[idx + 1]
            alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0.0

            u0 = self._idw_apply(self.u_frames[idx], idxs, weights, land_mask)
            v0 = self._idw_apply(self.v_frames[idx], idxs, weights, land_mask)
            u1 = self._idw_apply(self.u_frames[idx + 1], idxs, weights, land_mask)
            v1 = self._idw_apply(self.v_frames[idx + 1], idxs, weights, land_mask)
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
        if self.polar is None or heading_rad is None or wind_u is None or wind_v is None:
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
    simulated_track: Optional[list] = None        # [(x, y), ...]
    simulated_track_times: Optional[list] = None  # [elapsed_s, ...] parallel to simulated_track
    debug: Optional[dict] = None                  # rich diagnostics for NPZ export

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

    def save_npz(self, path):
        """Save all route data + diagnostics to a compressed .npz file.

        The file contains everything needed to reconstruct, re-plot, and
        analyze the route without re-running the router.
        """
        d = {}

        # Final waypoints
        wps_utm = np.array(self.waypoints_utm)
        wps_ll = np.array(self.waypoints_latlon)
        d['wp_x'] = wps_utm[:, 0]
        d['wp_y'] = wps_utm[:, 1]
        d['wp_lat'] = wps_ll[:, 0]
        d['wp_lon'] = wps_ll[:, 1]
        d['wp_leg_time_s'] = np.array(self.leg_times_s)
        d['wp_leg_dist_m'] = np.array(self.leg_distances_m)
        d['total_time_s'] = np.float64(self.total_time_s)
        d['total_distance_m'] = np.float64(self.total_distance_m)
        d['avg_sog_kt'] = np.float64(self.avg_sog_knots)
        d['boat_speed_kt'] = np.float64(self.boat_speed_knots)
        d['nodes_explored'] = np.int64(self.nodes_explored)

        # Per-leg headings and turn angles
        if len(wps_utm) >= 2:
            dx = np.diff(wps_utm[:, 0])
            dy = np.diff(wps_utm[:, 1])
            d['wp_heading_deg'] = np.degrees(np.arctan2(dx, dy)) % 360.0
            if len(wps_utm) >= 3:
                h = d['wp_heading_deg']
                raw_diff = np.abs(np.diff(h))
                d['wp_turn_deg'] = np.minimum(raw_diff, 360.0 - raw_diff)

        # Cumulative arrival time at each waypoint
        if self.leg_times_s:
            cum = np.concatenate([[0.0], np.cumsum(self.leg_times_s)])
            d['wp_arrival_s'] = cum

        # Simulated track
        if self.simulated_track:
            trk = np.array(self.simulated_track)
            d['track_x'] = trk[:, 0]
            d['track_y'] = trk[:, 1]
        if self.simulated_track_times:
            d['track_time_s'] = np.array(self.simulated_track_times)

        # Merge debug dict (raw path, smoothed path, explored, perf, …)
        if self.debug:
            for k, v in self.debug.items():
                if isinstance(v, np.ndarray):
                    d[k] = v
                elif isinstance(v, (list, tuple)):
                    try:
                        d[k] = np.asarray(v)
                    except (ValueError, TypeError):
                        pass
                elif isinstance(v, dict):
                    for sk, sv in v.items():
                        d[f'{k}_{sk}'] = np.float64(sv) if np.isscalar(sv) else np.asarray(sv)
                elif np.isscalar(v):
                    d[k] = np.asarray(v)

        np.savez_compressed(path, **d)
        print(f"Route NPZ saved → {Path(path).name}  "
              f"({len(d)} arrays, {sum(v.nbytes for v in d.values()) / 1024:.0f} KB)")


# ===================================================================
#  Heading sweep -- polar + current physics
# ===================================================================

# Pre-computed heading unit vectors at 2-degree resolution.
_SWEEP_DEGS = np.arange(0, 360, 2, dtype=np.float64)
_SWEEP_RADS = np.radians(_SWEEP_DEGS)
_SWEEP_HX = np.cos(_SWEEP_RADS)   # east component (math frame)
_SWEEP_HY = np.sin(_SWEEP_RADS)   # north component


def _fixed_speed_sog(d_hat_x, d_hat_y, cu, cv, boat_speed_ms):
    """Compute SOG (m/s) along d_hat for a fixed boat speed through water.

    Accounts for cross-track current by crabbing.  Returns ``np.inf``
    when the cross-current exceeds boat speed (impassable leg).

    Parameters
    ----------
    d_hat_x, d_hat_y : float
        Unit vector of desired ground track (east, north).
    cu, cv : float
        Current velocity (m/s, east/north).
    boat_speed_ms : float
        Boat speed through water in m/s.

    Returns
    -------
    float : SOG in m/s along d_hat, or ``np.inf`` if impassable.
    """
    c_par = cu * d_hat_x + cv * d_hat_y
    c_perp_x = cu - c_par * d_hat_x
    c_perp_y = cv - c_par * d_hat_y
    c_perp_mag = np.hypot(c_perp_x, c_perp_y)
    if c_perp_mag >= boat_speed_ms:
        return np.inf
    v_water_along = np.sqrt(boat_speed_ms**2 - c_perp_mag**2)
    v_sog = v_water_along + c_par
    if v_sog <= 0.01:
        return np.inf
    return v_sog


def _polar_boat_speeds(boat_model, wind_u, wind_v):
    """Vectorised polar lookup for all sweep headings.

    Returns an array of boat speeds (m/s) over water for every heading
    in ``_SWEEP_RADS``.  Falls back to ``base_speed_ms`` if no polar or
    if wind is negligible.
    """
    tws_kt = np.hypot(wind_u, wind_v) * MS_TO_KNOTS
    if boat_model.polar is None or tws_kt <= 1e-3:
        return np.full(len(_SWEEP_HX), boat_model.base_speed_ms)

    wind_from_rad = np.arctan2(-wind_v, -wind_u)
    delta = _SWEEP_RADS - wind_from_rad
    delta = (delta + np.pi) % (2 * np.pi) - np.pi
    twa_arr = np.degrees(np.abs(delta))

    polar = boat_model.polar
    twa_c = np.clip(twa_arr, polar._twas[0], polar._twas[-1])
    tws_c = float(np.clip(tws_kt, polar._twss[0], polar._twss[-1]))

    i0 = np.clip(np.searchsorted(polar._twas, twa_c, side='right') - 1,
                 0, len(polar._twas) - 2)
    i1 = i0 + 1
    j0 = int(np.clip(np.searchsorted(polar._twss, tws_c, side='right') - 1,
                     0, len(polar._twss) - 2))
    j1 = j0 + 1

    alpha = np.where(polar._twas[i1] > polar._twas[i0],
                     (twa_c - polar._twas[i0]) / (polar._twas[i1] - polar._twas[i0]),
                     0.0)
    beta = float((tws_c - polar._twss[j0]) / (polar._twss[j1] - polar._twss[j0])
                 if polar._twss[j1] > polar._twss[j0] else 0.0)

    V_kt = ((1 - alpha) * (1 - beta) * polar._speeds[i0, j0] +
            alpha       * (1 - beta) * polar._speeds[i1, j0] +
            (1 - alpha) * beta       * polar._speeds[i0, j1] +
            alpha       * beta       * polar._speeds[i1, j1])
    if polar.minimum_twa > 0:
        V_kt = np.where(twa_arr < polar.minimum_twa, 0.0, V_kt)
    return V_kt * KNOTS_TO_MS


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
    sog, _, _, _ = _solve_heading_full(d_hat_x, d_hat_y, cu, cv,
                                       wind_u, wind_v, boat_model, drift_tol)
    return sog


def _solve_heading_full(d_hat_x, d_hat_y, cu, cv, wind_u, wind_v,
                        boat_model, drift_tol=0.10):
    """Like _solve_heading but returns (sog, gvx, gvy, heading_rad).

    gvx/gvy are the *actual* ground velocity of the best heading --
    including cross-track components from current and sailing angle.
    This is the velocity that determines the true ground track.

    Returns (0, 0, 0, 0) if impassable.
    """
    V = _polar_boat_speeds(boat_model, wind_u, wind_v)

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
#  Mesh adjacency helper for MeshRouter
# ===================================================================

def build_mesh_adjacency(delaunay, valid_mask):
    """Extract node adjacency and edge distances from a Delaunay triangulation.

    An edge exists between nodes a and b if they share a valid water triangle.
    This produces the navigation graph for mesh-based A* routing.

    Parameters
    ----------
    delaunay : scipy.spatial.Delaunay
        Delaunay triangulation of SSCOFS element centers.
    valid_mask : ndarray of bool
        True for water triangles, False for land-spanning triangles.

    Returns
    -------
    adj : dict[int, set[int]]
        Adjacency list: adj[node_id] -> set of neighbor node_ids.
    edge_dist : dict[tuple[int,int], float]
        Edge distances in the same units as delaunay.points (typically UTM metres).
        Keys are (min(a,b), max(a,b)) for canonical ordering.
    """
    from collections import defaultdict

    adj = defaultdict(set)
    edge_dist = {}
    points = delaunay.points
    simplices = delaunay.simplices

    for t in range(len(simplices)):
        if not valid_mask[t]:
            continue
        v0, v1, v2 = simplices[t]
        for a, b in [(v0, v1), (v1, v2), (v2, v0)]:
            adj[a].add(b)
            adj[b].add(a)
            key = (min(a, b), max(a, b))
            if key not in edge_dist:
                dx = points[b, 0] - points[a, 0]
                dy = points[b, 1] - points[a, 1]
                edge_dist[key] = np.hypot(dx, dy)

    return dict(adj), edge_dist


# ===================================================================
#  CSR adjacency builder (used by MeshRouter + Numba kernel)
# ===================================================================

def _build_csr_adjacency(adj, n_nodes):
    """Convert adjacency dict-of-sets to CSR (Compressed Sparse Row) format.

    Returns
    -------
    offsets : int32[n_nodes+1]
        offsets[i] is the start index in ``targets`` for node i's neighbors.
    targets : int32[total_edges]
        Neighbor node ids, grouped by source node.
    """
    offsets = np.zeros(n_nodes + 1, dtype=np.int32)
    for node, neighbors in adj.items():
        offsets[node + 1] = len(neighbors)
    np.cumsum(offsets, out=offsets)

    total_edges = int(offsets[n_nodes])
    targets = np.empty(total_edges, dtype=np.int32)

    pos = offsets[:-1].copy().astype(np.int64)
    for node, neighbors in adj.items():
        for nbr in neighbors:
            targets[pos[node]] = nbr
            pos[node] += 1

    return offsets, targets


# ===================================================================
#  Numba A* kernel (no-polar / fixed boat speed through water)
# ===================================================================

if _NUMBA_AVAILABLE:
    @_numba_mod.njit(cache=True)
    def _mesh_astar_jit(
        adj_offsets,        # int32[n_nodes+1]  — CSR row offsets
        adj_targets,        # int32[n_edges]    — CSR column indices
        node_x,             # float64[n_nodes]  — UTM easting
        node_y,             # float64[n_nodes]  — UTM northing
        u_frames,           # float64[n_frames, n_nodes]
        v_frames,           # float64[n_frames, n_nodes]
        frame_times,        # float64[n_frames]
        start_node,         # int
        end_node,           # int
        start_time_s,       # float
        boat_speed_ms,      # float  — fixed speed through water
        max_speed,          # float  — boat + max current (for heuristic)
        tack_penalty_s,     # float
        tack_threshold_deg, # float
        n_bearing_buckets,  # int    — e.g. 16
        start_sentinel,     # int    — sentinel bucket for start node (= n_bearing_buckets)
        max_iterations,     # int
    ):
        """Numba-compiled time-dependent A* on the SSCOFS mesh.

        Uses pre-allocated numpy arrays instead of Python dicts, and a
        manual binary min-heap instead of Python heapq.  Only handles the
        fixed-boat-speed (no polar/wind) case.

        Returns
        -------
        best_cost        : float64[n_nodes, n_buckets]
        arrival_time     : float64[n_nodes, n_buckets]
        came_from_node   : int32[n_nodes, n_buckets]   (-1 = no predecessor)
        came_from_bucket : int32[n_nodes, n_buckets]
        explored         : int
            >=0 on success, -1 if max_iterations exceeded, -2 if heap full.
        """
        n_nodes = node_x.shape[0]
        n_frames = frame_times.shape[0]
        n_buckets = n_bearing_buckets + 1
        INF = np.inf
        BUCKET_DEG = 360.0 / n_bearing_buckets
        bs2 = boat_speed_ms * boat_speed_ms
        goal_x = node_x[end_node]
        goal_y = node_y[end_node]

        # State arrays [n_nodes × n_buckets] — replaces Python dicts
        best_cost = np.full((n_nodes, n_buckets), INF)
        arrival_time = np.full((n_nodes, n_buckets), INF)
        came_from_node = np.full((n_nodes, n_buckets), -1, dtype=np.int32)
        came_from_bucket = np.full((n_nodes, n_buckets), -1, dtype=np.int32)

        best_cost[start_node, start_sentinel] = 0.0
        arrival_time[start_node, start_sentinel] = start_time_s

        # Binary min-heap rows: [f_cost, g_cost, float(node), float(bucket)]
        max_heap = max(n_nodes * 8, 500000)
        heap = np.empty((max_heap, 4), dtype=np.float64)
        heap_size = 0

        hdx0 = goal_x - node_x[start_node]
        hdy0 = goal_y - node_y[start_node]
        heap[0, 0] = np.sqrt(hdx0 * hdx0 + hdy0 * hdy0) / max_speed
        heap[0, 1] = 0.0
        heap[0, 2] = float(start_node)
        heap[0, 3] = float(start_sentinel)
        heap_size = 1

        explored = 0

        while heap_size > 0:
            if explored >= max_iterations:
                explored = -1
                break

            # Pop minimum (move last element to root, sift down)
            cost = heap[0, 1]
            node = int(heap[0, 2])
            bucket_in = int(heap[0, 3])
            heap_size -= 1
            if heap_size > 0:
                heap[0, 0] = heap[heap_size, 0]
                heap[0, 1] = heap[heap_size, 1]
                heap[0, 2] = heap[heap_size, 2]
                heap[0, 3] = heap[heap_size, 3]
                i = 0
                while True:
                    left = 2 * i + 1
                    right = 2 * i + 2
                    smallest = i
                    if left < heap_size and heap[left, 0] < heap[smallest, 0]:
                        smallest = left
                    if right < heap_size and heap[right, 0] < heap[smallest, 0]:
                        smallest = right
                    if smallest == i:
                        break
                    for col in range(4):
                        tmp = heap[i, col]
                        heap[i, col] = heap[smallest, col]
                        heap[smallest, col] = tmp
                    i = smallest

            # Lazy deletion: skip if a better path was already found
            if cost > best_cost[node, bucket_in]:
                continue

            explored += 1

            if node == end_node:
                break

            arr_t = arrival_time[node, bucket_in]

            # Expand neighbors (CSR format — no Python dict lookup)
            for edge_i in range(adj_offsets[node], adj_offsets[node + 1]):
                nb = int(adj_targets[edge_i])

                # Outgoing bearing bucket
                ndx = node_x[nb] - node_x[node]
                ndy = node_y[nb] - node_y[node]
                dist = np.sqrt(ndx * ndx + ndy * ndy)
                if dist < 1e-6:
                    continue
                bearing = np.degrees(np.arctan2(ndx, ndy)) % 360.0
                bucket_out = int(bearing / BUCKET_DEG) % n_bearing_buckets

                d_hat_x = ndx / dist
                d_hat_y = ndy / dist

                # Time-interpolated current at edge midpoint
                t = arr_t
                if t < frame_times[0]:
                    t = frame_times[0]
                elif t > frame_times[-1]:
                    t = frame_times[-1]

                if n_frames == 1:
                    ua = u_frames[0, node]
                    va = v_frames[0, node]
                    ub = u_frames[0, nb]
                    vb = v_frames[0, nb]
                else:
                    # Binary search for frame bracket
                    lo = 0
                    hi = n_frames - 1
                    while lo < hi:
                        mid = (lo + hi + 1) // 2
                        if frame_times[mid] <= t:
                            lo = mid
                        else:
                            hi = mid - 1
                    fi = lo
                    if fi >= n_frames - 1:
                        fi = n_frames - 2
                    t0f = frame_times[fi]
                    t1f = frame_times[fi + 1]
                    alpha = (t - t0f) / (t1f - t0f) if t1f > t0f else 0.0
                    om = 1.0 - alpha
                    ua = u_frames[fi, node] * om + u_frames[fi + 1, node] * alpha
                    va = v_frames[fi, node] * om + v_frames[fi + 1, node] * alpha
                    ub = u_frames[fi, nb] * om + u_frames[fi + 1, nb] * alpha
                    vb = v_frames[fi, nb] * om + v_frames[fi + 1, nb] * alpha

                cu = 0.5 * (ua + ub)
                cv = 0.5 * (va + vb)

                # Fixed-speed SOG (crabbing to cancel cross-track current)
                c_par = cu * d_hat_x + cv * d_hat_y
                c_perp_x = cu - c_par * d_hat_x
                c_perp_y = cv - c_par * d_hat_y
                c_perp_sq = c_perp_x * c_perp_x + c_perp_y * c_perp_y
                if c_perp_sq >= bs2:
                    continue
                v_sog = np.sqrt(bs2 - c_perp_sq) + c_par
                if v_sog <= 0.01:
                    continue

                dt = dist / v_sog

                # Tacking penalty
                penalty = 0.0
                if tack_penalty_s > 0.0 and bucket_in != start_sentinel:
                    diff = abs(bucket_in - bucket_out)
                    if diff > n_bearing_buckets // 2:
                        diff = n_bearing_buckets - diff
                    if float(diff) * BUCKET_DEG > tack_threshold_deg:
                        penalty = tack_penalty_s

                new_cost = cost + dt + penalty
                if new_cost < best_cost[nb, bucket_out]:
                    best_cost[nb, bucket_out] = new_cost
                    arrival_time[nb, bucket_out] = arr_t + dt + penalty
                    came_from_node[nb, bucket_out] = node
                    came_from_bucket[nb, bucket_out] = bucket_in

                    if heap_size < max_heap:
                        hdx2 = goal_x - node_x[nb]
                        hdy2 = goal_y - node_y[nb]
                        new_f = new_cost + np.sqrt(hdx2 * hdx2 + hdy2 * hdy2) / max_speed
                        heap[heap_size, 0] = new_f
                        heap[heap_size, 1] = new_cost
                        heap[heap_size, 2] = float(nb)
                        heap[heap_size, 3] = float(bucket_out)
                        heap_size += 1
                        # Sift up
                        i = heap_size - 1
                        while i > 0:
                            parent = (i - 1) // 2
                            if heap[i, 0] < heap[parent, 0]:
                                for col in range(4):
                                    tmp = heap[i, col]
                                    heap[i, col] = heap[parent, col]
                                    heap[parent, col] = tmp
                                i = parent
                            else:
                                break
                    else:
                        # Frontier overflow: abort instead of silently dropping states.
                        return (best_cost, arrival_time,
                                came_from_node, came_from_bucket, -2)

        return best_cost, arrival_time, came_from_node, came_from_bucket, explored

else:
    _mesh_astar_jit = None


# ===================================================================
#  Vectorized sector graph builder (for SectorRouter + Numba kernel)
# ===================================================================

def _build_corridor_sector_graph(
    node_x, node_y, tree, delaunay, valid_triangle,
    corridor_mask,
    n_sectors, sector_width,
    k_candidates, min_edge_m, max_edge_m, n_los_samples,
):
    """Build sector adjacency for corridor nodes using vectorized ops.

    Instead of per-node lazy discovery with individual KD-tree + LOS queries,
    this does one bulk KD-tree call and one batch ``find_simplex`` for all
    line-of-sight checks.

    Returns
    -------
    offsets  : int32[n_nodes+1]   — CSR row offsets (global node indices)
    targets  : int32[n_edges]     — neighbor global node ids
    distances: float64[n_edges]   — edge distances in metres
    sectors  : int32[n_edges]     — sector index of each edge
    """
    n_nodes = len(node_x)
    corridor_ids = np.where(corridor_mask)[0]
    n_corridor = len(corridor_ids)

    empty = (np.zeros(n_nodes + 1, dtype=np.int32),
             np.empty(0, dtype=np.int32),
             np.empty(0, dtype=np.float64),
             np.empty(0, dtype=np.int32))
    if n_corridor == 0:
        return empty

    pts = np.column_stack([node_x[corridor_ids], node_y[corridor_ids]])
    dists_all, idxs_all = tree.query(pts, k=k_candidates)

    dx = node_x[idxs_all] - node_x[corridor_ids, np.newaxis]
    dy = node_y[idxs_all] - node_y[corridor_ids, np.newaxis]
    bearings = np.degrees(np.arctan2(dx, dy)) % 360.0
    sec_all = (bearings / sector_width).astype(np.int32) % n_sectors

    valid_cand = ((idxs_all != corridor_ids[:, np.newaxis]) &
                  (dists_all >= min_edge_m) & (dists_all <= max_edge_m))

    # Choose per-sector candidates nearest-first; if LOS fails, retry with the
    # next-nearest candidate in that sector.
    best_nb = np.full((n_corridor, n_sectors), -1, dtype=np.int32)
    best_d = np.full((n_corridor, n_sectors), np.inf)
    pending = np.ones((n_corridor, n_sectors), dtype=bool)
    fracs = np.linspace(0, 1, n_los_samples)
    # One fallback retry beyond nearest candidate.
    max_sector_los_attempts = 2

    def _batch_los_ok(src_nodes, dst_nodes):
        sx = node_x[src_nodes]
        sy = node_y[src_nodes]
        ex = node_x[dst_nodes]
        ey = node_y[dst_nodes]

        samp_x = sx[:, np.newaxis] + fracs[np.newaxis, :] * (ex - sx)[:, np.newaxis]
        samp_y = sy[:, np.newaxis] + fracs[np.newaxis, :] * (ey - sy)[:, np.newaxis]
        flat_pts = np.column_stack([samp_x.ravel(), samp_y.ravel()])

        CHUNK = 5_000_000
        flat_ok = np.zeros(len(flat_pts), dtype=bool)
        for c0 in range(0, len(flat_pts), CHUNK):
            c1 = min(c0 + CHUNK, len(flat_pts))
            simplex = delaunay.find_simplex(flat_pts[c0:c1])
            in_hull = simplex >= 0
            flat_ok[c0:c1][in_hull] = valid_triangle[simplex[in_hull]]
        return flat_ok.reshape(len(src_nodes), n_los_samples).all(axis=1)

    for _ in range(max_sector_los_attempts):
        if not np.any(pending):
            break

        cand_ci = []
        cand_sec = []
        cand_bk = []
        cand_dst = []
        cand_dis = []

        for s in range(n_sectors):
            active_rows = np.where(pending[:, s])[0]
            if active_rows.size == 0:
                continue
            d_sub = dists_all[active_rows]
            i_sub = idxs_all[active_rows]
            v_sub = valid_cand[active_rows]
            s_mask = (sec_all[active_rows] == s) & v_sub
            d_m = np.where(s_mask, d_sub, np.inf)
            bk = np.argmin(d_m, axis=1)
            bd = np.take_along_axis(d_m, bk[:, np.newaxis], axis=1)[:, 0]
            bi = np.take_along_axis(i_sub, bk[:, np.newaxis], axis=1)[:, 0]

            has_candidate = bd < np.inf
            if np.any(~has_candidate):
                pending[active_rows[~has_candidate], s] = False
            if not np.any(has_candidate):
                continue

            rows = active_rows[has_candidate].astype(np.int32)
            cand_ci.append(rows)
            cand_sec.append(np.full(len(rows), s, dtype=np.int32))
            cand_bk.append(bk[has_candidate].astype(np.int32))
            cand_dst.append(bi[has_candidate].astype(np.int32))
            cand_dis.append(bd[has_candidate])

        if not cand_ci:
            break

        ci_idx = np.concatenate(cand_ci)
        s_idx = np.concatenate(cand_sec)
        bk_idx = np.concatenate(cand_bk)
        edge_dst = np.concatenate(cand_dst)
        edge_dis = np.concatenate(cand_dis)
        edge_src = corridor_ids[ci_idx].astype(np.int32)

        los_pass = _batch_los_ok(edge_src, edge_dst)
        if np.any(los_pass):
            pass_ci = ci_idx[los_pass]
            pass_sec = s_idx[los_pass]
            best_nb[pass_ci, pass_sec] = edge_dst[los_pass]
            best_d[pass_ci, pass_sec] = edge_dis[los_pass]
            pending[pass_ci, pass_sec] = False

        if np.any(~los_pass):
            fail_ci = ci_idx[~los_pass]
            fail_sec = s_idx[~los_pass]
            fail_bk = bk_idx[~los_pass]
            valid_cand[fail_ci, fail_bk] = False
            # Keep failed sectors pending so they can try the next-nearest candidate.
            pending[fail_ci, fail_sec] = True

    has_edge = best_nb >= 0
    ci_idx, s_idx = np.nonzero(has_edge)
    if len(ci_idx) == 0:
        return empty
    f_src = corridor_ids[ci_idx].astype(np.int32)
    f_dst = best_nb[ci_idx, s_idx].astype(np.int32)
    f_dis = best_d[ci_idx, s_idx]
    f_sec = s_idx.astype(np.int32)

    order = np.argsort(f_src, kind='stable')
    f_src = f_src[order]
    f_dst = f_dst[order]
    f_dis = f_dis[order]
    f_sec = f_sec[order]

    counts = np.bincount(f_src, minlength=n_nodes).astype(np.int32)
    offsets = np.zeros(n_nodes + 1, dtype=np.int32)
    np.cumsum(counts, out=offsets[1:])

    return offsets, f_dst, f_dis, f_sec


# ===================================================================
#  Numba A* kernel for SectorRouter (fixed-speed AND polar+wind)
# ===================================================================

_KNOTS_TO_MS_C = KNOTS_TO_MS
_MS_TO_KNOTS_C = MS_TO_KNOTS

if _NUMBA_AVAILABLE:
    @_numba_mod.njit(cache=True)
    def _sector_astar_jit(
        adj_offsets, adj_targets, adj_dists, adj_sectors,
        node_x, node_y,
        u_frames, v_frames, frame_times,
        start_node, end_node, start_time_s,
        boat_speed_ms, max_speed,
        # Polar table (pass empty arrays + has_polar=False to disable)
        has_polar, polar_twas, polar_twss, polar_speeds, polar_min_twa,
        sweep_hx, sweep_hy, sweep_rads, n_sweep, polar_coarse_step,
        # Dense pre-baked polar (pass empty array + use_dense_polar=False to disable)
        use_dense_polar, polar_dense, polar_dense_max_tws,
        # Forward-hemisphere filter: skip headings pointing away from the target
        use_dot_filter,
        # Wind at nodes (pass empty arrays + has_wind=False to disable)
        has_wind, wind_wu_nodes, wind_wv_nodes, wind_frame_times_w,
        # Routing params
        tack_penalty_s, tack_threshold_deg,
        n_sectors, start_sentinel, max_iterations,
    ):
        """Numba-compiled time-dependent A* over a sector graph.

        Finds the minimum travel-time path from ``start_node`` to ``end_node``
        through a navigation graph whose edge costs depend on ocean current,
        wind, and boat polar performance.  The entire function is compiled to
        native code with ``@njit`` so all data structures are plain NumPy
        arrays and all loops are scalar.

        State space
        -----------
        Each A* state is a (node, sector_bucket) pair.  ``sector_bucket``
        encodes the compass sector (0 .. n_sectors-1) the boat was travelling
        in when it *arrived* at the node.  This lets the search track direction
        changes so a tack penalty can be applied when the outgoing sector on
        the next edge differs by more than ``tack_threshold_deg``.  The special
        value ``start_sentinel = n_sectors`` is used for the start node, which
        has no incoming direction.

        Graph representation (CSR adjacency)
        -------------------------------------
        adj_offsets : shape (n_nodes+1,)
            adj_offsets[i] .. adj_offsets[i+1] are the indices into
            adj_targets/adj_dists/adj_sectors for node i's outgoing edges.
        adj_targets : shape (n_edges,)  int32
            Neighbour node index for each edge.
        adj_dists   : shape (n_edges,)  float64
            Edge length in metres.
        adj_sectors : shape (n_edges,)  int32
            Compass sector (0-based) of the edge direction.

        Time-interpolated fields
        ------------------------
        u_frames, v_frames : shape (n_frames, n_nodes)
            Ocean current east/north components (m/s) sampled at ``frame_times``.
        frame_times : shape (n_frames,)
            Epoch-seconds for each current frame.
        wind_wu_nodes, wind_wv_nodes : shape (n_wind_frames, n_nodes)
            Wind east/north components (m/s), only used when ``has_wind=True``.
        wind_frame_times_w : shape (n_wind_frames,)
            Epoch-seconds for each wind frame.

        At each node expansion the current at the *arrival time* is linearly
        interpolated between the two bracketing frames using a binary-search
        bracket (lo/hi bisection).  Edge current is the average of the current
        at the two endpoint nodes; edge wind is averaged the same way.

        Edge cost calculation
        ---------------------
        For each outgoing edge the code computes ``best_sog`` — the maximum
        speed-over-ground achievable along the edge direction — then sets
        ``dt = dist / best_sog``.

        Polar + wind mode (``has_polar=True``, ``has_wind=True``):
            1. Compute mid-edge TWS (true wind speed) and look up the TWS
               bracket (j0/j1) in the polar table once per edge.
            2. Sweep all ``n_sweep`` candidate headings (every 2° around the
               full circle) in two passes:
               - Coarse pass: step through headings ``polar_coarse_step``
                 apart (default 5 → 10° steps → 18 evaluations) to find the
                 approximate best-heading index.
               - Refine pass: re-evaluate the (2*coarse_step - 1) headings
                 surrounding the coarse winner at full 2° resolution.
            3. For each candidate heading compute the TWA, look up the TWA
               bracket (i0/i1) in the polar table, bilinearly interpolate the
               boat speed in knots, convert to m/s, add the mid-edge current
               vector to get the ground velocity, and measure its along-track
               component (SOG) and cross-track drift.  Accept the heading only
               if ``sog > 0.01 m/s`` and ``drift <= 0.10 * sog``.
            4. Special case: if TWS < 0.001 kt fall back to the fixed-speed
               crabbing formula below.

        Fixed-speed mode (``has_polar=False`` or no wind):
            Decompose the current into along-track and cross-track components.
            The cross-track component must be overcome by crabbing; if it
            exceeds ``boat_speed_ms`` the edge is impassable.  SOG is
            ``sqrt(boat_speed^2 - c_perp^2) + c_par``.

        Tack penalty
        ------------
        After ``dt`` is computed, if the angular difference between
        ``bucket_in`` (arrival sector) and ``sector_out`` (departure sector)
        exceeds ``tack_threshold_deg``, ``tack_penalty_s`` seconds are added
        to the edge cost.  No penalty is applied from the start node.

        Priority queue
        --------------
        A hand-rolled binary min-heap stored in ``heap`` (shape
        ``(max_heap, 4)``): columns are (f_score, g_cost, node, bucket).
        ``f_score = g_cost + heuristic`` where the heuristic is
        ``straight_line_distance / max_speed`` (admissible, so A* is optimal).
        Sift-down is performed on pop; sift-up on push.

        Parameters
        ----------
        adj_offsets, adj_targets, adj_dists, adj_sectors
            CSR adjacency arrays (see above).
        node_x, node_y : shape (n_nodes,)
            Node positions in metres (UTM or similar projected CRS).
        u_frames, v_frames : shape (n_frames, n_nodes)
            Time-series ocean current east/north (m/s).
        frame_times : shape (n_frames,)
            Epoch-seconds corresponding to each current frame.
        start_node, end_node : int
            Source and destination node indices.
        start_time_s : float
            Departure time in epoch-seconds.
        boat_speed_ms : float
            Nominal boat speed through water (m/s), used as fixed speed when
            ``has_polar=False`` and as fallback when wind is negligible.
        max_speed : float
            Upper-bound speed (m/s) used only for the A* admissible heuristic.
        has_polar : bool
            Enable polar-based boat speed (requires ``has_wind=True`` in
            practice).
        polar_twas : shape (n_twa,)
            True-wind-angle breakpoints in the polar table (degrees, 0-180).
        polar_twss : shape (n_tws,)
            True-wind-speed breakpoints in the polar table (knots).
        polar_speeds : shape (n_twa, n_tws)
            Boat speeds (knots) at each (TWA, TWS) grid point.
        polar_min_twa : float
            Headings within this many degrees of dead upwind give zero speed.
        sweep_hx, sweep_hy : shape (n_sweep,)
            Pre-computed unit vectors (east, north) for every candidate
            heading.  Typically the module-level ``_SWEEP_HX/_SWEEP_HY``
            (180 headings at 2° spacing).
        sweep_rads : shape (n_sweep,)
            Corresponding angles in radians.
        n_sweep : int
            Number of candidate headings (len of sweep arrays).
        polar_coarse_step : int
            Step size for the coarse heading pass.  1 = exact (no coarse);
            5 = 10° coarse pass followed by 2° refine.
        use_dense_polar : bool
            When True, use the pre-baked dense lookup (``polar_dense``)
            instead of the sparse binary-search path.  Eliminates the TWA
            and TWS bracket searches from the hot loop.
        polar_dense : shape (182, polar_dense_max_tws+2)
            Dense lookup array at 1° TWA × 1 kt TWS resolution, in knots.
            Row index = int(twa_deg), column index = int(tws_kt).
        polar_dense_max_tws : int
            Maximum TWS column index (= number of 1-kt TWS steps covered).
        use_dot_filter : bool
            When True, skip any candidate heading whose unit vector has a
            negative dot product with ``d_hat`` (the desired ground track).
            This discards the entire backward hemisphere (~90 headings) with a
            single multiply-add per candidate before doing any polar work.
            Can be combined with ``polar_coarse_step`` and
            ``use_dense_polar``.  The filter is exact — no valid heading is
            discarded — because a heading pointing away from the target can
            never produce positive SOG along the track.
        has_wind : bool
            Whether wind arrays are populated and should be used.
        wind_wu_nodes, wind_wv_nodes : shape (n_wf, n_nodes)
            Wind east/north (m/s) at each node and wind frame.
        wind_frame_times_w : shape (n_wf,)
            Epoch-seconds for wind frames.
        tack_penalty_s : float
            Extra seconds added for each tack/gybe (0 to disable).
        tack_threshold_deg : float
            Minimum direction change (degrees) that triggers the tack penalty.
        n_sectors : int
            Number of compass sectors for state-space bucketing (e.g. 32).
        start_sentinel : int
            Bucket index used for the start node (= n_sectors).
        max_iterations : int
            Hard cap on node expansions; returns ``explored = -1`` if hit.

        Returns
        -------
        best_cost : ndarray, shape (n_nodes, n_sectors+1)
            Minimum travel time (seconds) to reach each (node, bucket) state.
        arrival_time : ndarray, shape (n_nodes, n_sectors+1)
            Wall-clock arrival time (epoch-seconds) for each state.
        came_from_node : ndarray int32, shape (n_nodes, n_sectors+1)
            Parent node index for path reconstruction (-1 = unvisited).
        came_from_bucket : ndarray int32, shape (n_nodes, n_sectors+1)
            Parent bucket index for path reconstruction (-1 = unvisited).
        explored : int
            Number of node expansions completed.  Special values:
            -1 = max_iterations reached; -2 = heap overflow (path aborted).
        """
        n_nodes = node_x.shape[0]
        n_frames = frame_times.shape[0]
        n_buckets = n_sectors + 1          # sectors 0..n_sectors-1 + start_sentinel
        INF = np.inf
        PI = np.float64(3.141592653589793)
        KNOTS_TO_MS_L = np.float64(0.514444)
        MS_TO_KNOTS_L = np.float64(1.0 / 0.514444)
        SECTOR_WIDTH = 360.0 / n_sectors   # degrees per sector bucket
        bs2 = boat_speed_ms * boat_speed_ms  # boat speed squared, reused in crabbing formula
        goal_x = node_x[end_node]
        goal_y = node_y[end_node]

        # Per-(node, bucket) best travel-time and wall-clock arrival time found so far.
        best_cost = np.full((n_nodes, n_buckets), INF)
        arrival_time = np.full((n_nodes, n_buckets), INF)
        # Parent pointers for path reconstruction after the search completes.
        came_from_node = np.full((n_nodes, n_buckets), -1, dtype=np.int32)
        came_from_bucket = np.full((n_nodes, n_buckets), -1, dtype=np.int32)

        best_cost[start_node, start_sentinel] = 0.0
        arrival_time[start_node, start_sentinel] = start_time_s

        # Min-heap rows: [f_score, g_cost, node (float), bucket (float)].
        # Stored as float64 so Numba can use a single typed array.
        max_heap = max(n_nodes * 4, 500000)
        heap = np.empty((max_heap, 4), dtype=np.float64)
        hdx0 = goal_x - node_x[start_node]
        hdy0 = goal_y - node_y[start_node]
        # f = g + h;  h = straight-line distance / max_speed (admissible heuristic).
        heap[0, 0] = np.sqrt(hdx0 * hdx0 + hdy0 * hdy0) / max_speed
        heap[0, 1] = 0.0                   # g = 0 at start
        heap[0, 2] = float(start_node)
        heap[0, 3] = float(start_sentinel)
        heap_size = 1
        explored = 0

        # TWS bracket (j0, beta_p) is recomputed per edge because wind varies
        # spatially; there is no single value we can hoist out of the main loop.

        while heap_size > 0:
            if explored >= max_iterations:
                explored = -1
                break

            # --- Pop minimum f-score state from heap ---
            cost = heap[0, 1]              # g_cost of this state
            node = int(heap[0, 2])
            bucket_in = int(heap[0, 3])    # sector the boat arrived from
            heap_size -= 1
            if heap_size > 0:
                # Move last entry to root, then sift down to restore heap order.
                heap[0, 0] = heap[heap_size, 0]
                heap[0, 1] = heap[heap_size, 1]
                heap[0, 2] = heap[heap_size, 2]
                heap[0, 3] = heap[heap_size, 3]
                i = 0
                while True:
                    left = 2 * i + 1
                    right = 2 * i + 2
                    sm = i
                    if left < heap_size and heap[left, 0] < heap[sm, 0]:
                        sm = left
                    if right < heap_size and heap[right, 0] < heap[sm, 0]:
                        sm = right
                    if sm == i:
                        break
                    for col in range(4):
                        tmp = heap[i, col]
                        heap[i, col] = heap[sm, col]
                        heap[sm, col] = tmp
                    i = sm

            # Stale entry: a cheaper path to this (node, bucket) was already found.
            if cost > best_cost[node, bucket_in]:
                continue

            explored += 1
            if node == end_node:
                break

            arr_t = arrival_time[node, bucket_in]  # wall-clock time we arrive here

            # --- Time-interpolate ocean current at this node ---
            # Clamp to the available frame range so edge cases don't extrapolate.
            t_c = arr_t
            if t_c < frame_times[0]:
                t_c = frame_times[0]
            elif t_c > frame_times[-1]:
                t_c = frame_times[-1]
            if n_frames == 1:
                u_node = u_frames[0, node]
                v_node = v_frames[0, node]
            else:
                # Binary search for the bracketing frame index fi such that
                # frame_times[fi] <= t_c < frame_times[fi+1].
                lo, hi = 0, n_frames - 1
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if frame_times[mid] <= t_c:
                        lo = mid
                    else:
                        hi = mid - 1
                fi = lo
                if fi >= n_frames - 1:
                    fi = n_frames - 2
                alpha_c = ((t_c - frame_times[fi]) /
                           (frame_times[fi + 1] - frame_times[fi])
                           if frame_times[fi + 1] > frame_times[fi] else 0.0)
                om_c = 1.0 - alpha_c       # weight for frame fi
                u_node = u_frames[fi, node] * om_c + u_frames[fi + 1, node] * alpha_c
                v_node = v_frames[fi, node] * om_c + v_frames[fi + 1, node] * alpha_c

            # --- Time-interpolate wind at this node (same pattern as current) ---
            wu_node = 0.0
            wv_node = 0.0
            if has_wind:
                n_wf = wind_frame_times_w.shape[0]
                t_w = arr_t
                if t_w < wind_frame_times_w[0]:
                    t_w = wind_frame_times_w[0]
                elif t_w > wind_frame_times_w[-1]:
                    t_w = wind_frame_times_w[-1]
                if n_wf == 1:
                    wu_node = wind_wu_nodes[0, node]
                    wv_node = wind_wv_nodes[0, node]
                else:
                    lo2, hi2 = 0, n_wf - 1
                    while lo2 < hi2:
                        mid2 = (lo2 + hi2 + 1) // 2
                        if wind_frame_times_w[mid2] <= t_w:
                            lo2 = mid2
                        else:
                            hi2 = mid2 - 1
                    wi = lo2
                    if wi >= n_wf - 1:
                        wi = n_wf - 2
                    aw = ((t_w - wind_frame_times_w[wi]) /
                          (wind_frame_times_w[wi + 1] - wind_frame_times_w[wi])
                          if wind_frame_times_w[wi + 1] > wind_frame_times_w[wi]
                          else 0.0)
                    omw = 1.0 - aw
                    wu_node = wind_wu_nodes[wi, node] * omw + wind_wu_nodes[wi + 1, node] * aw
                    wv_node = wind_wv_nodes[wi, node] * omw + wind_wv_nodes[wi + 1, node] * aw

            # --- Expand all outgoing edges from the current node ---
            for ei in range(adj_offsets[node], adj_offsets[node + 1]):
                nb = int(adj_targets[ei])
                dist = adj_dists[ei]
                sector_out = int(adj_sectors[ei])  # compass sector of this edge's direction

                if dist < 1e-6:
                    continue

                # Unit vector pointing from node → neighbour (desired ground-track direction).
                d_hat_x = (node_x[nb] - node_x[node]) / dist
                d_hat_y = (node_y[nb] - node_y[node]) / dist

                # Reuse the same frame bracket (fi, alpha_c) computed for the current node;
                # the neighbour is evaluated at the same arrival time.
                if n_frames == 1:
                    u_nb = u_frames[0, nb]
                    v_nb = v_frames[0, nb]
                else:
                    u_nb = u_frames[fi, nb] * om_c + u_frames[fi + 1, nb] * alpha_c
                    v_nb = v_frames[fi, nb] * om_c + v_frames[fi + 1, nb] * alpha_c

                # Mid-edge current: simple average of the two endpoint values.
                cu = 0.5 * (u_node + u_nb)
                cv = 0.5 * (v_node + v_nb)

                # Mid-edge wind: default to node wind, then average with neighbour if available.
                wu_mid = wu_node
                wv_mid = wv_node
                if has_wind:
                    if n_wf == 1:
                        wu_nb = wind_wu_nodes[0, nb]
                        wv_nb = wind_wv_nodes[0, nb]
                    else:
                        wu_nb = wind_wu_nodes[wi, nb] * omw + wind_wu_nodes[wi + 1, nb] * aw
                        wv_nb = wind_wv_nodes[wi, nb] * omw + wind_wv_nodes[wi + 1, nb] * aw
                    wu_mid = 0.5 * (wu_node + wu_nb)
                    wv_mid = 0.5 * (wv_node + wv_nb)

                # ----- Edge cost: find the best achievable SOG along d_hat -----
                best_sog = 0.0

                if has_polar:
                    tws_ms = np.sqrt(wu_mid * wu_mid + wv_mid * wv_mid)
                    tws_kt = tws_ms * MS_TO_KNOTS_L
                    if tws_kt < 1e-3:
                        # Wind negligible — fall back to fixed-speed crabbing.
                        c_par = cu * d_hat_x + cv * d_hat_y
                        c_px = cu - c_par * d_hat_x
                        c_py = cv - c_par * d_hat_y
                        c_perp_sq = c_px * c_px + c_py * c_py
                        if c_perp_sq < bs2:
                            s = np.sqrt(bs2 - c_perp_sq) + c_par
                            if s > 0.01:
                                best_sog = s
                    else:
                        # Direction the wind blows FROM (radians), used to compute TWA.
                        wind_from_rad = np.arctan2(-wv_mid, -wu_mid)
                        coarse_step = polar_coarse_step
                        if coarse_step < 1:
                            coarse_step = 1
                        best_idx = -1      # index of the best heading found so far (-1 = none)

                        if use_dense_polar:
                            # ---- Dense path: direct integer index, no binary search ----
                            # TWS bracket via direct integer index into dense array.
                            j_d = int(tws_kt)
                            if j_d >= polar_dense_max_tws:
                                j_d = polar_dense_max_tws - 1
                            j_d1 = j_d + 1
                            beta_d = tws_kt - float(j_d)   # fractional TWS within 1-kt cell
                            omb_d = 1.0 - beta_d

                            # --- Pass 1: coarse sweep (dense) ---
                            for h in range(0, n_sweep, coarse_step):
                                # Forward-hemisphere filter: a heading pointing away
                                # from d_hat can never produce positive SOG along it.
                                if use_dot_filter and (sweep_hx[h] * d_hat_x +
                                                       sweep_hy[h] * d_hat_y) < 0.0:
                                    continue
                                delta_h = sweep_rads[h] - wind_from_rad
                                delta_h = (delta_h + PI) % (2.0 * PI) - PI
                                twa_deg = np.abs(delta_h) * (180.0 / PI)

                                if twa_deg < polar_min_twa:
                                    V_ms = 0.0
                                else:
                                    # Direct index — no search needed.
                                    i_d = int(twa_deg)
                                    if i_d > 181:
                                        i_d = 181
                                    i_d1 = i_d + 1
                                    if i_d1 > 181:
                                        i_d1 = 181
                                    alpha_d = twa_deg - float(i_d)  # sub-degree fraction
                                    oma_d = 1.0 - alpha_d
                                    V_kt = (oma_d * omb_d * polar_dense[i_d,  j_d] +
                                            alpha_d * omb_d * polar_dense[i_d1, j_d] +
                                            oma_d * beta_d * polar_dense[i_d,  j_d1] +
                                            alpha_d * beta_d * polar_dense[i_d1, j_d1])
                                    V_ms = V_kt * KNOTS_TO_MS_L

                                gx = V_ms * sweep_hx[h] + cu
                                gy = V_ms * sweep_hy[h] + cv
                                sog_h = gx * d_hat_x + gy * d_hat_y
                                drift_h = np.abs(gx * (-d_hat_y) + gy * d_hat_x)
                                prog = max(sog_h, 1e-6)
                                if sog_h > 0.01 and drift_h <= 0.50 * prog:
                                    if sog_h > best_sog:
                                        best_sog = sog_h
                                        best_idx = h

                            # --- Pass 2: refine (dense) ---
                            # Dot filter not applied here: the refine window is small
                            # and centred on the coarse winner which already passed.
                            if coarse_step > 1 and best_idx >= 0:
                                for off in range(-(coarse_step - 1), coarse_step):
                                    h = (best_idx + off) % n_sweep
                                    delta_h = sweep_rads[h] - wind_from_rad
                                    delta_h = (delta_h + PI) % (2.0 * PI) - PI
                                    twa_deg = np.abs(delta_h) * (180.0 / PI)

                                    if twa_deg < polar_min_twa:
                                        V_ms = 0.0
                                    else:
                                        i_d = int(twa_deg)
                                        if i_d > 181:
                                            i_d = 181
                                        i_d1 = i_d + 1
                                        if i_d1 > 181:
                                            i_d1 = 181
                                        alpha_d = twa_deg - float(i_d)
                                        oma_d = 1.0 - alpha_d
                                        V_kt = (oma_d * omb_d * polar_dense[i_d,  j_d] +
                                                alpha_d * omb_d * polar_dense[i_d1, j_d] +
                                                oma_d * beta_d * polar_dense[i_d,  j_d1] +
                                                alpha_d * beta_d * polar_dense[i_d1, j_d1])
                                        V_ms = V_kt * KNOTS_TO_MS_L

                                    gx = V_ms * sweep_hx[h] + cu
                                    gy = V_ms * sweep_hy[h] + cv
                                    sog_h = gx * d_hat_x + gy * d_hat_y
                                    drift_h = np.abs(gx * (-d_hat_y) + gy * d_hat_x)
                                    prog = max(sog_h, 1e-6)
                                    if sog_h > 0.01 and drift_h <= 0.50 * prog:
                                        if sog_h > best_sog:
                                            best_sog = sog_h

                        else:
                            # ---- Sparse path: binary search on TWS then TWA ----
                            # Clamp TWS to the range covered by the polar table.
                            tws_c = tws_kt
                            if tws_c < polar_twss[0]:
                                tws_c = polar_twss[0]
                            elif tws_c > polar_twss[-1]:
                                tws_c = polar_twss[-1]
                            # Find the TWS bracket (j0, j1) in the polar table.
                            # This is done once per edge; beta_p/omb are reused for every heading.
                            n_tws = polar_twss.shape[0]
                            j0 = 0
                            for jj in range(n_tws - 1, 0, -1):
                                if polar_twss[jj] <= tws_c:
                                    j0 = jj
                                    break
                            if j0 >= n_tws - 1:
                                j0 = n_tws - 2
                            j1 = j0 + 1
                            beta_p = ((tws_c - polar_twss[j0]) /
                                      (polar_twss[j1] - polar_twss[j0])
                                      if polar_twss[j1] > polar_twss[j0] else 0.0)
                            omb = 1.0 - beta_p  # complement weight for j0 column

                            # --- Pass 1: coarse heading sweep ---
                            # Step through every coarse_step-th heading index (e.g. step=5
                            # → every 10°) to locate the approximate best heading cheaply.
                            for h in range(0, n_sweep, coarse_step):
                                if use_dot_filter and (sweep_hx[h] * d_hat_x +
                                                       sweep_hy[h] * d_hat_y) < 0.0:
                                    continue
                                delta_h = sweep_rads[h] - wind_from_rad
                                delta_h = (delta_h + PI) % (2.0 * PI) - PI  # wrap to [-π, π]
                                twa_deg = np.abs(delta_h) * (180.0 / PI)

                                if twa_deg < polar_min_twa:
                                    # Too close to dead upwind — polar gives zero speed.
                                    V_ms = 0.0
                                else:
                                    # Clamp TWA to the range covered by the polar table.
                                    twa_c = twa_deg
                                    if twa_c < polar_twas[0]:
                                        twa_c = polar_twas[0]
                                    elif twa_c > polar_twas[-1]:
                                        twa_c = polar_twas[-1]
                                    # Find bracketing TWA row (i0, i1) via reverse linear scan.
                                    n_twa = polar_twas.shape[0]
                                    i0 = 0
                                    for ii in range(n_twa - 1, 0, -1):
                                        if polar_twas[ii] <= twa_c:
                                            i0 = ii
                                            break
                                    if i0 >= n_twa - 1:
                                        i0 = n_twa - 2
                                    i1 = i0 + 1
                                    # Fractional position along the TWA axis.
                                    alpha_p = ((twa_c - polar_twas[i0]) /
                                               (polar_twas[i1] - polar_twas[i0])
                                               if polar_twas[i1] > polar_twas[i0]
                                               else 0.0)
                                    oma = 1.0 - alpha_p
                                    # Bilinear interpolation over the 2×2 polar cell.
                                    # Rows = TWA axis (i0/i1), columns = TWS axis (j0/j1).
                                    V_kt = (oma * omb * polar_speeds[i0, j0] +
                                            alpha_p * omb * polar_speeds[i1, j0] +
                                            oma * beta_p * polar_speeds[i0, j1] +
                                            alpha_p * beta_p * polar_speeds[i1, j1])
                                    V_ms = V_kt * KNOTS_TO_MS_L

                                # Ground velocity = boat-through-water vector + current vector.
                                gx = V_ms * sweep_hx[h] + cu
                                gy = V_ms * sweep_hy[h] + cv
                                # Along-track component of ground velocity.
                                sog_h = gx * d_hat_x + gy * d_hat_y
                                # Cross-track component — reject if it swamps forward progress.
                                drift_h = np.abs(gx * (-d_hat_y) + gy * d_hat_x)
                                prog = max(sog_h, 1e-6)
                                if sog_h > 0.01 and drift_h <= 0.50 * prog:
                                    if sog_h > best_sog:
                                        best_sog = sog_h
                                        best_idx = h

                            # --- Pass 2: refine around the coarse winner ---
                            # Re-evaluate the (2*coarse_step - 1) headings bracketing
                            # best_idx at full 2° resolution to recover the true optimum.
                            if coarse_step > 1 and best_idx >= 0:
                                for off in range(-(coarse_step - 1), coarse_step):
                                    h = (best_idx + off) % n_sweep  # wraps at 360°
                                    delta_h = sweep_rads[h] - wind_from_rad
                                    delta_h = (delta_h + PI) % (2.0 * PI) - PI
                                    twa_deg = np.abs(delta_h) * (180.0 / PI)

                                    if twa_deg < polar_min_twa:
                                        V_ms = 0.0
                                    else:
                                        twa_c = twa_deg
                                        if twa_c < polar_twas[0]:
                                            twa_c = polar_twas[0]
                                        elif twa_c > polar_twas[-1]:
                                            twa_c = polar_twas[-1]
                                        n_twa = polar_twas.shape[0]
                                        i0 = 0
                                        for ii in range(n_twa - 1, 0, -1):
                                            if polar_twas[ii] <= twa_c:
                                                i0 = ii
                                                break
                                        if i0 >= n_twa - 1:
                                            i0 = n_twa - 2
                                        i1 = i0 + 1
                                        alpha_p = ((twa_c - polar_twas[i0]) /
                                                   (polar_twas[i1] - polar_twas[i0])
                                                   if polar_twas[i1] > polar_twas[i0]
                                                   else 0.0)
                                        oma = 1.0 - alpha_p
                                        V_kt = (oma * omb * polar_speeds[i0, j0] +
                                                alpha_p * omb * polar_speeds[i1, j0] +
                                                oma * beta_p * polar_speeds[i0, j1] +
                                                alpha_p * beta_p * polar_speeds[i1, j1])
                                        V_ms = V_kt * KNOTS_TO_MS_L

                                    gx = V_ms * sweep_hx[h] + cu
                                    gy = V_ms * sweep_hy[h] + cv
                                    sog_h = gx * d_hat_x + gy * d_hat_y
                                    drift_h = np.abs(gx * (-d_hat_y) + gy * d_hat_x)
                                    prog = max(sog_h, 1e-6)
                                    if sog_h > 0.01 and drift_h <= 0.50 * prog:
                                        if sog_h > best_sog:
                                            best_sog = sog_h

                else:
                    # --- Fixed-speed mode (no polar): crabbing formula ---
                    # Decompose current into along-track and cross-track components.
                    # The boat crabs to cancel c_perp; if c_perp >= boat_speed the
                    # edge is impassable (current sweeps the boat sideways faster
                    # than it can swim).
                    c_par = cu * d_hat_x + cv * d_hat_y   # current along desired track
                    c_px = cu - c_par * d_hat_x            # cross-track current (east)
                    c_py = cv - c_par * d_hat_y            # cross-track current (north)
                    c_perp_sq = c_px * c_px + c_py * c_py
                    if c_perp_sq < bs2:
                        # SOG = water-speed component along track + along-track current.
                        s = np.sqrt(bs2 - c_perp_sq) + c_par
                        if s > 0.01:
                            best_sog = s

                if best_sog <= 0.01:
                    continue                         # edge is impassable; skip
                dt = dist / best_sog                # travel time for this edge (seconds)

                # --- Tack penalty ---
                # Apply a time penalty when the boat changes direction by more than
                # tack_threshold_deg between the incoming and outgoing sectors.
                penalty = 0.0
                if tack_penalty_s > 0.0 and bucket_in != start_sentinel:
                    diff = abs(bucket_in - sector_out)
                    if diff > n_sectors // 2:
                        diff = n_sectors - diff      # shortest angular distance across sectors
                    if float(diff) * SECTOR_WIDTH > tack_threshold_deg:
                        penalty = tack_penalty_s

                new_cost = cost + dt + penalty
                if new_cost < best_cost[nb, sector_out]:
                    # Better path found — update tables and push to heap.
                    best_cost[nb, sector_out] = new_cost
                    arrival_time[nb, sector_out] = arr_t + dt + penalty
                    came_from_node[nb, sector_out] = node
                    came_from_bucket[nb, sector_out] = bucket_in

                    if heap_size < max_heap:
                        hdx2 = goal_x - node_x[nb]
                        hdy2 = goal_y - node_y[nb]
                        # f = g + h (admissible heuristic: straight-line / max_speed).
                        new_f = new_cost + np.sqrt(hdx2 * hdx2 + hdy2 * hdy2) / max_speed
                        heap[heap_size, 0] = new_f
                        heap[heap_size, 1] = new_cost
                        heap[heap_size, 2] = float(nb)
                        heap[heap_size, 3] = float(sector_out)
                        heap_size += 1
                        # Sift the new entry up to restore the min-heap property.
                        i = heap_size - 1
                        while i > 0:
                            parent = (i - 1) // 2
                            if heap[i, 0] < heap[parent, 0]:
                                for col in range(4):
                                    tmp = heap[i, col]
                                    heap[i, col] = heap[parent, col]
                                    heap[parent, col] = tmp
                                i = parent
                            else:
                                break
                    else:
                        # Frontier overflow: abort instead of silently dropping states.
                        return (best_cost, arrival_time,
                                came_from_node, came_from_bucket, -2)

        # explored >= 0 : normal exit (goal reached or graph exhausted)
        # explored == -1: max_iterations cap hit
        return best_cost, arrival_time, came_from_node, came_from_bucket, explored

else:
    _sector_astar_jit = None


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
    # For direction i: (dr, dc) is the row/col offset.  dc > 0 = east,
    # dr > 0 = north (ys is ascending).  arctan2(dc, dr) gives the
    # compass-style bearing (0 = north, 90 = east) for consistent
    # angle-difference computation.
    _N_DIRS = 9  # 0-7 grid directions + 8 start sentinel
    _DIR_ANGLES = np.array([
        np.degrees(np.arctan2(dc, dr))
        for dr, dc in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    ])

    @classmethod
    def _build_angle_diff(cls):
        mat = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                d = abs(cls._DIR_ANGLES[i] - cls._DIR_ANGLES[j]) % 360.0
                mat[i, j] = min(d, 360.0 - d)
        return mat

    _ANGLE_DIFF = None  # populated below after class definition

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

    # Tight drift tolerance for smoothing: ensures shortcuts that bypass
    # tacking are evaluated at the actual pointing-limited speed, not VMG.
    _EDGE_DRIFT_TOL = 0.10

    # Relaxed drift tolerance for final route time: allows VMG-based
    # evaluation so that upwind segments yield finite times.
    _VMG_DRIFT_TOL = 2.0

    def _segment_travel_time(self, x0, y0, x1, y1, start_time_s,
                             n_samples=None, drift_tol=None):
        """Compute travel time (s) along an arbitrary straight segment
        by sampling currents (and wind if available) at multiple points.

        Parameters
        ----------
        drift_tol : float or None
            Override for the _solve_heading drift tolerance.  When None,
            uses ``_VMG_DRIFT_TOL`` (relaxed, for final route timing).
            The smoother passes ``_EDGE_DRIFT_TOL`` to prevent shortcuts
            from erasing tacking legs.

        Returns (time_s, dist_m).  time_s is np.inf if impassable.
        """
        if drift_tol is None:
            drift_tol = self._VMG_DRIFT_TOL

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
                                       wu, wv, self.boat,
                                       drift_tol=drift_tol)
                if v_sog <= 0.01:
                    return np.inf, dist
            else:
                v_sog = _fixed_speed_sog(d_hat_x, d_hat_y, cu, cv,
                                         self.boat.speed())
                if not np.isfinite(v_sog):
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

        The raw path baseline uses relaxed drift tolerance (VMG speed) so
        that diagonal tacking legs get their true fast time.  Shortcut
        candidates use tight drift tolerance so that straight lines
        through the no-go zone are penalised.

        Returns a new (shorter) list of (row, col) tuples.
        """
        if len(path_rc) <= 2:
            return path_rc

        dt_tight = self._EDGE_DRIFT_TOL

        # Pre-compute cumulative travel time along the raw grid path
        # using relaxed drift_tol so diagonal tacking steps evaluate at
        # their true (fast) VMG speed, not the pointing-limited speed.
        n = len(path_rc)
        cum_time = np.zeros(n)
        for k in range(1, n):
            rk0, ck0 = path_rc[k - 1]
            rk1, ck1 = path_rc[k]
            x0, y0 = xs[ck0], ys[rk0]
            x1, y1 = xs[ck1], ys[rk1]
            t_k, _ = self._segment_travel_time(
                x0, y0, x1, y1, arrival_time[rk0, ck0],
                drift_tol=self._VMG_DRIFT_TOL)
            cum_time[k] = cum_time[k - 1] + t_k

        # Pre-compute no-go zone check data for shortcut rejection.
        min_twa = self.boat.polar.minimum_twa if self._use_polar else 0.0

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

                # Reject shortcuts whose bearing falls in the no-go zone.
                if min_twa > 0 and self.wind is not None:
                    dx = xj - x0
                    dy = yj - y0
                    seg_heading = np.arctan2(dy, dx)
                    mx = 0.5 * (x0 + xj)
                    my = 0.5 * (y0 + yj)
                    t_mid = arrival_time[r0, c0]
                    wu, wv = self.wind.query(mx, my, elapsed_s=t_mid)
                    twa = compute_twa(seg_heading, wu, wv)
                    if twa < min_twa:
                        continue

                t_shortcut, _ = self._segment_travel_time(
                    x0, y0, xj, yj, arrival_time[r0, c0],
                    drift_tol=dt_tight)

                if not np.isfinite(t_shortcut):
                    continue

                if not (np.isfinite(cum_time[j]) and np.isfinite(cum_time[i])):
                    continue
                t_grid = cum_time[j] - cum_time[i]
                if t_shortcut <= t_grid * 1.005:
                    best_j = j
                    break

            smoothed.append(path_rc[best_j])
            i = best_j

        return smoothed

    def _remove_stubs(self, path_rc, xs, ys, water_mask=None):
        """Remove stub waypoints that create Y-junctions.

        A stub is an intermediate waypoint where one adjacent leg is
        much shorter than the other and the turn angle is large.
        Removing the stub lets the route go directly from the previous
        to the next waypoint, eliminating the visual and navigational
        artifact.  The check is purely geometric (no time comparison)
        so it runs fast.

        Stubs are NOT removed when doing so would create a segment
        whose bearing falls inside the polar no-go zone.
        """
        if len(path_rc) <= 2:
            return path_rc

        min_twa = self.boat.polar.minimum_twa if self._use_polar else 0.0

        changed = True
        while changed:
            changed = False
            new_path = [path_rc[0]]
            i = 1
            while i < len(path_rc) - 1:
                prev = new_path[-1]
                ax, ay = xs[prev[1]], ys[prev[0]]
                bx, by = xs[path_rc[i][1]],   ys[path_rc[i][0]]
                cx, cy = xs[path_rc[i+1][1]], ys[path_rc[i+1][0]]

                leg_in = np.hypot(bx - ax, by - ay)
                leg_out = np.hypot(cx - bx, cy - by)
                short_leg = min(leg_in, leg_out)
                long_leg = max(leg_in, leg_out)

                if short_leg > 0 and long_leg / short_leg > 3.0:
                    v1x, v1y = bx - ax, by - ay
                    v2x, v2y = cx - bx, cy - by
                    dot = v1x * v2x + v1y * v2y
                    cross = v1x * v2y - v1y * v2x
                    turn = abs(np.degrees(np.arctan2(cross, dot)))
                    if turn > 45.0:
                        if (water_mask is not None and not self._line_of_sight(
                                prev[0], prev[1],
                                path_rc[i+1][0], path_rc[i+1][1],
                                water_mask)):
                            new_path.append(path_rc[i])
                            i += 1
                            continue
                        # Guard: don't remove if the resulting A->C
                        # segment would enter the no-go zone.
                        if min_twa > 0 and self.wind is not None:
                            dx_ac = cx - ax
                            dy_ac = cy - ay
                            hd = np.arctan2(dy_ac, dx_ac)
                            wu, wv = self.wind.query(
                                0.5 * (ax + cx), 0.5 * (ay + cy))
                            twa = compute_twa(hd, wu, wv)
                            if twa < min_twa:
                                new_path.append(path_rc[i])
                                i += 1
                                continue
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

        v_sog = _fixed_speed_sog(d_hat_x, d_hat_y, cu, cv, self.boat.speed())
        if not np.isfinite(v_sog):
            return np.inf
        return dist / v_sog

    # ------------------------------------------------------------------
    #  A* search
    # ------------------------------------------------------------------

    def find_route(self, start_latlon, end_latlon, start_time_s=0.0,
                   max_iterations=2_000_000, return_debug=False):
        """Find the time-optimal route between two lat/lon points.

        Parameters
        ----------
        start_latlon : (lat, lon)
        end_latlon : (lat, lon)
        start_time_s : float
            Elapsed seconds from the forecast reference time at departure.
        max_iterations : int
            Hard limit on A* iterations to prevent runaway searches.
            Raises RuntimeError if exceeded.
        return_debug : bool
            If True, returns a 5th element: dict with 'raw_path_rc',
            'smoothed_path_rc', 'xs', 'ys', 'water_mask'.

        Returns
        -------
        (Route, xs, ys, water_mask[, debug_info])
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
            if explored > max_iterations:
                raise RuntimeError(
                    f"A* exceeded {max_iterations} iterations -- "
                    "route may be unreachable or grid too large.")

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

                # Tacking penalty: course change beyond threshold adds time cost
                # and advances the physical clock used for current/wind look-ups.
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
        path_rc = self._remove_stubs(path_rc, xs, ys, water_mask=water_mask)

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

        sim_track, sim_track_times = self.simulate_track(waypoints_utm, start_time_s)

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
            simulated_track_times=sim_track_times,
        )

        wall_s = _time.monotonic() - t_wall_0
        print(f"\nA* completed in {wall_s:.2f}s, explored {explored} nodes")
        print(f"Path smoothing: {n_raw} -> {n_smooth} waypoints")
        print(route.summary())

        if return_debug:
            debug_info = {
                'raw_path_rc': raw_path_rc,
                'smoothed_path_rc': path_rc,
                'xs': xs,
                'ys': ys,
                'water_mask': water_mask,
            }
            return route, xs, ys, water_mask, debug_info

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
                                       wu, wv, self.boat,
                                       drift_tol=self._VMG_DRIFT_TOL)
                if v_sog <= 0.01:
                    return np.inf, total_dist
            else:
                v_sog = _fixed_speed_sog(d_hat_x, d_hat_y, cu, cv,
                                         self.boat.speed())
                if not np.isfinite(v_sog):
                    return np.inf, total_dist

            dt = seg_len / v_sog
            total_time += dt
            elapsed += dt

        return total_time, total_dist

    # ------------------------------------------------------------------
    #  Ground-track simulation
    # ------------------------------------------------------------------

    def simulate_track(self, waypoints_utm, start_time_s=0.0, dt_s=10.0):
        """Sample the planned ground track into a dense time series.

        The route solver already returns a polyline of feasible ground-track
        waypoints.  For visualisation/export, we densify each segment using the
        same travel-time model as the solver and linearly interpolate position
        along that segment.  This keeps the exported track continuous and its
        timing consistent with ``route.total_time_s``.

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
        tuple[list[(x, y)], list[float]]
            ``(track, track_times)`` — position list and parallel
            list of elapsed-seconds values for each point.
        """
        if len(waypoints_utm) < 2:
            pts = list(waypoints_utm)
            return pts, [start_time_s] * len(pts)

        track = []
        track_times = []
        elapsed = start_time_s
        px, py = waypoints_utm[0]
        track.append((px, py))
        track_times.append(elapsed)

        for wp_idx in range(1, len(waypoints_utm)):
            tx, ty = waypoints_utm[wp_idx]
            seg_time, seg_dist = self._segment_travel_time(
                px, py, tx, ty, elapsed)
            if not np.isfinite(seg_time) or seg_dist < 1e-6:
                px, py = tx, ty
                track.append((px, py))
                track_times.append(elapsed)
                continue

            # Use enough samples for both temporal smoothness and spatial
            # smoothness, while always landing exactly on the segment endpoint.
            n_steps = max(
                1,
                int(np.ceil(seg_time / dt_s)),
                int(np.ceil(seg_dist / max(self.resolution / 2.0, 1.0))),
            )

            x0, y0 = px, py
            for step_idx in range(1, n_steps + 1):
                alpha = step_idx / n_steps
                px = x0 + alpha * (tx - x0)
                py = y0 + alpha * (ty - y0)
                t = elapsed + alpha * seg_time
                track.append((px, py))
                track_times.append(t)

            elapsed += seg_time

        return track, track_times


# Populate the classmethod-built table now that the class is defined.
Router._ANGLE_DIFF = Router._build_angle_diff()


# ===================================================================
#  MeshRouter -- time-dependent A* on the SSCOFS Delaunay mesh
# ===================================================================

class MeshRouter:
    """Find the time-optimal route using A* directly on the SSCOFS mesh.

    Instead of overlaying a regular grid, this router uses the natural
    Delaunay triangulation of SSCOFS element centers as the search graph.
    Edges connect nodes that share a valid water triangle. This provides:

    - Exact velocity at nodes (no IDW interpolation for edge costs)
    - Adaptive resolution (dense near coast, coarser in open water)
    - Alignment with the same water boundary used for display

    State space is (node_id, bearing_bucket) where bearing_bucket quantizes
    the incoming direction into 16 bins (22.5 deg each) for tacking penalty.
    """

    N_BEARING_BUCKETS = 16
    BEARING_BUCKET_DEG = 360.0 / N_BEARING_BUCKETS
    START_SENTINEL = N_BEARING_BUCKETS  # bucket index for start node

    def __init__(self, current_field: CurrentField, boat: BoatModel,
                 wind: 'WindField | None' = None,
                 tack_penalty_s: float = 60.0,
                 tack_threshold_deg: float = 90.0):
        """
        Parameters
        ----------
        current_field : CurrentField
            Must have delaunay and valid_triangle attributes set.
        boat : BoatModel
            Boat performance model.
        wind : WindField, optional
            Wind field for polar-based routing.
        tack_penalty_s : float
            Time penalty (seconds) added for course changes > tack_threshold_deg.
        tack_threshold_deg : float
            Minimum angle change that triggers tack penalty.
        """
        self.cf = current_field
        self.boat = boat
        self.wind = wind
        self.tack_penalty_s = tack_penalty_s
        self.tack_threshold_deg = tack_threshold_deg

        if current_field.delaunay is None or current_field.valid_triangle is None:
            raise ValueError("CurrentField must have Delaunay triangulation for MeshRouter")

        self.adj, self.edge_dist = build_mesh_adjacency(
            current_field.delaunay, current_field.valid_triangle)

        self.node_x = current_field._x_utm
        self.node_y = current_field._y_utm
        self.n_nodes = len(self.node_x)

        # Pre-build CSR adjacency for Numba kernel (built once, reused every route)
        self._adj_csr_offsets, self._adj_csr_targets = _build_csr_adjacency(
            self.adj, self.n_nodes)

        # Stack current frames into contiguous 2D arrays: shape (n_frames, n_nodes)
        self._u_frames_2d = np.stack(current_field.u_frames).astype(np.float64)
        self._v_frames_2d = np.stack(current_field.v_frames).astype(np.float64)

    @property
    def _use_polar(self):
        return self.wind is not None and self.boat.polar is not None

    def _bearing_bucket(self, dx, dy):
        """Convert a direction vector to a bearing bucket index (0-15)."""
        angle_deg = np.degrees(np.arctan2(dx, dy)) % 360.0
        return int(angle_deg / self.BEARING_BUCKET_DEG) % self.N_BEARING_BUCKETS

    def _bucket_angle_diff(self, bucket1, bucket2):
        """Compute minimum angle difference between two bearing buckets."""
        if bucket1 == self.START_SENTINEL or bucket2 == self.START_SENTINEL:
            return 0.0
        diff = abs(bucket1 - bucket2)
        if diff > self.N_BEARING_BUCKETS // 2:
            diff = self.N_BEARING_BUCKETS - diff
        return diff * self.BEARING_BUCKET_DEG

    def _interpolate_velocity(self, node_a, node_b, elapsed_s):
        """Get current velocity at edge midpoint via direct node lookup.

        Uses time interpolation between frames, but no spatial IDW.
        Returns (cu, cv) in m/s.
        """
        cf = self.cf
        t = np.clip(elapsed_s, cf.frame_times[0], cf.frame_times[-1])

        if cf.n_frames == 1:
            ua = cf.u_frames[0][node_a]
            va = cf.v_frames[0][node_a]
            ub = cf.u_frames[0][node_b]
            vb = cf.v_frames[0][node_b]
        else:
            idx = np.searchsorted(cf.frame_times, t, side='right') - 1
            idx = int(np.clip(idx, 0, cf.n_frames - 2))
            t0 = cf.frame_times[idx]
            t1 = cf.frame_times[idx + 1]
            alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0.0

            ua = cf.u_frames[idx][node_a] * (1 - alpha) + cf.u_frames[idx + 1][node_a] * alpha
            va = cf.v_frames[idx][node_a] * (1 - alpha) + cf.v_frames[idx + 1][node_a] * alpha
            ub = cf.u_frames[idx][node_b] * (1 - alpha) + cf.u_frames[idx + 1][node_b] * alpha
            vb = cf.v_frames[idx][node_b] * (1 - alpha) + cf.v_frames[idx + 1][node_b] * alpha

        cu = 0.5 * (ua + ub)
        cv = 0.5 * (va + vb)
        return cu, cv

    def _edge_cost(self, node_a, node_b, arrival_time_s):
        """Compute travel time (seconds) from node_a to node_b.

        Returns np.inf if the edge is impassable.
        """
        x1, y1 = self.node_x[node_a], self.node_y[node_a]
        x2, y2 = self.node_x[node_b], self.node_y[node_b]

        dx = x2 - x1
        dy = y2 - y1
        dist = np.hypot(dx, dy)
        if dist < 1e-6:
            return 0.0

        d_hat_x = dx / dist
        d_hat_y = dy / dist

        cu, cv = self._interpolate_velocity(node_a, node_b, arrival_time_s)

        if self._use_polar:
            mx, my = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
            wu, wv = self.wind.query(mx, my, elapsed_s=arrival_time_s)
            v_sog = _solve_heading(d_hat_x, d_hat_y, cu, cv, wu, wv, self.boat)
            if v_sog <= 0.01:
                return np.inf
            return dist / v_sog

        v_sog = _fixed_speed_sog(d_hat_x, d_hat_y, cu, cv, self.boat.speed())
        if not np.isfinite(v_sog):
            return np.inf
        return dist / v_sog

    def _heuristic(self, node, goal_x, goal_y, max_speed):
        """Admissible heuristic: straight-line distance / max possible speed."""
        dx = goal_x - self.node_x[node]
        dy = goal_y - self.node_y[node]
        return np.hypot(dx, dy) / max_speed

    def _segment_travel_time(self, x1, y1, x2, y2, start_time_s, n_samples=10):
        """Compute travel time along a straight segment using CurrentField.query.

        Used for path smoothing and ground-track simulation where we need
        continuous spatial queries (not just at mesh nodes).
        """
        dist = np.hypot(x2 - x1, y2 - y1)
        if dist < 1e-6:
            return 0.0, dist

        d_hat_x = (x2 - x1) / dist
        d_hat_y = (y2 - y1) / dist

        seg_len = dist / n_samples
        elapsed = start_time_s
        total_time = 0.0

        for i in range(n_samples):
            frac = (i + 0.5) / n_samples
            mx = x1 + frac * (x2 - x1)
            my = y1 + frac * (y2 - y1)

            cu, cv = self.cf.query(mx, my, elapsed_s=elapsed)
            if np.isnan(cu):
                return np.inf, dist

            if self._use_polar:
                wu, wv = self.wind.query(mx, my, elapsed_s=elapsed)
                v_sog = _solve_heading(d_hat_x, d_hat_y, cu, cv, wu, wv, self.boat)
                if v_sog <= 0.01:
                    return np.inf, dist
            else:
                v_sog = _fixed_speed_sog(d_hat_x, d_hat_y, cu, cv, self.boat.speed())
                if not np.isfinite(v_sog):
                    return np.inf, dist

            dt = seg_len / v_sog
            total_time += dt
            elapsed += dt

        return total_time, dist

    def _line_of_sight(self, x1, y1, x2, y2, n_samples=20):
        """Check if a straight line between two points crosses land.

        Samples points along the line and checks CurrentField.query for NaN.
        Returns True if all samples are over water.
        """
        for i in range(n_samples + 1):
            frac = i / n_samples
            mx = x1 + frac * (x2 - x1)
            my = y1 + frac * (y2 - y1)
            cu, _ = self.cf.query(mx, my, elapsed_s=0.0)
            if np.isnan(cu):
                return False
        return True

    def _smooth_path(self, path_nodes, arrival_times):
        """Remove unnecessary waypoints via greedy string-pulling.

        A shortcut is accepted if:
        1. Line-of-sight check passes (no land)
        2. Travel time via shortcut is no worse than original path segments
        """
        if len(path_nodes) <= 2:
            return path_nodes

        smoothed = [path_nodes[0]]
        smoothed_times = [arrival_times[0]]
        i = 0

        while i < len(path_nodes) - 1:
            best_j = i + 1
            best_time = arrival_times[i + 1] - arrival_times[i]

            for j in range(i + 2, len(path_nodes)):
                x1 = self.node_x[path_nodes[i]]
                y1 = self.node_y[path_nodes[i]]
                x2 = self.node_x[path_nodes[j]]
                y2 = self.node_y[path_nodes[j]]

                if not self._line_of_sight(x1, y1, x2, y2):
                    break

                shortcut_time, _ = self._segment_travel_time(
                    x1, y1, x2, y2, smoothed_times[-1])
                original_time = arrival_times[j] - arrival_times[i]

                if np.isfinite(shortcut_time) and shortcut_time <= original_time * 1.05:
                    best_j = j
                    best_time = shortcut_time

            smoothed.append(path_nodes[best_j])
            smoothed_times.append(smoothed_times[-1] + best_time)
            i = best_j

        return smoothed

    def _remove_stubs(self, path_nodes):
        """Remove stub waypoints that create Y-junctions.

        A stub is an intermediate waypoint where one adjacent leg is much
        shorter than the other and the turn angle is large.
        """
        if len(path_nodes) <= 2:
            return path_nodes

        result = [path_nodes[0]]
        for i in range(1, len(path_nodes) - 1):
            prev_node = result[-1]
            curr_node = path_nodes[i]
            next_node = path_nodes[i + 1]

            x0, y0 = self.node_x[prev_node], self.node_y[prev_node]
            x1, y1 = self.node_x[curr_node], self.node_y[curr_node]
            x2, y2 = self.node_x[next_node], self.node_y[next_node]

            leg_in = np.hypot(x1 - x0, y1 - y0)
            leg_out = np.hypot(x2 - x1, y2 - y1)

            if leg_in < 1e-6 or leg_out < 1e-6:
                result.append(curr_node)
                continue

            ratio = max(leg_in, leg_out) / min(leg_in, leg_out)

            v_in = ((x1 - x0) / leg_in, (y1 - y0) / leg_in)
            v_out = ((x2 - x1) / leg_out, (y2 - y1) / leg_out)
            dot = v_in[0] * v_out[0] + v_in[1] * v_out[1]
            turn_angle = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))

            is_stub = ratio > 3.0 and turn_angle > 45.0

            if is_stub and self._line_of_sight(x0, y0, x2, y2):
                shortcut_time, _ = self._segment_travel_time(x0, y0, x2, y2, 0.0)
                if np.isfinite(shortcut_time):
                    continue
            result.append(curr_node)

        result.append(path_nodes[-1])
        return result

    def find_route(self, start_latlon, end_latlon, start_time_s=0.0,
                   max_iterations=2_000_000, return_debug=False):
        """Find the time-optimal route between two lat/lon points.

        Parameters
        ----------
        start_latlon : (lat, lon)
        end_latlon : (lat, lon)
        start_time_s : float
            Elapsed seconds from the forecast reference time at departure.
        max_iterations : int
            Hard limit on A* iterations.
        return_debug : bool
            If True, returns a 5th element: dict with 'raw_path' (node ids),
            'raw_times', 'smoothed_path', and 'perf' timing breakdown.

        Returns
        -------
        (Route, xs, ys, water_mask[, debug_info])
        """
        _perf = {}
        t_total = _time.monotonic()

        # ------------------------------------------------------------------
        # Setup: coordinate transform + snap start/end to mesh nodes
        # ------------------------------------------------------------------
        t0 = _time.monotonic()
        transformer = self.cf.transformer
        sx, sy = transformer.transform(start_latlon[1], start_latlon[0])
        ex, ey = transformer.transform(end_latlon[1], end_latlon[0])

        _, start_node = self.cf.tree.query([sx, sy])
        _, end_node = self.cf.tree.query([ex, ey])
        start_node = int(start_node)
        end_node = int(end_node)

        if start_node not in self.adj:
            raise ValueError(f"Start point snapped to isolated node {start_node}")
        if end_node not in self.adj:
            raise ValueError(f"End point snapped to isolated node {end_node}")

        goal_x, goal_y = self.node_x[end_node], self.node_y[end_node]
        boat_speed = self.boat.speed()
        max_speed = boat_speed + self.cf.max_current_speed
        _perf['setup'] = _time.monotonic() - t0

        _use_numba = _NUMBA_AVAILABLE and not self._use_polar and _mesh_astar_jit is not None
        _mode = ('Numba JIT' if _use_numba
                 else ('Python + polar' if self._use_polar else 'Python'))
        print(f"MeshRouter: {self.n_nodes:,} nodes, {len(self.adj):,} with edges  [{_mode}]")
        print(f"Start node: {start_node}, End node: {end_node}")

        # ------------------------------------------------------------------
        # A* search
        # ------------------------------------------------------------------
        t0 = _time.monotonic()

        if _use_numba:
            best_cost_arr, arrival_arr, cf_node_arr, cf_bucket_arr, explored = \
                _mesh_astar_jit(
                    self._adj_csr_offsets, self._adj_csr_targets,
                    self.node_x, self.node_y,
                    self._u_frames_2d, self._v_frames_2d,
                    self.cf.frame_times,
                    start_node, end_node,
                    start_time_s, boat_speed, max_speed,
                    self.tack_penalty_s, self.tack_threshold_deg,
                    self.N_BEARING_BUCKETS, self.START_SENTINEL,
                    max_iterations,
                )
            explored = int(explored)
            _perf['astar'] = _time.monotonic() - t0

            if explored < 0:
                if explored == -1:
                    raise RuntimeError(
                        f"A* exceeded {max_iterations} iterations (Numba kernel) -- "
                        "route may be unreachable.")
                if explored == -2:
                    raise RuntimeError(
                        "A* frontier heap overflow in Numba kernel -- "
                        "increase heap sizing or reduce search space.")
                raise RuntimeError("A* failed in Numba kernel (internal error).")

            # Path reconstruction from numpy arrays
            t0 = _time.monotonic()
            n_buckets = self.N_BEARING_BUCKETS + 1
            goal_costs = best_cost_arr[end_node, :]
            final_bucket = int(np.argmin(goal_costs))
            if not np.isfinite(goal_costs[final_bucket]):
                raise RuntimeError("No path found between start and end")

            raw_path = []
            raw_times = []
            node = end_node
            bucket = final_bucket
            while True:
                raw_path.append(node)
                raw_times.append(float(arrival_arr[node, bucket]))
                prev_node = int(cf_node_arr[node, bucket])
                if prev_node < 0:
                    break
                prev_bucket = int(cf_bucket_arr[node, bucket])
                node = prev_node
                bucket = prev_bucket
            raw_path.reverse()
            raw_times.reverse()
            _perf['reconstruct'] = _time.monotonic() - t0

        else:
            # Python fallback (also handles polar/wind mode)
            INF = float('inf')
            N_BUCKETS = self.N_BEARING_BUCKETS + 1
            best_cost = {}
            came_from = {}
            arrival_time = {}
            start_state = (start_node, self.START_SENTINEL)
            best_cost[start_state] = 0.0
            arrival_time[start_state] = start_time_s
            open_set = [(self._heuristic(start_node, goal_x, goal_y, max_speed),
                         0.0, start_node, self.START_SENTINEL)]
            explored = 0

            while open_set:
                if explored >= max_iterations:
                    raise RuntimeError(f"A* exceeded {max_iterations} iterations")
                _, cost, node, bucket_in = heapq.heappop(open_set)
                state = (node, bucket_in)
                if cost > best_cost.get(state, INF):
                    continue
                explored += 1
                if node == end_node:
                    break
                arr_t = arrival_time[state]
                for neighbor in self.adj.get(node, []):
                    dx = self.node_x[neighbor] - self.node_x[node]
                    dy = self.node_y[neighbor] - self.node_y[node]
                    bucket_out = self._bearing_bucket(dx, dy)
                    dt = self._edge_cost(node, neighbor, arr_t)
                    if dt == INF:
                        continue
                    penalty = 0.0
                    angle_diff = self._bucket_angle_diff(bucket_in, bucket_out)
                    if self.tack_penalty_s > 0 and angle_diff > self.tack_threshold_deg:
                        penalty = self.tack_penalty_s
                    new_cost = cost + dt + penalty
                    new_state = (neighbor, bucket_out)
                    if new_cost < best_cost.get(new_state, INF):
                        best_cost[new_state] = new_cost
                        arrival_time[new_state] = arr_t + dt + penalty
                        came_from[new_state] = state
                        h = self._heuristic(neighbor, goal_x, goal_y, max_speed)
                        heapq.heappush(open_set, (new_cost + h, new_cost, neighbor, bucket_out))

            _perf['astar'] = _time.monotonic() - t0

            goal_states = [(best_cost.get((end_node, b), INF), b)
                           for b in range(N_BUCKETS)]
            best_goal_cost, final_bucket = min(goal_states)
            if best_goal_cost == INF:
                raise RuntimeError("No path found between start and end")

            t0 = _time.monotonic()
            raw_path = []
            raw_times = []
            state = (end_node, final_bucket)
            while state in came_from or state == start_state:
                node, _ = state
                raw_path.append(node)
                raw_times.append(arrival_time.get(state, start_time_s))
                if state == start_state:
                    break
                state = came_from.get(state)
                if state is None:
                    break
            raw_path.reverse()
            raw_times.reverse()
            _perf['reconstruct'] = _time.monotonic() - t0

        # ------------------------------------------------------------------
        # Path smoothing
        # ------------------------------------------------------------------
        t0 = _time.monotonic()
        path_nodes = self._smooth_path(raw_path, raw_times)
        path_nodes = self._remove_stubs(path_nodes)
        _perf['smooth'] = _time.monotonic() - t0

        # ------------------------------------------------------------------
        # Route timing and waypoints
        # ------------------------------------------------------------------
        t0 = _time.monotonic()
        inv_transformer = Transformer.from_crs(
            transformer.target_crs, transformer.source_crs, always_xy=True)

        def _compute_leg_times(nodes):
            wps = [(self.node_x[n], self.node_y[n]) for n in nodes]
            times, dists, t = [], [], start_time_s
            for k in range(len(wps) - 1):
                x1, y1 = wps[k]
                x2, y2 = wps[k + 1]
                t_seg, d_seg = self._segment_travel_time(x1, y1, x2, y2, t)
                times.append(t_seg)
                dists.append(d_seg)
                t += t_seg
            return wps, times, dists

        def _use_astar_times(nodes, arr_times):
            wps = [(self.node_x[n], self.node_y[n]) for n in nodes]
            times, dists = [], []
            for k in range(len(wps) - 1):
                x1, y1 = wps[k]
                x2, y2 = wps[k + 1]
                dists.append(np.hypot(x2 - x1, y2 - y1))
                times.append(arr_times[k + 1] - arr_times[k])
            return wps, times, dists

        waypoints_utm, leg_times, leg_dists = _compute_leg_times(path_nodes)
        total_time = sum(leg_times)
        if not np.isfinite(total_time):
            path_nodes = raw_path
            waypoints_utm, leg_times, leg_dists = _use_astar_times(path_nodes, raw_times)
            total_time = sum(leg_times)

        total_dist = sum(leg_dists)
        avg_sog = (total_dist / total_time * MS_TO_KNOTS) if (
            total_time > 0 and np.isfinite(total_time)) else 0.0

        waypoints_latlon = []
        for x, y in waypoints_utm:
            lon, lat = inv_transformer.transform(x, y)
            waypoints_latlon.append((lat, lon))
        _perf['route_time'] = _time.monotonic() - t0

        # ------------------------------------------------------------------
        # Track simulation and plotting grid
        # ------------------------------------------------------------------
        t0 = _time.monotonic()
        sim_track, sim_track_times = self._simulate_track(waypoints_utm, start_time_s)
        _perf['simulate'] = _time.monotonic() - t0

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
            simulated_track_times=sim_track_times,
        )

        t0 = _time.monotonic()
        xs, ys, water_mask = self._build_grid_for_plotting(
            waypoints_utm, start_latlon, end_latlon)
        _perf['plot_grid'] = _time.monotonic() - t0

        # ------------------------------------------------------------------
        # Performance report
        # ------------------------------------------------------------------
        wall_s = _time.monotonic() - t_total
        _perf['total'] = wall_s
        n_raw = len(raw_path)
        n_smooth = len(path_nodes)

        print(f"\nMesh A* completed in {wall_s:.2f}s ({_mode}), explored {explored:,} nodes")
        print(f"  Setup:               {_perf['setup']:.3f}s")
        print(f"  A* search:           {_perf['astar']:.3f}s")
        print(f"  Path reconstruction: {_perf['reconstruct']:.3f}s")
        print(f"  Path smoothing:      {_perf['smooth']:.3f}s")
        print(f"  Route timing:        {_perf['route_time']:.3f}s")
        print(f"  Track simulation:    {_perf['simulate']:.3f}s")
        print(f"  Plotting grid:       {_perf['plot_grid']:.3f}s")
        print(f"  TOTAL:               {wall_s:.3f}s")
        if _use_numba and _perf['astar'] > 8.0:
            print("  (Numba JIT compiled on this run — subsequent runs use cache)")
        print(f"Path smoothing: {n_raw} -> {n_smooth} waypoints")
        print(route.summary())

        if return_debug:
            debug_info = {
                'raw_path': raw_path,
                'raw_times': raw_times,
                'explored_nodes': set(raw_path),
                'smoothed_path': path_nodes,
                'perf': _perf,
            }
            return route, xs, ys, water_mask, debug_info

        return route, xs, ys, water_mask

    def _simulate_track(self, waypoints_utm, start_time_s, dt_s=10.0):
        """Sample the ground track into a dense time series."""
        if len(waypoints_utm) < 2:
            return list(waypoints_utm), [start_time_s] * len(waypoints_utm)

        track = []
        track_times = []
        elapsed = start_time_s
        px, py = waypoints_utm[0]
        track.append((px, py))
        track_times.append(elapsed)

        for wp_idx in range(1, len(waypoints_utm)):
            tx, ty = waypoints_utm[wp_idx]
            seg_time, seg_dist = self._segment_travel_time(px, py, tx, ty, elapsed)

            if not np.isfinite(seg_time) or seg_dist < 1e-6:
                px, py = tx, ty
                track.append((px, py))
                track_times.append(elapsed)
                continue

            n_steps = max(1, int(np.ceil(seg_time / dt_s)),
                          int(np.ceil(seg_dist / 150.0)))

            x0, y0 = px, py
            for step_idx in range(1, n_steps + 1):
                alpha = step_idx / n_steps
                px = x0 + alpha * (tx - x0)
                py = y0 + alpha * (ty - y0)
                t = elapsed + alpha * seg_time
                track.append((px, py))
                track_times.append(t)

            elapsed += seg_time

        return track, track_times

    def _build_grid_for_plotting(self, waypoints_utm, start_latlon, end_latlon,
                                  resolution=300.0, padding=2000.0):
        """Generate a regular grid overlay for plot_route compatibility."""
        transformer = self.cf.transformer
        sx, sy = transformer.transform(start_latlon[1], start_latlon[0])
        ex, ey = transformer.transform(end_latlon[1], end_latlon[0])

        all_x = [sx, ex] + [p[0] for p in waypoints_utm]
        all_y = [sy, ey] + [p[1] for p in waypoints_utm]

        x0 = min(all_x) - padding
        x1 = max(all_x) + padding
        y0 = min(all_y) - padding
        y1 = max(all_y) + padding

        xs = np.arange(x0, x1 + resolution, resolution)
        ys = np.arange(y0, y1 + resolution, resolution)

        u_grid, v_grid = self.cf.query_grid(xs, ys, elapsed_s=0.0)
        water_mask = ~np.isnan(u_grid)

        return xs, ys, water_mask

    def straight_line_time(self, start_latlon, end_latlon, start_time_s=0.0):
        """Estimate travel time along a straight line."""
        transformer = self.cf.transformer
        sx, sy = transformer.transform(start_latlon[1], start_latlon[0])
        ex, ey = transformer.transform(end_latlon[1], end_latlon[0])

        total_time, total_dist = self._segment_travel_time(
            sx, sy, ex, ey, start_time_s, n_samples=50)
        return total_time, total_dist


# ===================================================================
#  SectorRouter: Heading-binned connectivity on SSCOFS nodes
# ===================================================================

class SectorRouter:
    """Find the time-optimal route using A* with heading-binned connectivity.

    Instead of using raw Delaunay mesh adjacency (which is tied to the PDE
    mesh topology), this router connects each SSCOFS node to the nearest
    reachable neighbor in each of 32 heading sectors (11.25 deg each).

    This gives the A* 32 angular choices at every decision point, decoupling
    the navigation action lattice from the hydrodynamics mesh while still
    using SSCOFS element centers as routing nodes (inheriting their adaptive
    density: ~100m near shore, ~500m offshore).

    Key differences from MeshRouter:
    - Connectivity: 32 heading sectors vs. Delaunay edge adjacency
    - Edge cost: CurrentField.query at midpoint vs. averaged node velocities
    - Graph build: Lazy (during A*) vs. eager (all edges upfront)
    - Path quality: Directly steerable headings vs. mesh topology wobble
    """

    N_SECTORS = 32
    SECTOR_WIDTH = 360.0 / N_SECTORS  # 11.25 deg
    START_SENTINEL = N_SECTORS  # sector index for start node (no incoming)
    POLAR_SWEEP_COARSE_STEP = 5  # 2 deg base sweep -> 10 deg coarse pass
    CORRIDOR_PAD_FACTORS = (0.85, 1.00, 1.35)  # adaptive retry from tight to wide
    CORRIDOR_CACHE_MAX = 4
    SMOOTH_TACK_THRESHOLD_DEG = 45.0   # turns >= this are tack waypoints
    SMOOTH_DP_EPSILON_M = 120.0        # DP perpendicular tolerance within curved segments
    SMOOTH_MIN_TACK_SPACING_M = 400.0  # minimum distance between consecutive tack waypoints

    def __init__(self, current_field: CurrentField, boat: BoatModel,
                 wind: 'WindField | None' = None,
                 tack_penalty_s: float = 60.0,
                 tack_threshold_deg: float = 90.0,
                 max_edge_m: float = 2000.0,
                 min_edge_m: float = 50.0,
                 k_candidates: int = 50,
                 polar_sweep_coarse_step: int | None = None,
                 corridor_pad_factors: tuple[float, ...] | list[float] | None = None,
                 corridor_cache_max: int | None = None,
                 use_dense_polar: bool = False,
                 use_dot_filter: bool = False):
        """
        Parameters
        ----------
        current_field : CurrentField
            Must have tree (KD-tree) and _x_utm, _y_utm arrays.
        boat : BoatModel
            Boat performance model.
        wind : WindField, optional
            Wind field for polar-based routing.
        tack_penalty_s : float
            Time penalty (seconds) for course changes > tack_threshold_deg.
        tack_threshold_deg : float
            Minimum angle change that triggers tack penalty.
        max_edge_m : float
            Maximum edge length in meters.
        min_edge_m : float
            Minimum edge length in meters (skip very short edges).
        k_candidates : int
            Number of KD-tree neighbors to consider when filling sectors.
        polar_sweep_coarse_step : int, optional
            Coarse stride for polar heading sweep in the Numba kernel.
            1 means exact full sweep. Higher values are faster but approximate.
        corridor_pad_factors : sequence[float], optional
            Adaptive corridor pad multipliers (tight to wide) applied to the
            base pad. Defaults to class constant CORRIDOR_PAD_FACTORS.
        corridor_cache_max : int, optional
            Maximum number of corridor graphs to keep in memory cache.
        use_dense_polar : bool, optional
            When True, use the pre-baked 1°×1kt dense polar lookup table
            instead of binary search for every heading evaluation.  Eliminates
            both the TWS bracket search (per edge) and the TWA bracket search
            (per heading) from the Numba hot loop.  Default False.
        use_dot_filter : bool, optional
            When True, skip any candidate heading whose unit vector has a
            negative dot product with the desired ground track, pruning the
            backward hemisphere (~90 of 180 headings) with a single
            multiply-add.  Works with both dense and sparse polar paths.
            Default False.
        """
        self.cf = current_field
        self.boat = boat
        self.wind = wind
        self.tack_penalty_s = tack_penalty_s
        self.tack_threshold_deg = tack_threshold_deg
        self.max_edge_m = max_edge_m
        self.min_edge_m = min_edge_m
        self.k_candidates = k_candidates
        if polar_sweep_coarse_step is None:
            polar_sweep_coarse_step = self.POLAR_SWEEP_COARSE_STEP
        self._polar_sweep_coarse_step = max(1, int(polar_sweep_coarse_step))

        if corridor_pad_factors is None:
            corridor_pad_factors = self.CORRIDOR_PAD_FACTORS
        pads = [float(v) for v in corridor_pad_factors if float(v) > 0.0]
        if not pads:
            pads = list(self.CORRIDOR_PAD_FACTORS)
        self._corridor_pad_factors = tuple(pads)

        if corridor_cache_max is None:
            corridor_cache_max = self.CORRIDOR_CACHE_MAX
        self._corridor_cache_max = max(1, int(corridor_cache_max))

        if current_field.tree is None:
            raise ValueError("CurrentField must have KD-tree for SectorRouter")

        self.node_x = current_field._x_utm
        self.node_y = current_field._y_utm
        self.n_nodes = len(self.node_x)

        # Lazy neighbor cache: node_id -> list of (neighbor_id, sector, distance)
        self._neighbor_cache = {}
        self._corridor_graph_cache = {}

        # Line-of-sight samples per edge
        self._los_samples = 15

        # Pre-stacked current frames for Numba kernel: (n_frames, n_nodes)
        self._u_frames_2d = np.stack(current_field.u_frames).astype(np.float64)
        self._v_frames_2d = np.stack(current_field.v_frames).astype(np.float64)

        # Pre-compute wind values at SSCOFS nodes for Numba kernel
        self._wind_at_nodes_wu = None  # shape (n_wind_frames, n_nodes)
        self._wind_at_nodes_wv = None
        self._wind_frame_times = None
        self._numba_wind_ready = False
        if wind is not None:
            self._precompute_wind_at_nodes()

        # Dense polar lookup arrays for the Numba kernel
        self._use_dense_polar = bool(use_dense_polar)
        if use_dense_polar and boat.polar is not None:
            self._polar_dense_arr = boat.polar._polar_dense.astype(np.float64)
            self._polar_dense_max_tws = int(boat.polar._polar_dense_max_tws)
            print(f"Dense polar lookup enabled "
                  f"({self._polar_dense_arr.shape[0]}×{self._polar_dense_arr.shape[1]} table)")
        else:
            # Pass dummy arrays when not used; Numba requires concrete types.
            self._polar_dense_arr = np.zeros((1, 1), dtype=np.float64)
            self._polar_dense_max_tws = 0
            self._use_dense_polar = False

        # Dot-product forward-hemisphere filter (works with both dense and sparse paths)
        self._use_dot_filter = bool(use_dot_filter)
        if self._use_dot_filter:
            mode = "dense" if self._use_dense_polar else "sparse"
            print(f"Dot-product forward-hemisphere filter enabled ({mode} polar path)")

    @property
    def _use_polar(self):
        return self.wind is not None and self.boat.polar is not None

    def _corridor_cache_key(self, start_node, end_node, pad_m):
        lo = start_node if start_node < end_node else end_node
        hi = end_node if start_node < end_node else start_node
        return (
            int(lo), int(hi), int(round(pad_m)),
            int(self.N_SECTORS), int(self.k_candidates),
            int(round(self.min_edge_m)), int(round(self.max_edge_m)),
            int(self._los_samples),
        )

    def _corridor_cache_get(self, key):
        val = self._corridor_graph_cache.get(key)
        if val is None:
            return None
        # Touch key for recency.
        self._corridor_graph_cache.pop(key)
        self._corridor_graph_cache[key] = val
        return val

    def _corridor_cache_put(self, key, value):
        if key in self._corridor_graph_cache:
            self._corridor_graph_cache.pop(key)
        self._corridor_graph_cache[key] = value
        while len(self._corridor_graph_cache) > self._corridor_cache_max:
            self._corridor_graph_cache.pop(next(iter(self._corridor_graph_cache)))

    def _precompute_wind_at_nodes(self):
        """Map wind field onto SSCOFS node positions for Numba kernel."""
        w = self.wind
        if w._mode == 'constant':
            self._wind_at_nodes_wu = np.full(
                (1, self.n_nodes), w._wu, dtype=np.float64)
            self._wind_at_nodes_wv = np.full(
                (1, self.n_nodes), w._wv, dtype=np.float64)
            self._wind_frame_times = np.array([0.0], dtype=np.float64)
            self._numba_wind_ready = True
        elif w._mode == 'temporal_const':
            nf = len(w._wu_frames)
            wu = np.empty((nf, self.n_nodes), dtype=np.float64)
            wv = np.empty((nf, self.n_nodes), dtype=np.float64)
            for i in range(nf):
                wu[i, :] = float(w._wu_frames[i])
                wv[i, :] = float(w._wv_frames[i])
            self._wind_at_nodes_wu = wu
            self._wind_at_nodes_wv = wv
            self._wind_frame_times = w._frame_times.astype(np.float64)
            self._numba_wind_ready = True
        elif w._mode == 'temporal_nodes' and w._node_tree is not None:
            pts = np.column_stack([self.node_x, self.node_y])
            _, nn = w._node_tree.query(pts)
            nn = np.asarray(nn, dtype=np.int64).ravel()
            self._wind_at_nodes_wu = np.ascontiguousarray(
                w._wu_frames[:, nn].astype(np.float64))
            self._wind_at_nodes_wv = np.ascontiguousarray(
                w._wv_frames[:, nn].astype(np.float64))
            self._wind_frame_times = w._frame_times.astype(np.float64)
            self._numba_wind_ready = True
        elif w._mode == 'grid':
            # Spatially varying but time-constant wind.
            pts = np.column_stack([self.node_x, self.node_y])
            wu_nodes, wv_nodes = w.query(pts[:, 0], pts[:, 1], elapsed_s=0.0)
            self._wind_at_nodes_wu = np.ascontiguousarray(
                np.asarray(wu_nodes, dtype=np.float64).reshape(1, self.n_nodes))
            self._wind_at_nodes_wv = np.ascontiguousarray(
                np.asarray(wv_nodes, dtype=np.float64).reshape(1, self.n_nodes))
            self._wind_frame_times = np.array([0.0], dtype=np.float64)
            self._numba_wind_ready = True
        elif w._mode == 'temporal_grid':
            # Spatially and temporally varying wind on a regular grid.
            # Pre-sample each frame onto SSCOFS nodes so Numba can use
            # node-indexed arrays during A*.
            n_wf = len(w._frame_times)
            pts = np.column_stack([self.node_x, self.node_y])
            wu = np.empty((n_wf, self.n_nodes), dtype=np.float64)
            wv = np.empty((n_wf, self.n_nodes), dtype=np.float64)
            for i in range(n_wf):
                t_s = float(w._frame_times[i])
                wu_i, wv_i = w.query(pts[:, 0], pts[:, 1], elapsed_s=t_s)
                wu[i, :] = np.asarray(wu_i, dtype=np.float64)
                wv[i, :] = np.asarray(wv_i, dtype=np.float64)
            self._wind_at_nodes_wu = np.ascontiguousarray(wu)
            self._wind_at_nodes_wv = np.ascontiguousarray(wv)
            self._wind_frame_times = w._frame_times.astype(np.float64)
            self._numba_wind_ready = True

    def _bearing_to_sector(self, dx, dy):
        """Convert a direction vector to a sector index (0-(N_SECTORS-1)).

        Sector 0 is centered on North (0 deg), sector 4 on East (90 deg), etc.
        """
        angle_deg = np.degrees(np.arctan2(dx, dy)) % 360.0
        return int(angle_deg / self.SECTOR_WIDTH) % self.N_SECTORS

    def _sector_angle_diff(self, sector1, sector2):
        """Compute minimum angle difference between two sectors."""
        if sector1 == self.START_SENTINEL or sector2 == self.START_SENTINEL:
            return 0.0
        diff = abs(sector1 - sector2)
        if diff > self.N_SECTORS // 2:
            diff = self.N_SECTORS - diff
        return diff * self.SECTOR_WIDTH

    def _line_of_sight(self, x1, y1, x2, y2):
        """Check if a straight line between two points crosses land.

        Samples points along the line and checks CurrentField.query for NaN.
        Returns True if all samples are over water.
        """
        n = self._los_samples
        for i in range(n + 1):
            frac = i / n
            mx = x1 + frac * (x2 - x1)
            my = y1 + frac * (y2 - y1)
            cu, _ = self.cf.query(mx, my, elapsed_s=0.0)
            if np.isnan(cu):
                return False
        return True

    def _find_sector_neighbors(self, node_id):
        """Find reachable neighbors in each heading sector.

        Uses KD-tree to find candidate neighbors, assigns each to a sector,
        keeps the nearest valid candidate per sector.

        Returns list of (neighbor_id, sector, distance) tuples.
        """
        if node_id in self._neighbor_cache:
            return self._neighbor_cache[node_id]

        x0 = self.node_x[node_id]
        y0 = self.node_y[node_id]

        # Query k_candidates nearest neighbors
        dists, idxs = self.cf.tree.query([x0, y0], k=self.k_candidates)
        dists = np.atleast_1d(dists)
        idxs = np.atleast_1d(idxs)

        # Candidate list per sector: sector -> [(distance, neighbor_id), ...]
        sector_candidates = {}

        for d, idx in zip(dists, idxs):
            idx = int(idx)
            if idx == node_id:
                continue
            if d < self.min_edge_m or d > self.max_edge_m:
                continue

            x1 = self.node_x[idx]
            y1 = self.node_y[idx]
            dx = x1 - x0
            dy = y1 - y0
            sector = self._bearing_to_sector(dx, dy)

            if sector not in sector_candidates:
                sector_candidates[sector] = []
            sector_candidates[sector].append((float(d), idx))

        # Validate nearest-first candidates with line-of-sight.
        neighbors = []
        for sector, candidates in sector_candidates.items():
            candidates.sort(key=lambda item: item[0])
            for d, idx in candidates:
                x1 = self.node_x[idx]
                y1 = self.node_y[idx]
                if self._line_of_sight(x0, y0, x1, y1):
                    neighbors.append((idx, sector, d))
                    break

        self._neighbor_cache[node_id] = neighbors
        return neighbors

    def _edge_cost(self, node_a, node_b, dist, arrival_time_s):
        """Compute travel time (seconds) from node_a to node_b.

        Uses CurrentField.query at edge midpoint for current velocity.
        Returns np.inf if the edge is impassable.
        """
        x1, y1 = self.node_x[node_a], self.node_y[node_a]
        x2, y2 = self.node_x[node_b], self.node_y[node_b]

        if dist < 1e-6:
            return 0.0

        d_hat_x = (x2 - x1) / dist
        d_hat_y = (y2 - y1) / dist

        # Query current at edge midpoint
        mx, my = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        cu, cv = self.cf.query(mx, my, elapsed_s=arrival_time_s)

        if np.isnan(cu):
            return np.inf

        if self._use_polar:
            wu, wv = self.wind.query(mx, my, elapsed_s=arrival_time_s)
            v_sog = _solve_heading(d_hat_x, d_hat_y, cu, cv, wu, wv, self.boat)
            if v_sog <= 0.01:
                return np.inf
            return dist / v_sog

        v_sog = _fixed_speed_sog(d_hat_x, d_hat_y, cu, cv, self.boat.speed())
        if not np.isfinite(v_sog):
            return np.inf
        return dist / v_sog

    def _heuristic(self, node, goal_x, goal_y, max_speed):
        """Admissible heuristic: straight-line distance / max possible speed."""
        dx = goal_x - self.node_x[node]
        dy = goal_y - self.node_y[node]
        return np.hypot(dx, dy) / max_speed

    def _segment_travel_time(self, x1, y1, x2, y2, start_time_s, n_samples=10):
        """Compute travel time along a straight segment using CurrentField.query.

        Used for path smoothing and ground-track simulation.
        """
        dist = np.hypot(x2 - x1, y2 - y1)
        if dist < 1e-6:
            return 0.0, dist

        d_hat_x = (x2 - x1) / dist
        d_hat_y = (y2 - y1) / dist

        seg_len = dist / n_samples
        elapsed = start_time_s
        total_time = 0.0

        for i in range(n_samples):
            frac = (i + 0.5) / n_samples
            mx = x1 + frac * (x2 - x1)
            my = y1 + frac * (y2 - y1)

            cu, cv = self.cf.query(mx, my, elapsed_s=elapsed)
            if np.isnan(cu):
                return np.inf, dist

            if self._use_polar:
                wu, wv = self.wind.query(mx, my, elapsed_s=elapsed)
                v_sog = _solve_heading(d_hat_x, d_hat_y, cu, cv, wu, wv, self.boat)
                if v_sog <= 0.01:
                    return np.inf, dist
            else:
                v_sog = _fixed_speed_sog(d_hat_x, d_hat_y, cu, cv, self.boat.speed())
                if not np.isfinite(v_sog):
                    return np.inf, dist

            dt = seg_len / v_sog
            total_time += dt
            elapsed += dt

        return total_time, dist

    # ------------------------------------------------------------------
    # Sailable route builder (replaces old smooth → stub → tack pipeline)
    # ------------------------------------------------------------------

    def _build_sailable_route(self, raw_path, raw_times):
        """Convert raw A* path into a simplified, fully-sailable route.

        Every segment between consecutive output waypoints is verified
        sailable (finite ``_segment_travel_time``).  The route preserves
        genuine tack points and follows the actual sailing curve within
        each tack leg.

        Algorithm
        ---------
        1. Detect tack points in the raw path (heading change >= threshold).
        2. Merge nearby tacks that are closer than min spacing.
        3. For each inter-tack segment, run Douglas-Peucker-like
           simplification where the split criterion is **sailability**:
           a segment is kept as-is only if ``_segment_travel_time`` is
           finite (sailable heading, no land crossing) *and* the path
           deviation is small.  Otherwise, split at the most-deviating
           raw node and recurse.
        """
        if len(raw_path) <= 2:
            return list(raw_path)

        n = len(raw_path)
        px = np.array([self.node_x[nd] for nd in raw_path])
        py = np.array([self.node_y[nd] for nd in raw_path])

        tack_threshold = self.SMOOTH_TACK_THRESHOLD_DEG
        min_spacing    = self.SMOOTH_MIN_TACK_SPACING_M
        shape_tol      = self.SMOOTH_DP_EPSILON_M

        # ── 1. Compute per-segment headings ─────────────────────────────
        dx = np.diff(px)
        dy = np.diff(py)
        headings = np.degrees(np.arctan2(dx, dy)) % 360.0  # len = n-1

        def _hdiff(h1, h2):
            d = abs(h2 - h1)
            return d if d <= 180.0 else 360.0 - d

        # ── 2. Detect raw tack indices ──────────────────────────────────
        raw_tacks = []
        for i in range(1, len(headings)):
            if _hdiff(headings[i - 1], headings[i]) >= tack_threshold:
                raw_tacks.append(i)

        # ── 3. Merge nearby tacks ───────────────────────────────────────
        seg_dists = np.hypot(dx, dy)
        cum_dist  = np.concatenate([[0.0], np.cumsum(seg_dists)])

        merged = []
        for ti in raw_tacks:
            if merged and cum_dist[ti] - cum_dist[merged[-1]] < min_spacing:
                prev_ti = merged[-1]
                prev_h_before = headings[max(0, prev_ti - 1)]
                prev_h_after  = headings[min(prev_ti, len(headings) - 1)]
                this_h_before = headings[max(0, ti - 1)]
                this_h_after  = headings[min(ti, len(headings) - 1)]
                if _hdiff(this_h_before, this_h_after) > \
                   _hdiff(prev_h_before, prev_h_after):
                    merged[-1] = ti
            else:
                merged.append(ti)

        # Segment boundaries: start, tack points, end
        boundaries = sorted(set([0] + merged + [n - 1]))

        # ── 4. Sailable DP simplification per tack segment ──────────────
        _seg_time = self._segment_travel_time

        def _sailable_dp(seg_nodes, t_start):
            """Simplify seg_nodes keeping every segment sailable."""
            if len(seg_nodes) <= 2:
                return list(seg_nodes)

            first, last = seg_nodes[0], seg_nodes[-1]
            x1, y1 = self.node_x[first], self.node_y[first]
            x2, y2 = self.node_x[last],  self.node_y[last]

            sailable = False
            t_direct, _ = _seg_time(x1, y1, x2, y2, t_start)
            if np.isfinite(t_direct):
                seg_len = np.hypot(x2 - x1, y2 - y1)
                if seg_len < 1e-6:
                    return [first, last]
                spx = np.array([self.node_x[nd] for nd in seg_nodes])
                spy = np.array([self.node_y[nd] for nd in seg_nodes])
                devs = np.abs((x2 - x1) * (y1 - spy) -
                              (x1 - spx) * (y2 - y1)) / seg_len
                if devs.max() <= shape_tol:
                    sailable = True

            if sailable:
                return [first, last]

            # Must split — pick most-deviating point
            spx = np.array([self.node_x[nd] for nd in seg_nodes])
            spy = np.array([self.node_y[nd] for nd in seg_nodes])
            seg_len = np.hypot(x2 - x1, y2 - y1)
            if seg_len < 1e-6:
                return [first, last]

            devs = np.abs((x2 - x1) * (y1 - spy) -
                          (x1 - spx) * (y2 - y1)) / seg_len
            devs[0] = devs[-1] = 0.0
            split_idx = int(np.argmax(devs))
            if split_idx == 0:
                split_idx = len(seg_nodes) // 2

            left  = _sailable_dp(seg_nodes[:split_idx + 1], t_start)

            t_at_split = t_start
            t_left, _ = _seg_time(x1, y1,
                                  self.node_x[seg_nodes[split_idx]],
                                  self.node_y[seg_nodes[split_idx]],
                                  t_start)
            if np.isfinite(t_left):
                t_at_split = t_start + t_left

            right = _sailable_dp(seg_nodes[split_idx:], t_at_split)

            return left + right[1:]

        result = []
        for i in range(len(boundaries) - 1):
            s, e = boundaries[i], boundaries[i + 1]
            seg = [raw_path[j] for j in range(s, e + 1)]
            t0 = raw_times[s] if raw_times else 0.0
            simplified = _sailable_dp(seg, t0)
            if result and result[-1] == simplified[0]:
                result.extend(simplified[1:])
            else:
                result.extend(simplified)

        return result

    def _smooth_path(self, path_nodes, arrival_times,
                      tack_threshold_deg=None, dp_epsilon_m=None,
                      min_tack_spacing_m=None):
        """Two-pass path simplification: preserve tacks, simplify curves.

        Sailing routes have two kinds of heading changes:
        - **Tacks** (genuine course changes, ~45-90°): must be preserved
        - **Natural curves** (gradual drift from current/shore, 0-20°): simplified

        Algorithm:
        1. Find all raw tack points (turn angle >= tack_threshold_deg)
        2. Merge nearby tacks: skip tacks < min_tack_spacing_m from the previous
           kept tack (prevents consecutive rapid-fire tacks each becoming a wpt)
        3. Apply Douglas-Peucker within each inter-tack segment to simplify curves

        Parameters
        ----------
        tack_threshold_deg : float
            Turn angles at or above this are tacks (default 45°).
        dp_epsilon_m : float
            DP perpendicular tolerance within curved segments (default 120m).
        min_tack_spacing_m : float
            Minimum distance between consecutive tack waypoints (default 400m).
        """
        if tack_threshold_deg is None:
            tack_threshold_deg = self.SMOOTH_TACK_THRESHOLD_DEG
        if dp_epsilon_m is None:
            dp_epsilon_m = self.SMOOTH_DP_EPSILON_M
        if min_tack_spacing_m is None:
            min_tack_spacing_m = self.SMOOTH_MIN_TACK_SPACING_M

        if len(path_nodes) <= 2:
            return path_nodes

        px = np.array([self.node_x[n] for n in path_nodes])
        py = np.array([self.node_y[n] for n in path_nodes])
        n = len(path_nodes)

        # Cumulative distance along path
        seg_dists = np.hypot(np.diff(px), np.diff(py))
        cum_dist  = np.concatenate([[0.0], np.cumsum(seg_dists)])

        def _hdiff(h1, h2):
            d = abs(h2 - h1)
            return d if d <= 180 else 360 - d

        def _heading(i, j):
            return np.degrees(np.arctan2(px[j] - px[i], py[j] - py[i])) % 360.0

        # --- Pass 1: find raw tack indices ---
        raw_tacks = []
        for i in range(1, n - 1):
            h_in  = _heading(i - 1, i)
            h_out = _heading(i, i + 1)
            if _hdiff(h_in, h_out) >= tack_threshold_deg:
                raw_tacks.append(i)

        # --- Pass 1b: merge nearby tacks ---
        # Consecutive tacks < min_tack_spacing_m apart are collapsed: keep only
        # the one that is furthest from the previous kept tack waypoint.
        tack_boundaries = [0]
        for ti in raw_tacks:
            last = tack_boundaries[-1]
            dist_from_last = cum_dist[ti] - cum_dist[last]
            if dist_from_last >= min_tack_spacing_m:
                tack_boundaries.append(ti)
            else:
                # Replace last tack with this one if it has a bigger turn
                if last != 0:
                    last_turn = _hdiff(_heading(last - 1, last), _heading(last, last + 1))
                    this_turn = _hdiff(_heading(ti - 1, ti), _heading(ti, ti + 1))
                    if this_turn > last_turn:
                        tack_boundaries[-1] = ti
        tack_boundaries.append(n - 1)

        # --- Pass 2: Douglas-Peucker within each inter-tack segment ---
        def _dp_segment(start, end):
            """Simplify path[start..end] via Douglas-Peucker."""
            if end - start <= 1:
                return [start]
            x1, y1 = px[start], py[start]
            x2, y2 = px[end],   py[end]
            seg_len = np.hypot(x2 - x1, y2 - y1)
            if seg_len < 1e-6:
                return [start]

            max_d, max_i = 0.0, start
            for k in range(start + 1, end):
                d = abs((x2 - x1) * (y1 - py[k]) - (x1 - px[k]) * (y2 - y1)) / seg_len
                if d > max_d:
                    max_d, max_i = d, k

            if max_d > dp_epsilon_m:
                return _dp_segment(start, max_i) + _dp_segment(max_i, end)
            return [start]

        kept = []
        for seg_i in range(len(tack_boundaries) - 1):
            kept.extend(_dp_segment(tack_boundaries[seg_i], tack_boundaries[seg_i + 1]))
        kept.append(n - 1)
        kept = sorted(set(kept))

        return [path_nodes[i] for i in kept]

    def _remove_stubs(self, path_nodes):
        """Remove stub waypoints that create Y-junctions."""
        if len(path_nodes) <= 2:
            return path_nodes

        result = [path_nodes[0]]
        for i in range(1, len(path_nodes) - 1):
            prev_node = result[-1]
            curr_node = path_nodes[i]
            next_node = path_nodes[i + 1]

            x0, y0 = self.node_x[prev_node], self.node_y[prev_node]
            x1, y1 = self.node_x[curr_node], self.node_y[curr_node]
            x2, y2 = self.node_x[next_node], self.node_y[next_node]

            leg_in = np.hypot(x1 - x0, y1 - y0)
            leg_out = np.hypot(x2 - x1, y2 - y1)

            if leg_in < 1e-6 or leg_out < 1e-6:
                result.append(curr_node)
                continue

            ratio = max(leg_in, leg_out) / min(leg_in, leg_out)

            v_in = ((x1 - x0) / leg_in, (y1 - y0) / leg_in)
            v_out = ((x2 - x1) / leg_out, (y2 - y1) / leg_out)
            dot = v_in[0] * v_out[0] + v_in[1] * v_out[1]
            turn_angle = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))

            is_stub = ratio > 3.0 and turn_angle > 45.0

            if is_stub and self._line_of_sight(x0, y0, x2, y2):
                shortcut_time, _ = self._segment_travel_time(x0, y0, x2, y2, 0.0)
                if np.isfinite(shortcut_time):
                    continue
            result.append(curr_node)

        result.append(path_nodes[-1])
        return result

    TACK_ONLY_MIN_ANGLE = 30.0
    TACK_ONLY_MIN_LEG_M = 400.0

    def _tack_only_filter(self, path_nodes,
                          min_angle_deg=None, min_leg_m=None):
        """Keep only start, end, and genuine tack waypoints.

        Pass 1 — drop every waypoint whose turn angle (measured from
                 the previous *kept* point) is below min_angle_deg.
        Pass 2 — iteratively remove tack points that create legs shorter
                 than min_leg_m, collapsing micro-tack sequences.

        Land avoidance is handled downstream by ``_compute_leg_times``,
        which falls back to raw sub-paths when a simplified leg crosses
        land.  So this filter can be aggressive.
        """
        if min_angle_deg is None:
            min_angle_deg = self.TACK_ONLY_MIN_ANGLE
        if min_leg_m is None:
            min_leg_m = self.TACK_ONLY_MIN_LEG_M

        if len(path_nodes) <= 2:
            return path_nodes

        def _turn(a, b, c):
            h1 = np.degrees(np.arctan2(
                self.node_x[b] - self.node_x[a],
                self.node_y[b] - self.node_y[a])) % 360.0
            h2 = np.degrees(np.arctan2(
                self.node_x[c] - self.node_x[b],
                self.node_y[c] - self.node_y[b])) % 360.0
            d = abs(h2 - h1)
            return d if d <= 180 else 360 - d

        def _dist(a, b):
            return np.hypot(self.node_x[b] - self.node_x[a],
                            self.node_y[b] - self.node_y[a])

        # Pass 1: keep only real tacks
        kept = [path_nodes[0]]
        for i in range(1, len(path_nodes) - 1):
            prev = kept[-1]
            curr = path_nodes[i]
            nxt = path_nodes[i + 1]
            if _turn(prev, curr, nxt) >= min_angle_deg:
                kept.append(curr)
        kept.append(path_nodes[-1])

        # Pass 2: iteratively merge short tack legs (but preserve
        # approach tacks whose short leg connects to start or end)
        start_node, end_node = kept[0], kept[-1]
        changed = True
        while changed and len(kept) > 2:
            changed = False
            new_kept = [kept[0]]
            i = 1
            while i < len(kept) - 1:
                prev = new_kept[-1]
                curr = kept[i]
                nxt = kept[i + 1]
                leg_in = _dist(prev, curr)
                leg_out = _dist(curr, nxt)

                touches_endpoint = (prev == start_node or
                                    nxt == end_node)
                if (min(leg_in, leg_out) < min_leg_m
                        and not touches_endpoint):
                    changed = True
                    i += 1
                    continue
                new_kept.append(curr)
                i += 1
            new_kept.append(kept[-1])
            kept = new_kept

        return kept

    def find_route(self, start_latlon, end_latlon, start_time_s=0.0,
                   max_iterations=2_000_000, return_debug=False):
        """Find the time-optimal route between two lat/lon points.

        Uses a vectorized sector graph + Numba A* kernel when available,
        falling back to the original Python A* loop otherwise.
        """
        _perf = {}
        t_total = _time.monotonic()

        # ------------------------------------------------------------------
        # Setup
        # ------------------------------------------------------------------
        t0 = _time.monotonic()
        transformer = self.cf.transformer
        sx, sy = transformer.transform(start_latlon[1], start_latlon[0])
        ex, ey = transformer.transform(end_latlon[1], end_latlon[0])

        _, start_node = self.cf.tree.query([sx, sy])
        _, end_node = self.cf.tree.query([ex, ey])
        start_node = int(start_node)
        end_node = int(end_node)

        cu, _ = self.cf.query(self.node_x[start_node], self.node_y[start_node])
        if np.isnan(cu):
            raise ValueError(f"Start point snapped to land node {start_node}")
        cu, _ = self.cf.query(self.node_x[end_node], self.node_y[end_node])
        if np.isnan(cu):
            raise ValueError(f"End point snapped to land node {end_node}")

        goal_x, goal_y = self.node_x[end_node], self.node_y[end_node]
        boat_speed = self.boat.speed()
        if self._use_polar:
            max_boat_speed = self.boat.polar.max_speed_ms
        else:
            max_boat_speed = boat_speed
        max_speed = max_boat_speed + self.cf.max_current_speed
        _perf['setup'] = _time.monotonic() - t0

        _numba_reason = []
        if not _NUMBA_AVAILABLE or _sector_astar_jit is None:
            _numba_reason.append("numba_unavailable")
        if self.cf.delaunay is None or self.cf.valid_triangle is None:
            _numba_reason.append("missing_delaunay")
            if getattr(self.cf, "delaunay_error", None):
                _numba_reason.append(f"delaunay_error={self.cf.delaunay_error}")
        if self._use_polar and (not self.boat.polar or not self._numba_wind_ready):
            if not self.boat.polar:
                _numba_reason.append("polar_missing")
            if not self._numba_wind_ready:
                w_mode = getattr(self.wind, "_mode", "unknown") if self.wind is not None else "none"
                _numba_reason.append(f"wind_not_numba_ready(mode={w_mode})")

        _can_numba = len(_numba_reason) == 0
        _mode = 'Numba JIT' if _can_numba else 'Python'
        print(f"SectorRouter: {self.n_nodes:,} nodes, {self.N_SECTORS} sectors/node  [{_mode}]")
        if not _can_numba:
            print("  Numba disabled:", "; ".join(_numba_reason))
        print(f"Start node: {start_node}, End node: {end_node}")

        # ------------------------------------------------------------------
        # A* search
        # ------------------------------------------------------------------
        if _can_numba:
            # Prepare polar arrays (dummy if no polar)
            has_polar = self._use_polar
            if has_polar:
                p = self.boat.polar
                polar_twas = np.ascontiguousarray(p._twas.astype(np.float64))
                polar_twss = np.ascontiguousarray(p._twss.astype(np.float64))
                polar_speeds = np.ascontiguousarray(p._speeds.astype(np.float64))
                polar_min_twa = float(p.minimum_twa)
            else:
                polar_twas = np.empty(0, dtype=np.float64)
                polar_twss = np.empty(0, dtype=np.float64)
                polar_speeds = np.empty((0, 0), dtype=np.float64)
                polar_min_twa = 0.0

            has_wind = self._numba_wind_ready
            if has_wind:
                wind_wu = self._wind_at_nodes_wu
                wind_wv = self._wind_at_nodes_wv
                wind_ft = self._wind_frame_times
            else:
                wind_wu = np.empty((0, 0), dtype=np.float64)
                wind_wv = np.empty((0, 0), dtype=np.float64)
                wind_ft = np.empty(0, dtype=np.float64)

            # Build corridor adaptively and reuse graph from cache when possible.
            _perf['corridor'] = 0.0
            _perf['graph_build'] = 0.0
            _perf['graph_cache_hits'] = 0
            _perf['corridor_attempts'] = 0
            _perf['astar'] = 0.0

            sx_n = self.node_x[start_node]
            sy_n = self.node_y[start_node]
            ex_n = self.node_x[end_node]
            ey_n = self.node_y[end_node]
            route_dist = np.hypot(ex_n - sx_n, ey_n - sy_n)
            base_pad = max(route_dist, self.max_edge_m * 5, 5000.0)

            pad_values = []
            _seen = set()
            for fac in self._corridor_pad_factors:
                pad_i = max(base_pad * float(fac), self.max_edge_m * 3, 2500.0)
                key_i = int(round(pad_i))
                if key_i in _seen:
                    continue
                _seen.add(key_i)
                pad_values.append(float(key_i))
            pad_values.sort()

            best_cost_arr = None
            arrival_arr = None
            cf_node_arr = None
            cf_bucket_arr = None
            explored = -1

            for i_try, pad in enumerate(pad_values, start=1):
                _perf['corridor_attempts'] += 1
                cache_key = self._corridor_cache_key(start_node, end_node, pad)
                cached = self._corridor_cache_get(cache_key)
                if cached is not None:
                    g_off, g_tgt, g_dis, g_sec, n_corr, n_edges = cached
                    _perf['graph_cache_hits'] += 1
                    print(f"Corridor attempt {i_try}/{len(pad_values)}: "
                          f"pad {pad:.0f}m, {n_corr:,} nodes, {n_edges:,} edges (cache hit)")
                else:
                    t0 = _time.monotonic()
                    x_lo = min(sx_n, ex_n) - pad
                    x_hi = max(sx_n, ex_n) + pad
                    y_lo = min(sy_n, ey_n) - pad
                    y_hi = max(sy_n, ey_n) + pad
                    corridor = ((self.node_x >= x_lo) & (self.node_x <= x_hi) &
                                (self.node_y >= y_lo) & (self.node_y <= y_hi))
                    n_corr = int(corridor.sum())
                    _perf['corridor'] += _time.monotonic() - t0

                    t0 = _time.monotonic()
                    g_off, g_tgt, g_dis, g_sec = _build_corridor_sector_graph(
                        self.node_x, self.node_y, self.cf.tree,
                        self.cf.delaunay, self.cf.valid_triangle,
                        corridor, self.N_SECTORS, self.SECTOR_WIDTH,
                        self.k_candidates, self.min_edge_m, self.max_edge_m,
                        self._los_samples,
                    )
                    n_edges = len(g_tgt)
                    dt_build = _time.monotonic() - t0
                    _perf['graph_build'] += dt_build
                    self._corridor_cache_put(
                        cache_key, (g_off, g_tgt, g_dis, g_sec, n_corr, n_edges))
                    print(f"Corridor attempt {i_try}/{len(pad_values)}: "
                          f"pad {pad:.0f}m, {n_corr:,} nodes, {n_edges:,} edges "
                          f"(built in {dt_build:.2f}s)")

                t0 = _time.monotonic()
                best_cost_arr, arrival_arr, cf_node_arr, cf_bucket_arr, explored = \
                    _sector_astar_jit(
                        g_off, g_tgt, g_dis, g_sec,
                        self.node_x, self.node_y,
                        self._u_frames_2d, self._v_frames_2d,
                        self.cf.frame_times,
                        start_node, end_node, start_time_s,
                        boat_speed, max_speed,
                        has_polar, polar_twas, polar_twss, polar_speeds,
                        polar_min_twa,
                        _SWEEP_HX, _SWEEP_HY, _SWEEP_RADS, len(_SWEEP_HX),
                        self._polar_sweep_coarse_step,
                        self._use_dense_polar, self._polar_dense_arr,
                        self._polar_dense_max_tws,
                        self._use_dot_filter,
                        has_wind, wind_wu, wind_wv, wind_ft,
                        self.tack_penalty_s, self.tack_threshold_deg,
                        self.N_SECTORS, self.START_SENTINEL, max_iterations,
                    )
                explored = int(explored)
                _perf['astar'] += _time.monotonic() - t0

                if explored < 0:
                    if explored == -1:
                        raise RuntimeError(
                            f"A* exceeded {max_iterations} iterations (Numba) -- "
                            "route may be unreachable.")
                    if explored == -2:
                        raise RuntimeError(
                            "A* frontier heap overflow in Numba kernel -- "
                            "increase heap sizing or reduce search space.")
                    raise RuntimeError("A* failed in Numba kernel (internal error).")

                gc_try = best_cost_arr[end_node, :]
                fs_try = int(np.argmin(gc_try))
                if np.isfinite(gc_try[fs_try]):
                    break
                if i_try < len(pad_values):
                    print("  No path in this corridor; expanding search window...")

            # Reconstruct path
            t0 = _time.monotonic()
            n_buckets = self.N_SECTORS + 1
            gc = best_cost_arr[end_node, :]
            final_sector = int(np.argmin(gc))
            if not np.isfinite(gc[final_sector]):
                raise RuntimeError("No path found between start and end")

            raw_path = []
            raw_times = []
            node = end_node
            bucket = final_sector
            while True:
                raw_path.append(node)
                raw_times.append(float(arrival_arr[node, bucket]))
                pn = int(cf_node_arr[node, bucket])
                if pn < 0:
                    break
                pb = int(cf_bucket_arr[node, bucket])
                node = pn
                bucket = pb
            raw_path.reverse()
            raw_times.reverse()

            # All nodes that were reached by A* (any bucket with finite cost)
            explored_mask = np.any(best_cost_arr < np.inf, axis=1)
            explored_nodes = set(np.where(explored_mask)[0].tolist())
            _perf['reconstruct'] = _time.monotonic() - t0

        else:
            # ---- Python fallback ----
            _perf['corridor'] = 0.0
            _perf['graph_build'] = 0.0
            t0 = _time.monotonic()
            INF = float('inf')
            N_BUCKETS = self.N_SECTORS + 1
            best_cost = {}
            came_from = {}
            arrival_time_d = {}
            start_state = (start_node, self.START_SENTINEL)
            best_cost[start_state] = 0.0
            arrival_time_d[start_state] = start_time_s
            open_set = [(self._heuristic(start_node, goal_x, goal_y, max_speed),
                         0.0, start_node, self.START_SENTINEL)]
            explored = 0
            explored_nodes = set()

            while open_set:
                if explored >= max_iterations:
                    raise RuntimeError(f"A* exceeded {max_iterations} iterations")
                _, cost, node, sector_in = heapq.heappop(open_set)
                state = (node, sector_in)
                if cost > best_cost.get(state, INF):
                    continue
                explored += 1
                explored_nodes.add(node)
                if node == end_node:
                    break
                arr_t = arrival_time_d[state]
                neighbors = self._find_sector_neighbors(node)
                for neighbor, sector_out, dist in neighbors:
                    dt = self._edge_cost(node, neighbor, dist, arr_t)
                    if dt == INF:
                        continue
                    penalty = 0.0
                    angle_diff = self._sector_angle_diff(sector_in, sector_out)
                    if self.tack_penalty_s > 0 and angle_diff > self.tack_threshold_deg:
                        penalty = self.tack_penalty_s
                    new_cost = cost + dt + penalty
                    new_state = (neighbor, sector_out)
                    if new_cost < best_cost.get(new_state, INF):
                        best_cost[new_state] = new_cost
                        arrival_time_d[new_state] = arr_t + dt + penalty
                        came_from[new_state] = state
                        h = self._heuristic(neighbor, goal_x, goal_y, max_speed)
                        heapq.heappush(open_set,
                                       (new_cost + h, new_cost, neighbor, sector_out))

            _perf['astar'] = _time.monotonic() - t0
            goal_states = [(best_cost.get((end_node, b), INF), b)
                           for b in range(N_BUCKETS)]
            best_goal_cost, final_sector = min(goal_states)
            if best_goal_cost == INF:
                raise RuntimeError("No path found between start and end")

            t0 = _time.monotonic()
            raw_path = []
            raw_times = []
            state = (end_node, final_sector)
            while state in came_from or state == start_state:
                node, _ = state
                raw_path.append(node)
                raw_times.append(arrival_time_d.get(state, start_time_s))
                if state == start_state:
                    break
                state = came_from.get(state)
                if state is None:
                    break
            raw_path.reverse()
            raw_times.reverse()
            _perf['reconstruct'] = _time.monotonic() - t0

        # ------------------------------------------------------------------
        # Sailable route simplification
        # ------------------------------------------------------------------
        t0 = _time.monotonic()
        path_nodes = self._build_sailable_route(raw_path, raw_times)
        _perf['smooth'] = _time.monotonic() - t0

        # ------------------------------------------------------------------
        # Route timing and waypoints
        # ------------------------------------------------------------------
        t0 = _time.monotonic()
        inv_transformer = Transformer.from_crs(
            transformer.target_crs, transformer.source_crs, always_xy=True)

        def _compute_leg_times(nodes):
            """Compute per-leg timing.  Every segment should be sailable;
            fall back to raw sub-path summation only as a safety net."""
            raw_node_idx = {nd: i for i, nd in enumerate(raw_path)}
            wps, times, dists = [], [], []
            fallback_speed = max(self.boat.base_speed_knots * 0.514, 0.1)
            t = start_time_s
            for k in range(len(nodes) - 1):
                na, nb = nodes[k], nodes[k + 1]
                x1, y1 = self.node_x[na], self.node_y[na]
                x2, y2 = self.node_x[nb], self.node_y[nb]
                t_seg, d_seg = self._segment_travel_time(x1, y1, x2, y2, t)
                if np.isfinite(t_seg):
                    wps.append((x1, y1))
                    times.append(t_seg)
                    dists.append(d_seg)
                    t += t_seg
                else:
                    ia = raw_node_idx.get(na, -1)
                    ib = raw_node_idx.get(nb, -1)
                    sub = raw_path[ia:ib + 1] if (ia >= 0 and ib > ia) else [na, nb]
                    leg_t = 0.0
                    leg_d = 0.0
                    for j in range(len(sub) - 1):
                        sx1 = self.node_x[sub[j]]
                        sy1 = self.node_y[sub[j]]
                        sx2 = self.node_x[sub[j + 1]]
                        sy2 = self.node_y[sub[j + 1]]
                        t_s, _ = self._segment_travel_time(
                            sx1, sy1, sx2, sy2, t + leg_t)
                        d_s = np.hypot(sx2 - sx1, sy2 - sy1)
                        if not np.isfinite(t_s):
                            t_s = d_s / fallback_speed
                        leg_t += t_s
                        leg_d += d_s
                    wps.append((x1, y1))
                    times.append(leg_t)
                    dists.append(leg_d)
                    t += leg_t
            wps.append((self.node_x[nodes[-1]],
                        self.node_y[nodes[-1]]))
            return list(nodes), wps, times, dists

        smooth_ids_before_timing = list(path_nodes)
        path_nodes, waypoints_utm, leg_times, leg_dists = _compute_leg_times(path_nodes)
        total_time = sum(leg_times) if leg_times else float('nan')
        total_dist = sum(leg_dists)
        avg_sog = (total_dist / total_time * MS_TO_KNOTS) if (
            total_time > 0 and np.isfinite(total_time)) else 0.0

        waypoints_latlon = []
        for x, y in waypoints_utm:
            lon, lat = inv_transformer.transform(x, y)
            waypoints_latlon.append((lat, lon))
        _perf['route_time'] = _time.monotonic() - t0

        # ------------------------------------------------------------------
        # Track from raw A* path (follows actual tacking, correct timing)
        # ------------------------------------------------------------------
        t0 = _time.monotonic()
        sim_track = [(self.node_x[n], self.node_y[n]) for n in raw_path]
        sim_track_times = list(raw_times)
        _perf['simulate'] = _time.monotonic() - t0

        # ------------------------------------------------------------------
        # Build rich diagnostics (always collected, cheap)
        # ------------------------------------------------------------------
        t0 = _time.monotonic()
        _dbg = {}

        # ---- Raw A* path (before smoothing) ----
        raw_ids = np.array(raw_path, dtype=np.int32)
        _dbg['raw_node_ids'] = raw_ids
        _dbg['raw_x'] = self.node_x[raw_ids]
        _dbg['raw_y'] = self.node_y[raw_ids]
        _dbg['raw_arrival_s'] = np.array(raw_times, dtype=np.float64)

        raw_ll = np.empty((len(raw_ids), 2))
        for ri, nid in enumerate(raw_ids):
            lon, lat = inv_transformer.transform(
                self.node_x[nid], self.node_y[nid])
            raw_ll[ri] = [lat, lon]
        _dbg['raw_lat'] = raw_ll[:, 0]
        _dbg['raw_lon'] = raw_ll[:, 1]

        if len(raw_ids) >= 2:
            rdx = np.diff(_dbg['raw_x'])
            rdy = np.diff(_dbg['raw_y'])
            _dbg['raw_heading_deg'] = np.degrees(np.arctan2(rdx, rdy)) % 360.0
            _dbg['raw_seg_dist_m'] = np.hypot(rdx, rdy)
            if len(raw_ids) >= 3:
                rh = _dbg['raw_heading_deg']
                rd = np.abs(np.diff(rh))
                _dbg['raw_turn_deg'] = np.minimum(rd, 360.0 - rd)

        # Current and wind at each raw-path node at its arrival time
        n_raw = len(raw_ids)
        raw_cu = np.empty(n_raw)
        raw_cv = np.empty(n_raw)
        raw_wu = np.zeros(n_raw)
        raw_wv = np.zeros(n_raw)
        for ri in range(n_raw):
            nid = int(raw_ids[ri])
            et = float(_dbg['raw_arrival_s'][ri])
            cu_v, cv_v = self.cf.query(self.node_x[nid], self.node_y[nid],
                                       elapsed_s=et)
            raw_cu[ri] = cu_v
            raw_cv[ri] = cv_v
            if self.wind is not None:
                wu_v, wv_v = self.wind.query(self.node_x[nid],
                                             self.node_y[nid], elapsed_s=et)
                raw_wu[ri] = wu_v
                raw_wv[ri] = wv_v
        _dbg['raw_cu'] = raw_cu
        _dbg['raw_cv'] = raw_cv
        if self.wind is not None:
            _dbg['raw_wu'] = raw_wu
            _dbg['raw_wv'] = raw_wv

        # Per-segment SOG along the raw path
        if n_raw >= 2:
            raw_dt = np.diff(_dbg['raw_arrival_s'])
            raw_dd = _dbg['raw_seg_dist_m']
            with np.errstate(divide='ignore', invalid='ignore'):
                _dbg['raw_sog_kt'] = np.where(
                    raw_dt > 0, raw_dd / raw_dt * MS_TO_KNOTS, 0.0)

        # ---- Smoothed path (after smooth + stub + tack filter, before leg timing) ----
        smooth_ids = np.array(smooth_ids_before_timing, dtype=np.int32)
        _dbg['smooth_node_ids'] = smooth_ids
        _dbg['smooth_x'] = self.node_x[smooth_ids]
        _dbg['smooth_y'] = self.node_y[smooth_ids]

        # ---- Explored nodes ----
        exp_arr = np.array(sorted(explored_nodes), dtype=np.int32)
        _dbg['explored_node_ids'] = exp_arr
        _dbg['explored_x'] = self.node_x[exp_arr]
        _dbg['explored_y'] = self.node_y[exp_arr]

        # ---- Scalars / metadata ----
        _dbg['start_node'] = np.int32(start_node)
        _dbg['end_node'] = np.int32(end_node)
        _dbg['start_time_s'] = np.float64(start_time_s)
        _dbg['n_nodes_total'] = np.int32(self.n_nodes)
        _dbg['router_mode'] = np.array(_mode)

        # ---- Performance timings ----
        _dbg['perf'] = _perf

        _perf['diagnostics'] = _time.monotonic() - t0

        # ------------------------------------------------------------------
        # Assemble Route
        # ------------------------------------------------------------------
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
            simulated_track_times=sim_track_times,
            debug=_dbg,
        )

        t0 = _time.monotonic()
        xs, ys, water_mask = self._build_grid_for_plotting(
            waypoints_utm, start_latlon, end_latlon)
        _perf['plot_grid'] = _time.monotonic() - t0

        # ------------------------------------------------------------------
        # Performance report
        # ------------------------------------------------------------------
        wall_s = _time.monotonic() - t_total
        _perf['total'] = wall_s
        n_raw_ct = len(raw_path)
        n_smooth_ct = len(smooth_ids_before_timing)

        print(f"\nSector A* completed in {wall_s:.2f}s ({_mode}), explored {explored:,} nodes")
        print(f"  Setup:               {_perf['setup']:.3f}s")
        if _can_numba:
            print(f"  Corridor select:     {_perf['corridor']:.3f}s")
            print(f"  Graph build:         {_perf['graph_build']:.3f}s")
            print(f"  Corridor attempts:   {int(_perf.get('corridor_attempts', 1))} "
                  f"(cache hits {int(_perf.get('graph_cache_hits', 0))})")
        print(f"  A* search:           {_perf['astar']:.3f}s")
        print(f"  Path reconstruction: {_perf['reconstruct']:.3f}s")
        print(f"  Path smoothing:      {_perf['smooth']:.3f}s")
        print(f"  Route timing:        {_perf['route_time']:.3f}s")
        print(f"  Track simulation:    {_perf['simulate']:.3f}s")
        print(f"  Diagnostics:         {_perf['diagnostics']:.3f}s")
        print(f"  Plotting grid:       {_perf['plot_grid']:.3f}s")
        print(f"  TOTAL:               {wall_s:.3f}s")
        if _can_numba and _perf['astar'] > 10.0:
            print("  (Numba JIT compiled on this run — subsequent runs use cache)")
        print(f"Path smoothing: {n_raw_ct} -> {n_smooth_ct} waypoints")
        print(route.summary())

        if return_debug:
            return route, xs, ys, water_mask, _dbg

        return route, xs, ys, water_mask

    def _simulate_track(self, waypoints_utm, start_time_s, dt_s=10.0):
        """Sample the ground track into a dense time series."""
        if len(waypoints_utm) < 2:
            return list(waypoints_utm), [start_time_s] * len(waypoints_utm)

        track = []
        track_times = []
        elapsed = start_time_s
        px, py = waypoints_utm[0]
        track.append((px, py))
        track_times.append(elapsed)

        for wp_idx in range(1, len(waypoints_utm)):
            tx, ty = waypoints_utm[wp_idx]
            seg_time, seg_dist = self._segment_travel_time(px, py, tx, ty, elapsed)

            if not np.isfinite(seg_time) or seg_dist < 1e-6:
                px, py = tx, ty
                track.append((px, py))
                track_times.append(elapsed)
                continue

            n_steps = max(1, int(np.ceil(seg_time / dt_s)),
                          int(np.ceil(seg_dist / 150.0)))

            x0, y0 = px, py
            for step_idx in range(1, n_steps + 1):
                alpha = step_idx / n_steps
                px = x0 + alpha * (tx - x0)
                py = y0 + alpha * (ty - y0)
                t = elapsed + alpha * seg_time
                track.append((px, py))
                track_times.append(t)

            elapsed += seg_time

        return track, track_times

    def _build_grid_for_plotting(self, waypoints_utm, start_latlon, end_latlon,
                                  resolution=300.0, padding=2000.0):
        """Generate a regular grid overlay for plot_route compatibility."""
        transformer = self.cf.transformer
        sx, sy = transformer.transform(start_latlon[1], start_latlon[0])
        ex, ey = transformer.transform(end_latlon[1], end_latlon[0])

        all_x = [sx, ex] + [p[0] for p in waypoints_utm]
        all_y = [sy, ey] + [p[1] for p in waypoints_utm]

        x0 = min(all_x) - padding
        x1 = max(all_x) + padding
        y0 = min(all_y) - padding
        y1 = max(all_y) + padding

        xs = np.arange(x0, x1 + resolution, resolution)
        ys = np.arange(y0, y1 + resolution, resolution)

        u_grid, v_grid = self.cf.query_grid(xs, ys, elapsed_s=0.0)
        water_mask = ~np.isnan(u_grid)

        return xs, ys, water_mask

    def straight_line_time(self, start_latlon, end_latlon, start_time_s=0.0):
        """Estimate travel time along a straight line."""
        transformer = self.cf.transformer
        sx, sy = transformer.transform(start_latlon[1], start_latlon[0])
        ex, ey = transformer.transform(end_latlon[1], end_latlon[0])

        total_time, total_dist = self._segment_travel_time(
            sx, sy, ex, ey, start_time_s, n_samples=50)
        return total_time, total_dist


# ===================================================================
#  Visualization
# ===================================================================

def plot_route(route, xs, ys, water_mask, current_field,
               start_latlon, end_latlon,
               straight_time_s=None, straight_dist_m=None,
               save_path=None, subsample_arrows=3, show=True,
               wind_field=None, elapsed_s=0.0):
    """Plot the optimised route over the current field.

    Parameters
    ----------
    wind_field : WindField, optional
        If provided, creates a side-by-side plot with current (left)
        and wind (right) vector fields.
    elapsed_s : float
        Time offset for querying current/wind fields (default 0).
    """
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    import matplotlib.colors as mcolors

    transformer = current_field.transformer
    xx, yy = np.meshgrid(xs, ys)
    margin = 500

    u_grid, v_grid = current_field.query_grid(xs, ys, elapsed_s=elapsed_s)
    speed_grid = np.sqrt(u_grid**2 + v_grid**2) * MS_TO_KNOTS
    speed_grid[~water_mask] = np.nan

    def _draw_shoreline(ax):
        draw_shoreline(ax, transformer)

    def _draw_route(ax, show_legend=True):
        if route.simulated_track and len(route.simulated_track) > 1:
            stx = [p[0] for p in route.simulated_track]
            sty = [p[1] for p in route.simulated_track]
            ax.plot(stx, sty, color='#e63946', linewidth=3, zorder=5,
                    label='Ground track' if show_legend else None,
                    path_effects=[pe.Stroke(linewidth=5, foreground='white'),
                                  pe.Normal()])
            rx = [p[0] for p in route.waypoints_utm[1:-1]]
            ry = [p[1] for p in route.waypoints_utm[1:-1]]
            if rx:
                ax.plot(rx, ry, 'o', color='#e63946', markersize=5, zorder=6,
                        markeredgecolor='white', markeredgewidth=1,
                        label='Tack points' if show_legend else None)
        else:
            rx = [p[0] for p in route.waypoints_utm]
            ry = [p[1] for p in route.waypoints_utm]
            ax.plot(rx, ry, color='#e63946', linewidth=3, zorder=5,
                    label='Optimal route' if show_legend else None,
                    path_effects=[pe.Stroke(linewidth=5, foreground='white'),
                                  pe.Normal()])

        sx, sy = transformer.transform(start_latlon[1], start_latlon[0])
        ex, ey = transformer.transform(end_latlon[1], end_latlon[0])
        ax.plot([sx, ex], [sy, ey], '--', color='#457b9d', linewidth=2,
                zorder=4, label='Straight line' if show_legend else None,
                path_effects=[pe.Stroke(linewidth=3.5, foreground='white',
                                        alpha=0.6),
                              pe.Normal()])
        ax.plot(sx, sy, 'o', color='#2a9d8f', markersize=12, zorder=6,
                markeredgecolor='white', markeredgewidth=2,
                label='Start' if show_legend else None)
        ax.plot(ex, ey, 's', color='#e76f51', markersize=12, zorder=6,
                markeredgecolor='white', markeredgewidth=2,
                label='End' if show_legend else None)

    def _setup_ax(ax, title):
        ax.set_facecolor('#f0e6d3')
        water_bg = np.where(water_mask, 0.0, np.nan)
        ax.pcolormesh(xs, ys, water_bg, cmap='Greys', vmin=0, vmax=1,
                      alpha=0.05, shading='auto', zorder=0)
        ax.set_xlabel('Easting (m, UTM)')
        ax.set_ylabel('Northing (m, UTM)')
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.15, color='#999999')
        ax.set_xlim(xs[0] - margin, xs[-1] + margin)
        ax.set_ylim(ys[0] - margin, ys[-1] + margin)

    if wind_field is None:
        fig, ax = plt.subplots(figsize=(14, 10))

        _setup_ax(ax, 'Sail Routing -- Time-Optimal Path Through Currents')

        speed_max = np.nanmax(speed_grid) if np.any(~np.isnan(speed_grid)) else 1.0
        im = ax.pcolormesh(xs, ys, speed_grid,
                           cmap='cividis', alpha=0.35, shading='auto',
                           vmin=0, vmax=max(speed_max, 0.5), zorder=1)
        fig.colorbar(im, ax=ax, label='Current speed (knots)',
                     pad=0.02, shrink=0.8)

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
        visible = (mag > 0.02) & ~np.isnan(spd_sub)
        u_norm[~visible] = np.nan
        v_norm[~visible] = np.nan

        arrow_scale = 1.0 / (step * (xs[1] - xs[0])) * 0.7
        ax.quiver(
            xx_sub, yy_sub, u_norm, v_norm, spd_sub,
            cmap='plasma', clim=(0, max(speed_max, 0.5)),
            scale=arrow_scale, scale_units='xy',
            width=0.004, headwidth=4, headlength=5, headaxislength=4,
            alpha=0.85, zorder=3, edgecolors='#333333', linewidth=0.3,
        )

        _draw_shoreline(ax)
        _draw_route(ax)

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
        ax.legend(loc='upper right', framealpha=0.9, fontsize=9)

    else:
        fig, (ax_cur, ax_wind) = plt.subplots(1, 2, figsize=(20, 10))

        _setup_ax(ax_cur, 'Ocean Current')
        _setup_ax(ax_wind, 'Wind')

        speed_max = np.nanmax(speed_grid) if np.any(~np.isnan(speed_grid)) else 1.0
        im_cur = ax_cur.pcolormesh(xs, ys, speed_grid,
                                   cmap='cividis', alpha=0.35, shading='auto',
                                   vmin=0, vmax=max(speed_max, 0.5), zorder=1)
        fig.colorbar(im_cur, ax=ax_cur, label='Current (knots)',
                     pad=0.02, shrink=0.7)

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
        visible = (mag > 0.02) & ~np.isnan(spd_sub)
        u_norm[~visible] = np.nan
        v_norm[~visible] = np.nan

        arrow_scale = 1.0 / (step * (xs[1] - xs[0])) * 0.7
        ax_cur.quiver(
            xx_sub, yy_sub, u_norm, v_norm, spd_sub,
            cmap='plasma', clim=(0, max(speed_max, 0.5)),
            scale=arrow_scale, scale_units='xy',
            width=0.004, headwidth=4, headlength=5, headaxislength=4,
            alpha=0.85, zorder=3, edgecolors='#333333', linewidth=0.3,
        )

        wu_grid = np.zeros_like(u_grid)
        wv_grid = np.zeros_like(v_grid)
        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                if water_mask[i, j]:
                    wu, wv = wind_field.query(x, y, elapsed_s=elapsed_s)
                    wu_grid[i, j] = wu
                    wv_grid[i, j] = wv
                else:
                    wu_grid[i, j] = np.nan
                    wv_grid[i, j] = np.nan

        wind_speed_grid = np.sqrt(wu_grid**2 + wv_grid**2) * MS_TO_KNOTS
        wind_max = np.nanmax(wind_speed_grid) if np.any(~np.isnan(wind_speed_grid)) else 10.0

        im_wind = ax_wind.pcolormesh(xs, ys, wind_speed_grid,
                                     cmap='YlOrRd', alpha=0.35, shading='auto',
                                     vmin=0, vmax=max(wind_max, 5.0), zorder=1)
        fig.colorbar(im_wind, ax=ax_wind, label='Wind (knots)',
                     pad=0.02, shrink=0.7)

        wu_sub = wu_grid[::step, ::step].copy()
        wv_sub = wv_grid[::step, ::step].copy()
        wspd_sub = wind_speed_grid[::step, ::step].copy()

        wmag = np.sqrt(wu_sub**2 + wv_sub**2)
        wmag_safe = np.where(wmag < 1e-8, 1.0, wmag)
        wu_norm = wu_sub / wmag_safe
        wv_norm = wv_sub / wmag_safe
        wvisible = (wmag > 0.1) & ~np.isnan(wspd_sub)
        wu_norm[~wvisible] = np.nan
        wv_norm[~wvisible] = np.nan

        ax_wind.quiver(
            xx_sub, yy_sub, wu_norm, wv_norm, wspd_sub,
            cmap='Reds', clim=(0, max(wind_max, 5.0)),
            scale=arrow_scale, scale_units='xy',
            width=0.004, headwidth=4, headlength=5, headaxislength=4,
            alpha=0.85, zorder=3, edgecolors='#333333', linewidth=0.3,
        )

        _draw_shoreline(ax_cur)
        _draw_shoreline(ax_wind)
        _draw_route(ax_cur, show_legend=True)
        _draw_route(ax_wind, show_legend=False)

        ax_cur.legend(loc='upper right', framealpha=0.9, fontsize=9)

        stats_lines = [
            f"Optimal:  {route.total_time_s / 60:.1f} min, "
            f"{route.total_distance_m / 1852:.2f} nm, "
            f"SOG {route.avg_sog_knots:.2f} kt",
            f"Boat STW: {route.boat_speed_knots:.1f} kt",
        ]
        if straight_time_s is not None and np.isfinite(straight_time_s):
            saved = straight_time_s - route.total_time_s
            stats_lines.insert(1, f"Time saved: {saved / 60:.1f} min")
        stats_text = "\n".join(stats_lines)
        ax_cur.text(0.02, 0.02, stats_text, transform=ax_cur.transAxes,
                    fontsize=9, verticalalignment='bottom', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                              edgecolor='#cccccc'))

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

_SURFACE_CACHE_DIR = Path(__file__).parent / ".sscofs_surface_cache"


def _surface_cache_path(run_date, cycle_hour, fh):
    return _SURFACE_CACHE_DIR / f"sscofs_{run_date.replace('-','')}_{cycle_hour:02d}z_f{fh:03d}.npz"


def _fetch_frames_byterange(run_date, cycle_hour, fh_list, max_workers=8):
    """Load surface u,v frames, hitting a lightweight local cache first.

    Cache stores only the extracted arrays (~1.5 MB compressed .npz per
    frame).  On a miss, fetches only the needed variables via S3
    byte-range reads (~3 MB per file instead of 200 MB), in parallel.

    Returns
    -------
    (lonc, latc, results)
        results : dict  fh -> (u, v)  numpy arrays, NaN replaced with 0.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import s3fs
    import xarray as xr
    from fetch_sscofs import build_sscofs_url

    _SURFACE_CACHE_DIR.mkdir(exist_ok=True)

    # 1. Check cache for each frame.
    results = {}
    lonc = latc = None
    to_fetch = []

    for fh in fh_list:
        cache_file = _surface_cache_path(run_date, cycle_hour, fh)
        if cache_file.exists():
            data = np.load(cache_file)
            results[fh] = (data["u"], data["v"])
            if lonc is None:
                lonc = data["lonc"]
                latc = data["latc"]
        else:
            to_fetch.append(fh)

    n_cached = len(fh_list) - len(to_fetch)
    if n_cached:
        print(f"  {n_cached} frame(s) from cache")
    if not to_fetch:
        return lonc, latc, results

    # 2. Byte-range fetch only the misses.
    print(f"  {len(to_fetch)} frame(s) from S3 (byte-range)...")

    _fs_holder = {}

    def _get_fs():
        if "fs" not in _fs_holder:
            _fs_holder["fs"] = s3fs.S3FileSystem(
                anon=True,
                default_block_size=8 * 1024 * 1024,
                default_fill_cache=True,
            )
        return _fs_holder["fs"]

    def _s3_key(date_str, cyc, fh):
        url = build_sscofs_url(date_str, cyc, fh)
        return url.replace(
            "https://noaa-nos-ofs-pds.s3.amazonaws.com/", "noaa-nos-ofs-pds/"
        )

    def _load_and_cache(fh):
        fs = _get_fs()
        key = _s3_key(run_date, cycle_hour, fh)
        with fs.open(key, "rb") as fobj:
            ds = xr.open_dataset(fobj, engine="h5netcdf",
                                 drop_variables=["siglay", "siglev"])
            u = np.nan_to_num(ds["u"].isel(time=0, siglay=0).values, nan=0.0)
            v = np.nan_to_num(ds["v"].isel(time=0, siglay=0).values, nan=0.0)
            fc_lonc = ds["lonc"].values
            fc_latc = ds["latc"].values
            ds.close()

        cache_file = _surface_cache_path(run_date, cycle_hour, fh)
        np.savez_compressed(cache_file, u=u, v=v, lonc=fc_lonc, latc=fc_latc)
        return fh, fc_lonc, fc_latc, u, v

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_load_and_cache, fh): fh for fh in to_fetch}
        for fut in as_completed(futures):
            fh_done, fc_lonc, fc_latc, u, v = fut.result()
            if lonc is None:
                lonc = fc_lonc
                latc = fc_latc
            results[fh_done] = (u, v)

    return lonc, latc, results


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
    from fetch_sscofs import build_sscofs_url, compute_file_for_datetime
    from latest_cycle import latest_cycle_and_url_for_local_hour

    depart_utc = None
    if depart_dt is not None:
        if depart_dt.tzinfo is None:
            depart_dt = depart_dt.replace(tzinfo=ZoneInfo("America/Los_Angeles"))
        depart_utc = depart_dt.astimezone(_dt.timezone.utc)

    # Determine model cycle metadata (no data download).
    now_utc = _dt.datetime.now(_dt.timezone.utc)
    if depart_utc is not None and (now_utc - depart_utc).total_seconds() > 3 * 3600:
        info = compute_file_for_datetime(depart_utc)
    else:
        local_now = now_utc.astimezone(ZoneInfo("America/Los_Angeles"))
        local_hhmm = local_now.hour * 100 + local_now.minute
        info = latest_cycle_and_url_for_local_hour(local_hhmm,
                                                   "America/Los_Angeles")

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
        depart_local = depart_utc.astimezone(depart_dt.tzinfo)
        print(f"Departure (local):  {depart_local:%Y-%m-%d %H:%M}")
        print(f"Departure (UTC):    {depart_utc:%Y-%m-%d %H:%M UTC}")

    # Collect the full list of forecast hours to fetch.
    all_fh = sorted(set(forecast_hours)) if forecast_hours else [base_hour]

    # Fast path: local surface cache → S3 byte-range reads for misses.
    t0 = _time.monotonic()
    print(f"Loading {len(all_fh)} SSCOFS surface frames...")
    lonc, latc, fh_data = _fetch_frames_byterange(
        run_date, cycle_hour_int, all_fh, max_workers=min(len(all_fh), 10))
    elapsed = _time.monotonic() - t0
    print(f"  Done in {elapsed:.1f}s")

    if lonc.max() > 180:
        lonc = np.where(lonc > 180, lonc - 360, lonc)

    transformer, _, _ = create_utm_transformer(
        float(np.mean(latc)), float(np.mean(lonc)))

    u_frames = []
    v_frames = []
    frame_times = []
    for fh in all_fh:
        u, v = fh_data[fh]
        u_frames.append(u)
        v_frames.append(v)
        frame_times.append(float((fh - base_hour) * 3600))

    cf = CurrentField(lonc, latc, u_frames, v_frames, frame_times,
                      transformer, nav_mask_path=NAV_MASK_PATH)

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
    parser.add_argument("--minimum-twa", type=float, default=0.0,
                        help="No-go zone: zero boat speed below this TWA (degrees)")
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
        polar = PolarTable(args.polar, minimum_twa=args.minimum_twa)
        nogo = f", no-go < {args.minimum_twa:.0f}°" if args.minimum_twa > 0 else ""
        print(f"Polar loaded: max speed {polar.max_speed_kt:.1f} kt{nogo}")

    wind = None
    if args.wind_speed is not None and args.wind_direction is not None:
        wind = WindField.from_met(args.wind_speed, args.wind_direction)
        print(f"Wind: {args.wind_speed:.1f} kt from {args.wind_direction:.0f} deg")
    elif (args.wind_speed is not None) != (args.wind_direction is not None):
        print("Warning: both --wind-speed and --wind-direction are required "
              "together; ignoring wind.")

    boat = BoatModel(base_speed_knots=args.boat_speed, polar=polar)

    print(f"Using SectorRouter ({SectorRouter.N_SECTORS}-sector heading-binned connectivity)")
    router = SectorRouter(cf, boat, wind=wind,
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
