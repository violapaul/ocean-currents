"""
shoreline_utils.py — Puget Sound shoreline overlay for matplotlib/UTM plots.

Usage:
    from shoreline_utils import draw_shoreline
    draw_shoreline(ax, transformer)          # light background
    draw_shoreline(ax, transformer, dark=True)  # dark background
"""

import json
from pathlib import Path

import numpy as np

_DEFAULT_PATH = Path(__file__).parent / "data" / "shoreline_puget.geojson"

# Colour presets: visible but unobtrusive on each background type.
_COLOR_LIGHT = "#6b7d8a"   # muted blue-grey on sandy/white backgrounds
_COLOR_DARK  = "#3d5a7a"   # darker slate-blue on near-black backgrounds


def draw_shoreline(
    ax,
    transformer,
    *,
    dark: bool = False,
    color: str | None = None,
    linewidth: float = 0.8,
    zorder: int = 2,
    alpha: float = 1.0,
    path: Path | None = None,
) -> None:
    """Overlay the Puget Sound shoreline on a UTM-coordinate matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis (must be in UTM metres, matching *transformer* output).
    transformer : pyproj.Transformer
        Converts (lon, lat) → (easting, northing).  Typically the same
        ``cf.transformer`` used to build the current-field grid.
    dark : bool
        Use the dark-background colour preset (default False → light preset).
    color : str, optional
        Override the automatic colour choice.
    linewidth : float
        Line width in points (default 0.8).
    zorder : int
        Matplotlib draw order (default 2, behind data but above background).
    alpha : float
        Line opacity (default 1.0).
    path : Path, optional
        GeoJSON file to read.  Defaults to ``data/shoreline_puget.geojson``
        relative to this file.
    """
    p = path or _DEFAULT_PATH
    if not p.exists():
        return

    c = color if color is not None else (_COLOR_DARK if dark else _COLOR_LIGHT)

    try:
        with open(p) as f:
            gj = json.load(f)

        for feat in gj.get("features", []):
            geom = feat.get("geometry") or {}
            gtype = geom.get("type", "")
            if gtype == "LineString":
                rings = [geom["coordinates"]]
            elif gtype == "MultiLineString":
                rings = geom["coordinates"]
            else:
                continue

            for coords in rings:
                arr = np.asarray(coords, dtype=float)  # (N, 2) lon/lat
                if arr.ndim != 2 or arr.shape[1] < 2 or len(arr) < 2:
                    continue
                sx, sy = transformer.transform(arr[:, 0], arr[:, 1])
                ax.plot(sx, sy, color=c, linewidth=linewidth,
                        zorder=zorder, alpha=alpha)
    except Exception:
        pass
