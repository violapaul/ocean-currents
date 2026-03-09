"""
test_ecmwf_wind.py
------------------

Unit tests for ECMWF/Open-Meteo wind discovery/fetch helpers.
Network calls are mocked so tests stay deterministic and offline.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ecmwf_wind as ew


def test_build_route_bbox_grid_has_expected_extent():
    wps = [(47.60, -122.40), (47.50, -122.20)]
    pts = ew.build_route_bbox_grid(wps, padding_deg=0.10, step_deg=0.10)
    lats = [p[0] for p in pts]
    lons = [p[1] for p in pts]
    assert min(lats) == pytest.approx(47.40, abs=1e-8)
    assert max(lats) == pytest.approx(47.70, abs=1e-8)
    assert min(lons) == pytest.approx(-122.50, abs=1e-8)
    assert max(lons) == pytest.approx(-122.10, abs=1e-8)


def test_discover_nodes_dedupes_snapped_results(monkeypatch, tmp_path):
    def fake_batch(latitudes, longitudes, **kwargs):
        out = []
        for lat, lon in zip(latitudes, longitudes):
            # Snap all points in this test to the same node.
            if lat < 47.0:
                out.append({"latitude": 46.9000, "longitude": -122.5000})
            else:
                out.append({"latitude": 47.1000, "longitude": -122.3000})
        return out

    monkeypatch.setattr(ew, "request_open_meteo_batch", fake_batch)

    points = [
        (46.95, -122.55),
        (46.96, -122.54),
        (47.11, -122.31),
        (47.12, -122.30),
    ]
    out_csv = tmp_path / "nodes.csv"
    nodes = ew.discover_ecmwf_nodes(points, output_csv=out_csv, verbose=False)

    assert list(nodes.columns) == ["latitude", "longitude"]
    assert len(nodes) == 2
    assert out_csv.exists()


def test_fetch_wind_for_nodes_shapes_and_utc(monkeypatch, tmp_path):
    def fake_batch(latitudes, longitudes, timezone, model, **kwargs):
        payloads = []
        for lat, lon in zip(latitudes, longitudes):
            payloads.append({
                "latitude": lat,
                "longitude": lon,
                "hourly": {
                    "time": ["2026-03-08T10:00", "2026-03-08T11:00"],
                    "wind_speed_10m": [10.0, 12.0],
                    "wind_direction_10m": [180.0, 200.0],
                    "wind_gusts_10m": [14.0, 16.0],
                },
            })
        return payloads

    monkeypatch.setattr(ew, "request_open_meteo_batch", fake_batch)

    nodes = pd.DataFrame({
        "node": [0, 1],
        "latitude": [47.6, 47.7],
        "longitude": [-122.3, -122.2],
    })
    out_nc = tmp_path / "wind.nc"
    tidy, ds = ew.fetch_ecmwf_wind_for_nodes(
        nodes=nodes,
        timezone="America/Los_Angeles",
        output_netcdf=out_nc,
        verbose=False,
    )

    assert len(tidy) == 4
    assert np.issubdtype(tidy["time"].dtype, np.datetime64)
    # 10:00 local should become 17:00 UTC on this date.
    assert str(tidy["time"].min()) == "2026-03-08 17:00:00"
    assert ds.sizes["time"] == 2
    assert ds.sizes["node"] == 2
    assert set(ds.data_vars) == {
        "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"
    }
    assert np.allclose(ds["latitude"].values, [47.6, 47.7])
    assert np.allclose(ds["longitude"].values, [-122.3, -122.2])
    assert ds.attrs["model"] == "ecmwf_ifs"
    assert out_nc.exists()


def test_fetch_route_wind_dataset_reuses_cached_nodes(monkeypatch, tmp_path):
    nodes_csv = tmp_path / "route_nodes.csv"
    pd.DataFrame({
        "latitude": [47.60, 47.62],
        "longitude": [-122.30, -122.28],
    }).to_csv(nodes_csv, index=False)

    called = {"discover": 0}

    def fake_discover(*args, **kwargs):
        called["discover"] += 1
        raise AssertionError("discover_ecmwf_nodes should not be called when cache exists")

    def fake_fetch(nodes, **kwargs):
        times = np.array(["2026-03-08T17:00:00", "2026-03-08T18:00:00"], dtype="datetime64[ns]")
        ds = ew.xr.Dataset(
            data_vars={
                "wind_speed_10m": (("time", "node"), np.array([[10.0, 11.0], [12.0, 13.0]], dtype=np.float32)),
                "wind_direction_10m": (("time", "node"), np.array([[180.0, 190.0], [200.0, 210.0]], dtype=np.float32)),
                "wind_gusts_10m": (("time", "node"), np.array([[15.0, 16.0], [17.0, 18.0]], dtype=np.float32)),
            },
            coords={
                "time": times,
                "node": np.array([0, 1], dtype=np.int32),
                "latitude": ("node", np.array([47.60, 47.62])),
                "longitude": ("node", np.array([-122.30, -122.28])),
            },
        )
        tidy = pd.DataFrame({
            "time": times.repeat(2),
            "node": [0, 1, 0, 1],
            "latitude": [47.60, 47.62, 47.60, 47.62],
            "longitude": [-122.30, -122.28, -122.30, -122.28],
            "wind_speed_10m": [10.0, 11.0, 12.0, 13.0],
            "wind_direction_10m": [180.0, 190.0, 200.0, 210.0],
            "wind_gusts_10m": [15.0, 16.0, 17.0, 18.0],
        })
        return tidy, ds

    monkeypatch.setattr(ew, "discover_ecmwf_nodes", fake_discover)
    monkeypatch.setattr(ew, "fetch_ecmwf_wind_for_nodes", fake_fetch)

    wps = [(47.60, -122.30), (47.55, -122.25)]
    tidy, ds, nodes = ew.fetch_route_wind_dataset(
        waypoints_latlon=wps,
        nodes_csv=nodes_csv,
        use_cached_nodes=True,
        verbose=False,
    )

    assert called["discover"] == 0
    assert len(nodes) == 2
    assert len(tidy) == 4
    assert ds.sizes["node"] == 2


def test_fetch_wind_falls_back_when_first_model_is_all_null(monkeypatch):
    def fake_batch(latitudes, longitudes, timezone, model, **kwargs):
        payloads = []
        bad_model = model == "ecmwf_ifs04"
        for lat, lon in zip(latitudes, longitudes):
            payloads.append({
                "latitude": lat,
                "longitude": lon,
                "hourly": {
                    "time": ["2026-03-08T10:00", "2026-03-08T11:00"],
                    "wind_speed_10m": [None, None] if bad_model else [10.0, 12.0],
                    "wind_direction_10m": [None, None] if bad_model else [180.0, 200.0],
                    "wind_gusts_10m": [None, None] if bad_model else [14.0, 16.0],
                },
            })
        return payloads

    monkeypatch.setattr(ew, "request_open_meteo_batch", fake_batch)

    nodes = pd.DataFrame({
        "node": [0, 1],
        "latitude": [47.6, 47.7],
        "longitude": [-122.3, -122.2],
    })
    tidy, ds = ew.fetch_ecmwf_wind_for_nodes(
        nodes=nodes,
        timezone="America/Los_Angeles",
        models=["ecmwf_ifs04", "ecmwf_ifs025"],
        min_non_null_coverage=0.5,
        verbose=False,
    )

    assert tidy["wind_speed_10m"].notna().all()
    assert ds.attrs["model"] == "ecmwf_ifs025"


def test_fetch_wind_raises_when_all_models_have_low_coverage(monkeypatch):
    def fake_batch(latitudes, longitudes, timezone, model, **kwargs):
        payloads = []
        for lat, lon in zip(latitudes, longitudes):
            payloads.append({
                "latitude": lat,
                "longitude": lon,
                "hourly": {
                    "time": ["2026-03-08T10:00", "2026-03-08T11:00"],
                    "wind_speed_10m": [None, None],
                    "wind_direction_10m": [None, None],
                    "wind_gusts_10m": [None, None],
                },
            })
        return payloads

    monkeypatch.setattr(ew, "request_open_meteo_batch", fake_batch)

    nodes = pd.DataFrame({
        "node": [0],
        "latitude": [47.6],
        "longitude": [-122.3],
    })

    with pytest.raises(RuntimeError, match="No usable Open-Meteo wind data found"):
        ew.fetch_ecmwf_wind_for_nodes(
            nodes=nodes,
            timezone="America/Los_Angeles",
            models=["ecmwf_ifs04"],
            min_non_null_coverage=0.1,
            verbose=False,
        )
