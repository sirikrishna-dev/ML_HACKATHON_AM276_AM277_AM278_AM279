"""Smoke tests for pathsnap.viz.folium_maps."""

from __future__ import annotations

import time

import folium
import numpy as np
import pandas as pd

from pathsnap.io.loader import Trip
from pathsnap.viz.folium_maps import (
    _arc_segments,
    _clip_network,
    _dedupe_consecutive,
    make_trip_map,
)


def test_map_has_all_layers(trip):
    fmap = make_trip_map(trip, reference_arcs=[0, 1, 2, 3, 4], predicted_arcs=[0, 0, 1, 2, 2, 3, 4])
    assert isinstance(fmap, folium.Map)
    html = fmap.get_root().render()
    for name in ("Road network", "GPS track", "Reference route", "Predicted route", "Start / end"):
        assert name in html


def test_clip_drops_far_arc(trip):
    kept = set(_clip_network(trip, buffer_m=150).tolist())
    assert kept == {0, 1, 2, 3, 4}


def test_clip_keeps_long_arc_passing_near_track(trip):
    # Arc from 1 km south to 1 km north crosses the track; both ends are far away.
    lon_mid = float(trip.track["lon"].mean())
    lat0 = float(trip.track["lat"].mean())
    extra = pd.DataFrame({"lon": [lon_mid, lon_mid], "lat": [lat0 - 0.009, lat0 + 0.009]})
    nodes = pd.concat([trip.nodes, extra], ignore_index=True)
    arcs = pd.concat(
        [trip.arcs, pd.DataFrame({"from_node": [len(nodes) - 2], "to_node": [len(nodes) - 1]})],
        ignore_index=True,
    )
    t2 = Trip(trip.trip_id, trip.split, trip.track, nodes, arcs)
    assert 6 in set(_clip_network(t2, buffer_m=150).tolist())


def test_unknown_arcs_are_skipped(trip):
    assert len(_arc_segments(trip, [0, 999, -5])) == 1
    make_trip_map(trip, predicted_arcs=[0, 999])  # must not raise


def test_dedupe_consecutive():
    assert _dedupe_consecutive([5, 5, 7, None, 7, float("nan"), 5]) == [5, 7, 5]
    assert _dedupe_consecutive(None) == []


def test_large_network_clips_fast():
    rng = np.random.default_rng(42)
    n_arcs = 75_000  # 150k nodes
    base = rng.uniform([77.0, 12.5], [78.0, 13.5], size=(n_arcs, 2))
    nodes = pd.DataFrame(np.vstack([base, base + 0.0005]), columns=["lon", "lat"])
    arcs = pd.DataFrame({"from_node": np.arange(n_arcs), "to_node": np.arange(n_arcs) + n_arcs})
    track = pd.DataFrame(
        {"lon": np.linspace(77.40, 77.45, 3600), "lat": np.linspace(12.90, 12.95, 3600), "t": np.arange(3600.0)}
    )
    big = Trip("big", "test", track, nodes, arcs)
    t0 = time.perf_counter()
    kept = _clip_network(big, buffer_m=150)
    assert time.perf_counter() - t0 < 5.0
    assert 0 < len(kept) < n_arcs // 10


def test_bounds_padded_for_flat_track(trip):
    from pathsnap.viz.folium_maps import _track_bounds

    (s, w), (n, e) = _track_bounds(trip.track)
    assert n - s > 0 and s < trip.track["lat"].min() and e > trip.track["lon"].max()
