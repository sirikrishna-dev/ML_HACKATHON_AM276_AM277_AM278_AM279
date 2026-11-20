"""Shared synthetic fixtures so tests run without the course data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pathsnap.io.loader import Trip

LAT0, LON0 = 12.9716, 77.5946  # arbitrary origin
DEG_PER_M_LAT = 1 / 111_195.0


def make_synthetic_trip(with_route: bool = False) -> Trip:
    """A 500 m east-west road (arcs 0-4) plus one arc 2 km north (arc 5).

    The GPS track runs along the road, about 5 m north of it, with 20 fixes.
    """
    deg_per_m_lon = DEG_PER_M_LAT / np.cos(np.radians(LAT0))
    road_lon = LON0 + np.arange(6) * 100 * deg_per_m_lon
    nodes = pd.DataFrame(
        {
            "lon": np.r_[road_lon, LON0, LON0 + 100 * deg_per_m_lon],
            "lat": np.r_[np.full(6, LAT0), LAT0 + 2000 * DEG_PER_M_LAT, LAT0 + 2000 * DEG_PER_M_LAT],
        }
    )
    arcs = pd.DataFrame({"from_node": [0, 1, 2, 3, 4, 6], "to_node": [1, 2, 3, 4, 5, 7]})
    n = 20
    track = pd.DataFrame(
        {
            "lon": np.linspace(road_lon[0], road_lon[-1], n),
            "lat": np.full(n, LAT0 + 5 * DEG_PER_M_LAT),
            "t": np.arange(n, dtype=float),
        }
    )
    route = [0, 1, 2, 3, 4] if with_route else None
    return Trip("synthetic", "test", track, nodes, arcs, route)


@pytest.fixture
def trip() -> Trip:
    """Synthetic trip without a route (prediction-safe)."""
    return make_synthetic_trip()


@pytest.fixture
def trip_with_route() -> Trip:
    """Synthetic trip that carries a ground-truth route (must be refused by the pipeline)."""
    return make_synthetic_trip(with_route=True)
