"""Folium map helpers for PathSnap trips.

P4 owns this module; P1 uses ``make_trip_map`` for EDA and the UI uses it to
show predictions. Nothing here reads ``trip.route`` on its own: reference arcs
are drawn only when the caller passes them in explicitly.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Sequence

import folium
import numpy as np
import shapely
from branca.element import Element

logger = logging.getLogger(__name__)

EARTH_RADIUS_M = 6_371_008.8
MAX_GPS_DOTS = 300
ESRI_BASE = "https://server.arcgisonline.com/ArcGIS/rest/services"
ESRI_GREY_URL = f"{ESRI_BASE}/Canvas/World_Light_Gray_Base/MapServer/tile/{{z}}/{{y}}/{{x}}"
ESRI_STREETS_URL = f"{ESRI_BASE}/World_Street_Map/MapServer/tile/{{z}}/{{y}}/{{x}}"
ESRI_ATTR = "Tiles &copy; Esri"

# Okabe-Ito palette: distinguishable for colour-blind viewers.
COLOURS = {
    "network": "#9E9E9E",
    "gps": "#0072B2",
    "reference": "#009E73",
    "predicted": "#D55E00",
    "start": "#009E73",
    "end": "#CC3311",
}


def _local_xy(
    lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float
) -> tuple[np.ndarray, np.ndarray]:
    """Project lon/lat degrees to metres on a flat plane centred at (lon0, lat0).

    Equirectangular approximation: good to well under a metre over the extent
    of a single trip, which is enough for drawing and clipping.
    """
    x = np.radians(np.asarray(lon, dtype=float) - lon0) * EARTH_RADIUS_M * np.cos(np.radians(lat0))
    y = np.radians(np.asarray(lat, dtype=float) - lat0) * EARTH_RADIUS_M
    return x, y


def _clip_network(trip, buffer_m: float) -> np.ndarray:
    """Return arc IDs whose line passes within ``buffer_m`` metres of the GPS track.

    Two passes: a cheap bounding-box filter on arc endpoints removes most of a
    large network, then an exact line-to-line distance test keeps arcs that
    really come close (including long arcs whose endpoints are both far away).
    """
    track, nodes, arcs = trip.track, trip.nodes, trip.arcs
    if len(track) == 0 or len(arcs) == 0:
        return np.array([], dtype=int)

    lon0, lat0 = float(track["lon"].mean()), float(track["lat"].mean())
    tx, ty = _local_xy(track["lon"].to_numpy(), track["lat"].to_numpy(), lon0, lat0)
    nx, ny = _local_xy(nodes["lon"].to_numpy(), nodes["lat"].to_numpy(), lon0, lat0)

    src = arcs["from_node"].to_numpy()
    dst = arcs["to_node"].to_numpy()
    valid = (src >= 0) & (src < len(nodes)) & (dst >= 0) & (dst < len(nodes))
    arc_ids = arcs.index.to_numpy()[valid]
    src, dst = src[valid], dst[valid]

    # Pass 1: keep arcs whose own bounding box overlaps the padded track box.
    xmin, xmax = tx.min() - buffer_m, tx.max() + buffer_m
    ymin, ymax = ty.min() - buffer_m, ty.max() + buffer_m
    ax0, ax1 = np.minimum(nx[src], nx[dst]), np.maximum(nx[src], nx[dst])
    ay0, ay1 = np.minimum(ny[src], ny[dst]), np.maximum(ny[src], ny[dst])
    in_box = (ax1 >= xmin) & (ax0 <= xmax) & (ay1 >= ymin) & (ay0 <= ymax)
    arc_ids, src, dst = arc_ids[in_box], src[in_box], dst[in_box]
    if len(arc_ids) == 0:
        return arc_ids

    # Pass 2: exact distance from each arc segment to the track line.
    coords = np.column_stack([tx, ty])
    track_geom = shapely.Point(coords[0]) if len(coords) == 1 else shapely.LineString(coords)
    seg_coords = np.stack(
        [np.column_stack([nx[src], ny[src]]), np.column_stack([nx[dst], ny[dst]])], axis=1
    )
    segments = shapely.linestrings(seg_coords)
    near = shapely.dwithin(segments, track_geom, buffer_m)
    return arc_ids[near]


def _arc_segments(trip, arc_ids: Iterable[int]) -> list[list[list[float]]]:
    """Turn arc IDs into ``[[lat, lon], [lat, lon]]`` segments for Folium.

    Unknown arc IDs or arcs pointing at missing nodes are skipped with a warning
    rather than crashing the map.
    """
    nodes, arcs = trip.nodes, trip.arcs
    ids = np.asarray(list(arc_ids), dtype=np.int64)
    known = np.isin(ids, arcs.index.to_numpy())
    src = arcs.loc[ids[known], "from_node"].to_numpy(dtype=np.int64)
    dst = arcs.loc[ids[known], "to_node"].to_numpy(dtype=np.int64)
    ok = (src >= 0) & (src < len(nodes)) & (dst >= 0) & (dst < len(nodes))
    skipped = int((~known).sum() + (~ok).sum())
    if skipped:
        logger.warning("Trip %s: skipped %d unknown arc id(s)", trip.trip_id, skipped)

    lat, lon = nodes["lat"].to_numpy(dtype=float), nodes["lon"].to_numpy(dtype=float)
    src, dst = src[ok], dst[ok]
    starts = np.column_stack([lat[src], lon[src]])
    ends = np.column_stack([lat[dst], lon[dst]])
    return np.stack([starts, ends], axis=1).tolist()


def _dedupe_consecutive(arc_ids: Optional[Sequence[int]]) -> list[int]:
    """Collapse runs of the same arc ID, e.g. ``[5, 5, 7, 7, 5]`` -> ``[5, 7, 5]``.

    The decoder emits one arc per GPS point; the map only needs each arc once
    per visit. ``None`` and NaN entries (unmatched points) are dropped.
    """
    if arc_ids is None:
        return []
    out: list[int] = []
    for a in arc_ids:
        if a is None or (isinstance(a, float) and np.isnan(a)):
            continue
        a = int(a)
        if not out or out[-1] != a:
            out.append(a)
    return out


def _add_legend(fmap: folium.Map, entries: list[tuple[str, str]]) -> None:
    """Add a fixed legend box in the bottom-left corner of the map."""
    rows = "".join(
        f'<div><span style="display:inline-block;width:14px;height:4px;'
        f'background:{colour};margin-right:6px;vertical-align:middle"></span>{label}</div>'
        for label, colour in entries
    )
    html = (
        '<div style="position:fixed;bottom:24px;left:24px;z-index:9999;'
        "background:white;padding:8px 10px;border:1px solid #bbb;border-radius:4px;"
        f'font:12px/1.6 sans-serif;color:#222">{rows}</div>'
    )
    fmap.get_root().html.add_child(Element(html))


def _add_lines(
    fmap: folium.Map, segments: list, name: str, colour: str, weight: float, opacity: float
) -> None:
    """Add all segments as ONE multi-polyline inside a toggleable FeatureGroup."""
    group = folium.FeatureGroup(name=name)
    if segments:
        folium.PolyLine(segments, color=colour, weight=weight, opacity=opacity).add_to(group)
    group.add_to(fmap)


def _add_gps(fmap: folium.Map, trip) -> None:
    """Add the GPS track line plus a thinned set of hoverable dots."""
    track = trip.track
    latlon = track[["lat", "lon"]].to_numpy().tolist()
    group = folium.FeatureGroup(name="GPS track")
    if len(latlon) > 1:
        folium.PolyLine(latlon, color=COLOURS["gps"], weight=2, opacity=0.8).add_to(group)
    stride = max(1, int(np.ceil(len(track) / MAX_GPS_DOTS)))
    has_t = "t" in track.columns
    for i in range(0, len(track), stride):
        tip = f"obs {i}" + (f", t={track['t'].iat[i]}" if has_t else "")
        folium.CircleMarker(
            latlon[i], radius=2, color=COLOURS["gps"], fill=True, fill_opacity=0.9, weight=1, tooltip=tip
        ).add_to(group)
    group.add_to(fmap)


def _add_endpoints(fmap: folium.Map, trip) -> None:
    """Mark the first and last GPS fix with labelled circle markers."""
    track = trip.track
    group = folium.FeatureGroup(name="Start / end")
    for idx, label, colour in ((0, "Start", COLOURS["start"]), (len(track) - 1, "End", COLOURS["end"])):
        folium.CircleMarker(
            [float(track["lat"].iat[idx]), float(track["lon"].iat[idx])],
            radius=7, color="white", weight=2, fill=True, fill_color=colour, fill_opacity=1.0, tooltip=label,
        ).add_to(group)
    group.add_to(fmap)


def _add_basemaps(fmap: folium.Map) -> None:
    """Add two Esri background maps: light grey (default) and streets.

    Esri is used because OpenStreetMap's servers return 403 for pages opened
    from disk or embedded without a Referer header, and CartoDB now needs an
    API key. Esri only has real tiles up to zoom 16 (grey) / 17 (streets); past
    that, ``max_native_zoom`` stretches the last real tiles instead of showing
    "Map data not yet available".
    """
    folium.TileLayer(
        ESRI_GREY_URL, attr=ESRI_ATTR, name="Light grey", max_native_zoom=16, max_zoom=20
    ).add_to(fmap)
    folium.TileLayer(
        ESRI_STREETS_URL, attr=ESRI_ATTR, name="Streets", max_native_zoom=17, max_zoom=20, show=False
    ).add_to(fmap)


def make_trip_map(
    trip,
    gps: bool = True,
    network: bool = True,
    reference_arcs: Optional[Sequence[int]] = None,
    predicted_arcs: Optional[Sequence[int]] = None,
    buffer_m: float = 150.0,
    tiles: bool = True,
) -> folium.Map:
    """Build an interactive Folium map of one trip.

    Args:
        trip: a ``Trip`` (track, nodes, arcs). ``trip.route`` is never read here.
        gps: draw the raw GPS track.
        network: draw road arcs within ``buffer_m`` of the track.
        reference_arcs: ground-truth arc IDs, drawn in green when given.
        predicted_arcs: predicted arc IDs (one per point is fine; repeats collapse).
        buffer_m: clipping distance for the network layer, in metres.
        tiles: add background maps (see ``_add_basemaps``); ``False`` for blank.

    Returns:
        A ``folium.Map`` with one toggleable layer per item, a legend and the view
        fitted to the GPS track.
    """
    track = trip.track
    if len(track) == 0:
        raise ValueError(f"Trip {trip.trip_id} has an empty track")

    centre = [float(track["lat"].mean()), float(track["lon"].mean())]
    fmap = folium.Map(location=centre, tiles=None, control_scale=True, max_zoom=20)
    if tiles:
        _add_basemaps(fmap)
    legend: list[tuple[str, str]] = []

    if network:
        kept = _clip_network(trip, buffer_m)
        _add_lines(fmap, _arc_segments(trip, kept), "Road network", COLOURS["network"], 1.5, 0.7)
        legend.append((f"Road network (within {buffer_m:g} m)", COLOURS["network"]))
    if gps:
        _add_gps(fmap, trip)
        legend.append(("GPS track", COLOURS["gps"]))
    if reference_arcs is not None:
        segs = _arc_segments(trip, _dedupe_consecutive(reference_arcs))
        _add_lines(fmap, segs, "Reference route", COLOURS["reference"], 7, 0.55)
        legend.append(("Reference route", COLOURS["reference"]))
    if predicted_arcs is not None:
        segs = _arc_segments(trip, _dedupe_consecutive(predicted_arcs))
        _add_lines(fmap, segs, "Predicted route", COLOURS["predicted"], 3.5, 0.95)
        legend.append(("Predicted route", COLOURS["predicted"]))

    _add_endpoints(fmap, trip)
    legend += [("Start", COLOURS["start"]), ("End", COLOURS["end"])]
    _add_legend(fmap, legend)
    folium.LayerControl(collapsed=False).add_to(fmap)

    fmap.fit_bounds(_track_bounds(track))
    return fmap


def _track_bounds(track, rel_pad: float = 0.05, min_pad_deg: float = 0.0005) -> list[list[float]]:
    """``[[south, west], [north, east]]`` around the track, padded on every side.

    Padding keeps start/end markers off the edge, and the minimum pad (about
    50 m) avoids a zero-height box for a perfectly straight east-west track,
    which makes Leaflet jump to maximum zoom.
    """
    lat_min, lat_max = float(track["lat"].min()), float(track["lat"].max())
    lon_min, lon_max = float(track["lon"].min()), float(track["lon"].max())
    pad_lat = max((lat_max - lat_min) * rel_pad, min_pad_deg)
    pad_lon = max((lon_max - lon_min) * rel_pad, min_pad_deg)
    return [[lat_min - pad_lat, lon_min - pad_lon], [lat_max + pad_lat, lon_max + pad_lon]]
