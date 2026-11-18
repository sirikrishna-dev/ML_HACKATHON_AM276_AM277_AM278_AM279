"""Load trips from disk into the shared Trip object (Contract 1)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SPLIT_DIRS = {
    "train": Path("data/raw/train"),
    "student_test": Path("data/raw/student_test"),
    "evaluator_test": Path("data/evaluator_test"),
}
_SEP = r"[\s,]+"  # spaces, tabs or commas


@dataclass(frozen=True)
class Trip:
    """One trip with its own local road network.

    Attributes:
        trip_id: Trip folder name, e.g. "00000048". Also the CV group key.
        split: "train", "student_test" or "evaluator_test".
        track: GPS points, columns [lon, lat, t]. Row index is obs_idx.
        nodes: Network nodes, columns [lon, lat]. Row index is node_id.
        arcs: Directed arcs, columns [from_node, to_node]. Row index is arc_id.
        route: Ground-truth arc_ids in travel order, NOT one per GPS point.
            None unless explicitly loaded.
        issues: Data-quality problems found on load. Empty means clean.
    """

    trip_id: str
    split: str
    track: pd.DataFrame
    nodes: pd.DataFrame
    arcs: pd.DataFrame
    route: list[int] | None = None
    issues: list[str] = field(default_factory=list)


def data_root() -> Path:
    """Project data root; override with env PATHSNAP_DATA_ROOT."""
    return Path(os.environ.get("PATHSNAP_DATA_ROOT", "."))


def _split_dir(split: str) -> Path:
    if split not in SPLIT_DIRS:
        raise ValueError(f"unknown split {split!r}; expected one of {list(SPLIT_DIRS)}")
    return data_root() / SPLIT_DIRS[split]


def list_trips(split: str) -> list[str]:
    """Sorted trip folder names for a split (empty list if the split is absent)."""
    d = _split_dir(split)
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir())


def _read_table(path: Path, columns: list[str], dtype, issues: list[str]) -> pd.DataFrame:
    """Read a headerless numeric table; on any problem record an issue and return empty."""
    empty = pd.DataFrame({c: pd.Series(dtype=dtype) for c in columns})
    if not path.is_file():
        issues.append(f"missing file: {path.name}")
        return empty
    if path.stat().st_size == 0:
        return empty
    try:
        df = pd.read_csv(path, header=None, sep=_SEP, engine="python")
    except Exception as exc:  # noqa: BLE001 - we want to report, not crash
        issues.append(f"{path.name}: unreadable ({exc})")
        return empty
    if df.shape[1] != len(columns):
        issues.append(f"{path.name}: expected {len(columns)} columns, found {df.shape[1]}")
        return empty
    df.columns = columns
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.astype(dtype) if dtype is not float and not df.isna().any().any() else df


def _read_route(path: Path, issues: list[str]) -> list[int] | None:
    """Read arc ids in travel order; None if the file does not exist."""
    if not path.is_file():
        return None
    try:
        text = path.read_text().replace(",", " ").split()
        return [int(x) for x in text]
    except ValueError as exc:
        issues.append(f"{path.name}: non-integer value ({exc})")
        return None


def validate_trip(trip: Trip) -> list[str]:
    """Return a list of data-quality issues (empty if the trip is clean)."""
    out: list[str] = []
    tr, nd, ar = trip.track, trip.nodes, trip.arcs
    if tr.empty:
        out.append("track is empty")
    else:
        if tr[["lon", "lat", "t"]].isna().any().any():
            out.append("track has NaN or non-numeric values")
        if ((tr["lon"].abs() > 180) | (tr["lat"].abs() > 90)).any():
            out.append("track has coordinates outside lon±180 / lat±90")
        dt = np.diff(tr["t"].to_numpy(dtype=float))
        if (dt == 0).any():
            out.append(f"track has {int((dt == 0).sum())} duplicate timestamps")
        if (dt < 0).any():
            out.append(f"track has {int((dt < 0).sum())} out-of-order timestamps")
    if nd.empty:
        out.append("nodes is empty")
    if ar.empty:
        out.append("arcs is empty")
    elif not nd.empty:
        if ar.isna().any().any():
            out.append("arcs has NaN or non-numeric values")
        else:
            bad = (ar.to_numpy() < 0) | (ar.to_numpy() >= len(nd))
            if bad.any():
                out.append(f"arcs reference node ids outside 0..{len(nd) - 1}")
    if trip.route is not None and len(ar) > 0:
        r = np.asarray(trip.route)
        if r.size and ((r < 0) | (r >= len(ar))).any():
            out.append(f"route has arc ids outside 0..{len(ar) - 1}")
    return out


def load_trip(split: str, trip_id: str, with_route: bool = True) -> Trip:
    """Load one trip. Never raises on bad data; problems land in ``trip.issues``.

    With ``with_route=False`` the .route file is never opened. A missing .route
    gives ``route=None`` with no issue (student_test/evaluator trips have none).
    """
    folder = _split_dir(split) / trip_id
    issues: list[str] = []
    if not folder.is_dir():
        issues.append(f"missing trip folder: {folder}")
    track = _read_table(folder / f"{trip_id}.track", ["lon", "lat", "t"], float, issues)
    nodes = _read_table(folder / f"{trip_id}.nodes", ["lon", "lat"], float, issues)
    arcs = _read_table(folder / f"{trip_id}.arcs", ["from_node", "to_node"], "int64", issues)
    route = _read_route(folder / f"{trip_id}.route", issues) if with_route else None
    trip = Trip(trip_id, split, track, nodes, arcs, route, [])
    trip.issues.extend(issues + validate_trip(trip))
    return trip