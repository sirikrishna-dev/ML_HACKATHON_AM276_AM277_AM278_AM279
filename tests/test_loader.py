"""Smoke tests for pathsnap.io.loader using a tiny synthetic trip."""
from pathlib import Path

import pytest

from pathsnap.io import loader


def _make(root: Path, split_dir: str, tid: str, track: str, route: str | None):
    d = root / split_dir / tid
    d.mkdir(parents=True)
    (d / f"{tid}.track").write_text(track)
    (d / f"{tid}.nodes").write_text("77.0 12.0\n77.001 12.0\n77.002 12.0\n")
    (d / f"{tid}.arcs").write_text("0 1\n1 2\n")
    if route is not None:
        (d / f"{tid}.route").write_text(route)


@pytest.fixture()
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("PATHSNAP_DATA_ROOT", str(tmp_path))
    good = "77.0 12.0 0\n77.0005 12.0 1\n77.0015 12.0 2\n"
    _make(tmp_path, "data/raw/train", "00000001", good, "0\n1\n")
    _make(tmp_path, "data/raw/student_test", "00000002", good, None)
    _make(tmp_path, "data/raw/train", "00000003", "77.0 12.0 5\n77.0 12.0 5\n200 12.0 3\n", "0\n9\n")
    return tmp_path


def test_clean_trip_loads(root):
    trip = loader.load_trip("train", "00000001")
    assert trip.issues == [] and trip.route == [0, 1]
    assert list(trip.track.columns) == ["lon", "lat", "t"] and len(trip.track) == 3


def test_route_not_read_when_disabled(root):
    assert loader.load_trip("train", "00000001", with_route=False).route is None


def test_missing_route_is_not_an_issue(root):
    trip = loader.load_trip("student_test", "00000002")
    assert trip.route is None and trip.issues == []


def test_bad_trip_reports_issues_instead_of_raising(root):
    trip = loader.load_trip("train", "00000003")
    text = " ".join(trip.issues)
    assert "duplicate timestamps" in text and "out-of-order" in text
    assert "outside lon" in text and "route has arc ids" in text


def test_missing_folder_does_not_crash(root):
    assert loader.load_trip("train", "99999999").issues


def test_list_trips_sorted(root):
    assert loader.list_trips("train") == ["00000001", "00000003"]