"""Smoke tests for pathsnap.pipeline using stubbed P1/P3 stages."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pathsnap import pipeline
from pathsnap.pipeline import Artifacts, load_artifacts, normalise_per_point, predict_trip

FAR_ARC = 5


def fake_candidates(trip, data_cfg) -> pd.DataFrame:
    """Two candidates per point: the true road arc (rank 0) and the far arc (rank 1)."""
    n = len(trip.track)
    true_arc = np.minimum(np.arange(n) * 5 // n, 4)
    rows = []
    for i in range(n):
        rows.append({"obs_idx": i, "arc_id": int(true_arc[i]), "cand_rank": 0, "dist_m": 5.0})
        rows.append({"obs_idx": i, "arc_id": FAR_ARC, "cand_rank": 1, "dist_m": 2000.0})
    df = pd.DataFrame(rows)
    df.insert(0, "trip_id", trip.trip_id)
    return df


@pytest.fixture
def baseline_art() -> Artifacts:
    return Artifacts(model=None, feature_columns=[], hmm_cfg={}, data_cfg={})


@pytest.fixture(autouse=True)
def stub_stages(monkeypatch):
    monkeypatch.setattr(pipeline, "_generate_candidates", fake_candidates)


def test_refuses_trip_with_route(trip_with_route, baseline_art):
    with pytest.raises(ValueError):
        predict_trip(trip_with_route, baseline_art)


def test_baseline_picks_nearest_and_falls_back_to_top1(trip, baseline_art):
    res = predict_trip(trip, baseline_art)
    assert res.mode == "baseline"
    assert res.decoder == "top1"  # P3 decoder still raises NotImplementedError
    assert len(res.points) == len(trip.track)
    assert FAR_ARC not in res.points["arc_id"].tolist()
    assert res.route_arcs == [0, 1, 2, 3, 4]
    assert {"candidates", "features", "model", "decode", "total"} <= res.timings_ms.keys()


def test_uses_hmm_decoder_when_available(monkeypatch, trip, baseline_art):
    def fake_decode(cands, trip_, cfg):
        assert trip_.route is None
        out = cands[cands["cand_rank"] == 0][["obs_idx", "arc_id"]]
        return out.assign(break_flag=False)

    monkeypatch.setattr(pipeline, "_decode", fake_decode)
    assert predict_trip(trip, baseline_art).decoder == "hmm"


def test_model_mode_uses_predict_proba(monkeypatch, trip):
    class FarArcLover:
        """Dummy model that prefers the far arc, to prove its scores are used."""

        def predict_proba(self, X):
            p = (X["dist_m"] > 100).astype(float).to_numpy() * 0.9 + 0.05
            return np.column_stack([1 - p, p])

    monkeypatch.setattr(pipeline, "_build_features", lambda c: c)
    art = Artifacts(model=FarArcLover(), feature_columns=["dist_m"], hmm_cfg={}, data_cfg={})
    res = predict_trip(trip, art)
    assert res.mode == "model"
    assert set(res.points["arc_id"].tolist()) == {FAR_ARC}


def test_normalise_per_point_sums_to_one_and_handles_zeros():
    cands = pd.DataFrame({"obs_idx": [0, 0, 1, 1, 1]})
    out = normalise_per_point(cands, np.array([3.0, 1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(out, [0.75, 0.25, 1 / 3, 1 / 3, 1 / 3])


def test_points_without_candidates_get_break(monkeypatch, trip, baseline_art):
    monkeypatch.setattr(
        pipeline, "_generate_candidates", lambda t, c: fake_candidates(t, c).query("obs_idx != 3")
    )
    pts = predict_trip(trip, baseline_art).points
    row = pts.loc[pts["obs_idx"] == 3].iloc[0]
    assert pd.isna(row["arc_id"]) and bool(row["break_flag"])


def test_load_artifacts_without_model_is_baseline(tmp_path):
    art = load_artifacts(str(tmp_path))
    assert art.mode == "baseline"
    assert "sigma" in art.hmm_cfg  # falls back to configs/hmm.yaml
