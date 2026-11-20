"""End-to-end prediction pipeline: candidates -> features -> model -> HMM.

P4 wires the stages together; each stage is owned by another module:
candidates (P1), features (P1), model (P2), decoder (P3). The pipeline never
reads ``trip.route`` and refuses trips that carry one.

Until real artifacts exist the pipeline runs in *baseline* mode: candidates
are scored by ``exp(-cand_rank)`` (nearest candidate wins) and, if the HMM
decoder is not implemented yet, each point simply takes its top-scoring arc.
``PredictionResult.mode`` always says which path ran.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional

import joblib
import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS = REPO_ROOT / "artifacts"
DEFAULT_HMM_CFG = REPO_ROOT / "configs" / "hmm.yaml"
DEFAULT_DATA_CFG = REPO_ROOT / "configs" / "data.yaml"

MODE_MODEL = "model"
MODE_BASELINE = "baseline"


@dataclass
class Artifacts:
    """Everything the pipeline loads from disk once per process."""

    model: Any | None
    feature_columns: list[str]
    hmm_cfg: dict
    data_cfg: dict
    metadata: dict = field(default_factory=dict)

    @property
    def mode(self) -> str:
        """``"model"`` when a trained model is loaded, else ``"baseline"``."""
        return MODE_MODEL if self.model is not None else MODE_BASELINE


@dataclass
class PredictionResult:
    """Output of ``predict_trip``.

    ``points`` has one row per GPS observation: ``obs_idx, arc_id, score,
    break_flag``. ``arc_id`` is ``<NA>`` where a point had no candidate.
    """

    trip_id: str
    points: pd.DataFrame
    route_arcs: list[int]
    timings_ms: dict[str, float]
    mode: str
    decoder: str


def _read_yaml(path: Path) -> dict:
    """Read a YAML file, returning ``{}`` if it does not exist."""
    if not path.exists():
        return {}
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=4)
def load_artifacts(path: str = str(DEFAULT_ARTIFACTS)) -> Artifacts:
    """Load model, feature list and configs from ``path`` (cached per path).

    Missing ``final_model.joblib`` means baseline mode. ``hmm.yaml`` in the
    artifacts folder (the tuned copy) wins over ``configs/hmm.yaml``.
    """
    root = Path(path)
    model_path = root / "final_model.joblib"
    model = joblib.load(model_path) if model_path.exists() else None

    cols_path = root / "feature_columns.json"
    feature_columns = json.loads(cols_path.read_text()) if cols_path.exists() else []
    if model is not None and not feature_columns:
        raise FileNotFoundError(f"{cols_path} is required when a model is present")

    hmm_path = root / "hmm.yaml"
    hmm_cfg = _read_yaml(hmm_path if hmm_path.exists() else DEFAULT_HMM_CFG)
    meta_path = root / "metadata.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    art = Artifacts(model, feature_columns, hmm_cfg, _read_yaml(DEFAULT_DATA_CFG), metadata)
    logger.info("Loaded artifacts from %s (mode=%s)", root, art.mode)
    return art


def _generate_candidates(trip, data_cfg: dict) -> pd.DataFrame:
    """Call P1's candidate generator (imported lazily so this module loads without it)."""
    from pathsnap.candidates.generate import generate_candidates

    return generate_candidates(
        trip, radius_m=data_cfg.get("candidate_radius_m", 100), k=data_cfg.get("candidate_k", 8)
    )


def _build_features(cands: pd.DataFrame) -> pd.DataFrame:
    """Call P1's feature builder."""
    from pathsnap.features.build import build_features

    return build_features(cands)


def _decode(cands: pd.DataFrame, trip, cfg: dict) -> pd.DataFrame:
    """Call P3's HMM/Viterbi decoder."""
    from pathsnap.hmm.viterbi import decode

    return decode(cands, trip, cfg)


def score_candidates(cands: pd.DataFrame, art: Artifacts) -> np.ndarray:
    """Raw score per candidate row: model probability, or ``exp(-cand_rank)`` in baseline mode."""
    if art.model is None:
        return np.exp(-cands["cand_rank"].to_numpy(dtype=float))
    return art.model.predict_proba(cands[art.feature_columns])[:, 1]


def normalise_per_point(cands: pd.DataFrame, raw: np.ndarray) -> np.ndarray:
    """Rescale scores so they sum to 1 within each ``obs_idx``.

    A point whose scores are all zero gets an equal share for each candidate,
    so the decoder never sees a point with no probability mass.
    """
    s = pd.Series(np.asarray(raw, dtype=float), index=cands.index)
    groups = cands["obs_idx"]
    totals = s.groupby(groups).transform("sum")
    counts = s.groupby(groups).transform("size")
    return np.where(totals > 0, s / totals.where(totals > 0, 1.0), 1.0 / counts).astype(float)


def top1_per_point(cands: pd.DataFrame) -> pd.DataFrame:
    """Fallback decoder: each point takes its highest-scoring arc, no continuity."""
    best = cands.loc[cands.groupby("obs_idx")["score"].idxmax(), ["obs_idx", "arc_id"]]
    return best.assign(break_flag=False).reset_index(drop=True)


def _assemble_points(decoded: pd.DataFrame, cands: pd.DataFrame, n_obs: int) -> pd.DataFrame:
    """Give every GPS observation a row; points without candidates get ``<NA>`` and a break."""
    scores = cands[["obs_idx", "arc_id", "score"]]
    pts = decoded[["obs_idx", "arc_id", "break_flag"]].merge(scores, on=["obs_idx", "arc_id"], how="left")
    pts = pts.set_index("obs_idx").reindex(range(n_obs))
    pts["arc_id"] = pts["arc_id"].astype("Int64")
    pts["break_flag"] = pts["break_flag"].astype("boolean").fillna(True).astype(bool)
    return pts.reset_index().rename(columns={"index": "obs_idx"})[["obs_idx", "arc_id", "score", "break_flag"]]


def _route_from_points(points: pd.DataFrame) -> list[int]:
    """Ordered arc list with consecutive repeats and unmatched points removed."""
    route: list[int] = []
    for a in points["arc_id"].dropna():
        if not route or route[-1] != int(a):
            route.append(int(a))
    return route


def _timed(timings: dict[str, float], key: str, fn: Callable[[], Any]) -> Any:
    """Run ``fn`` and store its wall time in milliseconds under ``key``."""
    t0 = time.perf_counter()
    out = fn()
    timings[key] = round((time.perf_counter() - t0) * 1000, 2)
    return out


def predict_trip(trip, artifacts: Optional[Artifacts] = None) -> PredictionResult:
    """Predict one arc per GPS point for ``trip``, without ground truth.

    Stages: candidates -> features (model mode only) -> scores -> per-point
    normalisation -> HMM decode (top-1 fallback if the decoder is missing).

    Raises:
        ValueError: if ``trip.route`` is set; load trips with ``with_route=False``.
    """
    if trip.route is not None:
        raise ValueError("predict_trip received a trip with a route; load it with with_route=False")
    art = artifacts or load_artifacts()
    timings: dict[str, float] = {}
    t_start = time.perf_counter()

    cands = _timed(timings, "candidates", lambda: _generate_candidates(trip, art.data_cfg))
    if art.model is not None:
        cands = _timed(timings, "features", lambda: _build_features(cands))
    else:
        timings["features"] = 0.0
    raw = _timed(timings, "model", lambda: score_candidates(cands, art))
    cands = cands.assign(score=normalise_per_point(cands, raw))

    decoder = "hmm"
    try:
        decoded = _timed(timings, "decode", lambda: _decode(cands, trip, art.hmm_cfg))
    except NotImplementedError:
        decoder = "top1"
        decoded = _timed(timings, "decode", lambda: top1_per_point(cands))
    if decoder == "top1":
        logger.warning("HMM decoder not implemented; used per-point top-1 for %s", trip.trip_id)

    points = _assemble_points(decoded, cands, len(trip.track))
    timings["total"] = round((time.perf_counter() - t_start) * 1000, 2)
    return PredictionResult(trip.trip_id, points, _route_from_points(points), timings, art.mode, decoder)
