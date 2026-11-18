# PathSnap — GPS Map Matching

PathSnap reconstructs likely road routes from noisy GPS trajectories by matching GPS observations to candidate road segments, scoring candidates with machine learning, and decoding a connected route with an HMM/Viterbi algorithm.

## Pipeline

1. Load a trip (GPS track, nodes, arcs, and optional reference route).
2. Project coordinates into a local metric coordinate system.
3. Generate candidate road arcs for each GPS observation.
4. Build candidate features and train/evaluate classifiers.
5. Decode the most likely connected arc sequence with HMM + Viterbi.
6. Serve predictions through FastAPI and display them in a UI.

## Team ownership

- **P1 — Data, labels and candidates:** `pathsnap/io`, `pathsnap/geo`, `pathsnap/labels`, `pathsnap/candidates`, `pathsnap/features`, `notebooks/01_eda.ipynb`, `scripts/build_candidates.py`
- **P2 — Model experiments:** `pathsnap/models`, `pathsnap/eval/metrics.py`, `scripts/train_models.py`, `notebooks/02_models.ipynb`
- **P3 — Bayesian HPO + HMM/Viterbi:** `pathsnap/hpo`, `pathsnap/hmm`, `scripts/tune.py`, `scripts/decode_eval.py`, `notebooks/03_hpo_hmm.ipynb`
- **P4 — API, UI, maps and integration:** `api`, `ui`, `pathsnap/viz`, `pathsnap/pipeline.py`, `Makefile`, final report assembly

## Shared contracts

### Trip
`Trip(trip_id, split, track, nodes, arcs, route=None)`

- `track`: DataFrame with `lon`, `lat`, `t`
- `nodes`: DataFrame with `lon`, `lat`
- `arcs`: DataFrame with `from_node`, `to_node`
- `route`: optional list of arc IDs; prediction code must use `route=None`

### Candidate table
One row per `(trip_id, obs_idx, arc_id)`. Required columns:

`trip_id, obs_idx, t, arc_id, cand_rank, <FEATURE_COLUMNS...>, label, has_true_cand`

`label` is training-only. `has_true_cand` supports candidate miss-rate measurement.

### Model
Every model follows the scikit-learn estimator interface and is persisted with joblib. `predict_proba(X[FEATURE_COLUMNS])[:, 1]` supplies candidate scores. Scores are normalised within each GPS observation before Top-1 selection and HMM decoding.

### Decoder
`decode(cands_df, trip, cfg)` returns a DataFrame containing:

`obs_idx, arc_id, break_flag`

The decoder must not use `trip.route`.

### API
- `GET /health`
- `GET /trips`
- `POST /predict/{trip_id}`

Ground truth is loaded only by the server-side evaluation module after prediction; it must never enter the prediction pipeline or UI.

## Data and evaluation rules

- Keep raw data, evaluator data, processed Parquet files and large generated artifacts out of Git.
- Use trip-level `GroupKFold` on the 80 training trips.
- Use the 10 `student_test` trips only for the comparisons allowed by the hackathon instructions; do not fit models or tune model hyperparameters on them.
- Evaluator data and evaluator ground truth must never be used during development.
- Never hardcode predictions or metrics.
- Use random seed `42` where applicable.
- Record experiments and report only metrics produced by executed code.

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Data files are intentionally not included in this skeleton. Place the provided data under `data/raw/train/` and `data/raw/student_test/` locally. Do not commit them.
