# Design

This document explains how `mvts-forecast` is put together: the layering, the
data flow through a single walk-forward window, the leakage invariants the
compute core guarantees, and the testing strategy that keeps the honest NULL
honest. For *why* individual contested choices were made, see the numbered ADRs
in [`docs/decisions/`](decisions/).

## Goals and non-goals

**Goals**

- A pure, typed (`mypy --strict`, `py.typed`), side-effect-free compute core that
  can be audited line by line and vendored into a backend without dragging
  torch, an inference engine, UI, or network dependencies along at import time.
- A faithful, leakage-free multivariate-transformer benchmark: a PatchTST-style
  encoder and a simplified interpretable variable-selection transformer raced
  against an LSTM, per-series ARIMA, and a naive random walk — on the **same**
  windows, the **same** RevIN, the **same** purged walk-forward, so any
  difference is the model, not the harness.
- A statistically defensible verdict (`deep_beats_naive`) that survives
  multiplicity correction (Deflated Sharpe) and a Diebold-Mariano test, and is
  *mechanically* prevented from over-claiming.
- Train heavy (torch, offline) → serve lean (onnxruntime, no torch), with the
  served ONNX graph parity-certified to the trained torch forward pass.

**Non-goals**

- Beating a random walk on noisy daily returns. The honest finding — documented
  and regression-locked — is that the deep models do **not**, by a
  DM-significant margin with a positive Deflated Sharpe.
- A live trading system or a profit claim. This is a research/benchmark library.
- A price-level forecaster. There is **no price-level R²** anywhere
  ([ADR-0005](decisions/0005-no-price-level-r2.md)); skill is judged in return
  space only.

## Layered architecture

The package is strictly layered; each layer imports only from the ones below it.
`src/` has **zero import-time side effects** (no torch, no onnxruntime, no
statsmodels, no plotly, no I/O, no RNG draw at import), guarded by a subprocess
import-purity test. torch lives behind the `[train]` extra and is reachable only
from the offline `train.py`; onnxruntime lives behind `[serve]` and is imported
lazily inside the model layer.

```
              cli.py (Typer, lazy)   plots.py (Plotly, lazy)   serve.py (API entry)
                     |                      |                       |
   ┌─────────────────┴───────────────────────┴───────────────────────┘
   │                          evaluation/
   │      metrics.py · diebold_mariano.py · dsr.py · verdict.py
   │  (return-space metrics · DM+HAC · Deflated Sharpe · pure verdict deriver)
   ├────────────────────────────────────────────────────────────────────────
   │                          models/
   │   naive.py · arima.py        lstm.py · patchtst.py · transformer_vs.py
   │   (live, torch-free)         (torch [train], offline)      onnx_runtime.py
   │                                                            (serve, no torch)
   ├────────────────────────────────────────────────────────────────────────
   │                          windowing/
   │        windows.py · revin.py · costs.py
   │  (purge+embargo walk-forward · input-window-only RevIN · per-side bps)
   ├────────────────────────────────────────────────────────────────────────
   │   data/synthetic.py · data/loaders.py     foundation (no internal deps)
   │   (seeded multivariate panel ·            _validation · _constants · _typing
   │    yfinance→Stooq + FRED release lags)     _exceptions · _manifest · _rng
   └────────────────────────────────────────────────────────────────────────
```

### Foundation (`_*.py`)

- `_constants.py` — `EPS`, period constants; one source of truth.
- `_validation.py` — input guards (DataFrame coercion, shape, finiteness).
- `_typing.py` / `_exceptions.py` — shared aliases (`FloatArray`,
  `SequenceTensor`) and the exception taxonomy (`MvtsForecastError` base +
  `ValidationError`, `InsufficientDataError`, `ArtifactError`).
- `_manifest.py` / `_rng.py` — `RunManifest` (BLAKE2b config-hash + git SHA) plus
  seeded PCG64 substreams. The same `(seed, basket, …)` reproduces the panel,
  the forecasts, and `metrics.json` byte-for-byte.

### `data/`

`synthetic.py` is the honest-null testbed: a single weak common factor + dominant
idiosyncratic Gaussian noise + mild AR(1), seeded through `_rng`. The conditional
mean sits far below the noise floor, so the next-step return is effectively a
random walk and the naive forecast is the OOS floor *by construction*.
`loaders.py` is the offline real-data path (yfinance → Stooq prices, FRED-CSV
macro lagged to **release** dates, never reference dates).

### `windowing/`

`windows.py` builds the supervised problem and enforces the leakage guards:

- **Sliding windows** — window `i` spans rows `[i, i+look_back)` and predicts the
  target return at `i + look_back + horizon - 1` (strictly *after* the window).
- **No-target-in-features** — the target column's same-step transform is excluded
  from the encoder input (`drop_target_feature=True`); leaving it in is refused.
- **Purge (≥ `look_back`) + embargo** — `make_folds` removes a gap of at least
  `look_back` samples at every train/test boundary so no window straddles a split
  ([ADR-0002](decisions/0002-purge-embargo-walkforward.md)).
- **Train-only standardizer** — `fit_standardizer` fits per-feature mean/std on
  the **train fold only** and `transform` applies (never re-fits) to the test
  fold — the de-leak fix for the classic full-series-scaler bug.

`revin.py` is reversible instance normalization computed from the **input window
only** ([ADR-0001](decisions/0001-revin-input-window-only.md)): per-window mean/std
over the look-back axis, captured so the forecast is de-normalized with the same
statistics. Because no statistic touches a future row or the train fold, RevIN is
causal and leakage-free by construction. `costs.py` is the per-side bps model.

### `models/`

Five models behind a common windowed interface, each returning a frozen result
dataclass:

- `naive.py` — last-value / random walk (`r_hat = 0` in return space), the floor,
  live and torch-free.
- `arima.py` — per-series ARIMA (statsmodels / optional pmdarima), live.
- `lstm.py`, `patchtst.py`, `transformer_vs.py` — the deep models built and
  trained with torch (`[train]`, offline only). PatchTST uses patching +
  channel-independence; `transformer_vs` is a simplified interpretable
  variable-selection transformer whose selection weights sum to 1.
- `onnx_runtime.py` — the **serve** path: loads the committed `artifacts/*.onnx`
  graphs with onnxruntime and runs a forward pass; torch is **never** imported
  ([ADR-0003](decisions/0003-onnx-serve-no-torch.md)). onnxruntime is lazy.

### `evaluation/`

`metrics.py` is **return-space only** (RMSE, MAE, MASE-vs-naive, directional
accuracy with a binomial test, net-of-cost PnL Sharpe with a `shift(1)` tradable
signal) — **never** a price-level R² ([ADR-0005](decisions/0005-no-price-level-r2.md)).
`diebold_mariano.py` tests equal predictive accuracy of a model vs. the naive
random walk on squared-error loss differentials, using a Newey-West Bartlett HAC
long-run variance. `dsr.py` is the Probabilistic / Deflated Sharpe (full
`(k+2)/4` kurtosis term) with the **full configuration grid** as `n_trials`.
`verdict.py` is a **pure function** mapping `(dm_statistic, dm_pvalue,
deflated_sharpe, n_effective_trials)` to a fixed verdict enum
([ADR-0004](decisions/0004-honest-null-vs-naive.md)).

### Delivery (`train.py`, `serve.py`, `cli.py`, `plots.py`)

`train.py` is the offline pipeline (synthetic → walk-forward → train torch models
→ export ONNX with a `1e-4` parity gate → precompute OOS forecasts → write
`metrics.json`); it is the only module that imports torch. `serve.py` exposes
`run_forecast` for the backend (naive + ARIMA live, deep from ONNX, pure verdict).
`cli.py` (Typer, lazy) exposes `train` / `forecast` / `compare`. `plots.py`
(Plotly, lazy) builds the actual-vs-forecast and error-by-model figures.

## Data flow through one walk-forward window

```
input window  ──►  fit_standardizer on TRAIN fold ONLY ──► transform test fold
  (returns)            │
                       ├─► RevIN: per-window mean/std over the look-back axis ONLY
                       │            (de-norm captured for the reversible inverse)
                       ▼
   purge (≥ look_back) + embargo at the train/test boundary — no window straddles
                       │
                       ▼
   per-model forecast of the next-step RETURN (target's same-step transform
   excluded from the encoder input — no-target-in-features)
        ├─ naive:        r_hat = 0                          (live)
        ├─ arima:        per-series statsmodels             (live)
        └─ lstm/patchtst/transformer: committed ONNX forward (serve, no torch)
                       │
                       ▼  signal = sign(forecast).shift(1)   (tradable, no lookahead)
   OOS window  ──►  return-space RMSE/MAE · directional accuracy · net-of-cost PnL
                       │
                       ▼ (aggregate across folds)
     Diebold-Mariano vs. naive (Newey-West HAC) · Deflated Sharpe (full n_trials)
                       │
                       ▼
            verdict.py  ──►  deep_beats_naive (pure-derived bool)
```

The headline comparison is **each deep model vs. the naive random walk**, on
identical windows and RevIN, so any RMSE/Sharpe gap is the model — not a
mismatched harness.

## Key invariants

The compute core guarantees, and tests enforce:

1. **Input-window-only RevIN.** The normalization of window `i` depends on rows
   `[i, i+look_back)` only; perturbing rows at `i+look_back..` cannot change the
   normalized input or the de-normalized forecast at `i` (Hypothesis test).
2. **Train-only scaler.** The standardizer is fitted on the train fold and
   applied (never re-fitted) to the test fold.
3. **No window straddles a split.** The purge gap is ≥ `look_back`, plus an
   embargo after each test block.
4. **No target in features.** The target column's same-step transform is excluded
   from the encoder input (asserted; leaving it in is refused).
5. **No price-level R².** It is never computed, stored, or reported; all skill is
   return-space.
6. **ONNX ↔ torch parity.** Each exported deep model's ONNX forward matches its
   torch forward to `1e-4` — the served graph *is* the trained model.
7. **DSR monotonicity / honesty.** The Deflated Sharpe uses the full
   configuration grid as `n_trials` and is non-increasing in it.
8. **Verdict safety.** `deep_beats_naive` cannot read `True` unless a deep model
   beats naive with a DM-significant margin signed in its favour **and** a
   positive Deflated Sharpe (truth-table unit-tested).
9. **Determinism.** Same `RunManifest` seed → byte-identical panel, forecasts,
   and `metrics.json`.
10. **Import purity.** Importing any `src/mvtsforecast` module triggers no torch,
    no onnxruntime, no I/O, no network, no RNG draw (subprocess-tested).

## Testing strategy

Tests are partitioned by intent under `tests/` (markers in `pyproject.toml`),
with seeded `conftest.py` fixtures (`synthetic_panel`, `random_walk`,
`weak_factor`) giving every layer deterministic, adversarial inputs:

- **`unit/`** — isolated kernels: RevIN math, windowing bounds, the metric
  formulas, the verdict truth table, the model result dataclasses.
- **`property/`** (Hypothesis) — the invariants above: future-perturbation
  invariance (RevIN + windowing), no-target-in-features, naive-is-the-floor on a
  pure random walk.
- **`parity/`** — golden checks: ONNX vs. torch forward at `1e-4` (needs the
  `[train]` extra, marked slow), Deflated Sharpe vs. the reused HRP reference at
  `1e-10`, Diebold-Mariano vs. the closed-form + Newey-West HAC oracle.
- **`regression/`** — the honest null, locked: on a pure random walk the deep
  models do **not** beat naive (`deep_beats_naive = false`); the golden
  `metrics.json` values; the import-purity subprocess test.
- **`integration/`** — end-to-end synthetic → windows → naive + ARIMA live + ONNX
  deep → compare, with **no torch** on the serve path.

Coverage gate `fail_under = 85` (currently ~93%), ruff + strict mypy clean. The
torch / `[train]` tests are marked slow; the serve path, naive, and ARIMA run
torch-free.

## Backend & frontend boundary

The compute core is decoupled from delivery. The backend vendors
`mvts-forecast[serve]` (onnxruntime, **not** torch) under `api/lib/mvts_forecast/`,
including the committed `artifacts/` (the ONNX deep models + `metrics.json`), and
exposes `POST /tools/mvts-forecast/run`. A module-level `_SESSION = None` lazily
loads the ONNX graphs; naive + ARIMA are computed live; the request path
**never** trains. The response returns JSON-safe summary scalars (RMSE,
directional accuracy, DM p-value, Deflated Sharpe, `deep_beats_naive`,
`data_source`) plus two Plotly `{data, layout}` figures. The frontend surfaces
the pure-derived `deep_beats_naive` as a prominent **"Deep beats naive: NO"**
badge — the first thing a visitor reads.
