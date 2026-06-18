# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-06-18

Initial release: a leakage-free multivariate-transformer forecast benchmark with
a measured, honest NULL. On the synthetic panel (`seed=7`, `horizon=1`,
`lookback=60`, `n_effective_trials=3`) the best model is `naive` and
`deep_beats_naive = false`.

### Added

- **Package** — src-layout, import name `mvtsforecast`, import-pure with
  `py.typed`. `import mvtsforecast` imports NO torch / onnxruntime / statsmodels /
  plotly at module load (subprocess-tested).
- **Foundation** reused from `hrp-portfolio` (renamed `hrp` → `mvtsforecast`):
  `_constants`, `_typing`, `_exceptions` (`MvtsForecastError` base +
  `ValidationError` / `InsufficientDataError` / `ArtifactError`), `_validation`,
  `_manifest` (`RunManifest` with BLAKE2b config-hash + git SHA), `_rng` (seeded
  PCG64 generator + substream spawning).
- **Data** — the seeded synthetic multivariate RETURNS generator
  (`data/synthetic.py` — a weak common factor + dominant idiosyncratic noise +
  mild AR(1), the honest-null DGP by construction) and the offline real-data
  loader (`data/loaders.py` — yfinance→Stooq prices + FRED-CSV macro lagged to
  RELEASE dates).
- **Windowing** — `windows.py` purge-aware (≥ `look_back`) sliding windows +
  embargoed anchored/expanding walk-forward folds + train-only `Standardizer` +
  the no-target-in-features guard; `revin.py` input-window-only reversible
  instance normalization; `costs.py` `FixedBpsCost`.
- **Models** — `naive` (random-walk floor, live) and `arima` (per-series, live,
  torch-free); `lstm`, `patchtst`, and `transformer_vs` (torch, `[train]`,
  offline); `onnx_runtime` (serve via onnxruntime, NEVER torch). Each exposes a
  frozen result/config dataclass with a JSON-safe `to_dict`.
- **Evaluation** — `metrics.py` (return-space RMSE/MAE, MASE-vs-naive, directional
  accuracy + binomial, net-of-cost PnL Sharpe; NO price-level R²),
  `diebold_mariano.py` (DM with Newey-West HAC), reused `dsr.py` (PSR/DSR with the
  full `(k+2)/4` kurtosis term + full-grid `n_trials`), and `verdict.py` (the PURE
  `deep_beats_naive` deriver, truth-table tested).
- **Offline training + artifacts** — `train.py` (synthetic → walk-forward → train
  torch → export ONNX with a `1e-4` parity gate → precompute OOS forecasts →
  `metrics.json`); committed `artifacts/{lstm,patchtst,transformer_vs}.onnx`
  (each < 10 MB) + `metrics.json`. `serve.py::run_forecast` for the backend;
  `cli.py` (Typer, lazy) `train`/`forecast`/`compare`; `plots.py` (Plotly, lazy).
- **Tooling** — `pyproject.toml` extras `[data]` / `[serve]` (onnxruntime, never
  torch) / `[train]` / `[viz]` / `[dev]` (no `[all]`); `ci.yml` (lean extras,
  matrix py3.11–3.13) + `no-ai-attribution.yml` guard;
  `CHANGELOG`/`CONTRIBUTING`/`LICENSE` (MIT)/`CITATION.cff`.
- **Tests** — partitioned `tests/` (unit/parity/property/regression/integration)
  with seeded fixtures (`synthetic_panel`, `random_walk`, `weak_factor`):
  ONNX↔torch `1e-4` parity, DSR `1e-10` parity, DM correctness, RevIN
  future-perturbation invariance, no-target-in-features, and the random-walk
  anti-leakage lock (`deep_beats_naive=false`). ruff + strict mypy clean,
  coverage `fail_under=85` (~93%).
- **Docs** — `README` with the measured honest-NULL results table + validation +
  reproduce + limitations + references; `docs/DESIGN.md`; ADRs
  `0001`–`0005` (RevIN input-window-only, purge/embargo walk-forward, ONNX serve
  without torch, the pure `deep_beats_naive` verdict, no price-level R²).

[Unreleased]: https://github.com/FatihHekim0glu/mvts-forecast/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/FatihHekim0glu/mvts-forecast/releases/tag/v0.1.0
