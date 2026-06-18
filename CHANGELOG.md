# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-06-18

### Added

- Initial package scaffold (src-layout, import name `mvtsforecast`, import-pure
  with `py.typed`). `import mvtsforecast` imports NO torch / onnxruntime /
  statsmodels / plotly at module load.
- Core infra reused from `hrp-portfolio` (renamed `hrp` → `mvtsforecast`):
  `_constants`, `_typing`, `_exceptions` (`MvtsForecastError` base +
  `ArtifactError`), `_validation`, `_manifest` (`RunManifest` with BLAKE2b
  config-hash), and `_rng` (seeded PCG64 generator + substream spawning).
- Reused honest-statistics layer: `evaluation/dsr.py` (PSR/DSR with the full
  kurtosis term + true `n_trials`) and `windowing/costs.py` (`FixedBpsCost`).
- Implemented now (load-bearing for the honest NULL): the seeded synthetic
  multivariate RETURNS generator (`data/synthetic.py::synthetic_panel` — a weak
  common factor + dominant idiosyncratic noise + mild AR(1)) and the naive
  `persistence_returns` floor.
- Typed stub signatures with full contracts for the new modules: `data/loaders`
  (yfinance→Stooq prices + FRED-CSV release-date-lagged macro), `windowing/`
  (`windows` purge-aware sliding windows + walk-forward folds + train-only
  scaler; `revin` input-window-only instance norm), `models/`
  (`naive`, `arima`, `lstm`, `patchtst`, `transformer_vs`, `onnx_runtime`),
  `evaluation/` (`metrics`, `diebold_mariano`, `verdict`), `train`, `serve`,
  `plots`, and `cli`. Each model exposes a frozen result/config dataclass with a
  JSON-safe `to_dict`.
- Curated top-level `__init__.py` re-exporting the public API with NO torch /
  onnxruntime imported at module load.
- Adapted `pyproject.toml` extras: `[data]` (numpy/pandas/scipy/statsmodels/
  pyarrow/diskcache), `[serve]` (onnxruntime — NEVER torch), `[train]`
  (torch/onnx, offline only), `[viz]` (plotly/kaleido), `[dev]`
  (pytest/pytest-cov/hypothesis/ruff/mypy + the lean serve stack). No `[all]`.
- CI (`ci.yml` lean extras + matrix py3.11–3.13; `no-ai-attribution.yml` guard),
  `CHANGELOG`/`CONTRIBUTING`/`LICENSE` (MIT)/`CITATION.cff`/`README` stub.
- Seeded test fixtures (`synthetic_panel`, `random_walk`, `weak_factor`) and the
  partitioned `tests/` tree (unit/parity/property/regression/integration).

[Unreleased]: https://github.com/FatihHekim0glu/mvts-forecast/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/FatihHekim0glu/mvts-forecast/releases/tag/v0.1.0
