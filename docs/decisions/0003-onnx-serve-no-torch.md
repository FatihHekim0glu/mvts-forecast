# ADR-0003: Serve via ONNX / onnxruntime — torch is train-only

- **Status:** Accepted
- **Date:** 2026-06-18
- **Deciders:** mvts-forecast maintainers
- **Related:** [DESIGN.md](../DESIGN.md) (layering & import purity)

## Context

The deep models — a small LSTM, a PatchTST-style encoder, and a simplified
interpretable variable-selection transformer — are built and trained with
**torch**. But the hosted tool runs in a small, shared API container alongside the
other portfolio tools. torch is a heavy dependency (hundreds of MB, slow cold
start, large memory footprint) and importing it at request time — or at package
import time — would be unacceptable for a lean inference service. The package must
also be **import-pure**: `import mvtsforecast` must not pull in torch, an
inference engine, or any I/O. (The leaky-stock-predictor footgun is a package that
spins up a model — or a scheduler, or a `while True` — at import time.)

We need to train with torch but serve without it.

## Decision

**Training and serving use different engines, split across optional extras:**

- **Train (`[train]` extra, offline, heavy):** build and fit the models with
  torch, then export each trained graph to a tiny ONNX artifact and commit it
  inside the package (`src/mvtsforecast/artifacts/{lstm,patchtst,transformer_vs}.onnx`,
  each well under 10 MB).
- **Serve (`[serve]` extra, container, lean):** load and run those committed ONNX
  artifacts with **onnxruntime** (numpy + onnxruntime only) via
  `models/onnx_runtime.py::OnnxForecaster`. The container **never imports torch.**
- The **naive and ARIMA baselines are torch-free** (pure numpy / statsmodels) and
  run **live** in the request path; only the deep models come from ONNX.

A **parity gate** in `train.py::export_onnx` runs the same `(N, look_back,
n_features) → (N, 1)` window through the trained torch module and the exported
ONNX graph and asserts they agree to **`1e-4`** (rtol/atol), so the thing the
container serves is exactly the model the `[train]` path produced. The
`tests/parity/test_onnx_torch_parity.py` test re-certifies this for every deep
model (marked slow; needs `[train]`).

Import purity is enforced: onnxruntime is imported **lazily** inside the model
layer on first call, torch is never reachable from a plain import, and a
subprocess test verifies no import-time side effects.

## Consequences

- **Positive.** The serve container is tiny and fast; no torch at all, instant
  cold start, small memory footprint.
- **Positive.** The committed artifacts are reproducible from the same seed and
  certified equivalent to torch (`1e-4` parity), so serve = train.
- **Positive.** `import mvtsforecast` stays side-effect-free and vendorable into
  the backend.
- **Cost.** An ONNX export step and a torch↔ONNX parity test that needs the
  `[train]` extra (marked slow, skipped in the lean CI run).
- **Risk addressed.** "torch leaks into the inference container / the package
  trains or loads a model at import time" — both are structurally prevented.
