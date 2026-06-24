# Contributing

Thanks for your interest in `mvts-forecast`. This project uses
[uv](https://docs.astral.sh/uv/) for environment and dependency management.

## Dev setup

```bash
# 1. Install uv (https://docs.astral.sh/uv/getting-started/installation/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create the env and install the project with the lean extras + dev tooling.
#    NOTE: the [train] extra (torch + onnx) is heavy and is NOT needed to run the
#    test suite — install it only when retraining/exporting the ONNX models.
uv venv
uv pip install -e ".[data,serve,viz,dev]"
```

Prefix commands with `uv run` to use the env without activating it.

## Quality gates

These are exactly what CI runs (see `.github/workflows/ci.yml`). Run them locally
before opening a pull request:

```bash
uv run ruff check src tests                                              # lint
uv run mypy src                                                          # types (strict)
uv run pytest -q -m "not slow" --cov=mvtsforecast --cov-report=term --cov-fail-under=85
```

- **Lint** (`ruff`) must pass.
- **Types** (`mypy --strict`) is run on every PR. It is currently non-blocking in
  CI while residual strict-mode issues are burned down, but new code should not
  add type errors.
- **Tests** (`pytest`) must pass with **coverage ≥ 85%** (the gate also lives in
  `[tool.coverage.report] fail_under` in `pyproject.toml`). The torch
  train/export path is marked `slow` and excluded from the default run; it must
  never require a GPU.

CI runs the full matrix on Python 3.11, 3.12, and 3.13.

## The point of this project

This is a leakage-free, honest-NULL **multivariate-transformer benchmark**. When
contributing, preserve the non-negotiables:

- The target is the next-step **RETURN**, never the price level.
- **No price-level R²** as a metric, anywhere.
- RevIN / instance-norm is computed from the **input window only**; any
  `StandardScaler` is fitted on the **train fold only**, per fold.
- Walk-forward uses **purge (≥ `look_back`) + embargo** at every boundary, and the
  encoder input never contains the target's same-step transform.
- `deep_beats_naive` is a **pure function** of the inference and must read `False`
  unless a deep model beats naive with a **Diebold-Mariano-significant** margin
  **and** a **Deflated Sharpe ≥ 0.95 (1 − alpha)**.
- `src/mvtsforecast/` is **import-pure**: no torch / onnxruntime / statsmodels /
  plotly / network / training at import time (heavy imports live behind functions
  or `__main__`). Training is **offline**; the serve container runs onnxruntime
  only — **never torch**.

## Commit hygiene

- Use clear, present-tense commit messages.
- **Do not** add AI-attribution trailers — no `Co-Authored-By: Claude`,
  no "Generated with Claude", no robot-emoji attribution lines. The
  `.github/workflows/no-ai-attribution.yml` guard fails any PR that contains them.

## Pull requests

- Branch off `main`; keep PRs focused.
- Make sure the three quality gates above are green locally.
- Update `CHANGELOG.md` (under `[Unreleased]`) when behaviour changes.
