# ADR-0001: RevIN / instance-norm from the input window only

- **Status:** Accepted
- **Date:** 2026-06-18
- **Deciders:** mvts-forecast maintainers
- **Related:** [DESIGN.md](../DESIGN.md) (windowing layer), [ADR-0002](0002-purge-embargo-walkforward.md) (purge/embargo)

## Context

Deep time-series forecasters are sensitive to distribution shift: the level and
scale of a returns window drift over time, and a model trained on one regime
generalizes poorly to another. Reversible Instance Normalization (RevIN; Kim et
al., 2022) addresses this by normalizing each input and de-normalizing the
forecast — and it is, alongside the scaler, the single most common **leakage
surface** in stock-prediction repos.

The leak happens when the normalization statistics are computed over more than the
input window: a full-series mean/std, or statistics that include the target row or
any future row, lets the model "see" information it would not have at inference
time. The result is an inflated, irreproducible OOS metric. This is precisely the
kind of subtle leak that produces the deceptive numbers this library exists to
debunk.

## Decision

**All instance normalization is computed from the input look-back window ONLY.**

For window `i` spanning rows `[i, i + look_back)`, `revin_normalize` computes the
per-feature mean and (EPS-floored) std over the **time axis of that window only**
and stores them in a frozen `RevInStats`. `revin_denormalize` inverts the
forecast with those same per-window statistics. No statistic depends on:

- any row at or after `i + look_back` (the target or the future), or
- the train fold, or
- the full series.

Separately, the per-feature `Standardizer` is fitted on the **train fold only**
and applied (never re-fitted) to the test fold (`windows.py::fit_standardizer`).
RevIN is per-window and causal; the standardizer is per-fold and train-only — two
independent guards, neither of which can read the future.

The leakage-freedom is **mechanically tested**, not asserted in prose: a
Hypothesis future-perturbation-invariance test alters rows at `i + look_back ..`
and asserts the normalized input and the de-normalized forecast at `i` are
unchanged.

## Consequences

- **Positive.** RevIN cannot leak by construction; the property test makes the
  guarantee executable and regression-proof.
- **Positive.** Per-window normalization also delivers RevIN's intended benefit
  (robustness to distribution shift) honestly.
- **Cost.** Statistics are recomputed per window rather than once globally — a
  negligible cost at this scale, and the correct trade for a leakage-free harness.
- **Risk addressed.** "Normalization statistics leak the target / the future / the
  full series" — the defining subtle bug of leaky forecasting repos — is
  structurally excluded and tested.
