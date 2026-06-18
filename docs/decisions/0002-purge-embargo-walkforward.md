# ADR-0002: Purge (≥ look_back) + embargo in the walk-forward split

- **Status:** Accepted
- **Date:** 2026-06-18
- **Deciders:** mvts-forecast maintainers
- **Related:** [DESIGN.md](../DESIGN.md) (windowing layer), [ADR-0001](0001-revin-input-window-only.md) (RevIN)

## Context

The supervised problem is built from sliding windows: sample `i` is the
`(look_back, n_features)` tensor over rows `[i, i + look_back)` and its label is
the target return at `i + look_back + horizon - 1`. Adjacent windows **overlap by
`look_back - 1` rows**. A naive train/test split therefore leaks: a window that
ends just before the split boundary shares rows with windows that begin just
after it, so a training observation and a test observation can be built from
overlapping data. The model effectively trains on part of its test set, and the
OOS metric is optimistically biased.

A plain `train_test_split` or a vanilla k-fold ignores this entirely. We need a
split that respects the window length and the forecast horizon.

## Decision

**The walk-forward split purges a gap of at least `look_back` samples at every
train/test boundary, plus an embargo after each test block.**

`windows.py::make_folds` builds anchored/expanding folds where, between the train
slice ending at `train_end` and the test slice starting at `test_start`, there is
a purge gap:

- **Purge = `look_back`.** No `look_back`-length window can straddle the boundary,
  so no test window shares a single row with any train window.
- **Embargo (default 5).** An extra gap after each test block before the next
  fold, guarding against any residual short-range autocorrelation bleeding across
  the boundary.
- **Anchored or rolling.** With `anchored=True` the train slice starts at sample 0
  and expands; otherwise it rolls forward by a fixed window.

When a forecast drives a tradable signal it is additionally `shift(1)`-ed
(`sign(forecast).shift(1)`) so the position is taken strictly after the forecast
is known. Returns use `pct_change(fill_method=None)` to avoid forward-filling
across gaps.

The purge size is **derived from `look_back`**, not cargo-culted from a sibling
repo's event-label config — the leakage surface here is window overlap, and the
purge is sized to exactly close it.

## Consequences

- **Positive.** The OOS metric is honest: no test window is contaminated by
  overlapping training rows.
- **Positive.** The purge scales automatically with `look_back`, so changing the
  window length cannot silently reintroduce the leak.
- **Cost.** Each fold sacrifices `look_back` (+ embargo) samples of usable data at
  the boundary. On the synthetic panel this is affordable; on short real series it
  must be budgeted (`make_folds` raises `InsufficientDataError` rather than
  shrinking the purge).
- **Risk addressed.** "Overlapping windows straddle the split / a vanilla k-fold
  leaks across the boundary" is structurally excluded and unit-tested on the fold
  bounds.
