# ADR-0005: Never report a price-level R² (the debunked trap)

- **Status:** Accepted
- **Date:** 2026-06-18
- **Deciders:** mvts-forecast maintainers
- **Related:** [ADR-0004](0004-honest-null-vs-naive.md) (honest verdict), [DESIGN.md](../DESIGN.md) (evaluation layer)

## Context

The single most misleading number in the "deep stock predictor" genre is the
**price-level R²**: the coefficient of determination of predicted vs. actual
*price levels*. It is routinely reported as 0.95+ and presented as proof the model
"works."

It is a trap. A price series has a **unit root** — it is integrated of order one —
so it is dominated by its own lag. Predicting `P_{t+1} ≈ P_t` (i.e. doing
*nothing*, the naive random walk) already explains almost all of the level's
variance, because that variance *is* the trend. A high level R² therefore
certifies that the series trends, not that the model forecasts. Pair it with a
leaked scaler or window-overlapping split and you get a beautiful, completely
hollow result.

## Decision

**The price-level R² is banned.** It is not computed, not stored, and not reported
anywhere in the codebase, the API response, the frontend, or the docs. All skill
is judged in **return space** (`evaluation/metrics.py`):

- return-space RMSE / MAE;
- **MASE vs. naive** (`≥ 1` ⇒ no improvement over the random walk);
- directional accuracy with a two-sided binomial test vs. 0.5;
- net-of-cost PnL Sharpe with a `shift(1)` tradable signal;
- the **Diebold-Mariano** (1995) test vs. naive, with a Newey-West HAC long-run
  variance, feeding the pure verdict ([ADR-0004](0004-honest-null-vs-naive.md)).

The target is always the next-step **return**, never the price level. This ADR
exists so the trap is documented **once, explicitly**, as a debunked metric — and
so the *absence* of a level R² is a deliberate, defensible choice rather than an
oversight a reviewer might mistake for a gap.

## Consequences

- **Positive.** The headline metrics cannot be inflated by the unit-root artifact;
  every reported number reflects actual return-space accuracy.
- **Positive.** Readers who expect the familiar "R² = 0.97" chart get an explicit
  explanation of why it is meaningless instead.
- **Cost.** The project's headline numbers look modest (RMSE comparable to naive,
  directional accuracy ≈ 0.50) next to a level-R² showcase. That honesty is the
  deliverable.
- **Risk addressed.** "Reporting a unit-root-inflated R² as forecasting skill" —
  the defining mistake of the genre — is structurally excluded.
