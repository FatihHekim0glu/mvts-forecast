# ADR-0004: The honest NULL — a pure `deep_beats_naive` verdict

- **Status:** Accepted
- **Date:** 2026-06-18
- **Deciders:** mvts-forecast maintainers
- **Related:** [ADR-0005](0005-no-price-level-r2.md) (return space), [DESIGN.md](../DESIGN.md) (evaluation layer)

## Context

The genre this library answers — "predict the stock with a transformer" — is
dominated by results that quietly over-claim: a leaked scaler, a price-level R², a
single lucky seed, or a Sharpe ratio that was selected from dozens of unreported
configurations. The reader is told the deep model "works."

On noisy **daily returns** the literature-consistent, defensible finding is the
opposite: a PatchTST-style encoder, an interpretable transformer, and an LSTM do
**not** reliably beat a naive random walk out-of-sample on directional accuracy or
risk-adjusted PnL after costs. The conditional mean is tiny relative to the noise;
any apparent edge is marginal and disappears under multiplicity correction.

The danger is that even an honest pipeline can be *narrated* into a win — a point
estimate that happens to favour the model, reported without its significance or
its selection cost. We need the verdict to be a **consequence of the evidence**,
not a sentence someone wrote.

## Decision

**`deep_beats_naive` is a PURE FUNCTION of the inference outputs, gated on three
conditions that ALL must hold.** `evaluation/verdict.py::derive_verdict` returns
`DEEP_BEATS_NAIVE` iff, for the best deep model (lowest RMSE):

1. the **Diebold-Mariano** test vs. naive is significant (`dm_pvalue < alpha`,
   default 0.05), using a Newey-West HAC long-run variance; **and**
2. the DM statistic is **signed in the model's favour** (`dm_statistic < 0` —
   strictly lower squared-error loss than the naive forecast); **and**
3. the **Deflated Sharpe** clears the 1-alpha confidence threshold
   (`deflated_sharpe >= 0.95 (1 - alpha)`) against the multiplicity-inflated
   benchmark, where `n_trials` is the **full configuration grid** (architectures ×
   HP configs), not 1. The DSR is a probability in `[0, 1]` (a CDF), so gating at
   `> 0` would be vacuous; the portfolio standard gates at `1 - alpha = 0.95`.

If **any** gate fails, the verdict is `NO_SIGNIFICANT_DIFFERENCE` and
`deep_beats_naive = false`, regardless of any favourable point estimate. The
function raises on out-of-range inputs and its truth table is unit-tested. There
is **no narrative override** anywhere in the code, the API, or the frontend — the
"Deep beats naive: NO" badge renders whatever the pure function returns.

The shipped synthetic result (`seed=7`, `n_effective_trials=3`) exercises this:
the deep models' DM tests are extremely significant but signed **against** them
(naive has the lower RMSE), directional accuracy is a coin flip, and the Deflated
Sharpes collapse toward zero — so `deep_beats_naive = false`. A regression test
locks the same outcome on a pure random walk.

## Consequences

- **Positive.** The headline cannot over-claim: a win requires a significant,
  correctly-signed DM margin *and* a multiplicity-corrected Sharpe that clears
  the 1-alpha confidence threshold (DSR ≥ 0.95).
- **Positive.** The verdict is reproducible and auditable — same evidence, same
  bool — and the honest NULL is the *measured* result, not an assertion.
- **Cost.** The project's headline is deliberately modest
  (`deep_beats_naive = false`). That honesty is the deliverable, and it corrects
  the over-claiming outlier in the genre.
- **Risk addressed.** "A favourable point estimate gets narrated into a win
  without significance or multiplicity correction" is structurally excluded.
