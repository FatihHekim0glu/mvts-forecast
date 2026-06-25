"""Parity + honesty guards for the Deflated Sharpe Ratio and its ``n_trials``.

The DSR is the overfitting yardstick: it deflates a realized Sharpe by an
expected-maximum benchmark that GROWS with ``n_trials`` (the FULL architecture x
HP configuration grid). These tests pin:

- PSR parity against the standard-normal CDF closed form;
- the DSR ``n_trials`` honesty guard — DSR is NON-INCREASING in ``n_trials`` (a
  larger explored grid can only make a given Sharpe less significant); and
- that under-counting ``n_trials`` (the data-snooping footgun) inflates the DSR,
  i.e. the honest full-grid count yields a STRICTLY LOWER DSR than a naive
  single-trial count.

Reuses :mod:`mvtsforecast.evaluation.dsr` (the copied, cross-checked
implementation) — no re-derivation here, only the contract.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mvtsforecast._exceptions import ValidationError
from mvtsforecast.evaluation.diebold_mariano import diebold_mariano
from mvtsforecast.evaluation.dsr import (
    _norm_cdf,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    variance_of_trial_sharpes,
)
from mvtsforecast.evaluation.metrics import net_pnl_sharpe, rmse
from mvtsforecast.evaluation.verdict import derive_verdict

pytestmark = pytest.mark.parity


def test_psr_matches_normal_cdf_reference() -> None:
    # Gaussian returns (skew 0, full kurtosis 3): the bracket variance is
    # 1 - 0*SR + 0.5*SR^2; PSR = Phi((SR - SR*) sqrt(n-1) / sqrt(variance)).
    sr, n = 0.15, 250
    psr = probabilistic_sharpe_ratio(sr, n_obs=n, skew=0.0, kurtosis=3.0)
    variance = 1.0 - 0.0 * sr + 0.25 * (3.0 - 1.0) * sr * sr
    z = sr * math.sqrt(n - 1) / math.sqrt(variance)
    assert psr == pytest.approx(_norm_cdf(z), rel=1e-12)


def test_psr_at_observed_equals_benchmark_is_half() -> None:
    # If the observed Sharpe equals the benchmark, the PSR is exactly 0.5.
    psr = probabilistic_sharpe_ratio(0.2, n_obs=300, benchmark_sharpe=0.2)
    assert psr == pytest.approx(0.5, abs=1e-12)


def test_dsr_single_trial_reduces_to_psr_vs_zero() -> None:
    # With n_trials == 1 the expected-maximum benchmark collapses to zero, so the
    # DSR equals the plain PSR against a zero benchmark.
    sr, n = 0.18, 252
    dsr = deflated_sharpe_ratio(sr, n_obs=n, n_trials=1, variance_of_trial_sharpes=0.04)
    psr0 = probabilistic_sharpe_ratio(sr, n_obs=n, benchmark_sharpe=0.0)
    assert dsr == pytest.approx(psr0, rel=1e-12)


def test_dsr_is_non_increasing_in_n_trials() -> None:
    # The n_trials honesty guard: deflating against MORE trials cannot raise the
    # DSR. Monotonic non-increase across a realistic grid count.
    sr, n, v = 0.20, 252, 0.05
    prev = deflated_sharpe_ratio(sr, n_obs=n, n_trials=1, variance_of_trial_sharpes=v)
    for nt in (2, 5, 12, 24, 60, 120):
        cur = deflated_sharpe_ratio(sr, n_obs=n, n_trials=nt, variance_of_trial_sharpes=v)
        assert cur <= prev + 1e-12
        prev = cur


def test_under_counting_trials_inflates_dsr() -> None:
    # The data-snooping footgun: claiming a single trial when the honest grid is
    # (architectures x HP configs) OVERSTATES significance. The honest count must
    # yield a strictly lower DSR than the dishonest single-trial count.
    sr, n, v = 0.22, 252, 0.06
    n_architectures = 3  # naive-deep family: lstm, patchtst, transformer
    n_hp_configs = 8  # the per-architecture HP grid actually swept offline
    honest_trials = n_architectures * n_hp_configs  # the FULL config grid
    dishonest = deflated_sharpe_ratio(sr, n_obs=n, n_trials=1, variance_of_trial_sharpes=v)
    honest = deflated_sharpe_ratio(sr, n_obs=n, n_trials=honest_trials, variance_of_trial_sharpes=v)
    assert honest < dishonest


def test_dsr_rejects_invalid_arguments() -> None:
    with pytest.raises(ValidationError):
        deflated_sharpe_ratio(0.1, n_obs=1, n_trials=4, variance_of_trial_sharpes=0.01)
    with pytest.raises(ValidationError):
        deflated_sharpe_ratio(0.1, n_obs=252, n_trials=0, variance_of_trial_sharpes=0.01)
    with pytest.raises(ValidationError):
        deflated_sharpe_ratio(0.1, n_obs=252, n_trials=4, variance_of_trial_sharpes=-0.01)


# --------------------------------------------------------------------------- #
# Honest cross-trial variance ``V`` (the migrated bug fix)                     #
# --------------------------------------------------------------------------- #
# The Deflated-Sharpe benchmark needs the REAL cross-trial variance ``V`` of the
# per-observation Sharpe ratios of the models actually compared — NOT a fabricated
# ``1 / n_obs`` heuristic. These guards pin: (a) ``V`` is the genuine sample
# variance of the trial Sharpes; (b) the single-series fallback; (c) the honest
# NULL does not flip a model to a false edge; and (d) a genuinely-skilled
# forecaster DOES fire the verdict (the positive control).


def test_variance_of_trial_sharpes_is_real_sample_variance() -> None:
    # ``V`` is the ddof=1 sample variance of the per-observation trial Sharpes,
    # NOT ``1 / n_obs`` or any other constant heuristic.
    trial_sharpes = [0.0, -0.05, 0.12, 0.04]
    v = variance_of_trial_sharpes(trial_sharpes)
    assert v == pytest.approx(float(np.var(trial_sharpes, ddof=1)))
    # And it is materially different from the old fabricated 1/n_obs value.
    assert v != pytest.approx(1.0 / 240)


def test_variance_of_trial_sharpes_single_series_fallback_is_zero() -> None:
    # Fewer than two finite trial Sharpes => 0.0 (the documented single-series
    # fallback), which collapses the DSR benchmark to plain PSR-against-zero.
    assert variance_of_trial_sharpes([0.1]) == 0.0
    assert variance_of_trial_sharpes([]) == 0.0
    dsr_v0 = deflated_sharpe_ratio(0.2, n_obs=252, n_trials=3, variance_of_trial_sharpes=0.0)
    psr0 = probabilistic_sharpe_ratio(0.2, n_obs=252, benchmark_sharpe=0.0)
    assert dsr_v0 == pytest.approx(psr0, rel=1e-12)


def test_honest_null_does_not_flip_with_real_v() -> None:
    # RE-VERIFY THE HONEST NULL: near-zero deep forecasts (indistinguishable from
    # naive) score a slightly-negative net Sharpe; with the REAL cross-trial V the
    # Deflated Sharpe stays far below the 0.95 gate, so deep CANNOT beat naive.
    rng = np.random.default_rng(7)
    n = 240
    y = rng.normal(0.0, 0.01, size=n)
    naive = np.zeros(n)
    deep = {m: np.full(n, 1e-7) for m in ("lstm", "patchtst", "transformer")}

    trial_sharpes = [net_pnl_sharpe(y, p) for p in (naive, *deep.values())]
    v_real = variance_of_trial_sharpes(trial_sharpes)
    # The real V differs from the fabricated 1/n_obs heuristic that was removed.
    assert v_real != pytest.approx(1.0 / n)

    best_stat, best_p, best_dsr = 0.0, 1.0, 0.0
    for pred in deep.values():
        dsr = deflated_sharpe_ratio(
            net_pnl_sharpe(y, pred), n_obs=n, n_trials=3, variance_of_trial_sharpes=v_real
        )
        assert dsr < 0.95  # the null model never clears the DSR gate
        stat, p = diebold_mariano(y, pred, naive)
        if stat < best_stat:
            best_stat, best_p, best_dsr = stat, p, dsr

    verdict = derive_verdict("best", best_stat, best_p, best_dsr, n_effective_trials=3)
    assert verdict.deep_beats_naive is False


def test_positive_control_skilled_forecaster_fires_verdict() -> None:
    # POSITIVE CONTROL: a genuinely-skilled forecaster (an accurate return
    # forecast) MUST beat naive on BOTH gates with the REAL cross-trial V —
    # strictly lower squared-error loss (DM significant + negative) AND a Deflated
    # Sharpe that clears 0.95. This proves the verdict still fires on real skill;
    # the V fix tightens the null without disabling true detection.
    rng = np.random.default_rng(0)
    n = 1000
    regime = np.sign(np.sin(np.linspace(0.0, 8.0 * np.pi, n)))  # low-turnover drift
    y = regime * 0.004 + rng.normal(0.0, 0.006, size=n)
    naive = np.zeros(n)
    skilled = y + rng.normal(0.0, 0.002, size=n)  # accurate return forecast

    assert rmse(y, skilled) < rmse(y, naive)  # genuinely lower forecast error

    trial_sharpes = [net_pnl_sharpe(y, naive), net_pnl_sharpe(y, skilled)]
    v_real = variance_of_trial_sharpes(trial_sharpes)
    dsr = deflated_sharpe_ratio(
        net_pnl_sharpe(y, skilled), n_obs=n, n_trials=3, variance_of_trial_sharpes=v_real
    )
    assert dsr >= 0.95  # the real edge clears the DSR confidence gate

    dm_stat, dm_pvalue = diebold_mariano(y, skilled, naive)
    assert dm_stat < 0.0 and dm_pvalue < 0.05  # DM significant + favours the model

    verdict = derive_verdict("skilled", dm_stat, dm_pvalue, dsr, n_effective_trials=3)
    assert verdict.deep_beats_naive is True
