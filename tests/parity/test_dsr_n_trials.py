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

import pytest

from mvtsforecast._exceptions import ValidationError
from mvtsforecast.evaluation.dsr import (
    _norm_cdf,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)

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
