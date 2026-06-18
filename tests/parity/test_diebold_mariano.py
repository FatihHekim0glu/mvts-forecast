"""Parity / correctness tests for the Diebold-Mariano test and its HAC kernel.

The DM statistic ``d_bar / HAC_SE(d)`` is validated against:

- an INDEPENDENT closed-form reference for the loss differential mean and a
  hand-rolled Newey-West Bartlett long-run variance (the same algebra, computed
  separately), and
- statsmodels' ``cov_hac`` sandwich when available, so the HAC long-run variance
  matches the canonical econometrics implementation.

The p-value is checked against the two-sided standard-normal tail. The sign
convention (NEGATIVE statistic favours the model) and the
``dm_favours_model`` gate are pinned.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mvtsforecast._exceptions import ValidationError
from mvtsforecast.evaluation.diebold_mariano import (
    _norm_sf,
    diebold_mariano,
    dm_favours_model,
)
from mvtsforecast.evaluation.metrics import andrews_lag, hac_standard_error

pytestmark = pytest.mark.parity


def _reference_dm(
    y_true: np.ndarray,
    y_pred_model: np.ndarray,
    y_pred_naive: np.ndarray | None = None,
    lag: int | None = None,
) -> tuple[float, float]:
    """A fully independent DM reference (squared-error loss, NW-Bartlett HAC)."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred_model, dtype=np.float64)
    naive = np.zeros_like(yt) if y_pred_naive is None else np.asarray(y_pred_naive, np.float64)
    d = (yt - yp) ** 2 - (yt - naive) ** 2
    t = d.size
    if lag is None:
        lag = math.ceil(4.0 * (t / 100.0) ** (2.0 / 9.0))
    c = d - d.mean()
    gamma0 = float(c @ c / t)
    omega = gamma0
    for h in range(1, min(lag, t - 1) + 1):
        w = 1.0 - h / (lag + 1.0)
        omega += 2.0 * w * float(c[h:] @ c[:-h] / t)
    omega = max(omega, 0.0)
    se = math.sqrt(omega / t)
    stat = d.mean() / se
    p = 2.0 * (0.5 * math.erfc(abs(stat) / math.sqrt(2.0)))
    return float(stat), float(min(1.0, p))


def test_dm_matches_independent_reference() -> None:
    rng = np.random.default_rng(42)
    yt = rng.standard_normal(250) * 0.01
    yp = yt * 0.3 + rng.standard_normal(250) * 0.01  # a partially-informative model
    stat, p = diebold_mariano(yt, yp)
    ref_stat, ref_p = _reference_dm(yt, yp)
    assert stat == pytest.approx(ref_stat, rel=1e-10, abs=1e-12)
    assert p == pytest.approx(ref_p, rel=1e-10, abs=1e-12)


def test_dm_uses_andrews_lag_by_default() -> None:
    rng = np.random.default_rng(1)
    yt = rng.standard_normal(180) * 0.01
    yp = rng.standard_normal(180) * 0.01
    stat_default, _ = diebold_mariano(yt, yp)
    stat_explicit, _ = diebold_mariano(yt, yp, lag=andrews_lag(180))
    assert stat_default == pytest.approx(stat_explicit, rel=1e-12)


def test_dm_pvalue_is_two_sided_normal_tail() -> None:
    rng = np.random.default_rng(9)
    yt = rng.standard_normal(300) * 0.01
    yp = yt * 0.5 + rng.standard_normal(300) * 0.005
    stat, p = diebold_mariano(yt, yp)
    assert p == pytest.approx(2.0 * _norm_sf(abs(stat)), rel=1e-12)


def test_dm_hac_matches_statsmodels_cov_hac() -> None:
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(123)
    # Build a loss-differential-like autocorrelated series and compare the HAC
    # long-run variance of the mean to statsmodels' Newey-West sandwich on a
    # constant-only OLS (whose coefficient SE IS the HAC SE of the mean).
    e = rng.standard_normal(400)
    d = np.empty(400)
    d[0] = e[0]
    for i in range(1, 400):
        d[i] = 0.4 * d[i - 1] + e[i]
    t = d.size
    lag = andrews_lag(t)
    ours = hac_standard_error(d, lag=lag)

    x = np.ones((t, 1))
    # ``use_correction=False`` turns off statsmodels' optional small-sample
    # T/(T-k) scaling so its sandwich uses the same 1/T weighting our kernel
    # does; the two HAC long-run variances then agree to machine precision.
    model = sm.OLS(d, x).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": lag, "kernel": "bartlett", "use_correction": False},
    )
    sm_se = float(np.sqrt(model.cov_params()[0, 0]))
    assert ours == pytest.approx(sm_se, rel=1e-9)


def test_dm_negative_statistic_favours_model() -> None:
    # A model that is uniformly better than naive => lower squared error =>
    # negative loss differential mean => negative DM statistic.
    rng = np.random.default_rng(7)
    yt = rng.standard_normal(400) * 0.01
    yp = yt * 0.6  # shrinks toward the truth, reliably lower error than zeros
    stat, p = diebold_mariano(yt, yp)
    assert stat < 0.0
    assert dm_favours_model(stat, p, alpha=0.05) is True


def test_dm_identical_forecasts_returns_neutral() -> None:
    # model == naive (both zeros) => zero loss differential => neutral (0, 1).
    yt = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
    stat, p = diebold_mariano(yt, np.zeros_like(yt), np.zeros_like(yt))
    assert stat == 0.0
    assert p == 1.0


def test_dm_raises_on_zero_variance_nonzero_mean() -> None:
    # A constant target with a constant forecast makes the squared-error loss
    # differential a NON-ZERO CONSTANT: dispersion is zero but the mean is not,
    # so the asymptotic DM statistic is undefined and must raise. (The scale-
    # aware degeneracy guard catches this even though the raw HAC SE is a tiny
    # ~1e-20 float residue rather than an exact zero.)
    yt = np.full(50, 0.02)
    yp = np.full(50, 0.05)  # loss_model = (0.02-0.05)^2 = 9e-4 vs naive 4e-4 -> const diff
    with pytest.raises(ValidationError):
        diebold_mariano(yt, yp)


def test_dm_neutral_on_identical_constant_forecasts() -> None:
    # model == naive on a constant target: the loss differential is identically
    # zero, so the test is neutral (0, 1) rather than an error.
    yt = np.full(40, 0.02)
    stat, p = diebold_mariano(yt, np.zeros_like(yt), np.zeros_like(yt))
    assert stat == 0.0
    assert p == 1.0


def test_dm_requires_two_observations() -> None:
    with pytest.raises(ValidationError):
        diebold_mariano(np.array([0.01]), np.array([0.0]))


def test_dm_rejects_length_mismatch() -> None:
    with pytest.raises(ValidationError):
        diebold_mariano(np.array([0.01, 0.02]), np.array([0.0]))


@pytest.mark.parametrize(
    ("stat", "p", "expected"),
    [
        (-3.0, 0.001, True),  # significant AND negative -> favours model
        (3.0, 0.001, False),  # significant but POSITIVE -> against model
        (-3.0, 0.20, False),  # negative but INSIGNIFICANT
        (0.0, 1.0, False),  # neutral
    ],
)
def test_dm_favours_model_truth_table(stat: float, p: float, expected: bool) -> None:
    assert dm_favours_model(stat, p, alpha=0.05) is expected
