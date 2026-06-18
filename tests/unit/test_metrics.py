"""Unit tests for return-space forecast metrics (``evaluation/metrics.py``).

Covers RMSE/MAE correctness, MASE-vs-naive semantics, directional accuracy +
binomial p-value, the cost-aware net-PnL Sharpe (tradable, shift-by-one signal),
the Andrews lag rule, the HAC standard error, and the ``forecast_metrics``
bundle. Validation-failure paths are exercised so the honest, leakage-safe
boundary is locked in.

A standalone assertion confirms NO price-level R² is exposed by the module.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mvtsforecast._exceptions import ValidationError
from mvtsforecast.evaluation import metrics as metrics_mod
from mvtsforecast.evaluation.metrics import (
    ForecastMetrics,
    andrews_lag,
    directional_accuracy,
    forecast_metrics,
    hac_standard_error,
    mae,
    mase_vs_naive,
    net_pnl_sharpe,
    rmse,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# rmse / mae                                                                   #
# --------------------------------------------------------------------------- #
def test_rmse_known_value() -> None:
    yt = np.array([1.0, 2.0, 3.0])
    yp = np.array([1.0, 2.0, 0.0])  # one error of 3
    # sqrt(mean([0, 0, 9])) = sqrt(3)
    assert rmse(yt, yp) == pytest.approx(math.sqrt(3.0))


def test_mae_known_value() -> None:
    yt = np.array([1.0, 2.0, 3.0])
    yp = np.array([0.0, 2.0, 1.0])  # errors 1, 0, 2
    assert mae(yt, yp) == pytest.approx(1.0)


def test_rmse_zero_on_perfect_forecast() -> None:
    yt = np.array([0.1, -0.2, 0.05])
    assert rmse(yt, yt) == 0.0
    assert mae(yt, yt) == 0.0


def test_rmse_matches_numpy_reference() -> None:
    rng = np.random.default_rng(0)
    yt = rng.standard_normal(200)
    yp = rng.standard_normal(200)
    ref = float(np.sqrt(np.mean((yt - yp) ** 2)))
    assert rmse(yt, yp) == pytest.approx(ref, rel=1e-12)


@pytest.mark.parametrize("bad", [rmse, mae])
def test_error_funcs_reject_empty(bad) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValidationError):
        bad(np.array([]), np.array([]))


@pytest.mark.parametrize("bad", [rmse, mae])
def test_error_funcs_reject_length_mismatch(bad) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValidationError):
        bad(np.array([1.0, 2.0]), np.array([1.0]))


@pytest.mark.parametrize("bad", [rmse, mae])
def test_error_funcs_reject_nonfinite_true(bad) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValidationError):
        bad(np.array([1.0, np.nan]), np.array([1.0, 2.0]))


@pytest.mark.parametrize("bad", [rmse, mae])
def test_error_funcs_reject_nonfinite_pred(bad) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValidationError):
        bad(np.array([1.0, 2.0]), np.array([1.0, np.inf]))


def test_mase_rejects_nonfinite_naive_vector() -> None:
    yt = np.array([0.01, -0.02, 0.03])
    with pytest.raises(ValidationError):
        mase_vs_naive(yt, yt, np.array([0.0, np.nan, 0.0]))


# --------------------------------------------------------------------------- #
# mase_vs_naive                                                               #
# --------------------------------------------------------------------------- #
def test_mase_equals_one_when_model_is_naive() -> None:
    yt = np.array([0.01, -0.02, 0.03, -0.04])
    # The default naive forecast is zeros; a zero-forecast model == naive.
    assert mase_vs_naive(yt, np.zeros_like(yt)) == pytest.approx(1.0)


def test_mase_below_one_when_model_beats_naive() -> None:
    yt = np.array([0.01, -0.02, 0.03, -0.04])
    # A model that halves every error has MAE = 0.5 * naive MAE.
    yp = yt - 0.5 * yt  # forecast = 0.5 * truth, error = 0.5 * truth
    assert mase_vs_naive(yt, yp) < 1.0


def test_mase_explicit_naive_vector() -> None:
    yt = np.array([1.0, 1.0, 1.0])
    yp = np.array([0.5, 0.5, 0.5])  # MAE 0.5
    naive = np.array([0.0, 0.0, 0.0])  # MAE 1.0
    assert mase_vs_naive(yt, yp, naive) == pytest.approx(0.5)


def test_mase_raises_on_degenerate_naive() -> None:
    yt = np.array([0.0, 0.0, 0.0])  # naive (zeros) MAE == 0 -> undefined
    with pytest.raises(ValidationError):
        mase_vs_naive(yt, np.array([0.1, 0.2, 0.3]))


def test_mase_rejects_wrong_length_naive() -> None:
    yt = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValidationError):
        mase_vs_naive(yt, yt, np.array([0.0, 0.0]))


# --------------------------------------------------------------------------- #
# directional_accuracy + binomial                                            #
# --------------------------------------------------------------------------- #
def test_directional_accuracy_perfect() -> None:
    yt = np.array([0.01, -0.02, 0.03, -0.04])
    acc, p = directional_accuracy(yt, yt)
    assert acc == 1.0
    assert 0.0 <= p <= 1.0


def test_directional_accuracy_half() -> None:
    yt = np.array([0.01, -0.02, 0.03, -0.04])
    yp = np.array([0.01, 0.02, -0.03, -0.04])  # 2 of 4 correct
    acc, p = directional_accuracy(yt, yp)
    assert acc == pytest.approx(0.5)
    # Exactly at the null -> the two-sided p-value is 1.0.
    assert p == pytest.approx(1.0)


def test_directional_accuracy_excludes_zero_realized_direction() -> None:
    # The middle observation has a zero realized direction (not scoreable).
    yt = np.array([0.01, 0.0, -0.02])
    yp = np.array([0.01, 0.5, -0.02])
    acc, _ = directional_accuracy(yt, yp)
    # 2 scoreable, both hit -> accuracy 1.0 over the 2 scoreable trials.
    assert acc == pytest.approx(1.0)


def test_directional_accuracy_all_zero_truth_raises() -> None:
    with pytest.raises(ValidationError):
        directional_accuracy(np.zeros(5), np.ones(5))


def test_binomial_pvalue_matches_scipy() -> None:
    scipy_stats = pytest.importorskip("scipy.stats")
    # A clearly directional series: many correct sign hits.
    yt = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0])
    yp = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # 7 of 8 hits
    acc, p = directional_accuracy(yt, yp)
    assert acc == pytest.approx(7.0 / 8.0)
    ref = scipy_stats.binomtest(7, 8, 0.5, alternative="two-sided").pvalue
    assert p == pytest.approx(float(ref), rel=1e-9, abs=1e-12)


# --------------------------------------------------------------------------- #
# net_pnl_sharpe                                                              #
# --------------------------------------------------------------------------- #
def test_net_pnl_sharpe_signal_is_shifted_not_lookahead() -> None:
    # If the signal were NOT shifted, a forecast that perfectly matches the sign
    # of the SAME-day return would be hugely profitable. The shift-by-one means
    # the position at t is sign(forecast_{t-1}), so same-day skill cannot leak.
    yt = np.array([0.01, -0.01, 0.01, -0.01, 0.01, -0.01])
    yp = yt.copy()  # "perfect" same-day forecast
    sharpe = net_pnl_sharpe(yt, yp, cost_bps=0.0)
    # With the lag, the perfectly-alternating series yields a position that is
    # systematically WRONG (last sign predicts the opposite next sign), so the
    # Sharpe is NOT a large positive number — the leakage is closed.
    assert sharpe < 0.0


def test_net_pnl_sharpe_costs_reduce_sharpe() -> None:
    rng = np.random.default_rng(3)
    yt = rng.standard_normal(300) * 0.01
    yp = rng.standard_normal(300) * 0.01
    s0 = net_pnl_sharpe(yt, yp, cost_bps=0.0)
    s_hi = net_pnl_sharpe(yt, yp, cost_bps=50.0)
    # Higher per-side costs can only lower (or hold) the net Sharpe of a
    # turnover-incurring strategy.
    assert s_hi <= s0 + 1e-12


def test_net_pnl_sharpe_flat_position_is_zero() -> None:
    # An all-zero forecast => sign 0 => flat position => zero PnL dispersion.
    yt = np.array([0.01, -0.02, 0.03, -0.01])
    assert net_pnl_sharpe(yt, np.zeros_like(yt)) == 0.0


def test_net_pnl_sharpe_rejects_negative_cost() -> None:
    yt = np.array([0.01, -0.02, 0.03])
    with pytest.raises(ValidationError):
        net_pnl_sharpe(yt, yt, cost_bps=-1.0)


def test_net_pnl_sharpe_first_step_position_is_flat() -> None:
    # Manually replicate: position[0] = 0 (flat before first decision),
    # position[t] = sign(forecast_{t-1}). Verify against a hand computation.
    yt = np.array([0.02, 0.03, -0.01])
    yp = np.array([0.5, -0.5, 0.5])  # signs +, -, +
    # positions: t0 flat(0); t1 = sign(yp0)=+1; t2 = sign(yp1)=-1
    # gross: 0*0.02, +1*0.03, -1*-0.01 = 0, 0.03, 0.01
    # turnover (cost 0): |0-0|, |1-0|, |-1-1| = 0,1,2 ; with cost_bps=0 -> net=gross
    net = np.array([0.0, 0.03, 0.01])
    expected = float(np.mean(net) / np.std(net))
    assert net_pnl_sharpe(yt, yp, cost_bps=0.0) == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# andrews_lag / hac_standard_error                                           #
# --------------------------------------------------------------------------- #
def test_andrews_lag_formula() -> None:
    # ceil(4 * (T/100)^(2/9))
    for t in (50, 100, 250, 1000):
        assert andrews_lag(t) == math.ceil(4.0 * (t / 100.0) ** (2.0 / 9.0))


def test_andrews_lag_rejects_nonpositive() -> None:
    with pytest.raises(ValidationError):
        andrews_lag(0)
    with pytest.raises(ValidationError):
        andrews_lag(-5)


def test_hac_se_lag0_equals_iid_se() -> None:
    # With lag=0 the HAC variance is just gamma0, so the SE is the population
    # standard error of the mean sqrt(var / T) using the 1/T (biased) variance.
    rng = np.random.default_rng(11)
    x = rng.standard_normal(500)
    t = x.size
    gamma0 = float(np.dot(x - x.mean(), x - x.mean()) / t)
    expected = math.sqrt(gamma0 / t)
    assert hac_standard_error(x, lag=0) == pytest.approx(expected, rel=1e-12)


def test_hac_se_nonnegative_and_finite() -> None:
    rng = np.random.default_rng(7)
    # An autocorrelated series (AR(1)) — HAC must stay finite and non-negative.
    e = rng.standard_normal(400)
    x = np.empty(400)
    x[0] = e[0]
    for i in range(1, 400):
        x[i] = 0.6 * x[i - 1] + e[i]
    se = hac_standard_error(x)
    assert math.isfinite(se)
    assert se >= 0.0


def test_hac_se_requires_two_observations() -> None:
    with pytest.raises(ValidationError):
        hac_standard_error(np.array([1.0]))


def test_hac_se_rejects_negative_lag() -> None:
    with pytest.raises(ValidationError):
        hac_standard_error(np.array([1.0, 2.0, 3.0]), lag=-1)


# --------------------------------------------------------------------------- #
# forecast_metrics bundle                                                      #
# --------------------------------------------------------------------------- #
def test_forecast_metrics_bundle_fields() -> None:
    rng = np.random.default_rng(5)
    yt = rng.standard_normal(120) * 0.01
    yp = rng.standard_normal(120) * 0.01
    fm = forecast_metrics(yt, yp)
    assert isinstance(fm, ForecastMetrics)
    assert fm.n_obs == 120
    assert fm.rmse_return == pytest.approx(rmse(yt, yp))
    assert fm.mae_return == pytest.approx(mae(yt, yp))
    assert fm.mase_vs_naive == pytest.approx(mase_vs_naive(yt, yp))
    acc, dir_p = directional_accuracy(yt, yp)
    assert fm.directional_accuracy == pytest.approx(acc)
    assert fm.directional_pvalue == pytest.approx(dir_p)
    assert fm.net_pnl_sharpe == pytest.approx(net_pnl_sharpe(yt, yp))


def test_forecast_metrics_is_frozen() -> None:
    fm = forecast_metrics(np.array([0.01, -0.02, 0.03]), np.array([0.0, 0.0, 0.0]))
    with pytest.raises((AttributeError, TypeError)):
        fm.rmse_return = 0.0  # type: ignore[misc]


def test_forecast_metrics_to_dict_is_json_safe() -> None:
    import json

    fm = forecast_metrics(np.array([0.01, -0.02, 0.03]), np.array([0.0, 0.01, 0.0]))
    d = fm.to_dict()
    json.dumps(d)  # must not raise
    assert set(d) == {
        "rmse_return",
        "mae_return",
        "mase_vs_naive",
        "directional_accuracy",
        "directional_pvalue",
        "net_pnl_sharpe",
        "n_obs",
    }


# --------------------------------------------------------------------------- #
# NO price-level R² — the honest-null contract                                #
# --------------------------------------------------------------------------- #
def test_no_price_level_r2_is_reported() -> None:
    """The evaluation layer must NEVER expose a price-level R² metric.

    The honest-null deliverable judges skill in RETURN space only; a price-level
    R² is a unit-root artifact, not forecasting skill. Assert no public symbol
    references an ``r2`` and the metric bundle carries no such field.
    """
    public_names = [n for n in dir(metrics_mod) if not n.startswith("_")]
    assert not any("r2" in n.lower() for n in public_names)
    assert not any("r_squared" in n.lower() for n in public_names)
    # No field on the frozen bundle is an R².
    assert not any("r2" in f.lower() for f in ForecastMetrics.__annotations__)
