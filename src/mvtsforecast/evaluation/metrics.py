"""Return-space forecast metrics and risk-adjusted PnL (NO price-level R²).

Everything here lives in RETURN space, where the honest comparison happens:

- :func:`rmse` / :func:`mae` — out-of-sample error of a model's return forecast;
- :func:`mase_vs_naive` — Mean Absolute Scaled Error against the naive random-walk
  baseline; ``MASE >= 1`` means the model does NOT beat the naive last-value
  forecast;
- :func:`directional_accuracy` — sign-hit rate, with a binomial test vs. 0.5;
- :func:`net_pnl_sharpe` — the per-observation Sharpe of a sign-following strategy
  net of per-side transaction costs (the signal is ``sign(forecast).shift(1)`` so
  it is tradable, never lookahead);
- :func:`forecast_metrics` — assembles the frozen :class:`ForecastMetrics` bundle.

DEBUNKED TRAP (documented once, never computed as a metric): a price-LEVEL R²
looks deceptively high because the integrated/trended price level is dominated by
its own lag — that is a unit-root artifact, NOT forecasting skill. We therefore
NEVER report a price-level R². All skill is judged in return space.

Importing this module has no side effects.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from mvtsforecast._exceptions import ValidationError
from mvtsforecast._typing import FloatArray

# quantcore-candidate: HAC long-run variance mirrors
# pairs-trading:evaluation/hac.py (Newey-West, Bartlett, Andrews lag).


def _coerce_pair(
    y_true: FloatArray,
    y_pred: FloatArray,
    *,
    true_name: str = "y_true",
    pred_name: str = "y_pred",
) -> tuple[FloatArray, FloatArray]:
    """Coerce a forecast pair to aligned, finite, equal-length float64 arrays.

    Both inputs are flattened to 1-D, checked for non-emptiness, equal length,
    and finiteness. This is the single boundary every metric in this module
    funnels its inputs through.

    Parameters
    ----------
    y_true, y_pred:
        Realized and forecast next-step returns.
    true_name, pred_name:
        Human-readable labels for error messages.

    Returns
    -------
    tuple[FloatArray, FloatArray]
        The two coerced 1-D float64 arrays.

    Raises
    ------
    ValidationError
        If either array is empty, lengths differ, or any value is non-finite.
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    if yt.size == 0 or yp.size == 0:
        raise ValidationError(f"{true_name} and {pred_name} must be non-empty.")
    if yt.size != yp.size:
        raise ValidationError(
            f"{true_name} (len {yt.size}) and {pred_name} (len {yp.size}) "
            "must have the same length."
        )
    if not np.isfinite(yt).all():
        raise ValidationError(f"{true_name} contains non-finite values.")
    if not np.isfinite(yp).all():
        raise ValidationError(f"{pred_name} contains non-finite values.")
    return yt, yp


def _naive_or_zeros(y_true: FloatArray, y_pred_naive: FloatArray | None) -> FloatArray:
    """Return the naive forecast: the given vector, else the random walk (zeros).

    The naive last-value / random-walk next-step return forecast is ``r_hat = 0``.

    Parameters
    ----------
    y_true:
        Realized returns (defines the expected length).
    y_pred_naive:
        The naive forecasts; ``None`` => an all-zeros vector.

    Returns
    -------
    FloatArray
        The naive forecast vector.

    Raises
    ------
    ValidationError
        If a provided vector is the wrong length or non-finite.
    """
    if y_pred_naive is None:
        return np.zeros_like(y_true)
    naive = np.asarray(y_pred_naive, dtype=np.float64).ravel()
    if naive.size != y_true.size:
        raise ValidationError(
            f"y_pred_naive (len {naive.size}) must match y_true (len {y_true.size})."
        )
    if not np.isfinite(naive).all():
        raise ValidationError("y_pred_naive contains non-finite values.")
    return naive


@dataclass(frozen=True, slots=True)
class ForecastMetrics:
    """Immutable bundle of return-space out-of-sample forecast metrics.

    Attributes
    ----------
    rmse_return:
        Root-mean-squared error of the model's next-step return forecast.
    mae_return:
        Mean absolute error of the model's next-step return forecast.
    mase_vs_naive:
        MAE scaled by the naive baseline's MAE. ``>= 1`` => no improvement.
    directional_accuracy:
        Fraction of next-step return signs correctly predicted.
    directional_pvalue:
        Binomial-test p-value for ``directional_accuracy > 0.5``.
    net_pnl_sharpe:
        Per-observation Sharpe of the sign-following strategy net of costs.
    n_obs:
        Number of out-of-sample forecasts evaluated.
    """

    rmse_return: float
    mae_return: float
    mase_vs_naive: float
    directional_accuracy: float
    directional_pvalue: float
    net_pnl_sharpe: float
    n_obs: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of these metrics."""
        return asdict(self)


def rmse(y_true: FloatArray, y_pred: FloatArray) -> float:
    """Return the root-mean-squared error of ``y_pred`` against ``y_true``.

    Parameters
    ----------
    y_true:
        Realized next-step returns.
    y_pred:
        Forecast next-step returns (same length).

    Returns
    -------
    float
        ``sqrt(mean((y_true - y_pred)**2))``.

    Raises
    ------
    ValidationError
        If the inputs are empty or length-mismatched.
    """
    raise NotImplementedError


def mae(y_true: FloatArray, y_pred: FloatArray) -> float:
    """Return the mean absolute error of ``y_pred`` against ``y_true``.

    Parameters
    ----------
    y_true:
        Realized next-step returns.
    y_pred:
        Forecast next-step returns (same length).

    Returns
    -------
    float
        ``mean(|y_true - y_pred|)``.

    Raises
    ------
    ValidationError
        If the inputs are empty or length-mismatched.
    """
    raise NotImplementedError


def mase_vs_naive(
    y_true: FloatArray,
    y_pred_model: FloatArray,
    y_pred_naive: FloatArray | None = None,
) -> float:
    r"""Mean Absolute Scaled Error of the model relative to the naive baseline.

    Returns ``MAE(model) / MAE(naive)`` where the naive forecast is ``r_hat = 0``
    (the random walk). A value ``>= 1`` means the model does NOT beat the naive
    baseline in return space — the expected, honest outcome on noisy daily
    returns.

    Parameters
    ----------
    y_true:
        Realized next-step returns.
    y_pred_model:
        The model's forecasts.
    y_pred_naive:
        The naive forecasts; defaults to an all-zeros vector.

    Returns
    -------
    float
        The MASE ratio.

    Raises
    ------
    ValidationError
        If inputs are empty/mismatched or the baseline MAE is zero.
    """
    raise NotImplementedError


def directional_accuracy(y_true: FloatArray, y_pred: FloatArray) -> tuple[float, float]:
    """Return the sign-hit rate and a two-sided binomial-test p-value vs. 0.5.

    Counts observations where ``sign(y_pred) == sign(y_true)`` (observations with
    a zero realized direction are not scoreable) and tests the hit rate against
    the no-skill rate 0.5 with a two-sided binomial test.

    Parameters
    ----------
    y_true:
        Realized next-step returns.
    y_pred:
        Forecast next-step returns.

    Returns
    -------
    tuple[float, float]
        ``(accuracy, binomial_pvalue)``.

    Raises
    ------
    ValidationError
        If inputs are empty, length-mismatched, or have no scoreable direction.
    """
    raise NotImplementedError


def net_pnl_sharpe(
    y_true: FloatArray,
    y_pred: FloatArray,
    *,
    cost_bps: float = 1.0,
) -> float:
    r"""Per-observation Sharpe of a sign-following strategy net of costs.

    The position at ``t`` is ``sign(forecast_{t-1})`` (the signal is shifted by
    one so it is TRADABLE, never using same-step information); the gross PnL is
    ``position_t * r_t``; the cost is ``cost_bps / 10_000`` charged on each unit
    of one-way turnover ``|position_t - position_{t-1}|``. The returned statistic
    is ``mean(net_pnl) / std(net_pnl)`` (non-annualized; the DSR layer handles
    multiplicity).

    Parameters
    ----------
    y_true:
        Realized next-step returns.
    y_pred:
        The model's forecasts (drive the sign signal).
    cost_bps:
        Per-side transaction cost in basis points (``>= 0``).

    Returns
    -------
    float
        The net per-observation Sharpe ratio (``0.0`` if PnL variance is zero).

    Raises
    ------
    ValidationError
        If inputs are empty/mismatched or ``cost_bps`` is negative.
    """
    raise NotImplementedError


def hac_standard_error(series: FloatArray, *, lag: int | None = None) -> float:
    """Newey-West HAC standard error of the sample mean of ``series``.

    Uses Bartlett weights; ``lag=None`` selects the Andrews (1991) automatic
    truncation ``ceil(4 * (T/100)**(2/9))``. Used to build the Diebold-Mariano
    statistic's denominator from the loss-differential series.

    Parameters
    ----------
    series:
        A 1-D series (e.g. the DM loss differential).
    lag:
        Bartlett lag truncation; ``None`` => Andrews rule.

    Returns
    -------
    float
        ``sqrt(omega_hat / T)``, the HAC standard error of the mean.

    Raises
    ------
    ValidationError
        If ``series`` has fewer than two finite observations or ``lag < 0``.
    """
    # quantcore-candidate: mirrors pairs-trading:evaluation/hac.py::newey_west_se.
    raise NotImplementedError


def andrews_lag(t: int) -> int:
    """Andrews (1991) automatic Bartlett lag truncation ``ceil(4*(T/100)**(2/9))``.

    Parameters
    ----------
    t:
        Sample size (must be positive).

    Returns
    -------
    int
        The non-negative lag truncation.

    Raises
    ------
    ValidationError
        If ``t <= 0``.
    """
    raise NotImplementedError


def forecast_metrics(
    y_true: FloatArray,
    y_pred_model: FloatArray,
    y_pred_naive: FloatArray | None = None,
    *,
    cost_bps: float = 1.0,
) -> ForecastMetrics:
    """Compute the full return-space metric bundle in one call.

    Assembles RMSE, MAE, MASE-vs-naive, directional accuracy + binomial p-value,
    and the net-of-cost PnL Sharpe into a frozen :class:`ForecastMetrics`.
    Deliberately omits any price-level R² (the debunked trap).

    Parameters
    ----------
    y_true:
        Realized next-step returns.
    y_pred_model:
        The model's forecasts.
    y_pred_naive:
        The naive forecasts; defaults to all zeros.
    cost_bps:
        Per-side transaction cost in basis points for the PnL Sharpe.

    Returns
    -------
    ForecastMetrics
        The frozen metric bundle.

    Raises
    ------
    ValidationError
        If inputs are empty or length-mismatched.
    """
    raise NotImplementedError


def _norm_sf(x: float) -> float:
    """Standard-normal survival function ``1 - Phi(x)`` via the error function."""
    return 0.5 * math.erfc(x / math.sqrt(2.0))
