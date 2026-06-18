"""Per-series ARIMA baseline (statsmodels / optional pmdarima) — torch-free.

A classical linear baseline fitted per target series on the train fold and rolled
forward over the OOS test fold. ARIMA is pure ``statsmodels`` (with an optional
``pmdarima`` auto-order search behind the ``data`` extra), so — like
:mod:`mvtsforecast.models.naive` — it runs LIVE in the serve container with NO
torch and NO ONNX. On noisy daily returns its forecast is, in practice, also
statistically indistinguishable from naive; it is included as the honest
classical reference point between naive and the deep models.

``statsmodels`` is imported LAZILY inside the functions (the ``data`` extra), so
importing this module pulls in nothing heavy and has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from mvtsforecast._typing import FloatArray


@dataclass(frozen=True, slots=True)
class ArimaResult:
    """Immutable result of a rolled per-series ARIMA forecast over the OOS samples.

    Attributes
    ----------
    name:
        Model identifier (``"arima"``).
    forecast:
        The ``(n_samples,)`` next-step return forecast.
    order:
        The fitted ``(p, d, q)`` ARIMA order.
    n_obs:
        Number of out-of-sample forecasts.
    """

    name: str
    forecast: FloatArray
    order: tuple[int, int, int]
    n_obs: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result."""
        out: dict[str, Any] = asdict(self)
        out["forecast"] = [float(x) for x in np.asarray(self.forecast).ravel()]
        out["order"] = list(self.order)
        return out


def arima_forecast(
    train_returns: FloatArray,
    n_test: int,
    *,
    order: tuple[int, int, int] = (1, 0, 0),
    auto: bool = False,
) -> ArimaResult:
    """Fit an ARIMA on the train returns and roll one-step forecasts over the test.

    LAZY IMPORT: ``statsmodels`` (and, when ``auto=True``, ``pmdarima``) is imported
    inside this function. The model is fitted on ``train_returns`` and produces
    ``n_test`` successive one-step-ahead forecasts (the series is extended with
    realized values as it rolls, never with future information).

    Parameters
    ----------
    train_returns:
        The in-sample (train-fold) target RETURN series.
    n_test:
        Number of out-of-sample one-step forecasts to produce.
    order:
        The fixed ``(p, d, q)`` order; on daily returns ``d=0`` (returns are
        already stationary — differencing the return would over-difference).
    auto:
        If ``True``, search the order with ``pmdarima.auto_arima`` instead of using
        the fixed ``order``.

    Returns
    -------
    ArimaResult
        The frozen ARIMA result.

    Raises
    ------
    ValidationError
        If ``train_returns`` is too short for the order, ``n_test < 1``, or
        ``d != 0`` would over-difference a return series.
    """
    raise NotImplementedError
