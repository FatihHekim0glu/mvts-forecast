"""Unit tests for the torch-free live baselines: naive random-walk + ARIMA.

These cover the two FLOOR/reference models that run live in the serve container
(no torch, no ONNX):

- ``naive``  — the random-walk last-value forecaster: its return forecast is
  identically zero and it is the OOS prediction floor on a pure random walk;
- ``arima``  — a per-series fixed/auto-order ARIMA via statsmodels: it must fit,
  forecast a finite vector of the requested length, validate its inputs, refuse
  to over-difference a return series, and be deterministic.

The ARIMA tests import statsmodels lazily through the public function (never at
module load), matching the import-purity guard.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from mvtsforecast._exceptions import ValidationError
from mvtsforecast.models.arima import (
    ArimaResult,
    _coerce_forecast,
    arima_forecast,
)
from mvtsforecast.models.naive import (
    NaiveResult,
    naive_forecast,
    persistence_returns,
)

_HAS_PMDARIMA = importlib.util.find_spec("pmdarima") is not None


# --------------------------------------------------------------------------- #
# naive                                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_naive_forecast_is_all_zeros_aligned_to_y_true() -> None:
    """The random-walk return forecast is a zero vector the length of y_true."""
    y_true = np.array([0.01, -0.02, 0.005, 0.0, -0.011])
    result = naive_forecast(y_true)

    assert isinstance(result, NaiveResult)
    assert result.name == "naive"
    assert result.n_obs == y_true.size
    assert result.forecast.shape == (y_true.size,)
    assert np.array_equal(result.forecast, np.zeros(y_true.size))


@pytest.mark.unit
def test_naive_forecast_accepts_list_and_array() -> None:
    """Loosely-typed inputs are coerced; the forecast length follows the input."""
    from_list = naive_forecast([0.1, 0.2, 0.3])  # type: ignore[arg-type]
    from_array = naive_forecast(np.array([0.1, 0.2, 0.3], dtype=np.float64))

    assert from_list.n_obs == 3
    assert np.array_equal(from_list.forecast, from_array.forecast)


@pytest.mark.unit
def test_naive_forecast_to_dict_is_json_safe() -> None:
    """``to_dict`` returns plain Python scalars/lists (JSON-serializable)."""
    out = naive_forecast(np.array([0.01, -0.02])).to_dict()
    assert out["name"] == "naive"
    assert out["n_obs"] == 2
    assert out["forecast"] == [0.0, 0.0]
    assert all(isinstance(x, float) for x in out["forecast"])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("bad", "match"),
    [
        (np.array([]), "non-empty"),
        (np.array([0.1, np.nan, 0.2]), "non-finite"),
        (np.array([0.1, np.inf]), "non-finite"),
    ],
)
def test_naive_forecast_validation(bad: np.ndarray, match: str) -> None:
    """Empty or non-finite ``y_true`` raises a clear ValidationError."""
    with pytest.raises(ValidationError, match=match):
        naive_forecast(bad)


@pytest.mark.unit
def test_persistence_returns_is_the_zero_floor() -> None:
    """The persistence helper returns an all-zeros float64 vector."""
    forecast = persistence_returns(7)
    assert forecast.shape == (7,)
    assert forecast.dtype == np.float64
    assert np.array_equal(forecast, np.zeros(7))


@pytest.mark.unit
def test_persistence_returns_rejects_non_positive() -> None:
    """``n_obs < 1`` raises ValidationError."""
    with pytest.raises(ValidationError, match="n_obs"):
        persistence_returns(0)


@pytest.mark.unit
def test_naive_is_the_prediction_floor_on_a_random_walk() -> None:
    """On a pure random walk no constant beats the zero forecast in MSE.

    For i.i.d. mean-zero returns the conditional mean is 0, so the MSE-minimizing
    constant forecast is exactly the naive 0. Any other constant prediction has
    strictly larger expected squared error, so naive is the floor.
    """
    rng = np.random.Generator(np.random.PCG64(7))
    returns = rng.normal(0.0, 0.01, size=5000)

    naive_mse = float(np.mean((returns - naive_forecast(returns).forecast) ** 2))
    # A handful of non-zero constant forecasts must not do better than naive.
    for c in (-0.01, -0.001, 0.001, 0.01):
        other_mse = float(np.mean((returns - c) ** 2))
        assert naive_mse <= other_mse + 1e-12


# --------------------------------------------------------------------------- #
# arima                                                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_arima_forecast_fits_and_returns_finite_vector() -> None:
    """A fixed-order ARIMA fits and emits ``n_test`` finite forecasts."""
    rng = np.random.Generator(np.random.PCG64(11))
    train = rng.normal(0.0, 0.01, size=250)

    result = arima_forecast(train, n_test=5)

    assert isinstance(result, ArimaResult)
    assert result.name == "arima"
    assert result.order == (1, 0, 0)
    assert result.n_obs == 5
    assert result.forecast.shape == (5,)
    assert np.isfinite(result.forecast).all()


@pytest.mark.unit
def test_arima_forecast_is_deterministic() -> None:
    """Two fits on identical train data yield identical forecasts."""
    rng = np.random.Generator(np.random.PCG64(11))
    train = rng.normal(0.0, 0.01, size=200)

    a = arima_forecast(train, n_test=3)
    b = arima_forecast(train, n_test=3)

    assert np.array_equal(a.forecast, b.forecast)
    assert a.order == b.order


@pytest.mark.unit
def test_arima_forecast_honours_custom_order() -> None:
    """A non-default ``(p, d, q)`` order is fitted and reported back."""
    rng = np.random.Generator(np.random.PCG64(13))
    train = rng.normal(0.0, 0.01, size=300)

    result = arima_forecast(train, n_test=4, order=(2, 0, 1))

    assert result.order == (2, 0, 1)
    assert result.forecast.shape == (4,)
    assert np.isfinite(result.forecast).all()


@pytest.mark.unit
def test_arima_forecast_on_near_constant_series_is_finite_and_near_level() -> None:
    """A (near-)zero-variance series still yields a finite forecast near its level.

    statsmodels may converge to a value extremely close to the constant or the
    code falls back to the train mean; either way the forecast must be finite and
    sit at the constant level.
    """
    train = np.full(100, 0.001)
    result = arima_forecast(train, n_test=3)
    assert result.forecast.shape == (3,)
    assert np.isfinite(result.forecast).all()
    assert np.allclose(result.forecast, 0.001, atol=1e-4)


@pytest.mark.unit
def test_coerce_forecast_keeps_good_vector_and_falls_back_otherwise() -> None:
    """A finite ``n_test`` forecast passes through; a bad one falls back to mean."""
    train = np.array([0.01, 0.03, 0.02])
    good = np.array([0.1, 0.2, 0.3])
    assert np.array_equal(_coerce_forecast(good, 3, train), good)

    # Wrong length -> mean-persistence fallback.
    short = np.array([0.1])
    assert np.allclose(_coerce_forecast(short, 3, train), float(np.mean(train)))

    # Non-finite -> mean-persistence fallback.
    nan_vec = np.array([0.1, np.nan, 0.3])
    assert np.allclose(_coerce_forecast(nan_vec, 3, train), float(np.mean(train)))


@pytest.mark.unit
def test_arima_forecast_to_dict_is_json_safe() -> None:
    """``to_dict`` returns plain lists/scalars for the forecast and order."""
    rng = np.random.Generator(np.random.PCG64(17))
    out = arima_forecast(rng.normal(0.0, 0.01, size=120), n_test=2).to_dict()

    assert out["name"] == "arima"
    assert out["order"] == [1, 0, 0]
    assert isinstance(out["order"], list)
    assert len(out["forecast"]) == 2
    assert all(isinstance(x, float) for x in out["forecast"])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"order": (1, 1, 0)}, "over-difference"),
        ({"order": (1, 2, 0)}, "over-difference"),
    ],
)
def test_arima_forecast_refuses_to_difference_returns(
    kwargs: dict[str, object], match: str
) -> None:
    """A non-zero differencing order on a return series is rejected."""
    train = np.random.Generator(np.random.PCG64(7)).normal(0.0, 0.01, size=100)
    with pytest.raises(ValidationError, match=match):
        arima_forecast(train, n_test=3, **kwargs)  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("train", "n_test", "match"),
    [
        (np.array([]), 3, "non-empty"),
        (np.array([0.1, np.nan, 0.2]), 3, "non-finite"),
        (np.array([0.1, 0.2, 0.3, 0.4]), 0, "n_test"),
    ],
)
def test_arima_forecast_input_validation(
    train: np.ndarray, n_test: int, match: str
) -> None:
    """Empty/non-finite train or non-positive ``n_test`` raise ValidationError."""
    with pytest.raises(ValidationError, match=match):
        arima_forecast(train, n_test=n_test)


@pytest.mark.unit
def test_arima_forecast_rejects_too_short_train() -> None:
    """A train fold shorter than the order's parameter count is rejected."""
    with pytest.raises(ValidationError, match="needs at least"):
        arima_forecast(np.array([0.01, -0.02]), n_test=1, order=(2, 0, 1))


@pytest.mark.unit
def test_arima_forecast_rejects_malformed_order() -> None:
    """An order that is not a (p, d, q) triple raises ValidationError."""
    train = np.random.Generator(np.random.PCG64(7)).normal(0.0, 0.01, size=50)
    with pytest.raises(ValidationError, match="triple"):
        arima_forecast(train, n_test=2, order=(1, 0))  # type: ignore[arg-type]


@pytest.mark.unit
def test_arima_forecast_rejects_negative_order_component() -> None:
    """Negative order components raise ValidationError."""
    train = np.random.Generator(np.random.PCG64(7)).normal(0.0, 0.01, size=50)
    with pytest.raises(ValidationError, match=">= 0"):
        arima_forecast(train, n_test=2, order=(-1, 0, 0))


@pytest.mark.unit
def test_arima_auto_without_pmdarima_raises_clear_error() -> None:
    """When ``pmdarima`` is absent, ``auto=True`` raises a helpful ValidationError."""
    if _HAS_PMDARIMA:
        pytest.skip("pmdarima is installed; the missing-dependency path is not exercised")
    train = np.random.Generator(np.random.PCG64(7)).normal(0.0, 0.01, size=80)
    with pytest.raises(ValidationError, match="pmdarima"):
        arima_forecast(train, n_test=3, auto=True)


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_PMDARIMA, reason="pmdarima (optional [data] extra) not installed")
def test_arima_auto_order_search_runs() -> None:
    """With pmdarima available, auto-order returns a d=0 order and finite forecasts."""
    rng = np.random.Generator(np.random.PCG64(19))
    train = rng.normal(0.0, 0.01, size=300)
    result = arima_forecast(train, n_test=4, auto=True)
    assert result.order[1] == 0  # d must stay 0 on returns
    assert result.forecast.shape == (4,)
    assert np.isfinite(result.forecast).all()
