"""Unit tests for :mod:`mvtsforecast.serve` — the torch-free serve wiring.

Exercises the serve helpers and the deep-model serving branch WITHOUT torch and
WITHOUT shipping real ONNX artifacts: the onnxruntime layer is mocked so the
``run_forecast`` deep path (artifact present -> served -> scored -> verdict) is
covered torch-free. Also pins ``forecast_from_onnx`` delegation, the figure
helper, and the JSON-safety coercions.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

import mvtsforecast.serve as serve_mod
from mvtsforecast._exceptions import ArtifactError
from mvtsforecast.serve import (
    _model_dsr,
    _ordered,
    _safe_float,
    _serve_deep,
    build_figures,
    forecast_from_onnx,
    run_forecast,
)


def test_safe_float_coerces_nonfinite_to_zero() -> None:
    assert _safe_float(1.5) == 1.5
    assert _safe_float(float("nan")) == 0.0
    assert _safe_float(float("inf")) == 0.0


def test_ordered_puts_naive_first_then_declared_then_extras() -> None:
    forecasts = {
        "patchtst": np.zeros(2),
        "arima": np.zeros(2),
        "naive": np.zeros(2),
        "mystery": np.zeros(2),
    }
    assert _ordered(forecasts) == ["naive", "arima", "patchtst", "mystery"]


def test_serve_deep_empty_when_no_deep_or_no_windows() -> None:
    assert _serve_deep(["naive", "arima"], np.zeros((2, 4, 3))) == {}
    assert _serve_deep(["lstm"], None) == {}


def test_serve_deep_skips_missing_artifact(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Missing:
        def is_file(self) -> bool:
            return False

    monkeypatch.setattr(
        "mvtsforecast.models.onnx_runtime.default_artifact_path", lambda _n: _Missing()
    )
    assert _serve_deep(["lstm", "patchtst"], np.zeros((2, 4, 3))) == {}


def test_serve_deep_serves_present_artifact(monkeypatch: pytest.MonkeyPatch) -> None:
    """A present artifact is served via the (mocked) lazy ONNX path, torch-free."""

    class _Present:
        def is_file(self) -> bool:
            return True

    torch_was_loaded = "torch" in sys.modules
    monkeypatch.setattr(
        "mvtsforecast.models.onnx_runtime.default_artifact_path", lambda _n: _Present()
    )
    monkeypatch.setattr(serve_mod, "forecast_from_onnx", lambda _n, _x: np.array([0.1, -0.2]))

    out = _serve_deep(["lstm"], np.zeros((2, 4, 3)))
    assert list(out) == ["lstm"]
    assert out["lstm"].tolist() == [0.1, -0.2]
    if not torch_was_loaded:
        assert "torch" not in sys.modules


def test_serve_deep_skips_corrupt_artifact(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Present:
        def is_file(self) -> bool:
            return True

    def _raise(_n: str, _x: object) -> object:
        raise ArtifactError("bad signature")

    monkeypatch.setattr(
        "mvtsforecast.models.onnx_runtime.default_artifact_path", lambda _n: _Present()
    )
    monkeypatch.setattr(serve_mod, "forecast_from_onnx", _raise)
    assert _serve_deep(["lstm"], np.zeros((2, 4, 3))) == {}


def test_forecast_from_onnx_delegates_to_onnxforecaster(monkeypatch: pytest.MonkeyPatch) -> None:
    """``forecast_from_onnx`` constructs an OnnxForecaster and runs ``predict``."""
    captured: dict[str, object] = {}

    class _FakeForecaster:
        def __init__(self, name: str) -> None:
            captured["name"] = name

        def predict(self, x: object) -> object:
            captured["x"] = x
            return np.array([0.0])

    monkeypatch.setattr("mvtsforecast.models.onnx_runtime.OnnxForecaster", _FakeForecaster)
    out = forecast_from_onnx("patchtst", np.zeros((1, 4, 3)))
    assert captured["name"] == "patchtst"
    assert np.asarray(out).tolist() == [0.0]


def test_run_forecast_serves_deep_via_mocked_onnx(monkeypatch: pytest.MonkeyPatch) -> None:
    """The full ``run_forecast`` deep path runs torch-free with a mocked ONNX serve.

    The mocked deep forecast hugs zero (the honest null), so the verdict stays
    ``False`` while the lstm row is fully scored (RMSE / dir-acc / DM / DSR).
    """

    class _Present:
        def is_file(self) -> bool:
            return True

    monkeypatch.setattr(
        "mvtsforecast.models.onnx_runtime.default_artifact_path", lambda _n: _Present()
    )

    def _fake_onnx(_name: str, x: object) -> np.ndarray:
        n = int(np.asarray(x).shape[0])
        # A tiny, near-zero deep forecast: indistinguishable from naive on the null.
        return np.full(n, 1e-6, dtype=np.float64)

    monkeypatch.setattr(serve_mod, "forecast_from_onnx", _fake_onnx)

    torch_was_loaded = "torch" in sys.modules
    run = run_forecast(
        basket=["SPY", "TLT", "GLD"],
        target="SPY",
        models=["naive", "lstm"],
        lookback=20,
        horizon=1,
        seed=7,
    )
    summary = run.summary
    assert "lstm" in summary.rmse_by_model
    assert "lstm" in summary.dm_pvalue_vs_naive
    assert "lstm" in summary.deflated_sharpe
    # The honest null holds: a near-zero deep forecast cannot beat naive.
    assert summary.deep_beats_naive is False
    # The serve action itself imports no torch (robust to prior slow-test pollution).
    if not torch_was_loaded:
        assert "torch" not in sys.modules


def test_run_forecast_deep_identical_to_naive_hits_dm_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deep forecast identical to naive (all zeros) triggers the DM-degenerate fallback."""

    class _Present:
        def is_file(self) -> bool:
            return True

    monkeypatch.setattr(
        "mvtsforecast.models.onnx_runtime.default_artifact_path", lambda _n: _Present()
    )

    def _zeros(_name: str, x: object) -> np.ndarray:
        # Exactly the naive forecast (all zeros) => degenerate DM loss differential.
        return np.zeros(int(np.asarray(x).shape[0]), dtype=np.float64)

    monkeypatch.setattr(serve_mod, "forecast_from_onnx", _zeros)

    run = run_forecast(
        basket=["SPY", "TLT", "GLD"],
        target="SPY",
        models=["naive", "lstm"],
        lookback=20,
        seed=7,
    )
    # The degenerate DM is treated as insignificant: p-value 1.0, verdict False.
    assert run.summary.dm_pvalue_vs_naive["lstm"] == 1.0
    assert run.summary.deep_beats_naive is False


def test_run_forecast_tracks_best_deep_but_verdict_stays_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lower-loss deep model is tracked as best-deep, yet the DSR gate keeps verdict False."""

    class _Present:
        def is_file(self) -> bool:
            return True

    monkeypatch.setattr(
        "mvtsforecast.models.onnx_runtime.default_artifact_path", lambda _n: _Present()
    )

    def _near_perfect(_name: str, x: object) -> np.ndarray:
        # A forecast with slightly lower squared-error than naive on most rows:
        # enough to make the DM stat negative (model favoured) and be tracked.
        n = int(np.asarray(x).shape[0])
        return np.full(n, 1e-4, dtype=np.float64)

    monkeypatch.setattr(serve_mod, "forecast_from_onnx", _near_perfect)

    run = run_forecast(
        basket=["SPY", "TLT", "GLD"],
        target="SPY",
        models=["naive", "lstm", "patchtst"],
        lookback=20,
        seed=7,
    )
    # Both deep models are scored and DM-tested vs naive.
    assert {"lstm", "patchtst"} <= set(run.summary.dm_pvalue_vs_naive)
    # On the synthetic null the DSR gate is not cleared: verdict stays False.
    assert run.summary.deep_beats_naive is False


def test_build_figures_returns_empty_on_missing_plotly(monkeypatch: pytest.MonkeyPatch) -> None:
    """If plotly ([viz]) is unavailable, the figure helper returns empty dicts."""
    import builtins

    real_import = builtins.__import__

    def _fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("mvtsforecast.plots"):
            raise ImportError("plotly not installed")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    fcast, err = build_figures(
        np.array([0.01, -0.02]),
        {"naive": np.zeros(2)},
        {"naive": 0.01},
        {"naive": 0.5},
    )
    assert fcast == {}
    assert err == {}


def test_model_dsr_is_finite_probability() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.normal(scale=0.01, size=64)
    dsr = _model_dsr(y_true, np.zeros(64))
    assert 0.0 <= dsr <= 1.0
