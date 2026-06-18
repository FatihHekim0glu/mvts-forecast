"""Unit tests for :mod:`mvtsforecast.plots` — valid ``{data, layout}`` + finite.

The two figure builders must return plain, JSON-safe ``{data, layout}`` dicts
(no plotly objects), with finite numeric ``y`` values, and must reject malformed
inputs with :class:`ValidationError` (so the FastAPI layer can map them to 422).
These tests exercise the LAZY-plotly path (plotly is in the ``[viz]`` / ``[dev]``
extras and present in CI); no torch / onnxruntime is touched.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pytest

from mvtsforecast._exceptions import ValidationError
from mvtsforecast.plots import error_figure, forecast_figure


def _assert_valid_figure(fig: dict[str, Any]) -> None:
    """Assert ``fig`` is a JSON-safe ``{data, layout}`` dict with finite y-values."""
    assert isinstance(fig, dict)
    assert set(fig) >= {"data", "layout"}
    assert isinstance(fig["data"], list)
    assert len(fig["data"]) >= 1
    assert isinstance(fig["layout"], dict)
    # The figure must round-trip through JSON unchanged (no plotly objects, no
    # numpy scalars), exactly as the API response embeds it.
    assert json.loads(json.dumps(fig)) == fig
    for trace in fig["data"]:
        assert isinstance(trace, dict)
        for value in trace.get("y", []):
            if value is not None:
                assert math.isfinite(float(value))


def test_forecast_figure_valid_and_finite() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.normal(scale=0.01, size=32)
    forecasts = {
        "naive": np.zeros(32),
        "arima": rng.normal(scale=0.001, size=32),
        "patchtst": rng.normal(scale=0.001, size=32),
    }

    fig = forecast_figure(y_true, forecasts)

    _assert_valid_figure(fig)
    # actual + one trace per model.
    names = [trace.get("name") for trace in fig["data"]]
    assert names == ["actual", "naive", "arima", "patchtst"]


def test_forecast_figure_custom_title() -> None:
    fig = forecast_figure(np.array([0.01, -0.02, 0.0]), {"naive": np.zeros(3)}, title="custom")
    assert fig["layout"]["title"]["text"] == "custom"


def test_forecast_figure_length_mismatch_raises() -> None:
    with pytest.raises(ValidationError, match="align"):
        forecast_figure(np.zeros(10), {"naive": np.zeros(9)})


def test_forecast_figure_empty_actual_raises() -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        forecast_figure(np.array([]), {"naive": np.array([])})


def test_forecast_figure_nonfinite_actual_raises() -> None:
    with pytest.raises(ValidationError, match="finite"):
        forecast_figure(np.array([0.0, np.nan, 0.0]), {"naive": np.zeros(3)})


def test_error_figure_valid_and_finite() -> None:
    rmse = {"naive": 0.0101, "arima": 0.0102, "patchtst": 0.0103}
    dir_acc = {"naive": 0.50, "arima": 0.49, "patchtst": 0.51}

    fig = error_figure(rmse, dir_acc)

    _assert_valid_figure(fig)
    assert fig["layout"]["barmode"] == "group"
    trace_names = {trace.get("name") for trace in fig["data"]}
    assert trace_names == {"RMSE", "directional accuracy"}
    # The RMSE bar carries the model x-categories in stable (declared) order.
    rmse_trace = next(t for t in fig["data"] if t["name"] == "RMSE")
    assert list(rmse_trace["x"]) == ["naive", "arima", "patchtst"]


def test_error_figure_custom_title() -> None:
    fig = error_figure({"naive": 0.01}, {"naive": 0.5}, title="errors!")
    assert fig["layout"]["title"]["text"] == "errors!"


def test_error_figure_empty_raises() -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        error_figure({}, {})


def test_error_figure_key_mismatch_raises() -> None:
    with pytest.raises(ValidationError, match="same model keys"):
        error_figure({"naive": 0.01, "arima": 0.02}, {"naive": 0.5})


def test_error_figure_nonfinite_value_raises() -> None:
    with pytest.raises(ValidationError, match="finite"):
        error_figure({"naive": float("nan")}, {"naive": 0.5})


def test_plots_import_is_torch_free() -> None:
    """Importing plots must not pull in torch / onnxruntime (import purity).

    Verified in a FRESH interpreter so the assertion is robust to ``sys.modules``
    pollution from earlier ``slow`` torch tests in the same pytest process.
    """
    import subprocess
    import sys

    code = (
        "import sys;"
        "import mvtsforecast.plots;"
        "bad=[m for m in ('torch','onnxruntime') if m in sys.modules];"
        "print(','.join(bad))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "", f"importing plots leaked: {out.stdout.strip()}"
