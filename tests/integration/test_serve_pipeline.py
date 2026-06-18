"""Integration: the end-to-end serve pipeline runs TORCH-FREE on the null.

The deployed request path is exercised end to end — synthetic panel -> purged,
leakage-safe windows -> live naive + ARIMA baselines -> scored comparison ->
PURE ``deep_beats_naive`` verdict + two Plotly figures — WITHOUT any deep models
(no committed ONNX artifacts in a fresh checkout) and WITHOUT importing torch.

These tests pin the honest NULL: on the synthetic panel the live baselines do not
beat the naive random walk, so ``deep_beats_naive`` is ``False`` and the summary
is fully JSON-safe for the FastAPI response. ``run_compare`` (the workflow brief's
serve-entrypoint name) is asserted to be the same callable as ``run_forecast``.
"""

from __future__ import annotations

import json
import sys

import pytest

from mvtsforecast._exceptions import ValidationError
from mvtsforecast.serve import ForecastRun, run_compare, run_forecast


def _assert_torch_free(torch_was_loaded: bool) -> None:
    """Assert the action under test did not newly import torch / onnxruntime.

    Robust to ``sys.modules`` pollution from earlier ``slow`` torch tests: only
    assert the serve action itself stayed engine-free when nothing was loaded yet.
    """
    if not torch_was_loaded:
        assert "torch" not in sys.modules
    # onnxruntime is never imported when no deep model is served (none ship in a
    # fresh checkout), regardless of any prior pollution: it is import-pure here.
    assert "onnxruntime" not in sys.modules


@pytest.mark.integration
def test_run_compare_is_run_forecast() -> None:
    """The brief's ``run_compare`` serve entrypoint is the canonical pipeline."""
    assert run_compare is run_forecast


@pytest.mark.integration
def test_end_to_end_naive_arima_no_torch() -> None:
    """Synthetic -> windows -> naive+ARIMA -> compare runs torch-free; verdict False."""
    torch_was_loaded = "torch" in sys.modules

    run = run_forecast(
        basket=["SPY", "TLT", "GLD"],
        target="SPY",
        models=["naive", "arima"],
        lookback=20,
        horizon=1,
        seed=7,
    )

    assert isinstance(run, ForecastRun)
    summary = run.summary
    # Both live baselines were scored; naive always participates as the floor.
    assert set(summary.rmse_by_model) == {"naive", "arima"}
    assert set(summary.directional_acc_by_model) == {"naive", "arima"}
    assert summary.data_source == "synthetic"
    # The honest NULL: no deep model is served, so the verdict cannot be True.
    assert summary.deep_beats_naive is False
    assert summary.n_effective_trials >= 1
    # ARIMA does not beat naive at 5% on the synthetic null (DM insignificant).
    assert summary.dm_pvalue_vs_naive["arima"] >= 0.05
    # Naive is (at worst tied for) the lowest-RMSE model on the null.
    assert summary.rmse_by_model[summary.best_model] <= summary.rmse_by_model["arima"] + 1e-12

    # The whole summary must be JSON-safe for the FastAPI response.
    payload = json.loads(json.dumps(run.to_dict()))
    assert payload["summary"]["deep_beats_naive"] is False
    assert set(payload["forecast_figure"]) >= {"data", "layout"}
    assert set(payload["error_figure"]) >= {"data", "layout"}

    _assert_torch_free(torch_was_loaded)


@pytest.mark.integration
def test_naive_only_request_runs_and_is_torch_free() -> None:
    """A naive-only request still produces a valid, torch-free summary."""
    torch_was_loaded = "torch" in sys.modules

    run = run_forecast(basket=["SPY", "TLT"], target="TLT", models=["naive"], lookback=16, seed=11)

    assert set(run.summary.rmse_by_model) == {"naive"}
    assert run.summary.best_model == "naive"
    assert run.summary.deep_beats_naive is False
    _assert_torch_free(torch_was_loaded)


@pytest.mark.integration
def test_short_panel_falls_back_to_baselines() -> None:
    """A panel too short for a deep window still scores the live baselines."""
    torch_was_loaded = "torch" in sys.modules

    # A long look-back forces the no-deep-window fallback branch in the split.
    run = run_forecast(
        basket=["SPY", "TLT", "GLD"],
        target="SPY",
        models=["naive", "arima"],
        lookback=60,
        seed=7,
    )

    assert set(run.summary.rmse_by_model) == {"naive", "arima"}
    assert run.summary.deep_beats_naive is False
    _assert_torch_free(torch_was_loaded)


@pytest.mark.integration
def test_horizon_five_runs_torch_free() -> None:
    """The 5-step horizon path runs end to end, torch-free, on the null."""
    torch_was_loaded = "torch" in sys.modules

    run = run_forecast(
        basket=["SPY", "TLT", "GLD"],
        target="SPY",
        models=["naive", "arima"],
        lookback=20,
        horizon=5,
        seed=7,
    )

    assert set(run.summary.rmse_by_model) == {"naive", "arima"}
    assert run.summary.deep_beats_naive is False
    _assert_torch_free(torch_was_loaded)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"basket": [], "target": "SPY"}, "at least one ticker"),
        ({"basket": ["SPY", "TLT"], "target": "ZZZ"}, "target"),
        ({"basket": ["SPY", "TLT"], "target": "SPY", "horizon": 3}, "horizon"),
        ({"basket": ["SPY"], "target": "SPY", "models": ["bogus"]}, "unknown model"),
    ],
)
def test_invalid_requests_raise_validation_error(kwargs: dict[str, object], match: str) -> None:
    """Bad requests are rejected with :class:`ValidationError` (mapped to 422)."""
    with pytest.raises(ValidationError, match=match):
        run_forecast(lookback=16, **kwargs)  # type: ignore[arg-type]
