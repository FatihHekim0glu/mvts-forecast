"""Serve entrypoints the backend calls (onnxruntime + numpy + statsmodels, NO torch).

The FastAPI router calls :func:`run_forecast` to compare the models on the
synthetic panel (or a real basket loaded via the CLI path) and return a JSON-safe
summary plus two Plotly figures. The naive and ARIMA baselines are computed LIVE
(pure numpy / statsmodels); the deep models (LSTM, PatchTST, the interpretable
transformer) are served from their committed ONNX artifacts via onnxruntime —
torch is NEVER imported. The honest ``deep_beats_naive`` verdict is the PURE
function of the inference outputs.

The committed offline-trained metrics may also be read from
``artifacts/metrics.json`` so the deployed default returns instantly without
re-running the deep forward pass; either way the request path NEVER trains.

Importing this module has no side effects (onnxruntime / statsmodels are imported
lazily inside the deep / ARIMA paths).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from mvtsforecast.data.synthetic import DEFAULT_BASKET


@dataclass(frozen=True, slots=True)
class ForecastSummary:
    """Immutable, JSON-safe summary of the deep-vs-naive comparison.

    Attributes
    ----------
    rmse_by_model:
        Return-space OOS RMSE keyed by model name.
    directional_acc_by_model:
        Directional accuracy keyed by model name.
    dm_pvalue_vs_naive:
        Diebold-Mariano p-value vs. naive keyed by model name.
    deflated_sharpe:
        Deflated Sharpe (FULL-grid ``n_trials``) keyed by model name.
    best_model:
        Name of the lowest-RMSE model overall.
    deep_beats_naive:
        The PURE verdict: ``True`` iff a deep model beats naive with a
        DM-significant margin AND a positive DSR.
    n_effective_trials:
        The honest multiplicity count used for the DSR.
    data_source:
        Provenance of the input panel (``"synthetic"`` / ``"yfinance"`` / ...).
    """

    rmse_by_model: dict[str, float]
    directional_acc_by_model: dict[str, float]
    dm_pvalue_vs_naive: dict[str, float]
    deflated_sharpe: dict[str, float]
    best_model: str
    deep_beats_naive: bool
    n_effective_trials: int
    data_source: str

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this summary."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ForecastRun:
    """Immutable bundle returned to the backend: summary + two Plotly figures.

    Attributes
    ----------
    summary:
        The :class:`ForecastSummary`.
    forecast_figure:
        A Plotly ``{data, layout}`` dict: realized target vs. each model forecast.
    error_figure:
        A Plotly ``{data, layout}`` dict: RMSE / directional accuracy by model.
    """

    summary: ForecastSummary
    forecast_figure: dict[str, Any] = field(default_factory=dict)
    error_figure: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this run."""
        return {
            "summary": self.summary.to_dict(),
            "forecast_figure": self.forecast_figure,
            "error_figure": self.error_figure,
        }


def run_forecast(
    *,
    basket: Sequence[str] = DEFAULT_BASKET,
    target: str = "SPY",
    horizon: int = 1,
    models: Sequence[str] = ("naive", "arima", "lstm", "patchtst", "transformer"),
    lookback: int = 60,
    data_source_pref: str = "synthetic",
    seed: int = 7,
) -> ForecastRun:
    """Run the end-to-end comparison and return a JSON-safe summary + figures.

    Builds (or loads) the panel, windows it leakage-safely, computes the naive and
    ARIMA forecasts LIVE, serves the requested deep models from their committed
    ONNX artifacts (onnxruntime, NO torch), scores everything in return space,
    derives the PURE ``deep_beats_naive`` verdict, and assembles the two Plotly
    figures. NEVER trains on the request path.

    Parameters
    ----------
    basket:
        Asset tickers in the panel.
    target:
        The forecast target column (must be in ``basket``).
    horizon:
        Forecast horizon in steps (1 or 5).
    models:
        Subset of ``{"naive", "arima", "lstm", "patchtst", "transformer"}``.
    lookback:
        Window length (default 60).
    data_source_pref:
        ``"synthetic"`` (default) or ``"auto"``/``"yfinance"`` for the real path.
    seed:
        Master RNG seed for the synthetic panel.

    Returns
    -------
    ForecastRun
        The summary and figures for the backend response.

    Raises
    ------
    ValidationError
        If the request is invalid (target absent, unknown model, bad horizon).
    ArtifactError
        If a requested deep model's ONNX artifact cannot be loaded.
    """
    raise NotImplementedError


def forecast_from_onnx(
    model_name: str,
    x: Any,
) -> Any:
    """Serve one deep model's OOS forecasts from its committed ONNX artifact.

    A thin convenience wrapper over
    :class:`mvtsforecast.models.onnx_runtime.OnnxForecaster` for the backend: load
    the named artifact lazily (onnxruntime, NO torch) and run the forward pass on
    the pre-scaled windows.

    Parameters
    ----------
    model_name:
        One of the deep-model names (``"lstm"`` / ``"patchtst"`` /
        ``"transformer"``).
    x:
        A ``(n_samples, look_back, n_features)`` pre-scaled sequence tensor.

    Returns
    -------
    Any
        The ``(n_samples,)`` next-step return forecast.

    Raises
    ------
    ArtifactError
        If the artifact is missing/corrupt or its signature mismatches ``x``.
    """
    raise NotImplementedError
