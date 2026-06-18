"""Plotly figure builders (LAZY plotly, [viz]) — serialize to ``{data, layout}``.

Two figures back the frontend tool:

- :func:`forecast_figure` — the realized target return vs. each model's forecast
  over the OOS window (so the eye can see that the deep forecasts hug zero, like
  naive);
- :func:`error_figure` — a grouped bar of return-space RMSE and directional
  accuracy by model (so the lack of a deep edge is legible at a glance).

``plotly`` is imported LAZILY inside the builders (the ``[viz]`` extra), and each
figure is returned as a plain ``{data, layout}`` dict via
``json.loads(pio.to_json(fig, validate=False))`` so the API response carries no
plotly objects. Importing this module has no side effects.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mvtsforecast._typing import FloatArray


def forecast_figure(
    y_true: FloatArray,
    forecasts_by_model: Mapping[str, FloatArray],
    *,
    title: str = "Actual vs. forecast (next-step returns)",
) -> dict[str, Any]:
    """Build the actual-vs-forecast Plotly figure as a ``{data, layout}`` dict.

    LAZY IMPORT: ``plotly`` is imported inside this function (the ``[viz]`` extra).
    Plots the realized target return as one trace and each model's forecast as an
    overlaid trace, then serializes to a plain dict.

    Parameters
    ----------
    y_true:
        The realized next-step target returns over the OOS window.
    forecasts_by_model:
        Map of model name -> its ``(n_obs,)`` forecast (same length as
        ``y_true``).
    title:
        Figure title.

    Returns
    -------
    dict[str, Any]
        A Plotly ``{data, layout}`` dict (no plotly objects).

    Raises
    ------
    ImportError
        If the ``[viz]`` extra (plotly) is not installed.
    ValidationError
        If a forecast length does not match ``y_true``.
    """
    raise NotImplementedError


def error_figure(
    rmse_by_model: Mapping[str, float],
    directional_acc_by_model: Mapping[str, float],
    *,
    title: str = "RMSE and directional accuracy by model",
) -> dict[str, Any]:
    """Build the per-model error bar Plotly figure as a ``{data, layout}`` dict.

    LAZY IMPORT: ``plotly`` is imported inside this function (the ``[viz]`` extra).
    Draws a grouped bar chart of return-space RMSE and directional accuracy keyed
    by model, then serializes to a plain dict.

    Parameters
    ----------
    rmse_by_model:
        Map of model name -> return-space OOS RMSE.
    directional_acc_by_model:
        Map of model name -> directional accuracy.
    title:
        Figure title.

    Returns
    -------
    dict[str, Any]
        A Plotly ``{data, layout}`` dict (no plotly objects).

    Raises
    ------
    ImportError
        If the ``[viz]`` extra (plotly) is not installed.
    ValidationError
        If the two maps do not share the same model keys.
    """
    raise NotImplementedError


def _figure_to_dict(figure: Any) -> dict[str, Any]:
    """Serialize a Plotly ``Figure`` to a plain ``{data, layout}`` dict.

    Uses ``json.loads(pio.to_json(fig, validate=False))`` so the result is a plain
    JSON-safe dict carrying no plotly objects — exactly the form the API response
    embeds. ``plotly.io`` is imported lazily inside this helper.

    Parameters
    ----------
    figure:
        A ``plotly.graph_objects.Figure``.

    Returns
    -------
    dict[str, Any]
        The ``{data, layout}`` dict.
    """
    raise NotImplementedError


def _ordered_models(*maps: Mapping[str, Any]) -> Sequence[str]:
    """Return a stable, de-duplicated model ordering across one or more maps."""
    seen: list[str] = []
    for mapping in maps:
        for key in mapping:
            if key not in seen:
                seen.append(key)
    return seen
