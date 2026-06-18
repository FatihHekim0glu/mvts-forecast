"""RevIN / reversible instance normalization — leakage-safe BY CONSTRUCTION.

Reversible Instance Normalization (Kim et al., 2022) normalizes each input window
using ONLY that window's own statistics, then de-normalizes the forecast with the
same statistics. Because the mean/std come exclusively from the look-back window
(never from the train fold, never from any future row), RevIN cannot leak: it is
a per-window, causal transform.

This module is the explicit, testable counterpart to the future-perturbation
property test — normalizing window ``i`` depends on rows ``[i, i + look_back)``
ONLY, so altering rows at ``i + look_back ..`` cannot change the normalized input
or the de-normalized forecast at ``i``.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from mvtsforecast._typing import FloatArray, SequenceTensor


@dataclass(frozen=True, slots=True)
class RevInStats:
    """Per-window, per-feature location/scale captured for reversible de-norm.

    Attributes
    ----------
    mean:
        Per-window per-feature means, shape ``(n_samples, 1, n_features)``.
    std:
        Per-window per-feature standard deviations (EPS-floored), same shape.
    """

    mean: FloatArray
    std: FloatArray

    def to_dict(self) -> dict[str, Any]:
        """Return a plain ``dict`` of the captured statistics (shapes preserved)."""
        return asdict(self)


def revin_normalize(
    windows: SequenceTensor,
    *,
    eps: float | None = None,
) -> tuple[SequenceTensor, RevInStats]:
    r"""Normalize each window with ITS OWN statistics (leakage-safe).

    For each sample ``i`` and feature ``f``, subtract the within-window mean and
    divide by the within-window standard deviation computed over the ``look_back``
    time axis ONLY:

    .. math::

        \tilde{x}_{i,t,f} = \frac{x_{i,t,f} - \mu_{i,f}}{\sigma_{i,f} + \epsilon},
        \quad \mu_{i,f}, \sigma_{i,f} \text{ over } t \in [0, \text{look\_back}).

    The returned :class:`RevInStats` lets :func:`revin_denormalize` invert the
    transform on the model's forecast. NO statistic depends on any row outside
    the window, so this transform is causal and leakage-free by construction.

    Parameters
    ----------
    windows:
        A ``(n_samples, look_back, n_features)`` sequence tensor.
    eps:
        Numerical floor added to each window std; ``None`` => the project ``EPS``.

    Returns
    -------
    tuple[SequenceTensor, RevInStats]
        The normalized tensor (same shape) and the per-window statistics needed
        to reverse it.

    Raises
    ------
    ValidationError
        If ``windows`` is not a 3-D tensor or is empty.
    """
    raise NotImplementedError


def revin_denormalize(
    normalized_forecast: FloatArray,
    stats: RevInStats,
    *,
    feature_index: int = 0,
) -> FloatArray:
    r"""Invert the RevIN transform on a per-window forecast.

    Maps a normalized forecast back to return space with the window's own
    statistics: :math:`\hat{y}_i = \tilde{y}_i \,\sigma_{i,f} + \mu_{i,f}` for the
    selected target feature ``f`` = ``feature_index``.

    Parameters
    ----------
    normalized_forecast:
        A ``(n_samples,)`` forecast in the normalized space.
    stats:
        The :class:`RevInStats` returned by :func:`revin_normalize`.
    feature_index:
        Index of the target feature whose location/scale reverse the forecast.

    Returns
    -------
    FloatArray
        The ``(n_samples,)`` forecast back in return space.

    Raises
    ------
    ValidationError
        If the forecast length does not match the captured statistics or
        ``feature_index`` is out of range.
    """
    raise NotImplementedError
