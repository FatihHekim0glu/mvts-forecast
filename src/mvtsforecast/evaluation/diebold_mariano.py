"""Diebold-Mariano (1995) test of equal predictive accuracy vs. the naive baseline.

The DM test compares two forecasts' out-of-sample squared-error losses. With
per-observation losses ``e_model^2`` and ``e_naive^2``, the loss differential
``d_t = e_model_t^2 - e_naive_t^2`` has mean ``d_bar``; the DM statistic is
``d_bar / HAC_SE(d)``, asymptotically standard normal under the null of equal
accuracy. A NEGATIVE statistic with a small p-value means the model beats the
naive random walk; a p-value ``>= alpha`` means the difference is INSIGNIFICANT
(the honest NULL on noisy daily returns).

The HAC standard error of the loss-differential mean uses a Newey-West Bartlett
long-run variance with the Andrews automatic lag — reused from
:func:`mvtsforecast.evaluation.metrics.hac_standard_error`.

Importing this module has no side effects.
"""

from __future__ import annotations

from mvtsforecast._typing import FloatArray

# quantcore-candidate: mirrors pairs-trading:evaluation/hac.py +
# lstm-forecast:evaluation/metrics.py::diebold_mariano.


def diebold_mariano(
    y_true: FloatArray,
    y_pred_model: FloatArray,
    y_pred_naive: FloatArray | None = None,
    *,
    lag: int | None = None,
) -> tuple[float, float]:
    r"""Diebold-Mariano (1995) test of equal predictive accuracy vs. the random walk.

    With per-observation squared-error losses ``e_model^2`` and ``e_naive^2``, the
    loss differential ``d_t = e_model_t^2 - e_naive_t^2`` has mean ``d_bar``; the
    DM statistic is ``d_bar / HAC_SE(d)``, asymptotically standard normal under
    the null of equal accuracy. A NEGATIVE statistic with a small p-value means
    the model beats the naive baseline; a p-value ``>= alpha`` means the
    difference is insignificant (the honest NULL on noisy daily returns).

    Parameters
    ----------
    y_true:
        Realized next-step returns.
    y_pred_model:
        The model's forecasts.
    y_pred_naive:
        The naive forecasts; defaults to all zeros (the random walk).
    lag:
        HAC Bartlett lag; ``None`` => Andrews automatic rule.

    Returns
    -------
    tuple[float, float]
        ``(dm_statistic, two_sided_pvalue)``. A negative statistic favours the
        model; the p-value is clipped to ``[0, 1]``.

    Raises
    ------
    ValidationError
        If inputs are empty/mismatched, or the loss-differential HAC variance is
        zero with a non-zero mean (the statistic is undefined).
    """
    raise NotImplementedError


def dm_favours_model(dm_statistic: float, dm_pvalue: float, *, alpha: float = 0.05) -> bool:
    """Return ``True`` iff DM is significant AND signed in the model's favour.

    The model beats the naive baseline only when the two-sided p-value clears the
    significance threshold (``dm_pvalue < alpha``) AND the statistic is strictly
    negative (lower squared-error loss than the naive forecast). This is the
    DM-side of the pure :func:`mvtsforecast.evaluation.verdict.derive_verdict`
    gate.

    Parameters
    ----------
    dm_statistic:
        The Diebold-Mariano statistic (negative favours the model).
    dm_pvalue:
        The two-sided DM p-value.
    alpha:
        Significance level (default ``0.05``).

    Returns
    -------
    bool
        ``True`` iff ``dm_pvalue < alpha and dm_statistic < 0``.
    """
    raise NotImplementedError
