"""Purge-aware sliding windows and the anchored/expanding walk-forward split.

This module builds the supervised learning problem for the multivariate encoders
and enforces, per fold, the leakage guards a leaky stock-price repo lacks:

1. **Sliding windows** — :func:`make_windows` turns a wide returns panel into a
   ``(n_samples, look_back, n_features)`` sequence tensor and its aligned
   ``(n_samples,)`` next-step TARGET vector. The window at sample ``i`` spans
   rows ``[i, i + look_back)`` and predicts row ``i + look_back`` of the target.
2. **No-target-in-features** — :func:`make_windows` excludes the target column's
   same-step transform from the encoder input when ``drop_target_feature=True``,
   so the window can never read the value it is trying to predict.
3. **Purge (>= look_back) + embargo** — :func:`make_folds` removes a gap of at
   least ``look_back`` rows at every train/test boundary so no ``look_back``-length
   window can straddle the split, plus an embargo after each test block.

The standardizer (see :func:`fit_standardizer`) is fitted on the TRAIN fold only
and APPLIED (never re-fitted) to the test fold — the headline fix for the
full-series-scaler leakage bug. The RevIN instance-norm is, separately, computed
from each input window only (see :mod:`mvtsforecast.windowing.revin`).

Importing this module has no side effects.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from mvtsforecast._typing import FloatArray, SequenceTensor

# quantcore-candidate: purge/embargo mirror pairs-trading:evaluation/_purge.py +
# lstm-forecast:walkforward/engine.py, generalized to the multivariate setting.


@dataclass(frozen=True, slots=True)
class WindowSpec:
    """Immutable specification of the supervised windowing problem.

    Attributes
    ----------
    look_back:
        Sequence window length (default 60); also the MINIMUM purge size.
    horizon:
        Forecast horizon in steps ahead (1 or 5); the target is the row
        ``look_back + horizon - 1`` ahead's return.
    target:
        Column label of the forecast target within the panel.
    feature_columns:
        Ordered feature column labels fed to the encoder (the target's same-step
        transform is excluded when ``drop_target_feature`` is set).
    drop_target_feature:
        If ``True``, the target column is excluded from the encoder input so the
        window cannot read the value it predicts.
    """

    look_back: int = 60
    horizon: int = 1
    target: str = "SPY"
    feature_columns: tuple[str, ...] = ()
    drop_target_feature: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this spec."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Fold:
    """Immutable index ranges for one purged, embargoed walk-forward fold.

    All bounds are half-open row positions into the windowed-sample axis.

    Attributes
    ----------
    train_start, train_end:
        Train-slice sample bounds ``[train_start, train_end)``.
    test_start, test_end:
        Test-slice sample bounds ``[test_start, test_end)`` (already past the
        purge gap after ``train_end``).
    """

    train_start: int
    train_end: int
    test_start: int
    test_end: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this fold."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Standardizer:
    """Immutable per-feature mean/std fitted on a TRAIN slice only.

    Attributes
    ----------
    mean:
        Per-feature means, shape ``(n_features,)``.
    std:
        Per-feature standard deviations (floored at ``EPS``), shape
        ``(n_features,)``.
    feature_columns:
        The feature column labels, in order, the statistics correspond to.
    """

    mean: FloatArray
    std: FloatArray
    feature_columns: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this standardizer."""
        return {
            "mean": [float(x) for x in np.asarray(self.mean).ravel()],
            "std": [float(x) for x in np.asarray(self.std).ravel()],
            "feature_columns": list(self.feature_columns),
        }

    def transform(self, tensor: SequenceTensor) -> SequenceTensor:
        """Standardize a ``(n, look_back, n_features)`` tensor with fitted stats.

        Parameters
        ----------
        tensor:
            A 3-D sequence tensor whose last axis matches ``feature_columns``.

        Returns
        -------
        SequenceTensor
            ``(tensor - mean) / std`` broadcast over the feature axis.

        Raises
        ------
        ValidationError
            If the tensor's feature axis does not match the fitted statistics.
        """
        raise NotImplementedError


def make_windows(
    panel: pd.DataFrame,
    spec: WindowSpec,
) -> tuple[SequenceTensor, FloatArray]:
    """Build the ``(X, y)`` supervised windows from a wide returns panel.

    The window at sample ``i`` spans panel rows ``[i, i + look_back)`` over the
    feature columns and predicts the target column at row
    ``i + look_back + horizon - 1``. When ``spec.drop_target_feature`` is set, the
    target column's same-step transform is excluded from ``X`` so the encoder
    input can never contain the value it is trying to forecast.

    Parameters
    ----------
    panel:
        A wide, time-indexed RETURNS panel (rows = time, columns = assets).
    spec:
        The :class:`WindowSpec` describing look-back, horizon, target, and which
        feature columns to feed the encoder.

    Returns
    -------
    tuple[SequenceTensor, FloatArray]
        ``X`` of shape ``(n_samples, look_back, n_features)`` and ``y`` of shape
        ``(n_samples,)`` (the next-``horizon``-step target return).

    Raises
    ------
    ValidationError
        If the target/feature columns are missing, ``look_back`` or ``horizon``
        is non-positive, or the target appears in the feature set while
        ``drop_target_feature`` is ``False`` and a leakage guard is active.
    InsufficientDataError
        If the panel has fewer than ``look_back + horizon`` rows (no window).
    """
    raise NotImplementedError


def assert_no_target_leakage(spec: WindowSpec) -> None:
    """Assert the encoder input cannot read the target's same-step transform.

    When ``spec.drop_target_feature`` is ``True`` the target column must NOT be
    present in ``spec.feature_columns``. This is the explicit, testable guard
    against the "target-in-features" footgun.

    Parameters
    ----------
    spec:
        The window specification to check.

    Raises
    ------
    ValidationError
        If ``drop_target_feature`` is ``True`` but the target column is still
        listed in ``feature_columns``.
    """
    raise NotImplementedError


def make_folds(
    n_samples: int,
    *,
    look_back: int = 60,
    n_folds: int = 3,
    test_size: int | None = None,
    embargo: int = 5,
    anchored: bool = True,
) -> list[Fold]:
    """Build anchored/expanding walk-forward folds with purge + embargo.

    Each fold's train slice is followed by a purge gap of at least ``look_back``
    samples (so no window straddles the boundary), then the test slice, then an
    ``embargo`` gap before the next fold. With ``anchored=True`` the train slice
    always starts at sample 0 and expands; otherwise it rolls forward.

    Parameters
    ----------
    n_samples:
        Total number of windowed samples (the output length of
        :func:`make_windows`).
    look_back:
        The window length; the purge gap is clamped up to this value.
    n_folds:
        Number of out-of-sample test blocks.
    test_size:
        Samples per test block; ``None`` => an even split of the post-warmup tail.
    embargo:
        Extra gap after each test block.
    anchored:
        If ``True``, the train slice anchors at sample 0 and expands; else rolls.

    Returns
    -------
    list[Fold]
        The ordered, non-overlapping folds.

    Raises
    ------
    ValidationError
        If any size is non-positive or ``look_back`` is negative.
    InsufficientDataError
        If there are too few samples to form ``n_folds`` purged test blocks.
    """
    raise NotImplementedError


def fit_standardizer(
    x_train: SequenceTensor,
    *,
    feature_columns: Sequence[str] = (),
) -> Standardizer:
    """Fit a per-feature standardizer on the TRAIN windows only.

    Computes the per-feature mean and (EPS-floored) standard deviation over all
    train-fold windows and time-steps. The returned :class:`Standardizer` is then
    APPLIED — never re-fitted — to the test fold, which is the de-leak fix for the
    classic full-series-scaler bug.

    Parameters
    ----------
    x_train:
        The train-fold sequence tensor ``(n_train, look_back, n_features)``.
    feature_columns:
        Optional feature labels recorded on the standardizer for provenance.

    Returns
    -------
    Standardizer
        The frozen mean/std fitted on the train fold only.

    Raises
    ------
    ValidationError
        If ``x_train`` is not 3-D or is empty.
    """
    raise NotImplementedError
