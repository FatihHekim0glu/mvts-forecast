"""Smoke tests for the seeded synthetic multivariate generator (the honest-null DGP).

These exercise the load-bearing, already-implemented parts: reproducibility,
shape/columns, and the persistence-floor helper. The remaining generators
(``weak_factor_panel``, ``random_walk_panel``) are filled in by a later author and
their tests live alongside.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mvtsforecast._exceptions import ValidationError
from mvtsforecast.data.synthetic import (
    DEFAULT_BASKET,
    random_walk_panel,
    synthetic_panel,
    weak_factor_panel,
)
from mvtsforecast.models.naive import persistence_returns


def _mean_abs_lag1_autocorr(panel: pd.DataFrame) -> float:
    """Mean absolute lag-1 autocorrelation across the panel's columns."""
    acs = [panel[col].autocorr(lag=1) for col in panel.columns]
    return float(np.mean(np.abs(acs)))


@pytest.mark.unit
def test_synthetic_panel_is_reproducible() -> None:
    """The same ``(seed, basket, n_obs)`` yields a byte-identical panel."""
    a = synthetic_panel(DEFAULT_BASKET, n_obs=200, seed=7)
    b = synthetic_panel(DEFAULT_BASKET, n_obs=200, seed=7)
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.unit
def test_synthetic_panel_shape_and_columns() -> None:
    """The panel has the requested shape, columns, and a business-day index."""
    panel = synthetic_panel(("SPY", "TLT", "GLD"), n_obs=128, seed=7)
    assert panel.shape == (128, 3)
    assert list(panel.columns) == ["SPY", "TLT", "GLD"]
    assert panel.index.is_monotonic_increasing
    assert np.isfinite(panel.to_numpy()).all()


@pytest.mark.unit
def test_different_seeds_give_different_panels() -> None:
    """Distinct seeds produce distinct realizations (the RNG is actually used)."""
    a = synthetic_panel(DEFAULT_BASKET, n_obs=200, seed=7)
    b = synthetic_panel(DEFAULT_BASKET, n_obs=200, seed=8)
    assert not np.allclose(a.to_numpy(), b.to_numpy())


@pytest.mark.unit
def test_persistence_returns_is_the_zero_floor() -> None:
    """The naive random-walk return forecast is an all-zeros vector."""
    forecast = persistence_returns(50)
    assert forecast.shape == (50,)
    assert np.array_equal(forecast, np.zeros(50))


# --------------------------------------------------------------------------- #
# synthetic_panel: target autocorrelation                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_synthetic_panel_has_target_autocorrelation() -> None:
    """The AR(1) recursion induces ~``ar1`` lag-1 autocorrelation on average."""
    ar1 = 0.30
    panel = synthetic_panel(
        DEFAULT_BASKET, n_obs=6000, seed=7, ar1=ar1, idio_vol=0.010, factor_vol=0.0
    )
    # With no common factor, each series is a pure AR(1); the empirical lag-1
    # autocorrelation should be close to the target coefficient.
    estimated = _mean_abs_lag1_autocorr(panel)
    assert abs(estimated - ar1) < 0.05


@pytest.mark.unit
def test_synthetic_panel_is_near_white_noise_by_default() -> None:
    """The shipped default keeps the predictable component far below the floor."""
    panel = synthetic_panel(DEFAULT_BASKET, n_obs=4000, seed=7)
    # Default ar1=0.02 — the panel is effectively a random walk in return space.
    assert _mean_abs_lag1_autocorr(panel) < 0.10


@pytest.mark.unit
def test_synthetic_panel_has_cross_sectional_correlation() -> None:
    """The common factor induces positive average pairwise correlation."""
    panel = synthetic_panel(DEFAULT_BASKET, n_obs=4000, seed=7)
    corr = panel.corr().to_numpy()
    off_diag = corr[~np.eye(corr.shape[0], dtype=bool)]
    assert float(np.mean(off_diag)) > 0.05


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"basket": []}, "non-empty"),
        ({"basket": ["SPY", "SPY"]}, "duplicates"),
        ({"n_obs": 1}, "n_obs"),
        ({"ar1": 1.0}, "ar1"),
        ({"factor_vol": -0.1}, "non-negative"),
        ({"idio_vol": -0.1}, "non-negative"),
    ],
)
def test_synthetic_panel_validation(kwargs: dict[str, object], match: str) -> None:
    """Bad parameters raise a clear :class:`ValidationError`."""
    with pytest.raises(ValidationError, match=match):
        synthetic_panel(**kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# weak_factor_panel                                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_weak_factor_panel_shape_and_reproducible() -> None:
    """Weak-factor panel has the requested shape and is byte-reproducible."""
    a = weak_factor_panel(DEFAULT_BASKET, n_obs=256, seed=7)
    b = weak_factor_panel(DEFAULT_BASKET, n_obs=256, seed=7)
    assert a.shape == (256, 3)
    assert list(a.columns) == list(DEFAULT_BASKET)
    pd.testing.assert_frame_equal(a, b)
    assert np.isfinite(a.to_numpy()).all()


@pytest.mark.unit
def test_weak_factor_has_weaker_correlation_than_default() -> None:
    """The 'weak factor' panel is less cross-correlated than the default DGP."""

    def avg_abs_offdiag(panel: pd.DataFrame) -> float:
        corr = panel.corr().to_numpy()
        return float(np.mean(np.abs(corr[~np.eye(corr.shape[0], dtype=bool)])))

    weak = avg_abs_offdiag(weak_factor_panel(DEFAULT_BASKET, n_obs=4000, seed=7))
    default = avg_abs_offdiag(synthetic_panel(DEFAULT_BASKET, n_obs=4000, seed=7))
    assert weak < default


@pytest.mark.unit
def test_weak_factor_propagates_validation() -> None:
    """Validation errors propagate up from :func:`synthetic_panel`."""
    with pytest.raises(ValidationError, match="non-empty"):
        weak_factor_panel([])


# --------------------------------------------------------------------------- #
# random_walk_panel                                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_random_walk_panel_shape_columns_reproducible() -> None:
    """Random-walk panel has the requested shape and is byte-reproducible."""
    a = random_walk_panel(("SPY", "TLT", "GLD"), n_obs=300, seed=7)
    b = random_walk_panel(("SPY", "TLT", "GLD"), n_obs=300, seed=7)
    assert a.shape == (300, 3)
    assert list(a.columns) == ["SPY", "TLT", "GLD"]
    pd.testing.assert_frame_equal(a, b)
    assert np.isfinite(a.to_numpy()).all()


@pytest.mark.unit
def test_random_walk_panel_is_serially_uncorrelated() -> None:
    """i.i.d. noise => near-zero lag-1 autocorrelation (naive is the floor)."""
    panel = random_walk_panel(DEFAULT_BASKET, n_obs=6000, seed=7)
    assert _mean_abs_lag1_autocorr(panel) < 0.05


@pytest.mark.unit
def test_random_walk_panel_is_cross_sectionally_uncorrelated() -> None:
    """No common factor => negligible average pairwise correlation."""
    panel = random_walk_panel(DEFAULT_BASKET, n_obs=6000, seed=7)
    corr = panel.corr().to_numpy()
    off_diag = corr[~np.eye(corr.shape[0], dtype=bool)]
    assert float(np.mean(np.abs(off_diag))) < 0.05


@pytest.mark.unit
def test_random_walk_different_seeds_differ() -> None:
    """Distinct seeds produce distinct realizations."""
    a = random_walk_panel(DEFAULT_BASKET, n_obs=200, seed=7)
    b = random_walk_panel(DEFAULT_BASKET, n_obs=200, seed=8)
    assert not np.allclose(a.to_numpy(), b.to_numpy())


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"basket": []}, "non-empty"),
        ({"basket": ["SPY", "SPY"]}, "duplicates"),
        ({"n_obs": 1}, "n_obs"),
        ({"idio_vol": -0.01}, "idio_vol"),
    ],
)
def test_random_walk_panel_validation(kwargs: dict[str, object], match: str) -> None:
    """Bad parameters raise a clear :class:`ValidationError`."""
    with pytest.raises(ValidationError, match=match):
        random_walk_panel(**kwargs)  # type: ignore[arg-type]
