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

from mvtsforecast.data.synthetic import DEFAULT_BASKET, synthetic_panel
from mvtsforecast.models.naive import persistence_returns


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
