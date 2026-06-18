"""Shared, SEEDED pytest fixtures for the mvts-forecast test suite.

Every fixture is deterministic (fixed seed) so the partitioned tests
(unit/parity/property/regression/integration) are reproducible across runs and
across the CI matrix. The fixtures deliberately span the honest-null spectrum:

- ``synthetic_panel`` — the deployed default DGP: a small basket of correlated
  return series (weak common factor + idiosyncratic noise + mild AR(1)) where the
  deep models cannot reliably beat naive by construction;
- ``random_walk`` — the strictest testbed: i.i.d. Gaussian returns, where naive is
  provably the OOS floor (drives the random-walk anti-leakage regression test);
- ``weak_factor`` — a panel whose only cross-sectional structure is a deliberately
  weak common factor.

These import only the import-pure parts of the package (no torch / onnxruntime).
"""

from __future__ import annotations

import pandas as pd
import pytest

from mvtsforecast.data import synthetic as _synth
from mvtsforecast.data.synthetic import DEFAULT_BASKET

#: Seed shared across fixtures so every panel is reproducible.
SEED: int = 7

#: Small basket + modest length keep the fixtures fast while still exercising the
#: multivariate windowing/walk-forward paths.
BASKET: tuple[str, ...] = DEFAULT_BASKET
N_OBS: int = 400


@pytest.fixture
def synthetic_panel() -> pd.DataFrame:
    """A seeded correlated multivariate RETURNS panel (the deployed-default DGP).

    A weak common factor + dominant idiosyncratic noise + mild AR(1) over the
    default basket. By construction the next-step return is dominated by
    unforecastable noise, so the honest NULL holds.
    """
    return _synth.synthetic_panel(BASKET, n_obs=N_OBS, seed=SEED)


@pytest.fixture
def random_walk() -> pd.DataFrame:
    """A seeded i.i.d.-Gaussian RETURNS panel (the strictest honest-null testbed).

    Each series is independent noise, so the next-step return is genuinely
    unpredictable and the naive forecast is provably the OOS floor. Drives the
    random-walk anti-leakage regression test (deep must NOT beat naive).
    """
    return _synth.random_walk_panel(BASKET, n_obs=N_OBS, seed=SEED)


@pytest.fixture
def weak_factor() -> pd.DataFrame:
    """A seeded panel whose only cross-sectional structure is a WEAK common factor.

    Cross-sectional correlation is present but small, so even a real (but tiny)
    common signal does not let a deep model beat naive after costs.
    """
    return _synth.weak_factor_panel(BASKET, n_obs=N_OBS, seed=SEED)
