"""Real-data loaders: yfinance -> Stooq prices + FRED-CSV macro (offline CLI path).

The default, deployed data path is the seeded synthetic generator
(:mod:`mvtsforecast.data.synthetic`) — no API keys, no survivorship questions, and
the honest NULL holds by construction. This module is the OFFLINE CLI path for
real data:

- :func:`load_prices` fetches a price panel via yfinance (curl_cffi Chrome
  impersonation) with a Stooq fallback, computes returns with
  ``pct_change(fill_method=None)``, and reports the resolved
  :data:`DataSource`;
- :func:`load_macro` loads FRED series from committed CSVs and lags each feature
  to its RELEASE date (never its reference date) so a macro feature observed at
  time ``t`` only uses information actually published by ``t``.

Heavy data dependencies (yfinance, curl_cffi, pyarrow, diskcache,
pandas-datareader) live behind the ``data`` extra and are imported LAZILY inside
these functions, so importing this module pulls in nothing heavy and has no side
effects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

# quantcore-candidate: mirrors hrp-portfolio:data.py (yfinance->stooq fallback)
# and lstm-forecast:data.py (CSV loader), extended with FRED release-date lags.

#: Where a price/return panel ultimately came from. Returned alongside the data so
#: callers (and the API ``data_source`` field) can report provenance.
DataSource = Literal["yfinance", "stooq", "synthetic", "cache"]


def load_prices(
    basket: list[str],
    *,
    start: str = "2015-01-01",
    end: str | None = None,
    cache_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, DataSource]:
    """Load a real price panel via yfinance, falling back to Stooq, then cache.

    LAZY IMPORTS: ``yfinance``/``curl_cffi``/``pandas-datareader``/``diskcache`` are
    imported inside this function (the ``data`` extra), so importing this module
    is cheap and side-effect-free. Prices are forward-filled across non-trading
    gaps but returns are computed with ``pct_change(fill_method=None)`` upstream so
    no synthetic zero-returns are injected.

    Parameters
    ----------
    basket:
        Tickers to fetch.
    start, end:
        Inclusive date span (``end=None`` => today).
    cache_dir:
        Optional diskcache directory; ``None`` disables on-disk caching.

    Returns
    -------
    tuple[pandas.DataFrame, DataSource]
        The wide price panel and the resolved data-source label.

    Raises
    ------
    ValidationError
        If ``basket`` is empty or the fetched panel is unusable.
    """
    raise NotImplementedError


def load_macro(
    csv_paths: dict[str, str | Path],
    *,
    index: pd.DatetimeIndex,
    release_lag_days: int = 1,
) -> pd.DataFrame:
    """Load FRED macro series from CSV, lagged to RELEASE dates (not reference dates).

    Each FRED CSV is read, its reference-date observations are shifted forward to
    the date the figure was actually PUBLISHED (reference date + ``release_lag_days``
    business days, a conservative proxy), and the result is reindexed onto
    ``index`` with forward-fill. This prevents the classic macro look-ahead bug
    where, e.g., a GDP figure stamped with the quarter-end is used on the
    quarter-end even though it is published weeks later.

    Parameters
    ----------
    csv_paths:
        Mapping of feature name -> path to a FRED ``DATE,VALUE`` CSV.
    index:
        The trading-day index to align the macro features onto.
    release_lag_days:
        Business-day lag applied to each reference date to approximate the
        release date.

    Returns
    -------
    pandas.DataFrame
        A macro feature panel aligned to ``index``, release-date-lagged.

    Raises
    ------
    ValidationError
        If a CSV is missing/malformed or ``release_lag_days`` is negative.
    """
    raise NotImplementedError


def returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert a wide price panel to simple returns with ``pct_change(fill_method=None)``.

    Uses ``fill_method=None`` so genuine gaps stay NaN (and are dropped) rather
    than being silently forward-filled into fake zero-returns — a subtle leakage /
    bias source. The first row (all-NaN) is dropped.

    Parameters
    ----------
    prices:
        A wide, time-indexed price panel.

    Returns
    -------
    pandas.DataFrame
        The aligned simple-returns panel (one row shorter than ``prices``).

    Raises
    ------
    ValidationError
        If ``prices`` is empty or has fewer than two rows.
    """
    raise NotImplementedError
