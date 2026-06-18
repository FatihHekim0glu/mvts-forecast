"""Tests for the real-data loaders (offline-safe; synthetic fallback default).

The ``data`` extra (yfinance / curl_cffi / pandas-datareader) is NOT assumed to be
installed, so these tests exercise the OFFLINE behaviour: ``load_prices`` falls back
to the deterministic synthetic panel, ``returns_from_prices`` is no-lookahead-safe,
and ``load_macro`` lags each FRED feature to its RELEASE date (never its reference
date). Importing the module must pull in nothing heavy.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mvtsforecast._exceptions import ValidationError
from mvtsforecast.data.loaders import (
    _extract_close,
    load_macro,
    load_prices,
    returns_from_prices,
)


# --------------------------------------------------------------------------- #
# load_prices: offline synthetic fallback                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_load_prices_falls_back_to_synthetic_offline() -> None:
    """With no network and no ``data`` extra, load_prices yields synthetic prices."""
    prices, source = load_prices(["SPY", "TLT", "GLD"])
    assert source == "synthetic"
    assert list(prices.columns) == ["SPY", "TLT", "GLD"]
    assert prices.shape[0] > 1
    # Compounded from returns: strictly positive and finite.
    assert np.isfinite(prices.to_numpy()).all()
    assert (prices.to_numpy() > 0.0).all()


@pytest.mark.unit
def test_load_prices_synthetic_fallback_is_reproducible() -> None:
    """The synthetic fallback is deterministic across calls."""
    a, _ = load_prices(["SPY", "TLT"])
    b, _ = load_prices(["SPY", "TLT"])
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.unit
def test_load_prices_rejects_empty_basket() -> None:
    """An empty basket is a validation error before any fetch is attempted."""
    with pytest.raises(ValidationError, match="non-empty"):
        load_prices([])


@pytest.mark.unit
def test_load_prices_accepts_cache_dir_noop(tmp_path: Path) -> None:
    """``cache_dir`` is accepted for API parity and is a no-op offline."""
    prices, source = load_prices(["SPY"], cache_dir=tmp_path)
    assert source == "synthetic"
    assert not prices.empty


# --------------------------------------------------------------------------- #
# returns_from_prices: no-lookahead                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_returns_from_prices_matches_pct_change() -> None:
    """Returns equal pct_change(fill_method=None) with the leading NaN row dropped."""
    idx = pd.bdate_range("2020-01-01", periods=5)
    prices = pd.DataFrame({"A": [100.0, 101.0, 99.0, 102.0, 103.0]}, index=idx)
    returns = returns_from_prices(prices)
    assert returns.shape == (4, 1)
    expected = prices["A"].pct_change(fill_method=None).iloc[1:].to_numpy()
    np.testing.assert_allclose(returns["A"].to_numpy(), expected)


@pytest.mark.unit
def test_returns_from_prices_does_not_forward_fill_gaps() -> None:
    """A genuine price gap stays NaN rather than becoming a fake zero return."""
    idx = pd.bdate_range("2020-01-01", periods=4)
    prices = pd.DataFrame({"A": [100.0, np.nan, 102.0, 103.0]}, index=idx)
    returns = returns_from_prices(prices)
    # The return across the NaN gap must be NaN (no ffill-then-diff), not 0.0.
    assert bool(np.isnan(returns["A"].iloc[0]))


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad",
    [
        pd.DataFrame({"A": [1.0]}),  # one row
        pd.DataFrame(index=pd.bdate_range("2020-01-01", periods=3)),  # no columns
    ],
)
def test_returns_from_prices_validation(bad: pd.DataFrame) -> None:
    """Too few rows or no columns raises a validation error."""
    with pytest.raises(ValidationError, match="at least two rows"):
        returns_from_prices(bad)


@pytest.mark.unit
def test_returns_from_prices_rejects_non_dataframe() -> None:
    """A non-DataFrame input is rejected."""
    with pytest.raises(ValidationError, match="DataFrame"):
        returns_from_prices([1.0, 2.0])  # type: ignore[arg-type]


@pytest.mark.unit
def test_load_prices_then_returns_roundtrip() -> None:
    """The fallback price panel converts cleanly to a finite returns panel."""
    prices, _ = load_prices(["SPY", "TLT", "GLD"])
    returns = returns_from_prices(prices)
    assert returns.shape[0] == prices.shape[0] - 1
    assert np.isfinite(returns.to_numpy()).all()


# --------------------------------------------------------------------------- #
# load_macro: RELEASE-date lags (no look-ahead)                               #
# --------------------------------------------------------------------------- #
def _write_fred_csv(path: Path, dates: list[str], values: list[float]) -> None:
    """Write a minimal FRED-style ``DATE,VALUE`` CSV."""
    pd.DataFrame({"DATE": dates, "VALUE": values}).to_csv(path, index=False)


@pytest.mark.unit
def test_load_macro_lags_to_release_date(tmp_path: Path) -> None:
    """A reference-date figure is only visible on/after its release date."""
    csv = tmp_path / "gdp.csv"
    # Reference date is a Wednesday; with a 2-business-day release lag the figure
    # is published on the following Friday.
    _write_fred_csv(csv, ["2021-01-06"], [5.0])
    index = pd.bdate_range("2021-01-04", periods=8)  # Mon Jan 4 .. Wed Jan 13
    macro = load_macro({"gdp": csv}, index=index, release_lag_days=2)

    release_date = pd.Timestamp("2021-01-08")  # 2 business days after Jan 6 (Wed)
    # Before release: NaN (no look-ahead). On/after release: the value.
    pre = macro.loc[macro.index < release_date, "gdp"]
    post = macro.loc[macro.index >= release_date, "gdp"]
    assert pre.isna().all()
    assert (post == 5.0).all()


@pytest.mark.unit
def test_load_macro_forward_fills_between_releases(tmp_path: Path) -> None:
    """The latest released value carries forward until the next release."""
    csv = tmp_path / "rate.csv"
    _write_fred_csv(csv, ["2021-01-04", "2021-01-11"], [1.0, 2.0])
    index = pd.bdate_range("2021-01-04", periods=12)
    macro = load_macro({"rate": csv}, index=index, release_lag_days=1)
    # After the first release the value is 1.0, then 2.0 after the second.
    assert macro["rate"].dropna().iloc[0] == 1.0
    assert macro["rate"].iloc[-1] == 2.0
    # Monotone non-decreasing once both are released here.
    released = macro["rate"].dropna()
    assert set(released.unique()) <= {1.0, 2.0}


@pytest.mark.unit
def test_load_macro_zero_lag_aligns_on_reference_date(tmp_path: Path) -> None:
    """A zero release lag aligns each value on its own reference date."""
    csv = tmp_path / "x.csv"
    _write_fred_csv(csv, ["2021-01-05"], [7.0])
    index = pd.bdate_range("2021-01-04", periods=5)
    macro = load_macro({"x": csv}, index=index, release_lag_days=0)
    assert bool(np.isnan(macro.loc["2021-01-04", "x"]))
    assert macro.loc["2021-01-05", "x"] == 7.0


@pytest.mark.unit
def test_load_macro_accepts_fred_series_id_column(tmp_path: Path) -> None:
    """A FRED export named ``DATE,<SERIES_ID>`` (no 'VALUE') is parsed."""
    csv = tmp_path / "dgs3mo.csv"
    pd.DataFrame({"DATE": ["2021-01-04"], "DGS3MO": [0.09]}).to_csv(csv, index=False)
    index = pd.bdate_range("2021-01-04", periods=4)
    macro = load_macro({"dgs3mo": csv}, index=index, release_lag_days=0)
    assert macro.loc["2021-01-04", "dgs3mo"] == pytest.approx(0.09)


@pytest.mark.unit
def test_load_macro_multiple_features(tmp_path: Path) -> None:
    """Multiple CSVs become multiple aligned columns."""
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    _write_fred_csv(a, ["2021-01-04"], [1.0])
    _write_fred_csv(b, ["2021-01-04"], [2.0])
    index = pd.bdate_range("2021-01-04", periods=4)
    macro = load_macro({"a": a, "b": b}, index=index, release_lag_days=0)
    assert list(macro.columns) == ["a", "b"]
    assert macro.loc["2021-01-04", "a"] == 1.0
    assert macro.loc["2021-01-04", "b"] == 2.0


@pytest.mark.unit
def test_load_macro_negative_lag_rejected(tmp_path: Path) -> None:
    """A negative release lag is a validation error."""
    csv = tmp_path / "x.csv"
    _write_fred_csv(csv, ["2021-01-04"], [1.0])
    index = pd.bdate_range("2021-01-04", periods=3)
    with pytest.raises(ValidationError, match="release_lag_days"):
        load_macro({"x": csv}, index=index, release_lag_days=-1)


@pytest.mark.unit
def test_load_macro_missing_file_rejected(tmp_path: Path) -> None:
    """A missing CSV path is a validation error."""
    index = pd.bdate_range("2021-01-04", periods=3)
    with pytest.raises(ValidationError, match="file not found"):
        load_macro({"x": tmp_path / "nope.csv"}, index=index)


@pytest.mark.unit
def test_load_macro_missing_date_column_rejected(tmp_path: Path) -> None:
    """A CSV without a date column is rejected."""
    csv = tmp_path / "bad.csv"
    pd.DataFrame({"foo": [1.0], "bar": [2.0]}).to_csv(csv, index=False)
    index = pd.bdate_range("2021-01-04", periods=3)
    # 'foo' is taken as date and fails to parse -> a parse-time error surfaces.
    with pytest.raises((ValidationError, ValueError)):
        load_macro({"x": csv}, index=index)


@pytest.mark.unit
def test_load_macro_rejects_empty_inputs(tmp_path: Path) -> None:
    """Empty csv_paths or a non-DatetimeIndex are validation errors."""
    index = pd.bdate_range("2021-01-04", periods=3)
    with pytest.raises(ValidationError, match="non-empty"):
        load_macro({}, index=index)
    csv = tmp_path / "x.csv"
    _write_fred_csv(csv, ["2021-01-04"], [1.0])
    with pytest.raises(ValidationError, match="DatetimeIndex"):
        load_macro({"x": csv}, index=pd.Index([0, 1, 2]))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# _extract_close: provider-response normalization                             #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_extract_close_single_ticker_ohlcv() -> None:
    """A single-ticker OHLCV frame collapses to a one-column close panel."""
    idx = pd.to_datetime(["2021-01-04", "2021-01-05"])
    raw = pd.DataFrame({"Open": [1.0, 2.0], "Close": [1.5, 2.5], "Volume": [10, 20]}, index=idx)
    out = _extract_close(raw, ["SPY"])
    assert list(out.columns) == ["SPY"]
    np.testing.assert_allclose(out["SPY"].to_numpy(), [1.5, 2.5])


@pytest.mark.unit
def test_extract_close_multiindex_keeps_request_order() -> None:
    """A multi-ticker frame keeps only requested tickers, in request order."""
    idx = pd.to_datetime(["2021-01-04", "2021-01-05"])
    cols = pd.MultiIndex.from_product([["Close"], ["TLT", "SPY", "GLD"]])
    raw = pd.DataFrame(np.arange(6, dtype="float64").reshape(2, 3), index=idx, columns=cols)
    out = _extract_close(raw, ["SPY", "GLD"])
    assert list(out.columns) == ["SPY", "GLD"]


@pytest.mark.unit
def test_extract_close_prefers_adj_close_level() -> None:
    """When an ``Adj Close`` level is present it is preferred over ``Close``."""
    idx = pd.to_datetime(["2021-01-04", "2021-01-05"])
    cols = pd.MultiIndex.from_tuples([("Adj Close", "SPY"), ("Close", "SPY")])
    raw = pd.DataFrame([[10.0, 99.0], [11.0, 98.0]], index=idx, columns=cols)
    out = _extract_close(raw, ["SPY"])
    # The adjusted-close values (10, 11), not the raw close (99, 98), are used.
    np.testing.assert_allclose(out["SPY"].to_numpy(), [10.0, 11.0])


@pytest.mark.unit
def test_extract_close_multiindex_without_close_level_uses_first_level() -> None:
    """With no Close/Adj Close level, the first field level is cross-sectioned."""
    idx = pd.to_datetime(["2021-01-04", "2021-01-05"])
    cols = pd.MultiIndex.from_product([["Price"], ["SPY", "TLT"]])
    raw = pd.DataFrame(np.arange(4, dtype="float64").reshape(2, 2), index=idx, columns=cols)
    out = _extract_close(raw, ["SPY", "TLT"])
    assert list(out.columns) == ["SPY", "TLT"]


@pytest.mark.unit
def test_extract_close_flat_panel_passthrough() -> None:
    """A flat (already wide) panel with no ``Close`` column passes through."""
    idx = pd.to_datetime(["2021-01-04", "2021-01-05"])
    raw = pd.DataFrame({"SPY": [1.0, 2.0], "TLT": [3.0, 4.0]}, index=idx)
    out = _extract_close(raw, ["TLT", "SPY"])
    assert list(out.columns) == ["TLT", "SPY"]


# --------------------------------------------------------------------------- #
# fetchers via mocked lazy imports + the fallback chain                       #
# --------------------------------------------------------------------------- #
def _install_fake_module(monkeypatch: pytest.MonkeyPatch, name: str, module: object) -> None:
    """Register ``module`` under ``name`` so a lazy ``import name`` resolves to it."""
    import sys

    monkeypatch.setitem(sys.modules, name, module)


@pytest.mark.unit
def test_fetch_yfinance_with_mocked_download(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_fetch_yfinance`` normalizes a mocked yfinance response."""
    import types

    from mvtsforecast.data import loaders

    idx = pd.to_datetime(["2021-01-04", "2021-01-05"])
    cols = pd.MultiIndex.from_product([["Close"], ["SPY", "TLT"]])
    panel = pd.DataFrame(np.arange(4, dtype="float64").reshape(2, 2), index=idx, columns=cols)

    fake_yf = types.ModuleType("yfinance")
    fake_yf.download = lambda *a, **k: panel  # type: ignore[attr-defined]
    _install_fake_module(monkeypatch, "yfinance", fake_yf)
    # Force the curl_cffi branch to fail so the session=None path is taken.
    _install_fake_module(monkeypatch, "curl_cffi", types.ModuleType("curl_cffi"))

    out = loaders._fetch_yfinance(["SPY", "TLT"], "2021-01-01", "2021-01-10")
    assert list(out.columns) == ["SPY", "TLT"]


@pytest.mark.unit
def test_fetch_yfinance_raises_on_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """An all-NaN yfinance response raises (so the chain falls through)."""
    import types

    from mvtsforecast.data import loaders

    idx = pd.to_datetime(["2021-01-04"])
    empty = pd.DataFrame({"SPY": [np.nan]}, index=idx)
    fake_yf = types.ModuleType("yfinance")
    fake_yf.download = lambda *a, **k: empty  # type: ignore[attr-defined]
    _install_fake_module(monkeypatch, "yfinance", fake_yf)
    _install_fake_module(monkeypatch, "curl_cffi", types.ModuleType("curl_cffi"))

    with pytest.raises(ValueError, match="no usable price data"):
        loaders._fetch_yfinance(["SPY"], "2021-01-01", None)


@pytest.mark.unit
def test_fetch_stooq_with_mocked_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_fetch_stooq`` sorts the descending-date Stooq response ascending."""
    import types

    from mvtsforecast.data import loaders

    # Stooq returns most-recent-first; the fetcher must sort ascending.
    idx = pd.to_datetime(["2021-01-05", "2021-01-04"])
    panel = pd.DataFrame({"Close": [2.0, 1.0]}, index=idx)

    data_mod = types.ModuleType("pandas_datareader.data")
    data_mod.DataReader = lambda *a, **k: panel  # type: ignore[attr-defined]
    pkg = types.ModuleType("pandas_datareader")
    pkg.data = data_mod  # type: ignore[attr-defined]
    _install_fake_module(monkeypatch, "pandas_datareader", pkg)
    _install_fake_module(monkeypatch, "pandas_datareader.data", data_mod)

    out = loaders._fetch_stooq(["SPY"], "2021-01-01", "2021-01-10")
    assert out.index.is_monotonic_increasing
    np.testing.assert_allclose(out["SPY"].to_numpy(), [1.0, 2.0])


@pytest.mark.unit
def test_load_prices_uses_yfinance_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the (mocked) yfinance fetch succeeds, its panel + label are returned."""
    from mvtsforecast.data import loaders

    idx = pd.to_datetime(["2021-01-04", "2021-01-05"])
    panel = pd.DataFrame({"SPY": [1.0, 2.0]}, index=idx)
    monkeypatch.setattr(loaders, "_fetch_yfinance", lambda *a, **k: panel)
    prices, source = loaders.load_prices(["SPY"])
    assert source == "yfinance"
    pd.testing.assert_frame_equal(prices, panel.astype("float64"))


@pytest.mark.unit
def test_load_prices_falls_through_empty_yfinance_to_stooq(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty yfinance panel falls through to the (mocked) Stooq fetch."""
    from mvtsforecast.data import loaders

    idx = pd.to_datetime(["2021-01-04", "2021-01-05"])
    stooq_panel = pd.DataFrame({"SPY": [3.0, 4.0]}, index=idx)
    monkeypatch.setattr(loaders, "_fetch_yfinance", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(loaders, "_fetch_stooq", lambda *a, **k: stooq_panel)
    prices, source = loaders.load_prices(["SPY"])
    assert source == "stooq"
    pd.testing.assert_frame_equal(prices, stooq_panel.astype("float64"))


# --------------------------------------------------------------------------- #
# load_macro: remaining error branches                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_load_macro_unparsable_csv_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read_csv failure surfaces as a clear validation error."""
    csv = tmp_path / "x.csv"
    _write_fred_csv(csv, ["2021-01-04"], [1.0])
    index = pd.bdate_range("2021-01-04", periods=3)

    def _boom(*_a: object, **_k: object) -> pd.DataFrame:
        raise ValueError("corrupt")

    monkeypatch.setattr(pd, "read_csv", _boom)
    with pytest.raises(ValidationError, match="could not parse"):
        load_macro({"x": csv}, index=index)


@pytest.mark.unit
def test_load_macro_date_only_csv_rejected(tmp_path: Path) -> None:
    """A CSV with only a date column (no value column) is rejected."""
    csv = tmp_path / "x.csv"
    pd.DataFrame({"DATE": ["2021-01-04"]}).to_csv(csv, index=False)
    index = pd.bdate_range("2021-01-04", periods=3)
    with pytest.raises(ValidationError, match="no value column"):
        load_macro({"x": csv}, index=index)


# --------------------------------------------------------------------------- #
# import purity                                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_loaders_import_pulls_in_no_heavy_dependency() -> None:
    """Importing the loaders module must not import yfinance / pandas_datareader."""
    code = (
        "import sys;"
        "import mvtsforecast.data.loaders;"
        "bad=[m for m in ('yfinance','pandas_datareader','curl_cffi') "
        "if m in sys.modules];"
        "print(','.join(bad))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "", f"loaders leaked heavy modules: {out.stdout!r}"
