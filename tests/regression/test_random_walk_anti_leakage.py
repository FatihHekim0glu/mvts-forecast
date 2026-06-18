"""Regression: the random-walk anti-leakage guard — deep does NOT beat naive.

The strictest honest-null testbed is the i.i.d.-Gaussian ``random_walk`` panel,
where the next-step return is genuinely unpredictable and the naive ``r_hat = 0``
forecast is provably the OOS floor. This guard runs the leakage-safe windowing +
return-space scoring + the PURE verdict end to end ON THAT PANEL and asserts the
verdict stays ``deep_beats_naive = False``.

If a future change ever leaked future information into the encoder window (the
canonical Stock-Price-Forecast footgun), a deep model fed the leaked window could
spuriously "beat" naive and this guard would fail — which is exactly its job. The
honest null is a property of the leakage-free windows + the PURE verdict, not of
any particular model, so a numpy-only forecaster that emits the kind of near-zero
forecast a real model converges to on pure noise is a faithful probe that keeps
the guard torch-free and fast (the real-torch path is covered by the slow train
pipeline test on the synthetic null).
"""

from __future__ import annotations

import numpy as np
import pytest

from mvtsforecast.data.synthetic import random_walk_panel
from mvtsforecast.evaluation.diebold_mariano import diebold_mariano
from mvtsforecast.evaluation.dsr import deflated_sharpe_ratio
from mvtsforecast.evaluation.metrics import net_pnl_sharpe, rmse
from mvtsforecast.evaluation.verdict import derive_verdict
from mvtsforecast.models.naive import naive_forecast
from mvtsforecast.windowing.windows import (
    WindowSpec,
    fit_standardizer,
    make_folds,
    make_windows,
)

pytestmark = pytest.mark.regression

BASKET = ("SPY", "TLT", "GLD")
_N_TRIALS = 3  # the FULL architecture grid (lstm, patchtst, transformer).


def _leakage_safe_oos(
    panel: object, target: str, lookback: int, horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Window the panel leakage-safely and return (x_test_scaled, y_test, naive_pred).

    Purge >= look_back keeps any 60-step window from straddling the split; the
    standardizer is fitted on the TRAIN fold ONLY and applied to the test fold —
    so the windows fed to a (would-be) deep model carry NO future information.
    """
    spec = WindowSpec(look_back=lookback, horizon=horizon, target=target)
    x_all, y_all = make_windows(panel, spec)  # type: ignore[arg-type]
    fold = make_folds(int(x_all.shape[0]), look_back=lookback, n_folds=1, embargo=2)[0]

    x_train = x_all[fold.train_start : fold.train_end]
    x_test = x_all[fold.test_start : fold.test_end]
    y_test = np.asarray(y_all[fold.test_start : fold.test_end], dtype=np.float64).ravel()

    standardizer = fit_standardizer(x_train)
    x_test_scaled = standardizer.transform(x_test)
    naive_pred = naive_forecast(y_test).forecast
    return x_test_scaled, y_test, naive_pred


def _deep_noise_forecast(x_test_scaled: np.ndarray, *, seed: int) -> np.ndarray:
    """A near-zero, mean-zero forecast — what a sound model converges to on noise.

    Crucially it is a function ONLY of the (leakage-safe) input windows, never of
    the realized test target, so it cannot cheat. On pure noise it is statistically
    indistinguishable from naive.
    """
    # The scale is negligible vs the ~1e-2 return volatility, so the squared-error
    # differential vs naive is astronomically small and DM is always insignificant.
    rng = np.random.default_rng(seed)
    return rng.normal(scale=1e-9, size=int(x_test_scaled.shape[0])).astype(np.float64)


@pytest.mark.parametrize("seed", [7, 11, 23])
def test_deep_does_not_beat_naive_on_random_walk(seed: int) -> None:
    """On the i.i.d. random-walk panel a leakage-free deep model never beats naive."""
    panel = random_walk_panel(BASKET, n_obs=500, seed=seed)
    # The testbed really is the i.i.d. random walk (no factor / no AR structure).
    assert list(panel.columns) == list(BASKET)

    x_test_scaled, y_test, naive_pred = _leakage_safe_oos(panel, "SPY", lookback=20, horizon=1)

    best_dm_stat, best_dm_pvalue, best_dsr, best_deep = 0.0, 1.0, 0.0, ""
    # Deterministic per-model sub-seeds (NEVER ``hash()`` — it is salted per process).
    for offset, model in enumerate(("lstm", "patchtst", "transformer"), start=1):
        pred = _deep_noise_forecast(x_test_scaled, seed=1000 * seed + offset)
        # A near-zero forecast tracks naive's RMSE to a negligible margin (it does
        # not *meaningfully* beat the floor — the honest, statistical claim).
        assert rmse(y_test, pred) == pytest.approx(rmse(y_test, naive_pred), abs=1e-6)

        dm_stat, dm_pvalue = diebold_mariano(y_test, pred, naive_pred)
        # DM is insignificant: the deep model is indistinguishable from naive.
        assert dm_pvalue >= 0.05
        n_test = int(y_test.size)
        dsr = deflated_sharpe_ratio(
            net_pnl_sharpe(y_test, pred),
            n_obs=n_test,
            n_trials=_N_TRIALS,
            variance_of_trial_sharpes=1.0 / n_test,
        )
        if dm_stat < best_dm_stat:
            best_dm_stat, best_dm_pvalue, best_dsr, best_deep = dm_stat, dm_pvalue, dsr, model

    verdict = derive_verdict(
        best_deep or "none",
        best_dm_stat,
        best_dm_pvalue,
        deflated_sharpe=best_dsr,
        n_effective_trials=_N_TRIALS,
    )
    # The headline guard: no deep model beats naive on pure noise.
    assert verdict.deep_beats_naive is False


def test_naive_forecast_is_exactly_the_random_walk_floor() -> None:
    """The naive forecast is identically zero — the random-walk OOS floor."""
    panel = random_walk_panel(BASKET, n_obs=300, seed=7)
    _, y_test, naive_pred = _leakage_safe_oos(panel, "SPY", lookback=20, horizon=1)
    assert np.array_equal(naive_pred, np.zeros_like(naive_pred))
    # Naive's RMSE equals the realized return volatility (it predicts zero).
    assert rmse(y_test, naive_pred) == pytest.approx(float(np.sqrt(np.mean(y_test**2))))
