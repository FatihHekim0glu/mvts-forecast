"""Property-based invariants for the leakage-safe windowing + RevIN layer.

These Hypothesis tests pin down the *structural* leakage guards that a leaky
stock-price repo lacks, expressed as invariants that must hold for ANY admissible
panel/window shape:

- **future-perturbation invariance** — the window at sample ``i`` (and its RevIN
  statistics) depend ONLY on rows ``[i, i + look_back)``; altering any row at or
  after the target position cannot change them. This is the testable counterpart
  to "no lookahead";
- **no-target-in-features** — the encoder input never contains the target column's
  same-step transform (both the explicit guard and the produced tensor width);
- **RevIN inverse round-trips** — ``revin_denormalize`` exactly inverts
  ``revin_normalize`` on the target feature;
- **window/target alignment** — ``X[i]`` are rows ``[i, i + look_back)`` and
  ``y[i]`` is the target return at row ``i + look_back + horizon - 1``.

The suite imports only the import-pure parts of the package (no torch / onnx).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from mvtsforecast._exceptions import ValidationError
from mvtsforecast.windowing.revin import (
    RevInStats,
    revin_denormalize,
    revin_normalize,
)
from mvtsforecast.windowing.windows import (
    WindowSpec,
    assert_no_target_leakage,
    fit_standardizer,
    make_folds,
    make_windows,
)

# --------------------------------------------------------------------------- #
# Strategies                                                                   #
# --------------------------------------------------------------------------- #
_COLS = ("SPY", "TLT", "GLD", "VIX")


@st.composite
def _panels(draw: st.DrawFn) -> tuple[pd.DataFrame, WindowSpec]:
    """Draw a finite returns panel and a window spec that fits it."""
    n_features = draw(st.integers(min_value=2, max_value=4))
    columns = list(_COLS[:n_features])
    look_back = draw(st.integers(min_value=1, max_value=6))
    horizon = draw(st.integers(min_value=1, max_value=3))
    # At least one window: n_rows >= look_back + horizon (+ a few spare rows).
    n_rows = draw(st.integers(min_value=look_back + horizon, max_value=look_back + horizon + 8))
    target = draw(st.sampled_from(columns))

    values = draw(
        hnp.arrays(
            dtype=np.float64,
            shape=(n_rows, n_features),
            elements=st.floats(
                min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False
            ),
        )
    )
    panel = pd.DataFrame(values, columns=columns)
    spec = WindowSpec(look_back=look_back, horizon=horizon, target=target)
    return panel, spec


def _window_tensors(data: st.DataObject) -> np.ndarray:
    """Draw a non-degenerate 3-D window tensor for RevIN."""
    n = data.draw(st.integers(min_value=1, max_value=5))
    look_back = data.draw(st.integers(min_value=2, max_value=8))
    n_features = data.draw(st.integers(min_value=1, max_value=4))
    arr: np.ndarray = data.draw(
        hnp.arrays(
            dtype=np.float64,
            shape=(n, look_back, n_features),
            elements=st.floats(
                min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False
            ),
        )
    )
    return arr


# --------------------------------------------------------------------------- #
# Window / target alignment                                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.property
@given(_panels())
@settings(max_examples=200, deadline=None)
def test_window_target_alignment(case: tuple[pd.DataFrame, WindowSpec]) -> None:
    """``X[i]`` is rows ``[i, i+L)``; ``y[i]`` is target at ``i+L+h-1``."""
    panel, spec = case
    x, y = make_windows(panel, spec)

    look_back, horizon = spec.look_back, spec.horizon
    expected_n = panel.shape[0] - look_back - horizon + 1
    assert x.shape[0] == expected_n == y.shape[0]
    assert x.shape[1] == look_back

    feature_cols = [c for c in panel.columns if c != spec.target]
    feat = panel.loc[:, feature_cols].to_numpy(dtype=np.float64)
    target = panel.loc[:, spec.target].to_numpy(dtype=np.float64)

    for i in range(x.shape[0]):
        # The window is exactly rows [i, i + look_back) over the feature columns.
        np.testing.assert_array_equal(x[i], feat[i : i + look_back])
        # The label is the target return strictly AFTER the window.
        assert y[i] == target[i + look_back + horizon - 1]


# --------------------------------------------------------------------------- #
# No-target-in-features                                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.property
@given(_panels())
@settings(max_examples=200, deadline=None)
def test_target_never_in_encoder_input(case: tuple[pd.DataFrame, WindowSpec]) -> None:
    """The encoder feature axis excludes the target's same-step transform."""
    panel, spec = case
    x, _ = make_windows(panel, spec)
    # One fewer feature than panel columns: the target column was dropped.
    assert x.shape[2] == panel.shape[1] - 1

    # The produced encoder tensor must equal the panel with the TARGET column
    # removed (in original order) — so the target's same-step transform is, by
    # construction, absent from every window/feature slice.
    feature_cols = [c for c in panel.columns if c != spec.target]
    feat = panel.loc[:, feature_cols].to_numpy(dtype=np.float64)
    for i in range(x.shape[0]):
        np.testing.assert_array_equal(x[i], feat[i : i + spec.look_back])


@pytest.mark.property
def test_assert_no_target_leakage_rejects_target_in_features() -> None:
    """The explicit guard raises when the target sits in the feature columns."""
    spec = WindowSpec(
        look_back=4,
        target="SPY",
        feature_columns=("SPY", "TLT", "GLD"),
        drop_target_feature=True,
    )
    with pytest.raises(ValidationError, match="must not read the value it predicts"):
        assert_no_target_leakage(spec)


@pytest.mark.property
def test_assert_no_target_leakage_accepts_clean_spec() -> None:
    """The guard is silent when the target is absent from the feature columns."""
    spec = WindowSpec(look_back=4, target="SPY", feature_columns=("TLT", "GLD"))
    # Must not raise.
    assert_no_target_leakage(spec)


@pytest.mark.property
def test_make_windows_refuses_target_when_drop_disabled() -> None:
    """With ``drop_target_feature=False`` an in-feature target is refused."""
    panel = pd.DataFrame(
        np.arange(40, dtype=np.float64).reshape(10, 4),
        columns=list(_COLS),
    )
    spec = WindowSpec(
        look_back=3,
        target="SPY",
        feature_columns=("SPY", "TLT"),
        drop_target_feature=False,
    )
    with pytest.raises(ValidationError, match="would leak"):
        make_windows(panel, spec)


# --------------------------------------------------------------------------- #
# Future-perturbation invariance (the leakage property)                       #
# --------------------------------------------------------------------------- #
@pytest.mark.property
@given(_panels(), st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False))
@settings(max_examples=200, deadline=None)
def test_future_perturbation_leaves_early_windows_unchanged(
    case: tuple[pd.DataFrame, WindowSpec], delta: float
) -> None:
    """Altering rows at/after a window's target cannot change that window or its
    RevIN statistics — the formal no-lookahead invariant."""
    panel, spec = case
    x, _ = make_windows(panel, spec)
    assume(x.shape[0] >= 2)

    # Perturb every row from the FIRST window's target position onward. Window 0
    # spans rows [0, look_back); its target is at look_back + horizon - 1, so
    # touching that row (and later) must not move window 0 nor its RevIN stats.
    cut = spec.look_back  # rows >= look_back are all strictly future to window 0
    perturbed = panel.copy()
    perturbed.iloc[cut:] = perturbed.iloc[cut:] + delta

    x2, _ = make_windows(perturbed, spec)

    # Window 0 (the only window that depends solely on rows [0, look_back)) is
    # byte-identical before and after the future perturbation.
    np.testing.assert_array_equal(x[0], x2[0])

    # And its RevIN statistics are identical, too.
    _, stats_a = revin_normalize(x[0:1])
    _, stats_b = revin_normalize(x2[0:1])
    np.testing.assert_array_equal(stats_a.mean, stats_b.mean)
    np.testing.assert_array_equal(stats_a.std, stats_b.std)


@pytest.mark.property
@given(st.data())
@settings(max_examples=150, deadline=None)
def test_revin_window_stats_are_purely_local(data: st.DataObject) -> None:
    """Each RevIN per-window statistic depends only on that window's own rows."""
    windows = _window_tensors(data)
    assume(windows.shape[0] >= 2)
    _, stats = revin_normalize(windows)

    # Perturb every window EXCEPT window 0; window 0's stats must be unchanged.
    perturbed = windows.copy()
    perturbed[1:] += data.draw(
        st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)
    )
    _, stats2 = revin_normalize(perturbed)
    np.testing.assert_array_equal(stats.mean[0], stats2.mean[0])
    np.testing.assert_array_equal(stats.std[0], stats2.std[0])


# --------------------------------------------------------------------------- #
# RevIN inverse round-trips                                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.property
@given(st.data())
@settings(max_examples=200, deadline=None)
def test_revin_roundtrips_on_target_feature(data: st.DataObject) -> None:
    """``denormalize(normalize(x)[..., f]) == x[..., f]`` for the target feature."""
    windows = _window_tensors(data)
    n_features = windows.shape[2]
    feature_index = data.draw(st.integers(min_value=0, max_value=n_features - 1))

    normalized, stats = revin_normalize(windows)
    # Take the LAST normalized time-step of the chosen feature as a stand-in for a
    # one-step forecast in normalized space, then invert it.
    norm_forecast = normalized[:, -1, feature_index]
    recovered = revin_denormalize(norm_forecast, stats, feature_index=feature_index)
    np.testing.assert_allclose(recovered, windows[:, -1, feature_index], rtol=1e-9, atol=1e-9)


@pytest.mark.property
def test_revin_normalizes_each_window_to_zero_mean() -> None:
    """Normalized windows have ~zero per-window mean (instance-norm semantics)."""
    rng = np.random.default_rng(7)
    windows = rng.standard_normal((6, 12, 3)) * 3.0 + 2.0
    normalized, _ = revin_normalize(windows)
    np.testing.assert_allclose(normalized.mean(axis=1), 0.0, atol=1e-9)


# --------------------------------------------------------------------------- #
# Validation surface                                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.property
def test_revin_normalize_rejects_non_3d() -> None:
    """RevIN requires a 3-D window tensor."""
    with pytest.raises(ValidationError, match="3-D"):
        revin_normalize(np.zeros((4, 5)))


@pytest.mark.property
def test_revin_denormalize_rejects_length_mismatch() -> None:
    """A forecast whose length differs from the captured stats is rejected."""
    _, stats = revin_normalize(np.ones((3, 5, 2)))
    with pytest.raises(ValidationError, match="does not match"):
        revin_denormalize(np.zeros(2), stats)


@pytest.mark.property
def test_revin_denormalize_rejects_bad_feature_index() -> None:
    """An out-of-range feature index is rejected."""
    _, stats = revin_normalize(np.ones((3, 5, 2)))
    with pytest.raises(ValidationError, match="out of range"):
        revin_denormalize(np.zeros(3), stats, feature_index=5)


@pytest.mark.property
def test_revin_stats_to_dict_preserves_shapes() -> None:
    """``RevInStats.to_dict`` round-trips the captured statistic arrays."""
    _, stats = revin_normalize(np.arange(2 * 5 * 2, dtype=np.float64).reshape(2, 5, 2))
    payload = stats.to_dict()
    assert isinstance(stats, RevInStats)
    np.testing.assert_array_equal(payload["mean"], stats.mean)
    np.testing.assert_array_equal(payload["std"], stats.std)


@pytest.mark.property
def test_make_windows_rejects_short_panel() -> None:
    """A panel too short to form one window raises ``InsufficientDataError``."""
    from mvtsforecast._exceptions import InsufficientDataError

    panel = pd.DataFrame(np.zeros((3, 3)), columns=["SPY", "TLT", "GLD"])
    spec = WindowSpec(look_back=5, horizon=1, target="SPY")
    with pytest.raises(InsufficientDataError, match="at least look_back"):
        make_windows(panel, spec)


@pytest.mark.property
def test_make_windows_rejects_missing_target() -> None:
    """An absent target column is rejected."""
    panel = pd.DataFrame(np.zeros((10, 2)), columns=["TLT", "GLD"])
    spec = WindowSpec(look_back=3, horizon=1, target="SPY")
    with pytest.raises(ValidationError, match="absent"):
        make_windows(panel, spec)


# --------------------------------------------------------------------------- #
# Walk-forward purge invariant                                                #
# --------------------------------------------------------------------------- #
@pytest.mark.property
@given(
    n_samples=st.integers(min_value=80, max_value=400),
    look_back=st.integers(min_value=1, max_value=20),
    n_folds=st.integers(min_value=1, max_value=4),
    embargo=st.integers(min_value=0, max_value=5),
    anchored=st.booleans(),
)
@settings(max_examples=200, deadline=None)
def test_folds_purge_keeps_windows_from_straddling(
    n_samples: int,
    look_back: int,
    n_folds: int,
    embargo: int,
    anchored: bool,
) -> None:
    """Every fold's train/test gap is at least ``look_back`` (no straddling)."""
    try:
        folds = make_folds(
            n_samples,
            look_back=look_back,
            n_folds=n_folds,
            embargo=embargo,
            anchored=anchored,
        )
    except ValidationError:
        # Insufficient-data configurations are a valid, documented outcome.
        return

    assert len(folds) == n_folds
    prev_test_end = -1
    for fold in folds:
        # Purge gap >= look_back: no look_back-length window can straddle.
        assert fold.test_start - fold.train_end >= look_back
        # Non-empty, ordered slices.
        assert fold.train_start < fold.train_end
        assert fold.test_start < fold.test_end
        assert fold.test_end <= n_samples
        # Test blocks march strictly forward and never overlap.
        assert fold.test_start > prev_test_end
        prev_test_end = fold.test_end
        if anchored:
            assert fold.train_start == 0


@pytest.mark.property
def test_fit_standardizer_is_train_only_and_applies() -> None:
    """The standardizer's stats come from the train tensor and apply elsewhere."""
    rng = np.random.default_rng(3)
    x_train = rng.standard_normal((40, 8, 3)) * 2.0 + 5.0
    sc = fit_standardizer(x_train, feature_columns=("SPY", "TLT", "GLD"))
    # Applied to its own train data the result is ~standardized.
    z = sc.transform(x_train)
    np.testing.assert_allclose(z.mean(axis=(0, 1)), 0.0, atol=1e-9)
    np.testing.assert_allclose(z.std(axis=(0, 1)), 1.0, atol=1e-6)
    # A held-out tensor is transformed with the SAME (train) stats, not re-fitted.
    x_test = rng.standard_normal((10, 8, 3)) * 2.0 + 9.0
    z_test = sc.transform(x_test)
    assert z_test.shape == x_test.shape
    # The held-out mean is shifted (proves the train stats, not test stats, used).
    assert not np.allclose(z_test.mean(axis=(0, 1)), 0.0, atol=1e-3)


# --------------------------------------------------------------------------- #
# Dataclass ``to_dict`` round-trips (provenance / API serialization)          #
# --------------------------------------------------------------------------- #
@pytest.mark.property
def test_window_spec_to_dict_is_plain() -> None:
    """``WindowSpec.to_dict`` is a plain, JSON-friendly mapping."""
    spec = WindowSpec(look_back=5, horizon=2, target="SPY", feature_columns=("TLT", "GLD"))
    payload = spec.to_dict()
    assert payload["look_back"] == 5
    assert payload["horizon"] == 2
    assert payload["target"] == "SPY"
    assert payload["feature_columns"] == ("TLT", "GLD")
    assert payload["drop_target_feature"] is True


@pytest.mark.property
def test_fold_to_dict_round_trips() -> None:
    """``Fold.to_dict`` exposes the four half-open bounds."""
    folds = make_folds(200, look_back=10, n_folds=2, embargo=2)
    payload = folds[0].to_dict()
    assert set(payload) == {"train_start", "train_end", "test_start", "test_end"}
    assert payload["test_start"] - payload["train_end"] >= 10


@pytest.mark.property
def test_standardizer_to_dict_lists_floats() -> None:
    """``Standardizer.to_dict`` renders mean/std as plain float lists."""
    rng = np.random.default_rng(11)
    sc = fit_standardizer(rng.standard_normal((20, 6, 2)), feature_columns=("SPY", "TLT"))
    payload = sc.to_dict()
    assert payload["feature_columns"] == ["SPY", "TLT"]
    assert len(payload["mean"]) == 2
    assert all(isinstance(v, float) for v in payload["mean"])
    assert all(isinstance(v, float) for v in payload["std"])


# --------------------------------------------------------------------------- #
# Standardizer + make_windows + make_folds validation surface                 #
# --------------------------------------------------------------------------- #
@pytest.mark.property
def test_standardizer_transform_rejects_non_3d() -> None:
    """A non-3-D tensor cannot be standardized."""
    sc = fit_standardizer(np.ones((4, 5, 2)))
    with pytest.raises(ValidationError, match="must be 3-D"):
        sc.transform(np.ones((5, 2)))


@pytest.mark.property
def test_standardizer_transform_rejects_feature_mismatch() -> None:
    """A tensor whose feature axis differs from the fitted stats is rejected."""
    sc = fit_standardizer(np.ones((4, 5, 2)))
    with pytest.raises(ValidationError, match="does not"):
        sc.transform(np.ones((3, 5, 4)))


@pytest.mark.property
def test_make_windows_rejects_non_positive_look_back() -> None:
    """A look_back < 1 is rejected."""
    panel = pd.DataFrame(np.zeros((10, 2)), columns=["SPY", "TLT"])
    with pytest.raises(ValidationError, match="look_back"):
        make_windows(panel, WindowSpec(look_back=0, target="SPY"))


@pytest.mark.property
def test_make_windows_rejects_non_positive_horizon() -> None:
    """A horizon < 1 is rejected."""
    panel = pd.DataFrame(np.zeros((10, 2)), columns=["SPY", "TLT"])
    with pytest.raises(ValidationError, match="horizon"):
        make_windows(panel, WindowSpec(look_back=3, horizon=0, target="SPY"))


@pytest.mark.property
def test_make_windows_rejects_unknown_feature_column() -> None:
    """A feature column absent from the panel is rejected."""
    panel = pd.DataFrame(np.zeros((10, 2)), columns=["SPY", "TLT"])
    spec = WindowSpec(look_back=3, target="SPY", feature_columns=("TLT", "NOPE"))
    with pytest.raises(ValidationError, match="absent"):
        make_windows(panel, spec)


@pytest.mark.property
def test_make_windows_rejects_target_only_panel() -> None:
    """A panel whose only column is the target leaves no encoder features."""
    panel = pd.DataFrame(np.zeros((10, 1)), columns=["SPY"])
    spec = WindowSpec(look_back=3, target="SPY")
    with pytest.raises(ValidationError, match="no feature columns remain"):
        make_windows(panel, spec)


@pytest.mark.property
def test_make_windows_accepts_explicit_feature_columns() -> None:
    """An explicit feature-column subset is honoured (target still dropped)."""
    panel = pd.DataFrame(np.arange(40, dtype=np.float64).reshape(10, 4), columns=list(_COLS))
    spec = WindowSpec(look_back=3, target="SPY", feature_columns=("SPY", "TLT", "GLD"))
    x, _ = make_windows(panel, spec)
    # SPY dropped from the explicit list => 2 encoder features.
    assert x.shape[2] == 2


@pytest.mark.property
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_samples": 0}, "n_samples"),
        ({"n_samples": 100, "look_back": -1}, "look_back"),
        ({"n_samples": 100, "n_folds": 0}, "n_folds"),
        ({"n_samples": 100, "embargo": -1}, "embargo"),
        ({"n_samples": 100, "test_size": 0}, "test_size"),
    ],
)
def test_make_folds_validation(kwargs: dict[str, int], match: str) -> None:
    """Bad fold parameters raise a clear ``ValidationError``."""
    with pytest.raises(ValidationError, match=match):
        make_folds(**kwargs)  # type: ignore[arg-type]


@pytest.mark.property
def test_make_folds_insufficient_samples_raises() -> None:
    """Too few samples for the requested folds raises ``InsufficientDataError``."""
    from mvtsforecast._exceptions import InsufficientDataError

    with pytest.raises(InsufficientDataError, match="cannot host"):
        make_folds(20, look_back=60, n_folds=3, embargo=5)


@pytest.mark.property
def test_make_folds_explicit_test_size_too_large_raises() -> None:
    """An explicit test_size that overflows the budget is rejected."""
    from mvtsforecast._exceptions import InsufficientDataError

    with pytest.raises(InsufficientDataError, match="cannot host"):
        make_folds(100, look_back=10, n_folds=3, test_size=80)


@pytest.mark.property
def test_make_folds_rolling_train_window_is_bounded() -> None:
    """A rolling (non-anchored) fold's train slice is bounded by look_back."""
    folds = make_folds(340, look_back=60, n_folds=3, embargo=5, anchored=False)
    for fold in folds:
        assert fold.train_end - fold.train_start <= 60
        assert fold.train_start >= 0


@pytest.mark.property
def test_fit_standardizer_rejects_non_3d() -> None:
    """A non-3-D train tensor cannot be fitted."""
    with pytest.raises(ValidationError, match="must be 3-D"):
        fit_standardizer(np.ones((5, 4)))


@pytest.mark.property
def test_fit_standardizer_rejects_empty() -> None:
    """An empty train tensor is rejected."""
    with pytest.raises(ValidationError, match="non-empty"):
        fit_standardizer(np.empty((0, 5, 3)))


@pytest.mark.property
def test_fit_standardizer_rejects_label_count_mismatch() -> None:
    """Mismatched feature_columns length is rejected."""
    with pytest.raises(ValidationError, match="feature"):
        fit_standardizer(np.ones((4, 5, 3)), feature_columns=("SPY", "TLT"))


@pytest.mark.property
def test_revin_rejects_empty_window() -> None:
    """An empty (zero-size) 3-D tensor is rejected by RevIN."""
    with pytest.raises(ValidationError, match="non-empty"):
        revin_normalize(np.empty((0, 4, 3)))


@pytest.mark.property
def test_revin_rejects_non_finite_window() -> None:
    """A window with NaN/Inf is rejected by RevIN."""
    bad = np.ones((2, 4, 3))
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValidationError, match="finite"):
        revin_normalize(bad)


@pytest.mark.property
def test_revin_rejects_negative_eps() -> None:
    """A negative eps floor is rejected."""
    with pytest.raises(ValidationError, match="eps"):
        revin_normalize(np.ones((2, 4, 3)), eps=-1.0)


@pytest.mark.property
def test_revin_denormalize_rejects_non_1d_forecast() -> None:
    """A non-1-D normalized forecast is rejected."""
    _, stats = revin_normalize(np.ones((3, 5, 2)))
    with pytest.raises(ValidationError, match="1-D"):
        revin_denormalize(np.zeros((3, 1)), stats)


@pytest.mark.property
def test_revin_denormalize_rejects_empty_forecast() -> None:
    """An empty normalized forecast is rejected."""
    _, stats = revin_normalize(np.ones((3, 5, 2)))
    with pytest.raises(ValidationError, match="non-empty"):
        revin_denormalize(np.zeros(0), stats)


@pytest.mark.property
def test_revin_denormalize_rejects_non_3d_stats() -> None:
    """Malformed (non-3-D) stats arrays are rejected on de-normalization."""
    bad_stats = RevInStats(mean=np.zeros(3), std=np.ones(3))
    with pytest.raises(ValidationError, match="3-D"):
        revin_denormalize(np.zeros(3), bad_stats)
