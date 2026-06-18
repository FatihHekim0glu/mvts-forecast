"""Unit tests for the small deep models (LSTM, PatchTST, variable-selection transformer).

Two layers:

- **config validation** (fast, torch-free) — every model's frozen ``*Config``
  validates its sizes / patch geometry / head-divisibility eagerly in
  ``__post_init__``, so a bad hyperparameter is rejected WITHOUT importing torch;
- **build + forward shape** (``slow``, torch) — each builder returns a
  ``torch.nn.Module`` mapping a tiny ``(batch, look_back, n_features)`` window to a
  ``(batch, 1)`` next-step return, and each trainer runs a few-epoch fit. These are
  marked ``slow`` and SKIPPED when the ``[train]`` extra (torch) is absent, so the
  torch-free serve suite still runs.

The torch import is deferred to the builders/trainers (never at module load), so
importing these model modules pulls in NO torch — the import-purity guard.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from mvtsforecast._exceptions import ValidationError
from mvtsforecast.models.lstm import LstmConfig, build_lstm, train_lstm
from mvtsforecast.models.patchtst import PatchTSTConfig, build_patchtst, train_patchtst
from mvtsforecast.models.transformer_vs import (
    TransformerVSConfig,
    build_transformer_vs,
    train_transformer_vs,
)

_HAS_TORCH = importlib.util.find_spec("torch") is not None
_requires_torch = pytest.mark.skipif(
    not _HAS_TORCH, reason="torch ([train] extra) not installed; the slow torch path is skipped"
)

#: Tiny shapes keep the slow torch tests fast while exercising real forward passes.
BATCH, LOOK_BACK, N_FEATURES = 8, 60, 2


def _tiny_batch() -> tuple[np.ndarray, np.ndarray]:
    """A small, seeded ``(X, y)`` train batch shaped for the default look-back."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal((BATCH, LOOK_BACK, N_FEATURES)).astype(np.float64)
    y = rng.standard_normal(BATCH).astype(np.float64)
    return x, y


# --------------------------------------------------------------------------- #
# config validation (torch-free)                                              #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_configs_round_trip_to_dict() -> None:
    """Every config serializes to a plain JSON-safe ``dict`` of its fields."""
    for cfg in (LstmConfig(), PatchTSTConfig(), TransformerVSConfig()):
        out = cfg.to_dict()
        assert isinstance(out, dict)
        assert out["look_back"] == 60
        assert out["n_features"] == 2


@pytest.mark.unit
def test_patchtst_n_patches_geometry() -> None:
    """``n_patches`` follows ``(look_back - patch_len) // stride + 1``."""
    cfg = PatchTSTConfig(look_back=60, patch_len=16, stride=8)
    assert cfg.n_patches == (60 - 16) // 8 + 1 == 6


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"look_back": 0}, "sizes must be >= 1"),
        ({"n_features": 0}, "sizes must be >= 1"),
        ({"hidden_size": 0}, "sizes must be >= 1"),
        ({"num_layers": 0}, "sizes must be >= 1"),
        ({"dropout": 1.0}, "dropout"),
        ({"dropout": -0.1}, "dropout"),
    ],
)
def test_lstm_config_validation(kwargs: dict[str, object], match: str) -> None:
    """An out-of-range LSTM hyperparameter is rejected without importing torch."""
    with pytest.raises(ValidationError, match=match):
        LstmConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"patch_len": 0}, "sizes must be >= 1"),
        ({"n_heads": 0}, "sizes must be >= 1"),
        ({"patch_len": 100}, "patch_len"),
        ({"d_model": 30, "n_heads": 4}, "divisible by n_heads"),
        ({"dropout": 1.5}, "dropout"),
    ],
)
def test_patchtst_config_validation(kwargs: dict[str, object], match: str) -> None:
    """Bad PatchTST size/geometry/divisibility is rejected eagerly (torch-free)."""
    with pytest.raises(ValidationError, match=match):
        PatchTSTConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"d_model": 0}, "sizes must be >= 1"),
        ({"n_layers": 0}, "sizes must be >= 1"),
        ({"d_model": 30, "n_heads": 4}, "divisible by n_heads"),
        ({"dropout": 2.0}, "dropout"),
    ],
)
def test_transformer_vs_config_validation(kwargs: dict[str, object], match: str) -> None:
    """Bad variable-selection-transformer config is rejected eagerly (torch-free)."""
    with pytest.raises(ValidationError, match=match):
        TransformerVSConfig(**kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# build + forward shape (slow, torch)                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@_requires_torch
def test_build_lstm_forward_shape() -> None:
    """The LSTM maps a ``(batch, look_back, n_features)`` window to ``(batch, 1)``."""
    import torch

    cfg = LstmConfig(look_back=LOOK_BACK, n_features=N_FEATURES, hidden_size=8)
    model = build_lstm(cfg)
    x, _ = _tiny_batch()
    out = model(torch.as_tensor(x, dtype=torch.float32))
    assert tuple(out.shape) == (BATCH, 1)


@pytest.mark.slow
@_requires_torch
def test_build_patchtst_forward_shape() -> None:
    """The PatchTST-style encoder maps the tiny window to ``(batch, 1)``."""
    import torch

    cfg = PatchTSTConfig(
        look_back=LOOK_BACK,
        n_features=N_FEATURES,
        patch_len=16,
        stride=8,
        d_model=16,
        n_heads=2,
        n_layers=1,
    )
    model = build_patchtst(cfg)
    x, _ = _tiny_batch()
    out = model(torch.as_tensor(x, dtype=torch.float32))
    assert tuple(out.shape) == (BATCH, 1)


@pytest.mark.slow
@_requires_torch
def test_build_transformer_vs_forward_shape_and_weights() -> None:
    """The transformer maps to ``(batch, 1)`` and exposes per-feature weights summing to 1."""
    import torch

    cfg = TransformerVSConfig(
        look_back=LOOK_BACK, n_features=N_FEATURES, d_model=16, n_heads=2, n_layers=1
    )
    model = build_transformer_vs(cfg)
    x, _ = _tiny_batch()
    out = model(torch.as_tensor(x, dtype=torch.float32))
    assert tuple(out.shape) == (BATCH, 1)

    weights = model.variable_weights()
    assert tuple(weights.shape) == (N_FEATURES,)
    assert abs(float(weights.sum()) - 1.0) < 1e-5


@pytest.mark.slow
@_requires_torch
@pytest.mark.parametrize("model_name", ["lstm", "patchtst", "transformer"])
def test_trainers_fit_and_are_deterministic(model_name: str) -> None:
    """A few-epoch fit runs and is reproducible for a fixed seed (forward shape ``(batch, 1)``)."""
    import torch

    x, y = _tiny_batch()

    def fit() -> torch.Tensor:
        if model_name == "lstm":
            model = train_lstm(
                x,
                y,
                LstmConfig(look_back=LOOK_BACK, n_features=N_FEATURES, hidden_size=8),
                epochs=3,
                seed=7,
            )
        elif model_name == "patchtst":
            model = train_patchtst(
                x,
                y,
                PatchTSTConfig(
                    look_back=LOOK_BACK,
                    n_features=N_FEATURES,
                    patch_len=16,
                    stride=8,
                    d_model=16,
                    n_heads=2,
                    n_layers=1,
                ),
                epochs=3,
                seed=7,
            )
        else:
            model = train_transformer_vs(
                x,
                y,
                TransformerVSConfig(
                    look_back=LOOK_BACK, n_features=N_FEATURES, d_model=16, n_heads=2, n_layers=1
                ),
                epochs=3,
                seed=7,
            )
        model.eval()
        with torch.no_grad():
            return model(torch.as_tensor(x, dtype=torch.float32)).detach()

    first = fit()
    second = fit()
    assert tuple(first.shape) == (BATCH, 1)
    # Same seed + same data => bit-identical forward outputs (reproducible fit).
    assert torch.allclose(first, second)


@pytest.mark.slow
@_requires_torch
@pytest.mark.parametrize(
    ("trainer", "config"),
    [
        (train_lstm, LstmConfig(look_back=LOOK_BACK, n_features=N_FEATURES, hidden_size=8)),
        (
            train_patchtst,
            PatchTSTConfig(look_back=LOOK_BACK, n_features=N_FEATURES, patch_len=16, stride=8),
        ),
        (train_transformer_vs, TransformerVSConfig(look_back=LOOK_BACK, n_features=N_FEATURES)),
    ],
)
def test_trainers_reject_misshaped_tensors(trainer: object, config: object) -> None:
    """A 2-D ``x_train`` (or mismatched sample counts) raises ValidationError."""
    bad_x = np.zeros((BATCH, LOOK_BACK))  # 2-D, not 3-D
    y = np.zeros(BATCH)
    with pytest.raises(ValidationError, match="3-D tensor"):
        trainer(bad_x, y, config)  # type: ignore[operator]

    good_x = np.zeros((BATCH, LOOK_BACK, N_FEATURES))
    with pytest.raises(ValidationError, match="must match"):
        trainer(good_x, np.zeros(BATCH + 1), config)  # type: ignore[operator]


@pytest.mark.slow
@_requires_torch
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"epochs": 0}, "epochs"),
        ({"lr": 0.0}, "lr"),
        ({"lr": -1.0}, "lr"),
    ],
)
def test_train_lstm_rejects_bad_budget(kwargs: dict[str, object], match: str) -> None:
    """A non-positive ``epochs`` / ``lr`` raises ValidationError before fitting."""
    x, y = _tiny_batch()
    cfg = LstmConfig(look_back=LOOK_BACK, n_features=N_FEATURES, hidden_size=8)
    with pytest.raises(ValidationError, match=match):
        train_lstm(x, y, cfg, **kwargs)  # type: ignore[arg-type]


@pytest.mark.slow
@_requires_torch
def test_train_lstm_rejects_empty_and_mismatched_dims() -> None:
    """An empty batch or a window whose dims disagree with the config is rejected."""
    cfg = LstmConfig(look_back=LOOK_BACK, n_features=N_FEATURES, hidden_size=8)

    empty = np.zeros((0, LOOK_BACK, N_FEATURES))
    with pytest.raises(ValidationError, match="non-empty"):
        train_lstm(empty, np.zeros(0), cfg)

    wrong_dims = np.zeros((BATCH, LOOK_BACK + 1, N_FEATURES))
    with pytest.raises(ValidationError, match="match config"):
        train_lstm(wrong_dims, np.zeros(BATCH), cfg)
