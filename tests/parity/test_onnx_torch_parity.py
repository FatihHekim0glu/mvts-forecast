"""Parity: the exported ONNX deep model matches its torch forward pass to 1e-4.

The serve container runs the deep models ONLY through onnxruntime (no torch), so
the committed ONNX graph MUST reproduce the trained torch module's forward pass.
For each model (LSTM, PatchTST-style, the interpretable transformer) we:

1. train a tiny torch module on a seeded batch ([train] extra);
2. export it to a temporary ONNX artifact (the legacy TorchScript exporter, which
   needs only ``onnx`` — the ``[train]`` extra — not ``onnxscript``);
3. run the SAME batch through the torch module and through
   :class:`mvtsforecast.models.onnx_runtime.OnnxForecaster` (onnxruntime);
4. assert the two forecasts agree to ``rtol``/``atol`` 1e-4.

These tests are ``slow`` and SKIPPED without torch; the serve-path tests in
``tests/unit/test_onnx_runtime.py`` cover the onnxruntime wrapper torch-free.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mvtsforecast.models.lstm import LstmConfig, train_lstm
from mvtsforecast.models.onnx_runtime import OnnxForecaster
from mvtsforecast.models.patchtst import PatchTSTConfig, train_patchtst
from mvtsforecast.models.transformer_vs import TransformerVSConfig, train_transformer_vs

_HAS_TORCH = importlib.util.find_spec("torch") is not None
_requires_torch = pytest.mark.skipif(
    not _HAS_TORCH, reason="torch ([train] extra) not installed; the parity oracle is skipped"
)

BATCH, LOOK_BACK, N_FEATURES = 24, 60, 2
PARITY_TOL = 1e-4


def _seeded_batch() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    x = rng.standard_normal((BATCH, LOOK_BACK, N_FEATURES)).astype(np.float64)
    y = rng.standard_normal(BATCH).astype(np.float64)
    return x, y


def _export_onnx(model: Any, x: np.ndarray, out_path: Path, model_name: str) -> np.ndarray:
    """Export ``model`` to ONNX, returning its torch forward pass on ``x``.

    Uses the legacy TorchScript exporter (``dynamo=False``) so only ``onnx`` is
    required — mirroring the offline train/export path's dependency set.
    """
    import torch

    model.eval()
    sample = torch.as_tensor(x, dtype=torch.float32)
    with torch.no_grad():
        torch_out = model(sample).detach().numpy().reshape(-1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (sample,),
        str(out_path),
        input_names=["sequence"],
        output_names=["return_hat"],
        dynamic_axes={"sequence": {0: "batch"}, "return_hat": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    return np.asarray(torch_out, dtype=np.float64)


@pytest.mark.parity
@pytest.mark.slow
@_requires_torch
def test_lstm_onnx_matches_torch(tmp_path: Path) -> None:
    """The exported LSTM ONNX graph reproduces the torch forward pass to 1e-4."""
    x, y = _seeded_batch()
    model = train_lstm(
        x,
        y,
        LstmConfig(look_back=LOOK_BACK, n_features=N_FEATURES, hidden_size=8),
        epochs=3,
        seed=7,
    )
    torch_out = _export_onnx(model, x, tmp_path / "lstm.onnx", "lstm")
    onnx_out = OnnxForecaster("lstm", artifact_path=tmp_path / "lstm.onnx").predict(x)
    assert onnx_out.shape == (BATCH,)
    assert np.allclose(torch_out, onnx_out, rtol=PARITY_TOL, atol=PARITY_TOL)


@pytest.mark.parity
@pytest.mark.slow
@_requires_torch
def test_patchtst_onnx_matches_torch(tmp_path: Path) -> None:
    """The exported PatchTST ONNX graph reproduces the torch forward pass to 1e-4."""
    x, y = _seeded_batch()
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
    torch_out = _export_onnx(model, x, tmp_path / "patchtst.onnx", "patchtst")
    onnx_out = OnnxForecaster("patchtst", artifact_path=tmp_path / "patchtst.onnx").predict(x)
    assert onnx_out.shape == (BATCH,)
    assert np.allclose(torch_out, onnx_out, rtol=PARITY_TOL, atol=PARITY_TOL)


@pytest.mark.parity
@pytest.mark.slow
@_requires_torch
def test_transformer_vs_onnx_matches_torch(tmp_path: Path) -> None:
    """The exported variable-selection-transformer ONNX matches torch to 1e-4."""
    x, y = _seeded_batch()
    model = train_transformer_vs(
        x,
        y,
        TransformerVSConfig(look_back=LOOK_BACK, n_features=N_FEATURES, d_model=16, n_heads=2),
        epochs=3,
        seed=7,
    )
    torch_out = _export_onnx(model, x, tmp_path / "transformer_vs.onnx", "transformer")
    onnx_out = OnnxForecaster(
        "transformer", artifact_path=tmp_path / "transformer_vs.onnx"
    ).predict(x)
    assert onnx_out.shape == (BATCH,)
    assert np.allclose(torch_out, onnx_out, rtol=PARITY_TOL, atol=PARITY_TOL)
