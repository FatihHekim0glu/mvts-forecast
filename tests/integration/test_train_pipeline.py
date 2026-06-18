"""Integration: the OFFLINE train pipeline (synthetic -> ONNX + metrics.json).

Two layers:

- **fast, torch-free** — importing :mod:`mvtsforecast.train` pulls in NO torch,
  and a bad request (empty basket / absent target / bad horizon/lookback) is
  rejected with :class:`ValidationError` BEFORE any torch import or panel build;
- **``slow``, torch** — the real end-to-end pipeline trains the three small deep
  models on a tiny synthetic panel, exports each to an ONNX artifact (parity-checked
  to the torch forward pass to 1e-4 inside ``export_onnx``), precomputes the OOS
  forecasts, scores them in RETURN space, and writes ``metrics.json``. The honest
  NULL is pinned: on the synthetic panel ``deep_beats_naive`` is ``False``. These
  are marked ``slow`` and skipped when the ``[train]`` extra (torch) is absent.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from mvtsforecast._exceptions import ValidationError
from mvtsforecast.train import TrainResult, train_pipeline

_HAS_TORCH = importlib.util.find_spec("torch") is not None
_requires_torch = pytest.mark.skipif(
    not _HAS_TORCH, reason="torch ([train] extra) not installed; the slow train path is skipped"
)


@pytest.mark.integration
def test_train_module_import_is_torch_free() -> None:
    """Importing the train module must not pull in torch (verified fresh)."""
    import subprocess

    code = "import sys;import mvtsforecast.train;print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


@pytest.mark.integration
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"basket": [], "target": "SPY"}, "at least one ticker"),
        ({"basket": ["SPY", "TLT"], "target": "ZZZ"}, "target"),
        ({"basket": ["SPY"], "target": "SPY", "horizon": 2}, "horizon"),
        ({"basket": ["SPY"], "target": "SPY", "lookback": 0}, "lookback"),
    ],
)
def test_train_rejects_bad_request_before_torch(kwargs: dict[str, object], match: str) -> None:
    """A bad request raises ValidationError before any torch import or training."""
    torch_was_loaded = "torch" in sys.modules
    with pytest.raises(ValidationError, match=match):
        train_pipeline(n_obs=120, **kwargs)  # type: ignore[arg-type]
    if not torch_was_loaded:
        assert "torch" not in sys.modules


@pytest.mark.integration
def test_train_pipeline_orchestration_is_torch_free(tmp_path: Path) -> None:
    """The full pipeline orchestration runs torch-free with an injected fake trainer.

    A numpy-only ``trainer`` stand-in returns a near-zero OOS forecast (the honest
    null) and writes a stub ``.onnx`` file, so every orchestration branch — scoring,
    DM, DSR, the PURE verdict, manifest capture, and metrics.json writing — is
    exercised WITHOUT importing torch. This is the torch-free coverage of train.py
    that the lean serve-container CI relies on.
    """
    import numpy as np

    torch_was_loaded = "torch" in sys.modules
    calls: list[str] = []

    def _fake_trainer(
        model_name: str,
        x_train_scaled: object,
        y_train: object,
        x_test_scaled: object,
        *,
        lookback: int,
        n_features: int,
        seed: int,
        out_dir: Path,
    ) -> tuple[object, Path]:
        calls.append(model_name)
        n_test = int(np.asarray(x_test_scaled).shape[0])
        onnx_path = out_dir / f"{model_name}.onnx"
        onnx_path.write_bytes(b"stub-onnx")  # a placeholder committed artifact
        # A near-zero deep forecast: indistinguishable from naive on the null.
        return np.full(n_test, 1e-7, dtype=np.float64), onnx_path

    result = train_pipeline(
        basket=["SPY", "TLT", "GLD"],
        target="SPY",
        lookback=20,
        horizon=1,
        n_obs=300,
        seed=7,
        artifacts_dir=tmp_path,
        trainer=_fake_trainer,
    )

    assert calls == ["lstm", "patchtst", "transformer"]
    assert set(result.artifact_paths) == {"lstm", "patchtst", "transformer"}
    assert result.n_effective_trials == 3
    # The honest null holds: a near-zero deep forecast cannot beat naive.
    assert result.deep_beats_naive is False

    payload = json.loads(Path(result.metrics_path).read_text(encoding="utf-8"))
    assert payload["deep_beats_naive"] is False
    assert payload["data_source"] == "synthetic"
    for model in ("naive", "lstm", "patchtst", "transformer"):
        row = payload["by_model"][model]
        assert "rmse_return" in row
        assert "dm_pvalue_vs_naive" in row
        assert "deflated_sharpe" in row
        assert "r2" not in row
    assert payload["manifest"]["seed"] == 7

    if not torch_was_loaded:
        assert "torch" not in sys.modules


@pytest.mark.integration
@pytest.mark.slow
@_requires_torch
def test_train_pipeline_writes_onnx_and_metrics(tmp_path: Path) -> None:
    """The full offline pipeline writes 3 ONNX artifacts + metrics.json; null holds."""
    result = train_pipeline(
        basket=["SPY", "TLT", "GLD"],
        target="SPY",
        lookback=20,
        horizon=1,
        n_obs=300,
        seed=7,
        artifacts_dir=tmp_path,
    )

    assert isinstance(result, TrainResult)
    # One committed ONNX artifact per deep model, each actually written to disk.
    assert set(result.artifact_paths) == {"lstm", "patchtst", "transformer"}
    for path in result.artifact_paths.values():
        assert Path(path).is_file()
        assert Path(path).suffix == ".onnx"

    # metrics.json is written, JSON-safe, and records the PURE honest-null verdict.
    metrics_path = Path(result.metrics_path)
    assert metrics_path.is_file()
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["data_source"] == "synthetic"
    assert payload["deep_beats_naive"] is False
    assert result.deep_beats_naive is False
    # Multiplicity count = the FULL architecture grid (3 deep models).
    assert result.n_effective_trials == 3
    assert payload["n_effective_trials"] == 3
    # Every model carries return-space metrics (NO price-level R^2 anywhere).
    for model in ("naive", "lstm", "patchtst", "transformer"):
        row = payload["by_model"][model]
        assert "rmse_return" in row
        assert "directional_accuracy" in row
        assert "r2" not in row and "price_r2" not in row
        assert row["dm_pvalue_vs_naive"] >= 0.0
    # The reproducibility manifest is captured (git state + config hash + seed).
    assert payload["manifest"]["seed"] == 7
    assert "config_hash" in payload["manifest"]


@pytest.mark.integration
@pytest.mark.slow
@_requires_torch
def test_train_pipeline_artifacts_serve_via_onnx_torch_free(tmp_path: Path) -> None:
    """Artifacts exported by train serve through onnxruntime WITHOUT re-importing torch."""
    import numpy as np

    train_pipeline(
        basket=["SPY", "TLT", "GLD"],
        target="SPY",
        lookback=20,
        horizon=1,
        n_obs=300,
        seed=7,
        artifacts_dir=tmp_path,
    )

    # Serve the just-exported LSTM artifact via the onnxruntime wrapper directly.
    from mvtsforecast.models.onnx_runtime import OnnxForecaster

    x = np.zeros((4, 20, 2), dtype=np.float64)
    out = OnnxForecaster("lstm", artifact_path=tmp_path / "lstm.onnx").predict(x)
    assert out.shape == (4,)
    assert np.isfinite(out).all()
