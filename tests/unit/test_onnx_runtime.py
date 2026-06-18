"""Unit tests for the ONNX SERVE path (onnxruntime, NEVER torch).

These exercise :class:`mvtsforecast.models.onnx_runtime.OnnxForecaster` and
:func:`default_artifact_path` WITHOUT torch: a tiny, shape-matched ONNX graph is
built directly with the ``onnx`` builder (a fixture), then loaded + run through
onnxruntime. The serve path must:

- resolve each deep-model name to its shipped artifact filename (pure path math);
- load a valid artifact lazily and run a ``(N, look_back, n_features) -> (N,)``
  forward pass;
- short-circuit an EMPTY test slice to an empty vector without touching the session;
- raise :class:`ArtifactError` for a missing artifact or a mis-shaped input;
- import NO torch anywhere on this path (a subprocess purity guard).

``onnxruntime`` and ``onnx`` are the lean ``[serve]`` / ``[dev]`` deps; torch is
never installed in the serve container, so this whole module runs torch-free.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mvtsforecast._exceptions import ArtifactError, ValidationError
from mvtsforecast.models.onnx_runtime import (
    ARTIFACT_FILENAMES,
    ONNX_MODEL_NAMES,
    OnnxForecaster,
    default_artifact_path,
)

LOOK_BACK, N_FEATURES = 60, 2


def _build_identity_mean_onnx(path: Path, *, look_back: int, n_features: int) -> Path:
    """Build a tiny ``(N, look_back, n_features) -> (N, 1)`` ONNX graph (no torch).

    The graph reduces each window to the mean of its first feature — a deterministic,
    shape-matched stand-in for a real exported deep model, sufficient to validate the
    onnxruntime serve wrapper end-to-end without the heavy ``[train]`` extra.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    inp = helper.make_tensor_value_info(
        "sequence", TensorProto.FLOAT, [None, look_back, n_features]
    )
    out = helper.make_tensor_value_info("return_hat", TensorProto.FLOAT, [None, 1])
    initializers = [
        numpy_helper.from_array(np.array([0], dtype=np.int64), "start0"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "end1"),
        numpy_helper.from_array(np.array([2], dtype=np.int64), "feataxis"),
    ]
    nodes = [
        # Take feature 0 -> (N, look_back, 1), then mean over the time axis -> (N, 1).
        # In opset 17 ReduceMean takes ``axes`` as an attribute (not a second input).
        helper.make_node("Slice", ["sequence", "start0", "end1", "feataxis"], ["feat0"]),
        helper.make_node("ReduceMean", ["feat0"], ["return_hat"], axes=[1], keepdims=0),
    ]
    graph = helper.make_graph(nodes, "identity_mean", [inp], [out], initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return path


@pytest.fixture
def tiny_artifact(tmp_path: Path) -> Path:
    """A committed-shaped tiny ONNX artifact for the default look-back/feature count."""
    return _build_identity_mean_onnx(
        tmp_path / "tiny.onnx", look_back=LOOK_BACK, n_features=N_FEATURES
    )


# --------------------------------------------------------------------------- #
# default_artifact_path                                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.parametrize("model_name", ONNX_MODEL_NAMES)
def test_default_artifact_path_resolves_known_models(model_name: str) -> None:
    """Each known deep-model name maps to its committed ``artifacts/<file>.onnx``."""
    path = default_artifact_path(model_name)
    assert path.name == ARTIFACT_FILENAMES[model_name]
    assert path.parent.name == "artifacts"


@pytest.mark.unit
def test_default_artifact_path_rejects_unknown_model() -> None:
    """An unknown model name raises a clear ValidationError listing the valid set."""
    with pytest.raises(ValidationError, match="unknown deep-model name"):
        default_artifact_path("gru")


# --------------------------------------------------------------------------- #
# OnnxForecaster: construction + lazy load + predict                          #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_forecaster_records_name_and_path_without_loading(tiny_artifact: Path) -> None:
    """Constructing a forecaster is cheap: it records name/path and loads nothing."""
    fc = OnnxForecaster("lstm", artifact_path=tiny_artifact)
    assert fc.model_name == "lstm"
    assert fc.artifact_path == tiny_artifact
    assert fc._session is None  # session is created lazily on first load/predict


@pytest.mark.unit
def test_forecaster_default_path_when_unspecified() -> None:
    """With no explicit path the forecaster resolves the shipped default artifact."""
    fc = OnnxForecaster("patchtst")
    assert fc.artifact_path == default_artifact_path("patchtst")


@pytest.mark.unit
def test_predict_runs_forward_pass_and_returns_1d(tiny_artifact: Path) -> None:
    """A valid shape-matched input yields a finite ``(n_samples,)`` forecast vector."""
    fc = OnnxForecaster("lstm", artifact_path=tiny_artifact)
    rng = np.random.default_rng(7)
    x = rng.standard_normal((5, LOOK_BACK, N_FEATURES)).astype(np.float64)

    out = fc.predict(x)
    assert out.shape == (5,)
    assert out.dtype == np.float64
    assert np.isfinite(out).all()
    # The fixture graph returns the per-window mean of feature 0 — verify it matches.
    expected = x[:, :, 0].mean(axis=1)
    assert np.allclose(out, expected, atol=1e-5)


@pytest.mark.unit
def test_predict_is_idempotent_on_session(tiny_artifact: Path) -> None:
    """Repeated predicts reuse a single lazily-created session (no re-init)."""
    fc = OnnxForecaster("lstm", artifact_path=tiny_artifact)
    x = np.zeros((3, LOOK_BACK, N_FEATURES), dtype=np.float64)
    fc.predict(x)
    session_after_first = fc._session
    assert session_after_first is not None
    fc.predict(x)
    assert fc._session is session_after_first


@pytest.mark.unit
def test_load_is_idempotent(tiny_artifact: Path) -> None:
    """Calling ``load`` twice returns ``self`` and keeps the same session object."""
    fc = OnnxForecaster("lstm", artifact_path=tiny_artifact)
    assert fc.load() is fc
    session = fc._session
    assert fc.load() is fc
    assert fc._session is session


@pytest.mark.unit
def test_predict_empty_slice_short_circuits(tiny_artifact: Path) -> None:
    """An empty test slice returns an empty vector WITHOUT initializing the session."""
    fc = OnnxForecaster("lstm", artifact_path=tiny_artifact)
    out = fc.predict(np.empty((0, LOOK_BACK, N_FEATURES), dtype=np.float64))
    assert out.shape == (0,)
    assert fc._session is None  # short-circuit never touched onnxruntime


@pytest.mark.unit
def test_predict_rejects_non_3d_input(tiny_artifact: Path) -> None:
    """A 2-D input raises ArtifactError before any forward pass."""
    fc = OnnxForecaster("lstm", artifact_path=tiny_artifact)
    with pytest.raises(ArtifactError, match="3-D"):
        fc.predict(np.zeros((5, LOOK_BACK), dtype=np.float64))


@pytest.mark.unit
def test_load_missing_artifact_raises(tmp_path: Path) -> None:
    """A missing artifact file raises ArtifactError on load."""
    fc = OnnxForecaster("lstm", artifact_path=tmp_path / "does_not_exist.onnx")
    with pytest.raises(ArtifactError, match="not found"):
        fc.load()


@pytest.mark.unit
def test_predict_wrong_feature_count_raises(tmp_path: Path) -> None:
    """An input whose feature axis mismatches the graph signature raises ArtifactError."""
    artifact = _build_identity_mean_onnx(
        tmp_path / "tiny.onnx", look_back=LOOK_BACK, n_features=N_FEATURES
    )
    fc = OnnxForecaster("lstm", artifact_path=artifact)
    # 3 features where the graph fixed 2 -> onnxruntime rejects the shape.
    bad = np.zeros((4, LOOK_BACK, N_FEATURES + 1), dtype=np.float64)
    with pytest.raises(ArtifactError, match="forward pass failed"):
        fc.predict(bad)


@pytest.mark.unit
def test_load_corrupt_artifact_is_normalized_to_artifact_error(tmp_path: Path) -> None:
    """A present-but-corrupt ``.onnx`` file surfaces as a normalized ArtifactError.

    The file exists (so the missing-file branch is skipped) but is not a valid ONNX
    model, so onnxruntime raises — the wrapper must wrap that as ArtifactError.
    """
    corrupt = tmp_path / "corrupt.onnx"
    corrupt.write_bytes(b"not a real onnx graph")
    fc = OnnxForecaster("lstm", artifact_path=corrupt)
    with pytest.raises(ArtifactError, match="failed to initialize onnxruntime session"):
        fc.load()


# --------------------------------------------------------------------------- #
# serve path imports NO torch                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_serve_path_imports_no_torch(tiny_artifact: Path) -> None:
    """Importing + running the ONNX serve wrapper must not import torch."""
    code = (
        "import sys;"
        "import numpy as np;"
        "from mvtsforecast.models.onnx_runtime import OnnxForecaster;"
        f"fc=OnnxForecaster('lstm', artifact_path={str(tiny_artifact)!r});"
        f"fc.predict(np.zeros((2, {LOOK_BACK}, {N_FEATURES})));"
        "print('torch' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "False", (
        f"the ONNX serve path imported torch: stdout={out.stdout!r} stderr={out.stderr!r}"
    )
