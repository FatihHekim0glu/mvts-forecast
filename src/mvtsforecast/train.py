"""Offline training pipeline: synthetic -> walk-forward -> train -> ONNX + metrics.

This is the OFFLINE path (the ``[train]`` extra, torch). It builds the synthetic
panel, windows it leakage-safely (RevIN from the input window only, scaler fitted
on the TRAIN fold only, purge >= ``look_back`` + embargo), trains the small deep
models (LSTM, PatchTST, the interpretable transformer) over the walk-forward
folds, exports each fitted graph to ONNX (parity-checked to the torch forward pass
to 1e-4), precomputes the OOS forecasts, scores everything in return space, and
writes a committed ``artifacts/metrics.json`` + a :class:`RunManifest`. The honest
multiplicity count ``n_effective_trials`` equals the FULL config grid
(#architectures x #HP configs).

torch is imported LAZILY inside the trainer calls, so importing this module pulls
in NO torch and has no side effects. NEVER invoked on the request path.

Importing this module has no side effects.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mvtsforecast.data.synthetic import DEFAULT_BASKET


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Immutable summary of an offline training run.

    Attributes
    ----------
    artifact_paths:
        Map of deep-model name -> exported ``.onnx`` artifact path.
    metrics_path:
        Path to the committed ``metrics.json``.
    n_effective_trials:
        The FULL multiplicity count (#architectures x #HP configs) used for DSR.
    deep_beats_naive:
        The PURE verdict at train time (expected ``False`` on the synthetic panel).
    manifest:
        The reproducibility manifest dict (git SHA, dirty flag, config hash, seed).
    """

    artifact_paths: dict[str, str]
    metrics_path: str
    n_effective_trials: int
    deep_beats_naive: bool
    manifest: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result."""
        return asdict(self)


def train_pipeline(
    *,
    basket: Sequence[str] = DEFAULT_BASKET,
    target: str = "SPY",
    lookback: int = 60,
    horizon: int = 1,
    n_obs: int = 1500,
    seed: int = 7,
    artifacts_dir: str | Path | None = None,
) -> TrainResult:
    """Run the offline train -> ONNX-export -> metrics pipeline end-to-end.

    LAZY IMPORT: ``torch`` (and the ``onnx`` exporter) are imported inside the
    per-model trainer calls — the ``[train]`` extra. Builds the synthetic panel,
    windows it leakage-safely, trains each deep model over the purged walk-forward
    folds, exports to ONNX with a 1e-4 parity check, precomputes OOS forecasts,
    scores in return space, derives the pure verdict, and writes the committed
    artifacts + ``metrics.json`` + manifest.

    Parameters
    ----------
    basket:
        Asset tickers for the synthetic panel.
    target:
        The forecast target column.
    lookback:
        Window length (default 60).
    horizon:
        Forecast horizon (1 or 5).
    n_obs:
        Number of synthetic observations to generate.
    seed:
        Master RNG seed.
    artifacts_dir:
        Output directory for the ``.onnx`` files + ``metrics.json``; ``None`` =>
        the package's ``artifacts/`` directory.

    Returns
    -------
    TrainResult
        Paths, multiplicity count, and the (expected ``False``) honest verdict.

    Raises
    ------
    ImportError
        If the ``[train]`` extra (torch) is not installed.
    ValidationError
        If the request is invalid (target absent, bad horizon/lookback).
    """
    raise NotImplementedError


def export_onnx(
    model: Any,
    sample_window: Any,
    out_path: str | Path,
    *,
    rtol: float = 1e-4,
) -> Path:
    """Export a trained torch module to ONNX and verify forward-pass parity.

    LAZY IMPORT: ``torch`` and ``onnxruntime`` are imported inside this function.
    Runs the torch forward pass and the exported ONNX forward pass on
    ``sample_window`` and asserts they agree to ``rtol`` (the parity gate). The
    serve path then uses ONLY the ONNX graph (no torch).

    Parameters
    ----------
    model:
        A trained ``torch.nn.Module``.
    sample_window:
        A ``(1, look_back, n_features)`` example input defining the export
        signature and driving the parity check.
    out_path:
        Destination ``.onnx`` path.
    rtol:
        Relative tolerance for the torch-vs-ONNX parity assertion (default 1e-4).

    Returns
    -------
    pathlib.Path
        The written ``.onnx`` path.

    Raises
    ------
    ImportError
        If torch / onnxruntime are not installed.
    ValidationError
        If the torch and ONNX forward passes disagree beyond ``rtol``.
    """
    raise NotImplementedError
