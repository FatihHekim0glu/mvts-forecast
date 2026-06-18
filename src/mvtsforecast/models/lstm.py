"""Small multivariate LSTM (torch, [train] only) — lazy, never imported at load.

A compact LSTM encoder over the ``(look_back, n_features)`` window that emits a
single next-step RETURN forecast for the target channel. ``torch`` is imported
LAZILY inside the builder/trainer (the ``[train]`` extra), so importing this
module pulls in NO torch and has no side effects — the import-purity guard the
Stock-Price-Forecast footgun violated. The trained graph is exported to ONNX and
served, torch-free, via :mod:`mvtsforecast.models.onnx_runtime`.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from mvtsforecast._typing import FloatArray, SequenceTensor


@dataclass(frozen=True, slots=True)
class LstmConfig:
    """Immutable hyperparameters for the small multivariate LSTM.

    Attributes
    ----------
    look_back:
        Input window length (default 60).
    n_features:
        Number of input channels (basket size, minus the dropped target column).
    hidden_size:
        LSTM hidden width.
    num_layers:
        Number of stacked LSTM layers.
    dropout:
        Inter-layer dropout probability.
    use_revin:
        Whether the encoder consumes RevIN-normalized windows.
    """

    look_back: int = 60
    n_features: int = 2
    hidden_size: int = 32
    num_layers: int = 1
    dropout: float = 0.0
    use_revin: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this config."""
        return asdict(self)


def build_lstm(config: LstmConfig) -> Any:
    """Build (but do not train) the small LSTM ``torch.nn.Module``.

    LAZY IMPORT: ``torch`` is imported inside this function. The returned module
    maps a ``(batch, look_back, n_features)`` tensor to a ``(batch, 1)`` next-step
    return forecast. Returned as ``Any`` so the package never references a torch
    type at module scope.

    Parameters
    ----------
    config:
        The :class:`LstmConfig` hyperparameters.

    Returns
    -------
    Any
        An untrained ``torch.nn.Module``.

    Raises
    ------
    ImportError
        If the ``[train]`` extra (torch) is not installed.
    ValidationError
        If any config field is out of range.
    """
    raise NotImplementedError


def train_lstm(
    x_train: SequenceTensor,
    y_train: FloatArray,
    config: LstmConfig,
    *,
    epochs: int = 30,
    lr: float = 1e-3,
    seed: int = 7,
) -> Any:
    """Train the small LSTM on the (pre-scaled) train-fold windows.

    LAZY IMPORT: ``torch`` is imported inside this function (the ``[train]`` extra).
    Training is OFFLINE — never invoked on the request path. The fitted module is
    later exported to ONNX for the torch-free serve path.

    Parameters
    ----------
    x_train:
        Train-fold sequence tensor ``(n_train, look_back, n_features)``.
    y_train:
        Train-fold next-step target returns ``(n_train,)``.
    config:
        The model hyperparameters.
    epochs, lr:
        Optimization budget and learning rate.
    seed:
        Seeds torch's RNG for a reproducible fit.

    Returns
    -------
    Any
        The trained ``torch.nn.Module``.

    Raises
    ------
    ImportError
        If torch is not installed.
    ValidationError
        If the tensors are mis-shaped or empty.
    """
    raise NotImplementedError
