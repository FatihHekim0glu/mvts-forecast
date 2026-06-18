"""PatchTST-style encoder (torch, [train] only) — patching + channel-independence.

A compact reimplementation of the PatchTST idea (Nie et al., 2023): split each
channel's look-back window into PATCHES, embed them, run a small CHANNEL-INDEPENDENT
transformer encoder (every channel shares the same weights and is processed
separately), and read out a single next-step RETURN forecast for the target
channel. RevIN instance-normalization (from the input window only) sits in front.

``torch`` is imported LAZILY inside the builder/trainer (the ``[train]`` extra), so
importing this module pulls in NO torch and has no side effects. The trained graph
is exported to ONNX and served torch-free.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from mvtsforecast._typing import FloatArray, SequenceTensor


@dataclass(frozen=True, slots=True)
class PatchTSTConfig:
    """Immutable hyperparameters for the small PatchTST-style encoder.

    Attributes
    ----------
    look_back:
        Input window length (default 60).
    n_features:
        Number of input channels (processed channel-independently).
    patch_len:
        Length of each patch along the time axis.
    stride:
        Stride between consecutive patches.
    d_model:
        Transformer embedding width.
    n_heads:
        Number of self-attention heads.
    n_layers:
        Number of transformer encoder blocks.
    dropout:
        Dropout probability.
    use_revin:
        Whether RevIN instance-norm (input-window-only) is applied first.
    """

    look_back: int = 60
    n_features: int = 2
    patch_len: int = 16
    stride: int = 8
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.0
    use_revin: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this config."""
        return asdict(self)


def build_patchtst(config: PatchTSTConfig) -> Any:
    """Build (but do not train) the PatchTST-style ``torch.nn.Module``.

    LAZY IMPORT: ``torch`` is imported inside this function. The module patches the
    ``(batch, look_back, n_features)`` input channel-independently, embeds and
    encodes the patches with a small transformer, and emits ``(batch, 1)``.

    Parameters
    ----------
    config:
        The :class:`PatchTSTConfig` hyperparameters.

    Returns
    -------
    Any
        An untrained ``torch.nn.Module``.

    Raises
    ------
    ImportError
        If the ``[train]`` extra (torch) is not installed.
    ValidationError
        If ``patch_len``/``stride`` are inconsistent with ``look_back`` or
        ``d_model`` is not divisible by ``n_heads``.
    """
    raise NotImplementedError


def train_patchtst(
    x_train: SequenceTensor,
    y_train: FloatArray,
    config: PatchTSTConfig,
    *,
    epochs: int = 30,
    lr: float = 1e-3,
    seed: int = 7,
) -> Any:
    """Train the PatchTST-style encoder on the (pre-scaled) train-fold windows.

    LAZY IMPORT: ``torch`` is imported inside this function (the ``[train]`` extra).
    Training is OFFLINE — never on the request path. The fitted module is exported
    to ONNX for the torch-free serve path.

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
