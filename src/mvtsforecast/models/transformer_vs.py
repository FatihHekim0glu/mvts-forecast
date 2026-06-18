"""Interpretable variable-selection transformer (torch, [train] only) — lazy.

A simplified, interpretable transformer in the spirit of the Temporal Fusion
Transformer (Lim et al., 2021): a VARIABLE-SELECTION layer learns per-feature
gating weights (so the relative importance of each basket channel is readable),
followed by a small self-attention encoder over the look-back window and a
single next-step RETURN read-out for the target channel. RevIN instance-norm
(input-window-only) sits in front.

The variable-selection weights are exposed for interpretation, but the honest
headline is unchanged: on noisy daily returns this model does NOT reliably beat
naive. ``torch`` is imported LAZILY inside the builder/trainer (the ``[train]``
extra), so importing this module pulls in NO torch and has no side effects.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from mvtsforecast._typing import FloatArray, SequenceTensor


@dataclass(frozen=True, slots=True)
class TransformerVSConfig:
    """Immutable hyperparameters for the interpretable variable-selection transformer.

    Attributes
    ----------
    look_back:
        Input window length (default 60).
    n_features:
        Number of input channels gated by the variable-selection layer.
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
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 1
    dropout: float = 0.0
    use_revin: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this config."""
        return asdict(self)


def build_transformer_vs(config: TransformerVSConfig) -> Any:
    """Build (but do not train) the variable-selection transformer ``nn.Module``.

    LAZY IMPORT: ``torch`` is imported inside this function. The module gates the
    ``(batch, look_back, n_features)`` input per-feature (the interpretable
    variable-selection weights), encodes the gated sequence with a small
    self-attention stack, and emits ``(batch, 1)``.

    Parameters
    ----------
    config:
        The :class:`TransformerVSConfig` hyperparameters.

    Returns
    -------
    Any
        An untrained ``torch.nn.Module`` exposing its variable-selection weights.

    Raises
    ------
    ImportError
        If the ``[train]`` extra (torch) is not installed.
    ValidationError
        If ``d_model`` is not divisible by ``n_heads`` or a field is out of range.
    """
    raise NotImplementedError


def train_transformer_vs(
    x_train: SequenceTensor,
    y_train: FloatArray,
    config: TransformerVSConfig,
    *,
    epochs: int = 30,
    lr: float = 1e-3,
    seed: int = 7,
) -> Any:
    """Train the variable-selection transformer on the (pre-scaled) train windows.

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
