"""Typer CLI: ``train`` / ``forecast`` / ``compare`` (typer imported lazily).

The console entrypoint (``mvts-forecast``) exposes three commands:

- ``train``   — the OFFLINE pipeline (synthetic -> walk-forward -> train deep
  models -> ONNX + metrics.json); the only command that pulls in the ``[train]``
  extra (torch).
- ``forecast`` — serve the committed deep models (onnxruntime, NO torch) + the
  live naive/ARIMA baselines on a panel and print the summary.
- ``compare``  — the full deep-vs-naive comparison with the PURE
  ``deep_beats_naive`` verdict.

``typer`` is imported LAZILY inside :func:`build_app` (it lives in the ``[dev]``
extra), so importing this module pulls in NO typer and has no side effects. The
``main`` entrypoint builds and runs the app.

Importing this module has no side effects.
"""

from __future__ import annotations

from typing import Any


def build_app() -> Any:
    """Build and return the Typer application (``typer`` imported lazily here).

    Registers the ``train``, ``forecast``, and ``compare`` commands. Typer is
    imported inside this function so importing :mod:`mvtsforecast.cli` (and the
    package) never imports typer.

    Returns
    -------
    Any
        A configured ``typer.Typer`` instance.

    Raises
    ------
    ImportError
        If ``typer`` (the ``[dev]`` extra) is not installed.
    """
    raise NotImplementedError


def main() -> None:
    """Console-script entrypoint: build the Typer app and run it.

    Wired to the ``mvts-forecast`` console script in ``pyproject.toml``. Builds the
    app via :func:`build_app` and invokes it.

    Raises
    ------
    ImportError
        If ``typer`` (the ``[dev]`` extra) is not installed.
    """
    raise NotImplementedError


if __name__ == "__main__":  # pragma: no cover - module-as-script entrypoint
    main()
