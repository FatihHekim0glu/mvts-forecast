"""Import-purity guard: ``import mvtsforecast`` must pull in NO heavy dependency.

The Stock-Price-Forecast footgun was a package that imported a training framework
(and ran code) at module load. Here, importing the package — and every public
submodule — must NOT import torch, onnxruntime, statsmodels, plotly, or typer.
Those are imported LAZILY inside the functions that need them.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

#: Modules that must never be imported as a side effect of ``import mvtsforecast``.
_FORBIDDEN = ("torch", "onnxruntime", "statsmodels", "plotly", "typer", "yfinance")


@pytest.mark.unit
def test_import_mvtsforecast_pulls_in_no_heavy_dependency() -> None:
    """A fresh interpreter importing the package must not load any heavy module."""
    code = (
        "import sys;"
        "import mvtsforecast;"
        "bad=[m for m in "
        f"{_FORBIDDEN!r}"
        " if m in sys.modules];"
        "print(','.join(bad))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    leaked = out.stdout.strip()
    assert leaked == "", f"import mvtsforecast leaked heavy modules: {leaked}"


@pytest.mark.unit
def test_public_api_is_importable() -> None:
    """The curated ``__all__`` names must all resolve on the package."""
    import mvtsforecast

    missing = [name for name in mvtsforecast.__all__ if not hasattr(mvtsforecast, name)]
    assert missing == [], f"names in __all__ not bound on the package: {missing}"
