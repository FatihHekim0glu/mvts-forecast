"""Unit tests for :mod:`mvtsforecast.cli` — ``--help`` + a torch-free smoke.

The CLI must build via Typer (lazy import), expose ``train`` / ``forecast`` /
``compare`` in ``--help``, and run a tiny synthetic naive+ARIMA ``compare`` (and
``forecast``) WITHOUT importing torch. The ``train`` command (the only [train]
path) is not exercised here. These tests assert the honest null too: on the
synthetic panel ``deep beats naive`` prints ``False``.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
from typer.testing import CliRunner

import mvtsforecast.cli as cli_mod
from mvtsforecast._exceptions import ValidationError
from mvtsforecast.cli import _onnx_forecasts, _ordered, build_app, compare, forecast, train

runner = CliRunner()


def _assert_action_did_not_import_torch(torch_was_loaded: bool) -> None:
    """Assert the action under test did not newly import torch.

    The default ``pytest`` run executes the ``slow`` torch parity/deep tests in
    the same process, which legitimately leaves ``torch`` in ``sys.modules``. So
    a bare ``"torch" not in sys.modules`` check is order-dependent. Instead we
    snapshot whether torch was already loaded *before* the action and only assert
    the action itself did not pull it in. (Import purity of the modules is proven
    independently, in a fresh interpreter, by the ``*_import_is_torch_free`` tests.)
    """
    if torch_was_loaded:
        return
    assert "torch" not in sys.modules


def test_build_app_returns_fresh_typer_instances() -> None:
    import typer

    app_a = build_app()
    app_b = build_app()
    assert isinstance(app_a, typer.Typer)
    assert app_a is not app_b


def test_help_lists_all_commands() -> None:
    result = runner.invoke(build_app(), ["--help"])
    assert result.exit_code == 0
    for command in ("train", "forecast", "compare"):
        assert command in result.stdout


@pytest.mark.parametrize("command", ["train", "forecast", "compare"])
def test_subcommand_help(command: str) -> None:
    result = runner.invoke(build_app(), [command, "--help"])
    assert result.exit_code == 0
    assert "--seed" in result.stdout


def test_compare_smoke_naive_arima_no_torch() -> None:
    """A tiny synthetic naive+ARIMA compare runs torch-free and prints the verdict."""
    torch_was_loaded = "torch" in sys.modules
    result = runner.invoke(
        build_app(),
        ["compare", "--models", "naive,arima", "--n-obs", "120", "--lookback", "20"],
    )
    assert result.exit_code == 0, result.stdout
    assert "naive" in result.stdout
    assert "arima" in result.stdout
    # Honest null: deep models are not even served here, so the verdict is False.
    assert "deep beats naive   : False" in result.stdout
    # The serve/train path is NEVER triggered: torch must stay unimported.
    _assert_action_did_not_import_torch(torch_was_loaded)


def test_forecast_smoke_prints_metric_table() -> None:
    torch_was_loaded = "torch" in sys.modules
    result = runner.invoke(
        build_app(),
        ["forecast", "--n-obs", "120", "--lookback", "20"],
    )
    assert result.exit_code == 0, result.stdout
    assert "RMSE" in result.stdout
    assert "naive" in result.stdout
    assert "arima" in result.stdout
    _assert_action_did_not_import_torch(torch_was_loaded)


def test_compare_function_naive_only() -> None:
    code = compare(models="naive", n_obs=80, lookback=16)
    assert code == 0


def test_compare_unknown_model_returns_error(capsys: pytest.CaptureFixture[str]) -> None:
    code = compare(models="bogus", n_obs=80, lookback=16)
    assert code == 1
    assert "unknown model" in capsys.readouterr().out


def test_compare_bad_target_returns_error(capsys: pytest.CaptureFixture[str]) -> None:
    code = compare(target="ZZZ", models="naive,arima", n_obs=80, lookback=16)
    assert code == 1
    assert "target" in capsys.readouterr().out


def test_forecast_function_returns_zero() -> None:
    assert forecast(n_obs=80, lookback=16) == 0


def test_compare_via_runner_naive_only_function_path() -> None:
    """The Typer ``compare`` wrapper exits 0 for a naive-only run."""
    result = runner.invoke(build_app(), ["compare", "--models", "naive", "--n-obs", "80"])
    assert result.exit_code == 0


def test_ordered_puts_naive_first_and_keeps_extras() -> None:
    forecasts = {"arima": np.zeros(2), "naive": np.zeros(2), "mystery": np.zeros(2)}
    assert _ordered(forecasts) == ["naive", "arima", "mystery"]


def test_train_delegates_and_prints_summary(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``train`` delegates to ``train_pipeline`` (mocked) WITHOUT importing torch."""
    from mvtsforecast.train import TrainResult

    fake = TrainResult(
        artifact_paths={"lstm": "/tmp/lstm.onnx", "patchtst": "/tmp/patchtst.onnx"},
        metrics_path="/tmp/metrics.json",
        n_effective_trials=6,
        deep_beats_naive=False,
        manifest={},
    )
    monkeypatch.setattr("mvtsforecast.train.train_pipeline", lambda **_: fake)

    torch_was_loaded = "torch" in sys.modules
    code = train(basket="SPY,TLT,GLD", target="SPY", n_obs=120)

    assert code == 0
    out = capsys.readouterr().out
    assert "effective trials   : 6" in out
    assert "deep beats naive   : False" in out
    assert "artifact[lstm" in out
    _assert_action_did_not_import_torch(torch_was_loaded)


def test_train_handles_library_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _boom(**_: object) -> None:
        raise ValidationError("bad config")

    monkeypatch.setattr("mvtsforecast.train.train_pipeline", _boom)

    assert train(n_obs=120) == 1
    assert "error: bad config" in capsys.readouterr().out


def test_train_via_runner_exits_with_pipeline_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(**_: object) -> None:
        raise ValidationError("nope")

    monkeypatch.setattr("mvtsforecast.train.train_pipeline", _boom)
    result = runner.invoke(build_app(), ["train", "--n-obs", "120"])
    assert result.exit_code == 1
    assert "error: nope" in result.stdout


def test_onnx_forecasts_empty_when_no_deep_requested() -> None:
    assert _onnx_forecasts(["naive", "arima"], np.zeros((2, 4, 3))) == {}


def test_onnx_forecasts_empty_when_x_is_none() -> None:
    assert _onnx_forecasts(["lstm"], None) == {}


def test_onnx_forecasts_skips_missing_artifacts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deep models whose ONNX artifact is absent are skipped (no torch, no error)."""

    class _MissingPath:
        def is_file(self) -> bool:
            return False

    monkeypatch.setattr(cli_mod, "_DEEP_MODELS", ("lstm", "patchtst", "transformer"))
    monkeypatch.setattr(
        "mvtsforecast.models.onnx_runtime.default_artifact_path",
        lambda _name: _MissingPath(),
    )
    torch_was_loaded = "torch" in sys.modules
    assert _onnx_forecasts(["lstm", "patchtst"], np.zeros((2, 4, 3))) == {}
    _assert_action_did_not_import_torch(torch_was_loaded)


def test_onnx_forecasts_serves_present_artifact(monkeypatch: pytest.MonkeyPatch) -> None:
    """A present artifact is served via the lazy ONNX path (mocked, torch-free)."""

    class _PresentPath:
        def is_file(self) -> bool:
            return True

    monkeypatch.setattr(
        "mvtsforecast.models.onnx_runtime.default_artifact_path",
        lambda _name: _PresentPath(),
    )
    monkeypatch.setattr(
        "mvtsforecast.serve.forecast_from_onnx",
        lambda _name, _x: np.array([0.0, 0.0]),
    )
    torch_was_loaded = "torch" in sys.modules
    out = _onnx_forecasts(["lstm"], np.zeros((2, 4, 3)))
    assert list(out) == ["lstm"]
    assert out["lstm"].tolist() == [0.0, 0.0]
    _assert_action_did_not_import_torch(torch_was_loaded)


def test_onnx_forecasts_skips_corrupt_artifact(monkeypatch: pytest.MonkeyPatch) -> None:
    """A corrupt / signature-mismatched artifact is skipped, not fatal."""
    from mvtsforecast._exceptions import ArtifactError

    class _PresentPath:
        def is_file(self) -> bool:
            return True

    def _raise(_name: str, _x: object) -> object:
        raise ArtifactError("bad signature")

    monkeypatch.setattr(
        "mvtsforecast.models.onnx_runtime.default_artifact_path",
        lambda _name: _PresentPath(),
    )
    monkeypatch.setattr("mvtsforecast.serve.forecast_from_onnx", _raise)
    assert _onnx_forecasts(["lstm"], np.zeros((2, 4, 3))) == {}


def test_compare_empty_basket_returns_error(capsys: pytest.CaptureFixture[str]) -> None:
    code = compare(basket=" , ,", models="naive", n_obs=80, lookback=16)
    assert code == 1
    assert "at least one ticker" in capsys.readouterr().out


def test_forecast_bad_target_returns_error(capsys: pytest.CaptureFixture[str]) -> None:
    code = forecast(target="ZZZ", n_obs=80, lookback=16)
    assert code == 1
    assert "target" in capsys.readouterr().out


def test_cli_import_is_torch_free() -> None:
    """Importing the CLI must not pull in torch / onnxruntime (import purity).

    Verified in a FRESH interpreter so the assertion is robust to ``sys.modules``
    pollution from earlier ``slow`` torch tests in the same pytest process.
    """
    import subprocess

    code = (
        "import sys;"
        "import mvtsforecast.cli;"
        "bad=[m for m in ('torch','onnxruntime') if m in sys.modules];"
        "print(','.join(bad))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "", f"importing cli leaked: {out.stdout.strip()}"
