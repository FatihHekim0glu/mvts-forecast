"""Regression guard: the PURE ``deep_beats_naive`` verdict truth table.

``derive_verdict`` MUST read ``deep_beats_naive=True`` ONLY when the best deep
model clears ALL THREE gates simultaneously:

1. the Diebold-Mariano test is significant (``dm_pvalue < alpha``);
2. the DM statistic is signed in the model's favour (``dm_statistic < 0``);
3. the Deflated Sharpe is strictly positive (``deflated_sharpe > 0``).

If ANY gate fails, the verdict is ``NO_SIGNIFICANT_DIFFERENCE`` — the honest,
literature-consistent outcome on noisy daily returns. This is the anti-narration
guard that keeps the README honest; it is pinned here so a regression cannot
flip the headline.
"""

from __future__ import annotations

import pytest

from mvtsforecast._exceptions import ValidationError
from mvtsforecast.evaluation.verdict import Verdict, VerdictResult, derive_verdict

pytestmark = pytest.mark.regression


# (dm_statistic, dm_pvalue, deflated_sharpe) -> deep_beats_naive
_TRUTH_TABLE = [
    # All three gates pass -> the ONLY True row.
    ((-3.0, 0.001, 0.7), True),
    # DM insignificant (p >= alpha) -> False even with a great DSR and sign.
    ((-3.0, 0.20, 0.9), False),
    # DM significant but POSITIVE statistic (model worse) -> False.
    ((3.0, 0.001, 0.9), False),
    # DM significant and negative, but DSR == 0 (not strictly positive) -> False.
    ((-3.0, 0.001, 0.0), False),
    # DM significant and negative, but DSR negative -> False.
    ((-3.0, 0.001, -0.2), False),
    # Borderline: p exactly at alpha is NOT < alpha -> False.
    ((-3.0, 0.05, 0.9), False),
    # Everything fails -> False.
    ((1.5, 0.9, -0.5), False),
]


@pytest.mark.parametrize(("inputs", "expected"), _TRUTH_TABLE)
def test_verdict_truth_table(inputs: tuple[float, float, float], expected: bool) -> None:
    dm_stat, dm_p, dsr = inputs
    result = derive_verdict("patchtst", dm_stat, dm_p, dsr, n_effective_trials=24)
    assert isinstance(result, VerdictResult)
    assert result.deep_beats_naive is expected
    if expected:
        assert result.verdict is Verdict.DEEP_BEATS_NAIVE
    else:
        assert result.verdict is Verdict.NO_SIGNIFICANT_DIFFERENCE
    # The boolean and the enum must always agree.
    assert result.deep_beats_naive == (result.verdict is Verdict.DEEP_BEATS_NAIVE)


def test_verdict_carries_evidence_through() -> None:
    result = derive_verdict("transformer", -2.4, 0.012, 0.33, n_effective_trials=18)
    assert result.best_deep_model == "transformer"
    assert result.dm_pvalue == pytest.approx(0.012)
    assert result.deflated_sharpe == pytest.approx(0.33)
    assert result.n_effective_trials == 18


def test_verdict_alpha_is_respected() -> None:
    # With a stricter alpha, a p that previously cleared 0.05 now fails.
    inputs = (-3.0, 0.03, 0.9)
    assert derive_verdict("m", *inputs, n_effective_trials=4).deep_beats_naive is True
    strict = derive_verdict("m", *inputs, n_effective_trials=4, alpha=0.01)
    assert strict.deep_beats_naive is False


def test_verdict_to_dict_is_json_safe() -> None:
    import json

    result = derive_verdict("lstm", -2.0, 0.04, 0.1, n_effective_trials=6)
    d = result.to_dict()
    json.dumps(d)  # must not raise
    assert d["verdict"] == "deep_beats_naive"
    assert d["deep_beats_naive"] is True
    assert d["n_effective_trials"] == 6


@pytest.mark.parametrize("bad_p", [-0.1, 1.1, float("nan")])
def test_verdict_rejects_out_of_range_pvalue(bad_p: float) -> None:
    with pytest.raises(ValidationError):
        derive_verdict("m", -2.0, bad_p, 0.5, n_effective_trials=4)


def test_verdict_rejects_nonpositive_trials() -> None:
    with pytest.raises(ValidationError):
        derive_verdict("m", -2.0, 0.01, 0.5, n_effective_trials=0)


def test_verdict_rejects_nonfinite_statistic() -> None:
    with pytest.raises(ValidationError):
        derive_verdict("m", float("inf"), 0.01, 0.5, n_effective_trials=4)
    with pytest.raises(ValidationError):
        derive_verdict("m", -2.0, 0.01, float("nan"), n_effective_trials=4)
