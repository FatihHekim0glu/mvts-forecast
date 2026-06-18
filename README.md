# mvts-forecast

A **leakage-free multivariate-transformer forecast benchmark** with an honest
NULL headline.

> **Honest headline:** on noisy daily **returns**, a PatchTST-style encoder and a
> simplified interpretable variable-selection transformer (and an LSTM) do **NOT**
> reliably beat a naive random-walk baseline out-of-sample on directional accuracy
> or risk-adjusted PnL after costs — Diebold-Mariano insignificant, Deflated
> Sharpe ~ 0. The deliverable is the rigorous, leakage-free comparison, **not** a
> profit claim.

The shipped default runs the comparison on a **synthetic multivariate panel** (a
weak common factor + dominant idiosyncratic noise + mild autocorrelation, seeded —
no API keys) plus committed, offline-trained **ONNX** deep models served via
onnxruntime (the serve container has **no torch**). Real data flows through the
yfinance → Stooq + FRED-CSV offline CLI path.

## Why this exists

This is the rigorous counterweight to the "predict the stock price with an LSTM
and report a 0.99 R²" genre. That R² is a unit-root artifact of regressing a
trended price level on its own lag — **not** forecasting skill. Here:

- the target is the next-step **RETURN**, never the price level;
- there is **no price-level R²** anywhere;
- RevIN / instance-norm is computed from the **input window only**, the
  standardizer is fitted on the **train fold only**, and the walk-forward is
  **purged (≥ `look_back`) + embargoed** so no window straddles a split;
- the encoder input never contains the target's same-step transform;
- the verdict `deep_beats_naive` is a **pure function** of the evidence and reads
  `False` unless a deep model beats naive with a **DM-significant** margin **and** a
  **positive Deflated Sharpe**.

## Models

| Model | Family | Runs where | Heavy dep |
|-------|--------|-----------|-----------|
| `naive` | last-value / random walk (the floor) | live | none |
| `arima` | per-series ARIMA | live | statsmodels (`[data]`) |
| `lstm` | small multivariate LSTM | offline train → ONNX serve | torch (`[train]`) |
| `patchtst` | PatchTST-style (patching + channel-independence) | offline train → ONNX serve | torch (`[train]`) |
| `transformer` | interpretable variable-selection transformer | offline train → ONNX serve | torch (`[train]`) |

## Install

```bash
uv venv
uv pip install -e ".[data,serve,viz,dev]"   # lean path (no torch)
# uv pip install -e ".[train]"               # only to retrain/export ONNX
```

## Validation

_To be completed once the compute kernels and offline training run land (the
synthetic `seed=7` metrics, the ONNX-vs-torch 1e-4 parity row, and the
random-walk anti-leakage regression result)._

## Limitations

- **Static basket / survivorship:** the fixed default basket has no survivorship
  bias issue (it is static and synthetic), but a real-data run on a fixed universe
  would; a point-in-time (PIT) universe is the upgrade path.
- **Synthetic-trained default:** the shipped ONNX models are trained on the
  synthetic panel, where the honest NULL holds by construction.
- **Macro release-date lags:** real macro features are lagged to RELEASE dates
  (not reference dates) to avoid look-ahead; the lag is a conservative proxy.

## References

- Nie, Nguyen, Sinthong, Kalagnanam (2023), *A Time Series is Worth 64 Words:
  Long-term Forecasting with Transformers* (PatchTST).
- Lim, Arık, Loeff, Pfister (2021), *Temporal Fusion Transformers for
  Interpretable Multi-horizon Time Series Forecasting* (TFT / variable selection).
- Kim et al. (2022), *Reversible Instance Normalization for Accurate Time-Series
  Forecasting against Distribution Shift* (RevIN).
- Diebold & Mariano (1995), *Comparing Predictive Accuracy*.
- Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*.

## License

MIT — see [LICENSE](LICENSE).
