# Normalized EMA Momentum — Exploratory Backtesting

[![CI](https://github.com/aassiiss/EMA-Angle-Optimization/actions/workflows/tests.yml/badge.svg)](https://github.com/aassiiss/EMA-Angle-Optimization/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This repository evaluates whether normalized EMA-momentum filters reduce
low-momentum crossover participation under delayed execution and transaction
costs. Current results are mixed and should be interpreted as exploratory.

The experiment compares percentage EMA change (price-scale normalization), log EMA
change per elapsed hour (time-frequency normalization), and log EMA change divided
by trailing return volatility. It also includes controlled ablations and
independently optimized standard EMA, ROC, MACD, time-series-momentum, and
buy-and-hold baselines.

## Accounting model

All assets use fully collateralized, continuously divisible synthetic 1x exposure.
Gold (`GC=F`) and silver (`SI=F`) results are **not directly executable CME futures
performance**: contract multipliers, integer contracts, margin, contract-specific
fees, and independent rollover handling are outside the model. Commission and
slippage are assessed on every position change. Equity is marked to market on every
bar and reconciles exactly to compounded net returns.

## Validation design

| Fold | Training | Testing |
|---|---:|---:|
| 1 | first 50% | next 10% |
| 2 | first 60% | next 10% |
| 3 | first 70% | next 10% |
| 4 | first 80% | next 10% |
| 5 | first 90% | final 10% |

Each model is selected on training data only. Prior bars remain available for EMA
warm-up, while execution is prohibited before each test block. Test-block returns
are concatenated into synchronized out-of-sample series.

Inference uses stationary-bootstrap confidence intervals and p-values on daily
return differences, Holm correction across asset-frequency tests, and a
stationary-bootstrap max-t model-comparison diagnostic over the complete fixed
120-candidate universe. This diagnostic is not labeled as Hansen's SPA test.

The one-minute datasets span too few distinct days for defensible daily
time-series inference. They are retained as descriptive walk-forward results but
excluded from confidence intervals, p-values, multiple-testing adjustment, and
aggregate inference; missing inference is never replaced by zero.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python main.py
python audit_verification.py
```

Cached Yahoo Finance OHLCV files are stored under `data/`. Generated synchronized
returns and summary tables are written under `results/`.

## Important outputs

- `results/final_walk_forward/bar_returns/`: timestamped position, return, cost,
  equity, and drawdown ledgers;
- `results/final_walk_forward/csvs/`: OOS summaries, baselines, ablations,
  normalization comparisons, inference, tail risk, cost sensitivity, and the
  leave-Silver-1h-out aggregate;
- `results/final_walk_forward/logs/experiment_metadata.json`: configuration,
  data, result hashes, source commit, dependency versions, and random seeds;
- `results/legacy_invalid/`: preserved outputs produced before accounting and
  inference hardening.

## Tests

The regression suite covers scale invariance, transformation equivalence,
look-ahead isolation, entry/exit costs, long/short behavior, terminal liquidation,
mark-to-market drawdown, equity reconciliation, fold separation, daily risk-metric
annualization, and deterministic output hashes.

## License and citation

Released under the MIT License. Citation metadata is provided in `CITATION.cff`.
