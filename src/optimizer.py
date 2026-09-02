"""Anchored walk-forward selection with fair, independently optimized baselines."""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .backtest import metrics_from_ledger, run_backtest
from .config import (ANGLE_THRESHOLDS, COMMISSION_PCT, EMA_PAIRS,
                     FIXED_COMMISSION, INITIAL_EQUITY, SLIPPAGE_PCT)
from .indicators import add_ema_features

log = logging.getLogger(__name__)
FOLDS = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80),
         (0.80, 0.90), (0.90, 1.00)]
QUANTILES = (0.60, 0.70, 0.80, 0.90)
MACD_SIGNAL_PERIODS = (5, 9, 15)
MODELS = ("norm_angle", "log_velocity", "vol_scaled",
          "standard", "roc", "macd", "tsmom", "buy_hold")


def detect_regime(df: pd.DataFrame) -> str:
    if len(df) < 50:
        return "UNKNOWN"
    close = df["Close"]
    sma = close.rolling(50).mean()
    trend = "BULL" if close.iloc[-1] > sma.iloc[-1] and close.iloc[-1] > close.iloc[0] else (
        "BEAR" if close.iloc[-1] < sma.iloc[-1] and close.iloc[-1] < close.iloc[0] else "SIDEWAYS")
    atr_pct = ((df["High"] - df["Low"]).rolling(14).mean() / close).mean() * 100.0
    return f"{trend}_{'HIGH_VOL' if atr_pct > 1.0 else 'LOW_VOL'}"


def _thresholds(train: pd.DataFrame, fast: int, slow: int, mode: str,
                features: pd.DataFrame | None = None) -> Iterable[float]:
    if mode in ("standard", "buy_hold"):
        return (0.0,)
    if mode == "macd":
        return MACD_SIGNAL_PERIODS
    if mode == "roc":
        return (0.05, 0.10, 0.20, 0.50)
    if mode == "tsmom":
        return (0.0, 0.001, 0.0025, 0.005)
    feat = features if features is not None else add_ema_features(train, fast, slow).dropna()
    column = {"norm_angle": "angle_fast", "log_velocity": "log_velocity_fast",
              "vol_scaled": "vol_scaled_velocity_fast"}[mode]
    values = feat[column].abs().replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return (0.0,)
    return tuple(float(values.quantile(q)) for q in QUANTILES)


def _daily_mean(result: Dict[str, Any]) -> float:
    ledger = result["bar_returns"]
    if ledger.empty:
        return -np.inf
    daily = ledger["equity"].resample("1D").last().pct_change().dropna()
    return float(daily.mean()) if len(daily) else float(ledger["net_return"].mean())


def select_model(train: pd.DataFrame, mode: str) -> Tuple[int, int, float]:
    """Select one model using training data only and mean after-cost daily return."""
    pairs = (EMA_PAIRS[0],) if mode == "buy_hold" else EMA_PAIRS
    best: Tuple[int, int, float] | None = None
    best_score = -np.inf
    for fast, slow in pairs:
        features = add_ema_features(train, fast, slow).dropna()
        for threshold in _thresholds(train, fast, slow, mode, features):
            candidate_features = (add_ema_features(train, fast, slow, int(threshold)).dropna()
                                  if mode == "macd" else features)
            result = run_backtest(train, fast, slow, threshold, INITIAL_EQUITY,
                                  COMMISSION_PCT, SLIPPAGE_PCT, FIXED_COMMISSION, mode,
                                  precomputed_features=candidate_features)
            score = _daily_mean(result)
            if score > best_score:
                best_score = score
                best = (fast, slow, float(threshold))
    return best if best is not None else (EMA_PAIRS[0][0], EMA_PAIRS[0][1], 0.0)


def _combine(results: List[Dict[str, Any]], fast: int = 0, slow: int = 0,
             threshold: float = 0.0, mode: str = "combined") -> Dict[str, Any]:
    ledgers = [r["bar_returns"] for r in results if not r["bar_returns"].empty]
    if not ledgers:
        return metrics_from_ledger(pd.DataFrame(columns=["position", "gross_return", "turnover",
            "commission", "slippage", "net_return", "equity", "drawdown"]), [],
            INITIAL_EQUITY, fast, slow, threshold, mode)
    ledger = pd.concat(ledgers).sort_index()
    ledger = ledger[~ledger.index.duplicated(keep="first")].copy()
    ledger["equity"] = INITIAL_EQUITY * (1.0 + ledger["net_return"]).cumprod()
    ledger["drawdown"] = ledger["equity"] / ledger["equity"].cummax() - 1.0
    trades = [t for r in results for t in r["trades"]]
    return metrics_from_ledger(ledger, trades, INITIAL_EQUITY,
                               fast, slow, threshold, mode)


def run_walk_forward_analysis(data_store: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
    wfa: Dict[str, Any] = {}
    baselines: Dict[str, Any] = {}
    ablations: Dict[str, Any] = {}
    normalizations: Dict[str, Any] = {}
    regimes: Dict[str, Any] = {}

    for asset, tf_data in data_store.items():
        wfa[asset], baselines[asset], ablations[asset] = {}, {}, {}
        normalizations[asset], regimes[asset] = {}, {}
        for timeframe, df in tf_data.items():
            if df is None or len(df) < 250:
                continue
            df = df.sort_index()
            fold_records: List[Dict[str, Any]] = []
            model_runs: Dict[str, List[Dict[str, Any]]] = {m: [] for m in MODELS}
            candidate_runs: Dict[str, List[Dict[str, Any]]] = {
                f"ema_{f}_{s}_threshold_{threshold:g}": []
                for f, s in EMA_PAIRS for threshold in ANGLE_THRESHOLDS}
            ablation_runs = {m: [] for m in ("standard", "norm_slope_fast", "norm_angle")}

            for fold_id, (train_fraction, test_fraction) in enumerate(FOLDS, 1):
                train_end = int(len(df) * train_fraction)
                test_end = int(len(df) * test_fraction)
                train = df.iloc[:train_end]
                test_start_ts = df.index[train_end]
                test_end_ts = df.index[test_end - 1]
                selected: Dict[str, Tuple[int, int, float]] = {}
                for mode in MODELS:
                    param = select_model(train, mode)
                    selected[mode] = param
                    f, s, threshold = param
                    model_runs[mode].append(run_backtest(
                        df.iloc[:test_end], f, s, threshold, INITIAL_EQUITY,
                        COMMISSION_PCT, SLIPPAGE_PCT, FIXED_COMMISSION, mode,
                        execution_start=test_start_ts, execution_end=test_end_ts))

                # Complete fixed 10 x 12 candidate universe.  Each candidate is
                # evaluated only on the same held-out blocks as the selected model.
                for f, s in EMA_PAIRS:
                    for threshold in ANGLE_THRESHOLDS:
                        key = f"ema_{f}_{s}_threshold_{threshold:g}"
                        candidate_runs[key].append(run_backtest(
                            df.iloc[:test_end], f, s, threshold, INITIAL_EQUITY,
                            COMMISSION_PCT, SLIPPAGE_PCT, FIXED_COMMISSION, "norm_angle",
                            execution_start=test_start_ts, execution_end=test_end_ts))

                pf, ps, pt = selected["norm_angle"]
                for mode in ablation_runs:
                    threshold = pt if mode != "standard" else 0.0
                    ablation_runs[mode].append(run_backtest(
                        df.iloc[:test_end], pf, ps, threshold, INITIAL_EQUITY,
                        COMMISSION_PCT, SLIPPAGE_PCT, FIXED_COMMISSION, mode,
                        execution_start=test_start_ts, execution_end=test_end_ts))
                fold_records.append({"fold": fold_id, "train_end": train.index[-1],
                                     "test_start": test_start_ts, "test_end": test_end_ts,
                                     "selected": selected})

            combined = {mode: _combine(runs, mode=mode) for mode, runs in model_runs.items()}
            combined_ablation = {mode: _combine(runs, mode=f"ablation_{mode}")
                                 for mode, runs in ablation_runs.items()}
            combined_candidates = {key: _combine(runs, mode=key)
                                   for key, runs in candidate_runs.items()}
            wfa[asset][timeframe] = {"folds": fold_records,
                "out_of_sample_metrics": combined["norm_angle"],
                "candidate_universe": combined_candidates}
            baselines[asset][timeframe] = {m: combined[m] for m in
                ("standard", "roc", "macd", "tsmom", "buy_hold")}
            normalizations[asset][timeframe] = {m: combined[m] for m in
                ("norm_angle", "log_velocity", "vol_scaled")}
            ablations[asset][timeframe] = combined_ablation
            test_slice = df.iloc[int(len(df) * 0.5):]
            regimes[asset][timeframe] = {"regime": detect_regime(test_slice),
                "metrics": combined["norm_angle"]}
            log.info("Completed anchored WFA: %s %s", asset, timeframe)
    return {"wfa": wfa, "baselines": baselines, "ablations": ablations,
            "normalizations": normalizations, "regimes": regimes}


def build_wfa_ranking_table(wfa_results: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for asset, timeframes in wfa_results.items():
        for tf, data in timeframes.items():
            m = data["out_of_sample_metrics"]
            rows.append({"Market": asset, "Timeframe": tf, "Folds": len(data["folds"]),
                "OOS Trades": m["total_trades"], "OOS Win Rate (%)": m["win_rate"],
                "OOS Net Profit ($)": m["total_net_profit"], "OOS Sharpe": m["sharpe_ratio"],
                "OOS Max Drawdown (%)": m["max_drawdown"], "OOS Turnover": m["total_turnover"],
                "OOS Exposure (%)": m["exposure_pct"]})
    return pd.DataFrame(rows)


def build_baseline_comparison_table(baseline_results: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for asset, timeframes in baseline_results.items():
        for tf, modes in timeframes.items():
            for mode, m in modes.items():
                rows.append({"Market": asset, "Timeframe": tf, "Model": mode,
                    "Trades": m["total_trades"], "Win Rate (%)": m["win_rate"],
                    "Net Profit ($)": m["total_net_profit"], "Daily Sharpe": m["sharpe_ratio"],
                    "Max Drawdown (%)": m["max_drawdown"], "Turnover": m["total_turnover"],
                    "Exposure (%)": m["exposure_pct"]})
    return pd.DataFrame(rows)


def build_model_table(results: Dict[str, Any]) -> pd.DataFrame:
    return build_baseline_comparison_table(results)
