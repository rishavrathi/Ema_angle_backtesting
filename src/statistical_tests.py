"""Dependence-aware inference on synchronized out-of-sample return series."""
from __future__ import annotations

from typing import Any, Dict
import numpy as np
import pandas as pd


def stationary_bootstrap_indices(n: int, expected_block: float,
                                 rng: np.random.Generator) -> np.ndarray:
    """Politis--Romano stationary-bootstrap indices."""
    if n <= 0:
        return np.array([], dtype=int)
    restart_probability = 1.0 / expected_block
    out = np.empty(n, dtype=int)
    out[0] = rng.integers(n)
    for i in range(1, n):
        out[i] = rng.integers(n) if rng.random() < restart_probability else (out[i - 1] + 1) % n
    return out


def _daily_returns(metrics: Dict[str, Any]) -> pd.Series:
    ledger = metrics["bar_returns"]
    if ledger.empty:
        return pd.Series(dtype=float)
    return ledger["equity"].resample("1D").last().pct_change().dropna()


def stationary_bootstrap_test(difference: pd.Series, replications: int = 1999,
                              expected_block: float = 5.0,
                              seed: int = 20260831) -> Dict[str, float]:
    values = difference.dropna().to_numpy(dtype=float)
    n = len(values)
    if n < 5:
        return {"mean": np.nan, "median": np.nan, "ci_low": np.nan,
                "ci_high": np.nan, "p_value": np.nan,
                "probability_positive": np.nan, "n": n}
    rng = np.random.default_rng(seed)
    boot_means = np.empty(replications)
    centered_means = np.empty(replications)
    centered = values - values.mean()
    for b in range(replications):
        idx = stationary_bootstrap_indices(n, expected_block, rng)
        boot_means[b] = values[idx].mean()
        centered_means[b] = centered[idx].mean()
    observed = float(values.mean())
    return {"mean": observed, "median": float(np.median(values)),
            "ci_low": float(np.quantile(boot_means, 0.025)),
            "ci_high": float(np.quantile(boot_means, 0.975)),
            "p_value": float((1 + np.sum(centered_means >= observed)) / (replications + 1)),
            "probability_positive": float(np.mean(boot_means > 0)), "n": n}


def holm_adjust(p_values: pd.Series) -> pd.Series:
    valid = p_values.dropna().sort_values()
    adjusted = pd.Series(np.nan, index=p_values.index, dtype=float)
    running = 0.0
    m = len(valid)
    for rank, (idx, p_value) in enumerate(valid.items()):
        running = max(running, (m - rank) * float(p_value))
        adjusted.loc[idx] = min(running, 1.0)
    return adjusted


def max_t_model_comparison_p_value(models: Dict[str, Dict[str, Any]], benchmark: Dict[str, Any],
                                   replications: int = 999, expected_block: float = 5.0,
                                   seed: int = 20260831) -> float:
    """Stationary-bootstrap max-t diagnostic; this is not Hansen's SPA test."""
    benchmark_daily = _daily_returns(benchmark).rename("benchmark")
    columns = []
    for name, metrics in models.items():
        if name == "standard":
            continue
        aligned = pd.concat([_daily_returns(metrics).rename(name), benchmark_daily], axis=1).dropna()
        if len(aligned) >= 5:
            columns.append((name, aligned[name] - aligned["benchmark"]))
    if not columns:
        return np.nan
    matrix = pd.concat(dict(columns), axis=1).dropna().to_numpy(dtype=float)
    if len(matrix) < 5:
        return np.nan
    scale = matrix.std(axis=0, ddof=1)
    scale[scale == 0] = np.inf
    observed = float(np.max(np.sqrt(len(matrix)) * matrix.mean(axis=0) / scale))
    centered = matrix - matrix.mean(axis=0)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(replications):
        idx = stationary_bootstrap_indices(len(matrix), expected_block, rng)
        statistic = float(np.max(np.sqrt(len(matrix)) * centered[idx].mean(axis=0) / scale))
        exceed += statistic >= observed
    return float((exceed + 1) / (replications + 1))


def run_statistical_suite(baseline_results: Dict[str, Any], wfa_results: Dict[str, Any],
                          normalization_results: Dict[str, Any] | None = None) -> pd.DataFrame:
    rows = []
    for asset, timeframes in wfa_results.items():
        for tf, proposed_data in timeframes.items():
            # Intraday one-minute histories span too few distinct days for
            # dependence-aware daily inference. They remain descriptive outputs.
            if tf == "1m":
                continue
            proposed = proposed_data["out_of_sample_metrics"]
            standard = baseline_results[asset][tf]["standard"]
            aligned = pd.concat([_daily_returns(proposed).rename("filtered"),
                                 _daily_returns(standard).rename("standard")], axis=1).dropna()
            result = stationary_bootstrap_test(aligned["filtered"] - aligned["standard"])
            universe = proposed_data["candidate_universe"]
            rows.append({"Asset": asset, "Timeframe": tf,
                "Mean Daily Return Difference": result["mean"],
                "Median Difference": result["median"], "CI 2.5%": result["ci_low"],
                "CI 97.5%": result["ci_high"], "Bootstrap P-Value": result["p_value"],
                "Probability Difference Positive": result["probability_positive"],
                "OOS Days": result["n"],
                "Max-t Model-Comparison P-Value": max_t_model_comparison_p_value(
                    universe, standard)})
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["Holm-Adjusted P-Value"] = holm_adjust(frame["Bootstrap P-Value"])
    return frame


def leave_silver_1h_out_aggregate(baseline_results: Dict[str, Any],
                                  wfa_results: Dict[str, Any]) -> pd.DataFrame:
    """Equal-weight daily aggregate excluding descriptive 1m and Silver 1h."""
    differences = []
    for asset, timeframes in wfa_results.items():
        for tf, proposed_data in timeframes.items():
            if tf == "1m" or (asset == "Silver" and tf == "1h"):
                continue
            proposed = _daily_returns(proposed_data["out_of_sample_metrics"])
            standard = _daily_returns(baseline_results[asset][tf]["standard"])
            differences.append((proposed - standard).rename(f"{asset}_{tf}"))
    panel = pd.concat(differences, axis=1)
    aggregate = panel.mean(axis=1, skipna=True).dropna()
    result = stationary_bootstrap_test(aggregate)
    return pd.DataFrame([{"Excluded": "Silver_1h and all descriptive 1m series",
        "Series Included": len(differences), "OOS Days": result["n"],
        "Mean Daily Return Difference": result["mean"],
        "Median Difference": result["median"], "CI 2.5%": result["ci_low"],
        "CI 97.5%": result["ci_high"], "Bootstrap P-Value": result["p_value"],
        "Probability Difference Positive": result["probability_positive"]}])
