"""
main.py — Academic Execution Pipeline for EMA Angle Optimization (Phase 11).

This is the singular entry point for the backtesting framework. 
It orchestrates data acquisition, walk-forward parameter optimization, empirical analysis,
and publication-quality reporting.
"""

from src.reporting import generate_experiment_metadata
# We will skip old generate_all_charts for now, we will add new matplotlib generators later if needed
from src.optimizer import (run_walk_forward_analysis, build_wfa_ranking_table,
                           build_baseline_comparison_table, build_model_table)
from src.statistical_tests import run_statistical_suite, leave_silver_1h_out_aggregate
from src.data_loader import download_data
from src.config import ensure_directories, RESULT_ROOT
import sys
import logging
import pandas as pd
from pathlib import Path

# Ensure stdout handles UTF-8 for console output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Configure logging before importing local modules to capture all logs
log_dir = RESULT_ROOT / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
csv_dir = RESULT_ROOT / "csvs"
csv_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "run.log", encoding="utf-8", mode="w"),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)


def export_csvs(wfa_results, baseline_results, ablation_results,
                normalization_results, regime_results, stat_df):
    log.info("  -> Exporting Walk-Forward Results to CSV...")
    wfa_df = build_wfa_ranking_table(wfa_results)
    wfa_df.to_csv(csv_dir / "walk_forward_results.csv", index=False)
    
    log.info("  -> Exporting Baseline Comparisons to CSV...")
    base_df = build_baseline_comparison_table(baseline_results)
    base_df.to_csv(csv_dir / "baseline_results.csv", index=False)

    build_model_table(ablation_results).to_csv(csv_dir / "ablation_results.csv", index=False)
    build_model_table(normalization_results).to_csv(csv_dir / "normalization_results.csv", index=False)
    
    log.info("  -> Exporting Statistical Tests to CSV...")
    if not stat_df.empty:
        stat_df.to_csv(csv_dir / "statistical_tests.csv", index=False)
        
    log.info("  -> Exporting Regime Robustness to CSV...")
    # Quick regime DF
    rows = []
    for asset, tf_dict in regime_results.items():
        for tf, data in tf_dict.items():
            m = data["metrics"]
            rows.append({
                "Asset": asset, "Timeframe": tf, "Regime": data["regime"],
                "Net Profit": m.get("total_net_profit", 0.0),
                "Sharpe": m.get("sharpe_ratio", 0.0)
            })
    reg_df = pd.DataFrame(rows)
    if not reg_df.empty:
        reg_df.to_csv(csv_dir / "regime_results.csv", index=False)

    bar_dir = RESULT_ROOT / "bar_returns"
    bar_dir.mkdir(parents=True, exist_ok=True)
    for asset, timeframes in wfa_results.items():
        for tf, data in timeframes.items():
            data["out_of_sample_metrics"]["bar_returns"].to_csv(
                bar_dir / f"{asset}_{tf}_norm_angle.csv")
            for mode, metrics in baseline_results[asset][tf].items():
                metrics["bar_returns"].to_csv(bar_dir / f"{asset}_{tf}_{mode}.csv")
            for mode, metrics in normalization_results[asset][tf].items():
                metrics["bar_returns"].to_csv(bar_dir / f"{asset}_{tf}_{mode}.csv")

    risk_rows = []
    sensitivity_rows = []
    cost_levels = (0, 5, 10, 20, 30, 50)
    for asset, timeframes in wfa_results.items():
        for tf, data in timeframes.items():
            metrics = data["out_of_sample_metrics"]
            risk_rows.append({"Asset": asset, "Timeframe": tf,
                "VaR 95%": metrics["var_95"],
                "Expected Shortfall 95%": metrics["expected_shortfall_95"],
                "Break-Even Cost (bps)": metrics["break_even_cost_bps"]})
            ledger = metrics["bar_returns"]
            for bps in cost_levels:
                factors = (1 + ledger["gross_return"]) * (1 - bps / 10000) ** ledger["turnover"]
                sensitivity_rows.append({"Asset": asset, "Timeframe": tf,
                    "One-Way Cost (bps)": bps,
                    "Final Equity": 10000.0 * float(factors.prod()),
                    "Net Return": float(factors.prod() - 1)})
    pd.DataFrame(risk_rows).to_csv(csv_dir / "tail_risk_and_break_even.csv", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(csv_dir / "cost_sensitivity.csv", index=False)


def main() -> None:
    """Executes the complete end-to-end academic backtesting pipeline."""

    log.info("=" * 70)
    log.info("  EMA Angle Strategy — Empirical Hardening (Phase 11)")
    log.info("=" * 70)

    # Pre-execution environment setup
    ensure_directories()
    # 1. Data Acquisition
    log.info("\n[Phase 1] Acquiring Historical Market Data...")
    data_store = download_data()

    # 2. Walk-Forward Analysis (IS/OOS)
    log.info("\n[Phase 2] Executing Walk-Forward Optimization Engine...")
    results = run_walk_forward_analysis(data_store)
    
    wfa_results = results["wfa"]
    baseline_results = results["baselines"]
    regime_results = results["regimes"]
    ablation_results = results["ablations"]
    normalization_results = results["normalizations"]

    # 3. Statistical Suite
    log.info("\n[Phase 3] Running Non-Parametric Statistical Suite...")
    stat_df = run_statistical_suite(baseline_results, wfa_results, normalization_results)
    leave_silver_1h_out_aggregate(baseline_results, wfa_results).to_csv(
        csv_dir / "leave_silver_1h_out.csv", index=False)

    # 4. CSV Reporting
    log.info("\n[Phase 4] Generating CSV Artifacts...")
    export_csvs(wfa_results, baseline_results, ablation_results,
                normalization_results, regime_results, stat_df)
    generate_experiment_metadata()

    log.info("\n" + "=" * 70)
    log.info("  EMPIRICAL EXECUTION COMPLETE")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
