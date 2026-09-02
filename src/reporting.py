"""
reporting.py — Academic Report and Metadata Generation.

Generates the `Summary_of_Findings.md` for the `paper/` directory and
the `experiment_metadata.json` for the `results/logs/` directory to ensure
strict reproducibility standards.
"""

import json
import hashlib
import logging
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from .config import (
    ASSETS, TIMEFRAMES, EMA_PAIRS, ANGLE_THRESHOLDS,
    INITIAL_EQUITY, RESULT_LOGS_DIR, RESULT_ROOT, CONFIG_FILE, DATA_RAW_DIR
)

log = logging.getLogger(__name__)


def generate_experiment_metadata() -> None:
    """Captures the system state, dependency versions, and parameter sets."""
    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    data_hashes = {p.name: digest(p) for p in sorted(DATA_RAW_DIR.glob("*.csv"))}
    result_hashes = {str(p.relative_to(RESULT_ROOT)): digest(p)
                     for p in sorted(RESULT_ROOT.rglob("*"))
                     if p.is_file() and p.name != "experiment_metadata.json"}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=CONFIG_FILE.parent.parent,
            text=True).strip()
    except Exception:
        commit = "Unavailable"
    meta: Dict[str, Any] = {
        "execution_timestamp": datetime.now().isoformat(),
        "environment": {
            "os": platform.system(),
            "os_version": platform.version(),
            "python_version": platform.python_version(),
        },
        "parameters": {
            "initial_equity": INITIAL_EQUITY,
            "assets": list(ASSETS.keys()),
            "timeframes": list(TIMEFRAMES.keys()),
            "ema_pairs": [f"({f},{s})" for f, s in EMA_PAIRS],
            "angle_thresholds": ANGLE_THRESHOLDS,
            "total_combinations": len(ASSETS) * len(TIMEFRAMES) * len(EMA_PAIRS) * len(ANGLE_THRESHOLDS)
        },
        "provenance": {"source_commit": commit, "config_sha256": digest(CONFIG_FILE),
                       "data_sha256": data_hashes, "result_sha256": result_hashes,
                       "random_seeds": {"stationary_bootstrap": 20260831,
                                        "max_t_model_comparison": 20260831}}
    }

    # Attempt to capture pip package versions
    try:
        pip_freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True)
        meta["environment"]["dependencies"] = pip_freeze.strip().split("\n")
    except Exception:
        meta["environment"]["dependencies"] = "Unavailable"

    meta_file = RESULT_LOGS_DIR / "experiment_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)

    log.info(f"Experiment metadata generated: {meta_file.name}")


def generate_summary_report(
    results: Dict[str, Dict[str, Dict[Any, Dict[str, Any]]]],
    output_file: Path
) -> None:
    """Generates the academic Summary of Findings markdown report."""

    total_valid = 0
    total_fifty_plus = 0
    best_overall = None
    best_profit = -float('inf')

    for asset, tf_dict in results.items():
        for tf, pair_dict in tf_dict.items():
            for key, m in pair_dict.items():
                if m.get("total_trades", 0) > 0:
                    total_valid += 1
                    if m.get("win_rate", 0) >= 50:
                        total_fifty_plus += 1

                    if m.get("total_net_profit", 0) > best_profit:
                        best_profit = m["total_net_profit"]
                        best_overall = m

    report = [
        "# Summary of Findings: Normalized EMA Momentum Analysis",
        "",
        "## Executive Summary",
        "This exploratory study evaluates whether normalized EMA-momentum filters reduce low-momentum crossover participation under delayed execution and transaction costs.",
        "",
        "## Methodology",
        f"- **Initial Equity:** ${INITIAL_EQUITY:,.2f} (Zero leverage, 100% allocation)",
        f"- **Asset Universe:** {', '.join(ASSETS.keys())}",
        f"- **Timeframes Evaluated:** {', '.join(TIMEFRAMES.keys())}",
        f"- **Parameter Grid:** {len(EMA_PAIRS)} EMA Pairs × {len(ANGLE_THRESHOLDS)} Angle Thresholds",
        f"- **Total Combinations Tested:** {len(ASSETS) * len(TIMEFRAMES) * len(EMA_PAIRS) * len(ANGLE_THRESHOLDS)}",
        "",
        "## Statistical Findings",
        f"Out of the total combinations tested, **{total_valid}** generated active trading signals.",
        f"Among active configurations, **{total_fifty_plus}** achieved a win rate of $\\geq$ 50%. This descriptive count is not evidence of statistical superiority.",
        "",
        "### Top Performing Configuration",
    ]

    if best_overall:
        report.extend([
            f"- **Market:** {best_overall.get('ema_pair')} on {best_overall.get('angle_threshold')}° threshold",
            f"- **Win Rate:** {best_overall.get('win_rate', 0)}%",
            f"- **Total Profit:** ${best_overall.get('total_net_profit', 0):,.2f}",
            f"- **Sharpe Ratio:** {best_overall.get('sharpe_ratio', 0)}",
            f"- **Maximum Drawdown:** {best_overall.get('max_drawdown', 0)}%"
        ])
    else:
        report.append(
            "- *No profitable configurations identified in this dataset.*")

    report.extend([
        "",
        "## Conclusion",
        "Results must be interpreted from synchronized out-of-sample returns and dependence-aware inference; no universal profitability claim is made.",
        "",
        "---\n*Generated by the EMA Angle Backtesting Engine.*"
    ])

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    log.info(f"Summary Report generated: {output_file.name}")
