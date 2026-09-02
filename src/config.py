"""
config.py — Central configuration parser for the EMA Angle Strategy backtest.

This module loads all configurations from the `config/config.yaml` file,
ensuring no hardcoded values exist in the analytical codebase.
"""
import yaml
from pathlib import Path
from itertools import combinations
from typing import Any, Dict, List, Tuple

# ── Paths ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE = BASE_DIR / "config" / "config.yaml"

if not CONFIG_FILE.exists():
    raise FileNotFoundError(f"Configuration file not found at {CONFIG_FILE}")

with open(CONFIG_FILE, "r", encoding="utf-8") as file:
    _config: Dict[str, Any] = yaml.safe_load(file)

# ── Dynamic Configuration Values ──────────────────────────────────────
ASSETS: Dict[str, str] = _config.get("assets", {})
TIMEFRAMES: Dict[str, Dict[str, Any]] = _config.get("timeframes", {})

# Strategy Parameters
_base_emas: List[int] = _config.get("strategy", {}).get("base_ema_values", [])
EMA_PAIRS: List[Tuple[int, int]] = [(f, s)
                                    for f, s in combinations(_base_emas, 2)]
ANGLE_THRESHOLDS: List[float] = _config.get(
    "strategy", {}).get("angle_thresholds", [])

# Backtest Settings
INITIAL_EQUITY: float = float(_config.get(
    "backtest", {}).get("initial_equity", 10000.0))
COMMISSION_PCT: float = float(_config.get("backtest", {}).get("commission_pct", 0.0))
SLIPPAGE_PCT: float = float(_config.get("backtest", {}).get("slippage_pct", 0.0))
FIXED_COMMISSION: float = float(_config.get("backtest", {}).get("fixed_commission", 0.0))
BASELINES: List[str] = _config.get("backtest", {}).get("baselines", ["standard"])
RANK_METRIC: str = _config.get("backtest", {}).get(
    "rank_metric", "mean_daily_return")

# Data & Output Directories
RESULT_ROOT: Path = BASE_DIR / _config.get("paths", {}).get(
    "results_root", "results/final_walk_forward")
DATA_RAW_DIR: Path = BASE_DIR / \
    _config.get("paths", {}).get("data_raw", "data")
DATA_PROCESSED_DIR: Path = BASE_DIR / \
    _config.get("paths", {}).get("data_processed", "data/processed")
RESULT_EXCEL_DIR: Path = BASE_DIR / \
    _config.get("paths", {}).get("results_excel", "results/excel")
RESULT_FIGS_DIR: Path = BASE_DIR / \
    _config.get("paths", {}).get("results_figures", "results/figures")
RESULT_REPORTS_DIR: Path = BASE_DIR / \
    _config.get("paths", {}).get("results_reports", "results/reports")
RESULT_LOGS_DIR: Path = BASE_DIR / \
    _config.get("paths", {}).get("results_logs", "results/logs")
RESULT_TRADE_LOGS_DIR: Path = BASE_DIR / \
    _config.get("paths", {}).get("results_trade_logs", "results/trade_logs")

# Calculate Total Combinations statically for logging purposes
TOTAL_COMBOS: int = len(ASSETS) * len(TIMEFRAMES) * \
    len(EMA_PAIRS) * len(ANGLE_THRESHOLDS)


def ensure_directories() -> None:
    """Ensures all necessary output directories exist before execution."""
    for directory in [
        DATA_RAW_DIR, RESULT_ROOT, DATA_PROCESSED_DIR, RESULT_EXCEL_DIR,
        RESULT_FIGS_DIR, RESULT_REPORTS_DIR, RESULT_LOGS_DIR,
        RESULT_TRADE_LOGS_DIR
    ]:
        directory.mkdir(parents=True, exist_ok=True)
