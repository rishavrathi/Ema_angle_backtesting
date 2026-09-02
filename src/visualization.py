"""
visualization.py — Publication-Quality Figure Generation.

Generates high-resolution (300 DPI) figures for the academic report,
including equity curves, drawdown curves, and parameter heatmaps.
"""

import logging
import warnings
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mtick  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, Any, Optional  # noqa: E402

from .indicators import add_ema_features
from .config import ASSETS, TIMEFRAMES, RESULT_FIGS_DIR


warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

# Academic Grayscale / Professional Color Palette
PALETTE: Dict[str, str] = {
    "bg":       "#ffffff",
    "panel":    "#ffffff",
    "grid":     "#e0e0e0",
    "text":     "#000000",
    "accent":   "#004488",
    "green":    "#228833",
    "red":      "#cc3311",
    "orange":   "#ee6677",
    "purple":   "#aa3377",
    "fast_ema": "#ee6677",
    "slow_ema": "#004488",
}


def _apply_academic_style(fig: plt.Figure, axes: Any) -> None:
    """Applies a clean, publication-ready style to matplotlib figures."""
    fig.patch.set_facecolor(PALETTE["bg"])
    ax_list = axes if hasattr(axes, "__iter__") and not isinstance(
        axes, plt.Axes) else [axes]

    for ax in ax_list:
        if not isinstance(ax, plt.Axes):
            continue
        ax.set_facecolor(PALETTE["panel"])
        ax.tick_params(colors=PALETTE["text"], labelsize=10)
        ax.xaxis.label.set_color(PALETTE["text"])
        ax.yaxis.label.set_color(PALETTE["text"])
        ax.title.set_color(PALETTE["text"])
        ax.title.set_fontsize(12)
        ax.title.set_fontweight('bold')
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["text"])
            spine.set_linewidth(0.8)
        ax.grid(True, color=PALETTE["grid"], linewidth=0.5, alpha=0.7)


def _save(fig: plt.Figure, path: Path) -> None:
    """Saves the figure in 300 DPI PNG format suitable for academic journals."""
    fig.savefig(path, dpi=300, bbox_inches="tight",
                facecolor=PALETTE["bg"], edgecolor="none")
    plt.close(fig)
    log.info(f"Saved Publication Figure: {path.name}")


def _get_best_key(results_cell: Dict[Any, Dict[str, Any]]) -> Optional[tuple]:
    """Identifies the parameter set yielding the highest total profit."""
    if not results_cell:
        return None
    return max(results_cell.keys(), key=lambda k: results_cell[k].get("total_net_profit", -float('inf')))


def plot_price_ema_entries(
    df: pd.DataFrame,
    asset: str,
    tf: str,
    results_cell: Dict[Any, Dict[str, Any]],
    out_dir: Path = RESULT_FIGS_DIR,
) -> None:
    """Plots the raw price action alongside the Exponential Moving Averages and trade execution markers."""
    key = _get_best_key(results_cell)
    if key is None or df is None:
        return

    fast, slow, thr = key
    m: Dict[str, Any] = results_cell[key]
    trades: list = m.get("trades", [])

    feat: pd.DataFrame = add_ema_features(
        df, fast, slow).dropna(subset=["ema_fast", "ema_slow"])
    if feat.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    _apply_academic_style(fig, ax)

    ax.plot(feat.index, feat["Close"], color="#888888",
            linewidth=1.0, alpha=0.6, label="Close Price")
    ax.plot(feat.index, feat["ema_fast"], color=PALETTE["fast_ema"],
            linewidth=1.2, label=f"EMA({fast})")
    ax.plot(feat.index, feat["ema_slow"], color=PALETTE["slow_ema"],
            linewidth=1.2, label=f"EMA({slow})")

    long_x, long_y = [], []
    short_x, short_y = [], []
    for t in trades:
        if t["direction"] == "LONG":
            long_x.append(t["entry_date"])
            long_y.append(t["entry_price"])
        else:
            short_x.append(t["entry_date"])
            short_y.append(t["entry_price"])

    if long_x:
        ax.scatter(long_x, long_y, marker="^",
                   color=PALETTE["green"], s=80, edgecolors="white", label="Long Entry", zorder=3)
    if short_x:
        ax.scatter(short_x, short_y, marker="v",
                   color=PALETTE["red"], s=80, edgecolors="white", label="Short Entry", zorder=3)

    ax.set_title(
        f"Trade Executions: {asset} ({tf}) | EMA({fast},{slow}) | Angle $\\geq$ {thr}°")
    ax.set_ylabel("Price (USD)")
    ax.legend(loc="upper left", frameon=True,
              facecolor=PALETTE["panel"], edgecolor=PALETTE["grid"])

    fname = f"trades_{asset}_{tf}.png".replace("=", "").replace("-", "")
    _save(fig, out_dir / fname)


def plot_equity_curve(
    asset: str,
    tf: str,
    results_cell: Dict[Any, Dict[str, Any]],
    out_dir: Path = RESULT_FIGS_DIR,
) -> None:
    """Plots the cumulative equity curve for the optimal parameter set."""
    key = _get_best_key(results_cell)
    if not key:
        return

    m: Dict[str, Any] = results_cell[key]
    trades: list = m.get("trades", [])
    if not trades:
        return

    eq = [10000.0]
    dates = [trades[0]["entry_date"]]

    for t in trades:
        eq.append(eq[-1] + t["net_pnl"])
        dates.append(t["exit_date"])

    fig, ax = plt.subplots(figsize=(10, 4))
    _apply_academic_style(fig, ax)

    ax.plot(dates, eq, color=PALETTE["accent"], linewidth=2.0)
    ax.fill_between(dates, 10000.0, eq, where=(np.array(eq) >= 10000.0), color=PALETTE["green"], alpha=0.1)  # type: ignore
    ax.fill_between(dates, 10000.0, eq, where=(np.array(eq) < 10000.0), color=PALETTE["red"], alpha=0.1)  # type: ignore
    ax.axhline(10000.0, color="#888888", linestyle="--", linewidth=1.0)

    fast, slow, thr = key
    ax.set_title(
        f"Cumulative Equity: {asset} ({tf}) | EMA({fast},{slow}) | $\\theta \\geq$ {thr}°")
    ax.set_ylabel("Equity (USD)")
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))

    fname = f"equity_{asset}_{tf}.png".replace("=", "").replace("-", "")
    _save(fig, out_dir / fname)


def generate_all_charts(
    data_store: Dict[str, Dict[str, pd.DataFrame]],
    results: Dict[str, Dict[str, Dict[Any, Dict[str, Any]]]]
) -> None:
    """Orchestrates the generation of all publication-quality figures."""
    for asset in ASSETS:
        for tf in TIMEFRAMES:
            df = data_store.get(asset, {}).get(tf)
            res_cell = results.get(asset, {}).get(tf, {})

            plot_price_ema_entries(df, asset, tf, res_cell)
            plot_equity_curve(asset, tf, res_cell)

    log.info("Successfully generated all publication figures.")
