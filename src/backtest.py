"""Cost-aware, mark-to-market backtesting for fully collateralized 1x exposure.

Gold and silver are treated as divisible synthetic price exposures. This module
does not model CME multipliers, integer lots, margin, or contract rolls.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

from .indicators import add_ema_features
from .strategy import generate_signals


LEDGER_COLUMNS = ["timestamp", "position", "gross_return", "turnover",
                  "commission", "slippage", "net_return", "equity", "drawdown"]


def _empty_ledger() -> pd.DataFrame:
    return pd.DataFrame(columns=LEDGER_COLUMNS).set_index("timestamp")


def _risk_metrics(ledger: pd.DataFrame, initial_equity: float) -> Dict[str, float]:
    if ledger.empty:
        return dict(sharpe_ratio=0.0, sortino_ratio=0.0,
                    annualized_return=0.0, max_drawdown=0.0,
                    var_95=np.nan, expected_shortfall_95=np.nan)
    daily_equity = ledger["equity"].resample("1D").last().dropna()
    daily = daily_equity.pct_change().dropna()
    sharpe = 0.0
    sortino = 0.0
    if len(daily) > 1 and daily.std(ddof=1) > 0:
        sharpe = float(np.sqrt(252.0) * daily.mean() / daily.std(ddof=1))
    downside = daily[daily < 0]
    if len(downside) > 1 and downside.std(ddof=1) > 0:
        sortino = float(np.sqrt(252.0) * daily.mean() / downside.std(ddof=1))
    days = max((ledger.index[-1] - ledger.index[0]).total_seconds() / 86400.0, 1.0)
    cagr = float((ledger["equity"].iloc[-1] / initial_equity) ** (365.25 / days) - 1.0)
    q05 = float(daily.quantile(0.05)) if len(daily) else np.nan
    expected_shortfall = float(-daily[daily <= q05].mean()) if len(daily) else np.nan
    return dict(sharpe_ratio=sharpe, sortino_ratio=sortino,
                annualized_return=100.0 * cagr,
                max_drawdown=float(-100.0 * ledger["drawdown"].min()),
                var_95=float(-q05), expected_shortfall_95=expected_shortfall)


def _break_even_cost_bps(ledger: pd.DataFrame) -> float:
    """One-way proportional cost per unit turnover that reduces wealth to 1."""
    if ledger.empty or float((1 + ledger["gross_return"]).prod()) <= 1.0:
        return 0.0
    gross = ledger["gross_return"].to_numpy(float)
    turnover = ledger["turnover"].to_numpy(float)
    def wealth(cost: float) -> float:
        return float(np.prod((1.0 + gross) * np.power(1.0 - cost, turnover)))
    lo, hi = 0.0, 0.999999
    for _ in range(80):
        mid = (lo + hi) / 2
        if wealth(mid) > 1.0:
            lo = mid
        else:
            hi = mid
    return 10000.0 * (lo + hi) / 2


def metrics_from_ledger(ledger: pd.DataFrame, trades: list[Dict[str, Any]],
                        initial_equity: float, fast: int, slow: int,
                        threshold: float, mode: str) -> Dict[str, Any]:
    final_equity = initial_equity if ledger.empty else float(ledger["equity"].iloc[-1])
    if not ledger.empty:
        reconciled = initial_equity * float((1.0 + ledger["net_return"]).prod())
        if not np.isclose(final_equity, reconciled, rtol=1e-10, atol=1e-8):
            raise AssertionError("final-equity reconciliation failed")
    pnl = np.asarray([t["net_pnl"] for t in trades], dtype=float)
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    pf = float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() < 0 else 0.0
    risk = _risk_metrics(ledger, initial_equity)
    return {
        "bar_returns": ledger, "trades": trades, "total_trades": len(trades),
        "long_trades": sum(t["direction"] == "LONG" for t in trades),
        "short_trades": sum(t["direction"] == "SHORT" for t in trades),
        "winning_trades": int((pnl > 0).sum()), "losing_trades": int((pnl <= 0).sum()),
        "win_rate": float((pnl > 0).mean() * 100.0) if len(pnl) else 0.0,
        "total_net_profit": float(final_equity - initial_equity),
        "avg_trade_net": float(pnl.mean()) if len(pnl) else 0.0,
        "expectancy": float(pnl.mean()) if len(pnl) else 0.0,
        "final_equity": final_equity, "profit_factor": pf,
        **risk,
        "exposure_pct": float(100.0 * ledger["position"].abs().mean()) if not ledger.empty else 0.0,
        "total_turnover": float(ledger["turnover"].sum()) if not ledger.empty else 0.0,
        "position_changes": int((ledger["turnover"] > 0).sum()) if not ledger.empty else 0,
        "total_commission_paid": float(ledger["commission"].sum()) if not ledger.empty else 0.0,
        "total_slippage_paid": float(ledger["slippage"].sum()) if not ledger.empty else 0.0,
        "break_even_cost_bps": _break_even_cost_bps(ledger),
        "fast_ema": fast, "slow_ema": slow, "ema_pair": f"EMA({fast},{slow})",
        "angle_threshold": threshold, "baseline_mode": mode,
        "accounting_model": "fully_collateralized_synthetic_1x",
    }


def run_backtest(df: pd.DataFrame, fast: int, slow: int, threshold: float,
                 initial_equity: float, commission_pct: float = 0.0,
                 slippage_pct: float = 0.0, fixed_commission: float = 0.0,
                 baseline_mode: str = "standard",
                 execution_start: Optional[pd.Timestamp] = None,
                 execution_end: Optional[pd.Timestamp] = None,
                 precomputed_features: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Return timestamped net bar returns with prior history retained for warm-up."""
    if fixed_commission:
        raise ValueError("fixed dollar commission is undefined for synthetic exposure")
    if df is None or len(df) < max(fast, slow) + 2:
        return metrics_from_ledger(_empty_ledger(), [], initial_equity,
                                   fast, slow, threshold, baseline_mode)
    feat = (precomputed_features.copy() if precomputed_features is not None
            else add_ema_features(df.sort_index(), fast, slow,
                                  int(threshold) if baseline_mode == "macd" else 9).dropna().copy())
    if len(feat) < 2:
        return metrics_from_ledger(_empty_ledger(), [], initial_equity,
                                   fast, slow, threshold, baseline_mode)
    signals = generate_signals(feat, threshold, baseline_mode)
    start = feat.index[1] if execution_start is None else pd.Timestamp(execution_start)
    end = feat.index[-1] if execution_end is None else pd.Timestamp(execution_end)

    equity = float(initial_equity)
    peak = equity
    position = 0
    active_trade: Optional[Dict[str, Any]] = None
    trades: list[Dict[str, Any]] = []
    rows: list[Dict[str, Any]] = []

    index = feat.index
    signal_values = signals.to_numpy(dtype=np.int8)
    fast_values = feat["ema_fast"].to_numpy(dtype=float)
    slow_values = feat["ema_slow"].to_numpy(dtype=float)
    open_values = feat["Open"].to_numpy(dtype=float)
    close_values = feat["Close"].to_numpy(dtype=float)
    eligible = np.flatnonzero((index >= start) & (index <= end))
    for ordinal, i in enumerate(eligible):
        if i == 0:
            continue
        ts = index[i]
        previous_signal = int(signal_values[i - 1])
        previous_fast = fast_values[i - 1]
        previous_slow = slow_values[i - 1]
        target = position
        if baseline_mode == "buy_hold":
            target = 1
        elif position == 1 and previous_fast < previous_slow:
            target = previous_signal if previous_signal == -1 else 0
        elif position == -1 and previous_fast > previous_slow:
            target = previous_signal if previous_signal == 1 else 0
        elif target == 0:
            if previous_signal in (-1, 1):
                target = previous_signal

        equity_before = equity
        is_terminal = ordinal == len(eligible) - 1
        commission = 0.0
        slippage = 0.0

        # Events are ordered: close old position, then open the new position.  This
        # makes reversal attribution exact instead of assigning the new trade's
        # entry cost/return to the trade being closed.
        if position != 0 and target != position:
            exit_commission = equity * commission_pct
            exit_slippage = equity * slippage_pct
            commission += exit_commission
            slippage += exit_slippage
            equity -= exit_commission + exit_slippage
            if active_trade is None:
                raise AssertionError("open position has no active trade")
            active_trade.update(exit_date=ts,
                                net_pnl=float(equity - active_trade["entry_equity"]))
            trades.append(active_trade)
            active_trade = None
        if target != 0 and target != position:
            active_trade = {"entry_date": ts,
                            "direction": "LONG" if target == 1 else "SHORT",
                            "entry_equity": float(equity)}
            entry_commission = equity * commission_pct
            entry_slippage = equity * slippage_pct
            commission += entry_commission
            slippage += entry_slippage
            equity -= entry_commission + entry_slippage

        open_px = open_values[i]
        next_px = close_values[i] if is_terminal else open_values[eligible[ordinal + 1]]
        gross_return = float(target * (next_px / open_px - 1.0)) if open_px else 0.0
        equity *= 1.0 + gross_return

        # Liquidate after the terminal open-to-close return, at terminal equity.
        if is_terminal and target != 0:
            exit_commission = equity * commission_pct
            exit_slippage = equity * slippage_pct
            commission += exit_commission
            slippage += exit_slippage
            equity -= exit_commission + exit_slippage
            if active_trade is None:
                raise AssertionError("terminal position has no active trade")
            active_trade.update(exit_date=ts,
                                net_pnl=float(equity - active_trade["entry_equity"]))
            trades.append(active_trade)
            active_trade = None

        turnover = abs(target - position) + (abs(target) if is_terminal else 0)
        net_return = equity / equity_before - 1.0
        peak = max(peak, equity)
        rows.append({"timestamp": ts, "position": int(target), "gross_return": gross_return,
                     "turnover": float(turnover), "commission": float(commission),
                     "slippage": float(slippage), "net_return": float(net_return),
                     "equity": float(equity), "drawdown": float(equity / peak - 1.0)})

        position = target
        if is_terminal:
            position = 0

    ledger = pd.DataFrame(rows)
    ledger = _empty_ledger() if ledger.empty else ledger.set_index("timestamp")
    ledger.index = pd.DatetimeIndex(ledger.index)
    trade_pnl = sum(t["net_pnl"] for t in trades)
    if not np.isclose(trade_pnl, equity - initial_equity, rtol=1e-10, atol=1e-8):
        raise AssertionError("trade-P&L and portfolio-ledger reconciliation failed")
    return metrics_from_ledger(ledger, trades, initial_equity,
                               fast, slow, threshold, baseline_mode)
