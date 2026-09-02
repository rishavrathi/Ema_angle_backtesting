"""Independent reconciliation audit for committed datasets and current engine."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import run_backtest
from src.config import (COMMISSION_PCT, EMA_PAIRS, INITIAL_EQUITY,
                        SLIPPAGE_PCT, RESULT_LOGS_DIR)

ROOT = Path(__file__).resolve().parent


def run_audit():
    discrepancies = []
    checked = 0
    generated_ledgers_checked = 0
    max_reconciliation_error = 0.0
    for path in sorted((ROOT / "data").glob("*.csv")):
        data = pd.read_csv(path, parse_dates=["Datetime"]).set_index("Datetime")
        fast, slow = EMA_PAIRS[0]
        result = run_backtest(data, fast, slow, 0.05, INITIAL_EQUITY,
                              COMMISSION_PCT, SLIPPAGE_PCT,
                              baseline_mode="norm_angle")
        ledger = result["bar_returns"]
        reconciled = INITIAL_EQUITY if ledger.empty else INITIAL_EQUITY * float(
            (1 + ledger["net_return"]).prod())
        if not np.isclose(reconciled, result["final_equity"], rtol=1e-10, atol=1e-8):
            discrepancies.append({"file": path.name, "metric": "final_equity"})
        if not np.isclose(result["total_net_profit"],
                          result["final_equity"] - INITIAL_EQUITY, atol=1e-8):
            discrepancies.append({"file": path.name, "metric": "net_profit"})
        checked += 1
    ledger_root = ROOT / "results" / "final_walk_forward" / "bar_returns"
    for path in sorted(ledger_root.glob("*.csv")):
        ledger = pd.read_csv(path)
        if ledger.empty:
            continue
        compounded = INITIAL_EQUITY * float((1 + ledger["net_return"]).prod())
        reported = float(ledger["equity"].iloc[-1])
        error = abs(compounded - reported)
        max_reconciliation_error = max(max_reconciliation_error, error)
        if not np.isclose(compounded, reported, rtol=1e-10, atol=1e-8):
            discrepancies.append({"file": path.name, "metric": "exported_ledger"})
        generated_ledgers_checked += 1
    output = {"datasets_checked": checked, "discrepancies": discrepancies,
              "generated_ledgers_checked": generated_ledgers_checked,
              "max_final_equity_reconciliation_error": max_reconciliation_error,
              "status": "Pass" if not discrepancies else "Fail"}
    destination = RESULT_LOGS_DIR / "audit_report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    run_audit()
