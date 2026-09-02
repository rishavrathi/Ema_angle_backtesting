import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import json

from src.backtest import run_backtest
from src.config import EMA_PAIRS, ASSETS, TIMEFRAMES

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_FILE = BASE_DIR / "results" / "excel" / "EMA_Angle_Empirical_Results.xlsx"
OUTPUT_FILE = BASE_DIR / "results" / "logs" / "statistical_analysis.json"

def compute_buy_and_hold():
    results = {}
    for file in DATA_DIR.glob("*.csv"):
        parts = file.stem.split("_")
        if len(parts) != 2:
            continue
        asset, tf = parts[0], parts[1]
        
        df = pd.read_csv(file)
        if len(df) == 0:
            continue
            
        first_px = df['Open'].iloc[0]
        last_px = df['Close'].iloc[-1]
        
        initial_equity = 10000.0
        shares = initial_equity / first_px
        final_equity = shares * last_px
        net_profit = final_equity - initial_equity
        ret_pct = net_profit / initial_equity * 100.0
        
        df['Equity'] = shares * df['Close']
        df['Peak'] = df['Equity'].cummax()
        df['Drawdown'] = (df['Equity'] - df['Peak']) / df['Peak']
        max_dd = df['Drawdown'].min() * 100.0
        
        ret_series = df['Equity'].pct_change().dropna()
        if ret_series.std() == 0:
            sharpe = 0.0
        else:
            if tf == '1m': bars_per_year = 252 * 24 * 60
            elif tf == '5m': bars_per_year = 252 * 24 * 12
            elif tf == '15m': bars_per_year = 252 * 24 * 4
            elif tf == '1h': bars_per_year = 252 * 24
            else: bars_per_year = 252
            sharpe = (ret_series.mean() / ret_series.std()) * np.sqrt(bars_per_year)
            
        results[f"{asset}_{tf}"] = {
            "Net Profit": float(net_profit),
            "Return %": float(ret_pct),
            "Max Drawdown %": float(max_dd),
            "Sharpe Ratio": float(sharpe)
        }
    return results

def compare_baselines():
    # Execute threshold=0 (Standard EMA Crossover) vs threshold=0.05 (Normalized EMA Angle Crossover)
    baseline_metrics = []
    treatment_metrics = []
    
    for asset in ASSETS:
        for tf in TIMEFRAMES:
            file_path = DATA_DIR / f"{asset}_{tf}.csv"
            if not file_path.exists():
                continue
            
            try:
                df = pd.read_csv(file_path, parse_dates=['Datetime'])
                df.set_index('Datetime', inplace=True)
            except Exception:
                continue
            
            for fast, slow in EMA_PAIRS:
                # Baseline: 0 degree
                res_base = run_backtest(df, fast, slow, 0.0, 10000.0,
                                        baseline_mode="standard")
                # Treatment: 0.05 degree
                res_treat = run_backtest(df, fast, slow, 0.05, 10000.0,
                                         baseline_mode="norm_angle")
                
                if res_base['total_trades'] > 0 and res_treat['total_trades'] > 0:
                    baseline_metrics.append({
                        "Win Rate": res_base['win_rate'],
                        "Net Profit": res_base['total_net_profit'],
                        "Sharpe Ratio": res_base['sharpe_ratio'],
                        "Max Drawdown": res_base['max_drawdown']
                    })
                    treatment_metrics.append({
                        "Win Rate": res_treat['win_rate'],
                        "Net Profit": res_treat['total_net_profit'],
                        "Sharpe Ratio": res_treat['sharpe_ratio'],
                        "Max Drawdown": res_treat['max_drawdown']
                    })

    if len(baseline_metrics) < 5:
        return {"error": "Not enough data for statistical testing."}
        
    df_base = pd.DataFrame(baseline_metrics)
    df_treat = pd.DataFrame(treatment_metrics)
    
    stats_out = {}
    
    # Wilcoxon signed-rank test
    stat_wr, p_wr = stats.wilcoxon(df_base['Win Rate'], df_treat['Win Rate'], zero_method='zsplit')
    stat_np, p_np = stats.wilcoxon(df_base['Net Profit'], df_treat['Net Profit'], zero_method='zsplit')
    stat_sh, p_sh = stats.wilcoxon(df_base['Sharpe Ratio'], df_treat['Sharpe Ratio'], zero_method='zsplit')
    
    stats_out['Wilcoxon_WinRate'] = {"statistic": float(stat_wr), "p_value": float(p_wr)}
    stats_out['Wilcoxon_NetProfit'] = {"statistic": float(stat_np), "p_value": float(p_np)}
    stats_out['Wilcoxon_Sharpe'] = {"statistic": float(stat_sh), "p_value": float(p_sh)}
    
    stats_out['Mean_WinRate_Base'] = float(df_base['Win Rate'].mean())
    stats_out['Mean_WinRate_Treat'] = float(df_treat['Win Rate'].mean())
    stats_out['Mean_Sharpe_Base'] = float(df_base['Sharpe Ratio'].mean())
    stats_out['Mean_Sharpe_Treat'] = float(df_treat['Sharpe Ratio'].mean())
    stats_out['Mean_MaxDrawdown_Base'] = float(df_base['Max Drawdown'].mean())
    stats_out['Mean_MaxDrawdown_Treat'] = float(df_treat['Max Drawdown'].mean())

    return stats_out

if __name__ == "__main__":
    bnh = compute_buy_and_hold()
    stat_test = compare_baselines()
    
    out = {
        "Buy_and_Hold": bnh,
        "Statistical_Significance": stat_test
    }
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(out, f, indent=4)
        
    print(f"Successfully generated {OUTPUT_FILE}")
