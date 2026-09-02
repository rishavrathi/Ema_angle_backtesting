import hashlib

import numpy as np
import pandas as pd
import pytest

from src.backtest import _risk_metrics, run_backtest
from src.indicators import add_ema_features, calc_ema, calc_normalized_angle
from src.optimizer import FOLDS, MACD_SIGNAL_PERIODS, QUANTILES, _thresholds
from src.statistical_tests import run_statistical_suite
from src.strategy import generate_signals


@pytest.fixture
def prices() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=160, freq="h", tz="UTC")
    close = 100 + np.sin(np.arange(160) / 7) * 4 + np.arange(160) * 0.05
    return pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1,
                         "Close": close, "Volume": 1000}, index=index)


def deterministic_features(opens, directions):
    """Minimal precomputed feature frame with an exact standard-EMA signal path."""
    index = pd.date_range("2024-01-01", periods=len(opens), freq="h", tz="UTC")
    directions = np.asarray(directions)
    return pd.DataFrame({"Open": opens, "Close": opens,
        "ema_fast": 100 + directions, "ema_slow": np.full(len(opens), 100.0)},
        index=index)


def exact_run(opens, directions, commission=0.0, slippage=0.0):
    feat = deterministic_features(opens, directions)
    return run_backtest(feat, 1, 2, 0.0, 10000.0, commission, slippage,
                        baseline_mode="standard", execution_start=feat.index[1],
                        execution_end=feat.index[-1], precomputed_features=feat)


@pytest.mark.parametrize(("directions", "expected_direction", "expected_gross"), [
    ([1, 1, 1, 1], "LONG", (110 / 100) * (121 / 110) - 1),
    ([-1, -1, -1, -1], "SHORT", (2 - 90 / 100) * (2 - 81 / 90) - 1),
])
def test_exact_long_and_short_entry_exit(directions, expected_direction, expected_gross):
    opens = [100, 100, 110, 121] if expected_direction == "LONG" else [100, 100, 90, 81]
    result = exact_run(opens, directions)
    assert result["total_trades"] == 1
    assert result["trades"][0]["direction"] == expected_direction
    assert result["trades"][0]["net_pnl"] == pytest.approx(10000 * expected_gross)
    assert result["bar_returns"]["turnover"].tolist() == [1.0, 0.0, 1.0]


@pytest.mark.parametrize("directions", ([1, -1, -1, -1], [-1, 1, 1, 1]))
def test_exact_reversal_trade_boundaries_reconcile(directions):
    result = exact_run([100, 100, 110, 99], directions, commission=0.001,
                       slippage=0.002)
    assert [t["direction"] for t in result["trades"]] == (
        ["LONG", "SHORT"] if directions[0] == 1 else ["SHORT", "LONG"])
    assert result["bar_returns"]["turnover"].tolist() == [1.0, 2.0, 1.0]
    assert sum(t["net_pnl"] for t in result["trades"]) == pytest.approx(
        result["final_equity"] - 10000.0)


def test_exact_entry_exit_commission_slippage_and_terminal_liquidation():
    result = exact_run([100, 100, 110, 110], [1, 1, 1, 1], commission=0.01, slippage=0.02)
    ledger = result["bar_returns"]
    # Entry: 100 commission + 200 slippage. Terminal exit is charged on 10,670.
    assert ledger.iloc[0].commission == pytest.approx(100.0)
    assert ledger.iloc[0].slippage == pytest.approx(200.0)
    assert ledger.iloc[-1].commission == pytest.approx(106.7)
    assert ledger.iloc[-1].slippage == pytest.approx(213.4)
    assert result["final_equity"] == pytest.approx(10349.9)
    assert result["total_turnover"] == 2.0


def test_exact_mark_to_market_and_all_reconciliations():
    result = exact_run([100, 100, 120, 90], [1, 1, 1, 1])
    ledger = result["bar_returns"]
    assert ledger.equity.tolist() == pytest.approx([12000.0, 9000.0, 9000.0])
    assert ledger.drawdown.tolist() == pytest.approx([0.0, -0.25, -0.25])
    assert sum(t["net_pnl"] for t in result["trades"]) == pytest.approx(-1000.0)
    assert result["final_equity"] == pytest.approx(10000 * (1 + ledger.net_return).prod())
    assert result["total_net_profit"] == pytest.approx(result["final_equity"] - 10000)


def test_ema_first_value(prices):
    assert calc_ema(prices["Close"], 3).iloc[0] == prices["Close"].iloc[0]


def test_scale_invariance(prices):
    a = add_ema_features(prices, 9, 21)
    b = add_ema_features(prices * 100.0, 9, 21)
    np.testing.assert_allclose(a["angle_fast"], b["angle_fast"], equal_nan=True)


def test_angle_and_normalized_slope_signal_equivalence(prices):
    feat = add_ema_features(prices, 9, 21)
    angle = generate_signals(feat, 0.05, "norm_angle")
    slope_threshold = np.tan(np.radians(0.05))
    manual = pd.Series(0, index=feat.index)
    manual[(feat.ema_fast > feat.ema_slow) &
           (feat.norm_slope_fast >= slope_threshold) &
           (feat.angle_slow >= 0)] = 1
    manual[(feat.ema_fast < feat.ema_slow) &
           (feat.norm_slope_fast <= -slope_threshold) &
           (feat.angle_slow <= 0)] = -1
    pd.testing.assert_series_equal(angle, manual.astype(int))


def test_future_prices_do_not_change_past_results(prices):
    cutoff = prices.index[119]
    original = run_backtest(prices, 9, 21, 0.0, 10000, baseline_mode="standard",
                            execution_start=prices.index[40], execution_end=cutoff)
    changed = prices.copy()
    changed.loc[changed.index > cutoff, ["Open", "High", "Low", "Close"]] *= 20
    rerun = run_backtest(changed, 9, 21, 0.0, 10000, baseline_mode="standard",
                         execution_start=prices.index[40], execution_end=cutoff)
    pd.testing.assert_frame_equal(original["bar_returns"], rerun["bar_returns"])


def test_entry_exit_costs_and_terminal_liquidation(prices):
    result = run_backtest(prices, 9, 21, 0.0, 10000, commission_pct=0.001,
                          slippage_pct=0.001, baseline_mode="buy_hold",
                          execution_start=prices.index[40], execution_end=prices.index[80])
    ledger = result["bar_returns"]
    assert ledger["turnover"].sum() >= 2
    assert result["total_commission_paid"] > 0
    assert result["total_slippage_paid"] > 0
    assert result["total_trades"] == 1


def test_final_equity_reconciliation(prices):
    result = run_backtest(prices, 9, 21, 0.03, 10000, commission_pct=0.0005,
                          slippage_pct=0.0005, baseline_mode="norm_angle")
    expected = 10000 * (1 + result["bar_returns"].net_return).prod()
    assert result["final_equity"] == pytest.approx(expected, rel=1e-12)
    assert result["total_net_profit"] == pytest.approx(result["final_equity"] - 10000)


def test_mark_to_market_drawdown():
    index = pd.date_range("2024-01-01", periods=80, freq="h", tz="UTC")
    close = np.r_[np.linspace(100, 120, 40), np.linspace(120, 70, 40)]
    data = pd.DataFrame({"Open": close, "High": close, "Low": close,
                         "Close": close, "Volume": 1}, index=index)
    result = run_backtest(data, 3, 5, 0, 10000, baseline_mode="buy_hold",
                          execution_start=index[25])
    assert result["max_drawdown"] > 30
    assert (result["bar_returns"].drawdown < 0).any()


def test_long_short_symmetry(prices):
    upward = prices.copy()
    trend = np.linspace(100, 150, len(prices))
    for column in ["Open", "High", "Low", "Close"]:
        upward[column] = trend
    downward = upward.copy()
    inverse = 10000 / upward["Close"]
    for column in ["Open", "High", "Low", "Close"]:
        downward[column] = inverse.to_numpy()
    long_run = run_backtest(upward, 3, 9, 0, 10000, baseline_mode="standard")
    short_run = run_backtest(downward, 3, 9, 0, 10000, baseline_mode="standard")
    assert long_run["total_net_profit"] > 0
    assert short_run["total_net_profit"] > 0


def test_walk_forward_fold_separation():
    assert FOLDS == [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80),
                     (0.80, 0.90), (0.90, 1.00)]
    for train_end, test_end in FOLDS:
        assert 0 < train_end < test_end <= 1
        assert test_end - train_end == pytest.approx(0.10)


def test_predefined_macd_grid_and_training_only_quantiles(prices):
    assert MACD_SIGNAL_PERIODS == (5, 9, 15)
    train = prices.iloc[:80]
    for mode in ("norm_angle", "log_velocity", "vol_scaled"):
        thresholds = tuple(_thresholds(train, 9, 21, mode))
        assert len(thresholds) == len(QUANTILES)
        changed_future = prices.copy()
        changed_future.iloc[80:, changed_future.columns.get_indexer(
            ["Open", "High", "Low", "Close"])] *= 100
        assert thresholds == tuple(_thresholds(changed_future.iloc[:80], 9, 21, mode))


def test_one_minute_is_excluded_not_zero_filled():
    empty_metric = {"bar_returns": pd.DataFrame(columns=["equity"])}
    wfa = {"Gold": {"1m": {"out_of_sample_metrics": empty_metric,
                              "candidate_universe": {}}}}
    baselines = {"Gold": {"1m": {"standard": empty_metric}}}
    result = run_statistical_suite(baselines, wfa)
    assert result.empty


def test_daily_risk_metrics_are_frequency_independent():
    daily_index = pd.date_range("2024-01-01", periods=20, freq="D", tz="UTC")
    equity = 10000 * np.cumprod(1 + np.linspace(-0.01, 0.015, 20))
    base = pd.DataFrame({"equity": equity, "drawdown": equity / np.maximum.accumulate(equity) - 1},
                        index=daily_index)
    hourly = base.reindex(pd.date_range(daily_index[0], daily_index[-1], freq="h", tz="UTC")).ffill()
    assert _risk_metrics(base, 10000)["sharpe_ratio"] == pytest.approx(
        _risk_metrics(hourly, 10000)["sharpe_ratio"])


def test_deterministic_output_hash(prices):
    kwargs = dict(df=prices, fast=9, slow=21, threshold=0.03,
                  initial_equity=10000, baseline_mode="norm_angle")
    first = run_backtest(**kwargs)["bar_returns"].to_csv(float_format="%.12g")
    second = run_backtest(**kwargs)["bar_returns"].to_csv(float_format="%.12g")
    assert hashlib.sha256(first.encode()).hexdigest() == hashlib.sha256(second.encode()).hexdigest()
