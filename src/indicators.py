r"""
indicators.py — Mathematical computation of Exponential Moving Averages (EMA) and derived vector geometry.

This module is responsible for calculating price indicators and extracting
their normalized momentum properties.
"""

import numpy as np
import pandas as pd


def calc_ema(series: pd.Series, period: int) -> pd.Series:
    """Computes the Exponential Moving Average (EMA)."""
    return series.ewm(span=period, adjust=False).mean()

def calc_roc(series: pd.Series, period: int) -> pd.Series:
    """Computes Rate of Change."""
    return series.pct_change(periods=period) * 100

def calc_macd_hist(series: pd.Series, fast: int, slow: int, signal: int = 9) -> pd.Series:
    """Computes MACD Histogram."""
    ema_fast = calc_ema(series, fast)
    ema_slow = calc_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    return macd_line - signal_line
    
def calc_normalized_angle(ema_series: pd.Series) -> pd.Series:
    """
    Computes the Normalized Angle (\theta) of the EMA trajectory in degrees.
    """
    normalized_slope = (ema_series - ema_series.shift(1)) / ema_series.shift(1)
    return np.degrees(np.arctan(normalized_slope))
    
def calc_normalized_slope(ema_series: pd.Series) -> pd.Series:
    """
    Computes purely the normalized slope without arctangent squashing.
    """
    return (ema_series - ema_series.shift(1)) / ema_series.shift(1)


def calc_log_velocity(ema_series: pd.Series) -> pd.Series:
    """Log EMA change per elapsed hour (price- and frequency-normalized)."""
    log_change = np.log(ema_series).diff()
    if isinstance(ema_series.index, pd.DatetimeIndex):
        hours = ema_series.index.to_series().diff().dt.total_seconds() / 3600.0
        hours.index = ema_series.index
        return log_change / hours.replace(0, np.nan)
    return log_change


def calc_volatility_scaled_velocity(ema_series: pd.Series, price: pd.Series,
                                    window: int = 20) -> pd.Series:
    """One-bar log EMA change scaled by trailing price-return volatility."""
    numerator = np.log(ema_series).diff()
    denominator = np.log(price).diff().rolling(window, min_periods=window).std()
    return numerator / denominator.replace(0, np.nan)


def add_ema_features(df: pd.DataFrame, fast: int, slow: int,
                     macd_signal: int = 9) -> pd.DataFrame:
    """
    Appends dual EMA values and their normalized angles to the OHLCV dataframe.
    """
    feat: pd.DataFrame = df.copy()

    feat["ema_fast"] = calc_ema(feat["Close"], fast)
    feat["ema_slow"] = calc_ema(feat["Close"], slow)

    feat["angle_fast"] = calc_normalized_angle(feat["ema_fast"])
    feat["angle_slow"] = calc_normalized_angle(feat["ema_slow"])
    
    # Baselines
    feat["norm_slope_fast"] = calc_normalized_slope(feat["ema_fast"])
    feat["norm_slope_slow"] = calc_normalized_slope(feat["ema_slow"])
    feat["log_velocity_fast"] = calc_log_velocity(feat["ema_fast"])
    feat["vol_scaled_velocity_fast"] = calc_volatility_scaled_velocity(
        feat["ema_fast"], feat["Close"])
    feat["roc_fast"] = calc_roc(feat["Close"], fast)
    feat["macd_hist"] = calc_macd_hist(feat["Close"], fast, slow, macd_signal)
    feat["ts_momentum"] = feat["Close"].pct_change(slow)
    
    return feat
