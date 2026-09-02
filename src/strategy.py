"""
strategy.py — Signal Generation Logic.

Modes supported:
1. "standard": EMA spatial crossover (0 angle).
2. "norm_angle": Proposed methodology (arctangent normalized slope).
3. "norm_slope": Normalized slope without arctangent.
4. "roc": Rate of Change based entry.
5. "macd": MACD Histogram based entry.
"""
import pandas as pd


def generate_signals(df: pd.DataFrame, threshold: float, mode: str = "norm_angle") -> pd.Series:
    fast: pd.Series = df["ema_fast"]
    slow: pd.Series = df["ema_slow"]
    
    # Initialize neutral signal array
    signal: pd.Series = pd.Series(0, index=df.index, dtype=int)
    long_raw = pd.Series(False, index=df.index)
    short_raw = pd.Series(False, index=df.index)
    
    # Base spatial crossover condition
    base_long = fast > slow
    base_short = fast < slow

    if mode == "norm_angle":
        fast_angle: pd.Series = df["angle_fast"]
        slow_angle: pd.Series = df["angle_slow"]
        long_raw = base_long & (fast_angle >= threshold) & (slow_angle >= 0)
        short_raw = base_short & (fast_angle <= -threshold) & (slow_angle <= 0)
        
    elif mode == "standard":
        long_raw = base_long
        short_raw = base_short

    elif mode == "norm_slope_fast":
        import numpy as np
        slope_threshold = np.tan(np.radians(threshold))
        slope = df["norm_slope_fast"]
        long_raw = base_long & (slope >= slope_threshold)
        short_raw = base_short & (slope <= -slope_threshold)
        
    elif mode == "norm_slope":
        # threshold is passed as angle, convert back to slope for equivalent comparison
        # tan(theta * pi / 180) = slope
        import numpy as np
        slope_thresh = np.tan(np.radians(threshold))
        norm_slope = df["norm_slope_fast"]
        long_raw = base_long & (norm_slope >= slope_thresh)
        short_raw = base_short & (norm_slope <= -slope_thresh)
        
    elif mode == "roc":
        roc = df["roc_fast"]
        long_raw = base_long & (roc >= threshold)
        short_raw = base_short & (roc <= -threshold)
        
    elif mode == "macd":
        macd = df["macd_hist"]
        long_raw = base_long & (macd > 0)
        short_raw = base_short & (macd < 0)

    elif mode == "log_velocity":
        velocity = df["log_velocity_fast"]
        long_raw = base_long & (velocity >= threshold)
        short_raw = base_short & (velocity <= -threshold)

    elif mode == "vol_scaled":
        velocity = df["vol_scaled_velocity_fast"]
        long_raw = base_long & (velocity >= threshold)
        short_raw = base_short & (velocity <= -threshold)

    elif mode == "tsmom":
        momentum = df["ts_momentum"]
        long_raw = momentum > threshold
        short_raw = momentum < -threshold

    elif mode == "buy_hold":
        long_raw[:] = True

    # Apply conditions
    signal[long_raw] = 1
    signal[short_raw] = -1

    return signal
