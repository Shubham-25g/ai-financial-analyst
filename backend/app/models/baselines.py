"""
Shared baseline-forecast logic — used by training (to build residual
targets), inference (to add the drift back to the LSTM's residual
prediction), and the backtest script (to score baselines fairly against
the same drift definition the model was trained against).

Keeping this in one place means training/inference/backtest can't
silently drift out of sync with each other.
"""
from __future__ import annotations
import numpy as np


def compute_drift_log_returns(close_values: np.ndarray, i: int, horizon: int,
                               lookback: int = 60) -> np.ndarray:
    """
    Cumulative drift-baseline log-returns for steps 1..horizon, estimated
    from the average daily log-return over the `lookback` days strictly
    before index i (i.e. using only information available at prediction time).

    Returns an array of length `horizon`: drift[k-1] = avg_daily_return * k.
    """
    window = close_values[max(0, i - lookback - 1):i]
    if len(window) < 2:
        avg_daily_return = 0.0
    else:
        log_returns = np.diff(np.log(window))
        avg_daily_return = float(log_returns.mean())
    steps = np.arange(1, horizon + 1)
    return avg_daily_return * steps


def drift_price_forecast(close_values: np.ndarray, i: int, horizon: int,
                          lookback: int = 60) -> np.ndarray:
    """Drift baseline as actual prices (not log-returns) — used directly by the backtest."""
    last_close = close_values[i - 1]
    drift_log_returns = compute_drift_log_returns(close_values, i, horizon, lookback)
    return last_close * np.exp(drift_log_returns)


def naive_price_forecast(close_values: np.ndarray, i: int, horizon: int) -> np.ndarray:
    """Naive baseline: flat line at the last known price."""
    return np.full(horizon, close_values[i - 1])
