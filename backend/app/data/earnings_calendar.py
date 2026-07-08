"""
Earnings-calendar feature: flags when a forecast window overlaps a company's
earnings report date, since these are known, scheduled volatility events that
a pure price-history model has no way to see coming from OHLCV alone.

Motivation (from backtesting on real AMZN data): the LSTM's price-magnitude
error spiked sharply in a backtest window that happened to contain an
earnings report (directional accuracy held up reasonably; magnitude did not).
Giving the model visibility into "an earnings report lands inside my forecast
horizon" lets it learn to widen/dampen its point estimate around these known
events instead of being blindsided by them.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from app import config


def fetch_earnings_dates(ticker: str) -> list:
    """
    Returns a sorted list of pandas.Timestamp earnings-report dates (past and
    upcoming) for the ticker.

    --- Primary path: real data via yfinance ---
    yfinance's Ticker.get_earnings_dates() pulls actual historical + upcoming
    report dates. Falls back to a synthetic quarterly schedule (~91 calendar
    days apart) if that's unreachable (sandboxed/offline environments, or an
    occasional API hiccup) — same fallback pattern used elsewhere in this
    project (market_data.py, news_corpus.py, vector_store.py).
    """
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        edf = t.get_earnings_dates(limit=40)
        if edf is None or edf.empty:
            raise ValueError("empty earnings calendar returned")
        dates = sorted(pd.to_datetime(edf.index).tz_localize(None))
        return dates
    except Exception as e:
        print(f"[earnings_calendar] Could not fetch real earnings dates for {ticker} ({e}); "
              f"falling back to a synthetic quarterly schedule.")
        return _synthetic_earnings_dates(ticker)


def _synthetic_earnings_dates(ticker: str, n_quarters: int = 12) -> list:
    """Deterministic-per-ticker synthetic quarterly earnings schedule (~91
    calendar days apart), spanning past and future, for offline use. Seeded
    per ticker so it's stable across runs, same as market_data's synthetic
    price generator."""
    rng = np.random.default_rng(abs(hash(ticker)) % (2**32))
    anchor = pd.Timestamp.today().normalize() - pd.Timedelta(days=int(rng.integers(0, 91)))
    dates = [anchor + pd.Timedelta(days=91 * k) for k in range(-n_quarters, n_quarters)]
    return sorted(dates)


def add_earnings_features(df: pd.DataFrame, ticker: str, window_days: int = None) -> pd.DataFrame:
    """
    Adds two columns to df:
      - "days_to_earnings": signed calendar days to the NEAREST earnings date
        (0 on the report date itself, negative if the closest one is in the past).
      - "earnings_in_horizon": 1 if any earnings date falls within the next
        `window_days` (defaults to config.FORECAST_HORIZON) days of this row,
        else 0 — the more directly useful signal: "a known volatility event
        lands inside what I'm being asked to forecast right now."
    """
    window_days = window_days or config.FORECAST_HORIZON
    earnings_dates = fetch_earnings_dates(ticker)

    df = df.copy()
    row_dates = pd.to_datetime(df["date"]).values

    days_to_earnings = np.zeros(len(df), dtype=np.float32)
    earnings_in_horizon = np.zeros(len(df), dtype=np.float32)

    if not earnings_dates:
        df["days_to_earnings"] = days_to_earnings
        df["earnings_in_horizon"] = earnings_in_horizon
        return df

    earnings_arr = np.array([np.datetime64(d) for d in earnings_dates])

    for idx, d in enumerate(row_dates):
        deltas_days = (earnings_arr - d).astype("timedelta64[D]").astype(np.float32)
        nearest_idx = int(np.argmin(np.abs(deltas_days)))
        days_to_earnings[idx] = deltas_days[nearest_idx]
        earnings_in_horizon[idx] = float(np.any((deltas_days >= 0) & (deltas_days <= window_days)))

    df["days_to_earnings"] = days_to_earnings
    df["earnings_in_horizon"] = earnings_in_horizon
    return df
