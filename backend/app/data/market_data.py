"""
Market data layer.

`fetch_ohlcv()` is the ONE function to swap for a real data source.
Everything else (indicators, sequence building) works on the returned
DataFrame regardless of where it came from.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from app.data.earnings_calendar import add_earnings_features


def fetch_ohlcv(ticker: str, days: int = 500, seed: int | None = None) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(ticker, period="10y", interval="1d", progress=False)  # was "2y"
    df = df.reset_index()
    df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
    df = df.rename(columns={"date": "date"})
    return df[["date", "open", "high", "low", "close", "volume"]].tail(days).reset_index(drop=True)

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adds SMA, RSI, MACD columns. Uses `ta` if available, else manual fallback."""
    df = df.copy()
    try:
        from ta.trend import SMAIndicator, MACD
        from ta.momentum import RSIIndicator
        df["sma_10"] = SMAIndicator(df["close"], window=10).sma_indicator()
        df["sma_30"] = SMAIndicator(df["close"], window=30).sma_indicator()
        df["rsi_14"] = RSIIndicator(df["close"], window=14).rsi()
        df["macd"] = MACD(df["close"]).macd()
    except ImportError:
        df["sma_10"] = df["close"].rolling(10).mean()
        df["sma_30"] = df["close"].rolling(30).mean()
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi_14"] = 100 - (100 / (1 + rs))
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26

    df = df.bfill().ffill()
    return df


def get_prepared_data(ticker: str, days: int = 500) -> pd.DataFrame:
    df = fetch_ohlcv(ticker, days=days)
    df = add_technical_indicators(df)
    df = add_earnings_features(df, ticker)   # <- add this line
    return df


if __name__ == "__main__":
    d = get_prepared_data("NVDA")
    print(d.tail())
    print(d.shape)
