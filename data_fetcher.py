"""
Sonifier v3.0 — Data Fetching & Cleaning

Fetches OHLCV data from yfinance.
Returns a clean, gap-free DataFrame ready for sonification.
"""

import yfinance as yf
import pandas as pd
import warnings


def fetch_ohlcv(
    ticker: str,
    period: str = "3mo",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Yahoo Finance.

    Parameters
    ----------
    ticker:  Yahoo ticker symbol (e.g. "BTC-USD", "AAPL", "^GSPC").
    period:  yfinance period string  ("7d", "1mo", "3mo", "6mo", "1y", "5y").
    interval: yfinance interval string ("1m", "5m", "1h", "1d", "1wk", "1mo").

    Returns
    -------
    DataFrame with columns [Open, High, Low, Close, Volume],
    DatetimeIndex, NaN rows removed.

    Raises
    ------
    ValueError  if no data returned or ticker is invalid.
    ConnectionError  if download fails.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False)
        except Exception as exc:
            raise ConnectionError(f"Failed to download {ticker}: {exc}") from exc

    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}' "
                         f"(period={period}, interval={interval}). "
                         f"Check the symbol is valid for Yahoo Finance.")

    # Flatten MultiIndex columns if present (yfinance ≥ 0.2 quirk)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Ensure required columns exist
    required = {"Open", "High", "Low", "Close"}
    have = set(df.columns)
    missing = required - have
    if missing:
        raise ValueError(f"Data missing columns: {missing}. Got: {list(have)}")

    # Drop rows where Close is NaN (holidays / incomplete candles)
    df = df.dropna(subset=["Close"]).copy()

    # Ensure Volume column (fill missing with 0)
    if "Volume" not in df.columns:
        df["Volume"] = 0
    else:
        df["Volume"] = df["Volume"].fillna(0)

    df.index = pd.DatetimeIndex(df.index)
    return df[["Open", "High", "Low", "Close", "Volume"]]
