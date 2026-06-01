"""
Sonifier v3.0 — Candlestick Pattern Detection

Pure NumPy/Pandas implementations of common candlestick patterns.
Returns a dict of pattern_name -> list of (index, confidence) tuples.
Used by the UI to annotate the sonification.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def _body_frac(df: pd.DataFrame) -> np.ndarray:
    return np.abs(df["Close"].values - df["Open"].values) / (
        np.abs(df["Open"].values) + 1e-12
    )


def _upper_wick(df: pd.DataFrame) -> np.ndarray:
    close = df["Close"].values
    open_ = df["Open"].values
    high  = df["High"].values
    return high - np.maximum(close, open_)


def _lower_wick(df: pd.DataFrame) -> np.ndarray:
    close = df["Close"].values
    open_ = df["Open"].values
    low   = df["Low"].values
    return np.minimum(close, open_) - low


def _bullish(close, open_):
    return close >= open_


def _bearish(close, open_):
    return close < open_


def detect_patterns(df: pd.DataFrame) -> Dict[str, List[Tuple[int, float]]]:
    """
    Scan DataFrame for classic candlestick patterns.

    Returns
    -------
    dict: { pattern_name: [(index, confidence), ...] }
    Confidence is 0.0–1.0 (higher = stronger signal).
    """
    results: Dict[str, List[Tuple[int, float]]] = {}

    n = len(df)
    if n < 2:
        return results

    close = df["Close"].values
    open_ = df["Open"].values
    high  = df["High"].values
    low   = df["Low"].values
    body  = _body_frac(df)
    uwick = _upper_wick(df)
    lwick = _lower_wick(df)
    total_range = high - low + 1e-12

    # ── Doji ────────────────────────────────────────────────────
    doji_body = body < 0.03
    # Wick must be meaningful
    doji_wick = (total_range > 0) & ((body / total_range) < 0.15)
    doji_idx = np.where(doji_body & doji_wick)[0]
    if len(doji_idx):
        results["doji"] = [(int(i), 0.7 + 0.3 * (1 - body[i] / total_range]))
                           for i in doji_idx]

    # ── Hammer ──────────────────────────────────────────────────
    if n >= 1:
        hammer_lower = lwick > body * 2.0        # long lower shadow
        hammer_upper = uwick < body * 0.5        # small upper shadow
        hammer_body  = body > 0.01               # non-trivial body
        hammer_idx   = np.where(hammer_lower & hammer_upper & hammer_body)[0]
        if len(hammer_idx):
            results["hammer"] = [(int(i), 0.7 + 0.3 * min(1.0, lwick[i] / (body[i] * 4)))
                                  for i in hammer_idx]

    # ── Shooting Star ──────────────────────────────────────────
    if n >= 1:
        ss_upper = uwick > body * 2.0
        ss_lower = lwick < body * 0.5
        ss_body  = body > 0.01
        ss_idx   = np.where(ss_upper & ss_lower & ss_body)[0]
        if len(ss_idx):
            results["shooting_star"] = [(int(i), 0.7 + 0.3 * min(1.0, uwick[i] / (body[i] * 4)))
                                         for i in ss_idx]

    # ── Bullish Engulfing ───────────────────────────────────────
    if n >= 2:
        prev_bear  = _bearish(close[:-1], open_[:-1])
        curr_bull  = _bullish(close[1:],  open_[1:])
        engulf_body = body[1:] > body[:-1]
        engulf_idx  = np.where(prev_bear & curr_bull & engulf_body)[0] + 1
        if len(engulf_idx):
            results["engulfing_bull"] = [(int(i), 0.65 + 0.35 * min(1.0, body[i] / max(body[i - 1], 1e-12)))
                                          for i in engulf_idx]

    # ── Bearish Engulfing ───────────────────────────────────────
    if n >= 2:
        prev_bull  = _bullish(close[:-1], open_[:-1])
        curr_bear  = _bearish(close[1:],  open_[1:])
        engulf_body = body[1:] > body[:-1]
        engulf_idx  = np.where(prev_bull & curr_bear & engulf_body)[0] + 1
        if len(engulf_idx):
            results["engulfing_bear"] = [(int(i), 0.65 + 0.35 * min(1.0, body[i] / max(body[i - 1], 1e-12)))
                                          for i in engulf_idx]

    # ── Three White Soldiers ───────────────────────────────────
    if n >= 3:
        for i in range(2, n):
            b1 = _bullish(close[i],   open_[i])
            b2 = _bullish(close[i-1], open_[i-1])
            b3 = _bullish(close[i-2], open_[i-2])
            if b1 and b2 and b3:
                if close[i] > close[i-1] > close[i-2]:
                    conf = 0.6 + 0.4 * min(1.0, body[i] / 0.05)
                    results.setdefault("three_white", []).append((int(i), conf))

    # ── Three Black Crows ──────────────────────────────────────
    if n >= 3:
        for i in range(2, n):
            b1 = _bearish(close[i],   open_[i])
            b2 = _bearish(close[i-1], open_[i-1])
            b3 = _bearish(close[i-2], open_[i-2])
            if b1 and b2 and b3:
                if close[i] < close[i-1] < close[i-2]:
                    conf = 0.6 + 0.4 * min(1.0, body[i] / 0.05)
                    results.setdefault("three_black", []).append((int(i), conf))

    # ── Morning Star ───────────────────────────────────────────
    if n >= 3:
        for i in range(2, n):
            c0, c1, c2 = close[i-2], close[i-1], close[i]
            o0, o1, o2 = open_[i-2], open_[i-1], open_[i]
            if (_bearish(c0, o0) and
                abs(c1 - o1) < abs(c0 - o0) * 0.3 and
                _bullish(c2, o2) and
                close[i] > (open_[i-2] + close[i-2]) / 2):
                results.setdefault("morning_star", []).append((int(i), 0.7))

    # ── Evening Star ───────────────────────────────────────────
    if n >= 3:
        for i in range(2, n):
            c0, c1, c2 = close[i-2], close[i-1], close[i]
            o0, o1, o2 = open_[i-2], open_[i-1], open_[i]
            if (_bullish(c0, o0) and
                abs(c1 - o1) < abs(c0 - o0) * 0.3 and
                _bearish(c2, o2) and
                close[i] < (open_[i-2] + close[i-2]) / 2):
                results.setdefault("evening_star", []).append((int(i), 0.7))

    # ── Piercing Line ──────────────────────────────────────────
    if n >= 2:
        for i in range(1, n):
            if (_bearish(close[i-1], open_[i-1]) and
                _bullish(close[i], open_[i]) and
                open_[i] < low[i-1] and
                close[i] > (open_[i-1] + close[i-1]) / 2 and
                close[i] < open_[i-1]):
                results.setdefault("piercing", []).append((int(i), 0.75))

    # ── Dark Cloud Cover ───────────────────────────────────────
    if n >= 2:
        for i in range(1, n):
            if (_bullish(close[i-1], open_[i-1]) and
                _bearish(close[i], open_[i]) and
                open_[i] > high[i-1] and
                close[i] < (open_[i-1] + close[i-1]) / 2 and
                close[i] > open_[i-1]):
                results.setdefault("dark_cloud", []).append((int(i), 0.75))

    return results
