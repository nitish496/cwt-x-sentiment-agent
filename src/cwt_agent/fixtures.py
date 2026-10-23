"""Deterministic synthetic fixtures for software checks only."""

from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_demo(days: int = 240, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return prices/posts marked synthetic; never use as empirical evidence."""
    if days < 120:
        raise ValueError("synthetic demo requires at least 120 days")
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2025-01-01", periods=days, freq="D", tz="UTC")
    log_returns = rng.normal(0.0003, 0.025, days)
    close = 90_000 * np.exp(np.cumsum(log_returns))
    prices = pd.DataFrame({
        "timestamp_utc": timestamps,
        "open": np.r_[close[0], close[:-1]],
        "high": np.maximum(np.r_[close[0], close[:-1]], close) * 1.01,
        "low": np.minimum(np.r_[close[0], close[:-1]], close) * 0.99,
        "close": close,
        "volume": rng.uniform(100, 500, days),
        "symbol": "BTC-USD",
        "venue": "synthetic-fixture",
        "quote_currency": "USD",
        "source": "synthetic_fixture_not_market_data",
        "data_label": "synthetic fixture",
    })
    posts = pd.DataFrame({
        "post_id": [f"synthetic-{index:05d}" for index in range(days)],
        "created_at_utc": timestamps + pd.Timedelta(hours=12),
        "text": ["synthetic bullish example" if value > 0 else "synthetic bearish example" for value in log_returns],
        "sentiment_score": np.clip(log_returns / 0.05, -1, 1),
        "provider": "synthetic-fixture",
        "source_id": "synthetic-fixture-only",
        "data_label": "synthetic fixture",
    })
    return prices, posts
