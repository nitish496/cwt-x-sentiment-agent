"""Leakage-conscious daily sentiment and price feature construction.

Rows represent completed UTC days. Sentiment is aggregated from posts created
within each UTC day; price features use only closes on or before that day.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import lru_cache

import numpy as np
import pandas as pd

PRICE_LAGS = (1, 3, 7, 14, 30)
VOLATILITY_WINDOWS = (7, 14, 30)
SMA_WINDOWS = (7, 30)
HORIZONS = (1, 2, 3, 4, 5)

SENTIMENT_FEATURES = (
    "post_count",
    "mean_sentiment",
    "positive_share",
    "negative_share",
    "sentiment_dispersion",
    "has_posts",
)


@lru_cache(maxsize=1)
def _default_sentiment_analyzer():
    """Create VADER lazily so callers can inject a scorer in offline tests."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError as exc:  # pragma: no cover - depends on optional runtime setup
        raise RuntimeError(
            "Install the project dependencies or pass sentiment_scores explicitly."
        ) from exc
    return SentimentIntensityAnalyzer()


def score_texts(
    texts: Sequence[object], *, scorer: Callable[[str], float] | None = None
) -> pd.Series:
    """Return unweighted polarity scores in [-1, 1] for post text.

    The default scorer uses VADER compound polarity. A callable may be injected
    for tests or a separately configured sentiment model.
    """
    if scorer is None:
        analyzer = _default_sentiment_analyzer()
        scorer = lambda text: analyzer.polarity_scores(text)["compound"]

    scores: list[float] = []
    for value in texts:
        text = "" if value is None or pd.isna(value) else str(value)
        score = float(scorer(text))
        if not np.isfinite(score) or score < -1 or score > 1:
            raise ValueError("sentiment scores must be finite values in [-1, 1]")
        scores.append(score)
    return pd.Series(scores, dtype="float64")


def _utc_days(values: pd.Series, name: str) -> pd.Series:
    """Parse timestamp values, require timezone-aware inputs, and floor in UTC."""
    for value in values:
        if pd.isna(value):
            raise ValueError(f"{name} contains a missing or invalid timestamp")
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} contains a missing or invalid timestamp") from exc
        if timestamp.tzinfo is None:
            raise ValueError(f"{name} must contain timezone-aware timestamps")
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"{name} contains a missing or invalid timestamp")
    return parsed.dt.floor("D")


def aggregate_daily_sentiment(
    posts: pd.DataFrame,
    *,
    scorer: Callable[[str], float] | None = None,
    positive_threshold: float = 0.05,
) -> pd.DataFrame:
    """Aggregate normalized post records to UTC-day sentiment features.

    Accepts either a transient ``sentiment_score`` column or normalized raw
    ``text`` records. Scores are unweighted. Stable ``post_id`` values are
    deduplicated before aggregation when the column is present.
    """
    columns = ["timestamp_utc", *SENTIMENT_FEATURES]
    if posts.empty:
        result = pd.DataFrame(columns=columns)
        result["timestamp_utc"] = pd.Series(dtype="datetime64[ns, UTC]")
        result["post_count"] = pd.Series(dtype="int64")
        return result
    if "created_at_utc" not in posts:
        raise ValueError("posts must contain created_at_utc")
    if not 0 <= positive_threshold < 1:
        raise ValueError("positive_threshold must be in [0, 1)")

    frame = posts.copy()
    frame["timestamp_utc"] = _utc_days(frame["created_at_utc"], "created_at_utc")
    if "post_id" in frame:
        if frame["post_id"].isna().any():
            raise ValueError("post_id must be present to deduplicate normalized posts")
        frame = frame.drop_duplicates(subset="post_id", keep="first")
    else:
        raise ValueError("posts must contain post_id")

    if "sentiment_score" in frame:
        scores = pd.to_numeric(frame["sentiment_score"], errors="coerce")
        if scores.isna().any() or not np.isfinite(scores.to_numpy()).all():
            raise ValueError("sentiment_score contains a missing or non-finite value")
        if ((scores < -1) | (scores > 1)).any():
            raise ValueError("sentiment scores must be in [-1, 1]")
        frame["_score"] = scores.astype("float64")
    elif "text" in frame:
        frame["_score"] = score_texts(frame["text"].tolist(), scorer=scorer).to_numpy()
    else:
        raise ValueError("posts must contain text or sentiment_score")

    grouped = frame.groupby("timestamp_utc", sort=True)["_score"]
    result = grouped.agg(
        post_count="size",
        mean_sentiment="mean",
        sentiment_dispersion="std",
    )
    result["sentiment_dispersion"] = result["sentiment_dispersion"].fillna(0.0)
    result["positive_share"] = grouped.apply(
        lambda values: float((values > positive_threshold).mean())
    )
    result["negative_share"] = grouped.apply(
        lambda values: float((values < -positive_threshold).mean())
    )
    result["has_posts"] = 1.0
    result["post_count"] = result["post_count"].astype("int64")
    return result.reset_index()[columns]


def build_daily_features(
    prices: pd.DataFrame,
    posts: pd.DataFrame,
    *,
    scorer: Callable[[str], float] | None = None,
    price_lags: Sequence[int] = PRICE_LAGS,
    volatility_windows: Sequence[int] = VOLATILITY_WINDOWS,
    sma_windows: Sequence[int] = SMA_WINDOWS,
) -> pd.DataFrame:
    """Join daily post sentiment with completed daily close-derived features.

    The return index is a UTC-midnight ``timestamp_utc`` column. Missing post
    days retain a zero count and ``has_posts=0`` while sentiment statistics
    remain NaN for fold-local imputation. Inputs are not mutated.
    """
    required = {"timestamp_utc", "close"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"prices is missing required columns: {sorted(missing)}")
    if prices.empty:
        raise ValueError("prices must contain at least one completed daily bar")

    kept_price_columns = ["timestamp_utc", "close"]
    if "open" in prices:
        kept_price_columns.append("open")
    frame = prices[kept_price_columns].copy()
    frame["timestamp_utc"] = _utc_days(frame["timestamp_utc"], "price timestamp_utc")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    if frame["close"].isna().any() or (frame["close"] <= 0).any():
        raise ValueError("prices.close must contain positive finite values")
    if not np.isfinite(frame["close"].to_numpy()).all():
        raise ValueError("prices.close must contain positive finite values")
    if "open" in frame:
        frame["open"] = pd.to_numeric(frame["open"], errors="coerce")
        if frame["open"].isna().any() or (frame["open"] <= 0).any() or not np.isfinite(frame["open"].to_numpy()).all():
            raise ValueError("prices.open must contain positive finite values")
    frame = frame.sort_values("timestamp_utc").reset_index(drop=True)
    if frame["timestamp_utc"].duplicated().any():
        raise ValueError("prices must contain at most one bar per UTC day")
    if len(frame) > 1 and frame["timestamp_utc"].diff().dropna().ne(pd.Timedelta(days=1)).any():
        raise ValueError("daily BTC prices must have one consecutive bar for every UTC day")

    sentiment = aggregate_daily_sentiment(posts, scorer=scorer)
    daily = frame.merge(sentiment, on="timestamp_utc", how="left", validate="one_to_one")
    daily["post_count"] = daily["post_count"].fillna(0).astype("int64")
    daily["has_posts"] = daily["has_posts"].fillna(0.0)

    log_close = np.log(daily["close"])
    daily_log_return = log_close.diff()
    for lag in price_lags:
        if lag < 1:
            raise ValueError("price lags must be positive integers")
        daily[f"price_return_{lag}d"] = log_close.diff(lag)
    for window in volatility_windows:
        if window < 2:
            raise ValueError("volatility windows must be at least two days")
        daily[f"price_volatility_{window}d"] = daily_log_return.rolling(
            window=window, min_periods=window
        ).std(ddof=0)
    for window in sma_windows:
        if window < 1:
            raise ValueError("moving-average windows must be positive")
        trailing_mean = daily["close"].rolling(window=window, min_periods=window).mean()
        daily[f"price_vs_sma_{window}d"] = daily["close"] / trailing_mean - 1.0

    return daily.replace([np.inf, -np.inf], np.nan)


def build_supervised_frame(
    daily_features: pd.DataFrame, *, horizons: Sequence[int] = HORIZONS
) -> pd.DataFrame:
    """Add h-day log-return and target-date labels using future closes.

    Labels are deliberately separate from the feature columns and are only
    consumed by the walk-forward trainer. `n+0` is represented by origin close,
    not an estimated target.
    """
    required = {"timestamp_utc", "close"}
    missing = required - set(daily_features.columns)
    if missing:
        raise ValueError(f"daily_features is missing required columns: {sorted(missing)}")
    if daily_features["timestamp_utc"].duplicated().any():
        raise ValueError("daily_features must contain unique UTC days")
    frame = daily_features.sort_values("timestamp_utc").reset_index(drop=True).copy()
    if (frame["close"] <= 0).any() or not np.isfinite(frame["close"].to_numpy()).all():
        raise ValueError("close values must be positive and finite")
    for horizon in horizons:
        if horizon < 1:
            raise ValueError("forecast horizons must be positive integers")
        future_close = frame["close"].shift(-horizon)
        frame[f"target_log_return_h{horizon}"] = np.log(future_close / frame["close"])
        frame[f"target_close_h{horizon}"] = future_close
        frame[f"target_date_h{horizon}"] = frame["timestamp_utc"].shift(-horizon)
        if "open" in frame:
            frame[f"target_open_h{horizon}"] = frame["open"].shift(-horizon)
    return frame
