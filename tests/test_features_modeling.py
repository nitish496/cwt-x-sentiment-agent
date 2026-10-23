"""Synthetic unit coverage only; these values are not empirical performance."""

import numpy as np
import pandas as pd
import pytest

from cwt_agent.features import (
    HORIZONS,
    aggregate_daily_sentiment,
    build_daily_features,
    build_supervised_frame,
)
from cwt_agent.modeling import (
    expanding_walk_forward_predictions,
    forecast_metrics,
    forecast_next_horizons,
    one_day_long_flat_risk_metrics,
)


def synthetic_prices(rows: int = 110) -> pd.DataFrame:
    days = pd.date_range("2024-01-01", periods=rows, freq="D", tz="UTC")
    returns = 0.001 + 0.012 * np.sin(np.arange(rows) / 4.0)
    close = 40_000 * np.exp(np.cumsum(returns))
    open_prices = np.r_[40_000.0, close[:-1]]
    return pd.DataFrame({"timestamp_utc": days, "open": open_prices, "close": close})


def synthetic_posts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "post_id": ["a", "a", "b", "c"],
            "created_at_utc": [
                "2024-01-01T23:59:00Z",
                "2024-01-01T23:59:00Z",
                "2024-01-02T00:00:00Z",
                "2024-01-04T12:00:00Z",
            ],
            "sentiment_score": [0.8, -0.8, -0.4, 0.0],
        }
    )


def test_daily_sentiment_deduplicates_and_respects_utc_boundaries():
    daily = aggregate_daily_sentiment(synthetic_posts())

    jan1 = daily.loc[daily.timestamp_utc == pd.Timestamp("2024-01-01", tz="UTC")].iloc[0]
    jan2 = daily.loc[daily.timestamp_utc == pd.Timestamp("2024-01-02", tz="UTC")].iloc[0]
    assert jan1.post_count == 1
    assert jan1.mean_sentiment == pytest.approx(0.8)
    assert jan2.post_count == 1
    assert jan2.negative_share == pytest.approx(1.0)


def test_daily_features_keep_empty_days_and_use_only_current_or_prior_prices():
    prices = synthetic_prices(40)
    features = build_daily_features(prices, synthetic_posts(), price_lags=(1, 3), volatility_windows=(3,), sma_windows=(3,))

    jan3 = features.loc[features.timestamp_utc == pd.Timestamp("2024-01-03", tz="UTC")].iloc[0]
    assert jan3.post_count == 0
    assert jan3.has_posts == 0
    assert np.isnan(jan3.mean_sentiment)

    changed_future = prices.copy()
    changed_future.loc[10, "close"] *= 2
    changed_features = build_daily_features(
        changed_future,
        synthetic_posts(),
        price_lags=(1, 3),
        volatility_windows=(3,),
        sma_windows=(3,),
    )
    before_future_change = features.loc[features.timestamp_utc < prices.loc[10, "timestamp_utc"]]
    before_future_changed = changed_features.loc[
        changed_features.timestamp_utc < prices.loc[10, "timestamp_utc"]
    ]
    pd.testing.assert_frame_equal(
        before_future_change.reset_index(drop=True),
        before_future_changed.reset_index(drop=True),
    )


def test_daily_prices_require_utc_aware_timestamps_and_consecutive_days():
    prices = synthetic_prices(3)
    prices["timestamp_utc"] = prices["timestamp_utc"].dt.tz_localize(None)
    with pytest.raises(ValueError, match="timezone-aware"):
        build_daily_features(prices, pd.DataFrame())

    prices = synthetic_prices(3).drop(index=1).reset_index(drop=True)
    with pytest.raises(ValueError, match="consecutive bar"):
        build_daily_features(prices, pd.DataFrame())


def test_supervised_labels_are_forward_log_returns_and_end_at_five_day_horizon():
    features = build_daily_features(
        synthetic_prices(12),
        pd.DataFrame(columns=["created_at_utc", "sentiment_score"]),
        price_lags=(1,),
        volatility_windows=(),
        sma_windows=(),
    )
    supervised = build_supervised_frame(features)
    row = supervised.iloc[2]
    expected = np.log(supervised.iloc[3].close / row.close)
    assert row.target_log_return_h1 == pytest.approx(expected)
    assert row.target_date_h1 == supervised.iloc[3].timestamp_utc
    assert row.target_close_h5 == supervised.iloc[7].close
    assert supervised.iloc[-1][[f"target_log_return_h{h}" for h in HORIZONS]].isna().all()


def test_walk_forward_predictions_are_chronological_purged_and_include_random_walk():
    features = build_daily_features(
        synthetic_prices(),
        pd.DataFrame(columns=["created_at_utc", "sentiment_score"]),
        price_lags=(1, 3, 7),
        volatility_windows=(7,),
        sma_windows=(7,),
    )
    supervised = build_supervised_frame(features)
    predictions = expanding_walk_forward_predictions(
        supervised,
        n_splits=3,
        test_size=10,
        min_train_size=40,
        purge_bars=5,
    )

    assert len(predictions) > 0
    assert set(predictions.horizon_days) == set(HORIZONS)
    assert (predictions.train_end_utc < predictions.test_start_utc).all()
    assert (predictions.train_rows >= 40).all()
    assert (predictions.random_walk_log_return == 0).all()
    assert predictions.next_open.notna().all()
    assert np.allclose(predictions.random_walk_close, predictions.origin_close)
    assert (predictions.target_timestamp_utc > predictions.origin_timestamp_utc).all()

    metrics = forecast_metrics(predictions)
    assert set(metrics.model_name) == {"ridge", "random_walk"}
    assert set(metrics.horizon_days) == set(HORIZONS)
    assert (metrics.n_predictions > 0).all()
    assert np.isfinite(metrics.mae_price).all()


def test_walk_forward_rejects_target_columns_as_features():
    features = build_daily_features(
        synthetic_prices(),
        pd.DataFrame(columns=["created_at_utc", "sentiment_score"]),
        price_lags=(1,),
        volatility_windows=(),
        sma_windows=(),
    )
    supervised = build_supervised_frame(features)
    with pytest.raises(ValueError, match="target columns cannot be model features"):
        expanding_walk_forward_predictions(
            supervised,
            n_splits=2,
            test_size=10,
            min_train_size=30,
            feature_columns=["target_close_h1"],
        )


def test_current_forecasts_use_only_fully_observed_labels_and_anchor_n0():
    prices = synthetic_prices(130)
    features = build_daily_features(
        prices,
        pd.DataFrame(columns=["created_at_utc", "sentiment_score"]),
        price_lags=(1, 3),
        volatility_windows=(3,),
        sma_windows=(3,),
    )
    supervised = build_supervised_frame(features)
    forecasts = forecast_next_horizons(supervised, min_train_size=50)
    assert forecasts.horizon_days.tolist() == list(HORIZONS)
    latest = prices.iloc[-1]
    assert (forecasts.origin_timestamp_utc == latest.timestamp_utc).all()
    assert np.allclose(forecasts.origin_close, latest.close)
    assert (forecasts.target_timestamp_utc > forecasts.origin_timestamp_utc).all()
    assert forecasts.train_rows.tolist() == [len(prices) - horizon for horizon in HORIZONS]


def test_long_flat_risk_metrics_apply_costs_and_are_defined_for_zero_risk():
    predictions = pd.DataFrame(
        {
            "origin_timestamp_utc": pd.date_range("2024-01-01", periods=3, tz="UTC"),
            "horizon_days": [1, 1, 1],
            "actual_log_return": np.log1p([0.10, -0.20, 0.05]),
            "predicted_log_return": [0.02, 0.03, 0.01],
            "origin_close": [100.0, 100.0, 100.0],
            "next_open": [100.0, 100.0, 100.0],
            "actual_close": [110.0, 80.0, 105.0],
        }
    )
    metrics = one_day_long_flat_risk_metrics(predictions, round_trip_cost_bps=10)
    assert metrics["n_days"] == 3
    assert metrics["active_days"] == 3
    assert metrics["round_trips"] == pytest.approx(1.0)
    assert metrics["total_cost_fraction"] == pytest.approx(0.001)
    assert metrics["execution_assumption"] == "next_open_to_target_close"
    assert metrics["sharpe"] is not None
    assert metrics["max_drawdown"] < 0

    flat = predictions.assign(predicted_log_return=-1.0)
    flat_metrics = one_day_long_flat_risk_metrics(flat)
    assert flat_metrics["active_days"] == 0
    assert flat_metrics["sharpe"] is None
    assert flat_metrics["sortino"] is None
    assert flat_metrics["max_drawdown"] == 0.0
