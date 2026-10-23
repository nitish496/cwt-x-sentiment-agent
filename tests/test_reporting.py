import pandas as pd

from cwt_agent.reporting import (
    _prediction_chart_svg,
    _sentiment_price_chart_svg,
    write_evaluation,
    write_outlook,
)


def test_dependency_free_chart_svg_has_five_horizons():
    frame = pd.DataFrame(
        [
            {
                "horizon_days": horizon,
                "origin_timestamp_utc": timestamp,
                "actual_close": 100.0 + horizon + index,
                "predicted_close": 99.0 + horizon + index,
                "random_walk_close": 98.0 + horizon + index,
            }
            for horizon in range(1, 6)
            for index, timestamp in enumerate(pd.date_range("2026-01-01", periods=2, tz="UTC"))
        ]
    )
    svg = _prediction_chart_svg(frame)
    assert svg.startswith('<svg xmlns="http://www.w3.org/2000/svg"')
    assert all(f">n+{horizon}<" in svg for horizon in range(1, 6))
    assert "Random walk" in svg and "Ridge" in svg and "Actual" in svg


def test_outlook_shows_forecasts_and_marks_missing_live_context(tmp_path):
    prices = pd.DataFrame(
        [{"timestamp_utc": "2026-01-01T00:00:00Z", "close": 100, "source": "fixture", "venue": "fixture"}]
    )
    forecasts = pd.DataFrame(
        [{"horizon_days": horizon, "predicted_close": 100 + horizon} for horizon in range(1, 6)]
    )
    path = write_outlook(
        tmp_path / "outlook.md",
        symbol="BTC-USD",
        prices=prices,
        posts=pd.DataFrame(),
        news=pd.DataFrame(),
        forecasts=forecasts,
        data_label="synthetic fixture",
    )
    text = path.read_text()
    assert "Data label: **synthetic fixture**" in text
    assert "Unavailable: no X posts" in text
    assert "Unavailable: no current-news" in text
    assert all(f"| n+{horizon} |" in text for horizon in range(1, 6))


def test_sentiment_price_chart_is_labeled_and_leaves_missing_day_gaps():
    daily = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2026-01-01", periods=4, tz="UTC"),
            "close": [100.0, 102.0, 99.0, 105.0],
            "mean_sentiment": [-0.2, float("nan"), 0.5, 0.1],
        }
    )
    svg = _sentiment_price_chart_svg(daily)
    assert "Daily X sentiment vs BTC price" in svg
    assert "BTC close index (start = 100)" in svg
    assert "Mean sentiment: -1 to +1" in svg
    # A missing sentiment day creates no connecting segment across that day.
    assert svg.count('stroke="#b6654a"') == 2  # legend plus the single adjacent-day segment


def test_evaluation_writes_both_charts_and_links_them(tmp_path):
    predictions = pd.DataFrame(
        [
            {
                "horizon_days": horizon,
                "origin_timestamp_utc": pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=index),
                "actual_log_return": 0.01 * (index + 1),
                "predicted_log_return": 0.008 * (index + 1),
                "origin_close": 100.0,
                "actual_close": 100.0 + index + 1,
                "predicted_close": 100.0 + index + 0.8,
                "random_walk_close": 100.0,
            }
            for horizon in range(1, 6)
            for index in range(2)
        ]
    )
    daily = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2026-01-01", periods=3, tz="UTC"),
            "close": [100.0, 101.0, 99.0],
            "mean_sentiment": [-0.2, 0.4, 0.1],
        }
    )
    report, forecast_chart = write_evaluation(
        tmp_path, predictions, source_label="synthetic fixture", daily_features=daily
    )
    sentiment_chart = tmp_path / "sentiment_vs_price.svg"
    assert report.exists() and forecast_chart.exists() and sentiment_chart.exists()
    text = report.read_text()
    assert "walk_forward_predictions.svg" in text
    assert "sentiment_vs_price.svg" in text
