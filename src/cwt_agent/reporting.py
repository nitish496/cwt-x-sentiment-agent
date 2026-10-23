"""Human-readable outlook and evaluation artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import pandas as pd


def write_outlook(
    output: str | Path,
    *,
    symbol: str,
    prices: pd.DataFrame,
    posts: pd.DataFrame,
    news: pd.DataFrame,
    forecasts: pd.DataFrame | None = None,
    data_label: str = "live",
) -> Path:
    """Write a one-page current outlook; explicitly mark missing data."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"# {symbol} sentiment and price outlook", "", f"As of: {now}", f"Data label: **{data_label}**", ""]

    if prices.empty:
        lines += ["## Price snapshot", "Unavailable: no price candles were provided.", ""]
    else:
        ordered = prices.sort_values("timestamp_utc")
        latest = ordered.iloc[-1]
        lines += [
            "## Price snapshot",
            f"Latest available completed close: **{float(latest['close']):,.2f} {symbol.split('-')[-1]}** ",
            f"for {pd.Timestamp(latest['timestamp_utc']).isoformat()}.",
            f"Source: {latest.get('source', 'unspecified')} / {latest.get('venue', 'unspecified')}.",
            "",
        ]

    if posts.empty or "text" not in posts:
        lines += ["## X sentiment", "Unavailable: no X posts were supplied. No sentiment claim is made.", ""]
    else:
        from .features import aggregate_daily_sentiment

        daily = aggregate_daily_sentiment(posts)
        latest_day = daily.iloc[-1]
        lines += [
            "## X sentiment",
            (
                f"Latest represented UTC day: {pd.Timestamp(latest_day['timestamp_utc']).date()}; "
                f"{int(latest_day['post_count'])} posts; mean polarity {latest_day['mean_sentiment']:.3f} "
                "(unweighted VADER polarity, descriptive only)."
            ),
            "This small-window sentiment is not a standalone trading signal.",
            "",
        ]

    if news.empty:
        lines += ["## Current news context", "Unavailable: no current-news results were supplied.", ""]
    else:
        lines += ["## Current news context"]
        for _, item in news.head(5).iterrows():
            title = str(item.get("title", "Untitled")).replace("\n", " ")
            url = str(item.get("url", ""))
            provider = str(item.get("provider", "source not recorded"))
            lines.append(f"- [{title}]({url}) ({provider})")
        lines.append("")

    lines += ["## Forecast", "n+0 is the latest completed close and is a known anchor, not a prediction.", ""]
    if forecasts is None or forecasts.empty:
        lines += ["n+1 through n+5: **unavailable** (no forecast run supplied).", ""]
    else:
        horizon_column = "horizon_days" if "horizon_days" in forecasts else "horizon"
        lines += ["| Horizon | Predicted close |", "|---|---:|"]
        for _, row in forecasts.sort_values(horizon_column).iterrows():
            lines.append(f"| n+{int(row[horizon_column])} | {float(row['predicted_close']):,.2f} |")
        lines.append("")
    lines += [
        "## Limitations",
        (
            "This is a research summary, not financial advice. Forecasts are estimates, not guaranteed outcomes. "
            "A live outlook requires current X and news data; unavailable sections above are intentionally left unfilled."
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_evaluation(
    output_dir: str | Path,
    predictions: pd.DataFrame,
    *,
    source_label: str,
    coverage: dict[str, Any] | None = None,
    daily_features: pd.DataFrame | None = None,
) -> tuple[Path, Path]:
    """Save walk-forward metrics and a prediction-vs-actual visualization."""
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    report_path = folder / "evaluation.md"
    chart_path = folder / "walk_forward_predictions.svg"
    lines = ["# Walk-forward evaluation", "", f"Data label: **{source_label}**", ""]
    if coverage:
        lines += ["## Data coverage", ""]
        for key, value in coverage.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    if predictions.empty:
        lines += ["No out-of-sample predictions were generated.", ""]
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path, chart_path

    from .modeling import forecast_metrics, one_day_long_flat_risk_metrics

    all_metrics = forecast_metrics(predictions)
    metrics = all_metrics.loc[all_metrics["model_name"] == "ridge"].copy()
    baseline = all_metrics.loc[all_metrics["model_name"] == "random_walk"].set_index("horizon_days")
    lines += ["## Out-of-sample metrics", "", "Metrics use walk-forward predictions only. Synthetic data is not empirical evidence.", ""]
    lines += ["| Horizon | Model MAE (return) | Model RMSE (return) | Baseline MAE | Price MAE | Directional accuracy | N |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for _, row in metrics.iterrows():
        base = baseline.loc[row["horizon_days"]]
        lines.append(
            f"| n+{int(row['horizon_days'])} | {row['mae_log_return']:.6f} | {row['rmse_log_return']:.6f} "
            f"| {base['mae_log_return']:.6f} | {row['mae_price']:.2f} "
            f"| {row['directional_accuracy']:.3f} | {int(row['n_predictions'])} |"
        )
    lines += ["", "## Risk metrics", "", "Illustrative 1-day long/flat backtest from out-of-sample n+1 predictions. Position is 100% long when predicted log return > 0, otherwise cash; no leverage or shorting. This simple threshold is not tuned and is not a trading recommendation. Round-trip costs are applied at 5/10/25 bps; signals are evaluated only on non-overlapping h=1 origins.", ""]
    lines += ["| Round-trip cost (bps) | Sharpe (sqrt 365) | Sortino (zero hurdle) | Maximum drawdown | Total return | Active days |", "|---:|---:|---:|---:|---:|---:|"]
    for cost in (5.0, 10.0, 25.0):
        risk = one_day_long_flat_risk_metrics(predictions, round_trip_cost_bps=cost, annualization=365)
        fmt = lambda value: "N/A" if value is None else f"{value:.4f}"
        lines.append(
            f"| {cost:.0f} | {fmt(risk['sharpe'])} | {fmt(risk['sortino'])} "
            f"| {fmt(risk['max_drawdown'])} | {fmt(risk['total_return'])} | {risk['active_days']} |"
        )
    lines += ["", "Cash baseline return is 0% by construction. Buy-and-hold comparison and a separate untouched final holdout remain follow-up items before presenting this as a final empirical report.", ""]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    chart_path.write_text(_prediction_chart_svg(predictions), encoding="utf-8")
    if daily_features is not None:
        sentiment_chart = folder / "sentiment_vs_price.svg"
        sentiment_chart.write_text(_sentiment_price_chart_svg(daily_features), encoding="utf-8")
        with report_path.open("a", encoding="utf-8") as report:
            report.write("\n## Charts\n\n")
            report.write("- [Out-of-sample forecasts vs actuals](walk_forward_predictions.svg)\n")
            report.write("- [Daily X sentiment vs indexed BTC close](sentiment_vs_price.svg)\n")
    return report_path, chart_path


def _prediction_chart_svg(predictions: pd.DataFrame) -> str:
    """Build a dependency-free, labeled five-panel forecast comparison chart."""
    width, panel_height, top, bottom = 960, 190, 48, 22
    height = top + panel_height * 5 + bottom
    left, right = 76, 24
    plot_width = width - left - right
    colors = {
        "actual_close": "#202020",
        "predicted_close": "#b6654a",
        "random_walk_close": "#737373",
    }
    labels = {
        "actual_close": "Actual",
        "predicted_close": "Ridge",
        "random_walk_close": "Random walk",
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="title desc" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<title id="title">Walk-forward forecast compared with actual close</title>',
        '<desc id="desc">Five panels show out-of-sample actual, Ridge, and random-walk close values by forecast horizon.</desc>',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="24" y="28" font-family="sans-serif" font-size="17" font-weight="bold">Walk-forward forecast vs actual close</text>',
    ]
    for index, (column, label) in enumerate(labels.items()):
        x = 290 + index * 170
        parts += [
            f'<line x1="{x}" y1="26" x2="{x+24}" y2="26" stroke="{colors[column]}" stroke-width="3"/>',
            f'<text x="{x+30}" y="30" font-family="sans-serif" font-size="12">{label}</text>',
        ]
    horizons = sorted(int(value) for value in predictions["horizon_days"].unique())
    for panel, horizon in enumerate(horizons[:5]):
        part = predictions.loc[predictions["horizon_days"] == horizon].sort_values("origin_timestamp_utc")
        y_top = top + panel * panel_height
        y_bottom = y_top + panel_height - bottom
        values = [float(value) for column in labels for value in part[column].dropna()]
        low, high = min(values), max(values)
        if high <= low:
            padding = max(abs(high) * 0.01, 1.0)
            low, high = low - padding, high + padding
        else:
            padding = (high - low) * 0.08
            low, high = low - padding, high + padding
        count = max(len(part) - 1, 1)
        parts += [
            f'<line x1="{left}" y1="{y_bottom}" x2="{width-right}" y2="{y_bottom}" stroke="#c8c8c8"/>',
            f'<text x="18" y="{y_top+18}" font-family="sans-serif" font-size="13" font-weight="bold">n+{horizon}</text>',
            f'<text x="{left-8}" y="{y_top+35}" text-anchor="end" font-family="sans-serif" font-size="10">{high:,.0f}</text>',
            f'<text x="{left-8}" y="{y_bottom}" text-anchor="end" font-family="sans-serif" font-size="10">{low:,.0f}</text>',
        ]
        for column, color in colors.items():
            points = []
            for point_index, value in enumerate(part[column].astype(float)):
                x = left + plot_width * point_index / count
                y = y_bottom - (value - low) / (high - low) * (y_bottom - y_top - 40)
                points.append(f"{x:.1f},{y:.1f}")
            parts.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="1.6" points="{" ".join(points)}"/>'
            )
        if not part.empty:
            start = escape(pd.Timestamp(part["origin_timestamp_utc"].iloc[0]).strftime("%Y-%m-%d"))
            end = escape(pd.Timestamp(part["origin_timestamp_utc"].iloc[-1]).strftime("%Y-%m-%d"))
            parts += [
                f'<text x="{left}" y="{y_bottom+16}" font-family="sans-serif" font-size="10">{start}</text>',
                f'<text x="{width-right}" y="{y_bottom+16}" text-anchor="end" font-family="sans-serif" font-size="10">{end}</text>',
            ]
    parts.append("</svg>")
    return "\n".join(parts)


def _sentiment_price_chart_svg(daily: pd.DataFrame) -> str:
    """Compare daily sentiment and normalized close while preserving no-post gaps."""
    if not {"timestamp_utc", "close", "mean_sentiment"}.issubset(daily.columns):
        raise ValueError("daily features need timestamp_utc, close, and mean_sentiment")
    frame = daily.sort_values("timestamp_utc").reset_index(drop=True)
    if frame.empty:
        raise ValueError("daily features cannot be empty")
    prices = pd.to_numeric(frame["close"], errors="raise").astype(float)
    if (prices <= 0).any() or not prices.map(pd.notna).all():
        raise ValueError("daily close values must be finite and positive")
    index_values = prices / prices.iloc[0] * 100.0
    index_low, index_high = float(index_values.min()), float(index_values.max())
    if index_high == index_low:
        index_low -= 1.0
        index_high += 1.0
    sentiment = pd.to_numeric(frame["mean_sentiment"], errors="coerce")
    width, height = 1000, 430
    left, right, top, bottom = 82, 82, 76, 58
    plot_width, plot_height = width - left - right, height - top - bottom
    count = max(len(frame) - 1, 1)

    def x_position(index: int) -> float:
        return left + plot_width * index / count

    def price_y(value: float) -> float:
        return top + plot_height - (value - index_low) / (index_high - index_low) * plot_height

    def sentiment_y(value: float) -> float:
        return top + plot_height - (value + 1.0) / 2.0 * plot_height

    price_points = " ".join(
        f"{x_position(i):.1f},{price_y(value):.1f}" for i, value in enumerate(index_values)
    )
    sentiment_segments: list[str] = []
    for i in range(len(frame) - 1):
        first, second = sentiment.iloc[i], sentiment.iloc[i + 1]
        if pd.notna(first) and pd.notna(second):
            sentiment_segments.append(
                f'<line x1="{x_position(i):.1f}" y1="{sentiment_y(float(first)):.1f}" '
                f'x2="{x_position(i+1):.1f}" y2="{sentiment_y(float(second)):.1f}" '
                'stroke="#b6654a" stroke-width="2"/>'
            )
    start = escape(pd.Timestamp(frame["timestamp_utc"].iloc[0]).strftime("%Y-%m-%d"))
    end = escape(pd.Timestamp(frame["timestamp_utc"].iloc[-1]).strftime("%Y-%m-%d"))
    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="title desc" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<title id="title">Daily X sentiment and BTC close</title>',
        '<desc id="desc">BTC close indexed to 100 at the start of the period and daily unweighted X mean sentiment. Gaps mark days without posts.</desc>',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="24" y="30" font-family="sans-serif" font-size="18" font-weight="bold">Daily X sentiment vs BTC price</text>',
        f'<line x1="{left}" y1="{top+plot_height}" x2="{width-right}" y2="{top+plot_height}" stroke="#bbb"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}" stroke="#bbb"/>',
        f'<line x1="{width-right}" y1="{top}" x2="{width-right}" y2="{top+plot_height}" stroke="#bbb"/>',
        f'<polyline fill="none" stroke="#202020" stroke-width="2" points="{price_points}"/>',
        *sentiment_segments,
        f'<text x="{left}" y="{height-18}" font-family="sans-serif" font-size="11">{start}</text>',
        f'<text x="{width-right}" y="{height-18}" text-anchor="end" font-family="sans-serif" font-size="11">{end}</text>',
        f'<text x="{left}" y="{top-10}" font-family="sans-serif" font-size="11">BTC close index (start = 100)</text>',
        f'<text x="{width-right}" y="{top-10}" text-anchor="end" font-family="sans-serif" font-size="11">Mean sentiment: -1 to +1</text>',
        f'<text x="{left-8}" y="{top+12}" text-anchor="end" font-family="sans-serif" font-size="10">{index_high:.1f}</text>',
        f'<text x="{left-8}" y="{top+plot_height}" text-anchor="end" font-family="sans-serif" font-size="10">{index_low:.1f}</text>',
        f'<text x="{width-right+8}" y="{top+12}" font-family="sans-serif" font-size="10">+1</text>',
        f'<text x="{width-right+8}" y="{top+plot_height}" font-family="sans-serif" font-size="10">-1</text>',
        '<line x1="355" y1="52" x2="379" y2="52" stroke="#202020" stroke-width="2"/>',
        '<text x="385" y="56" font-family="sans-serif" font-size="11">BTC close index</text>',
        '<line x1="535" y1="52" x2="559" y2="52" stroke="#b6654a" stroke-width="2"/>',
        '<text x="565" y="56" font-family="sans-serif" font-size="11">X mean sentiment</text>',
        "</svg>",
    ])
