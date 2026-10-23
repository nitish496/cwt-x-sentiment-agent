# Quant methodology review

Role framing: senior quantitative research reviewer (30-years-equivalent standard of skepticism; this is a role brief, not a claim about a real biography).

## Forecast contract

Use UTC daily BTC-USD bars. Let `t` be the latest fully completed UTC day at the forecast cutoff. Return the latest observed close as `n+0` in the output table as the anchor; it is already known at the cutoff, so do not call it a forecast or include it in predictive performance metrics. For each origin `t`, train horizons `h=1..5` against `close[t+h]` (prefer log-return targets `log(close[t+h]/close[t])`, then reconstruct prices from the known origin close). If the employer expects n+0 to be a genuine predicted value, that conflicts with use of the current period's price/sentiment and must be clarified; a current close is either already known or unavailable until the period completes.

## Feature contract

All values must be available at the forecast cutoff. Candidate daily sentiment features: post count, mean polarity, positive/negative/neutral shares, and sentiment dispersion. Candidate price-only features: lagged log returns (1/3/7/14/30 days), realized volatility over trailing windows, and close relative to trailing moving averages. The model input must not contain raw text, ticker/news fields, timestamps as future cues, likes/reposts/views, follower counts, or news-derived variables unless the employer confirms that these are allowed under “sentiment and price only.” Engagement and followers can be retained in the raw ingestion table and shown as quality/context metadata, but omit them from the predictive feature matrix for strict compliance. Any sentiment weighting by engagement needs explicit documentation and a sensitivity comparison.

Aggregate only posts created before the forecast cutoff. Deduplicate by platform post ID. Keep zero-post days as missing-sentiment indicators plus zero volume; impute means using each training fold only. Store event time in UTC; if the assignment's EST means Eastern local time, render it for humans using an IANA timezone rather than fixed EST (DST).

## Validation protocol

1. Sort unique daily observations chronologically; never random-shuffle.
2. Reserve the final contiguous period as untouched test data. On earlier history, run expanding-window folds with fixed multiweek or monthly validation windows.
3. For each fold and horizon, train only on origins whose target date is before the validation start. Purge at least the final five training origins (or a horizon-specific `h` gap) to prevent overlapping 5-day targets from crossing the split.
4. Fit preprocessing, imputation, sentiment normalization, feature selection, and hyperparameter tuning using training data only. If tuning is needed, use inner chronological folds.
5. Generate and save predictions once for each out-of-sample origin/horizon. Report fold-by-fold and pooled results; do not evaluate in-sample fits as performance.
6. Compare with a random-walk baseline (future close = origin close / zero future return) at every horizon; optionally add a train-only historical-drift baseline. Prefer a simple regularized linear model before a tree-based challenger. Only keep a complex model if it beats baseline consistently across folds, not only pooled.

Report MAE/RMSE of reconstructed prices and log returns by horizon, directional accuracy/balanced accuracy, sample counts, date coverage, and confidence intervals across origins/folds. Price-level errors alone can look small for high-price assets and should not stand alone.

## Risk metrics and backtest mapping

Sharpe, Sortino, and maximum drawdown require a stated portfolio rule; they are not prediction metrics. Predeclare one one-day strategy from the `h=1` out-of-sample forecast, for example a bounded long/flat or long/short position when predicted next-day log return exceeds a threshold chosen inside training folds. Enter only after the signal is observable; use the next executable bar/price. Include turnover, fees, and slippage. Since these assumptions are not supplied, show a sensitivity grid (e.g. 5/10/25 bps per round trip) rather than choosing one cost as fact. Calculate net daily equity returns only on the untouched walk-forward predictions, use `sqrt(365)` for daily crypto annualization, calculate Sortino against a stated zero or cash hurdle, and compute max drawdown from the compounded net equity curve. Show buy-and-hold and cash baselines; state whether shorts/leverage are allowed. If no credible signal rule beats costs, report that honestly and still provide predictive metrics.

Do not infer a profitable strategy from the 5-day overlapping forecasts. If every horizon gets a separate trading backtest, define its execution/holding/rebalancing rule separately and account for overlapping positions.

## Data and evidence constraints

The linked dataset catalog is not the dataset itself. It includes huge sources and a StockTwits corpus (not X), so source names, platform, license, size, date range, schema, and actual downloaded coverage must be checked before any result is claimed. Do not combine old tweets with 2026 market data as if dates overlap. A model trained on historical X data ending in 2023 may be regime-stale; evaluate with an explicitly disclosed temporal gap and report it as a limitation. The live 180-day Apify collection can support a recent descriptive outlook, but by itself is too short for a robust daily five-horizon validation.

Recent news can be used in the current one-page outlook. Historical news is not part of the allowed feature set as written, and current news must never be backfilled into past forecast rows. Label dataset rows as live, historical, synthetic fixture, or unavailable; never mix fixture data into headline performance metrics.

## Ambiguities to resolve or state as defaults

- The meaning of `n+0`: known close anchor vs actual nowcast.
- Daily versus hourly `period`; BTC-USD/daily is a practical provisional choice, not stated in the assignment.
- “Sentiment and price only” versus required scraping of engagement/followers and ingestion of news.
- Price source/venue, bar cutoff convention, and what “last 180 days” means at submission time.
- Which historical corpus the employer supplied/intended; the linked catalog is a list, not data.
- The email request for raw APIFY/Tavily/Exa tokens is unsafe for reusable secrets. Supply actor IDs and reproducibility instructions; do not put credentials in Git or reports. Ask for a secure secret-sharing channel if they insist.
- Assignment timing conflicts: opening says within one week; timeline says 5–7 days. The user said they received it on 2026-10-22; confirm the exact due date with the owner.
