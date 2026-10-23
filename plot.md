# Project Plot — CWT X Sentiment Price Prediction

## Assignment source
Google Doc: `CWT X sentiment Agent - intern task` (read 2026-10-22). It asks for a Python pipeline for an asset's price forecasts at n+0 through n+5 using X sentiment and price, Apify-based collection, recent asset news (Tavily named; Exa listed in submission credentials), timestamp alignment, feature engineering, a one-page sentiment outlook, walk-forward validation, sentiment/price visualization, and an evaluation report with Sharpe, Sortino, and maximum drawdown. Submission requested: repository link, API tokens and actor IDs, report via email. The user said they received it on 2026-10-22; the Internshala notification states the due date is 2026-10-23, while the assignment document also gives broad “within 1 week” and “5–7 days” language.

Linked historical-data catalog: Google Doc `Historical Twitter/X Datasets — Verified Download Links` (read 2026-10-22). It describes possible datasets but contains a catalog, not attached tweet data files. It notes license/access differences and very large corpora. Dataset choice must be verified before download/use.

## Confirmed user defaults
- Asset: BTC-USD.
- Forecast sampling: daily bars; n+0 means the latest completed daily bar, and n+1..n+5 mean the next five completed daily bars.
- Storage: local SQLite plus reproducible CSV/Parquet exports where practical.
- Canonical timestamps: UTC; present Eastern time only for human-readable summaries, with DST unambiguous.
- No raw credentials in source, Git, reports, ZIPs, or chat. Load credentials from ignored `.env`; test offline with fixtures.
- Only public/authorized data, documented source and license. Do not represent historical offline data as fresh 180-day X coverage.
- No investment advice; results are an educational research artifact, not a trading recommendation.

## Required workflow / loop
1. Inspect workspace and source data; verify license, schema, date coverage, size, and source.
2. Define ingestion and timestamp contracts before integration.
3. Build one vertical slice from raw fixture/source through aligned features to a prediction and chart.
4. Run automated tests, leakage audit, and chronological walk-forward validation.
5. Have independent agents review sources, methodology, reproducibility, and security; root integrates only verified recommendations.
6. Fix, rerun tests/evaluation, regenerate report, and update this plot and `WORK_LOG.md` at each loop.
7. Package repository/report after privacy/license audit. Do not send email or share secrets without explicit authorization.

## Folder map
- `src/cwt_agent/`: pipeline implementation.
- `tests/`: offline, deterministic tests.
- `data/raw/`, `data/processed/`: local data outputs; never commit user credentials or restricted data.
- `reports/`: sentiment snapshot and evaluation report.
- `research/`: bounded independent agent memos.
- `.env.example`: names and format only; `.env` ignored.

## Decision gates and risks
- Historical dataset is still unselected; the linked document is a catalog rather than an attached file. Never claim validation until a suitable dataset is supplied and checked.
- No Apify, Tavily, or Exa credentials supplied. The assignment's linked dataset catalog contains metadata and download links, not actual tweet data. The user has authorized project work, but current X policy/terms and third-party corpus rights still require an approved acquisition method; do not scrape or use an unverified corpus until CWT confirms permission and the source is validated.
- The linked historical catalog includes large data (up to tens of GB), and one listed source is StockTwits, not X. Do not imply it is X. Select a small, licensed, relevant dataset only after source verification.
- Performance metrics require a deterministic signal-to-position/backtest rule, transaction-cost assumption, and out-of-sample predictions. Never invent or report metrics from in-sample fits.
- Current news can inform the sentiment outlook but must not leak into historical prediction features unless timestamped and available before each forecast cutoff.
- Employer requests raw API tokens by email. Treat tokens as secrets; prepare actor IDs and reproducible configuration instructions, but do not transmit raw tokens without the user's explicit authorization and a secure delivery decision.
- Repo creation/public push and external email are separate actions; no push or submission is presumed authorized by implementation alone.

## Status
Implementation and QA loops: core command flow, provider adapters, import, features, historical walk-forward evaluation, current n+1..n+5 forecasts, local persistence, and outlook/report generation are implemented. The Python 3.12 unit suite passes 26/26 and Ruff passes. Synthetic-only smoke tests generated forecast, outlook, evaluation, and both SVG charts under `/tmp`; no synthetic numbers were included in deliverables. A clean editable install in a fresh Python 3.12 virtual environment now succeeds. No live API credentials or approved historical tweet file have been provided; therefore current scraping and empirical metrics remain unavailable. GitHub publishing is pending a valid local CLI login; the Codex GitHub connector is authenticated but cannot create a new repository.

## Data contracts
- Posts: `post_id`, `created_at_utc`, `text`, nullable `likes`, `reposts`, `views`, `followers`, `provider`, `source_id`.
- Prices: `timestamp_utc`, OHLCV, `symbol`, `venue`, `quote_currency`.
- News: `title`, `url`, `published_at_utc`, `snippet`, `provider`, `retrieved_at_utc`.
- Daily modeling row: completed UTC daily price bar plus sentiment aggregates computable as of the close; current news excluded from historical model features.
- Model: direct h=1..5 log-return estimates, converted to price forecasts from each origin close; n+0 is a known close anchor, not an evaluated prediction.
- Offline sample fixtures are synthetic and used only for software behavior tests, never headline performance results.
