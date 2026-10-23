# CWT X Sentiment Price Prediction

Python research pipeline for the CrowdWisdomTrading internship assessment. The confirmed working setup is BTC-USD daily bars. `plot.md` records assignment requirements, decisions, and limits; `WORK_LOG.md` records verified progress.

## Current status

The project includes a local ingestion, feature, forecast, and reporting workflow. Live Apify/Tavily/Exa credentials and an approved historical X export are not available yet. The code supports an offline data path; synthetic fixture output is never empirical model performance. A clean editable install from the package index succeeded in Python 3.12; the suite currently contains 26 tests and Ruff checks are available.

## Setup

Requires Python 3.9 or newer. Network installs may be required for project dependencies.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
```

Keep `.env` local. Do not put tokens in Git, reports, screenshots, or assignment email. The assignment asks to email tokens; use dedicated revocable keys and an approved secure method if that request must be satisfied.

## Commands

Run from this project directory after installing dependencies:

```sh
python -m cwt_agent --help
python -m cwt_agent config-check
python -m cwt_agent prices --from 2026-04-25 --to 2026-10-22
python -m cwt_agent outlook
```

`prices` reads public Coinbase Exchange daily candles and requires network access. For a local CSV/JSONL historical X export, create a JSON column map from normalized names to the file's actual headers, then use `import-posts`; `reports/generated/import-manifest.json` records coverage, invalid rows, and the caller-supplied source/rights note. Permission and platform still need human verification.

Live X ingestion requires explicit `--live`, an `APIFY_API_TOKEN`, `APIFY_ACTOR_ID`, a reviewed Actor-specific JSON input file, and (when its output differs from normalized names) a reviewed `--field-map`. The provided `config/apify_actor_input.example.json` is only a template and is **not** guaranteed to match any Actor. Check its cost, permissions, schema, and date-filter behavior before a run. `news --live` requires Tavily; Exa is optional corroboration.

Evaluation requires real, aligned input files and labels the data source in the report:

```sh
python -m cwt_agent evaluate --prices data/processed/prices.csv --posts data/processed/posts.csv --data-label historical
```

The evaluator runs expanding chronological folds with a 5-day purge and writes predictions, metrics, a cost-sensitive illustrative h=1 risk table, a forecast-vs-actual SVG, and an X-sentiment-vs-indexed-BTC-price SVG. The risk table is an explicitly simple long/cash assumption, not an investment recommendation. Generate current n+1..n+5 forecasts from fully observed historical labels with `forecast`, then pass the resulting CSV to `outlook --forecasts`. An outlook without X/news data reports those sections as unavailable rather than inventing content.

## Working contract

The confirmed target is BTC-USD, daily UTC candles; n+0 is the latest completed close, while n+1..n+5 are the next five completed daily closes. This is the agreed interpretation because the assignment does not define the period. Read `docs/data-contracts.md` before connecting a source.

The model only uses price-derived features and unweighted text sentiment aggregates. Engagement/follower fields remain raw descriptive metadata unless the assignment owner clarifies that they belong in predictive features. Current news is used only for the current one-page outlook, never as historical model input.

No live scraping or emailed submission is automatic. Check the selected Apify Actor's input, historical coverage, pricing, and terms before a live run. See `docs/test-plan.md` for gates.
