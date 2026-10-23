# Data and model contracts

All event timestamps are parsed as UTC-aware values. A user-facing view may additionally render America/New_York; do not use a fixed EST offset across daylight-saving changes.

## Normalized records

- Posts: `post_id`, `created_at_utc`, `text`, nullable `likes`, `reposts`, `views`, `followers`, `provider`, `source_id`, `retrieved_at_utc`.
- Daily prices: `timestamp_utc`, `open`, `high`, `low`, `close`, `volume`, `symbol`, `venue`, `quote_currency`, `source`.
- News: `title`, `url`, `published_at_utc`, `snippet`, `provider`, `retrieved_at_utc`.
- Source coverage manifest: source/provider, dataset or Actor ID, version/build if known, retrieval time, requested and observed date range, row count, errors, and license/terms note. Never include API tokens.

## Alignment and features

- Price bars are grouped by UTC day. For the provisional daily contract, a row is eligible only after that full UTC day has closed.
- X posts are deduplicated by stable post ID and grouped by UTC creation day. Posts are only usable once their timestamps are before the forecast cutoff.
- Keep raw API fields for audit where permitted. Predictive features are restricted to lagged/known price and unweighted text sentiment aggregates: post count, mean sentiment, positive/negative share, and sentiment dispersion. Engagement/followers are descriptive metadata until clarified against the “sentiment and price only” clause. Current news appears in the current outlook only.
- For each origin date `t`, horizons `h=1..5` use target log return `log(close[t+h]/close[t])`. Reconstruct predicted closes from the observed origin close. `n+0` is the known current completed close, not a forecast score.
- Imputation, scaling, feature selection, and model fitting are fit inside each chronological training fold only. No random shuffle.

## API boundaries

- Live providers require an explicit CLI opt-in and environment credentials. Each network client has an injectable transport for deterministic tests.
- Apify actor input is actor-specific. Keep its schema in a local config file and require an actor ID; validate the schema before a paid run. Bound date range, result count, run time, and retries. Use Bearer headers.
- Tavily is the required current-news provider; Exa is optional corroboration because the assignment lists its token but does not describe its function.
- Offline fixtures are labeled synthetic and must never be included in empirical model-performance metrics.
