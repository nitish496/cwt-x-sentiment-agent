# Provider Completion Review

Role framing: senior API integration and reproducibility review. This is a task persona, not a claim of real employment history.

## Can the assignment run without API credentials?

Not end to end. Coinbase daily BTC-USD candles are the one genuinely keyless live source in the current implementation. The Apify run endpoint, Tavily search, and Exa search each require an API credential. Free signup or free credits may reduce cost, but they still require account creation and an issued key; they do not create a keyless path. The code correctly blocks live X/news collection when required keys are absent.

The local CSV/JSONL importer is a no-key alternative only when a historical X export is already available and its provenance, license, platform terms, schema, and observed coverage are approved. It cannot provide the required fresh collection by itself. Synthetic fixtures are only for software checks and cannot support empirical assignment claims.

## Concrete integration issue found

`fetch_tavily_news` sends explicit `start_date` and `end_date`, but sets `filter_by_published_date` to `false`. Tavily documents that with this flag false, results outside the requested date window and results with no detectable publish date are not removed. That makes the current-news window non-strict; an out-of-window or undated item may enter the stored current-news set and outlook.

Tavily's current API reference says `filter_by_published_date: true` removes results outside `start_date`/`end_date` and removes undated results, while also enabling each result's `published_date`. This was corrected in `src/cwt_agent/providers.py`; the provider mock now asserts the filter is enabled. Python 3.12 tests and Ruff pass after the change. Source: [Tavily Search API reference](https://docs.tavily.com/documentation/api-reference/endpoint/search).

No API calls, account creation, paid actions, or external submissions were made during this review.

## Readiness summary

- **Keyless:** price download and offline import/evaluation, subject to having an approved dataset.
- **Credentialed:** Apify X collection and Tavily news; Exa is optional corroboration.
- **Still needed for a real submission:** user-issued keys, a reviewed Apify Actor and input/output mapping, an approved historical X dataset with verified coverage, live run outputs, and fresh walk-forward metrics.
