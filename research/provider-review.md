# Provider integration review

Role: senior data-platform/API architecture review. This is a technical role framing, not a claim of real-world employment history.

## Recommended integration shape

- Keep providers behind small Python adapters and normalized records. The research/model pipeline should consume normalized data, not provider-specific JSON.
- All adapters accept an injectable client/transport and have fixture-backed tests. No live API call is needed for unit tests or for the no-credential demo.
- Credentials come only from environment variables loaded from an ignored `.env`: `APIFY_API_TOKEN`, `APIFY_ACTOR_ID`, `TAVILY_API_KEY`, and `EXA_API_KEY`. `.env.example` includes blank placeholders only. Never put tokens in URL query strings, logs, notebooks, report artifacts, screenshots, ZIPs, or Git history. The Apify docs show a token query parameter as one supported route, but use the Python client with a token configured out-of-band to reduce URL/log exposure.
- Config, code, and report record provider, actor ID, actor build/version if available, run ID, dataset ID, query, cutoff, retrieval time, and normalized row count. Avoid credentials and user/private data in run metadata.

## Apify X ingestion

Apify's documented flow is: POST to run an Actor with JSON input, wait for completion, and read its default dataset; the Python client documents `client.actor(actor_id).call(run_input)` and `client.dataset(defaultDatasetId).list_items()`. [Run Actors](https://docs.apify.com/actors/running)

Recommended implementation:

1. Require an explicit `APIFY_ACTOR_ID`; don't silently choose a Store actor. The exact search/input schema, historical depth, and whether author followers/views are available depend on the chosen actor and account plan. Inspect and pin that actor's input/output schema before enabling it.
2. Form searches from the configured symbol/cashtag/name, with UTC `from`/`until` bounds representing the preceding 180 days. Pass actor-specific fields only after validating its documented schema. Record the exact input (excluding secrets).
3. Start asynchronously for the full scrape, persist the run ID, poll the run, then page through dataset items. The synchronous endpoint has a 300-second ceiling and can time out; Apify docs recommend the asynchronous endpoint for longer work. [Synchronous run timeout](https://docs.apify.com/api/v2/actor-run-sync-get-dataset-items-post)
4. Normalize one canonical post ID, created timestamp, text, like/repost/view counts, author follower count, matched query, raw source ID, and fetched-at timestamp. Store raw JSON locally only when allowed by the actor's terms; normalized data has a unique constraint on `(provider, post_id)`. Upsert in bounded batches so reruns are idempotent.
5. Treat requested fields as nullable; do not replace missing counts with fabricated zero. Validate timestamp parsing and preserve `created_at_utc`; reject malformed timestamps into a diagnostic file.
6. Retry transient HTTP 429/5xx/network errors with bounded exponential backoff and `Retry-After`; don't retry permanent auth/schema errors. Bound run time, item count, and spend; expose CLI dry-run and explicit `--live` gate.
7. Before use, review actor terms/pricing, X/platform terms, and limits. Do not assume an Apify actor grants unrestricted access or a full historical archive. Capture observed coverage and gaps.

## News search: Tavily and Exa

The assignment names Tavily for current asset news and Exa in the submission credential list, without specifying Exa's research role. Tavily's current Search endpoint is `POST https://api.tavily.com/search`, authenticating with a bearer token; its request supports date bounds and published-date filtering. [Tavily Search API](https://docs.tavily.com/documentation/api-reference/endpoint/search)

- Use Tavily for the required current/recent news summary. Query asset aliases + material financial/news terms, constrain dates to the summary window, request published dates, and retain source title/URL/date/snippet/score and retrieval time. Deduplicate normalized URLs.
- Exa's current endpoint is `POST https://api.exa.ai/search` with `x-api-key`; its search supports `category: "news"`, `startPublishedDate`, `endPublishedDate`, `numResults`, and contents such as highlights/text. [Exa Search API](https://exa.ai/docs/reference/search)
- Since Exa is not otherwise specified in the task scope, make Exa optional as a secondary news-discovery/cross-check adapter, not a hidden requirement for model features. Keep Tavily and Exa results labeled by provider and never treat the same syndicated article as independent corroboration.
- Current web/news results belong in the current outlook only. Do not leak current articles into historical backtests. Historical news may only enter a feature row if its published timestamp was available before the forecast cutoff and the collection source has reproducible historical coverage.

## No-credentials path

- Use small, license-cleared synthetic fixtures for unit tests; mark synthetic fixtures conspicuously and do not use them to claim model performance.
- Provide a CSV/Parquet import path with documented expected columns for user-supplied licensed historical posts and prices. Validate schema, timezone, date range, and duplicates.
- If credentials are missing, local `demo` runs only on supplied fixtures or user data, clearly prints `live providers disabled`, and writes a coverage manifest. It must not silently fall back to fabricated/empty X history while claiming successful scraping.
- Generate evaluation metrics only when chronological out-of-sample predictions and the stated trading rule exist. Otherwise report `not computed` and the exact missing prerequisite.

## Open questions before enabling live adapters

- Which exact Apify Actor/actor ID is approved, what are its input keys and historical coverage, and what does it cost at requested volume?
- Can it access all required post metadata for 180 days under the user's plan and applicable terms?
- What exactly should Exa do in this project, given the assignment only names Tavily for news?
- Which historical dataset will be selected after checking its linked source, license, date coverage, and download size?
- The employer's request for API tokens by email is not a reason to store or transmit reusable secrets in the repo. If token proof is later required, use dedicated limited-scope/revocable keys and an approved secure delivery method.
