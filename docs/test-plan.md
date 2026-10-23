# Verification plan

## Unit tests
- Parse/normalize timestamp, source rows, and nullable engagement fields; reject malformed rows with a bounded diagnostics file.
- Deduplicate stable IDs, preserve UTC, and correctly handle UTC day boundaries and daylight-saving display conversion.
- Aggregate sentiment per day; preserve zero-post days and avoid counting posts after a forecast cutoff.
- Build lag features without using current/future close-derived values incorrectly; assert the h-step target dates.
- Verify each walk-forward fold is chronological, purges overlapping labels, and keeps preprocessing fit on training only.
- Verify random-walk forecasts, reconstructed-price MAE/RMSE, directional metrics, and risk-metric edge cases (zero variance, no downside, drawdown).
- Provider adapters: pagination, HTTP errors, 429 Retry-After, malformed payloads, and redaction; use mocked transports, never live credentials.

## Integration / pipeline checks
- Small, conspicuously synthetic fixture goes from CSV/JSON inputs through normalized SQLite rows, feature matrix, forecasts, and chart/report generation; label it synthetic and block accidental empirical use by default.
- Generate accessible dependency-free SVGs for five forecast horizons and for daily sentiment vs indexed price; preserve visible gaps on no-post days and verify outlook rows for n+1 through n+5.
- Repeat ingestion is idempotent; manifest records coverage and caller-supplied source/rights. Input hashes remain a follow-up.
- No credentials, raw tweet text, large datasets, synthetic fixtures presented as history, or generated reports are staged for publication.

## Acceptance gates
- All tests pass; CLI help and offline demo work without keys; build/package install works from a clean environment.
- Live CLI refuses to run without `--live`, required environment keys, Actor ID/input config, and explicit spend/item limits.
- Current forecast rows use only labels whose target dates are at or before the latest completed close; incomplete current-day Coinbase candles are excluded.
- No headline performance report unless a real source and actual out-of-sample predictions are present. State unavailable metrics as unavailable.
