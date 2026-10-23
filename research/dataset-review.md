# Dataset and Market-Data Feasibility Review

**Reviewer role:** senior market-data / licensing audit (30-years-equivalent role framing; this is a review persona, not a claim about a human work history).
**Scope:** source suitability for the provisional BTC-USD daily assignment. No data was downloaded; no credentials were used.

## Findings

The assignment's linked data catalog is a list of candidate sources, not the tweet files. It names the original **Bitcoin tweets — 16M tweets** Kaggle dataset, describes dates 2016-01-01 to 2019-03-29 and a 4.2 GB file, and labels it CC0. Those properties come from the assignment-provided catalog. I could not independently verify the current Kaggle license or file metadata from its dynamic listing page in this review. The direct Kaggle page is [alaix14/bitcoin-tweets-20160101-to-20190329](https://www.kaggle.com/datasets/alaix14/bitcoin-tweets-20160101-to-20190329). A published study citing that corpus reports 18,810,521 Bitcoin/BTC tweets across the same date interval; its author/date coverage differs slightly from the round “16M” name. Treat row count as corpus-version dependent, not guaranteed. [Dataset page](https://www.kaggle.com/datasets/alaix14/bitcoin-tweets-20160101-to-20190329) · [study describing corpus dates and count](https://file.ewadirect.com/press/media/upload/vol_pdf/ace/29.pdf)

## Candidate comparison

| Source | Verified facts | Fit and limits |
|---|---|---|
| [Original Kaggle Bitcoin tweets, 2016–2019](https://www.kaggle.com/datasets/alaix14/bitcoin-tweets-20160101-to-20190329) | The assignment catalog describes ~4.2 GB and CC0. A published study describes 18.81M BTC/Bitcoin posts, Jan 2016–Mar 2019. The source is a Kaggle dataset page. | Best listed historical candidate if the current file schema includes usable timestamps and engagement columns and the CC0 metadata is confirmed. Large (~4 GB), old, and does not cover today's required 180-day ingestion window. Download/process only after confirming disk space, actual license, and schema. Keep the corpus out of Git and follow the applicable source/platform terms. |
| [ExponentialScience/DLT-Tweets](https://huggingface.co/datasets/ExponentialScience/DLT-Tweets) | Dataset card reports 22,033,090 records / 3.58 GB; timestamped English DLT posts from 2013 to mid-2023; fields include tweet, timestamp, sentiment class/score; usernames removed. License is CC-BY-NC 4.0. | Strong timestamp/sentiment schema but broad DLT topic and no user/engagement fields. **Not a safe default for an employer-facing/company internship deliverable:** noncommercial-only licensing is explicit. Obtain rights-holder permission or choose a dataset with a verified compatible license before using it. The card notes no posts after mid-2023. |
| [StephanAkkerman/crypto-stock-tweets](https://huggingface.co/datasets/StephanAkkerman/crypto-stock-tweets) | Dataset card reports 8,024,269 rows / 1.73 GB, including ~4.82M Bitcoin tweets, CC-BY-4.0 as hosted; preview fields are `text` and `url`. It aggregates ten sources. | Not sufficient for reliable time alignment as shown: a timestamp sometimes appears inside tweet text, but there is no structured timestamp in the displayed schema, nor engagement/follower features. Composite dataset license may not resolve upstream tweet rights. Do not use as the time-series source without investigating component provenance and permissions. |
| [Bitcoin Tweets sentiment Kaggle mirror on Hugging Face](https://huggingface.co/datasets/cvnberk/bitcoin_tweets_sentiment_kaggle) | Hugging Face listing reports 109,218 rows and 9.87 MB in Parquet. | Convenient size, but this audit could not verify its README/schema, date range, or license. Not approved for training until those are checked. |
| [Bitcoin Tweets Dataset (2025–2026) Kaggle notebook input](https://www.kaggle.com/code/sumitchavhan7/bitcoin-tweets-dataset-2025-2026/input) | Kaggle notebook listing reports `bitcoin_tweets_latest.csv` at 2.1 GB. | Potentially newer coverage, but the listing did not establish row count, schema, date range, or license. Inspect metadata before selecting; it is still not a substitute for the requested Apify collection in the latest 180 days. |

The dataset catalog also mentions StockTwits. **StockTwits is a separate platform, not X/Twitter**, so it cannot be reported as the assignment's historical-X corpus. It may be considered only as a separately labeled sensitivity analysis, with a compatible license.

Kaggle's official documentation says dataset publishers choose a license (including CC0 and multiple restricted licenses). The dataset page's own current license field must therefore be checked rather than inferred from a mirror or filename. [Kaggle dataset licensing documentation](https://www.kaggle.com/docs/datasets)

## Market-price source

For an unauthenticated, reproducible public-price adapter, Binance documents a public market-data base URL `https://data-api.binance.vision` and Spot Kline/Candlestick endpoint `/api/v3/klines`. Daily OHLCV bars are available by symbol and time range; this source yields **BTCUSDT**, not BTC-USD. If the report uses it, label it BTC/USDT and state the quote-currency difference; do not silently call it BTC-USD. [Binance REST API overview](https://developers.binance.com/en/docs/products/spot/rest-api) · [Kline endpoint](https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints)

If strict USD denomination is required, choose a public BTC-USD market source and record the exchange in the report. Never combine one exchange's prices with another without documenting the choice. Validation should use the same source consistently for all dates and return targets.

## Recommended data policy and next checks

1. Keep the provisional BTC daily setup until the owner confirms asset/cadence.
2. Obtain the original Kaggle file metadata and inspect a small schema sample first. Confirm license, file version, timestamp column/timezone, engagement field availability, row count, coverage, and storage needs. The assignment-provided claim of CC0 is not independently confirmed here.
3. If full Kaggle access is impractical, use deterministic fixtures for software tests and clearly label any model run that uses a small historical proxy. Do not claim live 180-day X ingestion or evaluation until Apify data is actually collected.
4. Do not use the 22M-row DLT corpus for an employer-facing result without a license decision because CC-BY-NC 4.0 is stated on the dataset card.
5. Use a public exchange OHLCV endpoint without trading credentials. Keep raw inputs outside Git; publish code, source manifest, aggregates permissible under source terms, and reproducible acquisition steps instead of redistributing tweet text.

## Review status

### Additional public candidate checked on 2026-10-22

A live search surfaced a 76,797-row, 9-column Kaggle mirror titled “Bitcoin Tweets[Sentiment Analysis]” (9.23 MB), with fields `token`, `date`, reply/like/retweet/quote counts, `text`, `sentiment_label`, and `sentiment_score`. The mirror labels it CC BY 4.0 and cites Daniel Partida's Mendeley Data DOI. The Mendeley record itself describes “Bitcoin tweets,” dates its V1 to 2023-10-06, and lists CC BY 4.0. Sources: [Kaggle mirror metadata](https://baselight.app/u/kaggle/dataset/sujaykapadnis_bitcoin_tweets), [Mendeley Data record](https://data.mendeley.com/datasets/x7yvshrnxy/1).

This is a **candidate only, not approved for model use**: we have not downloaded a sample, verified observed timestamp coverage/timezone, established a stable unique post ID (the schema shows `token`, not an ID), or resolved whether the uploader's license is sufficient for underlying X content and this employer-facing use. The primary Mendeley record confirms CC BY 4.0 for the deposited dataset but does not establish the uploader's right to sublicense underlying X content. Current X policy restricts redistribution of downloadable X content and its Terms prohibit scraping without prior written consent. Do not label this candidate as the required recent 180-day Apify collection or use it for submitted performance claims without CWT's approved source/permission and a source sample review. Sources: [Mendeley record](https://data.mendeley.com/datasets/x7yvshrnxy/1), [X Developer Policy](https://docs.x.com/developer-terms/policy), [X Terms of Service](https://x.com/en/tos).

No dataset is yet approved for the final empirical report. Live Apify/Tavily/Exa credentials were not supplied and were not used.
