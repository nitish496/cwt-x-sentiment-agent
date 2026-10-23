from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx
import pandas as pd
import pytest

from cwt_agent import providers
from cwt_agent.providers import (
    ProviderError,
    fetch_coinbase_daily,
    fetch_exa_news,
    fetch_tavily_news,
    run_apify_posts,
)


def client_for(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_coinbase_daily_normalizes_candles_and_paginates_under_300(monkeypatch):
    monkeypatch.setattr(providers.time, "sleep", lambda _: None)
    calls = []

    def handler(request):
        calls.append(request)
        start = datetime.fromisoformat(request.url.params["start"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(request.url.params["end"].replace("Z", "+00:00"))
        assert request.url.params["granularity"] == "86400"
        assert (end - start).days <= 299
        candle = [int(start.timestamp()), 90, 110, 100, 105, 1234.5]
        return httpx.Response(200, json=[candle])

    with client_for(handler) as client:
        result = fetch_coinbase_daily(
            "BTC-USD", "2024-01-01", "2024-12-31", client=client
        )

    assert len(calls) == 2
    assert list(result.columns) == providers.PRICE_COLUMNS
    assert result.iloc[0]["symbol"] == "BTC-USD"
    assert result.iloc[0]["venue"] == "Coinbase Exchange"
    assert result.iloc[0]["quote_currency"] == "USD"
    assert result.iloc[0]["source"] == "coinbase_exchange"
    assert str(result["timestamp_utc"].dtype) == "datetime64[ns, UTC]"


def test_coinbase_retries_429_and_rejects_bad_ranges(monkeypatch):
    monkeypatch.setattr(providers.time, "sleep", lambda _: None)
    attempts = 0

    def handler(_request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, json={"secret": "do not leak"})
        return httpx.Response(200, json=[])

    with client_for(handler) as client:
        result = fetch_coinbase_daily("BTC-USD", "2024-01-01", "2024-01-02", client)
    assert result.empty
    assert attempts == 2
    with pytest.raises(ValueError, match="on or before"):
        fetch_coinbase_daily("BTC-USD", "2024-01-02", "2024-01-01", client)


def test_coinbase_excludes_incomplete_current_daily_candle():
    now = pd.Timestamp.now(tz="UTC")
    today = now.floor("D")
    prior = today - pd.Timedelta(days=1)

    def handler(_request):
        candles = [
            [int(today.timestamp()), 90, 110, 100, 105, 10],
            [int(prior.timestamp()), 80, 100, 90, 95, 11],
        ]
        return httpx.Response(200, json=candles)

    with client_for(handler) as client:
        result = fetch_coinbase_daily("BTC-USD", prior.date().isoformat(), today.date().isoformat(), client)
    assert result.timestamp_utc.tolist() == [prior]


def test_apify_uses_actor_template_auth_header_and_paged_normalization(monkeypatch):
    monkeypatch.setattr(providers.time, "sleep", lambda _: None)
    seen = []
    token = "secret-test-token"

    def handler(request):
        seen.append(request)
        assert token not in str(request.url)
        if request.url.path.endswith("/acts/example~x-actor/runs"):
            assert request.headers["Authorization"] == f"Bearer {token}"
            body = request.read().decode()
            assert "2024-01-01T00:00:00Z" in body
            assert "2024-01-03T00:00:00Z" in body
            return httpx.Response(201, json={"data": {"id": "run-1", "status": "RUNNING"}})
        if request.url.path.endswith("/actor-runs/run-1"):
            return httpx.Response(200, json={"data": {"id": "run-1", "status": "SUCCEEDED", "defaultDatasetId": "set-1"}})
        if request.url.path.endswith("/datasets/set-1/items"):
            return httpx.Response(200, json=[{
                "tweet_id": 7,
                "created": "2024-01-02T05:00:00-05:00",
                "body": "BTC is positive",
                "likes": 4,
                "retweets": 2,
                "views": None,
                "followers": 19,
            }])
        raise AssertionError(f"Unexpected request path: {request.url.path}")

    field_map = {
        "post_id": "tweet_id",
        "created_at_utc": "created",
        "text": "body",
        "likes": "likes",
        "reposts": "retweets",
        "views": "views",
        "followers": "followers",
    }
    with client_for(handler) as client:
        result = run_apify_posts(
            "example~x-actor",
            token,
            {"searchQueries": ["BTC"], "from": "{{START_DATE}}", "to": "{{END_DATE}}", "limit": "{{MAX_ITEMS}}"},
            "2024-01-01",
            "2024-01-03",
            100,
            client,
            field_map=field_map,
            poll_interval=0,
        )

    run_request = next(request for request in seen if request.url.path.endswith("/runs"))
    run_body = run_request.read().decode()
    assert "{{START_DATE}}" not in run_body
    assert "{{END_DATE}}" not in run_body
    assert '"limit":"100"' in run_body

    assert len(seen) == 3
    assert list(result.columns) == providers.POST_COLUMNS
    assert result.iloc[0]["post_id"] == "7"
    assert result.iloc[0]["created_at_utc"] == pd.Timestamp("2024-01-02T10:00:00Z")
    assert result.iloc[0]["reposts"] == 2
    assert pd.isna(result.iloc[0]["views"])
    assert result.iloc[0]["provider"] == "apify"


def test_apify_deduplicates_and_rejects_missing_output_mapping(monkeypatch):
    monkeypatch.setattr(providers.time, "sleep", lambda _: None)
    data = [
        {"post_id": "a", "created_at_utc": "2024-01-01T00:00:00Z", "text": "one"},
        {"post_id": "a", "created_at_utc": "2024-01-01T00:00:00Z", "text": "duplicate"},
    ]

    def handler(request):
        if request.method == "POST":
            return httpx.Response(201, json={"data": {"id": "r", "status": "SUCCEEDED", "defaultDatasetId": "d"}})
        if request.url.path.endswith("/datasets/d/items"):
            return httpx.Response(200, json=data)
        raise AssertionError(request.url.path)

    with client_for(handler) as client:
        result = run_apify_posts("a~b", "token", {}, "2024-01-01", "2024-01-01", 10, client)
    assert len(result) == 1
    assert result.iloc[0]["text"] == "one"

    with pytest.raises(ValueError, match="field_map"):
        providers._normalize_posts(data[:1], {"text": "text"}, datetime.now(timezone.utc))


@pytest.mark.parametrize("provider_fn,header", [(fetch_tavily_news, "authorization"), (fetch_exa_news, "x-api-key")])
def test_news_providers_normalize_and_keep_secrets_out_of_urls(provider_fn, header):
    secret = "sensitive-test-key"
    request_capture = []

    def handler(request):
        request_capture.append(request)
        assert secret not in str(request.url)
        assert request.headers[header]
        if header == "authorization":
            assert request.headers[header] == f"Bearer {secret}"
            body = json.loads(request.read())
            assert body["filter_by_published_date"] is True
            return httpx.Response(200, json={"results": [{
                "title": "BTC outlook", "url": "https://news.example/a",
                "published_date": "2024-01-02", "content": "Snippet A",
            }]})
        assert request.headers[header] == secret
        return httpx.Response(200, json={"results": [{
            "title": "BTC outlook", "url": "https://news.example/a",
            "publishedDate": "2024-01-02T10:00:00Z", "highlights": ["Snippet A", "Snippet B"],
        }]})

    with client_for(handler) as client:
        result = provider_fn(secret, "BTC news", "2024-01-01", "2024-01-03", client)
    assert len(result) == 1
    assert list(result.columns) == providers.NEWS_COLUMNS
    expected_time = (
        pd.Timestamp("2024-01-02T00:00:00Z")
        if header == "authorization"
        else pd.Timestamp("2024-01-02T10:00:00Z")
    )
    assert result.iloc[0]["published_at_utc"] == expected_time
    assert result.iloc[0]["provider"] == ("tavily" if header == "authorization" else "exa")
    assert len(request_capture) == 1


def test_non_transient_http_error_is_sanitized(monkeypatch):
    monkeypatch.setattr(providers.time, "sleep", lambda _: None)

    def handler(_request):
        return httpx.Response(401, json={"message": "secret-test-token"})

    with client_for(handler) as client, pytest.raises(ProviderError) as error:
        fetch_tavily_news("secret-test-token", "BTC", "2024-01-01", "2024-01-02", client)
    assert "401" in str(error.value)
    assert "secret-test-token" not in str(error.value)
