import pandas as pd
import pytest

from cwt_agent.storage import SQLiteStore


def test_posts_price_news_upsert_and_manifest(tmp_path):
    store = SQLiteStore(tmp_path / "nested" / "research.sqlite")
    posts = pd.DataFrame([{
        "post_id": "p1", "created_at_utc": "2026-01-01T00:00:00Z", "text": "hello",
        "provider": "fixture", "likes": pd.NA,
    }])
    assert store.upsert_posts(posts) == 1
    assert store.upsert_posts(posts.assign(text="updated")) == 1
    prices = pd.DataFrame([{
        "timestamp_utc": "2026-01-01T00:00:00Z", "symbol": "BTC-USD",
        "venue": "fixture", "quote_currency": "USD", "open": 1.0, "high": 2.0,
        "low": 0.5, "close": 1.5, "source": "fixture",
    }])
    assert store.upsert_prices(prices) == 1
    news = pd.DataFrame([{
        "url": "https://example.test", "title": "fixture", "provider": "fixture",
        "retrieved_at_utc": "2026-01-01T00:00:00Z",
    }])
    assert store.upsert_news(news) == 1
    store.write_manifest("run-1", "2026-01-01T00:00:00Z", {"data_label": "synthetic"})


def test_store_rejects_missing_required_columns(tmp_path):
    store = SQLiteStore(tmp_path / "research.sqlite")
    with pytest.raises(ValueError, match="Posts missing"):
        store.upsert_posts(pd.DataFrame([{"post_id": "p1"}]))
