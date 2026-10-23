"""Small local SQLite store for normalized source data and model artifacts."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


class SQLiteStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS posts (
                    post_id TEXT PRIMARY KEY,
                    created_at_utc TEXT NOT NULL,
                    text TEXT NOT NULL,
                    likes INTEGER,
                    reposts INTEGER,
                    views INTEGER,
                    followers INTEGER,
                    provider TEXT NOT NULL,
                    source_id TEXT,
                    retrieved_at_utc TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_posts_created_at ON posts(created_at_utc);
                CREATE TABLE IF NOT EXISTS prices (
                    timestamp_utc TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    venue TEXT NOT NULL,
                    quote_currency TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL,
                    source TEXT NOT NULL,
                    PRIMARY KEY(timestamp_utc, symbol, venue)
                );
                CREATE TABLE IF NOT EXISTS news (
                    url TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    published_at_utc TEXT,
                    snippet TEXT,
                    provider TEXT NOT NULL,
                    retrieved_at_utc TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS predictions (
                    origin_date TEXT NOT NULL,
                    horizon INTEGER NOT NULL,
                    actual_log_return REAL,
                    predicted_log_return REAL NOT NULL,
                    predicted_close REAL NOT NULL,
                    actual_close REAL,
                    fold INTEGER,
                    model_name TEXT NOT NULL,
                    PRIMARY KEY(origin_date, horizon, model_name)
                );
                CREATE TABLE IF NOT EXISTS run_manifests (
                    run_id TEXT PRIMARY KEY,
                    created_at_utc TEXT NOT NULL,
                    manifest_json TEXT NOT NULL
                );
                """
            )

    def upsert_posts(self, posts: pd.DataFrame) -> int:
        required = {"post_id", "created_at_utc", "text", "provider"}
        missing = sorted(required - set(posts.columns))
        if missing:
            raise ValueError(f"Posts missing required columns: {', '.join(missing)}")
        columns = [
            "post_id", "created_at_utc", "text", "likes", "reposts", "views",
            "followers", "provider", "source_id", "retrieved_at_utc",
        ]
        values = [
            tuple(_db_value(row.get(column)) for column in columns)
            for row in posts.to_dict(orient="records")
        ]
        with self._connect() as db:
            db.executemany(
                """INSERT INTO posts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(post_id) DO UPDATE SET
                   created_at_utc=excluded.created_at_utc, text=excluded.text,
                   likes=excluded.likes, reposts=excluded.reposts, views=excluded.views,
                   followers=excluded.followers, provider=excluded.provider,
                   source_id=excluded.source_id, retrieved_at_utc=excluded.retrieved_at_utc""",
                values,
            )
        return len(values)

    def upsert_prices(self, prices: pd.DataFrame) -> int:
        required = {"timestamp_utc", "symbol", "venue", "quote_currency", "open", "high", "low", "close", "source"}
        missing = sorted(required - set(prices.columns))
        if missing:
            raise ValueError(f"Prices missing required columns: {', '.join(missing)}")
        columns = [
            "timestamp_utc", "symbol", "venue", "quote_currency", "open", "high",
            "low", "close", "volume", "source",
        ]
        values = [
            tuple(_db_value(row.get(column)) for column in columns)
            for row in prices.to_dict(orient="records")
        ]
        with self._connect() as db:
            db.executemany(
                """INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(timestamp_utc, symbol, venue) DO UPDATE SET
                   quote_currency=excluded.quote_currency, open=excluded.open,
                   high=excluded.high, low=excluded.low, close=excluded.close,
                   volume=excluded.volume, source=excluded.source""",
                values,
            )
        return len(values)

    def upsert_news(self, news: pd.DataFrame) -> int:
        required = {"url", "title", "provider", "retrieved_at_utc"}
        missing = sorted(required - set(news.columns))
        if missing:
            raise ValueError(f"News missing required columns: {', '.join(missing)}")
        columns = ["url", "title", "published_at_utc", "snippet", "provider", "retrieved_at_utc"]
        values = [
            tuple(_db_value(row.get(column)) for column in columns)
            for row in news.to_dict(orient="records")
        ]
        with self._connect() as db:
            db.executemany(
                """INSERT INTO news VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(url) DO UPDATE SET title=excluded.title,
                   published_at_utc=excluded.published_at_utc, snippet=excluded.snippet,
                   provider=excluded.provider, retrieved_at_utc=excluded.retrieved_at_utc""",
                values,
            )
        return len(values)

    def write_manifest(self, run_id: str, created_at_utc: str, manifest: dict[str, Any]) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO run_manifests VALUES (?, ?, ?)",
                (run_id, created_at_utc, json.dumps(manifest, sort_keys=True, default=str)),
            )


def _db_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value
