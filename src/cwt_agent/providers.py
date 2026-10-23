"""Small, secret-safe clients for public prices, Apify posts, and current news."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote

import httpx
import pandas as pd

COINBASE_API = "https://api.exchange.coinbase.com"
APIFY_API = "https://api.apify.com/v2"
TAVILY_API = "https://api.tavily.com/search"
EXA_API = "https://api.exa.ai/search"

PRICE_COLUMNS = [
    "timestamp_utc", "open", "high", "low", "close", "volume",
    "symbol", "venue", "quote_currency", "source",
]
POST_COLUMNS = [
    "post_id", "created_at_utc", "text", "likes", "reposts", "views",
    "followers", "provider", "source_id", "retrieved_at_utc",
]
NEWS_COLUMNS = ["title", "url", "published_at_utc", "snippet", "provider", "retrieved_at_utc"]

_TRANSIENT_STATUSES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_MAX_APIFY_ITEMS = 100_000
_MAX_APIFY_PAGE = 1_000


class ProviderError(RuntimeError):
    """Sanitized provider failure; intentionally excludes response bodies and headers."""

    def __init__(self, provider: str, message: str, status_code: int | None = None):
        self.provider = provider
        self.status_code = status_code
        suffix = f" (HTTP {status_code})" if status_code is not None else ""
        super().__init__(f"{provider}: {message}{suffix}")


@contextmanager
def _client_context(client: Any | None) -> Iterator[Any]:
    if client is not None:
        yield client
        return
    with httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0)) as default_client:
        yield default_client


def _retry_after_seconds(response: Any, fallback: float) -> float:
    value = response.headers.get("Retry-After") if getattr(response, "headers", None) else None
    if not value:
        return fallback
    try:
        return min(max(float(value), 0.0), 8.0)
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return min(max((retry_at - datetime.now(timezone.utc)).total_seconds(), 0.0), 8.0)
        except (TypeError, ValueError, OverflowError):
            return fallback


def _request_json(
    client: Any,
    provider: str,
    method: str,
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    params: Mapping[str, Any] | None = None,
    json_body: Mapping[str, Any] | None = None,
    max_retries: int = _MAX_RETRIES,
) -> Any:
    """Issue bounded retries; never include body/headers in an exception."""
    for attempt in range(max_retries + 1):
        try:
            response = client.request(
                method,
                url,
                headers=dict(headers or {}),
                params=dict(params or {}),
                json=dict(json_body) if json_body is not None else None,
            )
        except (httpx.TimeoutException, httpx.NetworkError):
            if attempt >= max_retries:
                raise ProviderError(provider, "request failed after bounded retries") from None
            time.sleep(min(0.5 * (2**attempt), 8.0))
            continue

        status = int(response.status_code)
        if status in _TRANSIENT_STATUSES and attempt < max_retries:
            time.sleep(_retry_after_seconds(response, min(0.5 * (2**attempt), 8.0)))
            continue
        if status < 200 or status >= 300:
            raise ProviderError(provider, "request was rejected", status)
        try:
            return response.json()
        except (ValueError, json.JSONDecodeError):
            raise ProviderError(provider, "response was not valid JSON", status) from None
    raise ProviderError(provider, "request failed after bounded retries")


def _as_utc(value: str | date | datetime, field_name: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime(value.year, value.month, value.day)
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            try:
                parsed_date = date.fromisoformat(value)
                parsed = datetime(parsed_date.year, parsed_date.month, parsed_date.day)
            except ValueError:
                raise ValueError(f"{field_name} must be an ISO date or datetime") from None
    else:
        raise TypeError(f"{field_name} must be a date, datetime, or ISO string")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_timestamp(value: Any, provider: str) -> Any:
    if value in (None, ""):
        return pd.NaT
    try:
        return pd.to_datetime(value, utc=True, errors="raise")
    except (ValueError, TypeError, OverflowError):
        raise ProviderError(provider, "response contained an invalid timestamp") from None


def fetch_coinbase_daily(
    symbol: str,
    start: str | date | datetime,
    end: str | date | datetime,
    client: Any | None = None,
) -> pd.DataFrame:
    """Fetch daily Coinbase Exchange candles in ranges no larger than 300 days.

    The requested `start` and `end` are inclusive UTC day boundaries. Coinbase
    returns `[time, low, high, open, close, volume]` arrays in descending order.
    """
    product_id = symbol.strip().upper()
    if not re.fullmatch(r"[A-Z0-9]+-[A-Z0-9]+", product_id):
        raise ValueError("symbol must be a Coinbase product such as BTC-USD")
    start_utc, end_utc = _as_utc(start, "start"), _as_utc(end, "end")
    if start_utc > end_utc:
        raise ValueError("start must be on or before end")
    if start_utc.time() != datetime.min.time() or end_utc.time() != datetime.min.time():
        raise ValueError("daily candle range must use UTC day boundaries")

    rows: list[dict[str, Any]] = []
    # Use at most a 299-day inclusive span (300 possible daily buckets).
    cursor = start_utc
    with _client_context(client) as session:
        while cursor <= end_utc:
            chunk_end = min(cursor + timedelta(days=299), end_utc)
            payload = _request_json(
                session,
                "Coinbase",
                "GET",
                f"{COINBASE_API}/products/{quote(product_id, safe='-')}/candles",
                params={"start": _iso_z(cursor), "end": _iso_z(chunk_end), "granularity": 86400},
            )
            if not isinstance(payload, list):
                raise ProviderError("Coinbase", "candle response was not a list")
            for candle in payload:
                if not isinstance(candle, (list, tuple)) or len(candle) != 6:
                    raise ProviderError("Coinbase", "candle had an unexpected shape")
                try:
                    timestamp = pd.to_datetime(int(candle[0]), unit="s", utc=True)
                    low, high, open_, close, volume = (float(part) for part in candle[1:])
                except (TypeError, ValueError, OverflowError):
                    raise ProviderError("Coinbase", "candle contained invalid numeric data") from None
                # A daily candle is complete only after its UTC day has ended.
                if start_utc <= timestamp.to_pydatetime() <= end_utc and (
                    timestamp + pd.Timedelta(days=1) <= pd.Timestamp.now(tz="UTC")
                ):
                    rows.append({
                        "timestamp_utc": timestamp,
                        "open": open_, "high": high, "low": low, "close": close,
                        "volume": volume,
                        "symbol": product_id,
                        "venue": "Coinbase Exchange",
                        "quote_currency": product_id.split("-", 1)[1],
                        "source": "coinbase_exchange",
                    })
            cursor = chunk_end + timedelta(days=1)

    frame = pd.DataFrame(rows, columns=PRICE_COLUMNS)
    if not frame.empty:
        # Keep the normalized timestamp schema stable across Pandas 2.x/3.x;
        # newer pandas may otherwise infer microsecond precision from epoch data.
        frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True).astype(
            "datetime64[ns, UTC]"
        )
        frame = frame.drop_duplicates(subset=["timestamp_utc"]).sort_values("timestamp_utc")
        frame = frame.reset_index(drop=True)
    return frame


def _render_input(value: Any, replacements: Mapping[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: _render_input(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_render_input(item, replacements) for item in value]
    if isinstance(value, str):
        rendered = value
        for key, replacement in replacements.items():
            rendered = (
                rendered.replace("{{" + key + "}}", replacement)
                .replace("{" + key + "}", replacement)
                .replace("{{" + key.upper() + "}}", replacement)
                .replace("{" + key.upper() + "}", replacement)
            )
        return rendered
    return value


def _normalize_posts(
    records: list[Any],
    field_map: Mapping[str, str] | None,
    retrieved_at: datetime,
) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=POST_COLUMNS)
    if field_map is None:
        # Only exact contract field names are safe to infer. Other Actor schemas
        # require caller-supplied mappings rather than guessed field aliases.
        field_map = {column: column for column in POST_COLUMNS if column != "provider"}
    required = {"post_id", "created_at_utc", "text"}
    if not required.issubset(field_map):
        raise ValueError("field_map must map post_id, created_at_utc, and text")
    rows = []
    for item in records:
        if not isinstance(item, dict):
            raise ProviderError("Apify", "Actor dataset item was not an object")
        row: dict[str, Any] = {}
        for column in POST_COLUMNS:
            if column == "provider":
                row[column] = "apify"
            elif column == "retrieved_at_utc":
                row[column] = pd.Timestamp(retrieved_at)
            else:
                source_key = field_map.get(column)
                raw_value = item.get(source_key) if source_key else None
                if column == "created_at_utc":
                    row[column] = _parse_timestamp(raw_value, "Apify")
                elif column in {"likes", "reposts", "views", "followers"}:
                    if raw_value is None:
                        row[column] = None
                    else:
                        try:
                            row[column] = int(raw_value)
                        except (TypeError, ValueError, OverflowError):
                            raise ProviderError("Apify", f"Actor item had invalid {column}") from None
                else:
                    row[column] = raw_value
        if row["post_id"] in (None, "") or row["created_at_utc"] is pd.NaT or row["text"] is None:
            raise ProviderError("Apify", "Actor item lacked required mapped fields")
        row["post_id"] = str(row["post_id"])
        row["text"] = str(row["text"])
        row["source_id"] = str(row["source_id"] or row["post_id"])
        rows.append(row)
    frame = pd.DataFrame(rows, columns=POST_COLUMNS)
    return frame.drop_duplicates(subset=["post_id"]).reset_index(drop=True)


def run_apify_posts(
    actor_id: str,
    token: str,
    input_template: Mapping[str, Any],
    start: str | date | datetime,
    end: str | date | datetime,
    max_items: int,
    client: Any | None = None,
    *,
    field_map: Mapping[str, str] | None = None,
    poll_interval: float = 1.0,
    max_poll_attempts: int = 120,
) -> pd.DataFrame:
    """Run a caller-configured Apify Actor; its input and output remain Actor-specific."""
    if not actor_id or not token:
        raise ValueError("actor_id and token are required")
    if not isinstance(input_template, Mapping):
        raise TypeError("input_template must be an Actor-specific mapping")
    if not 1 <= max_items <= _MAX_APIFY_ITEMS:
        raise ValueError(f"max_items must be between 1 and {_MAX_APIFY_ITEMS}")
    if poll_interval < 0 or max_poll_attempts < 1 or max_poll_attempts > 600:
        raise ValueError("poll_interval or max_poll_attempts is outside the supported bound")
    start_utc, end_utc = _as_utc(start, "start"), _as_utc(end, "end")
    if start_utc > end_utc:
        raise ValueError("start must be on or before end")
    safe_actor_id = quote(actor_id.strip(), safe="~")
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    actor_input = _render_input(
        dict(input_template),
        {"start": _iso_z(start_utc), "end": _iso_z(end_utc),
         "start_date": _iso_z(start_utc), "end_date": _iso_z(end_utc),
         "max_items": str(max_items)},
    )
    retrieved_at = datetime.now(timezone.utc)
    with _client_context(client) as session:
        run_payload = _request_json(
            session,
            "Apify",
            "POST",
            f"{APIFY_API}/acts/{safe_actor_id}/runs",
            headers=headers,
            params={"maxItems": max_items},
            json_body=actor_input,
        )
        run = run_payload.get("data") if isinstance(run_payload, dict) else None
        if not isinstance(run, dict) or not run.get("id"):
            raise ProviderError("Apify", "run response lacked a run identifier")
        run_id = quote(str(run["id"]), safe="")
        state = str(run.get("status", "READY")).upper()
        for _ in range(max_poll_attempts):
            if state in {"SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"}:
                break
            if poll_interval:
                time.sleep(min(poll_interval, 30.0))
            status_payload = _request_json(
                session, "Apify", "GET", f"{APIFY_API}/actor-runs/{run_id}", headers=headers
            )
            status_data = status_payload.get("data") if isinstance(status_payload, dict) else None
            if not isinstance(status_data, dict):
                raise ProviderError("Apify", "run status response had an unexpected shape")
            state = str(status_data.get("status", "")).upper()
            run.update(status_data)
            if state not in {"READY", "RUNNING", "TIMING-OUT", "ABORTING", ""} and state not in {
                "SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"
            }:
                raise ProviderError("Apify", "run returned an unknown status")
        else:
            raise ProviderError("Apify", "run did not finish before the polling limit")
        if state != "SUCCEEDED":
            raise ProviderError("Apify", f"Actor run ended with status {state or 'unknown'}")
        dataset_id = run.get("defaultDatasetId")
        if not dataset_id:
            raise ProviderError("Apify", "successful run did not include a default dataset")
        dataset_id = quote(str(dataset_id), safe="")
        records: list[Any] = []
        offset = 0
        while len(records) < max_items:
            page_size = min(_MAX_APIFY_PAGE, max_items - len(records))
            page = _request_json(
                session,
                "Apify",
                "GET",
                f"{APIFY_API}/datasets/{dataset_id}/items",
                headers=headers,
                params={"offset": offset, "limit": page_size, "format": "json"},
            )
            if not isinstance(page, list):
                raise ProviderError("Apify", "dataset response was not a list")
            if not page:
                break
            records.extend(page[: max_items - len(records)])
            offset += len(page)
            if len(page) < page_size:
                break
    return _normalize_posts(records[:max_items], field_map, retrieved_at)


def _normalize_news(items: Any, provider: str) -> pd.DataFrame:
    if not isinstance(items, list):
        raise ProviderError(provider, "search response did not contain a results list")
    retrieved_at = pd.Timestamp.now(tz="UTC")
    rows = []
    seen_urls: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        if not isinstance(url, str) or not url.strip() or url in seen_urls:
            continue
        seen_urls.add(url)
        date_value = item.get("published_date") or item.get("publishedDate") or item.get("published_at")
        content = item.get("content") or item.get("snippet") or item.get("highlights")
        if isinstance(content, list):
            content = " ".join(str(part) for part in content if part)
        rows.append({
            "title": str(item.get("title") or ""),
            "url": url,
            "published_at_utc": _parse_timestamp(date_value, provider),
            "snippet": str(content or ""),
            "provider": provider.lower(),
            "retrieved_at_utc": retrieved_at,
        })
    return pd.DataFrame(rows, columns=NEWS_COLUMNS)


def fetch_tavily_news(
    api_key: str,
    query: str,
    start: str | date | datetime,
    end: str | date | datetime,
    client: Any | None = None,
) -> pd.DataFrame:
    if not api_key or not query.strip():
        raise ValueError("api_key and query are required")
    start_utc, end_utc = _as_utc(start, "start"), _as_utc(end, "end")
    if start_utc > end_utc:
        raise ValueError("start must be on or before end")
    body = {
        "query": query,
        "topic": "news",
        "search_depth": "basic",
        "max_results": 10,
        "start_date": start_utc.date().isoformat(),
        "end_date": end_utc.date().isoformat(),
        "include_published_date": True,
        "filter_by_published_date": True,
        "include_answer": False,
        "include_raw_content": False,
    }
    with _client_context(client) as session:
        payload = _request_json(
            session,
            "Tavily",
            "POST",
            TAVILY_API,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json_body=body,
        )
    return _normalize_news(payload.get("results") if isinstance(payload, dict) else None, "Tavily")


def fetch_exa_news(
    api_key: str,
    query: str,
    start: str | date | datetime,
    end: str | date | datetime,
    client: Any | None = None,
) -> pd.DataFrame:
    if not api_key or not query.strip():
        raise ValueError("api_key and query are required")
    start_utc, end_utc = _as_utc(start, "start"), _as_utc(end, "end")
    if start_utc > end_utc:
        raise ValueError("start must be on or before end")
    body = {
        "query": query,
        "type": "auto",
        "category": "news",
        "numResults": 10,
        "startPublishedDate": _iso_z(start_utc),
        "endPublishedDate": _iso_z(end_utc),
        "contents": {"highlights": True},
    }
    with _client_context(client) as session:
        payload = _request_json(
            session,
            "Exa",
            "POST",
            EXA_API,
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            json_body=body,
        )
    return _normalize_news(payload.get("results") if isinstance(payload, dict) else None, "Exa")
