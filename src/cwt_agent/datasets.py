"""Bounded, local-only import helpers for user-supplied historical data."""
from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

POST_COLUMNS = [
    "post_id", "created_at_utc", "text", "likes", "reposts", "views",
    "followers", "provider", "source_id", "retrieved_at_utc",
]
_OPTIONAL = ["likes", "reposts", "views", "followers", "provider", "source_id", "retrieved_at_utc"]
DEFAULT_MAX_ROWS = 1_000_000
DEFAULT_MAX_BYTES = 512 * 1024 * 1024


def _read_records(path: Path, fmt: str, max_rows: int, max_bytes: int) -> list[dict[str, Any]]:
    if max_rows < 1 or max_bytes < 1:
        raise ValueError("max_rows and max_bytes must be positive")
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size > max_bytes:
        raise ValueError(f"Input exceeds max_bytes ({max_bytes})")
    records: list[dict[str, Any]] = []
    if fmt == "csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError("CSV must include a header row")
            for row in reader:
                if len(records) >= max_rows:
                    raise ValueError(f"Input exceeds max_rows ({max_rows})")
                records.append(dict(row))
    elif fmt == "jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                if len(records) >= max_rows:
                    raise ValueError(f"Input exceeds max_rows ({max_rows})")
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise TypeError("Each JSONL line must contain an object")
                records.append(item)
    else:
        raise ValueError("format must be 'csv' or 'jsonl'")
    return records


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return pd.NaT
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp


def import_posts(
    path: str | Path,
    column_map: Mapping[str, str],
    source: str,
    *,
    format: str | None = None,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_bytes: int = DEFAULT_MAX_BYTES,
    dataset_id: str | None = None,
    version: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Import CSV/JSONL posts using explicit normalized-field -> input-field mapping.

    Naive timestamps are interpreted as UTC. Source and permission are supplied by
    the caller and are not independently verified by this importer.
    """
    source_path = Path(path)
    fmt = (format or source_path.suffix.lstrip(".")).lower()
    aliases = dict(column_map)
    required = {"post_id", "created_at_utc", "text"}
    missing = required - aliases.keys()
    if missing:
        raise ValueError(f"column_map missing required fields: {', '.join(sorted(missing))}")
    if not source or not source.strip():
        raise ValueError("source must identify the user-supplied data source")

    records = _read_records(source_path, fmt, max_rows, max_bytes)
    mapped: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, record in enumerate(records, start=1):
        try:
            item = {field: record.get(input_name) for field, input_name in aliases.items() if field in POST_COLUMNS}
            item["post_id"] = str(item.get("post_id") or "").strip()
            if not item["post_id"]:
                raise ValueError("missing post_id")
            item["text"] = "" if item.get("text") is None else str(item["text"])
            item["created_at_utc"] = _utc(item.get("created_at_utc"))
            if pd.isna(item["created_at_utc"]):
                raise ValueError("missing/invalid created_at_utc")
            if item.get("retrieved_at_utc") not in (None, ""):
                item["retrieved_at_utc"] = _utc(item["retrieved_at_utc"])
            else:
                item["retrieved_at_utc"] = pd.NaT
            for key in ("likes", "reposts", "views", "followers"):
                value = item.get(key)
                item[key] = pd.NA if value in (None, "") else int(float(value))
            for key in ("provider", "source_id"):
                item[key] = None if item.get(key) in (None, "") else str(item[key])
            mapped.append(item)
        except (TypeError, ValueError, OverflowError) as exc:
            errors.append(f"row {index}: {exc}")

    frame = pd.DataFrame(mapped, columns=POST_COLUMNS)
    if not frame.empty:
        frame = frame.drop_duplicates(subset="post_id", keep="first").reset_index(drop=True)
        frame["created_at_utc"] = pd.to_datetime(frame["created_at_utc"], utc=True)
        frame["retrieved_at_utc"] = pd.to_datetime(frame["retrieved_at_utc"], utc=True)
        for key in ("likes", "reposts", "views", "followers"):
            frame[key] = pd.array(frame[key], dtype="Int64")
    observed = None
    if not frame.empty:
        observed = {
            "start_utc": frame["created_at_utc"].min().isoformat(),
            "end_utc": frame["created_at_utc"].max().isoformat(),
        }
    manifest = {
        "source": source.strip(), "dataset_id": dataset_id, "version": version,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_range_utc": None, "observed_range_utc": observed,
        "input_row_count": len(records), "row_count": len(frame), "errors": errors,
        "license_terms_note": "Source and permission are user-supplied and require verification; rights are not verified by this importer.",
        "format": fmt,
    }
    return frame, manifest


def load_posts_csv(
    path: str | Path,
    column_map: Mapping[str, str],
    source: str,
    *,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_bytes: int = DEFAULT_MAX_BYTES,
    dataset_id: str | None = None,
    version: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convenience CSV loader; mapping keys are normalized fields."""
    return import_posts(path, column_map, source, format="csv", max_rows=max_rows,
                        max_bytes=max_bytes, dataset_id=dataset_id, version=version)


def load_posts_jsonl(
    path: str | Path,
    column_map: Mapping[str, str],
    source: str,
    *,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_bytes: int = DEFAULT_MAX_BYTES,
    dataset_id: str | None = None,
    version: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convenience JSONL loader; mapping keys are normalized fields."""
    return import_posts(path, column_map, source, format="jsonl", max_rows=max_rows,
                        max_bytes=max_bytes, dataset_id=dataset_id, version=version)
