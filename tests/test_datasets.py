import json

import pandas as pd
import pytest

from cwt_agent.datasets import load_posts_csv, load_posts_jsonl

MAP = {"post_id": "id", "created_at_utc": "created", "text": "body", "likes": "fav"}


def test_csv_mapping_utc_dedup_nullable_and_manifest(tmp_path):
    path = tmp_path / "posts.csv"
    path.write_text(
        "id,created,body,fav\n"
        "a,2024-01-01T01:00:00+01:00,hello,3\n"
        "a,2024-01-02T00:00:00Z,duplicate,9\n"
        "b,2024-01-01T00:00:00Z,world,\n",
        encoding="utf-8",
    )
    posts, manifest = load_posts_csv(path, MAP, "user-provided-export")
    assert list(posts.columns) == ["post_id", "created_at_utc", "text", "likes", "reposts", "views", "followers", "provider", "source_id", "retrieved_at_utc"]
    assert posts.post_id.tolist() == ["a", "b"]
    assert posts.created_at_utc.iloc[0] == pd.Timestamp("2024-01-01T00:00:00Z")
    assert str(posts.created_at_utc.dt.tz) == "UTC"
    assert pd.isna(posts.likes.iloc[1])
    assert manifest["input_row_count"] == 3
    assert manifest["row_count"] == 2
    assert "user-supplied" in manifest["license_terms_note"]
    assert manifest["observed_range_utc"]["start_utc"].startswith("2024-01-01")


def test_jsonl_explicit_mapping_and_invalid_row_reported(tmp_path):
    path = tmp_path / "posts.jsonl"
    path.write_text("\n".join([
        json.dumps({"key": 1, "at": "2024-03-01T12:00:00-05:00", "content": "ok"}),
        json.dumps({"key": 2, "at": "not-a-date", "content": "bad"}),
    ]), encoding="utf-8")
    posts, manifest = load_posts_jsonl(path, {"post_id": "key", "created_at_utc": "at", "text": "content"}, "manual")
    assert posts.post_id.tolist() == ["1"]
    assert posts.created_at_utc.iloc[0] == pd.Timestamp("2024-03-01T17:00:00Z")
    assert len(manifest["errors"]) == 1


def test_row_and_byte_limits(tmp_path):
    path = tmp_path / "posts.csv"
    path.write_text("id,created,body\na,2024-01-01T00:00:00Z,x\nb,2024-01-02T00:00:00Z,y\n")
    with pytest.raises(ValueError, match="max_rows"):
        load_posts_csv(path, {"post_id": "id", "created_at_utc": "created", "text": "body"}, "manual", max_rows=1)
    with pytest.raises(ValueError, match="max_bytes"):
        load_posts_csv(path, {"post_id": "id", "created_at_utc": "created", "text": "body"}, "manual", max_bytes=1)


def test_required_mapping_is_explicit(tmp_path):
    with pytest.raises(ValueError, match="column_map"):
        load_posts_csv(tmp_path / "missing.csv", {"post_id": "id"}, "manual")
