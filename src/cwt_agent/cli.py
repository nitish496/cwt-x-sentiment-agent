"""Command line entry points for the CWT research workflow."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Settings
from .datasets import load_posts_csv, load_posts_jsonl
from .features import build_daily_features, build_supervised_frame
from .modeling import expanding_walk_forward_predictions, forecast_next_horizons
from .providers import fetch_coinbase_daily, fetch_exa_news, fetch_tavily_news, run_apify_posts
from .storage import SQLiteStore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = PROJECT_ROOT / "data" / "processed" / "cwt_agent.sqlite3"
DEFAULT_REPORTS = PROJECT_ROOT / "reports" / "generated"


def _utc_range(start: str, end: str) -> tuple[datetime, datetime]:
    try:
        first = date.fromisoformat(start)
        last = date.fromisoformat(end)
    except ValueError as exc:
        raise ValueError("dates must use YYYY-MM-DD") from exc
    if first > last:
        raise ValueError("--from must be on or before --to")
    return datetime.combine(first, time.min, timezone.utc), datetime.combine(last, time.min, timezone.utc)


def _read_actor_input(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read Actor input JSON at {path}") from exc
    if not isinstance(value, dict):
        raise TypeError("Apify Actor input must be a JSON object")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read JSON mapping at {path}") from exc
    if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
        raise ValueError("column mapping must be a JSON object of string field names")
    return value


def _frame_from_db(path: Path, table: str) -> pd.DataFrame:
    import sqlite3

    if not path.exists():
        return pd.DataFrame()
    if table not in {"prices", "posts", "news"}:
        raise ValueError("unsupported local table")
    with sqlite3.connect(path) as connection:
        return pd.read_sql_query(f"SELECT * FROM {table}", connection)


def _save_posts(frame: pd.DataFrame, db: Path, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    SQLiteStore(db).upsert_posts(frame)


def _save_prices(frame: pd.DataFrame, db: Path, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    SQLiteStore(db).upsert_prices(frame)


def _cmd_config_check(args: argparse.Namespace) -> int:
    settings = Settings.from_env(Path(args.env_file) if args.env_file else None)
    status = {
        "apify": "configured" if settings.apify_api_token and settings.apify_actor_id else "missing",
        "tavily": "configured" if settings.tavily_api_key else "missing",
        "exa": "configured" if settings.exa_api_key else "optional/missing",
        "max_items": settings.max_items,
        "secrets_printed": False,
    }
    print(json.dumps(status, indent=2))
    return 0


def _cmd_prices(args: argparse.Namespace) -> int:
    start, end = _utc_range(args.start, args.end)
    frame = fetch_coinbase_daily(args.symbol, start, end)
    _save_prices(frame, Path(args.db), Path(args.output))
    print(f"Saved {len(frame)} Coinbase daily candles to {args.output}; no API key required.")
    if frame.empty:
        print("The provider returned no bars for this range.")
    return 0


def _cmd_ingest_x(args: argparse.Namespace) -> int:
    if not args.live:
        raise ValueError("X ingestion requires explicit --live acknowledgement")
    settings = Settings.from_env(Path(args.env_file) if args.env_file else None)
    missing = settings.missing_for_live_ingestion()
    if missing:
        raise ValueError(f"X ingestion unavailable; configure locally: {', '.join(missing)}")
    start, end = _utc_range(args.start, args.end)
    actor_input = _read_actor_input(Path(args.actor_input))
    mapping = _read_json(Path(args.field_map)) if args.field_map else None
    frame = run_apify_posts(
        settings.apify_actor_id or "", settings.apify_api_token or "", actor_input,
        start, end, min(args.max_items or settings.max_items, settings.max_items), field_map=mapping,
    )
    _save_posts(frame, Path(args.db), Path(args.output))
    print(f"Saved {len(frame)} posts from the configured Apify Actor.")
    return 0


def _cmd_import_posts(args: argparse.Namespace) -> int:
    mapping = _read_json(Path(args.column_map))
    loader = load_posts_jsonl if args.input.lower().endswith(".jsonl") else load_posts_csv
    frame, manifest = loader(
        args.input, mapping, args.source, max_rows=args.max_rows,
        dataset_id=args.dataset_id, version=args.version,
    )
    db = Path(args.db)
    _save_posts(frame, db, Path(args.output))
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Imported {len(frame)} posts; recorded {len(manifest['errors'])} invalid rows in the manifest.")
    print("Confirm source permission and platform before using the data in an assignment report.")
    return 0


def _cmd_news(args: argparse.Namespace) -> int:
    if not args.live:
        raise ValueError("Current-news retrieval requires explicit --live acknowledgement")
    settings = Settings.from_env(Path(args.env_file) if args.env_file else None)
    if not settings.tavily_api_key:
        raise ValueError("Current-news retrieval unavailable; configure TAVILY_API_KEY locally")
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)
    primary = fetch_tavily_news(settings.tavily_api_key, args.query, start, end)
    frames = [primary]
    if settings.exa_api_key and args.with_exa:
        frames.append(fetch_exa_news(settings.exa_api_key, args.query, start, end))
    combined = pd.concat(frames, ignore_index=True).drop_duplicates("url") if frames else pd.DataFrame()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    if not combined.empty:
        SQLiteStore(args.db).upsert_news(combined)
    print(f"Saved {len(combined)} current-news results. They are for the outlook only, not historical model features.")
    return 0


def _cmd_outlook(args: argparse.Namespace) -> int:
    from .reporting import write_outlook

    db = Path(args.db)
    prices = pd.read_csv(args.prices) if args.prices else _frame_from_db(db, "prices")
    posts = pd.read_csv(args.posts) if args.posts else _frame_from_db(db, "posts")
    news = pd.read_csv(args.news) if args.news else _frame_from_db(db, "news")
    forecasts = pd.read_csv(args.forecasts) if args.forecasts else None
    output = write_outlook(args.output, symbol=args.symbol, prices=prices, posts=posts, news=news,
                           forecasts=forecasts, data_label=args.data_label)
    print(f"Wrote outlook: {output}")
    return 0


def _cmd_forecast(args: argparse.Namespace) -> int:
    prices = pd.read_csv(args.prices)
    posts = pd.read_csv(args.posts)
    daily = build_daily_features(prices, posts)
    supervised = build_supervised_frame(daily)
    forecasts = forecast_next_horizons(supervised, min_train_size=args.min_train)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(output, index=False)
    print(f"Forecast n+1 through n+5 from the latest completed close; saved {output}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from .reporting import write_evaluation

    price_path, post_path = Path(args.prices), Path(args.posts)
    prices, posts = pd.read_csv(price_path), pd.read_csv(post_path)
    if args.data_label == "synthetic fixture" and not args.allow_synthetic:
        raise ValueError("synthetic data is test-only; pass --allow-synthetic to label this diagnostic report")
    daily = build_daily_features(prices, posts)
    supervised = build_supervised_frame(daily)
    predictions = expanding_walk_forward_predictions(
        supervised, n_splits=args.splits, test_size=args.test_size,
        min_train_size=args.min_train, purge_bars=5,
    )
    output_dir = Path(args.output_dir)
    report, chart = write_evaluation(
        output_dir, predictions, source_label=args.data_label,
        coverage={
            "price_rows": len(prices), "post_rows": len(posts),
            "price_start": str(prices.timestamp_utc.min()), "price_end": str(prices.timestamp_utc.max()),
            "synthetic": args.data_label == "synthetic fixture",
        },
        daily_features=daily,
    )
    predictions.to_csv(output_dir / "walk_forward_predictions.csv", index=False)
    print(f"Wrote {len(predictions)} out-of-sample origin/horizon predictions, {report}, and {chart}.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cwt-agent", description="CWT X sentiment and BTC research pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    config = commands.add_parser("config-check", help="report whether local integrations are configured (never print secrets)")
    config.add_argument("--env-file")
    config.set_defaults(func=_cmd_config_check)

    prices = commands.add_parser("prices", help="fetch public Coinbase daily candles")
    prices.add_argument("--from", dest="start", required=True)
    prices.add_argument("--to", dest="end", required=True)
    prices.add_argument("--symbol", default="BTC-USD")
    prices.add_argument("--db", default=str(DEFAULT_DB))
    prices.add_argument("--output", default=str(PROJECT_ROOT / "data/processed/prices.csv"))
    prices.set_defaults(func=_cmd_prices)

    x = commands.add_parser("ingest-x", help="run the reviewed Apify Actor and import posts")
    x.add_argument("--live", action="store_true", help="acknowledge a paid/external live data request")
    x.add_argument("--from", dest="start", required=True)
    x.add_argument("--to", dest="end", required=True)
    x.add_argument("--actor-input", required=True)
    x.add_argument("--field-map")
    x.add_argument("--max-items", type=int)
    x.add_argument("--env-file")
    x.add_argument("--db", default=str(DEFAULT_DB))
    x.add_argument("--output", default=str(PROJECT_ROOT / "data/processed/posts.csv"))
    x.set_defaults(func=_cmd_ingest_x)

    imp = commands.add_parser("import-posts", help="import a locally supplied CSV or JSONL with explicit mapping")
    imp.add_argument("input")
    imp.add_argument("--column-map", required=True)
    imp.add_argument("--source", required=True)
    imp.add_argument("--dataset-id")
    imp.add_argument("--version")
    imp.add_argument("--max-rows", type=int, default=1_000_000)
    imp.add_argument("--db", default=str(DEFAULT_DB))
    imp.add_argument("--output", default=str(PROJECT_ROOT / "data/processed/posts.csv"))
    imp.add_argument("--manifest", default=str(PROJECT_ROOT / "reports/generated/import-manifest.json"))
    imp.set_defaults(func=_cmd_import_posts)

    news = commands.add_parser("news", help="fetch recent news for the current outlook")
    news.add_argument("--live", action="store_true")
    news.add_argument("--query", default="Bitcoin BTC cryptocurrency")
    news.add_argument("--days", type=int, default=7)
    news.add_argument("--with-exa", action="store_true")
    news.add_argument("--env-file")
    news.add_argument("--db", default=str(DEFAULT_DB))
    news.add_argument("--output", default=str(PROJECT_ROOT / "data/processed/current_news.csv"))
    news.set_defaults(func=_cmd_news)

    outlook = commands.add_parser("outlook", help="write a one-page outlook with explicit unavailable sections")
    outlook.add_argument("--symbol", default="BTC-USD")
    outlook.add_argument("--db", default=str(DEFAULT_DB))
    outlook.add_argument("--prices")
    outlook.add_argument("--posts")
    outlook.add_argument("--news")
    outlook.add_argument("--forecasts")
    outlook.add_argument("--data-label", default="live")
    outlook.add_argument("--output", default=str(DEFAULT_REPORTS / "current_outlook.md"))
    outlook.set_defaults(func=_cmd_outlook)

    forecast = commands.add_parser("forecast", help="fit on fully observed history and forecast n+1 through n+5")
    forecast.add_argument("--prices", required=True)
    forecast.add_argument("--posts", required=True)
    forecast.add_argument("--min-train", type=int, default=60)
    forecast.add_argument("--output", default=str(DEFAULT_REPORTS / "current_forecasts.csv"))
    forecast.set_defaults(func=_cmd_forecast)

    evaluate = commands.add_parser("evaluate", help="run expanding walk-forward validation on supplied data")
    evaluate.add_argument("--prices", required=True)
    evaluate.add_argument("--posts", required=True)
    evaluate.add_argument("--data-label", required=True, choices=["historical", "live", "synthetic fixture"])
    evaluate.add_argument("--allow-synthetic", action="store_true")
    evaluate.add_argument("--splits", type=int, default=4)
    evaluate.add_argument("--test-size", type=int, default=21)
    evaluate.add_argument("--min-train", type=int, default=60)
    evaluate.add_argument("--output-dir", default=str(DEFAULT_REPORTS))
    evaluate.set_defaults(func=_cmd_evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
