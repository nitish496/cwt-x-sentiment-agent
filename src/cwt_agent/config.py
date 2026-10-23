"""Local settings. Secrets are loaded from environment and never rendered in logs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    apify_api_token: str | None
    apify_actor_id: str | None
    tavily_api_key: str | None
    exa_api_key: str | None
    max_items: int

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> Settings:
        if env_file is not None:
            load_dotenv(env_file, override=False)
        else:
            load_dotenv(override=False)
        try:
            max_items = int(os.getenv("APIFY_MAX_ITEMS", "5000"))
        except ValueError as exc:
            raise ValueError("APIFY_MAX_ITEMS must be an integer") from exc
        if max_items < 1 or max_items > 100_000:
            raise ValueError("APIFY_MAX_ITEMS must be between 1 and 100000")
        return cls(
            apify_api_token=os.getenv("APIFY_API_TOKEN") or None,
            apify_actor_id=os.getenv("APIFY_ACTOR_ID") or None,
            tavily_api_key=os.getenv("TAVILY_API_KEY") or None,
            exa_api_key=os.getenv("EXA_API_KEY") or None,
            max_items=max_items,
        )

    def missing_for_live_ingestion(self) -> list[str]:
        required = {
            "APIFY_API_TOKEN": self.apify_api_token,
            "APIFY_ACTOR_ID": self.apify_actor_id,
        }
        return [name for name, value in required.items() if not value]
