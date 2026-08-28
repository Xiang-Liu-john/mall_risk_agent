from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


PROJECT_DIR = Path(__file__).resolve().parents[1]
SECRETS_PATH = PROJECT_DIR / ".streamlit" / "secrets.toml"


@dataclass(frozen=True)
class RerankResult:
    index: int
    score: float
    model_name: str
    provider: str


class RerankerService(Protocol):
    provider_name: str
    model_name: str | None

    def rerank(self, query: str, documents: list[str]) -> list[RerankResult]:
        ...


def load_config() -> dict[str, str]:
    config = {key: value for key, value in os.environ.items() if key.startswith(("ARK_", "OPENAI_", "RAG_"))}
    if SECRETS_PATH.exists():
        try:
            data = tomllib.loads(SECRETS_PATH.read_text(encoding="utf-8"))
            for key, value in data.items():
                if value not in ("", None):
                    config.setdefault(str(key), str(value))
        except Exception:
            pass
    return {key: str(value) for key, value in config.items() if value not in ("", None)}


class DisabledRerankerService:
    provider_name = "disabled"
    model_name = None

    def rerank(self, query: str, documents: list[str]) -> list[RerankResult]:
        return []


def create_reranker_service(config: dict[str, str] | None = None) -> RerankerService:
    config = config or load_config()
    provider = config.get("RAG_RERANKER_PROVIDER", config.get("reranker_provider", "disabled"))
    if provider == "disabled":
        return DisabledRerankerService()
    return DisabledRerankerService()
