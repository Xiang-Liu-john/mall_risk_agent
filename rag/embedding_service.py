from __future__ import annotations

import hashlib
import math
import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


PROJECT_DIR = Path(__file__).resolve().parents[1]
SECRETS_PATH = PROJECT_DIR / ".streamlit" / "secrets.toml"
HASHING_VECTOR_SIZE = 96


@dataclass(frozen=True)
class EmbeddingResult:
    vector: list[float]
    model_name: str
    provider: str
    dimensions: int
    token_usage: dict | None = None
    production_ready: bool = True


class EmbeddingService(Protocol):
    provider_name: str
    model_name: str | None

    def embed_text(self, text: str) -> EmbeddingResult:
        ...

    def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        ...


def load_config() -> dict[str, str]:
    config = {key: value for key, value in os.environ.items() if key.startswith(("ARK_", "OPENAI_", "RAG_"))}
    if SECRETS_PATH.exists():
        try:
            secret_data = tomllib.loads(SECRETS_PATH.read_text(encoding="utf-8"))
            for key, value in secret_data.items():
                if value not in ("", None):
                    config.setdefault(str(key), str(value))
        except Exception:
            pass
    return {key: str(value) for key, value in config.items() if value not in ("", None)}


def tokenize(text: str) -> list[str]:
    normalized = str(text).lower()
    tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", normalized)
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]", normalized))
    tokens.extend(chinese[i : i + 2] for i in range(max(0, len(chinese) - 1)))
    return [token for token in tokens if token.strip()]


class HashingEmbeddingService:
    provider_name = "hashing_fallback"
    model_name = "local-token-hashing-96d"

    def embed_text(self, text: str) -> EmbeddingResult:
        vector = [0.0] * HASHING_VECTOR_SIZE
        for token in tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % HASHING_VECTOR_SIZE
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return EmbeddingResult(
            vector=vector,
            model_name=self.model_name,
            provider=self.provider_name,
            dimensions=HASHING_VECTOR_SIZE,
            production_ready=False,
        )

    def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        return [self.embed_text(text) for text in texts]


class OpenAICompatibleEmbeddingService:
    provider_name = "ark_openai_compatible"

    def __init__(self, api_key: str, base_url: str | None, model_name: str) -> None:
        if not model_name:
            raise ValueError("Embedding model must come from ARK_EMBEDDING_MODEL or OPENAI_EMBEDDING_MODEL.")
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name

    def _client(self):
        from openai import OpenAI

        if self.base_url:
            return OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=45)
        return OpenAI(api_key=self.api_key, timeout=45)

    def embed_text(self, text: str) -> EmbeddingResult:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        response = self._client().embeddings.create(
            model=self.model_name,
            input=texts,
            encoding_format="float",
        )
        usage = getattr(response, "usage", None)
        usage_dict = usage.model_dump() if hasattr(usage, "model_dump") else None
        results: list[EmbeddingResult] = []
        for item in response.data:
            vector = [float(value) for value in item.embedding]
            results.append(
                EmbeddingResult(
                    vector=vector,
                    model_name=self.model_name,
                    provider=self.provider_name,
                    dimensions=len(vector),
                    token_usage=usage_dict,
                    production_ready=True,
                )
            )
        return results


def create_embedding_service(config: dict[str, str] | None = None) -> EmbeddingService:
    config = config or load_config()
    provider = config.get("RAG_EMBEDDING_PROVIDER", config.get("embedding_provider", "ark_openai_compatible"))
    if provider in {"ark", "ark_openai_compatible", "openai_compatible"}:
        api_key = config.get("ARK_API_KEY") or config.get("OPENAI_API_KEY")
        model_name = config.get("ARK_EMBEDDING_MODEL") or config.get("OPENAI_EMBEDDING_MODEL")
        base_url = config.get("ARK_BASE_URL") or config.get("OPENAI_BASE_URL")
        if api_key and model_name:
            return OpenAICompatibleEmbeddingService(api_key=api_key, base_url=base_url, model_name=model_name)
    return HashingEmbeddingService()
