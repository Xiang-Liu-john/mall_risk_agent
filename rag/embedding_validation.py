from __future__ import annotations

import hashlib
from typing import Any

from .embedding_service import create_embedding_service, load_config
from .retriever import QdrantVectorStore


def validate_semantic_embedding() -> dict[str, Any]:
    config = load_config()
    ark_api_key = config.get("ARK_API_KEY")
    ark_base_url = config.get("ARK_BASE_URL")
    ark_embedding_model = config.get("ARK_EMBEDDING_MODEL")

    result: dict[str, Any] = {
        "embedding_provider": "ark_openai_compatible",
        "embedding_model": ark_embedding_model,
        "embedding_dimension": None,
        "ark_base_url": ark_base_url,
        "api_call_succeeded": False,
        "qdrant_write_succeeded": False,
        "qdrant_similarity_search_succeeded": False,
        "vector_db_status": "not-tested",
        "dense_retrieval_status": "not-tested",
        "production_ready": False,
        "reason": None,
    }

    if not ark_api_key:
        result["reason"] = "ARK_API_KEY is not configured."
        return result
    if not ark_embedding_model:
        result["reason"] = "ARK_EMBEDDING_MODEL is not configured; no model name was guessed."
        return result

    try:
        service = create_embedding_service(config)
        embedding = service.embed_text("购物中心经营风险 semantic embedding 验证")
        result.update(
            {
                "embedding_provider": embedding.provider,
                "embedding_model": embedding.model_name,
                "embedding_dimension": embedding.dimensions,
                "api_call_succeeded": True,
            }
        )
    except Exception as exc:
        result["reason"] = f"Embedding API call failed: {exc}"
        return result

    try:
        store = QdrantVectorStore(collection_name="mall_risk_embedding_validation")
        result["vector_db_status"] = "qdrant-local" if store.available else "qdrant-unavailable"
        if not store.available:
            result["reason"] = "Qdrant local client is unavailable."
            return result
        try:
            store.rebuild(int(result["embedding_dimension"]))
            chunk_id = hashlib.sha256(b"mall_risk_embedding_validation").hexdigest()
            store.upsert_chunks(
                [
                    {
                        "chunk_id": chunk_id,
                        "content": "购物中心经营风险 semantic embedding 验证",
                        "source": "embedding_validation",
                        "title": "semantic validation probe",
                        "metadata": {"validation": True},
                        "embedding": embedding.vector,
                    }
                ]
            )
            result["qdrant_write_succeeded"] = True
            hits = store.search(embedding.vector, top_k=1)
        finally:
            store.close()
        result["qdrant_similarity_search_succeeded"] = bool(hits)
        result["dense_retrieval_status"] = "ok" if hits else "no-dense-hits"
    except Exception as exc:
        result["reason"] = f"Qdrant validation failed: {exc}"
        return result

    result["production_ready"] = bool(
        result["api_call_succeeded"]
        and result["qdrant_write_succeeded"]
        and result["qdrant_similarity_search_succeeded"]
    )
    if not result["production_ready"] and result["reason"] is None:
        result["reason"] = "Semantic embedding validation did not complete all required checks."
    return result
