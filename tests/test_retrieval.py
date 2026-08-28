from rag.embedding_service import HashingEmbeddingService
from rag.retriever import HybridRetriever
from rag.reranker import DisabledRerankerService


def test_hashing_embedding_is_normalized_fallback():
    result = HashingEmbeddingService().embed_text("欠费 风险 门店")
    assert result.dimensions == 96
    assert result.production_ready is False
    assert abs(sum(value * value for value in result.vector) - 1.0) < 1e-6


def test_score_normalization_keeps_equal_scores_at_one(tmp_path):
    retriever = HybridRetriever(tmp_path / "rag.sqlite3", embedding_service=HashingEmbeddingService(), reranker=DisabledRerankerService())
    assert retriever._normalize_scores({"a": 2.0, "b": 2.0}) == {"a": 1.0, "b": 1.0}


def test_score_normalization_scales_range(tmp_path):
    retriever = HybridRetriever(tmp_path / "rag.sqlite3", embedding_service=HashingEmbeddingService(), reranker=DisabledRerankerService())
    assert retriever._normalize_scores({"a": 2.0, "b": 4.0}) == {"a": 0.0, "b": 1.0}
