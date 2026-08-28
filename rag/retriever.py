from __future__ import annotations

import json
import shutil
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from uuid import UUID

from .embedding_service import EmbeddingService, create_embedding_service, tokenize
from .reranker import RerankerService, create_reranker_service


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAG_DB_DIR = PROJECT_DIR / "rag_db"
QDRANT_PATH = RAG_DB_DIR / "qdrant"
CONFIG_PATH = Path(__file__).with_name("retrieval_config.json")


@dataclass(frozen=True)
class HybridSearchResult:
    chunk_id: str
    source: str
    title: str
    content: str
    metadata: dict[str, Any]
    dense_score: float = 0.0
    lexical_score: float = 0.0
    fusion_score: float = 0.0
    rerank_score: float | None = None


@dataclass(frozen=True)
class RetrievalDebugInfo:
    retrieval_latency_ms: int
    dense_count: int
    lexical_count: int
    merged_count: int
    reranked_count: int
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
    vector_db: str
    dense_retrieval_status: str
    config: dict[str, Any] = field(default_factory=dict)


def load_retrieval_config() -> dict[str, Any]:
    defaults = {
        "dense_weight": 0.62,
        "lexical_weight": 0.38,
        "rerank_top_n": 20,
        "final_top_k": 8,
        "dense_top_k": 40,
        "lexical_top_k": 40,
        "qdrant_collection": "mall_risk_chunks",
        "reranker_provider": "disabled",
    }
    if CONFIG_PATH.exists():
        try:
            defaults.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
        except Exception:
            pass
    total = float(defaults["dense_weight"]) + float(defaults["lexical_weight"])
    if total <= 0:
        defaults["dense_weight"], defaults["lexical_weight"] = 0.5, 0.5
    return defaults


class QdrantVectorStore:
    def __init__(self, path: Path = QDRANT_PATH, collection_name: str = "mall_risk_chunks") -> None:
        self.path = path
        self.collection_name = collection_name
        self.available = False
        self.client = None
        try:
            from qdrant_client import QdrantClient

            self.path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(self.path))
            self.available = True
        except Exception:
            self.client = None
            self.available = False

    def reset_collection(self, vector_size: int) -> None:
        if not self.available or self.client is None:
            return
        from qdrant_client.models import Distance, VectorParams

        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(collection_name=self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def upsert_chunks(self, chunks: Iterable[dict[str, Any]]) -> None:
        if not self.available or self.client is None:
            return
        from qdrant_client.models import PointStruct

        points = []
        for item in chunks:
            points.append(
                PointStruct(
                    id=str(UUID(hex=item["chunk_id"][:32])),
                    vector=item["embedding"],
                    payload={
                        "chunk_id": item["chunk_id"],
                        "content": item["content"],
                        "source": item["source"],
                        "title": item["title"],
                        "metadata": item["metadata"],
                    },
                )
            )
        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, vector: list[float], top_k: int) -> list[tuple[str, float]]:
        if not self.available or self.client is None:
            return []
        try:
            response = self.client.query_points(collection_name=self.collection_name, query=vector, limit=top_k)
            hits = response.points
        except Exception:
            return []
        return [(str(hit.payload.get("chunk_id")), float(hit.score)) for hit in hits if hit.payload]

    def rebuild(self, vector_size: int) -> None:
        if self.client is not None and hasattr(self.client, "close"):
            try:
                self.client.close()
            except Exception:
                pass
        if self.path.exists():
            shutil.rmtree(self.path)
        self.__init__(self.path, self.collection_name)
        self.reset_collection(vector_size)

    def close(self) -> None:
        if self.client is not None and hasattr(self.client, "close"):
            try:
                self.client.close()
            except Exception:
                pass


class HybridRetriever:
    def __init__(
        self,
        db_path: Path,
        embedding_service: EmbeddingService | None = None,
        reranker: RerankerService | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.db_path = db_path
        self.config = config or load_retrieval_config()
        self.embedding_service = embedding_service or create_embedding_service(self.config)
        self.reranker = reranker or create_reranker_service(self.config)
        self.vector_store = QdrantVectorStore(collection_name=str(self.config["qdrant_collection"]))

    def close(self) -> None:
        self.vector_store.close()

    def initialize_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            DROP TABLE IF EXISTS chunks_fts;
            DROP TABLE IF EXISTS chunks;

            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT NOT NULL UNIQUE,
                source TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                embedding_provider TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                embedding_dimensions INTEGER NOT NULL
            );

            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                title,
                content,
                source,
                content='chunks',
                content_rowid='id',
                tokenize='unicode61'
            );
            """
        )
        conn.commit()

    def upsert_documents(self, conn: sqlite3.Connection, docs: list[tuple[str, str, str, dict[str, Any]]]) -> None:
        if not docs:
            return
        texts = [f"{title}\n{content}" for _, title, content, _ in docs]
        embeddings = self.embedding_service.embed_texts(texts)
        vector_size = embeddings[0].dimensions
        self.vector_store.rebuild(vector_size)
        qdrant_chunks = []
        for index, ((source, title, content, metadata), embedding) in enumerate(zip(docs, embeddings), start=1):
            chunk_id = self._chunk_id(source, title, metadata, index)
            cur = conn.execute(
                """
                INSERT INTO chunks
                    (chunk_id, source, title, content, metadata_json, embedding_json,
                     embedding_provider, embedding_model, embedding_dimensions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk_id,
                    source,
                    title,
                    content,
                    json.dumps(metadata, ensure_ascii=False),
                    json.dumps(embedding.vector),
                    embedding.provider,
                    embedding.model_name,
                    embedding.dimensions,
                ),
            )
            conn.execute(
                "INSERT INTO chunks_fts(rowid, title, content, source) VALUES (?, ?, ?, ?)",
                (cur.lastrowid, title, content, source),
            )
            qdrant_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "content": content,
                    "source": source,
                    "title": title,
                    "metadata": metadata,
                    "embedding": embedding.vector,
                }
            )
        conn.commit()
        self.vector_store.upsert_chunks(qdrant_chunks)

    def search(self, query: str, top_k: int | None = None) -> tuple[list[HybridSearchResult], RetrievalDebugInfo]:
        started = time.perf_counter()
        final_top_k = int(top_k or self.config["final_top_k"])
        dense_top_k = int(self.config["dense_top_k"])
        lexical_top_k = int(self.config["lexical_top_k"])
        dense_weight = float(self.config["dense_weight"])
        lexical_weight = float(self.config["lexical_weight"])
        query_embedding = self.embedding_service.embed_text(query)
        dense_hits = (
            dict(self.vector_store.search(query_embedding.vector, dense_top_k))
            if dense_top_k > 0 and dense_weight > 0
            else {}
        )
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if not self._schema_ready(conn):
                debug = RetrievalDebugInfo(
                    retrieval_latency_ms=int((time.perf_counter() - started) * 1000),
                    dense_count=0,
                    lexical_count=0,
                    merged_count=0,
                reranked_count=0,
                embedding_provider=query_embedding.provider,
                embedding_model=query_embedding.model_name,
                embedding_dimension=query_embedding.dimensions,
                vector_db="schema-needs-rebuild",
                dense_retrieval_status="schema-needs-rebuild",
                config=self.config,
                )
                return [], debug
            lexical_hits = self._lexical_search(conn, query, lexical_top_k) if lexical_top_k > 0 and lexical_weight > 0 else {}
            chunk_ids = set(dense_hits) | set(lexical_hits)
            rows = self._fetch_chunks(conn, chunk_ids)
        dense_norm = self._normalize_scores(dense_hits)
        lexical_norm = self._normalize_scores(lexical_hits)
        weight_total = dense_weight + lexical_weight
        candidates = []
        for chunk_id, row in rows.items():
            fusion = (dense_norm.get(chunk_id, 0.0) * dense_weight + lexical_norm.get(chunk_id, 0.0) * lexical_weight) / weight_total
            candidates.append(
                HybridSearchResult(
                    chunk_id=chunk_id,
                    source=str(row["source"]),
                    title=str(row["title"]),
                    content=str(row["content"]),
                    metadata=json.loads(row["metadata_json"]),
                    dense_score=dense_norm.get(chunk_id, 0.0),
                    lexical_score=lexical_norm.get(chunk_id, 0.0),
                    fusion_score=fusion,
                )
            )
        candidates.sort(key=lambda item: item.fusion_score, reverse=True)
        reranked_count = 0
        rerank_top_n = int(self.config["rerank_top_n"])
        rerank_inputs = candidates[:rerank_top_n]
        rerank_results = self.reranker.rerank(query, [item.content for item in rerank_inputs])
        if rerank_results:
            reranked_count = len(rerank_results)
            score_by_index = {item.index: item.score for item in rerank_results}
            candidates = [
                HybridSearchResult(**{**candidate.__dict__, "rerank_score": score_by_index.get(index)})
                for index, candidate in enumerate(candidates)
            ]
            candidates.sort(
                key=lambda item: item.rerank_score if item.rerank_score is not None else item.fusion_score,
                reverse=True,
            )
        debug = RetrievalDebugInfo(
            retrieval_latency_ms=int((time.perf_counter() - started) * 1000),
            dense_count=len(dense_hits),
            lexical_count=len(lexical_hits),
            merged_count=len(candidates),
            reranked_count=reranked_count,
            embedding_provider=query_embedding.provider,
            embedding_model=query_embedding.model_name,
            embedding_dimension=query_embedding.dimensions,
            vector_db="qdrant-local" if self.vector_store.available else "qdrant-unavailable",
            dense_retrieval_status=(
                "ok"
                if dense_hits
                else ("qdrant-unavailable" if not self.vector_store.available else "no-dense-hits")
            ),
            config=self.config,
        )
        return candidates[:final_top_k], debug

    def _lexical_search(self, conn: sqlite3.Connection, query: str, limit: int) -> dict[str, float]:
        terms = " OR ".join(tokenize(query)[:8])
        if not terms:
            return {}
        try:
            rows = conn.execute(
                """
                SELECT chunks.chunk_id, bm25(chunks_fts) AS rank
                FROM chunks_fts
                JOIN chunks ON chunks.id = chunks_fts.rowid
                WHERE chunks_fts MATCH ?
                LIMIT ?
                """,
                (terms, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return {}
        return {str(row["chunk_id"]): 1 / (1 + abs(float(row["rank"]))) for row in rows}

    def _fetch_chunks(self, conn: sqlite3.Connection, chunk_ids: set[str]) -> dict[str, sqlite3.Row]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = conn.execute(
            f"SELECT chunk_id, source, title, content, metadata_json FROM chunks WHERE chunk_id IN ({placeholders})",
            tuple(chunk_ids),
        ).fetchall()
        return {str(row["chunk_id"]): row for row in rows}

    @staticmethod
    def _schema_ready(conn: sqlite3.Connection) -> bool:
        try:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()}
        except sqlite3.OperationalError:
            return False
        return {"chunk_id", "embedding_provider", "embedding_model", "embedding_dimensions"}.issubset(columns)

    @staticmethod
    def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        values = list(scores.values())
        min_score, max_score = min(values), max(values)
        if max_score == min_score:
            return {key: 1.0 for key in scores}
        return {key: (value - min_score) / (max_score - min_score) for key, value in scores.items()}

    @staticmethod
    def _chunk_id(source: str, title: str, metadata: dict[str, Any], index: int) -> str:
        raw = json.dumps({"source": source, "title": title, "metadata": metadata, "index": index}, ensure_ascii=False)
        return hashlib_sha256(raw)


def hashlib_sha256(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()
