"""Hybrid retrieval combining semantic and lexical search with reranking.

Architecture:
1. BM25 (lexical) - catches exact keyword matches
2. Vector search (semantic) - catches semantic similarity
3. Reciprocal Rank Fusion - combines both result sets
4. Cross-encoder reranking - final quality boost

Research shows this combination improves recall by 15-30% over single methods.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .bm25 import BM25Config, BM25Index
from .embeddings import EmbeddingConfig, EmbeddingService
from .reranker import Reranker, RerankerConfig
from .vectorstore import Document, VectorStore, VectorStoreConfig


@dataclass
class RetrievalResult:
    """Result from hybrid retrieval."""

    document: Document
    score: float
    bm25_rank: int | None = None
    semantic_rank: int | None = None
    reranker_score: float | None = None

    @property
    def id(self) -> str:
        return self.document.id

    @property
    def content(self) -> str:
        return self.document.content

    @property
    def metadata(self) -> dict[str, Any]:
        return self.document.metadata


@dataclass
class HybridConfig:
    """Configuration for hybrid retrieval."""

    # Base directory for persistence
    persist_directory: str = ".retrieval"

    # Embedding model configuration
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)

    # Vector store configuration
    vectorstore: VectorStoreConfig = field(default_factory=VectorStoreConfig)

    # BM25 configuration
    bm25: BM25Config = field(default_factory=BM25Config)

    # Reranker configuration (None to disable)
    reranker: RerankerConfig | None = field(default_factory=RerankerConfig)

    # Fusion parameters
    rrf_k: int = 60  # RRF constant (60 is standard)
    semantic_weight: float = 0.5  # Weight for semantic vs BM25 (0.5 = equal)

    # Retrieval parameters
    initial_k: int = 50  # Candidates to retrieve before fusion
    rerank_k: int = 20  # Candidates to rerank
    final_k: int = 10  # Final results to return

    # Whether to use reranking (can be expensive)
    use_reranker: bool = True

    def __post_init__(self):
        """Set up derived paths."""
        base = Path(self.persist_directory)

        # Update component paths
        self.vectorstore.persist_directory = str(base / "chroma")
        self.bm25.persist_path = str(base / "bm25_index.pkl")


class HybridRetriever:
    """Hybrid retrieval system combining semantic and lexical search.

    Usage:
        retriever = HybridRetriever()

        # Add documents
        retriever.add_texts(["doc1 content", "doc2 content"])

        # Search
        results = retriever.search("my query")
        for result in results:
            print(result.content, result.score)
    """

    def __init__(self, config: HybridConfig | None = None):
        """Initialize hybrid retriever.

        Args:
            config: Configuration (uses defaults if not provided)
        """
        self.config = config or HybridConfig()

        # Initialize components
        self._embedding_service = EmbeddingService(self.config.embedding)
        self._vector_store = VectorStore(
            self._embedding_service,
            self.config.vectorstore,
        )
        self._bm25_index = BM25Index(self.config.bm25)

        # Lazy-load reranker
        self._reranker: Reranker | None = None

    def _get_reranker(self) -> Reranker | None:
        """Get reranker (lazy initialization)."""
        if not self.config.use_reranker or self.config.reranker is None:
            return None

        if self._reranker is None:
            self._reranker = Reranker(self.config.reranker)

        return self._reranker

    def add(self, documents: list[Document]) -> list[str]:
        """Add documents to both indices.

        Args:
            documents: Documents to add

        Returns:
            List of document IDs
        """
        if not documents:
            return []

        # Add to vector store (handles embedding)
        ids = self._vector_store.add(documents)

        # Add to BM25 index
        self._bm25_index.add(documents)

        return ids

    def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Add texts to both indices.

        Args:
            texts: Text contents
            metadatas: Optional metadata per text
            ids: Optional IDs

        Returns:
            List of document IDs
        """
        if not texts:
            return []

        # Create documents
        documents = []
        for i, text in enumerate(texts):
            doc_id = ids[i] if ids else None
            metadata = metadatas[i] if metadatas else {}
            documents.append(Document.create(text, metadata, doc_id))

        return self.add(documents)

    def search(
        self,
        query: str,
        k: int | None = None,
        filter: dict[str, Any] | None = None,
        use_reranker: bool | None = None,
    ) -> list[RetrievalResult]:
        """Search using hybrid retrieval.

        Pipeline:
        1. Get top-k candidates from BM25
        2. Get top-k candidates from vector search
        3. Fuse results using Reciprocal Rank Fusion
        4. Optionally rerank top candidates with cross-encoder

        Args:
            query: Search query
            k: Number of results (default: config.final_k)
            filter: Optional metadata filter for vector search
            use_reranker: Override config.use_reranker

        Returns:
            List of RetrievalResult sorted by relevance
        """
        k = k or self.config.final_k
        should_rerank = use_reranker if use_reranker is not None else self.config.use_reranker

        # Get candidates from both indices
        initial_k = self.config.initial_k

        # BM25 search
        bm25_results = self._bm25_index.search(query, k=initial_k)
        bm25_docs = {doc.id: (doc, score, rank) for rank, (doc, score) in enumerate(bm25_results)}

        # Semantic search
        semantic_results = self._vector_store.search(query, k=initial_k, filter=filter)
        semantic_docs = {doc.id: (doc, score, rank) for rank, (doc, score) in enumerate(semantic_results)}

        # Reciprocal Rank Fusion
        fused_scores = self._reciprocal_rank_fusion(
            bm25_ranks={doc_id: rank for doc_id, (_, _, rank) in bm25_docs.items()},
            semantic_ranks={doc_id: rank for doc_id, (_, _, rank) in semantic_docs.items()},
        )

        # Merge documents
        all_docs = {}
        for doc_id, (doc, score, rank) in bm25_docs.items():
            all_docs[doc_id] = {
                "document": doc,
                "bm25_rank": rank,
                "bm25_score": score,
            }
        for doc_id, (doc, score, rank) in semantic_docs.items():
            if doc_id in all_docs:
                all_docs[doc_id]["semantic_rank"] = rank
                all_docs[doc_id]["semantic_score"] = score
            else:
                all_docs[doc_id] = {
                    "document": doc,
                    "semantic_rank": rank,
                    "semantic_score": score,
                }

        # Create initial results
        results = []
        for doc_id, data in all_docs.items():
            result = RetrievalResult(
                document=data["document"],
                score=fused_scores.get(doc_id, 0.0),
                bm25_rank=data.get("bm25_rank"),
                semantic_rank=data.get("semantic_rank"),
            )
            results.append(result)

        # Sort by fused score
        results.sort(key=lambda x: -x.score)

        # Rerank if enabled
        if should_rerank and len(results) > 0:
            reranker = self._get_reranker()
            if reranker is not None:
                rerank_k = min(self.config.rerank_k, len(results))
                candidates = results[:rerank_k]

                # Rerank candidates
                reranked = reranker.rerank(
                    query,
                    [r.document for r in candidates],
                    top_k=k,
                )

                # Update results with reranker scores
                reranked_results = []
                candidates_by_id = {r.document.id: r for r in candidates}
                for doc, rerank_score in reranked:
                    orig = candidates_by_id.get(doc.id)
                    if orig:
                        reranked_results.append(RetrievalResult(
                            document=doc,
                            score=rerank_score,
                            bm25_rank=orig.bm25_rank,
                            semantic_rank=orig.semantic_rank,
                            reranker_score=rerank_score,
                        ))

                results = reranked_results

        return results[:k]

    def _reciprocal_rank_fusion(
        self,
        bm25_ranks: dict[str, int],
        semantic_ranks: dict[str, int],
    ) -> dict[str, float]:
        """Combine rankings using Reciprocal Rank Fusion.

        RRF score = sum(1 / (k + rank))

        Args:
            bm25_ranks: doc_id -> rank from BM25
            semantic_ranks: doc_id -> rank from semantic search

        Returns:
            doc_id -> fused score
        """
        k = self.config.rrf_k
        semantic_weight = self.config.semantic_weight
        bm25_weight = 1.0 - semantic_weight

        fused = {}

        # Add BM25 contributions
        for doc_id, rank in bm25_ranks.items():
            score = bm25_weight / (k + rank + 1)
            fused[doc_id] = fused.get(doc_id, 0.0) + score

        # Add semantic contributions
        for doc_id, rank in semantic_ranks.items():
            score = semantic_weight / (k + rank + 1)
            fused[doc_id] = fused.get(doc_id, 0.0) + score

        return fused

    def search_semantic_only(
        self,
        query: str,
        k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Search using only semantic similarity.

        Args:
            query: Search query
            k: Number of results
            filter: Optional metadata filter

        Returns:
            List of RetrievalResult
        """
        results = self._vector_store.search(query, k=k, filter=filter)
        return [
            RetrievalResult(
                document=doc,
                score=score,
                semantic_rank=i,
            )
            for i, (doc, score) in enumerate(results)
        ]

    def search_bm25_only(
        self,
        query: str,
        k: int = 10,
    ) -> list[RetrievalResult]:
        """Search using only BM25 lexical matching.

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of RetrievalResult
        """
        results = self._bm25_index.search(query, k=k)
        return [
            RetrievalResult(
                document=doc,
                score=score,
                bm25_rank=i,
            )
            for i, (doc, score) in enumerate(results)
        ]

    def delete(self, ids: list[str]) -> None:
        """Delete documents from both indices.

        Args:
            ids: Document IDs to delete
        """
        self._vector_store.delete(ids)
        self._bm25_index.delete(ids)

    def clear(self) -> None:
        """Clear all documents from both indices."""
        self._vector_store.clear()
        self._bm25_index.clear()

    def count(self) -> int:
        """Get number of indexed documents."""
        return self._vector_store.count()

    def get(self, ids: list[str]) -> list[Document]:
        """Get documents by ID.

        Args:
            ids: Document IDs

        Returns:
            List of documents
        """
        return self._vector_store.get(ids)

    def stats(self) -> dict:
        """Get retriever statistics."""
        return {
            "document_count": self.count(),
            "embedding": self._embedding_service.cache_stats(),
            "vectorstore": self._vector_store.stats(),
            "bm25": self._bm25_index.stats(),
            "reranker": self._reranker.stats() if self._reranker else None,
            "config": {
                "semantic_weight": self.config.semantic_weight,
                "rrf_k": self.config.rrf_k,
                "initial_k": self.config.initial_k,
                "rerank_k": self.config.rerank_k,
                "use_reranker": self.config.use_reranker,
            },
        }


# Convenience function for quick setup
def create_retriever(
    persist_directory: str = ".retrieval",
    embedding_model: str = "BAAI/bge-large-en-v1.5",
    reranker_model: str = "BAAI/bge-reranker-large",
    use_reranker: bool = True,
) -> HybridRetriever:
    """Create a hybrid retriever with sensible defaults.

    Args:
        persist_directory: Where to store indices
        embedding_model: Embedding model name
        reranker_model: Reranker model name
        use_reranker: Whether to use reranking

    Returns:
        Configured HybridRetriever
    """
    config = HybridConfig(
        persist_directory=persist_directory,
        embedding=EmbeddingConfig(model_name=embedding_model),
        reranker=RerankerConfig(model_name=reranker_model) if use_reranker else None,
        use_reranker=use_reranker,
    )
    return HybridRetriever(config)
