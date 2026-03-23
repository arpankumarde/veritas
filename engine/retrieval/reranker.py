"""Cross-encoder reranker for high-quality final ranking.

Cross-encoders process query-document pairs together, providing more accurate
relevance scores than bi-encoders at the cost of speed. Used as a final
reranking step on top-k candidates from hybrid retrieval.

Research shows reranking improves retrieval quality by 10-20%.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class RerankerConfig:
    """Configuration for the reranker."""

    # Model selection - quality vs speed tradeoff
    # Best quality: "BAAI/bge-reranker-v2-m3" (multi-lingual, highest quality)
    # Good quality: "BAAI/bge-reranker-large" (English, very good)
    # Fast: "BAAI/bge-reranker-base" (English, good balance)
    # Very fast: "cross-encoder/ms-marco-MiniLM-L-6-v2" (fastest)
    model_name: str = "BAAI/bge-reranker-large"

    # Device selection
    device: str = "auto"  # "auto", "cuda", "mps", "cpu"

    # Batch size for reranking
    batch_size: int = 16

    # Max sequence length for query + document
    max_length: int = 512

    # Score normalization
    normalize_scores: bool = True


class Reranker:
    """Cross-encoder reranker for final relevance scoring."""

    def __init__(self, config: RerankerConfig | None = None):
        self.config = config or RerankerConfig()
        self._model = None

    def _get_device(self) -> str:
        """Determine best available device."""
        if self.config.device != "auto":
            return self.config.device

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _load_model(self):
        """Lazy load the reranker model."""
        if self._model is not None:
            return

        from sentence_transformers import CrossEncoder

        device = self._get_device()

        self._model = CrossEncoder(
            self.config.model_name,
            max_length=self.config.max_length,
            device=device,
        )

    def rerank(
        self,
        query: str,
        documents: list["Document"],
        top_k: int | None = None,
    ) -> list[tuple["Document", float]]:
        """Rerank documents by relevance to query.

        Args:
            query: Search query
            documents: Documents to rerank
            top_k: Optional limit on returned results

        Returns:
            List of (document, score) tuples, sorted by relevance
        """
        if not documents:
            return []

        self._load_model()

        # Create query-document pairs
        pairs = [[query, doc.content] for doc in documents]

        # Get relevance scores
        scores = self._model.predict(
            pairs,
            batch_size=self.config.batch_size,
            show_progress_bar=len(pairs) > 50,
        )

        # Normalize scores to 0-1 range if configured
        if self.config.normalize_scores:
            scores = self._normalize_scores(scores)

        # Pair documents with scores and sort
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: -x[1])

        if top_k is not None:
            doc_scores = doc_scores[:top_k]

        return doc_scores

    def score(self, query: str, document: str) -> float:
        """Score a single query-document pair.

        Args:
            query: Search query
            document: Document text

        Returns:
            Relevance score
        """
        self._load_model()

        score = self._model.predict([[query, document]])[0]

        if self.config.normalize_scores:
            # Apply sigmoid for single score normalization
            score = 1 / (1 + np.exp(-score))

        return float(score)

    def score_batch(
        self,
        query: str,
        documents: list[str],
    ) -> list[float]:
        """Score multiple documents against a query.

        Args:
            query: Search query
            documents: Document texts

        Returns:
            List of relevance scores
        """
        if not documents:
            return []

        self._load_model()

        pairs = [[query, doc] for doc in documents]
        scores = self._model.predict(
            pairs,
            batch_size=self.config.batch_size,
            show_progress_bar=len(pairs) > 50,
        )

        if self.config.normalize_scores:
            scores = self._normalize_scores(scores)

        return scores.tolist()

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to 0-1 range using sigmoid."""
        return 1 / (1 + np.exp(-scores))

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    def stats(self) -> dict:
        """Get reranker statistics."""
        return {
            "model_name": self.config.model_name,
            "model_loaded": self._model is not None,
            "device": self._get_device(),
            "max_length": self.config.max_length,
        }


# Optional: Lightweight reranker using embedding similarity
# Useful when cross-encoder is too slow
class LightweightReranker:
    """Fast reranker using embedding similarity.

    Less accurate than cross-encoder but much faster.
    Good for real-time applications or very large result sets.
    """

    def __init__(self, embedding_service: "EmbeddingService"):
        self.embedding_service = embedding_service

    def rerank(
        self,
        query: str,
        documents: list["Document"],
        top_k: int | None = None,
    ) -> list[tuple["Document", float]]:
        """Rerank using embedding similarity.

        Args:
            query: Search query
            documents: Documents to rerank
            top_k: Optional limit

        Returns:
            List of (document, score) tuples
        """
        if not documents:
            return []

        # Get query embedding
        query_emb = self.embedding_service.embed_query(query)

        # Get document embeddings
        doc_texts = [doc.content for doc in documents]
        doc_embs = self.embedding_service.embed_documents(doc_texts)

        # Calculate similarities
        scores = self.embedding_service.similarity(query_emb, doc_embs)

        # Pair and sort
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: -x[1])

        if top_k is not None:
            doc_scores = doc_scores[:top_k]

        return doc_scores
