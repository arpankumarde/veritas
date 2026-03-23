"""High-quality embedding service using sentence-transformers.

Supports multiple embedding models optimized for different use cases:
- BGE-large: Best overall quality for retrieval
- E5-large: Excellent for semantic similarity
- BGE-base: Good balance of quality and speed
"""

import hashlib
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

# Lazy imports for faster startup
_model = None
_model_name = None


@dataclass
class EmbeddingConfig:
    """Configuration for embedding service."""

    # Model selection - quality vs speed tradeoff
    # Best quality: "BAAI/bge-large-en-v1.5" (1024 dim, ~1.3GB)
    # Good quality: "BAAI/bge-base-en-v1.5" (768 dim, ~440MB)
    # Fast: "BAAI/bge-small-en-v1.5" (384 dim, ~130MB)
    # Alternative: "intfloat/e5-large-v2" (1024 dim, ~1.3GB)
    model_name: str = "BAAI/bge-large-en-v1.5"

    # Device selection
    device: str = "auto"  # "auto", "cuda", "mps", "cpu"

    # Batch size for encoding
    batch_size: int = 32

    # Whether to normalize embeddings (recommended for cosine similarity)
    normalize: bool = True

    # Cache directory for models
    cache_dir: str | None = None

    # Query prefix for BGE models (improves retrieval quality)
    query_prefix: str = "Represent this sentence for searching relevant passages: "

    # Document prefix (optional, some models benefit from this)
    document_prefix: str = ""

    # Max sequence length
    max_length: int = 512


class EmbeddingService:
    """High-quality embedding service with caching and batching."""

    def __init__(self, config: EmbeddingConfig | None = None):
        self.config = config or EmbeddingConfig()
        self._model = None
        self._dimension: int | None = None
        # [HARDENED] PERF-003: Use OrderedDict with max size to prevent unbounded growth
        self._embedding_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._MAX_CACHE_SIZE = 10000

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
        """Lazy load the embedding model."""
        if self._model is not None:
            return

        from sentence_transformers import SentenceTransformer

        device = self._get_device()
        cache_folder = self.config.cache_dir

        self._model = SentenceTransformer(
            self.config.model_name,
            device=device,
            cache_folder=cache_folder,
        )

        # Set max sequence length
        self._model.max_seq_length = self.config.max_length

        # Get embedding dimension
        self._dimension = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self._dimension is None:
            self._load_model()
        return self._dimension

    def _cache_key(self, text: str, is_query: bool) -> str:
        """Generate cache key for text."""
        prefix = "q:" if is_query else "d:"
        content = prefix + text
        return hashlib.md5(content.encode()).hexdigest()

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a search query with query-specific prefix.

        Args:
            query: The search query text

        Returns:
            Normalized embedding vector
        """
        cache_key = self._cache_key(query, is_query=True)
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        self._load_model()

        # Add query prefix for BGE models
        prefixed_query = self.config.query_prefix + query

        embedding = self._model.encode(
            prefixed_query,
            normalize_embeddings=self.config.normalize,
            convert_to_numpy=True,
        )

        self._embedding_cache[cache_key] = embedding
        # [HARDENED] PERF-003: Evict oldest entry if cache exceeds max size
        if len(self._embedding_cache) > self._MAX_CACHE_SIZE:
            self._embedding_cache.popitem(last=False)
        return embedding

    def embed_document(self, document: str) -> np.ndarray:
        """Embed a document for indexing.

        Args:
            document: The document text

        Returns:
            Normalized embedding vector
        """
        cache_key = self._cache_key(document, is_query=False)
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        self._load_model()

        # Add document prefix if configured
        text = self.config.document_prefix + document if self.config.document_prefix else document

        embedding = self._model.encode(
            text,
            normalize_embeddings=self.config.normalize,
            convert_to_numpy=True,
        )

        self._embedding_cache[cache_key] = embedding
        # [HARDENED] PERF-003: Evict oldest entry if cache exceeds max size
        if len(self._embedding_cache) > self._MAX_CACHE_SIZE:
            self._embedding_cache.popitem(last=False)
        return embedding

    def embed_documents(self, documents: list[str]) -> np.ndarray:
        """Embed multiple documents efficiently with batching.

        Args:
            documents: List of document texts

        Returns:
            Array of embedding vectors (n_docs, dimension)
        """
        self._load_model()

        # Check cache for each document
        embeddings = []
        uncached_indices = []
        uncached_docs = []

        for i, doc in enumerate(documents):
            cache_key = self._cache_key(doc, is_query=False)
            if cache_key in self._embedding_cache:
                embeddings.append((i, self._embedding_cache[cache_key]))
            else:
                uncached_indices.append(i)
                text = self.config.document_prefix + doc if self.config.document_prefix else doc
                uncached_docs.append(text)

        # Batch encode uncached documents
        if uncached_docs:
            new_embeddings = self._model.encode(
                uncached_docs,
                normalize_embeddings=self.config.normalize,
                convert_to_numpy=True,
                batch_size=self.config.batch_size,
                show_progress_bar=len(uncached_docs) > 100,
            )

            # [HARDENED] BUG-004: Use uncached_indices to look up original doc text,
            # not the full documents list, to avoid zip misalignment when some are cached.
            for idx, emb in zip(uncached_indices, new_embeddings):
                cache_key = self._cache_key(documents[idx], is_query=False)
                self._embedding_cache[cache_key] = emb
                # [HARDENED] PERF-003: Evict oldest entry if cache exceeds max size
                if len(self._embedding_cache) > self._MAX_CACHE_SIZE:
                    self._embedding_cache.popitem(last=False)
                embeddings.append((idx, emb))

        # Sort by original index and stack
        embeddings.sort(key=lambda x: x[0])
        return np.vstack([emb for _, emb in embeddings])

    def similarity(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and documents.

        Args:
            query_embedding: Query vector (dimension,)
            doc_embeddings: Document vectors (n_docs, dimension)

        Returns:
            Similarity scores (n_docs,)
        """
        # For normalized vectors, dot product equals cosine similarity
        if len(doc_embeddings.shape) == 1:
            doc_embeddings = doc_embeddings.reshape(1, -1)
        return np.dot(doc_embeddings, query_embedding)

    def clear_cache(self):
        """Clear embedding cache."""
        self._embedding_cache.clear()

    def cache_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "cached_embeddings": len(self._embedding_cache),
            "model_loaded": self._model is not None,
            "model_name": self.config.model_name,
            "dimension": self._dimension,
            "device": self._get_device(),
        }


# Global singleton for shared embedding service
_global_embedding_service: EmbeddingService | None = None


def get_embedding_service(config: EmbeddingConfig | None = None) -> EmbeddingService:
    """Get or create the global embedding service."""
    global _global_embedding_service
    if _global_embedding_service is None:
        _global_embedding_service = EmbeddingService(config)
    return _global_embedding_service


def reset_embedding_service():
    """Reset the global embedding service."""
    global _global_embedding_service
    _global_embedding_service = None
