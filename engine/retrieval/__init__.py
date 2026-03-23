"""Hybrid retrieval system combining semantic and lexical search.

This module provides high-quality retrieval combining:
- Semantic search (BGE embeddings + ChromaDB)
- Lexical search (BM25)
- Reciprocal Rank Fusion for hybrid combination
- Cross-encoder reranking for final quality boost

Usage:
    from engine.retrieval import HybridRetriever, create_retriever

    # Quick setup
    retriever = create_retriever()
    retriever.add_texts(["doc1", "doc2", "doc3"])
    results = retriever.search("my query")

    # For research evidence
    from engine.retrieval import EvidenceRetriever
    evidence_retriever = EvidenceRetriever()
    evidence_retriever.add_evidence(evidence, session_id)
    results = evidence_retriever.search("quantum computing")
"""

from .bm25 import BM25Config, BM25Index
from .deduplication import (
    DeduplicationConfig,
    DeduplicationResult,
    EvidenceDeduplicator,
    get_deduplicator,
    reset_deduplicator,
)
from .embeddings import EmbeddingConfig, EmbeddingService, get_embedding_service
from .evidence import (
    EvidenceSearchResult,
    EvidenceRetriever,
    get_evidence_retriever,
    reset_evidence_retriever,
)
from .hybrid import HybridConfig, HybridRetriever, RetrievalResult, create_retriever
from .memory_integration import (
    SemanticMemoryStore,
    SemanticSearchResult,
    create_semantic_memory,
)
from .query_expansion import (
    ExpandedQuery,
    QueryExpander,
    QueryExpansionConfig,
    QueryExpansionResult,
    merge_search_results,
)
from .reranker import LightweightReranker, Reranker, RerankerConfig
from .vectorstore import Document, VectorStore, VectorStoreConfig

# Backward-compat aliases (old name -> new name)
FindingsRetriever = EvidenceRetriever
get_findings_retriever = get_evidence_retriever

__all__ = [
    # Core components
    "EmbeddingService",
    "EmbeddingConfig",
    "get_embedding_service",
    "VectorStore",
    "VectorStoreConfig",
    "Document",
    "BM25Index",
    "BM25Config",
    "Reranker",
    "RerankerConfig",
    "LightweightReranker",
    # Hybrid retriever
    "HybridRetriever",
    "HybridConfig",
    "RetrievalResult",
    "create_retriever",
    # Evidence retriever
    "EvidenceRetriever",
    "EvidenceSearchResult",
    "get_evidence_retriever",
    "reset_evidence_retriever",
    # Backward-compat aliases
    "FindingsRetriever",
    "get_findings_retriever",
    # Memory integration
    "SemanticMemoryStore",
    "SemanticSearchResult",
    "create_semantic_memory",
    # Deduplication
    "EvidenceDeduplicator",
    "DeduplicationConfig",
    "DeduplicationResult",
    "get_deduplicator",
    "reset_deduplicator",
    # Query expansion
    "QueryExpander",
    "QueryExpansionConfig",
    "QueryExpansionResult",
    "ExpandedQuery",
    "merge_search_results",
]
