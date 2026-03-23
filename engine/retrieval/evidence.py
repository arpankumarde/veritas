"""Evidence retriever for research-specific semantic search.

This module provides specialized retrieval for research evidence,
integrated with the knowledge graph and manager agent workflow.
"""

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime

from ..models.evidence import Evidence, EvidenceType
from .embeddings import EmbeddingConfig
from .hybrid import HybridConfig, HybridRetriever, RetrievalResult
from .reranker import RerankerConfig
from .vectorstore import Document, VectorStoreConfig


@dataclass
class EvidenceSearchResult:
    """Result from evidence search."""

    evidence: Evidence
    score: float
    bm25_rank: int | None = None
    semantic_rank: int | None = None
    reranked: bool = False


class EvidenceRetriever:
    """Hybrid retriever specialized for research evidence.

    Provides:
    - Semantic search over evidence
    - Evidence type filtering
    - Source URL deduplication
    - Confidence-weighted ranking
    - Cross-session retrieval

    Usage:
        retriever = EvidenceRetriever(persist_dir=".evidence_index")

        # Index evidence as it comes in
        retriever.add_evidence(evidence, session_id, topic_id)

        # Search for relevant evidence
        results = retriever.search("quantum computing applications", limit=10)
    """

    def __init__(
        self,
        persist_dir: str = ".evidence_retrieval",
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        use_reranker: bool = True,
        reranker_model: str = "BAAI/bge-reranker-large",
        collection_name: str = "research_evidence",
    ):
        """Initialize evidence retriever.

        Args:
            persist_dir: Directory for persisting indices
            embedding_model: Sentence transformer model for embeddings
            use_reranker: Whether to use cross-encoder reranking
            reranker_model: Cross-encoder model for reranking
            collection_name: ChromaDB collection name (use session-specific names for isolation)
        """
        config = HybridConfig(
            persist_directory=persist_dir,
            embedding=EmbeddingConfig(
                model_name=embedding_model,
                # Optimize query prefix for research queries
                query_prefix="Represent this research query for finding relevant evidence: ",
            ),
            vectorstore=VectorStoreConfig(collection_name=collection_name),
            reranker=RerankerConfig(model_name=reranker_model) if use_reranker else None,
            use_reranker=use_reranker,
            # Retrieval parameters optimized for evidence
            initial_k=100,  # Get more candidates for better coverage
            rerank_k=30,    # Rerank top 30
            final_k=10,     # Return top 10
            semantic_weight=0.6,  # Slightly favor semantic for research
        )

        self._retriever = HybridRetriever(config)
        # [HARDENED] PERF-002: Use OrderedDict with max size to prevent unbounded growth
        self._evidence_cache: OrderedDict[str, Evidence] = OrderedDict()
        self._MAX_CACHE_SIZE = 5000

    def add_evidence(
        self,
        evidence: Evidence,
        session_id: str,
        topic_id: str | None = None,
    ) -> str:
        """Add an evidence item to the index.

        Args:
            evidence: The evidence to index
            session_id: Current research session ID
            topic_id: Optional topic ID

        Returns:
            Evidence ID
        """
        # Create searchable content (combine claim with context)
        content = evidence.content
        if hasattr(evidence, 'supporting_quote') and evidence.supporting_quote:
            content = f"{evidence.content}\n\nContext: {evidence.supporting_quote}"

        # Build metadata for filtering
        metadata = {
            "session_id": session_id,
            "evidence_type": evidence.evidence_type.value if hasattr(evidence.evidence_type, 'value') else str(evidence.evidence_type),
            "confidence": evidence.confidence,
            "source_url": evidence.source_url or "",
            "created_at": datetime.now().isoformat(),
        }

        if topic_id:
            metadata["topic_id"] = topic_id

        if hasattr(evidence, 'source_title') and evidence.source_title:
            metadata["source_title"] = evidence.source_title

        # Generate ID if not present (ensure it's a string for ChromaDB)
        evidence_id = getattr(evidence, 'id', None)
        if not evidence_id:
            import hashlib
            evidence_id = hashlib.md5(content.encode()).hexdigest()[:12]
        evidence_id = str(evidence_id)  # ChromaDB requires string IDs

        # Cache the evidence object
        self._evidence_cache[evidence_id] = evidence
        # [HARDENED] PERF-002: Evict oldest entry if cache exceeds max size
        if len(self._evidence_cache) > self._MAX_CACHE_SIZE:
            self._evidence_cache.popitem(last=False)

        # Add to retriever
        self._retriever.add_texts(
            texts=[content],
            metadatas=[metadata],
            ids=[evidence_id],
        )

        return evidence_id

    def add_evidence_batch(
        self,
        evidence_list: list[Evidence],
        session_id: str,
        topic_id: str | None = None,
    ) -> list[str]:
        """Add multiple evidence items efficiently.

        Args:
            evidence_list: Evidence items to index
            session_id: Current research session ID
            topic_id: Optional topic ID

        Returns:
            List of evidence IDs
        """
        if not evidence_list:
            return []

        documents = []
        ids = []

        for evidence in evidence_list:
            # Create searchable content
            content = evidence.content
            if hasattr(evidence, 'supporting_quote') and evidence.supporting_quote:
                content = f"{evidence.content}\n\nContext: {evidence.supporting_quote}"

            # Build metadata
            metadata = {
                "session_id": session_id,
                "evidence_type": evidence.evidence_type.value if hasattr(evidence.evidence_type, 'value') else str(evidence.evidence_type),
                "confidence": evidence.confidence,
                "source_url": evidence.source_url or "",
                "created_at": datetime.now().isoformat(),
            }

            if topic_id:
                metadata["topic_id"] = topic_id

            if hasattr(evidence, 'source_title') and evidence.source_title:
                metadata["source_title"] = evidence.source_title

            # Generate ID (ensure it's a string for ChromaDB)
            evidence_id = getattr(evidence, 'id', None)
            if not evidence_id:
                import hashlib
                evidence_id = hashlib.md5(content.encode()).hexdigest()[:12]
            evidence_id = str(evidence_id)  # ChromaDB requires string IDs

            # Cache
            self._evidence_cache[evidence_id] = evidence
            # [HARDENED] PERF-002: Evict oldest entry if cache exceeds max size
            if len(self._evidence_cache) > self._MAX_CACHE_SIZE:
                self._evidence_cache.popitem(last=False)

            documents.append(Document(
                id=evidence_id,
                content=content,
                metadata=metadata,
            ))
            ids.append(evidence_id)

        self._retriever.add(documents)
        return ids

    # Backward-compat aliases (old name -> new name)
    def add_findings(
        self,
        findings: list[Evidence],
        session_id: str,
        topic_id: str | None = None,
    ) -> list[str]:
        """Alias for add_evidence_batch (backward compat)."""
        return self.add_evidence_batch(findings, session_id, topic_id)

    def search(
        self,
        query: str,
        limit: int = 10,
        session_id: str | None = None,
        evidence_types: list[EvidenceType] | None = None,
        min_confidence: float | None = None,
        use_reranker: bool | None = None,
    ) -> list[EvidenceSearchResult]:
        """Search for relevant evidence.

        Args:
            query: Search query
            limit: Maximum results
            session_id: Optional filter by session
            evidence_types: Optional filter by evidence types
            min_confidence: Optional minimum confidence threshold
            use_reranker: Override default reranker setting

        Returns:
            List of EvidenceSearchResult sorted by relevance
        """
        # Build filter
        filter_dict = {}
        if session_id:
            filter_dict["session_id"] = session_id

        # Note: ChromaDB doesn't support IN queries directly,
        # so we'll filter evidence types post-retrieval
        chroma_filter = filter_dict if filter_dict else None

        # Get more results if we need to filter
        fetch_k = limit * 3 if (evidence_types or min_confidence) else limit

        # Run hybrid search
        results = self._retriever.search(
            query=query,
            k=fetch_k,
            filter=chroma_filter,
            use_reranker=use_reranker,
        )

        # Convert and filter results
        search_results = []

        for result in results:
            evidence_id = result.document.id

            # Get evidence from cache or reconstruct
            evidence = self._evidence_cache.get(evidence_id)
            if not evidence:
                # Reconstruct from document metadata
                evidence = self._reconstruct_evidence(result)

            # Apply filters
            if evidence_types:
                evidence_type = evidence.evidence_type
                if hasattr(evidence_type, 'value'):
                    if evidence_type not in evidence_types:
                        continue
                else:
                    type_str = str(evidence_type)
                    if not any(et.value == type_str or et.name == type_str for et in evidence_types):
                        continue

            if min_confidence and evidence.confidence < min_confidence:
                continue

            search_results.append(EvidenceSearchResult(
                evidence=evidence,
                score=result.score,
                bm25_rank=result.bm25_rank,
                semantic_rank=result.semantic_rank,
                reranked=result.reranker_score is not None,
            ))

            if len(search_results) >= limit:
                break

        return search_results

    def _reconstruct_evidence(self, result: RetrievalResult) -> Evidence:
        """Reconstruct an Evidence object from retrieval result."""
        metadata = result.document.metadata

        # Determine evidence type
        type_str = metadata.get("evidence_type", "FACT")
        try:
            evidence_type = EvidenceType(type_str)
        except (ValueError, KeyError):
            evidence_type = EvidenceType.FACT

        # [HARDENED] BUG-006: Include session_id in reconstructed Evidence
        return Evidence(
            content=result.document.content.split("\n\nContext:")[0],  # Extract main content
            evidence_type=evidence_type,
            confidence=float(metadata.get("confidence", 0.5)),
            source_url=metadata.get("source_url"),
            source_title=metadata.get("source_title"),
            session_id=metadata.get("session_id", ""),
        )

    def find_similar(
        self,
        evidence: Evidence,
        limit: int = 5,
        exclude_self: bool = True,
        session_id: str | None = None,
    ) -> list[EvidenceSearchResult]:
        """Find evidence similar to a given piece of evidence.

        Useful for:
        - Deduplication
        - Finding supporting evidence
        - Detecting contradictions

        Args:
            evidence: Evidence to find similar ones for
            limit: Maximum results
            exclude_self: Whether to exclude the input evidence
            session_id: Optional filter by session

        Returns:
            List of similar evidence
        """
        results = self.search(
            query=evidence.content,
            limit=limit + (1 if exclude_self else 0),
            session_id=session_id,
        )

        if exclude_self:
            # Remove the input evidence if it appears in results
            results = [
                r for r in results
                if r.evidence.content != evidence.content
            ][:limit]

        return results

    def find_by_source(
        self,
        source_url: str,
        limit: int = 20,
    ) -> list[Evidence]:
        """Find all evidence from a specific source.

        Args:
            source_url: Source URL to search for
            limit: Maximum results

        Returns:
            List of evidence from that source
        """
        # [HARDENED] BUG-005: Use source_url as query text instead of empty string
        results = self._retriever.search(
            query=source_url,
            k=limit,
            filter={"source_url": source_url},
            use_reranker=False,  # No need for reranking with filter-only
        )

        evidence_list = []
        for result in results:
            evidence = self._evidence_cache.get(result.document.id)
            if evidence:
                evidence_list.append(evidence)
            else:
                evidence_list.append(self._reconstruct_evidence(result))

        return evidence_list

    def get_session_evidence(
        self,
        session_id: str,
        limit: int = 100,
    ) -> list[Evidence]:
        """Get all evidence for a session.

        Args:
            session_id: Session ID
            limit: Maximum results

        Returns:
            List of evidence
        """
        # [HARDENED] BUG-007: Use empty query instead of biased "research evidence"
        results = self._retriever.search(
            query="",
            k=limit,
            filter={"session_id": session_id},
            use_reranker=False,
        )

        evidence_list = []
        for result in results:
            evidence = self._evidence_cache.get(result.document.id)
            if evidence:
                evidence_list.append(evidence)
            else:
                evidence_list.append(self._reconstruct_evidence(result))

        return evidence_list

    def delete_session(self, session_id: str) -> int:
        """Delete all evidence for a session.

        Args:
            session_id: Session to delete

        Returns:
            Number of evidence items deleted
        """
        # Get evidence for this session
        evidence_list = self.get_session_evidence(session_id, limit=10000)

        if not evidence_list:
            return 0

        # Delete from retriever
        # Note: ChromaDB doesn't support delete by filter directly
        # We need to delete by IDs
        ids_to_delete = []
        for evidence in evidence_list:
            evidence_id = getattr(evidence, 'id', None)
            if evidence_id:
                ids_to_delete.append(evidence_id)
                self._evidence_cache.pop(evidence_id, None)

        if ids_to_delete:
            self._retriever.delete(ids_to_delete)

        return len(ids_to_delete)

    def count(self) -> int:
        """Get total number of indexed evidence items."""
        return self._retriever.count()

    def stats(self) -> dict:
        """Get retriever statistics."""
        return {
            "total_evidence": self.count(),
            "cached_evidence": len(self._evidence_cache),
            "retriever": self._retriever.stats(),
        }

    def clear(self) -> None:
        """Clear all indexed evidence."""
        self._retriever.clear()
        self._evidence_cache.clear()


# Session-scoped retriever instances (keyed by session_id, None key for legacy global)
_evidence_retrievers: dict[str | None, EvidenceRetriever] = {}


def get_evidence_retriever(
    persist_dir: str = ".evidence_retrieval",
    session_id: str | None = None,
    **kwargs,
) -> EvidenceRetriever:
    """Get or create an evidence retriever, optionally scoped to a session.

    When session_id is provided, returns a session-specific retriever with an
    isolated ChromaDB collection. When session_id is None, behaves as a global
    singleton for backward compatibility.

    Args:
        persist_dir: Directory for persistence
        session_id: Optional session ID for isolation. If provided, the retriever
            uses a session-specific ChromaDB collection.
        **kwargs: Additional arguments for EvidenceRetriever

    Returns:
        EvidenceRetriever instance (session-scoped or global)
    """
    if session_id is not None and session_id not in _evidence_retrievers:
        # Sanitize: ChromaDB names must match [a-zA-Z0-9._-], no trailing special chars
        safe_id = session_id.strip("_- ") if session_id else "default"
        collection_name = f"evidence.{safe_id}" if safe_id else "evidence.default"
        _evidence_retrievers[session_id] = EvidenceRetriever(
            persist_dir=persist_dir,
            collection_name=collection_name,
            **kwargs,
        )
    elif session_id is None and None not in _evidence_retrievers:
        _evidence_retrievers[None] = EvidenceRetriever(persist_dir=persist_dir, **kwargs)

    return _evidence_retrievers[session_id]


def reset_evidence_retriever(session_id: str | None = None) -> None:
    """Reset an evidence retriever.

    Args:
        session_id: Session ID to reset, or None to reset the global instance.
            Pass "__all__" to reset all retrievers.
    """
    if session_id == "__all__":
        _evidence_retrievers.clear()
    else:
        _evidence_retrievers.pop(session_id, None)
