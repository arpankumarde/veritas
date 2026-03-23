"""Integration layer connecting hybrid retrieval with the memory system.

This module provides a SemanticMemoryStore that wraps the existing ExternalMemoryStore
and adds hybrid retrieval capabilities (semantic + BM25 + reranking).
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import aiosqlite

from ..memory.external import ExternalMemoryStore, StoredMemory
from .embeddings import EmbeddingConfig
from .hybrid import HybridConfig, HybridRetriever
from .vectorstore import Document


@dataclass
class SemanticSearchResult:
    """Result from semantic memory search."""

    memory: StoredMemory
    score: float
    retrieval_method: str  # "hybrid", "semantic", "bm25", "fts"

    @property
    def content(self) -> str:
        return self.memory.content


class SemanticMemoryStore:
    """External memory store enhanced with hybrid retrieval.

    Wraps ExternalMemoryStore and maintains a parallel hybrid index
    for high-quality semantic search.

    Usage:
        store = SemanticMemoryStore(db_path="research_memory.db")

        # Store memory (indexes in both FTS and hybrid)
        await store.store(session_id, content, memory_type, tags, metadata)

        # Semantic search (uses hybrid retrieval)
        results = await store.search_semantic(query, session_id, limit=10)
    """

    def __init__(
        self,
        db_path: str = "memory.db",
        retrieval_config: HybridConfig | None = None,
        enable_hybrid: bool = True,
    ):
        """Initialize semantic memory store.

        Args:
            db_path: Path to SQLite database for memory storage
            retrieval_config: Configuration for hybrid retriever
            enable_hybrid: Whether to enable hybrid retrieval (can be disabled for tests)
        """
        self._external_store = ExternalMemoryStore(db_path)
        self._db_path = Path(db_path)
        self._enable_hybrid = enable_hybrid

        if enable_hybrid:
            # Set up retriever with paths based on db_path
            db_base = Path(db_path).stem
            persist_dir = Path(db_path).parent / f".{db_base}_retrieval"

            if retrieval_config is None:
                retrieval_config = HybridConfig(
                    persist_directory=str(persist_dir),
                    # Use high-quality models
                    embedding=EmbeddingConfig(
                        model_name="BAAI/bge-large-en-v1.5",
                    ),
                )

            self._retriever = HybridRetriever(retrieval_config)
        else:
            self._retriever = None

    async def store(
        self,
        session_id: str,
        content: str,
        memory_type: str,
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Store content to memory with hybrid indexing.

        Args:
            session_id: Current research session ID
            content: Content to store
            memory_type: Type of memory (evidence, summary, context, decision)
            tags: Optional tags for retrieval
            metadata: Optional metadata

        Returns:
            ID of stored memory
        """
        # Store in SQLite (with FTS)
        memory_id = await self._external_store.store(
            session_id=session_id,
            content=content,
            memory_type=memory_type,
            tags=tags,
            metadata=metadata,
        )

        # Index in hybrid retriever
        if self._retriever:
            doc_metadata = {
                "memory_id": memory_id,
                "session_id": session_id,
                "memory_type": memory_type,
                "tags": ",".join(tags) if tags else "",
                "created_at": datetime.now().isoformat(),
            }
            if metadata:
                # Add custom metadata (flatten for ChromaDB compatibility)
                for k, v in metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        doc_metadata[f"meta_{k}"] = v

            self._retriever.add_texts(
                texts=[content],
                metadatas=[doc_metadata],
                ids=[memory_id],
            )

        return memory_id

    async def search_semantic(
        self,
        query: str,
        session_id: str | None = None,
        memory_type: str | None = None,
        limit: int = 10,
        use_reranker: bool = True,
    ) -> list[SemanticSearchResult]:
        """Search memory using hybrid retrieval.

        Args:
            query: Search query
            session_id: Optional filter by session
            memory_type: Optional filter by type
            limit: Maximum results
            use_reranker: Whether to use cross-encoder reranking

        Returns:
            List of SemanticSearchResult sorted by relevance
        """
        if not self._retriever:
            # Fall back to FTS search
            memories = await self._external_store.search(
                query=query,
                session_id=session_id,
                memory_type=memory_type,
                limit=limit,
            )
            return [
                SemanticSearchResult(
                    memory=m,
                    score=max(0.0, 1.0 - (i * 0.05)),  # Approximate score
                    retrieval_method="fts",
                )
                for i, m in enumerate(memories)
            ]

        # Build filter for ChromaDB
        filter_conditions = {}
        if session_id:
            filter_conditions["session_id"] = session_id
        if memory_type:
            filter_conditions["memory_type"] = memory_type

        chroma_filter = filter_conditions if filter_conditions else None

        # Run hybrid search
        results = self._retriever.search(
            query=query,
            k=limit,
            filter=chroma_filter,
            use_reranker=use_reranker,
        )

        # Batch-fetch all memories from SQLite in one query (fixes N+1)
        memory_ids = [
            r.document.metadata.get("memory_id", r.document.id) for r in results
        ]
        memories_by_id = await self._get_memories_by_ids(memory_ids)

        # Convert to SemanticSearchResult
        search_results = []
        for result, memory_id in zip(results, memory_ids):
            fetched = memories_by_id.get(memory_id)

            if fetched:
                search_results.append(SemanticSearchResult(
                    memory=fetched[0],
                    score=result.score,
                    retrieval_method="hybrid" if result.reranker_score else "rrf",
                ))
            else:
                # Memory might have been deleted from SQLite but still in vector store
                # Create a temporary memory object
                search_results.append(SemanticSearchResult(
                    memory=StoredMemory(
                        id=memory_id,
                        session_id=result.document.metadata.get("session_id", ""),
                        content=result.content,
                        memory_type=result.document.metadata.get("memory_type", "unknown"),
                        tags=result.document.metadata.get("tags", "").split(","),
                        created_at=datetime.now(),
                        metadata={},
                    ),
                    score=result.score,
                    retrieval_method="hybrid" if result.reranker_score else "rrf",
                ))

        return search_results

    async def _get_memory_by_id(self, memory_id: str) -> list[StoredMemory]:
        """Get memory by ID from SQLite."""
        results = await self._get_memories_by_ids([memory_id])
        return results.get(memory_id, [])

    async def _get_memories_by_ids(self, memory_ids: list[str]) -> dict[str, list[StoredMemory]]:
        """Get memories by IDs from SQLite in a single query.

        Args:
            memory_ids: List of memory IDs to fetch

        Returns:
            Dict mapping memory_id -> list of StoredMemory
        """
        if not memory_ids:
            return {}

        async with aiosqlite.connect(self._db_path) as conn:
            placeholders = ",".join("?" for _ in memory_ids)
            cursor = await conn.execute(f"""
                SELECT id, session_id, content, memory_type, tags, created_at, metadata
                FROM memories
                WHERE id IN ({placeholders})
            """, memory_ids)

            rows = await cursor.fetchall()

        result: dict[str, list[StoredMemory]] = {}
        for row in rows:
            memory = StoredMemory(
                id=row[0],
                session_id=row[1],
                content=row[2],
                memory_type=row[3],
                tags=json.loads(row[4]) if row[4] else [],
                created_at=datetime.fromisoformat(row[5]),
                metadata=json.loads(row[6]) if row[6] else {},
            )
            result.setdefault(row[0], []).append(memory)
        return result

    async def search(
        self,
        query: str,
        session_id: str | None = None,
        memory_type: str | None = None,
        limit: int = 10,
    ) -> list[StoredMemory]:
        """Search memory (delegates to FTS for backward compatibility).

        For semantic search, use search_semantic() instead.
        """
        return await self._external_store.search(query, session_id, memory_type, limit)

    async def get_by_session(
        self,
        session_id: str,
        memory_type: str | None = None,
    ) -> list[StoredMemory]:
        """Get all memories for a session."""
        return await self._external_store.get_by_session(session_id, memory_type)

    async def get_recent(
        self,
        session_id: str,
        limit: int = 10,
    ) -> list[StoredMemory]:
        """Get most recent memories for a session."""
        return await self._external_store.get_recent(session_id, limit)

    async def delete_session(self, session_id: str) -> int:
        """Delete all memories for a session."""
        # Get memory IDs for this session
        memories = await self._external_store.get_by_session(session_id)
        memory_ids = [m.id for m in memories]

        # Delete from hybrid retriever
        if self._retriever and memory_ids:
            self._retriever.delete(memory_ids)

        # Delete from SQLite
        return await self._external_store.delete_session(session_id)

    async def get_stats(self) -> dict:
        """Get memory store statistics."""
        stats = await self._external_store.get_stats()

        if self._retriever:
            stats["retrieval"] = self._retriever.stats()
            stats["hybrid_enabled"] = True
        else:
            stats["hybrid_enabled"] = False

        return stats

    async def reindex_all(self) -> int:
        """Reindex all memories in the hybrid retriever.

        Useful after enabling hybrid retrieval on existing data.

        Returns:
            Number of memories indexed
        """
        if not self._retriever:
            return 0

        # Clear existing index
        self._retriever.clear()

        # Get all memories from SQLite
        async with aiosqlite.connect(self._db_path) as conn:
            cursor = await conn.execute("""
                SELECT id, session_id, content, memory_type, tags, created_at, metadata
                FROM memories
            """)

            rows = await cursor.fetchall()

        if not rows:
            return 0

        # Build documents
        documents = []
        for row in rows:
            memory_id = row[0]
            session_id = row[1]
            content = row[2]
            memory_type = row[3]
            tags = json.loads(row[4]) if row[4] else []
            created_at = row[5]

            doc_metadata = {
                "memory_id": memory_id,
                "session_id": session_id,
                "memory_type": memory_type,
                "tags": ",".join(tags),
                "created_at": created_at,
            }

            documents.append(Document(
                id=memory_id,
                content=content,
                metadata=doc_metadata,
            ))

        # Index all at once
        self._retriever.add(documents)

        return len(documents)


# Convenience function
def create_semantic_memory(
    db_path: str = "research_memory.db",
    embedding_model: str = "BAAI/bge-large-en-v1.5",
    use_reranker: bool = True,
) -> SemanticMemoryStore:
    """Create a semantic memory store with sensible defaults.

    Args:
        db_path: Path to SQLite database
        embedding_model: Embedding model name
        use_reranker: Whether to enable reranking

    Returns:
        Configured SemanticMemoryStore
    """

    config = HybridConfig(
        embedding=EmbeddingConfig(model_name=embedding_model),
        use_reranker=use_reranker,
    )

    return SemanticMemoryStore(
        db_path=db_path,
        retrieval_config=config,
    )
