"""Vector store using ChromaDB for persistent local storage.

ChromaDB provides:
- Persistent local storage (SQLite + DuckDB backend)
- HNSW index for fast approximate nearest neighbor search
- Metadata filtering
- Automatic batching
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Document:
    """A document with content, metadata, and optional embedding."""

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: np.ndarray | None = None

    @classmethod
    def create(
        cls,
        content: str,
        metadata: dict[str, Any] | None = None,
        doc_id: str | None = None,
    ) -> "Document":
        """Create a document with auto-generated ID if not provided."""
        if doc_id is None:
            # Generate deterministic ID from content
            doc_id = hashlib.md5(content.encode()).hexdigest()[:16]

        return cls(
            id=doc_id,
            content=content,
            metadata=metadata or {},
        )


@dataclass
class VectorStoreConfig:
    """Configuration for vector store."""

    # Persistence path
    persist_directory: str = ".chroma"

    # Collection name
    collection_name: str = "research_evidence"

    # Distance metric: "cosine", "l2", "ip" (inner product)
    distance_metric: str = "cosine"

    # HNSW parameters for quality/speed tradeoff
    hnsw_space: str = "cosine"
    hnsw_construction_ef: int = 200  # Higher = better quality, slower build
    hnsw_search_ef: int = 100  # Higher = better quality, slower search
    hnsw_m: int = 32  # Connections per node, higher = better quality


class VectorStore:
    """ChromaDB-based vector store with persistent local storage."""

    def __init__(
        self,
        embedding_service: "EmbeddingService",
        config: VectorStoreConfig | None = None,
    ):
        """Initialize vector store.

        Args:
            embedding_service: Service for generating embeddings
            config: Store configuration
        """
        self.embedding_service = embedding_service
        self.config = config or VectorStoreConfig()
        self._client = None
        self._collection = None

    def _get_client(self):
        """Lazy initialize ChromaDB client."""
        if self._client is not None:
            return self._client

        import chromadb
        from chromadb.config import Settings

        # Create persistent client
        persist_path = Path(self.config.persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(persist_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        return self._client

    def _get_collection(self):
        """Get or create the collection."""
        if self._collection is not None:
            return self._collection

        client = self._get_client()

        # Collection metadata for HNSW configuration
        metadata = {
            "hnsw:space": self.config.hnsw_space,
            "hnsw:construction_ef": self.config.hnsw_construction_ef,
            "hnsw:search_ef": self.config.hnsw_search_ef,
            "hnsw:M": self.config.hnsw_m,
        }

        self._collection = client.get_or_create_collection(
            name=self.config.collection_name,
            metadata=metadata,
        )

        return self._collection

    def add(self, documents: list[Document]) -> list[str]:
        """Add documents to the vector store.

        Args:
            documents: Documents to add (embeddings will be generated if not provided)

        Returns:
            List of document IDs
        """
        if not documents:
            return []

        collection = self._get_collection()

        # Prepare data for ChromaDB
        ids = []
        contents = []
        metadatas = []
        embeddings = []
        # Track docs needing embeddings for batch generation
        needs_embedding_indices = []

        for i, doc in enumerate(documents):
            ids.append(doc.id)
            contents.append(doc.content)

            # Ensure metadata values are ChromaDB-compatible
            clean_metadata = self._clean_metadata(doc.metadata)
            # ChromaDB requires non-empty metadata
            if not clean_metadata:
                clean_metadata = {"_id": doc.id}
            metadatas.append(clean_metadata)

            if doc.embedding is not None:
                embeddings.append(doc.embedding.tolist())
            else:
                embeddings.append(None)  # Placeholder
                needs_embedding_indices.append(i)

        # Batch-generate embeddings for docs that need them
        if needs_embedding_indices:
            texts_to_embed = [contents[i] for i in needs_embedding_indices]
            batch_embeddings = self.embedding_service.embed_documents(texts_to_embed)
            for idx, emb in zip(needs_embedding_indices, batch_embeddings):
                embeddings[idx] = emb.tolist()

        # Upsert to handle duplicates
        collection.upsert(
            ids=ids,
            documents=contents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        return ids

    def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Convenience method to add raw texts.

        Args:
            texts: Text contents to add
            metadatas: Optional metadata for each text
            ids: Optional IDs (auto-generated if not provided)

        Returns:
            List of document IDs
        """
        if metadatas is None:
            metadatas = [{}] * len(texts)
        if ids is None:
            ids = [hashlib.md5(t.encode()).hexdigest()[:16] for t in texts]

        documents = [
            Document(id=id_, content=text, metadata=meta)
            for id_, text, meta in zip(ids, texts, metadatas)
        ]

        return self.add(documents)

    def search(
        self,
        query: str,
        k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """Search for similar documents.

        Args:
            query: Search query
            k: Number of results to return
            filter: Optional metadata filter (ChromaDB where clause)

        Returns:
            List of (document, score) tuples, sorted by relevance
        """
        collection = self._get_collection()

        # Generate query embedding
        query_embedding = self.embedding_service.embed_query(query)

        # Build query kwargs
        query_kwargs = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }

        if filter:
            query_kwargs["where"] = filter

        # Execute search
        results = collection.query(**query_kwargs)

        # Parse results
        documents_with_scores = []

        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                content = results["documents"][0][i] if results["documents"] else ""
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0.0

                # Convert distance to similarity score
                # ChromaDB returns distance, lower is better
                # For cosine: similarity = 1 - distance
                score = 1.0 - distance

                doc = Document(id=doc_id, content=content, metadata=metadata)
                documents_with_scores.append((doc, score))

        return documents_with_scores

    def search_by_embedding(
        self,
        embedding: np.ndarray,
        k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """Search using a pre-computed embedding.

        Args:
            embedding: Query embedding vector
            k: Number of results
            filter: Optional metadata filter

        Returns:
            List of (document, score) tuples
        """
        collection = self._get_collection()

        query_kwargs = {
            "query_embeddings": [embedding.tolist()],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }

        if filter:
            query_kwargs["where"] = filter

        results = collection.query(**query_kwargs)

        documents_with_scores = []

        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                content = results["documents"][0][i] if results["documents"] else ""
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0.0
                score = 1.0 - distance

                doc = Document(id=doc_id, content=content, metadata=metadata)
                documents_with_scores.append((doc, score))

        return documents_with_scores

    def delete(self, ids: list[str]) -> None:
        """Delete documents by ID.

        Args:
            ids: Document IDs to delete
        """
        if not ids:
            return

        collection = self._get_collection()
        collection.delete(ids=ids)

    def delete_by_filter(self, filter: dict[str, Any]) -> None:
        """Delete documents matching a filter.

        Args:
            filter: ChromaDB where clause
        """
        collection = self._get_collection()
        collection.delete(where=filter)

    def get(self, ids: list[str]) -> list[Document]:
        """Get documents by ID.

        Args:
            ids: Document IDs to retrieve

        Returns:
            List of documents
        """
        if not ids:
            return []

        collection = self._get_collection()
        results = collection.get(
            ids=ids,
            include=["documents", "metadatas"],
        )

        documents = []
        for i, doc_id in enumerate(results["ids"]):
            content = results["documents"][i] if results["documents"] else ""
            metadata = results["metadatas"][i] if results["metadatas"] else {}
            documents.append(Document(id=doc_id, content=content, metadata=metadata))

        return documents

    def count(self) -> int:
        """Get total number of documents in the store."""
        collection = self._get_collection()
        return collection.count()

    def clear(self) -> None:
        """Delete all documents from the collection."""
        client = self._get_client()
        client.delete_collection(self.config.collection_name)
        self._collection = None

    def _clean_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Clean metadata to be ChromaDB-compatible.

        ChromaDB only supports str, int, float, bool values.
        """
        clean = {}
        for key, value in metadata.items():
            if value is None:
                continue
            elif isinstance(value, (str, int, float, bool)):
                clean[key] = value
            elif isinstance(value, datetime):
                clean[key] = value.isoformat()
            elif isinstance(value, (list, tuple)):
                # Convert lists to comma-separated strings
                clean[key] = ",".join(str(v) for v in value)
            else:
                # Convert other types to string
                clean[key] = str(value)
        return clean

    def stats(self) -> dict:
        """Get store statistics."""
        collection = self._get_collection()
        return {
            "collection_name": self.config.collection_name,
            "document_count": collection.count(),
            "persist_directory": self.config.persist_directory,
        }
