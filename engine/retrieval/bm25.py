"""BM25 lexical search index for hybrid retrieval.

BM25 (Best Matching 25) is a probabilistic ranking function that excels at:
- Exact keyword matching
- Handling domain-specific terminology
- Cases where semantic similarity might miss exact matches
"""

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class BM25Config:
    """Configuration for BM25 index."""

    # BM25 parameters
    k1: float = 1.5  # Term frequency saturation parameter
    b: float = 0.75  # Length normalization parameter

    # Tokenization
    lowercase: bool = True
    remove_punctuation: bool = True
    min_token_length: int = 2

    # Optional stopwords removal
    remove_stopwords: bool = True
    custom_stopwords: set[str] = field(default_factory=set)

    # Persistence
    persist_path: str | None = None


# Common English stopwords
DEFAULT_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall", "can", "need",
    "it", "its", "this", "that", "these", "those", "i", "you", "he",
    "she", "we", "they", "what", "which", "who", "whom", "when", "where",
    "why", "how", "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "just", "also", "now", "here", "there",
}


@dataclass
class IndexedDocument:
    """A document in the BM25 index."""

    id: str
    content: str
    tokens: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


class BM25Index:
    """BM25 index for lexical search with persistence support."""

    def __init__(self, config: BM25Config | None = None):
        self.config = config or BM25Config()

        # Index state
        self._documents: dict[str, IndexedDocument] = {}
        self._doc_ids: list[str] = []  # Ordered list for scoring
        self._idf: dict[str, float] = {}  # Inverse document frequency
        self._doc_len: dict[str, int] = {}  # Document lengths
        self._avgdl: float = 0.0  # Average document length
        self._term_freqs: dict[str, dict[str, int]] = {}  # term -> {doc_id: freq}

        # Stopwords
        self._stopwords = DEFAULT_STOPWORDS.copy()
        if self.config.custom_stopwords:
            self._stopwords.update(self.config.custom_stopwords)

        # Load existing index if path provided
        if self.config.persist_path and Path(self.config.persist_path).exists():
            self.load()

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into terms."""
        if self.config.lowercase:
            text = text.lower()

        if self.config.remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)

        tokens = text.split()

        # Filter tokens
        tokens = [
            t for t in tokens
            if len(t) >= self.config.min_token_length
            and (not self.config.remove_stopwords or t not in self._stopwords)
        ]

        return tokens

    def add(self, documents: list["Document"]) -> list[str]:
        """Add documents to the index.

        Args:
            documents: Documents to index

        Returns:
            List of document IDs
        """
        ids = []

        for doc in documents:
            tokens = self._tokenize(doc.content)

            indexed_doc = IndexedDocument(
                id=doc.id,
                content=doc.content,
                tokens=tokens,
                metadata=doc.metadata,
            )

            # Update index
            self._documents[doc.id] = indexed_doc
            if doc.id not in self._doc_ids:
                self._doc_ids.append(doc.id)

            # Update term frequencies
            self._doc_len[doc.id] = len(tokens)
            term_freq = {}
            for token in tokens:
                term_freq[token] = term_freq.get(token, 0) + 1

            for term, freq in term_freq.items():
                if term not in self._term_freqs:
                    self._term_freqs[term] = {}
                self._term_freqs[term][doc.id] = freq

            ids.append(doc.id)

        # Recalculate IDF and average document length
        self._update_statistics()

        # Persist if configured
        if self.config.persist_path:
            self.save()

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
            ids: Optional IDs

        Returns:
            List of document IDs
        """
        from .vectorstore import Document

        if metadatas is None:
            metadatas = [{}] * len(texts)
        if ids is None:
            ids = [hashlib.md5(t.encode()).hexdigest()[:16] for t in texts]

        documents = [
            Document(id=id_, content=text, metadata=meta)
            for id_, text, meta in zip(ids, texts, metadatas)
        ]

        return self.add(documents)

    def _update_statistics(self):
        """Recalculate IDF scores and average document length."""
        n_docs = len(self._documents)
        if n_docs == 0:
            return

        # Average document length
        total_len = sum(self._doc_len.values())
        self._avgdl = total_len / n_docs

        # IDF for each term
        # IDF = log((N - n + 0.5) / (n + 0.5) + 1)
        # where N = total docs, n = docs containing term
        self._idf = {}
        for term, doc_freqs in self._term_freqs.items():
            n = len(doc_freqs)  # Number of docs containing term
            idf = np.log((n_docs - n + 0.5) / (n + 0.5) + 1)
            self._idf[term] = idf

    def search(
        self,
        query: str,
        k: int = 10,
        filter_fn: Callable[["IndexedDocument"], bool] | None = None,
    ) -> list[tuple["Document", float]]:
        """Search for documents matching the query.

        Args:
            query: Search query
            k: Number of results
            filter_fn: Optional function to filter documents (takes doc, returns bool)

        Returns:
            List of (document, score) tuples, sorted by relevance
        """
        from .vectorstore import Document

        if not self._documents:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # Calculate BM25 scores for all documents
        scores = {}

        for doc_id in self._doc_ids:
            doc = self._documents[doc_id]

            # Apply filter if provided
            if filter_fn and not filter_fn(doc):
                continue

            score = self._score_document(query_tokens, doc_id)
            if score > 0:
                scores[doc_id] = score

        # Sort by score descending
        sorted_docs = sorted(scores.items(), key=lambda x: -x[1])[:k]

        # Convert to Document objects
        results = []
        for doc_id, score in sorted_docs:
            indexed_doc = self._documents[doc_id]
            doc = Document(
                id=doc_id,
                content=indexed_doc.content,
                metadata=indexed_doc.metadata,
            )
            results.append((doc, score))

        return results

    def _score_document(self, query_tokens: list[str], doc_id: str) -> float:
        """Calculate BM25 score for a document.

        BM25 formula:
        score = sum(IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D|/avgdl)))

        where:
        - qi is query term
        - f(qi, D) is term frequency in document
        - |D| is document length
        - avgdl is average document length
        """
        score = 0.0
        doc_len = self._doc_len.get(doc_id, 0)

        if doc_len == 0 or self._avgdl == 0:
            return 0.0

        k1 = self.config.k1
        b = self.config.b

        for token in query_tokens:
            if token not in self._idf:
                continue

            idf = self._idf[token]
            tf = self._term_freqs.get(token, {}).get(doc_id, 0)

            if tf == 0:
                continue

            # BM25 term score
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_len / self._avgdl)
            score += idf * (numerator / denominator)

        return score

    def get_scores(self, query: str) -> dict[str, float]:
        """Get BM25 scores for all documents.

        Args:
            query: Search query

        Returns:
            Dictionary mapping doc_id to score
        """
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return {}

        scores = {}
        for doc_id in self._doc_ids:
            score = self._score_document(query_tokens, doc_id)
            scores[doc_id] = score

        return scores

    def delete(self, ids: list[str]) -> None:
        """Delete documents from the index.

        Args:
            ids: Document IDs to delete
        """
        for doc_id in ids:
            if doc_id not in self._documents:
                continue

            doc = self._documents[doc_id]

            # Remove from term frequencies
            for token in set(doc.tokens):
                if token in self._term_freqs:
                    self._term_freqs[token].pop(doc_id, None)
                    if not self._term_freqs[token]:
                        del self._term_freqs[token]

            # Remove document
            del self._documents[doc_id]
            self._doc_ids.remove(doc_id)
            self._doc_len.pop(doc_id, None)

        # Recalculate statistics
        self._update_statistics()

        # Persist if configured
        if self.config.persist_path:
            self.save()

    def count(self) -> int:
        """Get number of indexed documents."""
        return len(self._documents)

    def clear(self) -> None:
        """Clear the entire index."""
        self._documents.clear()
        self._doc_ids.clear()
        self._idf.clear()
        self._doc_len.clear()
        self._term_freqs.clear()
        self._avgdl = 0.0

        if self.config.persist_path:
            self.save()

    def save(self) -> None:
        """Save index to disk."""
        if not self.config.persist_path:
            return

        path = Path(self.config.persist_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # [HARDENED] SEC-001: Use JSON instead of pickle to prevent arbitrary code execution
        state = {
            "documents": {
                doc_id: {
                    "id": doc.id,
                    "content": doc.content,
                    "tokens": doc.tokens,
                    "metadata": doc.metadata,
                }
                for doc_id, doc in self._documents.items()
            },
            "doc_ids": self._doc_ids,
            "idf": self._idf,
            "doc_len": self._doc_len,
            "avgdl": self._avgdl,
            "term_freqs": self._term_freqs,
        }

        with open(path, "w") as f:
            json.dump(state, f)

    def load(self) -> None:
        """Load index from disk."""
        if not self.config.persist_path:
            return

        path = Path(self.config.persist_path)
        if not path.exists():
            return

        # [HARDENED] SEC-001: Use JSON instead of pickle to prevent arbitrary code execution
        with open(path) as f:
            try:
                state = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Old pickle format or corrupted file — rebuild from scratch
                logger.warning("Could not load BM25 index from %s, rebuilding", path)
                return

        self._documents = {
            doc_id: IndexedDocument(
                id=doc_data["id"],
                content=doc_data["content"],
                tokens=doc_data["tokens"],
                metadata=doc_data.get("metadata", {}),
            )
            for doc_id, doc_data in state.get("documents", {}).items()
        }
        self._doc_ids = state.get("doc_ids", [])
        self._idf = state.get("idf", {})
        self._doc_len = state.get("doc_len", {})
        self._avgdl = state.get("avgdl", 0.0)
        self._term_freqs = state.get("term_freqs", {})

    def stats(self) -> dict:
        """Get index statistics."""
        return {
            "document_count": len(self._documents),
            "vocabulary_size": len(self._idf),
            "average_document_length": self._avgdl,
            "total_tokens": sum(self._doc_len.values()),
        }
