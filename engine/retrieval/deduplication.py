"""MinHash LSH deduplication for research evidence.

Uses locality-sensitive hashing for efficient near-duplicate detection.
Prevents redundant evidence from being saved to the database.
"""

import hashlib
import re
from dataclasses import dataclass

try:
    from datasketch import MinHash, MinHashLSH
    HAS_DATASKETCH = True
except ImportError:
    MinHash = None  # type: ignore
    MinHashLSH = None  # type: ignore
    HAS_DATASKETCH = False


@dataclass
class DeduplicationConfig:
    """Configuration for deduplication."""
    # MinHash parameters
    num_perm: int = 128  # Number of permutations (higher = more accurate, slower)
    threshold: float = 0.7  # Jaccard similarity threshold for duplicates

    # Text processing
    ngram_size: int = 3  # Character n-gram size for shingling
    min_content_length: int = 20  # Minimum content length to consider

    # Behavior
    check_exact_match: bool = True  # Also check for exact content matches
    normalize_whitespace: bool = True  # Normalize whitespace before hashing


@dataclass
class DeduplicationResult:
    """Result of a deduplication check."""
    is_duplicate: bool
    duplicate_of: str | None = None  # ID of the duplicate evidence
    similarity: float = 0.0
    match_type: str = "none"  # none, exact, near


class EvidenceDeduplicator:
    """MinHash LSH-based deduplicator for research evidence.

    Uses shingling (character n-grams) to create MinHash signatures,
    then uses LSH for efficient approximate nearest neighbor search.

    This enables O(1) duplicate checking regardless of the number of
    existing evidence items.

    Example:
        dedup = EvidenceDeduplicator()

        # Add existing evidence
        dedup.add("evidence-1", "Machine learning improves NLP tasks")
        dedup.add("evidence-2", "Deep learning enables image recognition")

        # Check new evidence
        result = dedup.check("ML improves natural language processing")
        if result.is_duplicate:
            print(f"Duplicate of {result.duplicate_of}")
    """

    def __init__(self, config: DeduplicationConfig | None = None):
        """Initialize the deduplicator.

        Args:
            config: Configuration options. Uses defaults if not provided.
        """
        self.config = config or DeduplicationConfig()

        if not HAS_DATASKETCH:
            self._enabled = False
            return

        self._enabled = True

        # LSH index for approximate matching
        self.lsh = MinHashLSH(
            threshold=self.config.threshold,
            num_perm=self.config.num_perm,
        )

        # Store MinHash signatures for similarity calculation
        self.signatures: dict[str, MinHash] = {}

        # Exact match index (normalized content hash -> evidence_id)
        self.exact_index: dict[str, str] = {}

        # Original content storage for debugging
        self.content_store: dict[str, str] = {}

    @property
    def enabled(self) -> bool:
        """Whether deduplication is enabled (datasketch installed)."""
        return self._enabled

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison.

        - Lowercase
        - Remove extra whitespace
        - Remove punctuation (optional)
        """
        text = text.lower()

        if self.config.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _get_shingles(self, text: str) -> set[str]:
        """Extract character n-grams (shingles) from text.

        Character-level shingles are more robust to word variations
        than word-level shingles.
        """
        text = self._normalize_text(text)
        n = self.config.ngram_size

        if len(text) < n:
            return {text}

        return {text[i:i+n] for i in range(len(text) - n + 1)}

    def _create_minhash(self, text: str) -> "MinHash":
        """Create MinHash signature from text."""
        m = MinHash(num_perm=self.config.num_perm)

        for shingle in self._get_shingles(text):
            m.update(shingle.encode('utf-8'))

        return m

    def _content_hash(self, text: str) -> str:
        """Create exact content hash for fast exact-match checking."""
        normalized = self._normalize_text(text)
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()

    def add(self, evidence_id: str, content: str) -> bool:
        """Add evidence to the deduplication index.

        Args:
            evidence_id: Unique identifier for the evidence
            content: The evidence content text

        Returns:
            True if added successfully, False if already exists or too short
        """
        if not self._enabled:
            return False

        if len(content) < self.config.min_content_length:
            return False

        # Check if already in index
        if evidence_id in self.signatures:
            return False

        # Add to exact index
        if self.config.check_exact_match:
            content_hash = self._content_hash(content)
            if content_hash not in self.exact_index:
                self.exact_index[content_hash] = evidence_id

        # Create and store MinHash
        minhash = self._create_minhash(content)
        self.signatures[evidence_id] = minhash
        self.content_store[evidence_id] = content

        # Add to LSH index
        try:
            self.lsh.insert(evidence_id, minhash)
        except ValueError:
            # Key already exists
            pass

        return True

    def check(self, content: str, exclude_id: str | None = None) -> DeduplicationResult:
        """Check if content is a duplicate of existing evidence.

        Args:
            content: The content to check
            exclude_id: Optional ID to exclude from matches (e.g., self-check)

        Returns:
            DeduplicationResult with duplicate status and details
        """
        if not self._enabled:
            return DeduplicationResult(is_duplicate=False)

        if len(content) < self.config.min_content_length:
            return DeduplicationResult(is_duplicate=False)

        # Check exact match first (fast)
        if self.config.check_exact_match:
            content_hash = self._content_hash(content)
            if content_hash in self.exact_index:
                dup_id = self.exact_index[content_hash]
                if dup_id != exclude_id:
                    return DeduplicationResult(
                        is_duplicate=True,
                        duplicate_of=dup_id,
                        similarity=1.0,
                        match_type="exact",
                    )

        # Check near-duplicate via LSH
        minhash = self._create_minhash(content)
        candidates = self.lsh.query(minhash)

        # Filter out excluded ID
        if exclude_id and exclude_id in candidates:
            candidates = [c for c in candidates if c != exclude_id]

        if not candidates:
            return DeduplicationResult(is_duplicate=False)

        # Find the most similar candidate
        best_match = None
        best_similarity = 0.0

        for candidate_id in candidates:
            if candidate_id in self.signatures:
                similarity = minhash.jaccard(self.signatures[candidate_id])
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = candidate_id

        if best_match and best_similarity >= self.config.threshold:
            return DeduplicationResult(
                is_duplicate=True,
                duplicate_of=best_match,
                similarity=best_similarity,
                match_type="near",
            )

        return DeduplicationResult(is_duplicate=False)

    def check_and_add(
        self,
        evidence_id: str,
        content: str,
    ) -> DeduplicationResult:
        """Check for duplicates and add if not a duplicate.

        Convenience method that combines check() and add().

        Args:
            evidence_id: Unique identifier for the evidence
            content: The evidence content text

        Returns:
            DeduplicationResult - if not duplicate, the evidence is added
        """
        result = self.check(content)

        if not result.is_duplicate:
            self.add(evidence_id, content)

        return result

    def remove(self, evidence_id: str) -> bool:
        """Remove evidence from the index.

        Note: MinHashLSH doesn't support removal well, so this only
        removes from our tracking structures. The LSH index may still
        return the ID in queries, but we won't validate it.

        Args:
            evidence_id: The evidence ID to remove

        Returns:
            True if removed, False if not found
        """
        if evidence_id not in self.signatures:
            return False

        # Remove from signatures
        del self.signatures[evidence_id]

        # Remove from content store
        if evidence_id in self.content_store:
            content = self.content_store.pop(evidence_id)
            # Remove from exact index
            content_hash = self._content_hash(content)
            if content_hash in self.exact_index and self.exact_index[content_hash] == evidence_id:
                del self.exact_index[content_hash]

        return True

    def get_stats(self) -> dict:
        """Get statistics about the deduplication index."""
        return {
            "enabled": self._enabled,
            "total_evidence": len(self.signatures),
            "exact_index_size": len(self.exact_index),
            "config": {
                "threshold": self.config.threshold,
                "num_perm": self.config.num_perm,
                "ngram_size": self.config.ngram_size,
            },
        }

    def clear(self) -> None:
        """Clear all indexed evidence."""
        if not self._enabled:
            return

        self.lsh = MinHashLSH(
            threshold=self.config.threshold,
            num_perm=self.config.num_perm,
        )
        self.signatures.clear()
        self.exact_index.clear()
        self.content_store.clear()


# Global instance for convenience
_global_deduplicator: EvidenceDeduplicator | None = None


def get_deduplicator(config: DeduplicationConfig | None = None) -> EvidenceDeduplicator:
    """Get or create the global deduplicator instance.

    Args:
        config: Configuration (only used on first call)

    Returns:
        The global EvidenceDeduplicator instance
    """
    global _global_deduplicator

    if _global_deduplicator is None:
        _global_deduplicator = EvidenceDeduplicator(config)

    return _global_deduplicator


def reset_deduplicator() -> None:
    """Reset the global deduplicator instance."""
    global _global_deduplicator

    if _global_deduplicator:
        _global_deduplicator.clear()
    _global_deduplicator = None
