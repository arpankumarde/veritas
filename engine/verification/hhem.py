"""Source-grounding scorer using Vectara HHEM-2.1-Open.

Scores (source_text, evidence_text) pairs for factual consistency.
0.0 = hallucinated, 1.0 = fully grounded in source.
"""

import asyncio

from ..logging_config import get_logger

logger = get_logger(__name__)

# Maximum source text length to keep inference fast
_MAX_SOURCE_CHARS = 2000


class HHEMScorer:
    """Source-grounding scorer using Vectara HHEM-2.1-Open.

    Lazy-loads the model (~440MB) on first use. Runs inference in a thread
    pool since PyTorch is blocking. Returns -1.0 on any failure so it never
    blocks research.
    """

    def __init__(self):
        self._model = None
        self._lock = asyncio.Lock()

    def _load_model(self):
        """Lazy-load on first use. Cached by HuggingFace after first download."""
        from transformers import AutoModelForSequenceClassification

        self._model = AutoModelForSequenceClassification.from_pretrained(
            "vectara/hallucination_evaluation_model",
            trust_remote_code=True,
        )

    def _ensure_model(self):
        """Ensure model is loaded, loading it if necessary."""
        if self._model is None:
            self._load_model()

    def _score_sync(self, source_text: str, evidence_text: str) -> float:
        """Synchronous scoring (runs in thread pool)."""
        self._ensure_model()
        # Truncate source to keep inference fast
        source = source_text[:_MAX_SOURCE_CHARS]
        pairs = [(source, evidence_text)]
        scores = self._model.predict(pairs)
        return float(scores[0])

    def _score_batch_sync(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Synchronous batch scoring (runs in thread pool)."""
        self._ensure_model()
        truncated = [(s[:_MAX_SOURCE_CHARS], f) for s, f in pairs]
        scores = self._model.predict(truncated)
        return [float(s) for s in scores]

    async def score(self, source_text: str, evidence_text: str) -> float:
        """Score grounding of evidence against its source text.

        Args:
            source_text: The original source content the evidence was extracted from.
            evidence_text: The extracted evidence claim.

        Returns:
            Float 0.0 (hallucinated) to 1.0 (grounded), or -1.0 on failure.
        """
        try:
            async with self._lock:
                return await asyncio.to_thread(self._score_sync, source_text, evidence_text)
        except Exception as e:
            logger.warning("HHEM scoring failed: %s", e)
            return -1.0

    async def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score multiple (source_text, evidence_text) pairs at once.

        Args:
            pairs: List of (source_text, evidence_text) tuples.

        Returns:
            List of scores, or list of -1.0 on failure.
        """
        if not pairs:
            return []
        try:
            async with self._lock:
                return await asyncio.to_thread(self._score_batch_sync, pairs)
        except Exception as e:
            logger.warning("HHEM batch scoring failed: %s", e)
            return [-1.0] * len(pairs)
