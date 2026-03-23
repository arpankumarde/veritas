"""Chain-of-Verification (CoVe) implementation.

CoVe is a 4-step verification process:
1. Generate baseline response (already have the evidence)
2. Plan verification questions
3. Execute independent verification (answer questions without seeing original)
4. Generate final verified response

Reference: "Chain-of-Verification Reduces Hallucination in Large Language Models"
"""

import asyncio
import re
import time
from collections.abc import Callable

from ..logging_config import get_logger
from .confidence import ConfidenceCalibrator
from .json_utils import extract_json_array, parse_json_object
from .models import (
    VerificationConfig,
    VerificationMethod,
    VerificationQuestion,
    VerificationResult,
    VerificationStatus,
)

logger = get_logger(__name__)

# JSON schemas for structured output
QUESTIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "aspect": {
                        "type": "string",
                        "enum": [
                            "factual",
                            "temporal",
                            "source",
                            "quantitative",
                            "causal",
                        ],
                    },
                },
                "required": ["question", "aspect"],
            },
        },
    },
    "required": ["questions"],
}

INDEPENDENT_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["answer", "confidence"],
}

COMPARISON_SCHEMA = {
    "type": "object",
    "properties": {
        "supports": {"type": "boolean"},
        "reasoning": {"type": "string"},
    },
    "required": ["supports", "reasoning"],
}


class ChainOfVerification:
    """Chain-of-Verification for fact verification.

    Implements a 4-step verification process:
    1. Baseline: The original evidence (already have this)
    2. Plan: Generate verification questions
    3. Execute: Answer questions independently (with optional web search)
    4. Verify: Check consistency and calibrate confidence
    """

    def __init__(
        self,
        llm_callback: Callable,  # (prompt, model, **kwargs) -> response
        config: VerificationConfig | None = None,
        search_callback: Callable | None = None,
    ):
        """Initialize CoVe verifier.

        Args:
            llm_callback: Async function to call LLM. Accepts (prompt, model)
                and optionally output_format kwarg for structured output.
            config: Verification configuration
            search_callback: Optional async function for web search.
                Accepts (query) and returns search results.
        """
        self.llm_callback = llm_callback
        self.config = config or VerificationConfig()
        self.calibrator = ConfidenceCalibrator(config)
        self.search_callback = search_callback

    async def verify_streaming(
        self,
        evidence_content: str,
        evidence_id: str,
        original_confidence: float,
        source_url: str | None = None,
        search_query: str | None = None,
    ) -> VerificationResult:
        """Streaming verification - fast, lightweight for real-time use.

        Target latency: <500ms
        Uses 1-2 questions and Haiku model.

        Args:
            evidence_content: The content of the evidence to verify
            evidence_id: Unique identifier for the evidence
            original_confidence: Original confidence score (0-1)
            source_url: Optional source URL for context
            search_query: Optional search query that produced this evidence

        Returns:
            VerificationResult with calibrated confidence
        """
        start_time = time.time()
        logger.info("CoVe streaming: evidence=%s, confidence=%.2f", evidence_id, original_confidence)

        try:
            # Step 1: Generate 1-2 quick verification questions
            questions, question_error = await self._generate_questions_streaming(
                evidence_content, source_url, search_query
            )

            if not questions:
                logger.warning("CoVe streaming failed for %s: %s", evidence_id, question_error)
                return VerificationResult(
                    evidence_id=evidence_id,
                    original_confidence=original_confidence,
                    verified_confidence=original_confidence,
                    verification_status=VerificationStatus.SKIPPED,
                    verification_method=VerificationMethod.STREAMING,
                    verification_time_ms=(time.time() - start_time) * 1000,
                    error=question_error or "No verification questions generated (unknown reason)",
                )

            # Step 2: Answer questions independently (parallel for speed)
            answered_questions = await self._answer_questions_parallel(questions, evidence_content)

            # Step 3: Calculate consistency and calibrate
            consistency_score = self._calculate_consistency(answered_questions)

            calibration = self.calibrator.calibrate(
                original_confidence=original_confidence,
                cove_consistency_score=consistency_score,
            )

            return VerificationResult(
                evidence_id=evidence_id,
                original_confidence=original_confidence,
                verified_confidence=calibration.calibrated_confidence,
                verification_status=calibration.status,
                verification_method=VerificationMethod.STREAMING,
                questions_asked=answered_questions,
                consistency_score=consistency_score,
                verification_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error("CoVe streaming error for %s: %s", evidence_id, e, exc_info=True)
            return VerificationResult(
                evidence_id=evidence_id,
                original_confidence=original_confidence,
                verified_confidence=original_confidence,
                verification_status=VerificationStatus.SKIPPED,
                verification_method=VerificationMethod.STREAMING,
                verification_time_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )

    async def verify_batch(
        self,
        evidence_content: str,
        evidence_id: str,
        original_confidence: float,
        source_url: str | None = None,
        search_query: str | None = None,
        kg_support_score: float = 0.0,
        has_contradictions: bool = False,
    ) -> VerificationResult:
        """Batch verification - comprehensive, thorough analysis.

        Target latency: ~2s per piece of evidence
        Uses 3-5 questions and Sonnet model.

        Args:
            evidence_content: The content of the evidence to verify
            evidence_id: Unique identifier for the evidence
            original_confidence: Original confidence score (0-1)
            source_url: Optional source URL for context
            search_query: Optional search query that produced this evidence
            kg_support_score: KG corroboration score (0-1)
            has_contradictions: Whether KG found contradictions

        Returns:
            VerificationResult with calibrated confidence
        """
        start_time = time.time()
        logger.info("CoVe batch: evidence=%s, confidence=%.2f", evidence_id, original_confidence)

        try:
            # Step 1: Generate 3-5 comprehensive verification questions
            questions, question_error = await self._generate_questions_batch(
                evidence_content, source_url, search_query
            )

            if not questions:
                logger.warning("CoVe batch failed for %s: %s", evidence_id, question_error)
                return VerificationResult(
                    evidence_id=evidence_id,
                    original_confidence=original_confidence,
                    verified_confidence=original_confidence,
                    verification_status=VerificationStatus.SKIPPED,
                    verification_method=VerificationMethod.BATCH,
                    verification_time_ms=(time.time() - start_time) * 1000,
                    error=question_error or "No verification questions generated (unknown reason)",
                )

            # Step 2: Answer questions independently
            answered_questions = await self._answer_questions_parallel(
                questions, evidence_content, use_batch_model=True
            )

            # Step 3: Calculate consistency
            consistency_score = self._calculate_consistency(answered_questions)

            # Step 4: Calibrate with all signals
            calibration = self.calibrator.calibrate(
                original_confidence=original_confidence,
                cove_consistency_score=consistency_score,
                kg_support_score=kg_support_score,
                has_contradictions=has_contradictions,
            )

            return VerificationResult(
                evidence_id=evidence_id,
                original_confidence=original_confidence,
                verified_confidence=calibration.calibrated_confidence,
                verification_status=calibration.status,
                verification_method=VerificationMethod.BATCH,
                questions_asked=answered_questions,
                consistency_score=consistency_score,
                kg_support_score=kg_support_score,
                verification_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error("CoVe batch error for %s: %s", evidence_id, e, exc_info=True)
            return VerificationResult(
                evidence_id=evidence_id,
                original_confidence=original_confidence,
                verified_confidence=original_confidence,
                verification_status=VerificationStatus.SKIPPED,
                verification_method=VerificationMethod.BATCH,
                verification_time_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )

    async def _generate_questions_streaming(
        self,
        evidence_content: str,
        source_url: str | None,
        search_query: str | None,
    ) -> tuple[list[VerificationQuestion], str | None]:
        """Generate 1-2 quick verification questions.

        Returns:
            Tuple of (questions, error_reason). error_reason is None on success.
        """
        context = f"Source: {source_url}" if source_url else ""
        if search_query:
            context += f"\nSearch query: {search_query}"

        prompt = (
            f"Generate 1-2 quick verification questions for this claim.\n\n"
            f"CLAIM: {evidence_content}\n{context}\n\n"
            f"Questions should check the most critical facts.\n"
            f"Aspects: factual, temporal, source, quantitative, causal"
        )

        return await self._call_for_questions(
            prompt,
            self.config.streaming_model,
            max_q=self.config.max_cove_questions_streaming,
        )

    async def _generate_questions_batch(
        self,
        evidence_content: str,
        source_url: str | None,
        search_query: str | None,
    ) -> tuple[list[VerificationQuestion], str | None]:
        """Generate 3-5 comprehensive verification questions.

        Returns:
            Tuple of (questions, error_reason). error_reason is None on success.
        """
        context = f"Source: {source_url}" if source_url else ""
        if search_query:
            context += f"\nSearch query: {search_query}"

        prompt = (
            f"Generate 3-5 verification questions to thoroughly check "
            f"this claim.\n\n"
            f"CLAIM: {evidence_content}\n{context}\n\n"
            f"Cover different verification aspects:\n"
            f"- Factual accuracy (are the stated facts correct?)\n"
            f"- Temporal accuracy (are dates/timeframes correct?)\n"
            f"- Source attribution (is the source reliable?)\n"
            f"- Quantitative claims (are numbers accurate?)\n"
            f"- Causal claims (are cause-effect relationships valid?)"
        )

        return await self._call_for_questions(
            prompt,
            self.config.batch_model,
            max_q=self.config.max_cove_questions_batch,
        )

    async def _call_for_questions(
        self,
        prompt: str,
        model: str,
        max_q: int,
    ) -> tuple[list[VerificationQuestion], str | None]:
        """Call LLM for verification questions using structured output.

        Tries structured output first, falls back to text parsing.

        Returns:
            Tuple of (questions, error_reason). error_reason is None on success.
        """
        schema_fmt = {
            "type": "json_schema",
            "schema": QUESTIONS_SCHEMA,
        }

        try:
            try:
                response = await self.llm_callback(
                    prompt,
                    model,
                    output_format=schema_fmt,
                )
            except TypeError:
                # Callback doesn't support output_format kwarg -- fall back
                response = await self.llm_callback(prompt, model)
        except Exception as e:
            logger.error("CoVe question generation LLM error: %s", e, exc_info=True)
            return [], f"LLM call failed: {type(e).__name__}: {e}"

        # If structured output returned a dict directly, use it
        if isinstance(response, dict):
            items = response.get("questions", [])
            questions = [
                VerificationQuestion(
                    question=q["question"],
                    aspect=q.get("aspect", "factual"),
                )
                for q in items
                if isinstance(q, dict) and q.get("question")
            ][:max_q]
            if not questions:
                return [], "LLM returned structured response but no valid questions"
            return questions, None

        # Guard against None/empty response
        if not response:
            logger.warning("CoVe got empty LLM response for questions")
            return [], "LLM returned empty response"

        # Fallback: parse text response
        parsed = self._parse_questions(response, max_questions=max_q)
        if not parsed:
            return [], f"Could not parse questions from LLM text response ({len(response)} chars)"
        return parsed, None

    async def _search_for_evidence(self, question: str) -> str:
        """Search the web for evidence to answer a verification question.

        Uses the search callback to find relevant evidence. Formats results
        with titles, URLs, and snippets for the LLM to evaluate.

        Returns formatted evidence string, or empty string if unavailable.
        """
        if not self.search_callback:
            return ""

        try:
            results = await self.search_callback(question)
            if not results:
                return ""
            if isinstance(results, list):
                evidence_parts = []
                for r in results[:5]:
                    title = r.get("title", "") if isinstance(r, dict) else ""
                    snippet = r.get("snippet", "") if isinstance(r, dict) else str(r)
                    url = r.get("url", "") if isinstance(r, dict) else ""
                    # Include full page content if available (scrape callback)
                    content = r.get("content", "") if isinstance(r, dict) else ""
                    if content:
                        evidence_parts.append(f"SOURCE [{title}] ({url}):\n{content[:800]}")
                    elif snippet:
                        evidence_parts.append(f"SOURCE [{title}] ({url}): {snippet[:400]}")
                return "\n\n".join(evidence_parts) if evidence_parts else ""
            return str(results)[:1200]
        except Exception:
            logger.debug("Evidence search failed for question: %s", question[:100], exc_info=True)
            return ""

    async def _answer_questions_parallel(
        self,
        questions: list[VerificationQuestion],
        original_evidence: str,
        use_batch_model: bool = False,
    ) -> list[VerificationQuestion]:
        """Answer verification questions using factored CoVe (two-step).

        Per the CoVe paper, step 3 must answer questions INDEPENDENTLY
        (without seeing the original claim) so answers are not biased.
        A separate comparison step then checks support/contradiction.

        Step 1: Search web for evidence, then answer the question independently.
        Step 2: Compare the independent answer against the original claim.
        """
        model = self.config.batch_model if use_batch_model else self.config.streaming_model
        answer_schema = {"type": "json_schema", "schema": INDEPENDENT_ANSWER_SCHEMA}
        compare_schema = {"type": "json_schema", "schema": COMPARISON_SCHEMA}

        async def answer_and_compare(
            q: VerificationQuestion,
        ) -> VerificationQuestion:
            try:
                # --- Step 1a: Search web for evidence (if available) ---
                evidence = await self._search_for_evidence(q.question)

                # --- Step 1b: Independent answer with evidence context ---
                evidence_block = ""
                if evidence:
                    evidence_block = f"\n\nWEB EVIDENCE:\n{evidence}\n"
                    answer_prompt = (
                        "Based on the web evidence below, answer this "
                        "factual question. Your answer MUST be based on "
                        "what the evidence says -- quote or reference "
                        "specific parts. Do NOT say you need to search "
                        "or access a source -- the evidence is already "
                        "provided.\n\n"
                        f"QUESTION: {q.question}\n"
                        f"ASPECT: {q.aspect}"
                        f"{evidence_block}\n\n"
                        "Respond with:\n"
                        '- "answer": your factual answer based on the '
                        "evidence above\n"
                        '- "confidence": 0.0-1.0 (how strongly the '
                        "evidence supports your answer)"
                    )
                else:
                    answer_prompt = (
                        "Answer this factual question using your "
                        "knowledge. Provide your best assessment of "
                        "whether the stated fact is likely true, false, "
                        "or uncertain. Do NOT say you need to search or "
                        "cannot verify -- just give your best answer.\n\n"
                        f"QUESTION: {q.question}\n"
                        f"ASPECT: {q.aspect}\n\n"
                        "Respond with:\n"
                        '- "answer": your best factual answer\n'
                        '- "confidence": 0.0-1.0 (use lower confidence '
                        "if uncertain, but always answer)"
                    )

                try:
                    answer_resp = await self.llm_callback(
                        answer_prompt,
                        model,
                        output_format=answer_schema,
                    )
                except TypeError:
                    answer_resp = await self.llm_callback(answer_prompt, model)

                if isinstance(answer_resp, dict):
                    answer_data = answer_resp
                elif answer_resp:
                    answer_data = self._parse_json(answer_resp)
                    # Fallback: structured output failed and JSON parse
                    # failed -- use the raw text as the answer rather than
                    # discarding it entirely (the comparison step will
                    # still evaluate it against the claim)
                    if not answer_data and isinstance(answer_resp, str):
                        text = answer_resp.strip()
                        if len(text) > 10:
                            answer_data = {"answer": text, "confidence": 0.5}
                else:
                    answer_data = None

                if not answer_data or not answer_data.get("answer"):
                    q.independent_answer = None
                    q.confidence = 0.0
                    q.supports_original = None
                    return q

                q.independent_answer = answer_data["answer"]
                q.confidence = answer_data.get("confidence", 0.5)

                # --- Guard: detect "can't access" non-answers ---
                # If the LLM still says it can't verify, treat as neutral
                # (not a contradiction) with low confidence
                _refusal_markers = [
                    "cannot access",
                    "can't access",
                    "cannot verify",
                    "can't verify",
                    "not provided",
                    "no access",
                    "unable to verify",
                    "unable to access",
                    "don't have access",
                    "without access",
                    "not available",
                    "cannot be determined",
                    "i'll search",
                    "i will search",
                    "let me search",
                    "let me use",
                    "i need to search",
                    "i would need",
                    "i need access",
                    "need to look up",
                    "search for",
                    "look up the",
                    "use the web search",
                    "web search tool",
                ]
                answer_lower = q.independent_answer.lower()
                if any(m in answer_lower for m in _refusal_markers):
                    q.confidence = 0.1
                    q.supports_original = None  # Neutral, not contradicting
                    return q

                # --- Step 2: Compare independent answer against claim ---
                compare_prompt = (
                    "Does this independent answer broadly support or "
                    "contradict the claim below?\n\n"
                    f"CLAIM: {original_evidence}\n\n"
                    f"QUESTION: {q.question}\n"
                    f"INDEPENDENT ANSWER: {q.independent_answer}\n\n"
                    "IMPORTANT: Only mark as contradicts if the answer "
                    "directly opposes the claim's core conclusion. "
                    "Differences in wording, framing, or level of detail "
                    "are NOT contradictions -- if the overall direction "
                    "agrees, that counts as support.\n\n"
                    "Respond with:\n"
                    '- "supports": true if the answer broadly agrees with '
                    "the claim's main point, false ONLY if it directly "
                    "contradicts or refutes the claim\n"
                    '- "reasoning": brief explanation of why'
                )

                try:
                    compare_resp = await self.llm_callback(
                        compare_prompt,
                        model,
                        output_format=compare_schema,
                    )
                except TypeError:
                    compare_resp = await self.llm_callback(
                        compare_prompt,
                        model,
                    )

                if isinstance(compare_resp, dict):
                    compare_data = compare_resp
                elif compare_resp:
                    compare_data = self._parse_json(compare_resp)
                else:
                    compare_data = None

                if compare_data:
                    q.supports_original = compare_data.get("supports", True)
                else:
                    q.supports_original = None

            except Exception as e:
                q.independent_answer = None
                q.confidence = 0.0
                q.supports_original = None
                q.error = f"{type(e).__name__}: {e}"
                logger.warning("CoVe answer failed for question '%s': %s", q.question[:100], e, exc_info=True)

            return q

        # Run all questions in parallel with return_exceptions=True so a
        # single failure (including CancelledError from parent timeout)
        # doesn't discard all other answers.
        try:
            results = await asyncio.gather(
                *[answer_and_compare(q) for q in questions],
                return_exceptions=True,
            )
        except asyncio.CancelledError:
            logger.warning("CoVe answer_questions_parallel was cancelled")
            for q in questions:
                if q.independent_answer is None:
                    q.confidence = 0.0
                    q.supports_original = None
                    q.error = "Cancelled"
            return list(questions)

        answered = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                logger.warning("CoVe answer failed for question %d: %s", i, result)
                q = questions[i]
                q.independent_answer = None
                q.confidence = 0.0
                q.supports_original = None
                q.error = f"{type(result).__name__}: {result}"
                answered.append(q)
            else:
                answered.append(result)
        return answered

    def _calculate_consistency(self, questions: list[VerificationQuestion]) -> float:
        """Calculate consistency score from answered questions.

        Returns 0-1 score based on:
        - Support ratio: how many questions support vs contradict (primary signal)
        - Answered ratio: penalty for unanswered questions (data quality)
        - LLM confidence: minor weight (LLMs are systematically overconfident)
        - Questions with unknown support (None) are treated as neutral
        """
        if not questions:
            return 0.5  # No data = neutral, not penalizing

        supporting = 0
        contradicting = 0
        total_confidence = 0.0
        answered_count = 0

        for q in questions:
            if q.independent_answer is not None:
                answered_count += 1
                total_confidence += q.confidence
                # Only count explicit True/False, skip None (unknown)
                if q.supports_original is True:
                    supporting += 1
                elif q.supports_original is False:
                    contradicting += 1

        if answered_count == 0:
            return 0.5  # No data = neutral

        # Support ratio: supporting vs contradicting (ignore unknowns)
        decided_count = supporting + contradicting
        if decided_count > 0:
            support_ratio = supporting / decided_count
        else:
            # All answers have unknown support -> neutral
            support_ratio = 0.5

        avg_confidence = total_confidence / answered_count
        answered_ratio = answered_count / len(questions)

        # Weighted consistency score:
        # - 80% support ratio (the actual verify/contradict signal)
        # - 10% answered ratio (penalty for unanswered questions)
        # - 10% LLM confidence (unreliable, kept at low weight)
        return (support_ratio * 0.80) + (answered_ratio * 0.10) + (avg_confidence * 0.10)

    def _parse_questions(self, response: str, max_questions: int | None = None) -> list[VerificationQuestion]:
        """Parse verification questions from LLM response.

        Tries JSON first, then falls back to extracting plain-text questions
        (numbered lists, bullet points, or sentences ending with '?').
        """
        questions = []

        # Strategy 1: JSON array
        data = self._extract_json_array(response)
        if data:
            for item in data:
                if isinstance(item, dict) and item.get("question"):
                    questions.append(
                        VerificationQuestion(
                            question=item["question"],
                            aspect=item.get("aspect", "factual"),
                        )
                    )

        # Strategy 2: Plain text fallback -- extract lines that look like questions
        if not questions:
            for line in response.splitlines():
                line = line.strip()
                # Strip numbered/bulleted prefixes: "1. ", "- ", "* ", "1) "
                line = re.sub(r"^(?:\d+[.)]\s*|[-*•]\s*)", "", line).strip()
                if line.endswith("?") and len(line) > 15:
                    questions.append(
                        VerificationQuestion(question=line, aspect="factual")
                    )

        # Limit to configured max
        limit = max_questions if max_questions is not None else self.config.max_cove_questions_batch
        return questions[:limit]

    def _extract_json_array(self, response: str) -> list | None:
        return extract_json_array(response)

    def _parse_json(self, response: str) -> dict | None:
        return parse_json_object(response)
