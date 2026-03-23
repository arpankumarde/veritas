"""CRITIC verification implementation.

CRITIC is a tool-interactive verification method that:
1. Generates initial response (the evidence)
2. Critiques the response using external tools
3. Revises based on critique
4. Iterates until confident or max iterations

Reference: "CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing"
"""

import re
import time
from collections.abc import Callable
from typing import Any

from ..logging_config import get_logger
from .confidence import ConfidenceCalibrator
from .json_utils import parse_json_object
from .models import (
    VerificationConfig,
    VerificationMethod,
    VerificationResult,
    VerificationStatus,
)

logger = get_logger(__name__)

CRITIQUE_SCHEMA = {
    "type": "object",
    "properties": {
        "needs_correction": {"type": "boolean"},
        "issue": {"type": "string"},
        "issue_type": {
            "type": "string",
            "enum": [
                "factual",
                "logical",
                "temporal",
                "quantitative",
                "attribution",
            ],
        },
        "severity": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
        "verify_externally": {"type": "boolean"},
        "search_query": {"type": "string"},
        "suggestion": {"type": "string"},
    },
    "required": ["needs_correction"],
}


class CRITICVerifier:
    """CRITIC verification with tool-interactive critiquing.

    Uses external verification (web search) to validate claims
    and iteratively corrects evidence until confident.
    """

    def __init__(
        self,
        llm_callback: Callable,  # (prompt, model, **kwargs) -> response
        search_callback: Callable[[str], Any] | None = None,  # query -> results
        config: VerificationConfig | None = None,
    ):
        """Initialize CRITIC verifier.

        Args:
            llm_callback: Async function to call LLM with (prompt, model)
            search_callback: Optional async function to perform web search
            config: Verification configuration
        """
        self.llm_callback = llm_callback
        self.search_callback = search_callback
        self.config = config or VerificationConfig()
        self.calibrator = ConfidenceCalibrator(config)

    async def verify(
        self,
        evidence_content: str,
        evidence_id: str,
        original_confidence: float,
        source_url: str | None = None,
        cove_result: VerificationResult | None = None,
    ) -> VerificationResult:
        """Run CRITIC verification with iterative critique and correction.

        Args:
            evidence_content: The content of the evidence to verify
            evidence_id: Unique identifier for the evidence
            original_confidence: Original confidence score (0-1)
            source_url: Optional source URL for context
            cove_result: Optional previous CoVe result to build on

        Returns:
            VerificationResult with corrections and calibrated confidence
        """
        start_time = time.time()
        logger.info("CRITIC verify: evidence=%s", evidence_id)

        current_content = evidence_content
        corrections = []
        iteration = 0
        current_confidence = original_confidence
        external_verification_used = False

        # Get initial signals from CoVe if available
        consistency_score = (cove_result.consistency_score or 0.0) if cove_result else 0.0
        kg_support = (cove_result.kg_support_score or 0.0) if cove_result else 0.0
        questions_asked = cove_result.questions_asked if cove_result else []

        try:
            # Pre-fetch initial evidence so the first critique is grounded
            initial_evidence = None
            if self.search_callback:
                initial_evidence = await self._external_verify(current_content[:150])
                if initial_evidence:
                    external_verification_used = True

            while iteration < self.config.max_critic_iterations:
                iteration += 1

                # Step 1: Critique the content with evidence context
                evidence_for_critique = initial_evidence if iteration == 1 else None
                critique = await self._generate_critique(
                    current_content,
                    source_url,
                    external_evidence=evidence_for_critique,
                )

                # Step 2: Always fetch fresh evidence for correction decisions.
                # Don't let the LLM decide whether to search -- external
                # evidence is the whole point of CRITIC. Without it, CRITIC
                # degrades into unreliable self-correction.
                if self.search_callback:
                    search_query = critique.get("search_query", current_content[:100])
                    search_results = await self._external_verify(search_query)
                    if search_results:
                        external_verification_used = True
                        critique["external_evidence"] = search_results

                if not critique.get("needs_correction"):
                    # Critique says it's fine (after seeing evidence), we're done
                    break

                # Step 3: Generate correction
                corrected = await self._generate_correction(current_content, critique)

                if corrected and corrected != current_content:
                    corrections.append(
                        f"Iteration {iteration}: {critique.get('issue', 'Correction made')}"
                    )
                    current_content = corrected
                else:
                    # No meaningful correction possible
                    break

                # Step 4: Update confidence based on correction
                # Each correction slightly reduces confidence (original was wrong)
                current_confidence *= 0.9

            # Final calibration
            calibration = self.calibrator.calibrate(
                original_confidence=original_confidence,
                cove_consistency_score=consistency_score,
                kg_support_score=kg_support,
                critic_corrections=len(corrections),
            )

            return VerificationResult(
                evidence_id=evidence_id,
                original_confidence=original_confidence,
                verified_confidence=calibration.calibrated_confidence,
                verification_status=calibration.status,
                verification_method=VerificationMethod.CRITIC
                if not cove_result
                else VerificationMethod.COVE_CRITIC,
                questions_asked=questions_asked,
                consistency_score=consistency_score,
                kg_support_score=kg_support,
                critic_iterations=iteration,
                corrections_made=corrections,
                external_verification_used=external_verification_used,
                verification_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.warning("CRITIC error", exc_info=True)
            return VerificationResult(
                evidence_id=evidence_id,
                original_confidence=original_confidence,
                verified_confidence=original_confidence,
                verification_status=VerificationStatus.SKIPPED,
                verification_method=VerificationMethod.CRITIC,
                critic_iterations=iteration,
                corrections_made=corrections,
                verification_time_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )

    async def _generate_critique(
        self,
        content: str,
        source_url: str | None,
        external_evidence: str | None = None,
    ) -> dict:
        """Generate critique of the evidence content.

        When external evidence is provided (from web search), the critique
        is grounded in real data rather than just the LLM's parametric
        knowledge.
        """
        source_context = f"\nSource: {source_url}" if source_url else ""
        evidence_block = ""
        if external_evidence:
            evidence_block = (
                f"\n\nWEB EVIDENCE (use this to evaluate the evidence):\n{external_evidence}\n"
            )

        prompt = (
            "Critically analyze this research evidence for potential "
            "issues.\n\n"
            f"EVIDENCE: {content}{source_context}"
            f"{evidence_block}\n\n"
            "Check for:\n"
            "1. Factual accuracy - Are the stated facts verifiable?\n"
            "2. Logical consistency - Are there contradictions?\n"
            "3. Temporal accuracy - Are dates/timeframes plausible?\n"
            "4. Quantitative claims - Are numbers reasonable?\n"
            "5. Source attribution - Is the claim properly attributed?\n\n"
            "Always provide a search_query that could verify the core "
            "claim, even if you think it's correct."
        )

        schema_fmt = {"type": "json_schema", "schema": CRITIQUE_SCHEMA}
        try:
            response = await self.llm_callback(
                prompt,
                self.config.batch_model,
                output_format=schema_fmt,
            )
        except TypeError:
            response = await self.llm_callback(
                prompt,
                self.config.batch_model,
            )

        if isinstance(response, dict):
            return response

        return self._parse_json(response) or {"needs_correction": False}

    async def _external_verify(self, query: str) -> str | None:
        """Perform external verification via web search."""
        if not self.search_callback:
            return None

        try:
            results = await self.search_callback(query)
            if results:
                # Format results for LLM consumption
                if isinstance(results, list):
                    return "\n".join(
                        [
                            f"- {r.get('title', '')}: {r.get('snippet', '')[:200]}"
                            for r in results[:5]
                        ]
                    )
                return str(results)[:1000]
        except Exception:
            logger.warning("CRITIC error", exc_info=True)

        return None

    async def _generate_correction(
        self,
        content: str,
        critique: dict,
    ) -> str | None:
        """Generate corrected version of the evidence."""
        issue = critique.get("issue", "Unknown issue")
        suggestion = critique.get("suggestion", "")
        external_evidence = critique.get("external_evidence", "")

        external_context = ""
        if external_evidence:
            external_context = f"\n\nEXTERNAL EVIDENCE:\n{external_evidence}"

        prompt = f"""Correct this research evidence based on the identified issue.

ORIGINAL EVIDENCE: {content}

ISSUE: {issue}
SUGGESTION: {suggestion}{external_context}

Rules:
1. Only fix the identified issue
2. Preserve accurate parts of the original
3. If you cannot confidently correct it, return the original unchanged
4. Keep the same format and length

Return ONLY the corrected evidence text, no explanation."""

        response = await self.llm_callback(prompt, self.config.batch_model)

        # Handle non-string responses (e.g., dict from structured output)
        if response is None:
            return None
        if isinstance(response, dict):
            response = response.get("corrected_text", response.get("text", str(response)))
        if not isinstance(response, str):
            response = str(response)

        # Clean up response
        corrected = response.strip()
        if corrected.startswith('"') and corrected.endswith('"'):
            corrected = corrected[1:-1]

        return corrected if corrected else None

    def _parse_json(self, response: str) -> dict | None:
        return parse_json_object(response)


class HighStakesDetector:
    """Detect high-stakes evidence that warrants CRITIC verification.

    High-stakes evidence includes:
    - Quantitative claims (numbers, statistics)
    - Causal claims (X causes Y)
    - Medical/health claims
    - Financial/economic claims
    - Safety-related claims
    """

    HIGH_STAKES_PATTERNS = [
        # Quantitative
        r"\d+%",
        r"\$\d+",
        r"\d+\s*(million|billion|trillion)",
        r"increased by \d+",
        r"decreased by \d+",
        # Causal
        r"causes",
        r"leads to",
        r"results in",
        r"prevents",
        r"because of",
        # Medical
        r"treat(s|ed|ment)",
        r"cures?",
        r"disease",
        r"symptom",
        r"diagnos",
        r"mortality",
        # Financial
        r"invest(ment)?",
        r"profit",
        r"revenue",
        r"stock",
        r"market cap",
        # Safety
        r"danger(ous)?",
        r"risk",
        r"harm",
        r"safe(ty)?",
        r"warning",
    ]

    def __init__(self):
        self._patterns = [re.compile(p, re.IGNORECASE) for p in self.HIGH_STAKES_PATTERNS]

    def is_high_stakes(self, content: str) -> bool:
        """Check if evidence is high-stakes and warrants CRITIC."""
        return any(p.search(content) for p in self._patterns)

    def get_stakes_type(self, content: str) -> list[str]:
        """Get the types of high-stakes claims in the content."""
        stakes_types = []
        content_lower = content.lower()

        if any(p in content_lower for p in ["%", "million", "billion", "increased", "decreased"]):
            stakes_types.append("quantitative")
        if any(p in content_lower for p in ["causes", "leads to", "results in", "prevents"]):
            stakes_types.append("causal")
        if any(p in content_lower for p in ["treat", "cure", "disease", "diagnos", "mortality"]):
            stakes_types.append("medical")
        if any(p in content_lower for p in ["invest", "profit", "revenue", "stock"]):
            stakes_types.append("financial")
        if any(p in content_lower for p in ["danger", "risk", "harm", "safe", "warning"]):
            stakes_types.append("safety")

        return stakes_types
