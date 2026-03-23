"""Verification pipeline orchestrator."""

import asyncio
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional

from ..events import emit_agent_event
from ..logging_config import get_logger
from .confidence import ConfidenceCalibrator
from .cove import ChainOfVerification
from .critic import CRITICVerifier, HighStakesDetector
from .hhem import HHEMScorer
from .metrics import VerificationMetricsTracker
from .models import (
    BatchVerificationResult,
    ContradictionDetail,
    VerificationConfig,
    VerificationMethod,
    VerificationResult,
    VerificationStatus,
)

logger = get_logger(__name__)

if TYPE_CHECKING:
    from ..knowledge.graph import IncrementalKnowledgeGraph
    from ..models.evidence import Evidence


class VerificationPipeline:
    """Main orchestrator for the verification system.

    Coordinates:
    - Streaming verification for real-time evidence submission
    - Batch verification during synthesis
    - KG integration for corroboration
    - CRITIC for high-stakes evidence
    """

    def __init__(
        self,
        llm_callback: Callable[[str, str], Any],  # (prompt, model) -> response
        knowledge_graph: Optional["IncrementalKnowledgeGraph"] = None,
        search_callback: Callable[[str], Any] | None = None,
        config: VerificationConfig | None = None,
    ):
        """Initialize the verification pipeline.

        Args:
            llm_callback: Async function to call LLM with (prompt, model)
            knowledge_graph: Optional KG for corroboration
            search_callback: Optional async function for web search (CRITIC)
            config: Verification configuration
        """
        self.config = config or VerificationConfig()
        self.llm_callback = llm_callback
        self.knowledge_graph = knowledge_graph
        self.search_callback = search_callback

        # Initialize components -- pass search_callback to CoVe so it can
        # independently verify claims via web search instead of relying
        # solely on the LLM's parametric knowledge.
        self.cove = ChainOfVerification(
            llm_callback,
            self.config,
            search_callback=search_callback,
        )
        self.critic = CRITICVerifier(llm_callback, search_callback, self.config)
        self.calibrator = ConfidenceCalibrator(self.config)
        self.high_stakes_detector = HighStakesDetector()
        self.metrics = VerificationMetricsTracker()
        self.hhem = HHEMScorer() if self.config.enable_hhem else None

    async def verify_intern_evidence(
        self,
        evidence: "Evidence",
        session_id: str,
        source_content: str | None = None,
    ) -> VerificationResult:
        """Streaming verification when intern submits evidence.

        Fast, lightweight verification for real-time use.
        Target latency: <500ms

        Args:
            evidence: The evidence to verify
            session_id: Current session ID
            source_content: Original source text the evidence was extracted from

        Returns:
            VerificationResult with calibrated confidence
        """
        evidence_id = str(evidence.id or hash(evidence.content))

        if not self.config.enable_streaming_verification:
            return VerificationResult(
                evidence_id=evidence_id,
                original_confidence=evidence.confidence,
                verified_confidence=evidence.confidence,
                verification_status=VerificationStatus.SKIPPED,
                verification_method=VerificationMethod.STREAMING,
            )

        # HHEM source-grounding check (runs before CoVe to enable early reject)
        # Only use HHEM when source_content is substantial enough for reliable
        # grounding evaluation.  Search-result snippets (~100-200 chars) are too
        # short -- HHEM compares hypothesis against premise, and a tiny snippet
        # rarely contains enough evidence to support detailed evidence, causing
        # false low scores and mass-rejection.
        hhem_score = -1.0
        min_source_len_for_hhem = 500
        if self.hhem and source_content and len(source_content) >= min_source_len_for_hhem:
            hhem_score = await self.hhem.score(source_content, evidence.content)

            # Early reject only if clearly not grounded AND we had enough source
            # text to make a reliable judgement
            if 0 <= hhem_score < self.config.hhem_reject_threshold:
                result = VerificationResult(
                    evidence_id=evidence_id,
                    original_confidence=evidence.confidence,
                    verified_confidence=max(0.0, evidence.confidence - 0.15),
                    verification_status=VerificationStatus.REJECTED,
                    verification_method=VerificationMethod.STREAMING,
                    hhem_grounding_score=hhem_score,
                )
                await self.metrics.record_result(result)
                return result

        # Quick KG match if available
        kg_support_score = 0.0
        kg_entity_matches = 0
        kg_supporting_relations = 0
        if self.knowledge_graph and self.config.enable_kg_verification:
            (
                kg_support_score,
                kg_entity_matches,
                kg_supporting_relations,
            ) = await self._get_kg_support(evidence)

        # Run streaming CoVe
        result = await self.cove.verify_streaming(
            evidence_content=evidence.content,
            evidence_id=evidence_id,
            original_confidence=evidence.confidence,
            source_url=evidence.source_url,
            search_query=evidence.search_query,
        )

        # Attach HHEM and KG scores
        result.hhem_grounding_score = hhem_score
        result.kg_support_score = kg_support_score
        result.kg_entity_matches = kg_entity_matches
        result.kg_supporting_relations = kg_supporting_relations

        # Recalibrate with all signals (KG + HHEM + CoVe)
        if kg_support_score > 0 or hhem_score >= 0:
            # Pass -1 for CoVe consistency if CoVe was skipped (parsing failed),
            # so the calibrator treats it as "no data" rather than a negative signal
            cove_score = (
                result.consistency_score
                if result.consistency_score is not None
                and result.verification_status not in (VerificationStatus.SKIPPED, VerificationStatus.ERROR)
                else -1.0
            )
            calibration = self.calibrator.calibrate(
                original_confidence=evidence.confidence,
                cove_consistency_score=cove_score,
                kg_support_score=kg_support_score,
                hhem_grounding_score=hhem_score,
            )
            result.verified_confidence = calibration.calibrated_confidence
            result.verification_status = calibration.status

        # Track metrics
        await self.metrics.record_result(result)

        return result

    async def verify_batch(
        self,
        evidence_list: list["Evidence"],
        session_id: str,
    ) -> BatchVerificationResult:
        """Batch verification during manager synthesis.

        Comprehensive verification with KG consistency checks and CRITIC for high-stakes.
        Target latency: ~2s per piece of evidence average.

        Args:
            evidence_list: List of evidence to verify
            session_id: Current session ID

        Returns:
            BatchVerificationResult with all verification results
        """
        start_time = time.time()
        logger.info("Verification pipeline: %d evidence for session %s", len(evidence_list), session_id)

        if not self.config.enable_batch_verification:
            return BatchVerificationResult(
                session_id=session_id,
                total_evidence=len(evidence_list),
                verified_count=0,
                flagged_count=0,
                rejected_count=0,
                skipped_count=len(evidence_list),
            )

        # Emit progress so UI doesn't look stuck during lengthy verification
        total = len(evidence_list)
        try:
            await emit_agent_event(
                session_id=session_id,
                event_type="system",
                agent="manager",
                data={
                    "message": (
                        f"[VERIFY] Starting batch verification of {total} evidence "
                        f"(this may take several minutes)..."
                    ),
                },
            )
        except Exception:
            logger.debug("Failed to emit verification start event", exc_info=True)

        # Process in parallel batches
        results = []
        all_contradictions = []

        for i in range(0, len(evidence_list), self.config.parallel_batch_size):
            batch = evidence_list[i : i + self.config.parallel_batch_size]
            batch_num = (i // self.config.parallel_batch_size) + 1
            batch_size = self.config.parallel_batch_size
            total_batches = (total + batch_size - 1) // batch_size
            logger.info(
                "Verification batch %d/%d: evidence %d-%d of %d",
                batch_num, total_batches, i + 1, min(i + len(batch), total), total,
            )
            try:
                await emit_agent_event(
                    session_id=session_id,
                    event_type="system",
                    agent="manager",
                    data={
                        "message": (
                            f"[VERIFY] Verifying batch {batch_num}/{total_batches} "
                            f"({len(results)}/{total} complete)..."
                        ),
                    },
                )
            except Exception:
                logger.debug("Failed to emit verification batch event", exc_info=True)
            batch_results = await self._verify_batch_parallel(batch, session_id)
            results.extend(batch_results)

        # Collect contradictions
        for result in results:
            all_contradictions.extend(result.contradictions)

        # Count by status
        verified = sum(1 for r in results if r.verification_status == VerificationStatus.VERIFIED)
        flagged = sum(1 for r in results if r.verification_status == VerificationStatus.FLAGGED)
        rejected = sum(1 for r in results if r.verification_status == VerificationStatus.REJECTED)
        skipped = sum(1 for r in results if r.verification_status == VerificationStatus.SKIPPED)
        errored = sum(1 for r in results if r.verification_status == VerificationStatus.ERROR)

        total_time = (time.time() - start_time) * 1000

        try:
            await emit_agent_event(
                session_id=session_id,
                event_type="system",
                agent="manager",
                data={
                    "message": (
                        f"[VERIFY] Batch verification complete: "
                        f"{verified} verified, {flagged} flagged, "
                        f"{rejected} rejected, {skipped} skipped"
                        f"{f', {errored} errors' if errored else ''} "
                        f"({total_time / 1000:.0f}s)"
                    ),
                },
            )
        except Exception:
            logger.debug("Failed to emit verification complete event", exc_info=True)

        batch_result = BatchVerificationResult(
            session_id=session_id,
            total_evidence=len(evidence_list),
            verified_count=verified,
            flagged_count=flagged,
            rejected_count=rejected,
            skipped_count=skipped,
            results=results,
            contradictions_found=all_contradictions,
            total_time_ms=total_time,
            avg_time_per_evidence_ms=total_time / len(evidence_list) if evidence_list else 0,
        )

        # Track metrics
        await self.metrics.record_batch(batch_result)

        return batch_result

    async def _verify_batch_parallel(
        self,
        evidence_list: list["Evidence"],
        session_id: str,
    ) -> list[VerificationResult]:
        """Verify a batch of evidence in parallel."""
        tasks = [self._verify_single_batch(evidence, session_id) for evidence in evidence_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions - convert to skipped results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Batch verification failed for evidence: %s", result, exc_info=True)
                # Create a fallback result for failed verification
                evidence = evidence_list[i]
                processed_results.append(
                    VerificationResult(
                        evidence_id=str(evidence.id or hash(evidence.content)),
                        original_confidence=evidence.confidence,
                        verified_confidence=evidence.confidence,
                        verification_status=VerificationStatus.ERROR,
                        verification_method=VerificationMethod.BATCH,
                        error=str(result),
                    )
                )
            else:
                processed_results.append(result)

        return processed_results

    async def _verify_single_batch(
        self,
        evidence: "Evidence",
        session_id: str,
    ) -> VerificationResult:
        """Verify a single piece of evidence in batch mode."""
        evidence_id = str(evidence.id or hash(evidence.content))

        # Get KG signals
        kg_support_score = 0.0
        kg_entity_matches = 0
        kg_supporting_relations = 0
        has_contradictions = False
        contradictions = []

        if self.knowledge_graph and self.config.enable_kg_verification:
            (
                kg_support_score,
                kg_entity_matches,
                kg_supporting_relations,
            ) = await self._get_kg_support(evidence)
            contradiction_result = await self._check_kg_contradictions(evidence)
            has_contradictions = contradiction_result.get("has_contradictions", False)
            if has_contradictions:
                contradictions = [
                    ContradictionDetail(
                        evidence_id=evidence_id,
                        conflicting_evidence_id=c.get("conflicting_id", "unknown"),
                        description=c.get("description", "Contradiction detected"),
                        severity=c.get("severity", "medium"),
                    )
                    for c in contradiction_result.get("contradictions", [])
                ]

        # Run batch CoVe
        cove_result = await self.cove.verify_batch(
            evidence_content=evidence.content,
            evidence_id=evidence_id,
            original_confidence=evidence.confidence,
            source_url=evidence.source_url,
            search_query=evidence.search_query,
            kg_support_score=kg_support_score,
            has_contradictions=has_contradictions,
        )

        # Attach KG counts to CoVe result
        cove_result.kg_entity_matches = kg_entity_matches
        cove_result.kg_supporting_relations = kg_supporting_relations

        # Check if CRITIC is needed:
        # - Low CoVe consistency (verification questions contradicted the claim)
        # - Low confidence after calibration
        # - KG contradictions detected
        # High-stakes detection is used as a boost (run CRITIC at higher threshold)
        # but NOT as a gate -- any evidence with low consistency or contradictions
        # gets CRITIC regardless of content domain.
        low_consistency = (
            cove_result.consistency_score is not None and cove_result.consistency_score < 0.5
        )
        low_confidence = cove_result.verified_confidence < self.config.critic_confidence_threshold
        is_high_stakes = self.high_stakes_detector.is_high_stakes(evidence.content)

        use_critic = self.config.enable_critic and (
            low_consistency or has_contradictions or (low_confidence and is_high_stakes)
        )

        if use_critic:
            critic_result = await self.critic.verify(
                evidence_content=evidence.content,
                evidence_id=evidence_id,
                original_confidence=evidence.confidence,
                source_url=evidence.source_url,
                cove_result=cove_result,
            )
            critic_result.contradictions = contradictions
            critic_result.kg_entity_matches = kg_entity_matches
            critic_result.kg_supporting_relations = kg_supporting_relations
            return critic_result

        # Add contradictions to CoVe result
        cove_result.contradictions = contradictions
        return cove_result

    async def _get_kg_support(self, evidence: "Evidence") -> tuple[float, int, int]:
        """Get KG support score and entity/relation counts for evidence.

        Returns:
            Tuple of (score, entity_matches, supporting_relations)
        """
        if not self.knowledge_graph:
            return 0.0, 0, 0

        try:
            return await self.knowledge_graph.get_kg_support_score(
                content=evidence.content,
                source_url=evidence.source_url,
            )
        except Exception:
            logger.warning("KG support score retrieval failed", exc_info=True)
            return 0.0, 0, 0

    async def _check_kg_contradictions(self, evidence: "Evidence") -> dict:
        """Check for contradictions in the KG."""
        if not self.knowledge_graph:
            return {"has_contradictions": False, "contradictions": []}

        try:
            return await self.knowledge_graph.check_contradictions_detailed(
                content=evidence.content,
            )
        except Exception:
            logger.warning("KG contradiction check failed", exc_info=True)
            return {"has_contradictions": False, "contradictions": []}

    def get_metrics_summary(self) -> dict:
        """Get summary of verification metrics."""
        return self.metrics.get_summary()

    def get_metrics_report_section(self) -> str:
        """Get markdown section for report."""
        return self.metrics.get_report_section()

    def reset_metrics(self, session_id: str | None = None) -> None:
        """Reset metrics for a new session."""
        self.metrics.reset()
        self.metrics.session_id = session_id


def create_verification_pipeline(
    llm_callback: Callable[[str, str], Any],
    knowledge_graph: Optional["IncrementalKnowledgeGraph"] = None,
    search_callback: Callable[[str], Any] | None = None,
    config: VerificationConfig | None = None,
) -> VerificationPipeline:
    """Factory function to create a verification pipeline.

    Args:
        llm_callback: Async function to call LLM with (prompt, model)
        knowledge_graph: Optional KG for corroboration
        search_callback: Optional async function for web search
        config: Verification configuration

    Returns:
        Configured VerificationPipeline
    """
    return VerificationPipeline(
        llm_callback=llm_callback,
        knowledge_graph=knowledge_graph,
        search_callback=search_callback,
        config=config,
    )
