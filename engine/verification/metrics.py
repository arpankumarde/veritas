"""Verification metrics tracking."""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from .models import (
    BatchVerificationResult,
    VerificationMethod,
    VerificationResult,
    VerificationStatus,
)


@dataclass
class LatencyMetrics:
    """Latency metrics for verification operations."""
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float('inf')
    max_ms: float = 0.0

    def add(self, latency_ms: float) -> None:
        """Add a latency measurement."""
        self.count += 1
        self.total_ms += latency_ms
        self.min_ms = min(self.min_ms, latency_ms)
        self.max_ms = max(self.max_ms, latency_ms)

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count > 0 else 0.0

    @property
    def p95_target_met(self) -> bool:
        """Check if p95 target is likely met (rough heuristic)."""
        # If max is within 2x of avg, we're probably meeting p95
        return self.max_ms <= self.avg_ms * 2


@dataclass
class VerificationMetricsTracker:
    """Track verification metrics across sessions."""

    # Session tracking
    session_id: str | None = None
    started_at: datetime = field(default_factory=datetime.now)

    # Counts by status
    verified_count: int = 0
    flagged_count: int = 0
    rejected_count: int = 0
    skipped_count: int = 0

    # Counts by method
    streaming_count: int = 0
    batch_count: int = 0
    critic_count: int = 0

    # Latency tracking
    streaming_latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    batch_latency: LatencyMetrics = field(default_factory=LatencyMetrics)

    # Confidence tracking
    total_original_confidence: float = 0.0
    total_calibrated_confidence: float = 0.0
    confidence_improvements: int = 0
    confidence_reductions: int = 0

    # Contradiction tracking
    contradictions_detected: int = 0
    contradictions_by_severity: dict = field(default_factory=lambda: defaultdict(int))

    # KG integration
    kg_matches: int = 0
    kg_boosts: int = 0

    # CRITIC stats
    critic_iterations_total: int = 0
    corrections_made: int = 0
    external_verifications: int = 0

    # Error tracking
    errors: int = 0
    error_types: dict = field(default_factory=lambda: defaultdict(int))

    # Lock for thread-safe updates
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def record_result(self, result: VerificationResult) -> None:
        """Record a verification result."""
        async with self._lock:
            # Status counts
            if result.verification_status == VerificationStatus.VERIFIED:
                self.verified_count += 1
            elif result.verification_status == VerificationStatus.FLAGGED:
                self.flagged_count += 1
            elif result.verification_status == VerificationStatus.REJECTED:
                self.rejected_count += 1
            else:
                self.skipped_count += 1

            # Method counts and latency
            if result.verification_method == VerificationMethod.STREAMING:
                self.streaming_count += 1
                self.streaming_latency.add(result.verification_time_ms)
            elif result.verification_method in (VerificationMethod.BATCH, VerificationMethod.COVE):
                self.batch_count += 1
                self.batch_latency.add(result.verification_time_ms)
            elif result.verification_method in (VerificationMethod.CRITIC, VerificationMethod.COVE_CRITIC):
                self.critic_count += 1
                self.batch_latency.add(result.verification_time_ms)

            # Confidence tracking
            self.total_original_confidence += result.original_confidence
            self.total_calibrated_confidence += result.verified_confidence

            if result.verified_confidence > result.original_confidence:
                self.confidence_improvements += 1
            elif result.verified_confidence < result.original_confidence:
                self.confidence_reductions += 1

            # Contradiction tracking
            for contradiction in result.contradictions:
                self.contradictions_detected += 1
                self.contradictions_by_severity[contradiction.severity] += 1

            # KG tracking
            if (result.kg_support_score or 0) > 0:
                self.kg_matches += 1
            if (result.kg_support_score or 0) > 0.5:
                self.kg_boosts += 1

            # CRITIC tracking
            self.critic_iterations_total += result.critic_iterations or 0
            self.corrections_made += len(result.corrections_made)
            if result.external_verification_used:
                self.external_verifications += 1

            # Error tracking
            if result.error:
                self.errors += 1
                error_type = result.error.split(":")[0] if ":" in result.error else "Unknown"
                self.error_types[error_type] += 1

    async def record_batch(self, batch_result: BatchVerificationResult) -> None:
        """Record a batch verification result."""
        for result in batch_result.results:
            await self.record_result(result)

    @property
    def total_count(self) -> int:
        return self.verified_count + self.flagged_count + self.rejected_count + self.skipped_count

    @property
    def verification_rate(self) -> float:
        """Percentage of evidence that passed verification."""
        effective_total = self.total_count - self.skipped_count
        if effective_total == 0:
            return 0.0
        return self.verified_count / effective_total * 100

    @property
    def rejection_rate(self) -> float:
        """Percentage of evidence that was rejected."""
        effective_total = self.total_count - self.skipped_count
        if effective_total == 0:
            return 0.0
        return self.rejected_count / effective_total * 100

    @property
    def avg_confidence_delta(self) -> float:
        """Average change in confidence after calibration."""
        if self.total_count == 0:
            return 0.0
        avg_original = self.total_original_confidence / self.total_count
        avg_calibrated = self.total_calibrated_confidence / self.total_count
        return avg_calibrated - avg_original

    def get_summary(self) -> dict:
        """Get summary statistics."""
        return {
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(),
            "total_verifications": self.total_count,

            # Status breakdown
            "status": {
                "verified": self.verified_count,
                "flagged": self.flagged_count,
                "rejected": self.rejected_count,
                "skipped": self.skipped_count,
            },
            "verification_rate_pct": round(self.verification_rate, 1),
            "rejection_rate_pct": round(self.rejection_rate, 1),

            # Method breakdown
            "methods": {
                "streaming": self.streaming_count,
                "batch": self.batch_count,
                "critic": self.critic_count,
            },

            # Latency
            "latency": {
                "streaming_avg_ms": round(self.streaming_latency.avg_ms, 1),
                "streaming_max_ms": round(self.streaming_latency.max_ms, 1),
                "batch_avg_ms": round(self.batch_latency.avg_ms, 1),
                "batch_max_ms": round(self.batch_latency.max_ms, 1),
                "streaming_target_met": self.streaming_latency.avg_ms < 500,
                "batch_target_met": self.batch_latency.avg_ms < 2000,
            },

            # Confidence
            "confidence": {
                "avg_delta": round(self.avg_confidence_delta, 3),
                "improvements": self.confidence_improvements,
                "reductions": self.confidence_reductions,
            },

            # Contradictions
            "contradictions": {
                "total": self.contradictions_detected,
                "by_severity": dict(self.contradictions_by_severity),
            },

            # KG integration
            "kg_integration": {
                "matches": self.kg_matches,
                "confidence_boosts": self.kg_boosts,
            },

            # CRITIC
            "critic": {
                "total_iterations": self.critic_iterations_total,
                "corrections_made": self.corrections_made,
                "external_verifications": self.external_verifications,
            },

            # Errors
            "errors": {
                "total": self.errors,
                "by_type": dict(self.error_types),
            },
        }

    def get_report_section(self) -> str:
        """Generate a markdown section for the report."""
        summary = self.get_summary()

        lines = [
            "## Verification Metrics",
            "",
            f"**Total Evidence Verified:** {summary['total_verifications']}",
            "",
            "### Verification Status Breakdown",
            f"- Verified (>85% confidence): {summary['status']['verified']} ({summary['verification_rate_pct']:.1f}%)",
            f"- Flagged (50-85% confidence): {summary['status']['flagged']}",
            f"- Rejected (<50% confidence): {summary['status']['rejected']} ({summary['rejection_rate_pct']:.1f}%)",
            f"- Skipped: {summary['status']['skipped']}",
            "",
            "### Confidence Calibration",
            f"- Average confidence adjustment: {summary['confidence']['avg_delta']:+.1%}",
            f"- Evidence with improved confidence: {summary['confidence']['improvements']}",
            f"- Evidence with reduced confidence: {summary['confidence']['reductions']}",
            "",
        ]

        if summary['contradictions']['total'] > 0:
            lines.extend([
                "### Contradictions Detected",
                f"- Total: {summary['contradictions']['total']}",
            ])
            for severity, count in summary['contradictions']['by_severity'].items():
                lines.append(f"- {severity.capitalize()}: {count}")
            lines.append("")

        if summary['kg_integration']['matches'] > 0:
            lines.extend([
                "### Knowledge Graph Integration",
                f"- Evidence corroborated by KG: {summary['kg_integration']['matches']}",
                f"- Confidence boosts from KG: {summary['kg_integration']['confidence_boosts']}",
                "",
            ])

        lines.extend([
            "### Verification Latency",
            f"- Streaming: avg {summary['latency']['streaming_avg_ms']:.0f}ms (target <500ms: {'Met' if summary['latency']['streaming_target_met'] else 'Missed'})",
            f"- Batch: avg {summary['latency']['batch_avg_ms']:.0f}ms (target <2000ms: {'Met' if summary['latency']['batch_target_met'] else 'Missed'})",
            "",
        ])

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all metrics."""
        self.started_at = datetime.now()
        self.verified_count = 0
        self.flagged_count = 0
        self.rejected_count = 0
        self.skipped_count = 0
        self.streaming_count = 0
        self.batch_count = 0
        self.critic_count = 0
        self.streaming_latency = LatencyMetrics()
        self.batch_latency = LatencyMetrics()
        self.total_original_confidence = 0.0
        self.total_calibrated_confidence = 0.0
        self.confidence_improvements = 0
        self.confidence_reductions = 0
        self.contradictions_detected = 0
        self.contradictions_by_severity = defaultdict(int)
        self.kg_matches = 0
        self.kg_boosts = 0
        self.critic_iterations_total = 0
        self.corrections_made = 0
        self.external_verifications = 0
        self.errors = 0
        self.error_types = defaultdict(int)
