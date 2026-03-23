"""Verification system for fact checking and hallucination reduction.

This module implements Chain-of-Verification (CoVe) + CRITIC to achieve
50-70% hallucination reduction through:

1. Streaming verification - Real-time verification when interns submit evidence (<500ms)
2. Batch verification - Comprehensive verification during manager synthesis (~2s/evidence)

Confidence Thresholds:
- >85%: Auto-accept (VERIFIED)
- 50-85%: Flag for review (FLAGGED)
- <50%: Reject, trigger additional search (REJECTED)
"""

from .confidence import (
    CalibrationResult,
    ConfidenceCalibrator,
)
from .cove import ChainOfVerification
from .critic import (
    CRITICVerifier,
    HighStakesDetector,
)
from .hhem import HHEMScorer
from .metrics import (
    LatencyMetrics,
    VerificationMetricsTracker,
)
from .models import (
    BatchVerificationResult,
    ContradictionDetail,
    VerificationConfig,
    VerificationMethod,
    VerificationQuestion,
    VerificationResult,
    VerificationStatus,
)
from .pipeline import (
    VerificationPipeline,
    create_verification_pipeline,
)

__all__ = [
    # Models
    "VerificationStatus",
    "VerificationMethod",
    "VerificationQuestion",
    "VerificationResult",
    "BatchVerificationResult",
    "VerificationConfig",
    "ContradictionDetail",
    # Confidence
    "ConfidenceCalibrator",
    "CalibrationResult",
    # CoVe
    "ChainOfVerification",
    # CRITIC
    "CRITICVerifier",
    "HighStakesDetector",
    # HHEM
    "HHEMScorer",
    # Metrics
    "VerificationMetricsTracker",
    "LatencyMetrics",
    # Pipeline
    "VerificationPipeline",
    "create_verification_pipeline",
]
