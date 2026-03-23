"""Data models for the Veritas engine."""

from .evidence import (
    AgentMessage,
    AgentRole,
    CheckSession,
    Evidence,
    EvidenceType,
    EvidenceReport,
    InternReport,
    ManagerDirective,
    ManagerReport,
    SubClaim,
    Verdict,
    VerificationDirective,
    VerdictReport,
    is_meta_question,
)

# Backward-compat aliases used by some modules
Finding = Evidence
FindingType = EvidenceType
ResearchSession = CheckSession
ResearchTopic = SubClaim

__all__ = [
    "AgentMessage",
    "AgentRole",
    "CheckSession",
    "Evidence",
    "EvidenceReport",
    "EvidenceType",
    "Finding",
    "FindingType",
    "InternReport",
    "ManagerDirective",
    "ManagerReport",
    "ResearchSession",
    "ResearchTopic",
    "SubClaim",
    "Verdict",
    "VerdictReport",
    "VerificationDirective",
    "is_meta_question",
]
