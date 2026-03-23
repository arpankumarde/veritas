"""Data models for fact-check evidence and agent communication."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

# Phrases that indicate an LLM produced a meta-question or placeholder
# instead of a real evidence finding/follow-up.  Shared by agents to
# filter these out consistently.
META_QUESTION_PHRASES: tuple[str, ...] = (
    "please provide",
    "what information",
    "could you clarify",
    "what are you looking for",
    "what topic",
    "what subject",
    "what would you like",
    "can you specify",
    "please specify",
    "more details",
    "template or placeholder",
)


def is_meta_question(text: str) -> bool:
    """Return True if *text* looks like a meta-question/placeholder."""
    lower = text.lower()
    return any(phrase in lower for phrase in META_QUESTION_PHRASES)


class Verdict(str, Enum):
    """Verdict for a fact-checked claim."""

    TRUE = "true"
    MOSTLY_TRUE = "mostly_true"
    MIXED = "mixed"
    MOSTLY_FALSE = "mostly_false"
    FALSE = "false"
    UNVERIFIABLE = "unverifiable"


class EvidenceType(str, Enum):
    """Types of evidence gathered during fact-checking."""

    SUPPORTING = "supporting"
    CONTRADICTING = "contradicting"
    CONTEXTUAL = "contextual"
    SOURCE = "source"
    QUESTION = "question"
    # Keep backward compat aliases
    FACT = "fact"
    INSIGHT = "insight"
    CONNECTION = "connection"
    CONTRADICTION = "contradiction"


class AgentRole(str, Enum):
    """Agent roles in the hierarchy."""

    INTERN = "intern"
    MANAGER = "manager"
    DIRECTOR = "director"


class Evidence(BaseModel):
    """A single piece of evidence gathered during fact-checking."""

    id: int | None = None
    session_id: str  # 7-char hex ID
    content: str
    evidence_type: EvidenceType
    source_url: str | None = None
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    search_query: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    validated_by_manager: bool = False
    manager_notes: str | None = None

    topic_id: int | None = None

    # Verification fields
    verification_status: str | None = None  # verified/flagged/rejected/skipped
    verification_method: str | None = None  # cove/critic/kg_match/streaming/batch
    kg_support_score: float | None = None  # KG corroboration score (0-1)
    original_confidence: float | None = None  # Confidence before verification calibration


class SubClaim(BaseModel):
    """A sub-claim or subtopic to investigate during fact-checking."""

    id: int | None = None
    session_id: str  # 7-char hex ID
    topic: str
    parent_topic_id: int | None = None
    depth: int = 0
    status: str = "pending"  # pending, in_progress, completed, blocked
    priority: int = Field(default=5, ge=1, le=10)
    assigned_at: datetime | None = None
    completed_at: datetime | None = None
    findings_count: int = 0


class CheckSession(BaseModel):
    """A fact-checking session with iteration-based control."""

    id: str | None = None  # 7-char hex ID
    claim: str
    slug: str | None = None  # AI-generated short name for output folder
    verdict: str | None = None  # true/false/partially_true/unverifiable/misleading
    max_iterations: int = 5  # Manager ReAct loop iterations (controls verification depth)
    time_limit_minutes: int = 0  # Kept for backward compat / display only
    started_at: datetime = Field(default_factory=datetime.now)
    ended_at: datetime | None = None
    status: str = "active"  # active, running, paused, crashed, completed, error, interrupted
    total_findings: int = 0
    total_searches: int = 0
    depth_reached: int = 0

    # Pause/resume/crash recovery fields
    elapsed_seconds: float = 0.0  # Accumulated check time across pause/resume cycles
    paused_at: datetime | None = None  # Timestamp when last paused (None if running)
    iteration_count: int = 0  # Manager ReAct loop iteration for resume
    phase: str = "init"  # init | parallel_init | react_loop | synthesis | done

    @property
    def goal(self) -> str:
        """Backward-compat alias for 'claim'."""
        return self.claim


class AgentMessage(BaseModel):
    """Message passed between agents in the hierarchy."""

    id: int | None = None
    session_id: str  # 7-char hex ID
    from_agent: AgentRole
    to_agent: AgentRole
    message_type: str  # task, report, critique, question, directive
    content: str
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class ManagerDirective(BaseModel):
    """Directive from Manager to Intern."""

    action: str  # search, deep_dive, verify, expand, stop
    topic: str
    instructions: str
    priority: int = Field(default=5, ge=1, le=10)
    max_searches: int = 5


class InternReport(BaseModel):
    """Report from Intern to Manager."""

    topic: str
    evidence: list[Evidence]
    searches_performed: int
    suggested_followups: list[str]
    blockers: list[str] = Field(default_factory=list)


class ManagerReport(BaseModel):
    """Report from Manager to Director."""

    summary: str
    key_evidence: list[Evidence]
    topics_explored: list[str] = Field(default_factory=list)
    topics_remaining: list[str] = Field(default_factory=list)
    quality_assessment: str = ""
    recommended_next_steps: list[str] = Field(default_factory=list)
    time_elapsed_minutes: float = 0.0
    iterations_completed: int = 0
    searches_performed: int = 0
    verdict: str | None = None  # Verdict string for fact-check results
    # Backward-compat aliases (populated from sub_claims_* kwargs)
    sub_claims_explored: list[str] = Field(default_factory=list)
    sub_claims_remaining: list[str] = Field(default_factory=list)

    def model_post_init(self, __context) -> None:
        """Sync backward-compat aliases with canonical fields."""
        # If sub_claims_* were provided but topics_* were empty, copy them over
        if self.sub_claims_explored and not self.topics_explored:
            self.topics_explored = self.sub_claims_explored
        elif self.topics_explored and not self.sub_claims_explored:
            self.sub_claims_explored = self.topics_explored
        if self.sub_claims_remaining and not self.topics_remaining:
            self.topics_remaining = self.sub_claims_remaining
        elif self.topics_remaining and not self.sub_claims_remaining:
            self.sub_claims_remaining = self.topics_remaining


# Aliases used by various agent modules
VerificationDirective = ManagerDirective
EvidenceReport = InternReport
VerdictReport = ManagerReport
FindingType = EvidenceType
Finding = Evidence
ResearchSession = CheckSession
ResearchTopic = SubClaim
