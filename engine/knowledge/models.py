"""Data models for the knowledge graph."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False


# Entity types for research knowledge graphs
ENTITY_TYPES = {
    # Domain Knowledge Entities
    'CONCEPT': 'Abstract idea or topic being researched',
    'CLAIM': 'Specific assertion that can be true or false',
    'EVIDENCE': 'Data, study, or observation supporting/refuting claims',
    'METHOD': 'Technique, algorithm, or approach',
    'METRIC': 'Quantitative measurement or statistic',
    'TECHNOLOGY': 'Tool, system, or implementation',
    # Meta-Knowledge Entities
    'SOURCE': 'Document, paper, or website providing information',
    'AUTHOR': 'Person or organization authoring a source',
    'QUOTE': 'Direct quotation from a source',
    # Standard Named Entities
    'PERSON': 'Individual mentioned in research',
    'ORGANIZATION': 'Company, institution, or group',
    'LOCATION': 'Geographic location',
    'DATE': 'Temporal reference',
}

# Relation types
RELATION_TYPES = {
    # Epistemic Relations (knowledge about claims)
    'supports': 'Evidence supports a claim',
    'contradicts': 'Evidence contradicts a claim',
    'qualifies': 'Evidence adds conditions to a claim',
    'cites': 'Source cites another source',
    # Semantic Relations (domain knowledge)
    'is_a': 'Taxonomic relationship',
    'part_of': 'Compositional relationship',
    'causes': 'Causal relationship',
    'correlates_with': 'Statistical relationship',
    'enables': 'Functional dependency',
    'implements': 'Realization relationship',
    # Comparative Relations
    'outperforms': 'Performance comparison',
    'similar_to': 'Similarity relationship',
    'alternative_to': 'Alternative approaches',
    # Attribution Relations
    'authored_by': 'Authorship',
    'published_in': 'Publication venue',
    'mentioned_in': 'Reference in source',
    # Co-occurrence
    'co_occurs_with': 'Entities that appear together in the same evidence',
}


@dataclass
class Entity:
    """An entity in the knowledge graph."""
    id: str
    name: str
    entity_type: str
    embedding: Any | None = None  # np.ndarray if numpy available
    aliases: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    confidence: float = 1.0
    properties: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            'id': self.id,
            'name': self.name,
            'entity_type': self.entity_type,
            'aliases': self.aliases,
            'sources': self.sources,
            'confidence': self.confidence,
            'properties': self.properties,
            'created_at': self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Entity':
        """Create from dictionary."""
        return cls(
            id=data['id'],
            name=data['name'],
            entity_type=data['entity_type'],
            aliases=data.get('aliases', []),
            sources=data.get('sources', []),
            confidence=data.get('confidence', 1.0),
            properties=data.get('properties', {}),
            created_at=datetime.fromisoformat(data['created_at']) if 'created_at' in data else datetime.now(),
        )


@dataclass
class Relation:
    """A relation between entities in the knowledge graph."""
    id: str
    subject_id: str
    predicate: str
    object_id: str
    source_id: str
    confidence: float = 1.0
    timestamp: str | None = None
    properties: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            'id': self.id,
            'subject_id': self.subject_id,
            'predicate': self.predicate,
            'object_id': self.object_id,
            'source_id': self.source_id,
            'confidence': self.confidence,
            'timestamp': self.timestamp,
            'properties': self.properties,
            'created_at': self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Relation':
        """Create from dictionary."""
        return cls(
            id=data['id'],
            subject_id=data['subject_id'],
            predicate=data['predicate'],
            object_id=data['object_id'],
            source_id=data['source_id'],
            confidence=data.get('confidence', 1.0),
            timestamp=data.get('timestamp'),
            properties=data.get('properties', {}),
            created_at=datetime.fromisoformat(data['created_at']) if 'created_at' in data else datetime.now(),
        )


@dataclass
class KGEvidence:
    """A piece of evidence to be processed into the knowledge graph."""
    id: str
    content: str
    source_url: str
    source_title: str
    timestamp: str
    credibility_score: float
    evidence_type: str = "fact"
    search_query: str | None = None
    finding_type: str | None = None  # Backward-compat alias for evidence_type

    def __post_init__(self):
        # If finding_type was passed but evidence_type wasn't changed from default,
        # use finding_type as evidence_type (backward-compat).
        if self.finding_type is not None and self.evidence_type == "fact":
            self.evidence_type = self.finding_type

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'content': self.content,
            'source_url': self.source_url,
            'source_title': self.source_title,
            'timestamp': self.timestamp,
            'credibility_score': self.credibility_score,
            'evidence_type': self.evidence_type,
            'search_query': self.search_query,
        }


# Backward-compat alias
KGFinding = KGEvidence


@dataclass
class Contradiction:
    """A detected contradiction between relations."""
    id: str
    relation1_id: str
    relation2_id: str
    contradiction_type: str  # 'direct', 'transitive', 'semantic'
    description: str
    severity: str = 'medium'  # 'low', 'medium', 'high'
    resolution: str | None = None
    resolved_at: datetime | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'relation1_id': self.relation1_id,
            'relation2_id': self.relation2_id,
            'contradiction_type': self.contradiction_type,
            'description': self.description,
            'severity': self.severity,
            'resolution': self.resolution,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
        }


@dataclass
class KnowledgeGap:
    """An identified gap in knowledge."""
    gap_type: str  # 'insufficient_evidence', 'disconnected_topics', 'bridging_concept', 'missing_entity_type'
    entity: str | None = None
    entity_type: str | None = None
    current_count: int = 0
    recommendation: str = ""
    importance: float = 0.5

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'gap_type': self.gap_type,
            'entity': self.entity,
            'entity_type': self.entity_type,
            'current_count': self.current_count,
            'recommendation': self.recommendation,
            'importance': self.importance,
        }
