"""Knowledge graph module for real-time research synthesis."""

from .credibility import CredibilityScorer
from .fast_ner import ExtractedEntity, FastNER, FastNERConfig, get_fast_ner
from .graph import IncrementalKnowledgeGraph
from .models import Contradiction, Entity, KGEvidence, KGFinding, KnowledgeGap, Relation
from .query import ManagerQueryInterface
from .store import HybridKnowledgeGraphStore
from .visualize import KnowledgeGraphVisualizer

__all__ = [
    "IncrementalKnowledgeGraph",
    "HybridKnowledgeGraphStore",
    "Entity",
    "Relation",
    "KGEvidence",
    "KGFinding",
    "Contradiction",
    "KnowledgeGap",
    "ManagerQueryInterface",
    "CredibilityScorer",
    "KnowledgeGraphVisualizer",
    # Fast NER
    "FastNER",
    "FastNERConfig",
    "ExtractedEntity",
    "get_fast_ner",
]
