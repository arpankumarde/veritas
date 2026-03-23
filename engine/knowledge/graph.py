"""Real-time incremental knowledge graph construction."""

import asyncio
import json
import re
import uuid
from collections.abc import Callable
from typing import Any

try:
    import numpy as np  # noqa: F401  -- imported for HAS_NUMPY detection
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from ..logging_config import get_logger
from .credibility import CredibilityScorer
from .fast_ner import get_fast_ner
from .models import ENTITY_TYPES, Contradiction, Entity, KGEvidence, Relation
from .store import HybridKnowledgeGraphStore

logger = get_logger(__name__)


def _extract_json_array(text: str) -> list | None:
    """Extract a JSON array from an LLM response string.

    Returns the parsed ``list`` on success, or ``None`` if no valid array is
    found.
    """
    if not text or not isinstance(text, str):
        return None

    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = cleaned.replace("```", "").strip()

    # Strategy 1: direct parse
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: greedy regex
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

    return None


# Canonical predicate mapping -- normalizes free-form predicates to a fixed set
# for better graph connectivity. ~130 synonyms -> ~20 canonical forms.
PREDICATE_CANONICAL_MAP: dict[str, str] = {
    # === Canonical predicates (identity mappings) ===
    'supports': 'supports',
    'contradicts': 'contradicts',
    'qualifies': 'qualifies',
    'cites': 'cites',
    'is_a': 'is_a',
    'part_of': 'part_of',
    'causes': 'causes',
    'correlates_with': 'correlates_with',
    'enables': 'enables',
    'implements': 'implements',
    'outperforms': 'outperforms',
    'similar_to': 'similar_to',
    'alternative_to': 'alternative_to',
    'authored_by': 'authored_by',
    'published_in': 'published_in',
    'mentioned_in': 'mentioned_in',
    'co_occurs_with': 'co_occurs_with',
    # Directional predicates (kept separate for contradiction detection)
    'increases': 'increases',
    'decreases': 'decreases',
    'prevents': 'prevents',
    'blocks': 'blocks',
    'improves': 'improves',
    'worsens': 'worsens',
    'underperforms': 'underperforms',
    # === Synonyms -> supports ===
    'evidence_for': 'supports', 'backs': 'supports', 'confirms': 'supports',
    'validates': 'supports', 'demonstrates': 'supports', 'proves': 'supports',
    'shows': 'supports', 'indicates': 'supports', 'corroborates': 'supports',
    'suggests': 'supports', 'backs_up': 'supports',
    # === Synonyms -> contradicts ===
    'conflicts_with': 'contradicts', 'opposes': 'contradicts', 'refutes': 'contradicts',
    'disproves': 'contradicts', 'challenges': 'contradicts', 'negates': 'contradicts',
    'disputes': 'contradicts', 'denies': 'contradicts',
    # === Synonyms -> qualifies ===
    'limits': 'qualifies', 'conditions': 'qualifies', 'constrains': 'qualifies',
    'modifies': 'qualifies',
    # === Synonyms -> cites ===
    'references': 'cites', 'refers_to': 'cites', 'based_on': 'cites',
    'builds_on': 'cites', 'draws_from': 'cites', 'derived_from': 'cites',
    # === Synonyms -> is_a ===
    'is': 'is_a', 'type_of': 'is_a', 'kind_of': 'is_a', 'instance_of': 'is_a',
    'form_of': 'is_a', 'classified_as': 'is_a', 'subtype_of': 'is_a',
    'category_of': 'is_a',
    # === Synonyms -> part_of ===
    'component_of': 'part_of', 'belongs_to': 'part_of', 'subset_of': 'part_of',
    'included_in': 'part_of', 'element_of': 'part_of', 'contains': 'part_of',
    'comprises': 'part_of', 'consists_of': 'part_of', 'has': 'part_of',
    'includes': 'part_of',
    # === Synonyms -> causes ===
    'leads_to': 'causes', 'results_in': 'causes', 'produces': 'causes',
    'triggers': 'causes', 'induces': 'causes', 'generates': 'causes',
    'creates': 'causes', 'drives': 'causes', 'contributes_to': 'causes',
    'influences': 'causes', 'affects': 'causes', 'impacts': 'causes',
    'determines': 'causes',
    # === Synonyms -> correlates_with ===
    'associated_with': 'correlates_with', 'related_to': 'correlates_with',
    'linked_to': 'correlates_with', 'connected_to': 'correlates_with',
    'measures': 'correlates_with', 'predicts': 'correlates_with',
    'corresponds_to': 'correlates_with',
    # === Synonyms -> enables ===
    'allows': 'enables', 'facilitates': 'enables', 'permits': 'enables',
    'requires': 'enables', 'depends_on': 'enables', 'needs': 'enables',
    # === Synonyms -> implements ===
    'realizes': 'implements', 'applies': 'implements', 'uses': 'implements',
    'employs': 'implements', 'utilizes': 'implements', 'leverages': 'implements',
    'adopts': 'implements', 'extends': 'implements',
    # === Synonyms -> outperforms ===
    'better_than': 'outperforms', 'surpasses': 'outperforms', 'exceeds': 'outperforms',
    'superior_to': 'outperforms', 'beats': 'outperforms', 'faster_than': 'outperforms',
    # === Synonyms -> similar_to ===
    'resembles': 'similar_to', 'analogous_to': 'similar_to', 'comparable_to': 'similar_to',
    'equivalent_to': 'similar_to', 'same_as': 'similar_to',
    # === Synonyms -> alternative_to ===
    'replaces': 'alternative_to', 'substitute_for': 'alternative_to',
    'competes_with': 'alternative_to', 'instead_of': 'alternative_to',
    # === Synonyms -> authored_by ===
    'written_by': 'authored_by', 'created_by': 'authored_by', 'developed_by': 'authored_by',
    'invented_by': 'authored_by', 'designed_by': 'authored_by', 'proposed_by': 'authored_by',
    'introduced_by': 'authored_by',
    # === Synonyms -> published_in ===
    'appeared_in': 'published_in', 'presented_at': 'published_in',
    'reported_in': 'published_in', 'released_in': 'published_in',
    # === Synonyms -> mentioned_in ===
    'described_in': 'mentioned_in', 'discussed_in': 'mentioned_in',
    'found_in': 'mentioned_in', 'cited_in': 'mentioned_in',
}

# Canonical predicate names for inclusion in LLM prompts
CANONICAL_PREDICATES_PROMPT = (
    "supports, contradicts, causes, enables, implements, is_a, part_of, "
    "correlates_with, outperforms, similar_to, alternative_to, authored_by, "
    "published_in, increases, decreases, qualifies, cites, mentioned_in"
)


class IncrementalKnowledgeGraph:
    """Real-time knowledge graph that builds as evidence streams in.

    Based on iText2KG, KGGen, and Graphiti patterns:
    - Async processing for non-blocking LLM I/O
    - Entity resolution via embedding similarity
    - Automatic contradiction detection
    - Incremental updates without full reprocessing
    """

    # Predicate pairs that are contradictory
    CONTRADICTORY_PREDICATES = [
        ('increases', 'decreases'),
        ('causes', 'prevents'),
        ('supports', 'contradicts'),
        ('enables', 'blocks'),
        ('improves', 'worsens'),
        ('is', 'is not'),
        ('has', 'lacks'),
        ('before', 'after'),
        ('greater than', 'less than'),
        ('outperforms', 'underperforms'),
    ]

    def __init__(
        self,
        llm_callback: Callable[[str], Any],
        store: HybridKnowledgeGraphStore | None = None,
        similarity_threshold: float = 0.7,
        use_fast_ner: bool = True,
        credibility_audit_callback: Callable[[dict], Any] | None = None,
        session_id: str | None = None,
    ):
        """Initialize the incremental knowledge graph.

        Args:
            llm_callback: Async function to call LLM for extraction
            store: Storage backend (creates new if not provided)
            similarity_threshold: Threshold for entity matching (0.7 = 70% similar)
            use_fast_ner: Use spaCy for fast NER (with LLM fallback for domain types)
            credibility_audit_callback: Optional async callback to persist credibility audits
            session_id: Session ID to associate with entities/relations
        """
        self.llm_callback = llm_callback
        self.session_id = session_id
        self.store = store or HybridKnowledgeGraphStore(session_id=session_id)
        self.similarity_threshold = similarity_threshold
        self.credibility_scorer = CredibilityScorer()
        self.credibility_audit_callback = credibility_audit_callback

        # Fast NER using spaCy (falls back to LLM for domain types)
        self.use_fast_ner = use_fast_ner
        self.fast_ner = get_fast_ner() if use_fast_ner else None

        # In-memory indexes for fast entity resolution
        self.entity_embeddings: dict[str, Any] = {}
        self.entity_by_name: dict[str, str] = {}  # lowercase name -> entity_id
        self.entity_by_type: dict[str, list[str]] = {}

        # Contradiction tracking
        self.contradictions: list[Contradiction] = []

        # Processing lock for thread safety
        self._processing_lock = asyncio.Lock()

    async def _save_credibility_audit(self, audit_data: dict) -> None:
        """Fire-and-forget save of credibility audit data."""
        if not self.credibility_audit_callback:
            return
        try:
            result = self.credibility_audit_callback(audit_data)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            # Don't let audit errors affect main processing
            logger.debug("Credibility audit save failed", exc_info=True)
            pass

    async def add_evidence(self, evidence: KGEvidence, fast_mode: bool = True) -> dict:
        """Process new evidence and integrate it into the knowledge graph.

        Args:
            evidence: The evidence to process
            fast_mode: If True, use single-pass extraction (fewer LLM calls)

        Returns:
            Dict with extracted entities, relations, and any contradictions found
        """
        logger.info("KG add_evidence: id=%s, fast_mode=%s", evidence.id, fast_mode)
        async with self._processing_lock:
            result = {
                'entities': [],
                'relations': [],
                'contradictions_found': 0,
                'evidence_id': evidence.id,
            }

            # Score source credibility with full audit trail
            credibility, audit_data = self.credibility_scorer.score_source_with_audit(
                evidence.source_url,
                evidence.timestamp,
            )
            evidence.credibility_score = credibility.score

            # Queue async credibility audit write if callback provided
            if self.credibility_audit_callback:
                audit_data['evidence_id'] = evidence.id
                asyncio.create_task(self._save_credibility_audit(audit_data))

            if fast_mode:
                # Fast mode: Single LLM call for both entities and relations
                return await self._fast_extract(evidence)

            # Full mode: Multi-step extraction (slower but more thorough)
            # Step 1: Extract atomic facts
            atomic_facts = await self._extract_atomic_facts(evidence)

            # Step 2: Extract entities from each fact
            all_entities = []
            for fact in atomic_facts:
                entities = await self._extract_entities(fact, evidence.id)
                all_entities.extend(entities)

            # Step 3: Resolve entities against existing graph
            resolved_entities = []
            for entity in all_entities:
                resolved = await self._resolve_entity(entity)
                resolved_entities.append(resolved)
                result['entities'].append(resolved)

            # Step 4: Extract relations using resolved entities
            for fact in atomic_facts:
                relations = await self._extract_relations(fact, resolved_entities, evidence.id)
                for relation in relations:
                    # Check for contradictions before adding
                    contradiction = await self._check_contradiction(relation)
                    if contradiction:
                        self.contradictions.append(contradiction)
                        await self.store.add_contradiction(contradiction)
                        result['contradictions_found'] += 1

                    # Add relation to store
                    await self.store.add_relation(relation, self.session_id)
                    result['relations'].append(relation)

            return result

    async def _fast_extract(self, evidence: KGEvidence) -> dict:
        """Fast extraction using spaCy NER + LLM for relations.

        Uses spaCy for ~100x faster entity extraction, with LLM only for:
        - Domain-specific entity types (CONCEPT, CLAIM, METHOD, etc.)
        - Relation extraction between entities
        """
        logger.debug("KG fast_extract: evidence=%s", evidence.id)
        result = {
            'entities': [],
            'relations': [],
            'contradictions_found': 0,
            'evidence_id': evidence.id,
        }

        # Step 1: Fast entity extraction with spaCy + LLM for domain types
        if self.use_fast_ner and self.fast_ner and self.fast_ner.enabled:
            # Use spaCy for standard entities + LLM for domain-specific types
            entities = await self.fast_ner.extract_with_llm(
                evidence.content,
                self.llm_callback,
                source_id=evidence.id,
                extract_domain_types=True,  # Also use LLM for CONCEPT, CLAIM, etc.
            )

            # Resolve and add entities (improvement #1: raised cap 5->8)
            for entity in entities[:8]:
                # Improvement #7: propagate source URL to entity properties
                if evidence.source_url:
                    entity.properties.setdefault('source_urls', [])
                    if evidence.source_url not in entity.properties['source_urls']:
                        entity.properties['source_urls'].append(evidence.source_url)
                resolved = await self._resolve_entity(entity)
                result['entities'].append(resolved)

        else:
            # Fallback to LLM-only extraction
            return await self._llm_only_extract(evidence)

        # Step 2: Extract relations using LLM (needs semantic understanding)
        if len(result['entities']) >= 2:
            relations = await self._extract_relations_fast(
                evidence.content,
                result['entities'],
                evidence.id,
                credibility_score=evidence.credibility_score,  # improvement #5
            )

            for relation in relations:
                # Check for contradictions
                contradiction = await self._check_contradiction(relation)
                if contradiction:
                    self.contradictions.append(contradiction)
                    await self.store.add_contradiction(contradiction)
                    result['contradictions_found'] += 1

                await self.store.add_relation(relation, self.session_id)
                result['relations'].append(relation)

            # Improvement #4: cross-evidence co-occurrence links
            co_relations = self._build_co_occurrence_links(
                result['entities'], result['relations'], evidence,
            )
            for relation in co_relations:
                await self.store.add_relation(relation, self.session_id)
                result['relations'].append(relation)

        return result

    async def _llm_only_extract(self, evidence: KGEvidence) -> dict:
        """Fallback LLM-only extraction when spaCy is unavailable."""
        logger.debug("KG llm_only_extract: evidence=%s", evidence.id)
        result = {
            'entities': [],
            'relations': [],
            'contradictions_found': 0,
            'evidence_id': evidence.id,
        }

        entity_types_list = ", ".join(list(ENTITY_TYPES.keys())[:10])

        prompt = (
            "Extract key entities and their relationships from "
            "this research evidence.\n\n"
            f"Evidence: {evidence.content}\n\n"
            f"Entity Types: {entity_types_list}\n\n"
            "For each entity provide name, type, description.\n"
            "For METRIC entities, also include value and unit.\n"
            "For CLAIM entities, also include attributed_to.\n\n"
            f"Allowed predicates (use ONLY these):\n"
            f"{CANONICAL_PREDICATES_PROMPT}\n\n"
            "Extract up to 8 entities and 5 relations. "
            "Focus on the most important facts."
        )

        kg_schema = {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "type": {"type": "string"},
                                "description": {"type": "string"},
                                "value": {"type": "string"},
                                "unit": {"type": "string"},
                                "attributed_to": {"type": "string"},
                            },
                            "required": ["name", "type"],
                        },
                    },
                    "relations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "subject": {"type": "string"},
                                "predicate": {"type": "string"},
                                "object": {"type": "string"},
                            },
                            "required": [
                                "subject", "predicate", "object",
                            ],
                        },
                    },
                },
                "required": ["entities", "relations"],
            },
        }

        try:
            response = await self.llm_callback(
                prompt, output_format=kg_schema,
            )

            # Parse response (structured or text fallback)
            if isinstance(response, dict):
                data = response
            else:
                match = re.search(r'\{.*\}', response, re.DOTALL)
                if not match:
                    return result
                data = json.loads(match.group())

            # Process entities and relations from parsed data
            confidence = evidence.credibility_score if evidence.credibility_score else 0.8

            for e in data.get('entities', [])[:8]:
                if isinstance(e, dict) and e.get('name'):
                    entity_type = e.get('type', 'CONCEPT').upper()

                    props: dict[str, Any] = {}
                    if e.get('description'):
                        props['description'] = e['description']
                    if evidence.source_url:
                        props['source_urls'] = [evidence.source_url]
                    if entity_type == 'METRIC' and e.get('value'):
                        props['metric_value'] = e['value']
                        props['metric_unit'] = e.get('unit', '')
                    if entity_type == 'CLAIM':
                        if e.get('attributed_to'):
                            props['attributed_to'] = e['attributed_to']
                        if e.get('date_claimed'):
                            props['date_claimed'] = e['date_claimed']

                    entity = Entity(
                        id=self._generate_id(),
                        name=e['name'],
                        entity_type=entity_type,
                        aliases=[],
                        sources=[evidence.id],
                        properties=props,
                    )
                    resolved = await self._resolve_entity(entity)
                    result['entities'].append(resolved)

            name_to_id = {e.name.lower(): e.id for e in result['entities']}
            for r in data.get('relations', [])[:5]:
                if isinstance(r, dict):
                    subject_id = name_to_id.get(r.get('subject', '').lower())
                    object_id = name_to_id.get(r.get('object', '').lower())

                    if subject_id and object_id and subject_id != object_id:
                        relation = Relation(
                            id=self._generate_id(),
                            subject_id=subject_id,
                            predicate=self._normalize_predicate(
                                r.get('predicate', 'related_to')),
                            object_id=object_id,
                            source_id=evidence.id,
                            confidence=confidence,
                        )

                        contradiction = await self._check_contradiction(relation)
                        if contradiction:
                            self.contradictions.append(contradiction)
                            await self.store.add_contradiction(contradiction)
                            result['contradictions_found'] += 1

                        await self.store.add_relation(relation, self.session_id)
                        result['relations'].append(relation)

            co_relations = self._build_co_occurrence_links(
                result['entities'], result['relations'], evidence,
            )
            for relation in co_relations:
                await self.store.add_relation(relation, self.session_id)
                result['relations'].append(relation)

        except Exception as e:
            # KG extraction is non-blocking but we surface the error in the result
            logger.error("KG llm_only_extract failed: %s", e, exc_info=True)
            result['error'] = f"{type(e).__name__}: {e}"

        return result

    async def _extract_relations_fast(
        self,
        text: str,
        entities: list[Entity],
        source_id: str,
        credibility_score: float | None = None,
    ) -> list[Relation]:
        """Fast relation extraction for pre-extracted entities.

        Uses a compact prompt focused only on relations.
        """
        if len(entities) < 2:
            return []

        entity_names = ", ".join([e.name for e in entities[:8]])

        prompt = (
            f"Given these entities: {entity_names}\n\n"
            f"Extract relationships from this text:\n"
            f"{text[:500]}\n\n"
            f"Allowed predicates (use ONLY these):\n"
            f"{CANONICAL_PREDICATES_PROMPT}\n\n"
            "Return max 5 relations."
        )

        rel_schema = {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "relations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "subject": {"type": "string"},
                                "predicate": {"type": "string"},
                                "object": {"type": "string"},
                            },
                            "required": [
                                "subject", "predicate", "object",
                            ],
                        },
                    },
                },
                "required": ["relations"],
            },
        }

        try:
            response = await self.llm_callback(
                prompt, output_format=rel_schema,
            )

            if isinstance(response, dict):
                data = response.get("relations", [])
            else:
                match = re.search(r'\[.*\]', response, re.DOTALL)
                if not match:
                    return []
                data = json.loads(match.group())

            # Map entity names to IDs
            name_to_id = {e.name.lower(): e.id for e in entities}
            for e in entities:
                for alias in e.aliases:
                    name_to_id[alias.lower()] = e.id

            confidence = credibility_score if credibility_score else 0.8

            relations = []
            for r in data[:5]:
                if not isinstance(r, dict):
                    continue

                subject_id = name_to_id.get(r.get('subject', '').lower())
                object_id = name_to_id.get(r.get('object', '').lower())

                if subject_id and object_id and subject_id != object_id:
                    relation = Relation(
                        id=self._generate_id(),
                        subject_id=subject_id,
                        predicate=self._normalize_predicate(r.get('predicate', 'related_to')),
                        object_id=object_id,
                        source_id=source_id,
                        confidence=confidence,
                    )
                    relations.append(relation)

            return relations

        except Exception:
            logger.warning("KG extract_relations_fast failed", exc_info=True)
            return []

    async def add_evidence_batch(self, evidence_items: list[KGEvidence], batch_size: int = 5) -> dict:
        """Process multiple evidence items in batches for efficiency.

        Groups evidence items and extracts from multiple in a single LLM call.

        Args:
            evidence_items: List of evidence to process
            batch_size: Number of evidence items per LLM call

        Returns:
            Aggregated results from all evidence items
        """
        logger.info("KG batch: %d evidence items, batch_size=%d", len(evidence_items), batch_size)
        result = {
            'total_entities': 0,
            'total_relations': 0,
            'total_contradictions': 0,
            'processed': 0,
        }

        # Score credibility for all evidence items first
        for evidence in evidence_items:
            # Score source credibility with full audit trail
            credibility, audit_data = self.credibility_scorer.score_source_with_audit(
                evidence.source_url,
                evidence.timestamp,
            )
            evidence.credibility_score = credibility.score

            # Queue async credibility audit write if callback provided
            if self.credibility_audit_callback:
                audit_data['evidence_id'] = evidence.id
                asyncio.create_task(self._save_credibility_audit(audit_data))

        # Process in batches
        for i in range(0, len(evidence_items), batch_size):
            batch = evidence_items[i:i + batch_size]
            batch_result = await self._extract_batch(batch)

            result['total_entities'] += batch_result.get('entities_count', 0)
            result['total_relations'] += batch_result.get('relations_count', 0)
            result['total_contradictions'] += batch_result.get('contradictions', 0)
            result['processed'] += len(batch)

        return result

    async def _extract_batch(self, evidence_items: list[KGEvidence]) -> dict:
        """Extract entities and relations from a batch of evidence in one LLM call."""
        result = {'entities_count': 0, 'relations_count': 0, 'contradictions': 0}

        if not evidence_items:
            return result

        entity_types_list = ", ".join(list(ENTITY_TYPES.keys())[:10])

        # Format evidence items for batch processing
        evidence_text = "\n\n".join([
            f"Evidence {i+1}: {e.content[:500]}"
            for i, e in enumerate(evidence_items)
        ])

        prompt = (
            "Extract key entities and relationships from these "
            "research evidence items.\n\n"
            f"{evidence_text}\n\n"
            f"Entity Types: {entity_types_list}\n\n"
            "For each entity provide name, type, and description.\n"
            "For METRIC entities, also include value and unit.\n\n"
            f"Allowed predicates (use ONLY these):\n"
            f"{CANONICAL_PREDICATES_PROMPT}\n\n"
            "Max 5 entities and 4 relations per evidence item."
        )

        batch_schema = {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "evidence": {"type": "integer"},
                                "entities": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "type": {"type": "string"},
                                            "description": {
                                                "type": "string",
                                            },
                                            "value": {"type": "string"},
                                            "unit": {"type": "string"},
                                        },
                                        "required": ["name", "type"],
                                    },
                                },
                                "relations": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "subject": {
                                                "type": "string",
                                            },
                                            "predicate": {
                                                "type": "string",
                                            },
                                            "object": {
                                                "type": "string",
                                            },
                                        },
                                        "required": [
                                            "subject", "predicate",
                                            "object",
                                        ],
                                    },
                                },
                            },
                            "required": [
                                "evidence", "entities", "relations",
                            ],
                        },
                    },
                },
                "required": ["results"],
            },
        }

        try:
            response = await self.llm_callback(
                prompt, output_format=batch_schema,
            )

            # Parse response (structured or text fallback)
            if isinstance(response, dict):
                data = response.get("results", [])
            else:
                match = re.search(r'\[.*\]', response, re.DOTALL)
                if not match:
                    return result
                data = json.loads(match.group())

            # Process all items from parsed data
            for item in data:
                if not isinstance(item, dict):
                    continue

                evidence_idx = item.get('evidence', 1) - 1
                if evidence_idx < 0 or evidence_idx >= len(evidence_items):
                    evidence_idx = 0
                evidence = evidence_items[evidence_idx]
                confidence = evidence.credibility_score if evidence.credibility_score else 0.8

                entities = []
                for e in item.get('entities', [])[:5]:
                    if isinstance(e, dict) and e.get('name'):
                        entity_type = e.get('type', 'CONCEPT').upper()
                        props: dict[str, Any] = {}
                        if e.get('description'):
                            props['description'] = e['description']
                        if evidence.source_url:
                            props['source_urls'] = [evidence.source_url]
                        if entity_type == 'METRIC' and e.get('value'):
                            props['metric_value'] = e['value']
                            props['metric_unit'] = e.get('unit', '')

                        entity = Entity(
                            id=self._generate_id(),
                            name=e['name'],
                            entity_type=entity_type,
                            aliases=[],
                            sources=[evidence.id],
                            properties=props,
                        )
                        resolved = await self._resolve_entity(entity)
                        entities.append(resolved)
                        result['entities_count'] += 1

                name_to_id = {e.name.lower(): e.id for e in entities}
                for r in item.get('relations', [])[:4]:
                    if isinstance(r, dict):
                        subject_id = name_to_id.get(r.get('subject', '').lower())
                        object_id = name_to_id.get(r.get('object', '').lower())

                        if subject_id and object_id and subject_id != object_id:
                            relation = Relation(
                                id=self._generate_id(),
                                subject_id=subject_id,
                                predicate=self._normalize_predicate(
                                    r.get('predicate', 'related_to')),
                                object_id=object_id,
                                source_id=evidence.id,
                                confidence=confidence,
                            )

                            contradiction = await self._check_contradiction(relation)
                            if contradiction:
                                self.contradictions.append(contradiction)
                                await self.store.add_contradiction(contradiction)
                                result['contradictions'] += 1

                            await self.store.add_relation(relation, self.session_id)
                            result['relations_count'] += 1

        except Exception as e:
            logger.error("KG extract_batch failed: %s", e, exc_info=True)
            result['error'] = f"{type(e).__name__}: {e}"

        return result

    async def _extract_atomic_facts(self, evidence: KGEvidence) -> list[str]:
        """Break evidence into minimal, self-contained atomic facts."""
        prompt = f"""Extract atomic facts from this research evidence.
Each atomic fact should be:
- Self-contained (understandable without context)
- Minimal (single piece of information)
- Factual (not opinion unless attributed)

Evidence: {evidence.content}

Return as JSON array of strings.
Example: ["Fact 1", "Fact 2", "Fact 3"]"""

        facts_schema = {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "facts": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["facts"],
            },
        }

        try:
            response = await self.llm_callback(prompt, output_format=facts_schema)
        except TypeError:
            response = await self.llm_callback(prompt)

        if isinstance(response, dict):
            return response.get("facts", [])
        return self._parse_json_array(response)

    async def _extract_entities(self, text: str, source_id: str) -> list[Entity]:
        """Extract entities from text using LLM."""
        entity_types_list = "\n".join([f"- {k}: {v}" for k, v in list(ENTITY_TYPES.items())[:8]])

        prompt = f"""Extract key entities from this text.

Text: {text}

Entity Types:
{entity_types_list}

For each entity provide:
- name: The canonical name (singular, consistent capitalization)
- type: One of the entity types above
- aliases: Other names/mentions that refer to this entity

Return as JSON array:
[
  {{"name": "Entity name", "type": "CONCEPT", "aliases": ["alias1"]}}
]"""

        entities_schema = {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "type": {"type": "string"},
                                "aliases": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["name", "type"],
                        },
                    },
                },
                "required": ["entities"],
            },
        }

        try:
            response = await self.llm_callback(prompt, output_format=entities_schema)
        except TypeError:
            response = await self.llm_callback(prompt)

        if isinstance(response, dict):
            entities_data = response.get("entities", [])
        else:
            entities_data = self._parse_json_array(response)

        entities = []
        for e in entities_data:
            if not isinstance(e, dict):
                continue

            entity = Entity(
                id=self._generate_id(),
                name=e.get('name', ''),
                entity_type=e.get('type', 'CONCEPT').upper(),
                aliases=e.get('aliases', []),
                sources=[source_id],
            )
            entities.append(entity)

        return entities

    async def _resolve_entity(self, entity: Entity) -> Entity:
        """Resolve entity against existing global set.

        Uses name matching (embedding similarity if available).
        """
        # First check exact name match
        name_key = entity.name.lower().strip()
        if name_key in self.entity_by_name:
            existing_id = self.entity_by_name[name_key]
            existing = await self.store.get_entity(existing_id)
            if existing:
                # Merge: add this as a source
                if entity.sources:
                    existing.sources = list(set(existing.sources + entity.sources))
                existing.aliases = list(set(existing.aliases + entity.aliases + [entity.name]))
                # Merge source_urls from properties
                existing_urls = set(existing.properties.get('source_urls', []))
                new_urls = set(entity.properties.get('source_urls', []))
                if new_urls:
                    existing.properties['source_urls'] = list(existing_urls | new_urls)
                # Keep first description, don't overwrite
                if entity.properties.get('description'):
                    existing.properties.setdefault('description', entity.properties['description'])
                await self.store.add_entity(existing, self.session_id)  # Update
                return existing

        # Check aliases
        for existing_name, existing_id in list(self.entity_by_name.items()):
            existing = await self.store.get_entity(existing_id)
            if existing and entity.name.lower() in [a.lower() for a in existing.aliases]:
                # Found via alias
                if entity.sources:
                    existing.sources = list(set(existing.sources + entity.sources))
                # Merge source_urls
                existing_urls = set(existing.properties.get('source_urls', []))
                new_urls = set(entity.properties.get('source_urls', []))
                if new_urls:
                    existing.properties['source_urls'] = list(existing_urls | new_urls)
                await self.store.add_entity(existing, self.session_id)
                return existing

        # No match found - add as new entity
        await self.store.add_entity(entity, self.session_id)
        self.entity_by_name[name_key] = entity.id
        self.entity_by_type.setdefault(entity.entity_type, []).append(entity.id)

        # Also index aliases
        for alias in entity.aliases:
            alias_key = alias.lower().strip()
            if alias_key not in self.entity_by_name:
                self.entity_by_name[alias_key] = entity.id

        return entity

    async def _extract_relations(
        self, text: str, entities: list[Entity], source_id: str
    ) -> list[Relation]:
        """Extract relations using resolved entities as context."""
        if not entities:
            return []

        entity_context = "\n".join([
            f"- {e.name} ({e.entity_type})" for e in entities
        ])

        prompt = f"""Extract relationships between these entities from the text.

Text: {text}

Known entities:
{entity_context}

For each relationship provide:
- subject: The subject entity name (must be from known entities)
- predicate: The relationship (1-3 words max, e.g., "causes", "is part of")
- object: The object entity name (must be from known entities)
- confidence: 0.0-1.0 confidence score

Return as JSON array:
[
  {{"subject": "Entity1", "predicate": "causes", "object": "Entity2", "confidence": 0.9}}
]"""

        relations_schema = {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "relations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "subject": {"type": "string"},
                                "predicate": {"type": "string"},
                                "object": {"type": "string"},
                                "confidence": {"type": "number"},
                            },
                            "required": ["subject", "predicate", "object"],
                        },
                    },
                },
                "required": ["relations"],
            },
        }

        try:
            response = await self.llm_callback(prompt, output_format=relations_schema)
        except TypeError:
            response = await self.llm_callback(prompt)

        if isinstance(response, dict):
            relations_data = response.get("relations", [])
        else:
            relations_data = self._parse_json_array(response)

        # Map entity names to IDs
        name_to_id = {e.name.lower(): e.id for e in entities}
        for e in entities:
            for alias in e.aliases:
                name_to_id[alias.lower()] = e.id

        relations = []
        for r in relations_data:
            if not isinstance(r, dict):
                continue

            subject_name = r.get('subject', '').lower()
            object_name = r.get('object', '').lower()

            subject_id = name_to_id.get(subject_name)
            object_id = name_to_id.get(object_name)

            if subject_id and object_id and subject_id != object_id:
                relation = Relation(
                    id=self._generate_id(),
                    subject_id=subject_id,
                    predicate=self._normalize_predicate(r.get('predicate', 'related_to')),
                    object_id=object_id,
                    source_id=source_id,
                    confidence=r.get('confidence', 0.8),
                )
                relations.append(relation)

        return relations

    async def _check_contradiction(self, new_relation: Relation) -> Contradiction | None:
        """Check if new relation contradicts existing relations."""
        # Look for relations with same subject and object but different predicate
        existing_relations = await self.store.get_entity_relations(new_relation.subject_id)

        for existing in existing_relations.get('outgoing', []):
            if existing.get('target_id') == new_relation.object_id:
                existing_pred = existing.get('predicate', '')
                if self._predicates_contradict(existing_pred, new_relation.predicate):
                    return Contradiction(
                        id=self._generate_id(),
                        relation1_id=existing.get('relation_id', ''),
                        relation2_id=new_relation.id,
                        contradiction_type='direct',
                        description=(
                            f"Conflicting predicates: '{existing_pred}' vs '{new_relation.predicate}'"
                        ),
                        severity='high' if new_relation.confidence > 0.8 else 'medium',
                    )

        return None

    def _predicates_contradict(self, pred1: str, pred2: str) -> bool:
        """Check if two predicates are contradictory."""
        p1 = pred1.lower().strip()
        p2 = pred2.lower().strip()

        for contra_pair in self.CONTRADICTORY_PREDICATES:
            if (contra_pair[0] in p1 and contra_pair[1] in p2) or \
               (contra_pair[1] in p1 and contra_pair[0] in p2):
                return True

        return False

    def _normalize_predicate(self, predicate: str) -> str:
        """Normalize predicate to canonical form using PREDICATE_CANONICAL_MAP.

        Tries exact match, then 2-word prefix, then first word.
        Falls back to underscore-joined truncation if no canonical match.
        """
        words = predicate.lower().strip().split()[:4]
        if not words:
            return 'correlates_with'

        normalized = '_'.join(words)

        # Try exact match
        if normalized in PREDICATE_CANONICAL_MAP:
            return PREDICATE_CANONICAL_MAP[normalized]

        # Try first 2 words
        if len(words) >= 2:
            two_word = '_'.join(words[:2])
            if two_word in PREDICATE_CANONICAL_MAP:
                return PREDICATE_CANONICAL_MAP[two_word]

        # Try first word only
        if words[0] in PREDICATE_CANONICAL_MAP:
            return PREDICATE_CANONICAL_MAP[words[0]]

        # Fallback: return cleaned form (max 3 words)
        return '_'.join(words[:3])

    def _build_co_occurrence_links(
        self,
        entities: list[Entity],
        explicit_relations: list[Relation],
        evidence: KGEvidence,
    ) -> list[Relation]:
        """Create co_occurs_with relations between entities from the same evidence
        that don't already have an explicit relation. Capped at 5 per evidence item."""
        if len(entities) < 2:
            return []

        # Build set of entity pairs that already have explicit relations
        linked_pairs: set[tuple[str, str]] = set()
        for rel in explicit_relations:
            linked_pairs.add((rel.subject_id, rel.object_id))
            linked_pairs.add((rel.object_id, rel.subject_id))

        co_relations: list[Relation] = []
        for i, e1 in enumerate(entities):
            for e2 in entities[i + 1:]:
                if e1.id == e2.id:
                    continue
                if (e1.id, e2.id) in linked_pairs:
                    continue

                relation = Relation(
                    id=self._generate_id(),
                    subject_id=e1.id,
                    predicate='co_occurs_with',
                    object_id=e2.id,
                    source_id=evidence.id,
                    confidence=0.4,
                )
                co_relations.append(relation)

                if len(co_relations) >= 5:
                    return co_relations

        return co_relations

    def _generate_id(self) -> str:
        """Generate unique ID."""
        # [HARDENED] BUG-010: Use 16 chars to reduce collision probability
        return str(uuid.uuid4())[:16]

    def _parse_json_array(self, text: str | dict | list) -> list:
        """Parse JSON array from LLM response with multiple fallbacks."""
        # Handle structured output passthrough
        if isinstance(text, list):
            return text
        if isinstance(text, dict):
            for key in ("results", "entities", "relations", "facts"):
                if key in text and isinstance(text[key], list):
                    return text[key]
            return []

        return _extract_json_array(text) or []

    async def get_stats(self) -> dict:
        """Get knowledge graph statistics."""
        stats = await self.store.get_stats()
        stats['contradictions'] = len(self.contradictions)
        stats['indexed_names'] = len(self.entity_by_name)
        stats['fast_ner_enabled'] = self.use_fast_ner and self.fast_ner is not None
        if self.fast_ner:
            stats['fast_ner'] = self.fast_ner.get_stats()
        return stats

    def _find_entity_in_kg(
        self,
        name: str,
        snapshot: dict[str, str] | None = None,
    ) -> str | None:
        """Find an entity ID in the KG using exact, alias, and substring matching.

        Args:
            name: Entity name to look up
            snapshot: Optional pre-captured snapshot of entity_by_name.
                      Avoids dict-changed-during-iteration when called
                      from async methods that yield between reads.

        Returns entity_id if found, None otherwise.
        """
        names = snapshot if snapshot is not None else self.entity_by_name
        key = name.lower().strip()
        if not key or len(key) < 2:
            return None

        # 1. Exact match
        if key in names:
            return names[key]

        # 2. Check if any KG entity name contains this name or vice versa
        #    (e.g., "Cursor" matches "Cursor AI", "GitHub Copilot" matches "Copilot")
        for kg_name, entity_id in names.items():
            if len(kg_name) < 3 or len(key) < 3:
                continue
            if key in kg_name or kg_name in key:
                return entity_id

        return None

    async def get_kg_support_score(
        self,
        content: str,
        source_url: str | None = None,
    ) -> tuple[float, int, int]:
        """Calculate KG support score for evidence.

        Uses a two-pronged approach:
        1. Extract entities from content and look them up in KG
        2. Check which KG entity names appear in the content text

        Args:
            content: The evidence content to check
            source_url: Optional source URL for context

        Returns:
            Tuple of (support_score, entity_match_count, supporting_relation_count)
        """
        # Snapshot entity_by_name to avoid dict-changed-during-iteration
        # when add_evidence modifies it concurrently via _resolve_entity
        name_snapshot = dict(self.entity_by_name)

        matched_entity_ids: set[str] = set()
        supporting_relations = 0

        # --- Approach 1: Extract entities from content, look up in KG ---
        entities = await self._quick_entity_extract(content)
        for entity_name in entities:
            entity_id = self._find_entity_in_kg(
                entity_name, snapshot=name_snapshot,
            )
            if entity_id:
                matched_entity_ids.add(entity_id)

        # --- Approach 2: Check which KG entities appear in the content ---
        content_lower = content.lower()
        for kg_name, entity_id in name_snapshot.items():
            if len(kg_name) >= 3 and kg_name in content_lower:
                matched_entity_ids.add(entity_id)

        if not matched_entity_ids:
            return 0.0, 0, 0

        # Count supporting relations for matched entities
        for entity_id in matched_entity_ids:
            relations = await self.store.get_entity_relations(entity_id)
            if relations:
                supporting_relations += len(relations.get('outgoing', []))
                supporting_relations += len(relations.get('incoming', []))

        entity_match_count = len(matched_entity_ids)

        # Calculate score: 60% entity match density + 40% relation support
        # Use total KG entity count as denominator for a meaningful ratio
        total_kg_entities = max(len(set(name_snapshot.values())), 1)
        entity_score = min(entity_match_count / min(total_kg_entities, 10), 1.0)
        relation_score = min(supporting_relations / max(entity_match_count * 3, 1), 1.0)

        score = (entity_score * 0.6) + (relation_score * 0.4)
        return score, entity_match_count, supporting_relations

    async def _quick_entity_extract(self, content: str) -> list[str]:
        """Quick entity extraction without LLM call.

        Uses spaCy if available, otherwise simple NLP heuristics.
        Returns a combined list of NER entities + simple pattern matches.
        """
        entities: list[str] = []
        seen: set[str] = set()

        def _add(name: str) -> None:
            key = name.lower().strip()
            if key not in seen and len(key) > 2:
                seen.add(key)
                entities.append(name)

        # Use fast NER if available
        if self.use_fast_ner and self.fast_ner and self.fast_ner.enabled:
            for e in self.fast_ner.extract(content):
                _add(e.name)

        # Always also run pattern-based extraction to catch domain terms
        # that spaCy's standard NER misses (e.g., "Cursor", "Copilot",
        # "LLM", lowercase tech terms).

        # Capitalized phrases (proper nouns)
        for m in re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', content):
            _add(m)

        # Quoted terms
        for m in re.findall(r'"([^"]{2,50})"', content):
            _add(m)

        # Acronyms and tech terms (e.g., "AI", "GPT-4", "LLM", "VSCode")
        for m in re.findall(r'\b([A-Z][A-Z0-9][-A-Za-z0-9]*)\b', content):
            _add(m)

        # CamelCase terms (e.g., "GitHub", "DevOps", "TypeScript")
        for m in re.findall(r'\b([A-Z][a-z]+[A-Z][a-zA-Z]*)\b', content):
            _add(m)

        return entities[:15]

    async def check_contradictions_detailed(
        self,
        content: str,
    ) -> dict:
        """Check for contradictions with detailed results.

        Uses exact predicate matching (not substring) to avoid false positives
        like "enables" matching inside "implements".

        Args:
            content: The evidence content to check

        Returns:
            Dict with has_contradictions bool and list of contradiction details
        """
        contradictions = []

        # Antonym predicate pairs -- if the KG has one and the content uses
        # the other *for the same entity pair*, that's a contradiction.
        antonym_pairs = {
            'increases': 'decreases',
            'decreases': 'increases',
            'causes': 'prevents',
            'prevents': 'causes',
            'enables': 'blocks',
            'blocks': 'enables',
            'improves': 'worsens',
            'worsens': 'improves',
            'outperforms': 'underperforms',
            'underperforms': 'outperforms',
        }

        content_lower = content.lower()

        # Snapshot to avoid dict-changed-during-iteration if
        # add_evidence modifies entity_by_name concurrently
        name_snapshot = dict(self.entity_by_name)

        # Find all KG entities mentioned in the content
        matched_ids: set[str] = set()
        for kg_name, entity_id in name_snapshot.items():
            if len(kg_name) >= 3 and kg_name in content_lower:
                matched_ids.add(entity_id)

        for entity_id in matched_ids:
            relations = await self.store.get_entity_relations(entity_id)
            if not relations:
                continue

            for relation in relations.get('outgoing', []):
                predicate = relation.get('predicate', '').lower().strip()
                target_name = relation.get('target_name', '').lower().strip()

                # Both the target and an antonym predicate must appear in
                # the content for a real contradiction
                if not target_name or target_name not in content_lower:
                    continue

                # Exact match against canonical antonym (not substring)
                antonym = antonym_pairs.get(predicate)
                if not antonym:
                    continue

                # Check if the antonym appears as a standalone word in
                # the content (word-boundary match to avoid substrings)
                if re.search(rf'\b{re.escape(antonym)}\b', content_lower):
                    contradictions.append({
                        "conflicting_id": relation.get('relation_id', 'unknown'),
                        "description": (
                            f"KG relation '{predicate}' for target "
                            f"'{target_name}' but content uses '{antonym}'"
                        ),
                        "severity": "medium",
                    })

        return {
            "has_contradictions": len(contradictions) > 0,
            "contradictions": contradictions,
        }

    # Backward-compat aliases (old name -> new name)
    async def add_finding(self, finding: KGEvidence, fast_mode: bool = True) -> dict:
        """Alias for add_evidence (backward compat)."""
        return await self.add_evidence(finding, fast_mode=fast_mode)

    async def add_findings_batch(self, findings: list[KGEvidence], batch_size: int = 5) -> dict:
        """Alias for add_evidence_batch (backward compat)."""
        return await self.add_evidence_batch(findings, batch_size=batch_size)
