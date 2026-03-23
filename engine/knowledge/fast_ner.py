"""Fast NER using spaCy with LLM fallback.

Provides ~100x faster entity extraction compared to LLM-only approach
for common named entity types, with LLM fallback for domain-specific
entity types (CONCEPT, CLAIM, EVIDENCE, METHOD, etc.).
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

try:
    import spacy
    from spacy.language import Language
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

from ..logging_config import get_logger
from .models import ENTITY_TYPES, Entity

logger = get_logger(__name__)

# Allowed spaCy models (whitelist for security)
ALLOWED_SPACY_MODELS = {
    "en_core_web_sm",
    "en_core_web_md",
    "en_core_web_lg",
    "en_core_web_trf",
}


# Map spaCy NER labels to our entity types
SPACY_TO_ENTITY_TYPE = {
    # Standard NER labels from en_core_web_sm
    "PERSON": "PERSON",
    "ORG": "ORGANIZATION",
    "GPE": "LOCATION",  # Geopolitical entity
    "LOC": "LOCATION",
    "DATE": "DATE",
    "TIME": "DATE",
    "MONEY": "METRIC",
    "PERCENT": "METRIC",
    "QUANTITY": "METRIC",
    "CARDINAL": "METRIC",
    "ORDINAL": "METRIC",
    "PRODUCT": "TECHNOLOGY",
    "EVENT": "CONCEPT",
    "WORK_OF_ART": "SOURCE",
    "LAW": "SOURCE",
    "LANGUAGE": "CONCEPT",
    "FAC": "LOCATION",  # Facility
    "NORP": "ORGANIZATION",  # Nationalities, religious/political groups
}

# Entity types that require LLM extraction (domain-specific)
LLM_ONLY_TYPES = {
    "CONCEPT",
    "CLAIM",
    "EVIDENCE",
    "METHOD",
    "QUOTE",
    "AUTHOR",  # Often needs context to identify
}


@dataclass
class FastNERConfig:
    """Configuration for fast NER."""
    # spaCy model - use large model for better accuracy (still ~50x faster than LLM)
    model_name: str = "en_core_web_lg"
    auto_download_model: bool = False  # Opt-in for auto-download (security)

    # Extraction settings
    min_entity_length: int = 2
    max_entity_length: int = 100
    deduplicate: bool = True

    # LLM fallback
    use_llm_fallback: bool = True
    llm_entity_types: set[str] = field(default_factory=lambda: LLM_ONLY_TYPES.copy())

    # Performance
    batch_size: int = 100  # For batch processing


@dataclass
class ExtractedEntity:
    """Entity extracted by fast NER."""
    name: str
    entity_type: str
    start_char: int
    end_char: int
    confidence: float
    source: str  # "spacy" or "llm" or "heuristic"


class FastNER:
    """Fast named entity recognition using spaCy with LLM fallback.

    Uses spaCy's efficient NER for standard entity types (PERSON, ORG, etc.)
    and falls back to LLM for domain-specific types (CONCEPT, CLAIM, etc.).

    This provides ~100x speedup for standard entities while maintaining
    quality for complex domain entities.

    Example:
        ner = FastNER()

        # Extract entities from text
        entities = ner.extract("OpenAI released GPT-4 in March 2023")
        # [Entity(name="OpenAI", type="ORGANIZATION"),
        #  Entity(name="GPT-4", type="TECHNOLOGY"),
        #  Entity(name="March 2023", type="DATE")]

        # With LLM for domain entities
        async def llm_callback(prompt):
            return await call_claude(prompt)

        entities = await ner.extract_with_llm(
            "Transformers outperform RNNs on sequence tasks",
            llm_callback=llm_callback,
        )
    """

    def __init__(self, config: FastNERConfig | None = None):
        """Initialize fast NER.

        Args:
            config: Configuration options. Uses defaults if not provided.
        """
        self.config = config or FastNERConfig()
        self._nlp: Any | None = None
        self._load_attempted = False

    @property
    def enabled(self) -> bool:
        """Whether spaCy NER is available."""
        return HAS_SPACY and self._get_nlp() is not None

    def _get_nlp(self) -> Any | None:
        """Lazy-load spaCy model."""
        if self._nlp is not None:
            return self._nlp

        if self._load_attempted:
            return None

        self._load_attempted = True

        if not HAS_SPACY:
            return None

        try:
            self._nlp = spacy.load(self.config.model_name)
            return self._nlp
        except OSError:
            # Model not installed - try to download it if allowed
            if not self.config.auto_download_model:
                logger.error("spaCy model load failed", exc_info=True)
                return None

            # Validate model name against whitelist (security)
            if self.config.model_name not in ALLOWED_SPACY_MODELS:
                logger.error(
                    "spaCy model load failed: model %s not in whitelist",
                    self.config.model_name,
                )
                return None

            try:
                from spacy.cli import download
                download(self.config.model_name)
                self._nlp = spacy.load(self.config.model_name)
                return self._nlp
            except Exception:
                logger.error("spaCy model load failed", exc_info=True)
                return None

    def extract(self, text: str, source_id: str = "") -> list[ExtractedEntity]:
        """Extract named entities using spaCy.

        Fast extraction of standard entity types only.
        Does NOT extract domain-specific types (CONCEPT, CLAIM, etc.).

        Args:
            text: Text to extract entities from
            source_id: Source ID for provenance tracking

        Returns:
            List of extracted entities
        """
        nlp = self._get_nlp()
        if nlp is None:
            # Fallback to heuristic extraction
            return self._heuristic_extract(text, source_id)

        doc = nlp(text)
        entities = []
        seen = set()  # For deduplication

        for ent in doc.ents:
            # Filter by length
            if len(ent.text) < self.config.min_entity_length:
                continue
            if len(ent.text) > self.config.max_entity_length:
                continue

            # Map to our entity type
            entity_type = SPACY_TO_ENTITY_TYPE.get(ent.label_, "CONCEPT")

            # Deduplication
            if self.config.deduplicate:
                key = (ent.text.lower(), entity_type)
                if key in seen:
                    continue
                seen.add(key)

            entities.append(ExtractedEntity(
                name=ent.text,
                entity_type=entity_type,
                start_char=ent.start_char,
                end_char=ent.end_char,
                confidence=0.85,  # spaCy's en_core_web_sm is reasonably accurate
                source="spacy",
            ))

        # Also extract noun chunks as potential concepts
        entities.extend(self._extract_noun_chunks(doc, seen))

        if not entities:
            logger.warning("NER extraction returned empty")

        return entities

    def _extract_noun_chunks(
        self,
        doc: Any,
        seen: set,
    ) -> list[ExtractedEntity]:
        """Extract important noun chunks as potential concepts."""
        entities = []

        for chunk in doc.noun_chunks:
            # Filter short chunks
            if len(chunk.text) < 4:
                continue

            # Filter common/generic phrases
            if chunk.root.pos_ not in ("NOUN", "PROPN"):
                continue

            # Skip if already captured as named entity
            text_lower = chunk.text.lower()
            if (text_lower, "CONCEPT") in seen:
                continue

            # Skip very generic chunks
            if chunk.root.text.lower() in {"thing", "way", "time", "people", "year"}:
                continue

            # Only include chunks with meaningful modifiers or compounds
            if len(chunk.text.split()) < 2 and chunk.root.dep_ not in ("nsubj", "dobj", "pobj"):
                continue

            seen.add((text_lower, "CONCEPT"))

            entities.append(ExtractedEntity(
                name=chunk.text,
                entity_type="CONCEPT",
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                confidence=0.6,  # Lower confidence for noun chunks
                source="spacy",
            ))

        return entities[:5]  # Limit noun chunk extraction

    def _heuristic_extract(self, text: str, source_id: str) -> list[ExtractedEntity]:
        """Fallback heuristic extraction when spaCy is unavailable.

        Uses regex patterns to extract likely entities.
        """
        entities = []
        seen = set()

        # Extract capitalized phrases (likely proper nouns)
        cap_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        for match in re.finditer(cap_pattern, text):
            name = match.group(1)
            if len(name) < self.config.min_entity_length:
                continue

            key = name.lower()
            if key in seen:
                continue
            seen.add(key)

            # Guess type based on patterns
            entity_type = "CONCEPT"
            if re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+$', name):
                entity_type = "PERSON"  # Two capitalized words likely a name

            entities.append(ExtractedEntity(
                name=name,
                entity_type=entity_type,
                start_char=match.start(),
                end_char=match.end(),
                confidence=0.5,  # Low confidence for heuristic
                source="heuristic",
            ))

        # Extract quoted terms
        for match in re.finditer(r'"([^"]+)"', text):
            name = match.group(1)
            if len(name) < self.config.min_entity_length:
                continue
            if name.lower() in seen:
                continue
            seen.add(name.lower())

            entities.append(ExtractedEntity(
                name=name,
                entity_type="CONCEPT",
                start_char=match.start(),
                end_char=match.end(),
                confidence=0.6,
                source="heuristic",
            ))

        # Extract technology-like terms (CamelCase, acronyms)
        tech_pattern = r'\b([A-Z][a-z]+[A-Z][a-zA-Z]*|[A-Z]{2,}(?:-[A-Z0-9]+)*)\b'
        for match in re.finditer(tech_pattern, text):
            name = match.group(1)
            if name.lower() in seen:
                continue
            seen.add(name.lower())

            entities.append(ExtractedEntity(
                name=name,
                entity_type="TECHNOLOGY",
                start_char=match.start(),
                end_char=match.end(),
                confidence=0.6,
                source="heuristic",
            ))

        return entities[:15]  # Limit heuristic extraction

    async def extract_with_llm(
        self,
        text: str,
        llm_callback: Callable[[str], Any],
        source_id: str = "",
        extract_domain_types: bool = True,
    ) -> list[Entity]:
        """Extract entities using spaCy + LLM fallback.

        Uses spaCy for standard entities and LLM for domain-specific types.

        Args:
            text: Text to extract entities from
            llm_callback: Async function to call LLM
            source_id: Source ID for provenance
            extract_domain_types: Whether to use LLM for domain types

        Returns:
            List of Entity objects ready for knowledge graph
        """
        import uuid

        # First, fast extraction with spaCy
        fast_entities = self.extract(text, source_id)

        # Convert to Entity objects
        entities = []
        seen_names = set()

        for ext in fast_entities:
            # Skip low confidence heuristic extractions for non-standard types
            if ext.source == "heuristic" and ext.entity_type in LLM_ONLY_TYPES:
                continue

            seen_names.add(ext.name.lower())

            entities.append(Entity(
                id=str(uuid.uuid4())[:8],
                name=ext.name,
                entity_type=ext.entity_type,
                aliases=[],
                sources=[source_id] if source_id else [],
                confidence=ext.confidence,
            ))

        # LLM extraction for domain-specific types
        if extract_domain_types and self.config.use_llm_fallback:
            llm_entities = await self._llm_extract_domain_types(
                text, llm_callback, source_id, seen_names
            )
            entities.extend(llm_entities)

        return entities

    async def _llm_extract_domain_types(
        self,
        text: str,
        llm_callback: Callable[[str], Any],
        source_id: str,
        seen_names: set[str],
    ) -> list[Entity]:
        """Extract domain-specific entity types using LLM.

        Only extracts types not covered by spaCy.
        """
        import json
        import uuid

        domain_types = list(self.config.llm_entity_types)
        types_desc = "\n".join([
            f"- {t}: {ENTITY_TYPES.get(t, '')}"
            for t in domain_types
        ])

        prompt = (
            "Extract domain-specific entities from this text.\n\n"
            f"Text: {text}\n\n"
            f"Only extract these entity types:\n{types_desc}\n\n"
            "For METRIC entities, also include value and unit.\n"
            "For CLAIM entities, also include attributed_to.\n\n"
            "Max 8 entities. Focus on the most important domain "
            "concepts. Do NOT extract people, organizations, "
            "locations, or dates (those are already handled)."
        )

        ner_schema = {
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
                },
                "required": ["entities"],
            },
        }

        try:
            try:
                response = await llm_callback(
                    prompt, output_format=ner_schema,
                )
            except TypeError:
                response = await llm_callback(prompt)

            if isinstance(response, dict):
                data = response.get("entities", [])
            else:
                match = re.search(r'\[.*?\]', response, re.DOTALL)
                if not match:
                    return []
                data = json.loads(match.group())

            entities = []
            for item in data:
                if not isinstance(item, dict):
                    continue

                name = item.get("name", "").strip()
                entity_type = item.get("type", "CONCEPT").upper()

                # Skip if already seen
                if name.lower() in seen_names:
                    continue

                # Validate entity type
                if entity_type not in ENTITY_TYPES:
                    entity_type = "CONCEPT"

                seen_names.add(name.lower())

                # Build properties from LLM extraction
                props = {}
                if item.get("description"):
                    props["description"] = item["description"]
                if entity_type == "METRIC" and item.get("value"):
                    props["metric_value"] = item["value"]
                    props["metric_unit"] = item.get("unit", "")
                if entity_type == "CLAIM" and item.get("attributed_to"):
                    props["attributed_to"] = item["attributed_to"]

                entities.append(Entity(
                    id=str(uuid.uuid4())[:8],
                    name=name,
                    entity_type=entity_type,
                    aliases=[],
                    sources=[source_id] if source_id else [],
                    confidence=0.75,  # LLM extraction confidence
                    properties=props,
                ))

            return entities[:8]

        except Exception:
            logger.warning("LLM domain type extraction failed", exc_info=True)
            return []

    def extract_batch(
        self,
        texts: list[str],
        source_ids: list[str] | None = None,
    ) -> list[list[ExtractedEntity]]:
        """Extract entities from multiple texts efficiently.

        Uses spaCy's pipe() for batch processing.

        Args:
            texts: List of texts to process
            source_ids: Optional list of source IDs (same length as texts)

        Returns:
            List of entity lists, one per input text
        """
        nlp = self._get_nlp()
        if nlp is None:
            # Fallback to sequential heuristic extraction
            source_ids = source_ids or [""] * len(texts)
            return [
                self._heuristic_extract(text, sid)
                for text, sid in zip(texts, source_ids)
            ]

        results = []
        source_ids = source_ids or [""] * len(texts)

        # Process in batches using spaCy pipe
        for doc, _source_id in zip(
            nlp.pipe(texts, batch_size=self.config.batch_size),
            source_ids,
        ):
            entities = []
            seen = set()

            for ent in doc.ents:
                if len(ent.text) < self.config.min_entity_length:
                    continue
                if len(ent.text) > self.config.max_entity_length:
                    continue

                entity_type = SPACY_TO_ENTITY_TYPE.get(ent.label_, "CONCEPT")

                if self.config.deduplicate:
                    key = (ent.text.lower(), entity_type)
                    if key in seen:
                        continue
                    seen.add(key)

                entities.append(ExtractedEntity(
                    name=ent.text,
                    entity_type=entity_type,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    confidence=0.85,
                    source="spacy",
                ))

            results.append(entities)

        return results

    def get_stats(self) -> dict:
        """Get NER statistics."""
        return {
            "spacy_available": HAS_SPACY,
            "model_loaded": self._nlp is not None,
            "model_name": self.config.model_name if self._nlp else None,
            "llm_fallback_enabled": self.config.use_llm_fallback,
            "llm_entity_types": list(self.config.llm_entity_types),
        }


# Global instance
_global_ner: FastNER | None = None


def get_fast_ner(config: FastNERConfig | None = None) -> FastNER:
    """Get or create the global FastNER instance.

    Args:
        config: Configuration (only used on first call)

    Returns:
        The global FastNER instance
    """
    global _global_ner

    if _global_ner is None:
        _global_ner = FastNER(config)

    return _global_ner
