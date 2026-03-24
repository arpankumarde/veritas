"""Knowledge graph processing extracted from ManagerAgent.

Handles converting evidence to KG entities/relations and indexing
for hybrid retrieval. Reduces ManagerAgent's responsibilities.
"""

import logging
from collections.abc import Callable

from ..knowledge import IncrementalKnowledgeGraph, KGFinding
from ..models.findings import Evidence
from ..retrieval import FindingsRetriever

logger = logging.getLogger(__name__)

# Type alias for the log callback
LogCallback = Callable[[str, str], None]


class KGProcessor:
    """Processes evidence into the knowledge graph and retrieval index.

    Extracted from ManagerAgent to reduce its size and isolate KG concerns.
    """

    def __init__(
        self,
        knowledge_graph: IncrementalKnowledgeGraph,
        findings_retriever: FindingsRetriever,
        log: LogCallback | None = None,
    ):
        self.knowledge_graph = knowledge_graph
        self.findings_retriever = findings_retriever
        self._log = log or (lambda msg, style="": None)

    async def process_evidence(
        self, evidence_list: list[Evidence], session_id: str
    ) -> None:
        """Process evidence into the knowledge graph and hybrid retrieval index.

        Uses batch processing for speed (multiple evidence items per LLM call)
        while still building the full KG that agents can query during verification.
        Also indexes evidence for semantic search via hybrid retrieval.

        Args:
            evidence_list: List of evidence items to process
            session_id: Current check session ID
        """
        if not evidence_list:
            return

        self._log(
            f"[KG] Processing {len(evidence_list)} evidence items into knowledge graph",
            "dim",
        )

        # Index evidence for hybrid retrieval (semantic + lexical search)
        self._index_for_retrieval(evidence_list, session_id)

        # Convert and add to knowledge graph
        kg_findings = self._convert_to_kg_findings(evidence_list)
        await self._add_to_graph(kg_findings)

    def _index_for_retrieval(
        self, evidence_list: list[Evidence], session_id: str
    ) -> None:
        """Index evidence for semantic + lexical search."""
        if self.findings_retriever is None:
            return
        try:
            self.findings_retriever.add_findings(
                findings=evidence_list,
                session_id=session_id,
            )
            self._log(
                f"[RETRIEVAL] Indexed {len(evidence_list)} evidence items for semantic search",
                "dim",
            )
        except Exception as e:
            self._log(f"[RETRIEVAL] Error indexing evidence: {e}", "yellow")
            logger.warning("Retrieval indexing error: %s", e, exc_info=True)

    def _convert_to_kg_findings(self, evidence_list: list[Evidence]) -> list[KGFinding]:
        """Convert Evidence models to KGFinding format for the knowledge graph."""
        kg_findings = []
        for evidence in evidence_list:
            try:
                kg_finding = KGFinding(
                    id=str(evidence.id or hash(evidence.content)),
                    content=evidence.content,
                    source_url=evidence.source_url or "",
                    source_title=(
                        evidence.source_url.split("/")[-1] if evidence.source_url else ""
                    ),
                    timestamp=evidence.created_at.isoformat(),
                    credibility_score=evidence.confidence,
                    finding_type=evidence.evidence_type.value,
                    search_query=evidence.search_query,
                )
                kg_findings.append(kg_finding)
            except Exception as e:
                self._log(f"[KG] Error converting evidence: {e}", "dim")
                logger.warning("KG conversion error: %s", e, exc_info=True)
        return kg_findings

    async def _add_to_graph(self, kg_findings: list[KGFinding]) -> None:
        """Add KG findings to the knowledge graph (batch or individual)."""
        if not kg_findings:
            return

        if len(kg_findings) > 3:
            result = await self.knowledge_graph.add_findings_batch(
                kg_findings, batch_size=5
            )
            self._log(
                f"[KG] Extracted {result['total_entities']} entities, "
                f"{result['total_relations']} relations",
                "dim",
            )
            if result["total_contradictions"] > 0:
                self._log(
                    f"[KG] Contradictions detected: {result['total_contradictions']}",
                    "yellow",
                )
        else:
            for kg_finding in kg_findings:
                try:
                    result = await self.knowledge_graph.add_finding(
                        kg_finding, fast_mode=True
                    )
                    if result.get("contradictions_found", 0) > 0:
                        self._log(
                            f"[KG] Contradiction detected: "
                            f"{result['contradictions_found']} conflicts",
                            "yellow",
                        )
                except Exception as e:
                    self._log(f"[KG] Error processing evidence: {e}", "dim")
                    logger.warning("KG processing error: %s", e, exc_info=True)
