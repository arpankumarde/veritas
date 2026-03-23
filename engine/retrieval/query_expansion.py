"""Advanced query expansion for improved search recall.

This module provides multi-query generation, contextual expansion using
knowledge graph gaps, and FAIR-RAG sufficiency evaluation.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from ..logging_config import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from ..knowledge.query import ManagerQueryInterface
    from .deduplication import EvidenceDeduplicator


@dataclass
class ExpandedQuery:
    """A single expanded query with metadata."""
    query: str
    strategy: str  # perspective, specificity, temporal, contextual
    reasoning: str
    expansion_type: str = "multi"


@dataclass
class QueryExpansionResult:
    """Result of query expansion."""
    original_query: str
    expanded_queries: list[ExpandedQuery] = field(default_factory=list)
    kg_gaps_used: list[str] = field(default_factory=list)
    is_sufficient: bool = False
    sufficiency_score: float = 0.0


@dataclass
class QueryExpansionConfig:
    """Configuration for query expansion."""
    num_variations: int = 3
    use_kg_gaps: bool = True
    use_previous_evidence: bool = True
    enable_sufficiency_check: bool = True
    sufficiency_threshold: float = 0.75
    min_evidence_before_check: int = 5


class QueryExpander:
    """Advanced query expansion using multiple strategies.

    Features:
    - Multi-query generation (perspective, specificity, temporal)
    - Contextual expansion using KG gaps
    - FAIR-RAG sufficiency evaluation
    """

    def __init__(
        self,
        config: QueryExpansionConfig | None = None,
        llm_callback: Callable | None = None,
        kg_query: Optional["ManagerQueryInterface"] = None,
        deduplicator: Optional["EvidenceDeduplicator"] = None,
        decision_logger_callback: Callable | None = None,
    ):
        """Initialize the QueryExpander.

        Args:
            config: Expansion configuration
            llm_callback: Async callback to call LLM (prompt) -> str
            kg_query: Knowledge graph query interface for gap detection
            deduplicator: Evidence deduplicator for checking redundancy
            decision_logger_callback: Async callback for logging decisions
        """
        self.config = config or QueryExpansionConfig()
        self.llm_callback = llm_callback
        self.kg_query = kg_query
        self.deduplicator = deduplicator
        self.decision_logger_callback = decision_logger_callback

    def set_kg_query(self, kg_query: "ManagerQueryInterface") -> None:
        """Set the knowledge graph query interface."""
        self.kg_query = kg_query

    async def expand(
        self,
        query: str,
        session_id: str,
        previous_evidence: list | None = None,
        search_iteration: int = 0,
        year: int | None = None,
    ) -> QueryExpansionResult:
        """Expand a query using multiple strategies.

        Args:
            query: Original search query/topic
            session_id: Current research session ID
            previous_evidence: List of evidence gathered so far
            search_iteration: Current search iteration number
            year: Year for temporal scoping (defaults to current year)
        """
        from datetime import datetime
        if year is None:
            year = datetime.now().year

        previous_evidence = previous_evidence or []

        result = QueryExpansionResult(original_query=query)

        # Check sufficiency first if we have enough evidence
        if (
            self.config.enable_sufficiency_check
            and len(previous_evidence) >= self.config.min_evidence_before_check
            and search_iteration > 0
        ):
            is_sufficient, score = await self._evaluate_sufficiency(
                query, previous_evidence, search_iteration, session_id
            )
            result.is_sufficient = is_sufficient
            result.sufficiency_score = score
            if is_sufficient:
                return result

        # Generate multi-queries
        expanded = await self._generate_multi_queries(
            query, previous_evidence, year, session_id
        )
        result.expanded_queries.extend(expanded)

        # Add contextual queries from KG gaps
        if self.config.use_kg_gaps and self.kg_query:
            contextual = await self._contextual_expand(query, session_id)
            result.expanded_queries.extend(contextual)
            result.kg_gaps_used = [eq.reasoning for eq in contextual]

        # Deduplicate queries
        result.expanded_queries = self._deduplicate_queries(result.expanded_queries)

        return result

    async def _generate_multi_queries(
        self,
        query: str,
        previous_evidence: list,
        year: int,
        session_id: str,
    ) -> list[ExpandedQuery]:
        """Generate diverse query variations using different strategies.

        Strategies:
        - perspective: Different angle (challenges vs benefits)
        - specificity: Add concrete terms, names, technologies
        - temporal: Focus on recent developments
        """
        if not self.llm_callback:
            # Fallback: simple temporal expansion
            return [ExpandedQuery(
                query=f"{query} {year} latest developments",
                strategy="temporal",
                reasoning="Added temporal scope for recency",
                expansion_type="multi",
            )]

        evidence_summary = self._summarize_evidence(previous_evidence, max_evidence=5)

        prompt = (
            f'Generate 3 diverse search queries for researching this topic.\n\n'
            f'Topic: "{query}"\n\n'
            f'Strategies to use:\n'
            f'1. PERSPECTIVE: Different angle (implications, challenges, applications, benefits, limitations)\n'
            f'2. SPECIFICITY: Add concrete terms, names, technologies, specific examples\n'
            f'3. TEMPORAL: Focus on recent developments, {year} updates, latest research\n\n'
            f'Previous evidence to avoid redundant searches:\n'
            f'{evidence_summary}\n\n'
            f'Return ONLY valid JSON in this exact format:\n'
            f'{{"queries": [\n'
            f'  {{"query": "search query text", "strategy": "perspective", "reasoning": "brief explanation"}},\n'
            f'  {{"query": "search query text", "strategy": "specificity", "reasoning": "brief explanation"}},\n'
            f'  {{"query": "search query text", "strategy": "temporal", "reasoning": "brief explanation"}}\n'
            f']}}'
        )

        multi_schema = {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "strategy": {"type": "string"},
                                "reasoning": {"type": "string"},
                            },
                            "required": ["query", "strategy"],
                        },
                    },
                },
                "required": ["queries"],
            },
        }

        try:
            try:
                response = await self.llm_callback(
                    prompt, output_format=multi_schema,
                )
            except TypeError:
                response = await self.llm_callback(prompt)

            data = self._parse_json_response(response)
            queries_data = data.get("queries", [])

            expanded = []
            for q in queries_data[:self.config.num_variations]:
                if q.get("query"):
                    expanded.append(ExpandedQuery(
                        query=q["query"],
                        strategy=q.get("strategy", "multi"),
                        reasoning=q.get("reasoning", ""),
                        expansion_type="multi",
                    ))

            if self.decision_logger_callback:
                strategies = [eq.strategy for eq in expanded]
                await self.decision_logger_callback(
                    session_id=session_id,
                    decision_type="multi_query_gen",
                    decision_outcome=f"multi_query_{len(expanded)}",
                    reasoning=f"Strategies: {', '.join(strategies)}",
                    inputs={"original_query": query},
                    metrics={"query_count": len(expanded)},
                )

            return expanded

        except Exception:
            # [HARDENED] ERR-003: Log before falling back
            logger.warning("Multi-query expansion failed, using fallback", exc_info=True)
            return [ExpandedQuery(
                query=f"{query} {year} latest research developments",
                strategy="temporal",
                reasoning="Fallback: temporal expansion",
                expansion_type="multi",
            )]

    async def _contextual_expand(
        self,
        query: str,
        session_id: str,
    ) -> list[ExpandedQuery]:
        """Generate queries addressing knowledge graph gaps.

        Uses ManagerQueryInterface.identify_gaps() to find:
        - Claims with insufficient evidence
        - Disconnected topic clusters
        - Bridging concepts needing exploration
        """
        if not self.kg_query or not self.llm_callback:
            return []

        try:
            gaps = await self.kg_query.identify_gaps()
            if not gaps:
                return []

            # Sort by importance
            gaps_sorted = sorted(gaps, key=lambda g: getattr(g, 'importance', 0), reverse=True)
            top_gaps = gaps_sorted[:3]

            gaps_text = "\n".join([
                f"- {getattr(g, 'description', str(g))} (importance: {getattr(g, 'importance', 'unknown')})"
                for g in top_gaps
            ])

            prompt = (
                f'Research query: "{query}"\n\n'
                f'Knowledge gaps identified in our current research:\n'
                f'{gaps_text}\n\n'
                f'Generate 1-2 search queries that directly address the most critical gap(s).\n'
                f'Focus on finding evidence, connections, or information to fill these gaps.\n\n'
                f'Return ONLY valid JSON:\n'
                f'{{"contextual_queries": [\n'
                f'  {{"query": "search query", "gap_addressed": "which gap this addresses"}}\n'
                f']}}'
            )

            ctx_schema = {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "contextual_queries": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "gap_addressed": {
                                        "type": "string",
                                    },
                                },
                                "required": ["query"],
                            },
                        },
                    },
                    "required": ["contextual_queries"],
                },
            }

            try:
                response = await self.llm_callback(
                    prompt, output_format=ctx_schema,
                )
            except TypeError:
                response = await self.llm_callback(prompt)

            data = self._parse_json_response(response)
            contextual_data = data.get("contextual_queries", [])

            expanded = []
            for q in contextual_data[:2]:
                if q.get("query"):
                    expanded.append(ExpandedQuery(
                        query=q["query"],
                        strategy="contextual",
                        reasoning=f"Gap addressed: {q.get('gap_addressed', '')}",
                        expansion_type="contextual",
                    ))

            if self.decision_logger_callback and expanded:
                await self.decision_logger_callback(
                    session_id=session_id,
                    decision_type="contextual_expand",
                    decision_outcome="_contextual_queries",
                    reasoning=f"Gaps used: {len(top_gaps)}",
                    inputs={"gap_count": len(top_gaps)},
                    metrics={"contextual_query_count": len(expanded)},
                )

            return expanded

        except Exception:
            # [HARDENED] ERR-003: Log before falling back
            logger.warning("Contextual expansion failed", exc_info=True)
            return []

    async def _evaluate_sufficiency(
        self,
        query: str,
        previous_evidence: list,
        search_count: int,
        session_id: str,
    ) -> tuple[bool, float]:
        """Evaluate if enough information has been gathered (FAIR-RAG).

        Checks:
        - Coverage: Do evidence items cover main aspects?
        - Depth: Are evidence items detailed enough?
        - Quality: Are sources credible?

        Returns:
            Tuple of (is_sufficient, sufficiency_score)
        """
        if not self.llm_callback:
            return False, 0.0

        evidence_summary = self._summarize_evidence(previous_evidence, max_evidence=10)

        prompt = (
            f'Research question: "{query}"\n\n'
            f'Evidence gathered ({len(previous_evidence)}):\n'
            f'{evidence_summary}\n\n'
            f'Searches performed: {search_count}\n\n'
            f'Evaluate if we have gathered SUFFICIENT information to answer the research question.\n\n'
            f'Consider:\n'
            f'1. COVERAGE: Does the evidence cover the main aspects of the topic?\n'
            f'2. DEPTH: Is the evidence detailed enough to be useful?\n'
            f'3. QUALITY: Are the sources diverse and credible?\n\n'
            f'Return ONLY valid JSON:\n'
            f'{{\n'
            f'  "is_sufficient": true,\n'
            f'  "sufficiency_score": 0.0,\n'
            f'  "coverage_assessment": "brief assessment",\n'
            f'  "critical_gaps": ["gap1", "gap2"]\n'
            f'}}'
        )

        suff_schema = {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "is_sufficient": {"type": "boolean"},
                    "sufficiency_score": {"type": "number"},
                    "coverage_assessment": {"type": "string"},
                    "critical_gaps": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "is_sufficient", "sufficiency_score",
                ],
            },
        }

        try:
            try:
                response = await self.llm_callback(
                    prompt, output_format=suff_schema,
                )
            except TypeError:
                response = await self.llm_callback(prompt)

            data = self._parse_json_response(response)

            is_sufficient = bool(data.get("is_sufficient", False))
            score = float(data.get("sufficiency_score", 0.0))

            if self.decision_logger_callback:
                await self.decision_logger_callback(
                    session_id=session_id,
                    decision_type="sufficiency_check",
                    decision_outcome="sufficient" if is_sufficient else "insufficient",
                    reasoning=data.get("coverage_assessment", ""),
                    inputs={"search_count": search_count, "evidence_count": len(previous_evidence)},
                    metrics={
                        "sufficiency_score": score,
                        "critical_gaps": len(data.get("critical_gaps", [])),
                    },
                )

            return is_sufficient and score >= self.config.sufficiency_threshold, score

        except Exception:
            # [HARDENED] ERR-003: Log before falling back
            logger.warning("Sufficiency evaluation failed", exc_info=True)
            return False, 0.0

    def _summarize_evidence(self, evidence: list, max_evidence: int = 5) -> str:
        """Create a brief summary of evidence for prompts."""
        if not evidence:
            return "None yet."
        recent = evidence[-max_evidence:]
        lines = []
        for e in recent:
            content = getattr(e, 'content', str(e))
            etype = getattr(e, 'evidence_type', None)
            type_str = f"[{etype.value.upper()}] " if etype else ""
            lines.append(f"- {type_str}{content[:100]}")
        return "\n".join(lines)

    def _parse_json_response(self, response: str | dict) -> dict:
        """Parse JSON from LLM response, handling structured output."""
        if isinstance(response, dict):
            return response
        start = response.find("{")
        end = response.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(response[start:end])
            except json.JSONDecodeError:
                pass
        return {}

    def _deduplicate_queries(self, queries: list[ExpandedQuery]) -> list[ExpandedQuery]:
        """Remove duplicate or very similar queries."""
        seen = set()
        unique = []
        for q in queries:
            normalized = q.query.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                unique.append(q)
        return unique


def merge_search_results(
    queries: list[str],
    results_list: list,
    k: int = 60,
    max_results: int = 15,
) -> tuple[list, str]:
    """Merge results from multiple queries using Reciprocal Rank Fusion.

    Args:
        queries: List of search queries executed
        results_list: List of search results (one per query)
        k: RRF constant (default 60)
        max_results: Maximum results to return

    Returns:
        Tuple of (merged results list, summary string)
    """
    url_to_result = {}
    url_scores: dict[str, float] = {}

    for rank_list in results_list:
        if isinstance(rank_list, Exception):
            continue
        # Each element is a tuple (results, summary) from search_tool.search()
        if isinstance(rank_list, tuple) and len(rank_list) >= 1:
            results = rank_list[0]
        else:
            results = rank_list

        if not isinstance(results, list):
            continue

        for rank, result in enumerate(results):
            url = getattr(result, 'url', '') or str(rank)
            if url not in url_to_result:
                url_to_result[url] = result
            rrf_score = 1.0 / (k + rank + 1)
            url_scores[url] = url_scores.get(url, 0.0) + rrf_score

    # Sort by RRF score
    sorted_urls = sorted(url_scores.items(), key=lambda x: -x[1])
    merged = [url_to_result[url] for url, _ in sorted_urls[:max_results]]

    multi_hits = sum(1 for _, score in sorted_urls[:max_results] if score > 1.0 / (k + 1))
    summary = f"RRF merge: {len(queries)} queries -> {len(merged)} results ({multi_hits} multi-query hits)"

    return merged, summary
