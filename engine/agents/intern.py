import json
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from ..events import emit_action, emit_finding
from .base import AgentConfig, BaseAgent, DecisionType


def _get_current_year() -> int:
    """Get the current year for search queries."""
    return datetime.now().year


import asyncio

from rich.console import Console

from ..models.findings import (
    AgentRole,
    Evidence,
    EvidenceType,
    EvidenceReport,
    VerificationDirective,
    is_meta_question,
)
from ..retrieval.deduplication import get_deduplicator
from ..retrieval.query_expansion import (
    QueryExpander,
    QueryExpansionConfig,
    merge_search_results,
)
from ..storage.database import VeritasDatabase
from ..tools.web_search import SearchResult, WebSearchTool

from ..logging_config import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from ..verification import VerificationPipeline


def _is_academic_topic(topic: str) -> bool:
    """Detect if a topic would benefit from academic search."""
    academic_indicators = [
        "research",
        "study",
        "paper",
        "journal",
        "scientific",
        "evidence",
        "theory",
        "hypothesis",
        "experiment",
        "analysis",
        "review",
        "survey",
        "methodology",
        "framework",
        "algorithm",
        "clinical",
        "trial",
        "treatment",
        "disease",
        "medical",
        "mechanism",
        "model",
        "simulation",
        "data",
        "statistical",
        "peer-reviewed",
        "published",
        "findings",
        "literature",
        "thesis",
        "dissertation",
        "academic",
        "scholar",
        "university",
    ]
    topic_lower = topic.lower()
    return any(indicator in topic_lower for indicator in academic_indicators)


class InternAgent(BaseAgent):
    """The Intern agent searches the web and reports evidence to the Manager.

    Responsibilities:
    - Execute web searches based on Manager directives
    - Find evidence BOTH supporting AND contradicting claims
    - Extract relevant information from search results
    - Identify primary sources, official records, and expert opinions
    - Suggest follow-up angles for deeper investigation
    - Report evidence back to the Manager
    """

    def __init__(
        self,
        db: VeritasDatabase,
        config: AgentConfig | None = None,
        console: Console | None = None,
        verification_pipeline: Optional["VerificationPipeline"] = None,
        query_expansion_config: QueryExpansionConfig | None = None,
        agent_id: str | None = None,
    ):
        super().__init__(AgentRole.INTERN, db, config, console, agent_id=agent_id)
        self.search_tool = WebSearchTool(max_results=10)
        self.current_directive: VerificationDirective | None = None
        self.evidence: list[Evidence] = []
        self.searches_performed: int = 0
        self.suggested_followups: list[str] = []
        self.verification_pipeline = verification_pipeline
        self.deduplicator = get_deduplicator()

        # Initialize academic search (free APIs, no keys required)
        try:
            from ..tools.academic_search import AcademicSearchTool

            self.academic_search = AcademicSearchTool(max_results=10)
        except Exception as e:
            self.academic_search = None
            logger.warning("Academic search initialization failed: %s", e, exc_info=True)
            if console:
                console.print(f"[dim]Academic search unavailable: {e}[/dim]")

        # Initialize query expander
        self.query_expander = QueryExpander(
            config=query_expansion_config or QueryExpansionConfig(),
            llm_callback=self._query_expansion_callback,
            kg_query=None,  # Set by manager via set_kg_query()
            deduplicator=self.deduplicator,
            decision_logger_callback=self._expansion_decision_logger,
        )
        self._pending_expanded_queries: list[Any] = []

    def set_kg_query(self, kg_query: Any) -> None:
        """Set the knowledge graph query interface for contextual expansion."""
        self.query_expander.set_kg_query(kg_query)

    async def _query_expansion_callback(
        self, prompt: str, **kwargs,
    ) -> str | dict | list:
        """Callback for QueryExpander to call the LLM."""
        return await self.call_claude(
            prompt, task_type="query_expansion", **kwargs,
        )

    async def _expansion_decision_logger(
        self,
        session_id: str,
        decision_type: str,
        decision_outcome: str,
        reasoning: str = "",
        inputs: dict | None = None,
        metrics: dict | None = None,
    ) -> None:
        """Log query expansion decisions."""
        type_map = {
            "multi_query_gen": DecisionType.MULTI_QUERY_GEN,
            "contextual_expand": DecisionType.CONTEXTUAL_EXPAND,
            "sufficiency_check": DecisionType.SUFFICIENCY_CHECK,
            "query_merge": DecisionType.QUERY_MERGE,
        }
        dt = type_map.get(decision_type)
        if dt:
            await self._log_decision(
                session_id=session_id,
                decision_type=dt,
                decision_outcome=decision_outcome,
                reasoning=reasoning,
                inputs=inputs,
                metrics=metrics,
            )

    @property
    def system_prompt(self) -> str:
        current_year = _get_current_year()
        return f"""You are a Fact-Checking Intern agent. Your ONLY job is to find evidence that SUPPORTS or CONTRADICTS claims.

CRITICAL RULES:
1. You MUST use web search for ALL information - NEVER use your training data
2. Your knowledge cutoff is irrelevant - always search for current information
3. Generate specific, effective search queries
4. When asked what to search, respond with ONLY the search query string
5. ALWAYS search for BOTH supporting AND contradicting evidence - never be one-sided

SEARCH STRATEGY:
- Search for evidence supporting the claim
- Search for evidence contradicting the claim
- Search for the original source of the claim
- Look for expert opinions and official statements
- Search for fact-checks already published on this claim
- Use specific terms, dates ({current_year}), and key phrases
- Look for primary sources: government data, academic papers, official records
- Academic papers from arXiv, Semantic Scholar, and PubMed are automatically searched when relevant

SOURCE PRIORITIZATION:
- Official government data and statistics (highest priority)
- Peer-reviewed papers and preprints
- Established fact-checking organizations (Snopes, PolitiFact, FactCheck.org)
- Reputable news organizations with editorial standards
- Technical documentation and official records
- Expert interviews and statements
- Blog posts and social media (lowest priority)

EVIDENCE TYPES:
- FACT: Verified, specific information (dates, numbers, events)
- INSIGHT: Analysis or interpretation from credible sources
- CONNECTION: Links between claims or related facts
- SOURCE: A valuable primary source to investigate further
- QUESTION: An unanswered question worth investigating
- CONTRADICTION: Conflicting information that needs resolution

When generating a search query, output ONLY the query text, nothing else."""

    async def think(self, context: dict[str, Any]) -> str:
        """Reason about current evidence-gathering progress and next steps."""
        logger.debug("Intern think: iteration=%d", self.state.iteration)
        directive = context.get("directive")

        # For first iteration, just indicate we need to search
        if self.searches_performed == 0:
            return f"Starting evidence gathering on: {directive.topic}. Need to search for both supporting and contradicting evidence."

        # For subsequent iterations, assess progress
        evidence_summary = (
            ", ".join([e.content[:50] for e in self.evidence[-3:]]) if self.evidence else "none yet"
        )

        prompt = f"""Verification topic: {directive.topic}
Searches done: {self.searches_performed}/{directive.max_searches}
Evidence so far: {len(self.evidence)}
Recent evidence: {evidence_summary}

Should I continue searching or compile report? If continue, what aspect should I search next?
Remember: search for BOTH supporting AND contradicting evidence.
Be brief - just state your decision and reason."""

        return await self.call_claude(prompt)

    async def act(self, thought: str, context: dict[str, Any]) -> dict[str, Any]:
        """Execute a search or compile a report based on thinking."""
        logger.debug("Intern act: thought=%s", thought[:200])
        directive: VerificationDirective = context.get("directive")
        session_id = context.get("session_id", "")

        # Check if we should stop
        if self._should_stop_searching(thought, directive):
            # Log stop searching decision
            await self._log_decision(
                session_id=session_id,
                decision_type=DecisionType.STOP_SEARCHING,
                decision_outcome="stop",
                reasoning=thought[:500],
                inputs={"topic": directive.topic, "max_searches": directive.max_searches},
                metrics={
                    "searches_done": self.searches_performed,
                    "evidence_count": len(self.evidence),
                },
            )
            return {
                "action": "compile_report",
                "report": await self._compile_report(directive.topic, session_id),
            }

        # Use query expansion for better coverage
        results, search_summary, queries_used = await self._search_with_expansion(
            directive.topic, session_id
        )

        # Check for early stop due to sufficiency
        if not queries_used:
            self._log("[Sufficiency] Gathered enough evidence", style="bold magenta")
            return {
                "action": "compile_report",
                "report": await self._compile_report(directive.topic, session_id),
            }

        self.searches_performed += 1

        # Show search summary
        if search_summary:
            self._log("─" * 60, style="dim")
            self._log("[Search Summary]", style="bold cyan")
            summary_preview = (
                search_summary[:1500] + "..." if len(search_summary) > 1500 else search_summary
            )
            self.console.print(summary_preview)
            self._log("─" * 60, style="dim")

        # Show search results
        if results:
            self._log(f"[Search Results: {len(results)} found]", style="bold yellow")
            for i, r in enumerate(results[:5], 1):
                self._log(f"  {i}. {r.title}", style="yellow")
                if r.url:
                    self._log(f"     URL: {r.url}", style="dim")
                if r.snippet:
                    snippet = r.snippet[:200] + "..." if len(r.snippet) > 200 else r.snippet
                    self._log(f"     {snippet}", style="dim")

        # Process results and extract evidence (use primary query for logging)
        primary_query = queries_used[0] if queries_used else directive.topic
        new_evidence = await self._process_search_results(
            results, primary_query, session_id, search_summary
        )

        # Show extracted evidence
        if new_evidence:
            self._log(f"[Extracted {len(new_evidence)} Evidence Items]", style="bold green")
            for e in new_evidence:
                self._log(f"  [{e.evidence_type.value.upper()}] {e.content[:150]}...", style="green")
                if e.source_url:
                    self._log(f"    Source: {e.source_url}", style="dim")
                self._log(f"    Confidence: {e.confidence:.0%}", style="dim")

        return {
            "action": "search",
            "query": primary_query,
            "queries_used": queries_used,
            "results_count": len(results),
            "evidence_extracted": len(new_evidence),
            "results": results,
            "summary": search_summary,
        }

    async def observe(self, action_result: dict[str, Any]) -> str:
        """Process the result of a search action."""
        action = action_result.get("action")

        if action == "compile_report":
            report: EvidenceReport = action_result.get("report")
            return f"Report compiled: {len(report.evidence)} evidence items, {len(report.suggested_followups)} follow-up suggestions"

        if action == "search":
            query = action_result.get("query")
            results_count = action_result.get("results_count", 0)
            evidence_count = action_result.get("evidence_extracted", 0)

            if results_count == 0:
                return f"Search for '{query}' returned no results. Consider rephrasing or trying a different angle."

            return f"Search for '{query}' returned {results_count} results, extracted {evidence_count} evidence items. Total evidence now: {len(self.evidence)}"

        return "Unknown action result"

    def is_done(self, context: dict[str, Any]) -> bool:
        """Check if the intern has completed the current directive."""
        directive: VerificationDirective = context.get("directive")
        if not directive:
            return True

        # Stop if we hit max searches
        if self.searches_performed >= directive.max_searches:
            return True

        # Stop if action was to compile report
        last_action = context.get("last_action", {})
        if last_action.get("action") == "compile_report":
            return True

        # Stop if directive says to stop
        if directive.action == "stop":
            return True

        return False

    def _should_stop_searching(self, thought: str, directive: VerificationDirective) -> bool:
        """Determine if we should stop searching based on the thought."""
        thought_lower = thought.lower()
        stop_indicators = [
            "should stop",
            "enough information",
            "compile report",
            "ready to report",
            "sufficient evidence",
            "covered the topic",
        ]
        return any(indicator in thought_lower for indicator in stop_indicators)

    async def _search_with_expansion(
        self, topic: str, session_id: str
    ) -> tuple[list[SearchResult], str, list[str]]:
        """Search for evidence — uses direct search for speed.

        Skips the multi-query expansion pipeline (saves 1-2 LLM calls per search)
        and does a single direct web search instead.

        Returns:
            Tuple of (merged_results, combined_summary, queries_used)
        """
        # FAST PATH: Direct search without LLM-based query expansion.
        # The topic from the manager is already well-formed.
        query = topic.strip()
        if len(query) > 200:
            query = query[:200]

        self._log(f"[Search] {query[:80]}...", style="cyan")

        results, summary = await self.search_tool.search(query)

        # Also search academic if relevant (parallel)
        if self.academic_search and _is_academic_topic(topic):
            try:
                academic_results, _ = await self.academic_search.search(topic)
                if academic_results:
                    existing_urls = {r.url for r in results}
                    new_academic = [r for r in academic_results if r.url not in existing_urls]
                    results = new_academic[:3] + results
            except Exception:
                pass

        return results, summary, [query] if results else []

    async def _search_with_expansion_DISABLED(
        self, topic: str, session_id: str
    ) -> tuple[list[SearchResult], str, list[str]]:
        """ORIGINAL: Search using expanded queries for better coverage.
        Disabled for speed — kept for reference.
        """
        expansion = await self.query_expander.expand(
            query=topic,
            session_id=session_id,
            previous_evidence=self.evidence,
            search_iteration=self.searches_performed,
        )

        # Early stop if sufficient information gathered
        if expansion.is_sufficient:
            self._log(
                f"[Query Expansion] Sufficiency reached: {expansion.sufficiency_score:.0%}",
                style="magenta",
            )
            return [], "Sufficient evidence gathered", []

        queries = [eq.query for eq in expansion.expanded_queries]
        if not queries:
            queries = [topic]

        self._log(f"[Query Expansion] Executing {len(queries)} queries", style="cyan")
        for i, q in enumerate(queries, 1):
            self._log(f"  {i}. {q[:80]}...", style="dim") if len(q) > 80 else self._log(
                f"  {i}. {q}", style="dim"
            )

        # Execute searches in parallel
        for q in queries:
            logger.debug("Search: query=%s", q)
        tasks = [self.search_tool.search(q) for q in queries]

        # Also search academic databases if the topic seems academic
        academic_results = []
        if self.academic_search and _is_academic_topic(topic):
            self._log(
                "[Academic Search] Querying Semantic Scholar + arXiv",
                style="cyan",
            )
            tasks.append(self.academic_search.search(topic))

        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # Separate academic results if they were included
        if self.academic_search and _is_academic_topic(topic):
            academic_result = results_list[-1]
            results_list = list(results_list[:-1])
            if isinstance(academic_result, tuple) and len(academic_result) >= 1:
                ar = academic_result[0]
                academic_results = ar if isinstance(ar, list) else []
                if academic_results:
                    self._log(
                        f"[Academic Search] Found {len(academic_results)} papers",
                        style="cyan",
                    )
        else:
            results_list = list(results_list)

        # Merge results using RRF
        merged_results, merge_summary = merge_search_results(
            queries=queries,
            results_list=results_list,
            k=60,
            max_results=15,
        )

        # Append academic results (prioritized at top since they're high quality)
        if academic_results:
            # Add academic results that aren't already in merged results
            existing_urls = {r.url for r in merged_results}
            new_academic = [r for r in academic_results if r.url not in existing_urls]
            merged_results = new_academic[:5] + merged_results  # Top 5 academic papers first
            queries.append(f"[academic] {topic}")  # Track that academic search was used

        # Log query merge decision
        await self._log_decision(
            session_id=session_id,
            decision_type=DecisionType.QUERY_MERGE,
            decision_outcome=f"merged_{len(queries)}_queries",
            reasoning=merge_summary,
            inputs={"queries": queries[:3]},
            metrics={
                "query_count": len(queries),
                "total_results": len(merged_results),
                "kg_gaps_used": len(expansion.kg_gaps_used),
            },
        )

        # Combine summaries from all searches
        summaries = []
        for r in results_list:
            if isinstance(r, tuple) and len(r) > 1 and r[1]:
                summaries.append(r[1])

        combined_summary = "\n\n---\n\n".join(summaries[:2]) if summaries else ""

        return merged_results, combined_summary, queries

    async def _extract_search_query(
        self, thought: str, directive: VerificationDirective, session_id: str = ""
    ) -> str | None:
        """Extract a search query from the agent's thought."""
        # Check if thought indicates we should stop
        if self._should_stop_searching(thought, directive):
            return None

        # Generate a search query based on the directive and progress
        current_year = _get_current_year()
        if self.searches_performed == 0:
            # First search - use expanded query for broader coverage
            return await self._expand_query(directive.topic, current_year, session_id)

        # Subsequent searches - use diverse query expansion
        prompt = f"""Verification topic: {directive.topic}
Searches done: {self.searches_performed}
Previous evidence: {len(self.evidence)}

Recent evidence summary:
{self._get_evidence_summary()}

Generate ONE specific search query to find NEW evidence not covered by existing items.
Focus on:
- Evidence that CONTRADICTS the claim (if mostly supporting so far)
- Evidence that SUPPORTS the claim (if mostly contradicting so far)
- Different angles or perspectives
- Recent developments ({current_year})
- Primary sources (official records, research papers, government data)

Output ONLY the search query, nothing else."""

        response = await self.call_claude(prompt)
        query = response.strip().strip('"').strip("'")

        # Clean up the query - remove any preamble
        if ":" in query and len(query.split(":")[0]) < 20:
            query = query.split(":", 1)[1].strip()

        # Don't search for error messages or meta-text
        if "error" in query.lower() or len(query) > 200:
            return f"{directive.topic} recent evidence {current_year}"

        return query if query and query.upper() != "STOP" else None

    async def _expand_query(self, topic: str, year: int, session_id: str = "") -> str:
        """Expand a query to improve search coverage.

        Uses query expansion techniques:
        - Synonym expansion
        - Temporal scoping
        - Specificity adjustment
        """
        prompt = f"""Expand this fact-checking topic into an effective web search query.

Topic: {topic}

Create a search query that:
1. Includes specific keywords and synonyms
2. Targets recent information ({year})
3. Avoids overly generic terms
4. Is optimized for finding authoritative, primary sources
5. Looks for both supporting and contradicting evidence

Output ONLY the search query (15-25 words max), nothing else."""

        response = await self.call_claude(prompt, task_type="query_expansion")
        query = response.strip().strip('"').strip("'")

        # Determine expansion strategy
        used_fallback = False
        if not query or len(query) > 200 or "error" in query.lower():
            query = f"{topic} {year} fact check evidence"
            used_fallback = True

        # Log query expansion decision
        if session_id:
            await self._log_decision(
                session_id=session_id,
                decision_type=DecisionType.QUERY_EXPAND,
                decision_outcome="expanded" if not used_fallback else "fallback",
                reasoning=f"Expanded '{topic}' -> '{query[:100]}'",
                inputs={"original_topic": topic, "year": year},
                metrics={
                    "search_number": self.searches_performed + 1,
                    "used_fallback": used_fallback,
                },
            )

        return query

    def _get_evidence_summary(self) -> str:
        """Get a brief summary of recent evidence to avoid duplicate searches."""
        if not self.evidence:
            return "None yet"

        recent = self.evidence[-5:]  # Last 5 evidence items
        summaries = []
        for e in recent:
            content = e.content[:80] + "..." if len(e.content) > 80 else e.content
            summaries.append(f"- {content}")
        return "\n".join(summaries)

    async def _process_search_results(
        self,
        results: list[SearchResult],
        query: str,
        session_id: str,
        search_summary: str = "",
    ) -> list[Evidence]:
        """Process search results, deep-scrape top sources, and extract evidence."""
        if not results and not search_summary:
            return []

        # Deep-scrape top results in parallel for richer content
        scraped_content = await self._deep_scrape_results(results[:5], session_id)

        # Format results for Claude to analyze
        results_parts = []
        for r in results[:10]:
            parts = [f"Title: {r.title}", f"URL: {r.url}"]
            if r.snippet:
                parts.append(f"Snippet: {r.snippet}")
            if hasattr(r, "engine") and r.engine and r.engine != "google":
                parts.append(f"Engine: {r.engine}")
            # Attach scraped page content (truncated)
            page_md = scraped_content.get(r.url, {}).get("markdown", "")
            if page_md:
                parts.append(f"Page Content:\n{page_md[:3000]}")
            # Attach structured platform data
            platform_text = scraped_content.get(r.url, {}).get("platform_text", "")
            if platform_text:
                parts.append(f"Platform Data:\n{platform_text}")
            results_parts.append("\n".join(parts))
        results_text = "\n\n---\n\n".join(results_parts)

        # Include the search summary if available
        summary_section = ""
        if search_summary:
            summary_section = f"\n\nSearch Summary:\n{search_summary}\n"

        prompt = (
            f'Analyze these search results for the query: "{query}"\n\n'
            f"{results_text}\n{summary_section}\n"
            "Extract key evidence items. For each piece of evidence, provide:\n"
            "1. The evidence content (1-2 sentences)\n"
            "2. Type: FACT, INSIGHT, CONNECTION, SOURCE, QUESTION, "
            "or CONTRADICTION\n"
            "3. Source URL\n"
            "4. Confidence score (0.0-1.0)\n"
            "5. If the source contains quantitative results (statistics, "
            "percentages, measurements, etc.), include the specific numbers "
            "in the evidence content.\n"
            "6. For academic papers, note the dataset name and baseline "
            "comparison if mentioned.\n\n"
            "For evidence with quantitative results, also provide a "
            "'metrics' object with dataset, metric_name, metric_value, "
            "and baseline fields (all optional strings).\n\n"
            "Also suggest 2-3 follow-up search queries that could "
            "find additional supporting or contradicting evidence."
        )

        schema = {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "findings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string"},
                                "type": {
                                    "type": "string",
                                    "enum": [
                                        "FACT", "INSIGHT", "CONNECTION",
                                        "SOURCE", "QUESTION",
                                        "CONTRADICTION",
                                    ],
                                },
                                "url": {"type": "string"},
                                "confidence": {"type": "number"},
                                "metrics": {
                                    "type": "object",
                                    "properties": {
                                        "dataset": {"type": "string"},
                                        "metric_name": {"type": "string"},
                                        "metric_value": {"type": "string"},
                                        "baseline": {"type": "string"},
                                    },
                                },
                            },
                            "required": [
                                "content", "type", "url", "confidence",
                            ],
                        },
                    },
                    "followups": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["findings", "followups"],
            },
        }

        response = await self.call_claude(
            prompt, output_format=schema,
        )

        evidence_items = []

        # Try to parse response (structured or text)
        try:
            if isinstance(response, dict):
                data = response
            else:
                start = response.find("{")
                end = response.rfind("}") + 1
                if start == -1 or end <= start:
                    data = {"findings": [], "followups": []}
                else:
                    data = json.loads(response[start:end])

            for f in data.get("findings", []):
                content = f.get("content", "")

                # Check for duplicates before processing
                if self.deduplicator.enabled:
                    dedup_result = self.deduplicator.check(content)
                    if dedup_result.is_duplicate:
                        self._log(
                            f"[DEDUP] Skipping duplicate ({dedup_result.match_type}, "
                            f"sim={dedup_result.similarity:.0%})",
                            style="dim",
                        )
                        await self._log_decision(
                            session_id=session_id,
                            decision_type=DecisionType.DEDUP_SKIP,
                            decision_outcome="skipped",
                            reasoning=f"Content matched existing evidence: {content[:100]}",
                            inputs={"match_type": dedup_result.match_type},
                            metrics={"similarity": dedup_result.similarity},
                        )
                        continue

                source_url = f.get("url")

                # Append quantitative metrics to evidence content if present
                metrics = f.get("metrics")
                if metrics and isinstance(metrics, dict):
                    parts = []
                    if metrics.get("dataset"):
                        parts.append(f"Dataset: {metrics['dataset']}")
                    if metrics.get("metric_name") and metrics.get("metric_value"):
                        parts.append(
                            f"{metrics['metric_name']}: {metrics['metric_value']}"
                        )
                    if metrics.get("baseline"):
                        parts.append(f"Baseline: {metrics['baseline']}")
                    if parts:
                        content = f"{content} [{'; '.join(parts)}]"

                evidence_item = Evidence(
                    session_id=session_id,
                    content=content,
                    evidence_type=EvidenceType(f.get("type", "fact").lower()),
                    source_url=source_url,
                    confidence=f.get("confidence", 0.7),
                    search_query=query,
                )

                # Run streaming verification if pipeline is available
                verification_result = None
                if self.verification_pipeline:
                    try:
                        source_snippet = None
                        if source_url:
                            for r in results:
                                if r.url == source_url and r.snippet:
                                    source_snippet = r.snippet
                                    break

                        verification_result = (
                            await self.verification_pipeline.verify_intern_finding(
                                evidence_item, session_id, source_content=source_snippet
                            )
                        )
                        evidence_item.original_confidence = evidence_item.confidence
                        evidence_item.confidence = verification_result.verified_confidence
                        evidence_item.verification_status = (
                            verification_result.verification_status.value
                        )
                        evidence_item.verification_method = (
                            verification_result.verification_method.value
                        )
                        evidence_item.kg_support_score = verification_result.kg_support_score

                    except Exception as e:
                        logger.warning("Verification error: %s", e, exc_info=True)
                        self._log(f"[VERIFY] Error: {e}", style="yellow")
                        evidence_item.verification_status = "error"

                await self.db.save_finding(evidence_item)

                if verification_result and evidence_item.id:
                    try:
                        await self.db.save_verification_result(
                            session_id=session_id,
                            finding_id=evidence_item.id,
                            result_dict=verification_result.to_dict(),
                        )
                    except Exception:
                        logger.warning("Failed to save verification result for evidence %s", evidence_item.id, exc_info=True)
                        pass

                evidence_items.append(evidence_item)
                self.evidence.append(evidence_item)

                if self.session_id:
                    await emit_finding(
                        session_id=self.session_id,
                        agent=self.role.value,
                        content=content[:300],
                        source=source_url,
                        confidence=evidence_item.confidence,
                    )

                if self.deduplicator.enabled:
                    evidence_id = (
                        str(evidence_item.id) if evidence_item.id else f"{session_id}_{len(self.evidence)}"
                    )
                    self.deduplicator.add(evidence_id, content)

            for followup in data.get("followups", []):
                if not is_meta_question(followup) and followup not in self.suggested_followups:
                    self.suggested_followups.append(followup)

        except (json.JSONDecodeError, KeyError, ValueError):
            pass

        # Fallback: if no JSON evidence but we have search results, create evidence from them
        if not evidence_items and results:
            for r in results[:5]:
                if r.snippet:
                    content = r.snippet[:500]

                    # Check for duplicates
                    if self.deduplicator.enabled:
                        dedup_result = self.deduplicator.check(content)
                        if dedup_result.is_duplicate:
                            # Log dedup skip decision (fallback path)
                            await self._log_decision(
                                session_id=session_id,
                                decision_type=DecisionType.DEDUP_SKIP,
                                decision_outcome="skipped_fallback",
                                reasoning=f"Fallback content matched existing: {content[:100]}",
                                inputs={"match_type": dedup_result.match_type},
                                metrics={"similarity": dedup_result.similarity},
                            )
                            continue

                    evidence_item = Evidence(
                        session_id=session_id,
                        content=content,
                        evidence_type=EvidenceType.FACT,
                        source_url=r.url,
                        confidence=0.6,
                        search_query=query,
                    )

                    # Run streaming verification if pipeline is available
                    verification_result = None
                    if self.verification_pipeline:
                        try:
                            verification_result = (
                                await self.verification_pipeline.verify_intern_finding(
                                    evidence_item, session_id, source_content=r.snippet
                                )
                            )
                            evidence_item.original_confidence = evidence_item.confidence
                            evidence_item.confidence = verification_result.verified_confidence
                            evidence_item.verification_status = (
                                verification_result.verification_status.value
                            )
                            evidence_item.verification_method = (
                                verification_result.verification_method.value
                            )
                            evidence_item.kg_support_score = verification_result.kg_support_score
                        except Exception:
                            logger.warning(
                                "Verification error during fallback evidence extraction",
                                exc_info=True,
                            )
                            evidence_item.verification_status = "error"

                    await self.db.save_finding(evidence_item)

                    # Save detailed verification result AFTER save_finding (which assigns evidence_item.id)
                    if verification_result and evidence_item.id:
                        try:
                            await self.db.save_verification_result(
                                session_id=session_id,
                                finding_id=evidence_item.id,
                                result_dict=verification_result.to_dict(),
                            )
                        except Exception:
                            logger.warning("Failed to save verification result for fallback evidence %s", evidence_item.id, exc_info=True)
                            pass

                    evidence_items.append(evidence_item)
                    self.evidence.append(evidence_item)

                    # Add to deduplication index
                    if self.deduplicator.enabled:
                        evidence_id = (
                            str(evidence_item.id) if evidence_item.id else f"{session_id}_{len(self.evidence)}"
                        )
                        self.deduplicator.add(evidence_id, content)

        return evidence_items

    async def _deep_scrape_results(
        self, results: list[SearchResult], session_id: str
    ) -> dict[str, dict]:
        """Deep-scrape top search result URLs in parallel.

        For each URL, fetches:
        - Full page content as markdown
        - Structured platform data (if URL matches a known platform like X, Reddit, etc.)

        Returns dict mapping URL -> {"markdown": str, "platform_text": str}
        """
        if not results:
            return {}

        from ..tools.bright_data import detect_platform, format_platform_data

        urls = [r.url for r in results if r.url]
        if not urls:
            return {}

        # Emit scraping event so UI shows progress
        platform_urls = [u for u in urls if detect_platform(u)]
        if session_id and self.session_id:
            scrape_msg = f"Deep scraping {len(urls)} sources"
            if platform_urls:
                scrape_msg += f" ({len(platform_urls)} platform URLs detected)"
            await emit_action(
                session_id=self.session_id,
                agent=self.role.value,
                action="deep_scrape",
                details={
                    "urls_count": len(urls),
                    "platform_urls": len(platform_urls),
                    "platforms": list({detect_platform(u) for u in platform_urls}),
                },
            )

        self._log(f"[Deep Scrape] Scraping {len(urls)} sources...", style="cyan")

        try:
            scraped = await self.search_tool.deep_scrape_batch(urls)
        except Exception as e:
            logger.warning("Deep scrape batch failed: %s", e)
            return {}

        out: dict[str, dict] = {}
        for item in scraped:
            url = item.get("url", "")
            md = item.get("markdown", "")
            platform_data = item.get("platform_data")
            platform_name = item.get("platform")

            platform_text = ""
            if platform_data and platform_name:
                try:
                    from ..tools.bright_data import PlatformData as PD
                    pd = PD(platform=platform_name, url=url, data=platform_data)
                    platform_text = format_platform_data(pd)
                except Exception:
                    pass

            if md or platform_text:
                out[url] = {"markdown": md, "platform_text": platform_text}

                if platform_text:
                    self._log(f"  [{platform_name}] {url[:60]}", style="magenta")
                elif md:
                    self._log(f"  [scraped] {url[:60]} ({len(md)} chars)", style="dim")

        self._log(f"[Deep Scrape] Got content for {len(out)}/{len(urls)} URLs", style="cyan")
        return out

    async def _compile_report(self, topic: str, session_id: str) -> EvidenceReport:
        """Compile evidence into a report for the Manager."""
        return EvidenceReport(
            topic=topic,
            evidence=self.evidence.copy(),
            searches_performed=self.searches_performed,
            suggested_followups=self.suggested_followups.copy(),
            blockers=[],
        )

    def reset(self) -> None:
        """Reset state for a new directive."""
        self.current_directive = None
        self.evidence = []
        self.searches_performed = 0
        self.suggested_followups = []
        self.search_tool.reset_count()
        self.state = type(self.state)()
        self._pending_expanded_queries = []

    async def execute_directive(self, directive: VerificationDirective, session_id: str) -> EvidenceReport:
        """Execute a directive from the Manager and return a report."""
        logger.info("Executing directive: topic=%s, max_searches=%d", directive.topic, directive.max_searches)
        self.reset()
        self.current_directive = directive
        # Ensure WebSocket event emission uses the correct session ID
        self.session_id = session_id

        context = {
            "directive": directive,
            "session_id": session_id,
        }

        await self.run(context)

        return await self._compile_report(directive.topic, session_id)
