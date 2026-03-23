"""Manager agent - coordinates claim verification and critiques evidence."""

import asyncio
import json
import random
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from rich.console import Console

from ..audit import init_decision_logger
from ..knowledge import (
    CredibilityScorer,
    HybridKnowledgeGraphStore,
    IncrementalKnowledgeGraph,
    ManagerQueryInterface,
)
from ..logging_config import get_logger
from ..memory import ExternalMemoryStore, HybridMemory
from ..models.findings import (
    AgentRole,
    Evidence,
    EvidenceReport,
    VerificationDirective,
    VerdictReport,
    CheckSession,
    SubClaim,
    is_meta_question,
)
from ..retrieval import get_findings_retriever
from ..storage.database import VeritasDatabase
from ..verification import (
    BatchVerificationResult,
    VerificationConfig,
    VerificationPipeline,
)
from .base import AgentConfig, BaseAgent, DecisionType
from .intern import InternAgent
from .kg_processor import KGProcessor
from .parallel import ParallelInternPool

logger = get_logger(__name__)

if TYPE_CHECKING:
    from ..interaction import UserInteraction


class ManagerAgent(BaseAgent):
    """The Manager agent coordinates claim verification and critiques evidence.

    Responsibilities:
    - Decompose claims into specific sub-claims and verification angles
    - Create directives for the Intern agent to gather evidence
    - Critically evaluate the evidence gathered (for and against)
    - Identify gaps, inconsistencies, and areas needing deeper investigation
    - Synthesize evidence into a verdict (True/Mostly True/Mixed/Mostly False/False/Unverifiable)
    - Track verification progress and depth

    Uses Opus model with extended thinking for deep reasoning.
    """

    def __init__(
        self,
        db: VeritasDatabase,
        intern: InternAgent,
        config: AgentConfig | None = None,
        console: Console | None = None,
        pool_size: int = 3,
        use_parallel: bool = True,
        interaction: Optional["UserInteraction"] = None,
        max_depth: int = 5,
    ):
        # Force Opus model for manager's deep reasoning
        if config is None:
            config = AgentConfig()
        config.model = "opus"  # Use Opus for heavy reasoning
        super().__init__(AgentRole.MANAGER, db, config, console)
        self.intern = intern
        self.claim: str = ""
        self.session_id: str = ""
        self.topics_queue: list[SubClaim] = []
        self.completed_topics: list[SubClaim] = []
        self.all_evidence: list[Evidence] = []
        self.all_reports: list[EvidenceReport] = []
        self.current_depth: int = 0
        self.max_depth: int = max_depth
        self.start_time: datetime | None = None
        self.time_limit_minutes: int = 0  # Kept for stats only, not used for stopping
        self._current_phase: str = "init"

        # Locks for thread-safe state access (prevents race conditions in parallel execution)
        self._state_lock = asyncio.Lock()  # Protects topics_queue, all_evidence, all_reports

        # User interaction support
        self.interaction = interaction

        # Knowledge graph integration (connect() called lazily in run_verification)
        self.kg_store = HybridKnowledgeGraphStore(
            db_path=str(db.db_path).replace(".db", "_kg.db"),
            session_id=self.session_id,
        )
        self.knowledge_graph = IncrementalKnowledgeGraph(
            llm_callback=self._kg_llm_callback,
            store=self.kg_store,
            credibility_audit_callback=self._save_credibility_audit,
            session_id=self.session_id,
        )
        self.kg_query = ManagerQueryInterface(self.kg_store)
        self.credibility_scorer = CredibilityScorer()

        # Parallel execution pool
        self.use_parallel = use_parallel
        self.pool_size = pool_size
        self.intern_pool = (
            ParallelInternPool(
                db=db,
                pool_size=pool_size,
                config=config,
                console=console,
            )
            if use_parallel
            else None
        )

        # Hybrid memory for long verification sessions
        self.memory = HybridMemory(
            max_recent_tokens=8000,
            summary_threshold=0.8,
            llm_callback=self._memory_llm_callback,
        )
        self.external_memory = ExternalMemoryStore(
            db_path=str(db.db_path).replace(".db", "_memory.db")
        )

        # Hybrid retrieval for semantic search over evidence
        self.findings_retriever = get_findings_retriever(
            persist_dir=str(db.db_path).replace(".db", "_retrieval"),
            session_id=self.session_id,
            use_reranker=True,  # Quality is priority
        )

        # KG processing (extracted from ManagerAgent to reduce complexity)
        self.kg_processor = KGProcessor(
            knowledge_graph=self.knowledge_graph,
            findings_retriever=self.findings_retriever,
            log=self._log,
        )

        # Verification pipeline for hallucination reduction
        self.verification_config = VerificationConfig()
        self.verification_pipeline = VerificationPipeline(
            llm_callback=self._verification_llm_callback,
            knowledge_graph=self.knowledge_graph,
            search_callback=self._verification_search_callback,
            config=self.verification_config,
        )
        # Pass pipeline to intern and intern pool
        self.intern.verification_pipeline = self.verification_pipeline
        if self.intern_pool:
            self.intern_pool.set_verification_pipeline(self.verification_pipeline)

        # Track batch verification results for reports
        self.last_batch_verification: BatchVerificationResult | None = None

    async def _kg_llm_callback(
        self,
        prompt: str,
        **kwargs,
    ) -> str | dict | list:
        """LLM callback for knowledge graph extraction (uses faster model)."""
        logger.debug("KG LLM callback: prompt_len=%d", len(prompt))
        return await self.call_claude(prompt, model_override="sonnet", **kwargs)

    async def _save_credibility_audit(self, audit_data: dict) -> None:
        """Save credibility audit to database (fire-and-forget)."""
        try:
            await self.db.save_credibility_audit(
                session_id=self.session_id,
                finding_id=audit_data.get("finding_id"),
                url=audit_data.get("url", ""),
                domain=audit_data.get("domain", ""),
                final_score=audit_data.get("final_score", 0.0),
                domain_authority_score=audit_data.get("domain_authority_score", 0.0),
                recency_score=audit_data.get("recency_score", 0.5),
                source_type_score=audit_data.get("source_type_score", 0.6),
                https_score=audit_data.get("https_score", 0.5),
                path_depth_score=audit_data.get("path_depth_score", 0.8),
                credibility_label=audit_data.get("credibility_label", "Medium"),
            )
        except Exception as e:
            # Log error but don't stop processing
            self._log(f"[Credibility Audit Error] Failed to save audit: {e}", style="bold red")
            logger.warning("Failed to save credibility audit: %s", e, exc_info=True)

    async def _memory_llm_callback(self, prompt: str) -> str:
        """LLM callback for memory summarization (uses faster model)."""
        return await self.call_claude(prompt, model_override="haiku")

    _VERIFICATION_SYSTEM_PROMPT = (
        "You are a fact-verification assistant. Your job is to generate verification "
        "questions, independently answer them, and compare answers to assess factual "
        "accuracy. Always respond using the requested structured output format."
    )

    async def _verification_llm_callback(
        self,
        prompt: str,
        model: str = "sonnet",
        output_format: dict | None = None,
    ) -> str | dict | list:
        """LLM callback for verification (model specified by verification pipeline).

        Args:
            prompt: The prompt to send
            model: Model to use (e.g. "haiku", "sonnet")
            output_format: Optional JSON schema for structured output
        """
        logger.debug("Verification LLM callback: model=%s, prompt_len=%d", model, len(prompt))
        return await self.call_claude(
            prompt,
            output_format=output_format,
            model_override=model,
            system_prompt_override=self._VERIFICATION_SYSTEM_PROMPT,
        )

    async def _verification_search_callback(self, query: str) -> list[dict]:
        """Web search callback for CoVe/CRITIC independent verification.

        Uses the intern's search tool (Bright Data) to fetch evidence so
        the verification pipeline can ground answers in real web data
        instead of relying solely on parametric knowledge.

        Scrapes the top result's full page content for richer evidence
        when available, so the LLM can evaluate actual source material
        rather than just search snippets.
        """
        logger.debug("Verification search: query=%s", query)
        try:
            results, _ = await self.intern.search_tool.search(query)
            if not results:
                return []

            output = []
            for r in results[:5]:
                output.append(
                    {
                        "title": r.title,
                        "url": r.url,
                        "snippet": r.snippet,
                    }
                )

            # Scrape the top result for richer evidence context
            try:
                page_content = await self.intern.search_tool.fetch_page(results[0].url)
                if page_content and len(page_content) > 100:
                    output[0]["content"] = page_content[:1500]
            except Exception as e:
                logger.debug("Verification search scrape fallback: %s", e, exc_info=True)
                pass  # Snippet fallback is fine

            return output
        except Exception as e:
            logger.warning("Verification search failed: %s", e, exc_info=True)
            return []

    @property
    def system_prompt(self) -> str:
        return """You are a Claim Verification Manager agent. You coordinate the fact-checking process and determine verdicts.

RESPONSIBILITIES:
1. Decompose claims into specific, verifiable sub-claims
2. Create clear directives for the Evidence-Gathering Intern
3. Critically evaluate evidence gathered (both supporting AND contradicting)
4. Identify gaps, inconsistencies, and areas needing deeper investigation
5. Weigh evidence quality, source credibility, and consistency
6. Determine a verdict based on the totality of evidence

VERIFICATION STRATEGY:
- Decompose complex claims into atomic, verifiable sub-claims
- For each sub-claim, seek BOTH supporting and contradicting evidence
- Prioritize primary sources over secondary reporting
- Cross-reference multiple independent sources
- Check for context: is the claim technically true but misleading?
- Look for the original source of the claim
- Check if established fact-checkers have already evaluated this claim

EVIDENCE EVALUATION FRAMEWORK:
When reviewing evidence, consider:
- Accuracy: Is this evidence from a credible, verifiable source?
- Relevance: Does this directly address the claim being verified?
- Strength: Is this direct evidence or circumstantial?
- Independence: Are sources independently confirming, or echoing one source?
- Recency: Is the evidence current and applicable?
- Contradictions: Are there conflicting pieces of evidence?

VERDICT SCALE:
- TRUE: Strong, consistent evidence supports the claim
- MOSTLY TRUE: Evidence largely supports the claim but with minor caveats
- MIXED: Significant evidence both for and against; claim is partially true
- MOSTLY FALSE: Evidence largely contradicts the claim with minor true elements
- FALSE: Strong, consistent evidence contradicts the claim
- UNVERIFIABLE: Insufficient evidence to determine truth value

QUALITY STANDARDS:
- Reject evidence that is speculation presented as fact
- Flag contradictions for investigation
- Prioritize primary sources over secondary
- Note when confidence is low
- Weight verified evidence higher than unverified

OUTPUT FORMAT:
Provide structured analysis with clear reasoning. When creating directives:
- Be specific about what evidence to search for
- Explain why this angle matters for verification
- Set appropriate priority (1-10)
- Define what would constitute supporting vs contradicting evidence"""

    async def think(self, context: dict[str, Any]) -> str:
        """Reason about verification progress and next steps."""
        logger.debug("Manager think: iteration=%d, evidence=%d, topics=%d", self.state.iteration, len(self.all_evidence), len(self.topics_queue))
        time_elapsed = self._get_elapsed_minutes()
        iterations_remaining = self.config.max_iterations - self.state.iteration

        # Check for user guidance messages
        user_guidance = ""
        if self.interaction:
            messages = self.interaction.get_pending_messages()
            if messages:
                guidance_texts = [m.content for m in messages]
                user_guidance = (
                    "USER GUIDANCE (please incorporate this into your verification):\n"
                    + "\n".join(f"- {g}" for g in guidance_texts)
                )
                self._log(
                    f"[USER GUIDANCE] Received {len(messages)} message(s)", style="bold yellow"
                )
                for m in messages:
                    self._log(
                        f"  -> {m.content[:100]}{'...' if len(m.content) > 100 else ''}",
                        style="yellow",
                    )

        # Summarize current state
        evidence_summary = self._summarize_evidence()

        # Get knowledge graph insights
        kg_summary = await self.kg_query.get_research_summary()
        verification_directions = await self.kg_query.get_next_research_directions()

        # Use hybrid retrieval to find semantically relevant past evidence
        relevant_evidence_text = ""
        if self.findings_retriever.count() > 0:
            try:
                # Search for evidence relevant to the claim
                relevant = self.findings_retriever.search(
                    query=self.claim,
                    limit=5,
                    session_id=self.session_id,
                )
                if relevant:
                    relevant_evidence_text = "Most relevant evidence (via semantic search):\n"
                    for r in relevant:
                        ev = getattr(r, 'evidence', None) or getattr(r, 'finding', None)
                        if ev:
                            etype = getattr(ev, 'evidence_type', getattr(ev, 'finding_type', None))
                            etype_val = etype.value if hasattr(etype, 'value') else str(etype)
                            relevant_evidence_text += f"- [{etype_val}] {ev.content[:200]}... (score: {r.score:.2f})\n"
            except Exception as e:
                self._log(f"[RETRIEVAL] Search error: {e}", style="dim")
                logger.warning("Retrieval search error: %s", e, exc_info=True)

        # Get memory context for continuity
        memory_context = self.memory.get_context_for_prompt(max_tokens=2000)

        prompt = f"""Claim to Verify: {self.claim}

{user_guidance}

Current Status:
- Iteration: {self.state.iteration}/{self.config.max_iterations} ({iterations_remaining} remaining)
- Time elapsed: {time_elapsed:.1f} minutes
- Sub-claims completed: {len(self.completed_topics)}
- Sub-claims in queue: {len(self.topics_queue)}
- Total evidence gathered: {len(self.all_evidence)}
- Current depth: {self.current_depth}/{self.max_depth}

Last evidence report from Intern:
{context.get("last_report_summary", "No report yet")}

Evidence summary:
{evidence_summary}

{relevant_evidence_text}

{kg_summary}

Suggested verification directions from knowledge analysis:
{chr(10).join(["- " + d for d in verification_directions[:5]]) if verification_directions else "None yet"}

{f"Session Memory Context:{chr(10)}{memory_context}" if memory_context else ""}

What should I do next? Consider:
1. Are there gaps in the evidence identified by knowledge graph analysis?
2. Should I go deeper on any sub-claim?
3. Are there contradictions that need resolution?
4. Do I have both supporting AND contradicting evidence?
5. Is it time to determine a verdict and synthesize the report?

Think step by step about the best next action."""

        thought = await self.call_claude(prompt, use_thinking=True)

        # Track in memory
        await self.memory.add_message(
            role="assistant",
            content=f"Reasoning: {thought[:500]}...",
            metadata={"type": "thought", "iteration": self.state.iteration},
        )

        # Compress memory if needed
        await self.memory.maybe_compress()

        return thought

    async def act(self, thought: str, context: dict[str, Any]) -> dict[str, Any]:
        """Execute management actions based on thinking."""
        logger.info("Manager act: thought=%s", thought[:200])
        # Check if pause was requested before starting any long operation
        if self._pause_requested:
            return {"action": "paused"}

        # Periodic checkpoint for crash recovery
        await self._maybe_periodic_checkpoint()

        # Check if we should synthesize and stop
        if self._should_synthesize(thought):
            report = await self._synthesize_report()
            return {
                "action": "synthesize",
                "report": report,
            }

        # Check if we should create a new directive for the intern
        if self._should_create_directive(thought):
            directive = await self._create_directive(thought)
            if directive:
                self._log("=" * 70, style="bold blue")
                self._log(
                    f"[DIRECTIVE] {directive.action.upper()}: {directive.topic}", style="bold green"
                )
                self._log(f"  Instructions: {directive.instructions}", style="dim")
                self._log(
                    f"  Priority: {directive.priority}/10 | Max Searches: {directive.max_searches}",
                    style="dim",
                )
                self._log("=" * 70, style="bold blue")

                # Check pause before long intern operation
                if self._pause_requested:
                    return {"action": "paused"}

                # Execute the directive via the intern
                intern_report = await self.intern.execute_directive(directive, self.session_id)
                logger.info("Intern directive complete: topic=%s, evidence=%d", directive.topic, len(intern_report.evidence))
                async with self._state_lock:
                    self.all_reports.append(intern_report)
                    self.all_evidence.extend(intern_report.evidence)

                # Process evidence into knowledge graph
                await self.kg_processor.process_evidence(intern_report.evidence, self.session_id)

                # Critique the report
                critique = await self._critique_report(intern_report)

                # Show critique
                self._log("-" * 70, style="dim")
                self._log("[MANAGER CRITIQUE]", style="bold magenta")
                self.console.print(critique)
                self._log("-" * 70, style="dim")

                # Add follow-up topics to queue
                await self._process_followups(intern_report, directive)

                # Show follow-up topics added
                if intern_report.suggested_followups:
                    self._log(
                        f"[Follow-up Angles Added: {len(intern_report.suggested_followups)}]",
                        style="cyan",
                    )
                    for ft in intern_report.suggested_followups[:3]:
                        self._log(f"  -> {ft}", style="cyan")

                return {
                    "action": "intern_task",
                    "directive": directive,
                    "report": intern_report,
                    "critique": critique,
                }

        # Check pending topics - use parallel execution if multiple topics queued
        if self.topics_queue:
            # Use parallel execution if we have multiple topics and pool is available
            if len(self.topics_queue) >= 2 and self.intern_pool:
                # Pop multiple topics for parallel execution (with lock for thread safety)
                async with self._state_lock:
                    topics_to_run = []
                    for _ in range(min(self.pool_size, len(self.topics_queue))):
                        if self.topics_queue:
                            topics_to_run.append(self.topics_queue.pop(0))

                # Log topic selection decision
                await self._log_decision(
                    session_id=self.session_id,
                    decision_type=DecisionType.TOPIC_SELECTION,
                    decision_outcome="parallel_execution",
                    reasoning=f"Selected {len(topics_to_run)} sub-claims for parallel verification",
                    inputs={
                        "queue_size": len(self.topics_queue) + len(topics_to_run),
                        "selected_topics": [t.topic for t in topics_to_run],
                        "depths": [t.depth for t in topics_to_run],
                    },
                    metrics={
                        "evidence_count": len(self.all_evidence),
                        "completed_topics": len(self.completed_topics),
                    },
                )

                # Check pause before long parallel operation
                if self._pause_requested:
                    # Put topics back in queue
                    async with self._state_lock:
                        self.topics_queue = topics_to_run + self.topics_queue
                    return {"action": "paused"}

                self._log("=" * 70, style="bold blue")
                self._log(
                    f"[PARALLEL SUB-CLAIMS] Verifying {len(topics_to_run)} sub-claims in parallel",
                    style="bold green",
                )
                for t in topics_to_run:
                    self._log(f"  * {t.topic}", style="dim")
                self._log("=" * 70, style="bold blue")

                await self._run_parallel_topics(topics_to_run, max_parallel=self.pool_size)

                # Track in memory
                await self.memory.add_message(
                    role="system",
                    content=f"Completed parallel verification on {len(topics_to_run)} sub-claims",
                    metadata={"topics": [t.topic for t in topics_to_run]},
                )

                return {
                    "action": "parallel_topics",
                    "topics": topics_to_run,
                    "evidence_count": len(self.all_evidence),
                }

            # Single topic - use regular intern
            async with self._state_lock:
                topic = self.topics_queue.pop(0)

            # Log topic selection decision
            await self._log_decision(
                session_id=self.session_id,
                decision_type=DecisionType.TOPIC_SELECTION,
                decision_outcome="single_topic",
                reasoning=f"Selected sub-claim '{topic.topic}' from queue",
                inputs={
                    "queue_size": len(self.topics_queue) + 1,
                    "selected_topic": topic.topic,
                    "depth": topic.depth,
                    "priority": topic.priority,
                },
                metrics={
                    "evidence_count": len(self.all_evidence),
                    "completed_topics": len(self.completed_topics),
                },
            )

            directive = VerificationDirective(
                action="search",
                topic=topic.topic,
                instructions=f"Gather evidence for and against this sub-claim: {topic.topic}",
                priority=topic.priority,
                max_searches=5,
            )

            # Check pause before long intern operation
            if self._pause_requested:
                # Put topic back in queue
                async with self._state_lock:
                    self.topics_queue.insert(0, topic)
                return {"action": "paused"}

            self._log("=" * 70, style="bold blue")
            self._log(f"[QUEUED SUB-CLAIM] Depth {topic.depth}: {topic.topic}", style="bold green")
            self._log(
                f"  Priority: {topic.priority}/10 | Remaining in queue: {len(self.topics_queue)}",
                style="dim",
            )
            self._log("=" * 70, style="bold blue")

            intern_report = await self.intern.execute_directive(directive, self.session_id)
            logger.info("Intern directive complete: topic=%s, evidence=%d", directive.topic, len(intern_report.evidence))
            async with self._state_lock:
                self.all_reports.append(intern_report)
                self.all_evidence.extend(intern_report.evidence)
                self.completed_topics.append(topic)
                self.current_depth = max(self.current_depth, topic.depth)

            # Process evidence into knowledge graph
            await self.kg_processor.process_evidence(intern_report.evidence, self.session_id)

            # Track in memory
            await self.memory.add_message(
                role="system",
                content=f"Completed verification on: {topic.topic} - {len(intern_report.evidence)} evidence items",
                metadata={"topic": topic.topic, "evidence_count": len(intern_report.evidence)},
            )

            await self.db.update_topic_status(topic.id, "completed", len(intern_report.evidence))

            critique = await self._critique_report(intern_report)

            # Show critique
            self._log("-" * 70, style="dim")
            self._log("[MANAGER CRITIQUE]", style="bold magenta")
            self.console.print(critique)
            self._log("-" * 70, style="dim")

            await self._process_followups(intern_report, directive, topic)

            # Show follow-up angles added
            if intern_report.suggested_followups:
                self._log(
                    f"[Follow-up Angles Added: {len(intern_report.suggested_followups)}]",
                    style="cyan",
                )
                for ft in intern_report.suggested_followups[:3]:
                    self._log(f"  -> {ft}", style="cyan")

            return {
                "action": "intern_task",
                "directive": directive,
                "report": intern_report,
                "critique": critique,
            }

        # Nothing to do, synthesize
        report = await self._synthesize_report()
        return {
            "action": "synthesize",
            "report": report,
        }

    async def _maybe_periodic_checkpoint(self) -> None:
        """Fire-and-forget checkpoint every 2 iterations for crash recovery."""
        if self.state.iteration % 2 == 0 and self.session_id:
            try:
                session = await self.db.get_session(self.session_id)
                if session:
                    task = asyncio.create_task(self._periodic_checkpoint(session))
                    self._background_tasks.add(task)
                    task.add_done_callback(self._background_tasks.discard)
            except Exception:
                logger.debug("Periodic checkpoint failed", exc_info=True)

    async def observe(self, action_result: dict[str, Any]) -> str:
        """Process the result of a management action."""
        action = action_result.get("action")

        if action == "paused":
            return "Verification paused by user request"

        if action == "synthesize":
            report: VerdictReport = action_result.get("report")
            return f"Verdict report synthesized: {len(report.key_evidence)} key evidence items, verdict: {report.verdict}, {len(report.recommended_next_steps)} recommendations"

        if action == "intern_task":
            report: EvidenceReport = action_result.get("report")
            critique = action_result.get("critique", "")
            directive: VerificationDirective = action_result.get("directive")

            return f"""Intern completed evidence gathering on '{directive.topic}':
- Evidence items: {len(report.evidence)}
- Searches: {report.searches_performed}
- Follow-ups suggested: {len(report.suggested_followups)}
- Critique: {critique[:200]}..."""

        if action == "parallel_topics":
            topics = action_result.get("topics", [])
            evidence_count = action_result.get("evidence_count", 0)
            return f"""Parallel verification completed:
- Sub-claims verified: {len(topics)}
- Total evidence so far: {evidence_count}
- Sub-claims: {[t.topic for t in topics]}"""

        return "Unknown action"

    def is_done(self, context: dict[str, Any]) -> bool:
        """Check if the Manager should stop (iteration-based)."""
        # Check if synthesis complete
        last_action = context.get("last_action", {})
        if last_action.get("action") == "synthesize":
            return True

        # Check max iterations (the only stopping condition now)
        if self.state.iteration >= self.config.max_iterations:
            self._log(
                f"Iteration limit reached ({self.state.iteration}/{self.config.max_iterations})",
                style="yellow",
            )
            return True

        return False

    def _get_elapsed_minutes(self) -> float:
        """Get elapsed time in minutes."""
        if not self.start_time:
            return 0
        return (datetime.now() - self.start_time).total_seconds() / 60

    def _summarize_evidence(self) -> str:
        """Create a brief summary of all evidence."""
        if not self.all_evidence:
            return "No evidence yet."

        by_type = {}
        for e in self.all_evidence:
            t = e.evidence_type.value
            by_type.setdefault(t, []).append(e)

        summary_parts = []
        for etype, items in by_type.items():
            summary_parts.append(f"- {etype.upper()}: {len(items)} items")
            for e in items[:2]:  # Show first 2 of each type
                summary_parts.append(f"  * {e.content[:100]}...")

        return "\n".join(summary_parts)

    async def _maybe_ask_user(
        self,
        question: str,
        context: str = "",
        options: list[str] | None = None,
    ) -> str | None:
        """Ask the user a question during verification if interaction is enabled.

        This is non-blocking with a timeout - if the user doesn't respond,
        verification continues autonomously.

        Args:
            question: The question to ask
            context: Context about why this is being asked
            options: Optional suggested answers

        Returns:
            User's response, or None if no interaction or timeout
        """
        if not self.interaction:
            return None

        response = await self.interaction.ask_with_timeout(
            question=question,
            context=context,
            options=options,
        )

        if response:
            # Log that we got a response
            self._log(
                f"[USER RESPONSE] {response[:100]}{'...' if len(response) > 100 else ''}",
                style="bold green",
            )

            # Add to memory for context
            await self.memory.add_message(
                role="user",
                content=f"User guidance: {response}",
                metadata={"type": "mid_verification_response"},
            )

        return response

    def _should_synthesize(self, thought: str) -> bool:
        """Determine if it's time to synthesize a verdict (iteration-based)."""
        # Never synthesize if we have no evidence
        if not self.all_evidence:
            return False

        iterations_remaining = self.config.max_iterations - self.state.iteration

        # On last iteration, always synthesize if we have evidence
        if iterations_remaining <= 0 and self.all_evidence:
            task = asyncio.create_task(
                self._log_decision(
                    session_id=self.session_id,
                    decision_type=DecisionType.SYNTHESIS_TRIGGER,
                    decision_outcome="triggered_last_iteration",
                    reasoning=f"Last iteration with {len(self.all_evidence)} evidence items",
                    inputs={"evidence_count": len(self.all_evidence)},
                    metrics={
                        "iteration": self.state.iteration,
                        "max_iterations": self.config.max_iterations,
                    },
                )
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            return True

        # Don't allow early synthesis until at least 80% of iterations are done
        iteration_pct = (self.state.iteration / self.config.max_iterations) * 100
        if iteration_pct < 80:
            return False

        # Need at least some evidence before synthesizing
        if len(self.all_evidence) < 3:
            return False

        # Check explicit signals from LLM
        thought_lower = thought.lower()
        synthesis_signals = [
            "time to synthesize",
            "determine the verdict",
            "render a verdict",
            "final verdict",
            "conclude the verification",
            "sufficient evidence",
            "enough evidence",
            "ready to conclude",
            "wrap up the fact-check",
        ]
        should_synthesize = any(signal in thought_lower for signal in synthesis_signals)

        if should_synthesize:
            task = asyncio.create_task(
                self._log_decision(
                    session_id=self.session_id,
                    decision_type=DecisionType.SYNTHESIS_TRIGGER,
                    decision_outcome="triggered_explicit_signal",
                    reasoning=thought[:500],
                    inputs={"evidence_count": len(self.all_evidence)},
                    metrics={
                        "iteration": self.state.iteration,
                        "max_iterations": self.config.max_iterations,
                        "topics_completed": len(self.completed_topics),
                    },
                )
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        return should_synthesize

    def _should_create_directive(self, thought: str) -> bool:
        """Determine if we should create a new directive."""
        thought_lower = thought.lower()
        directive_signals = [
            "search for",
            "investigate",
            "look into",
            "explore",
            "gather evidence",
            "find out",
            "verify",
            "check whether",
            "fact-check",
            "confirm",
            "cross-reference",
        ]
        return any(signal in thought_lower for signal in directive_signals)

    async def _create_directive(self, thought: str) -> VerificationDirective | None:
        """Create a directive for the Intern based on reasoning."""
        prompt = (
            f"Based on this reasoning:\n{thought}\n\nCreate a directive for the Evidence-Gathering Intern. "
            "The directive should specify what evidence to search for, including both supporting and contradicting evidence."
        )

        schema = {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["search", "deep_dive", "verify"],
                    },
                    "topic": {"type": "string"},
                    "instructions": {"type": "string"},
                    "priority": {"type": "integer"},
                    "max_searches": {"type": "integer"},
                },
                "required": [
                    "action",
                    "topic",
                    "instructions",
                    "priority",
                    "max_searches",
                ],
            },
        }

        try:
            response = await self.call_claude(
                prompt,
                output_format=schema,
            )

            if isinstance(response, dict):
                data = response
            else:
                start = response.find("{")
                end = response.rfind("}") + 1
                if start == -1 or end <= start:
                    return None
                data = json.loads(response[start:end])

            directive = VerificationDirective(
                action=data.get("action", "search"),
                topic=data.get("topic", ""),
                instructions=data.get("instructions", ""),
                priority=data.get("priority", 5),
                max_searches=data.get("max_searches", 5),
            )

            # Log directive creation decision
            await self._log_decision(
                session_id=self.session_id,
                decision_type=DecisionType.DIRECTIVE_CREATE,
                decision_outcome=directive.action,
                reasoning=thought[:500],
                inputs={
                    "action": directive.action,
                    "topic": directive.topic,
                    "priority": directive.priority,
                    "max_searches": directive.max_searches,
                },
                metrics={
                    "evidence_count": len(self.all_evidence),
                    "iterations_remaining": self.config.max_iterations - self.state.iteration,
                },
            )

            return directive
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to create directive from LLM response: %s", e, exc_info=True)

        return None

    async def _apply_verification_results(self, batch_result, evidence_list: list) -> None:
        """Apply batch verification results to evidence and persist to DB.

        Used by both _critique_report and _synthesize_report to update evidence
        verification status, confidence, and save detailed results.
        """
        for result in batch_result.results:
            for e in evidence_list:
                if str(e.id or hash(e.content)) == result.finding_id:
                    e.original_confidence = e.confidence
                    e.confidence = result.verified_confidence
                    e.verification_status = result.verification_status.value
                    e.verification_method = result.verification_method.value
                    e.kg_support_score = result.kg_support_score

                    if e.id:
                        await self.db.update_finding_verification(
                            finding_id=e.id,
                            verification_status=e.verification_status,
                            verification_method=e.verification_method,
                            kg_support_score=e.kg_support_score,
                            original_confidence=e.original_confidence,
                            new_confidence=e.confidence,
                        )

                        await self.db.save_verification_result(
                            session_id=self.session_id,
                            finding_id=e.id,
                            result_dict=result.to_dict(),
                        )
                    break

    async def _critique_report(self, report: EvidenceReport) -> str:
        """Critique an Intern's evidence report with batch verification."""
        # Run batch verification on evidence if not already verified
        unverified = [e for e in report.evidence if not e.verification_status]
        if unverified and self.verification_config.enable_batch_verification:
            self._log(
                f"[VERIFY] Running batch verification on {len(unverified)} evidence items...", style="dim"
            )
            logger.info("Running batch verification on %d evidence items", len(unverified))
            batch_result = await self.verification_pipeline.verify_batch(
                unverified, self.session_id
            )
            self.last_batch_verification = batch_result

            await self._apply_verification_results(batch_result, report.evidence)

            # Log verification summary
            self._log(
                f"[VERIFY] Results: {batch_result.verified_count} verified, "
                f"{batch_result.flagged_count} flagged, {batch_result.rejected_count} rejected",
                style="dim",
            )
            logger.info("Verification results: verified=%d, flagged=%d, rejected=%d", batch_result.verified_count, batch_result.flagged_count, batch_result.rejected_count)

        # Separate evidence by verification status
        verified = [e for e in report.evidence if e.verification_status == "verified"]
        flagged = [e for e in report.evidence if e.verification_status == "flagged"]
        rejected = [e for e in report.evidence if e.verification_status == "rejected"]

        evidence_text = "\n".join(
            [
                f"- [{e.evidence_type.value}] {e.content} (confidence: {e.confidence:.0%}, status: {e.verification_status or 'pending'})"
                for e in report.evidence[:10]
            ]
        )

        verification_summary = ""
        if verified or flagged or rejected:
            verification_summary = f"""
Verification Summary:
- Verified (high confidence): {len(verified)}
- Flagged (needs review): {len(flagged)}
- Rejected (low confidence): {len(rejected)}
"""

        prompt = f"""Critique this evidence report for the claim verification:

Topic: {report.topic}
Searches: {report.searches_performed}
{verification_summary}
Evidence:
{evidence_text}

Suggested follow-ups: {report.suggested_followups}

Evaluate:
1. Quality of evidence (strength, accuracy, relevance to the claim)
2. Balance: Is there evidence both FOR and AGAINST the claim?
3. Verification status - pay special attention to flagged and rejected items
4. Coverage (what angles are missing?)
5. Credibility of sources
6. Suggestions for finding contradicting or confirming evidence

Be constructive but rigorous. Flag any rejected evidence that should be re-investigated."""

        return await self.call_claude(prompt)

    async def _process_followups(
        self,
        report: EvidenceReport,
        directive: VerificationDirective,
        parent_topic: SubClaim | None = None,
    ) -> None:
        """Process follow-up suggestions and add worthy ones to the queue."""
        if self.current_depth >= self.max_depth:
            return

        new_depth = (parent_topic.depth + 1) if parent_topic else self.current_depth + 1

        for followup in report.suggested_followups[:3]:  # Limit follow-ups
            if is_meta_question(followup):
                continue

            # Check if we already have this topic
            existing = [t for t in self.topics_queue if t.topic.lower() == followup.lower()]
            if existing:
                continue

            topic = await self.db.create_topic(
                session_id=self.session_id,
                topic=followup,
                parent_topic_id=parent_topic.id if parent_topic else None,
                depth=new_depth,
                priority=max(1, directive.priority - 1),
            )
            async with self._state_lock:
                self.topics_queue.append(topic)

    async def _synthesize_report(self) -> VerdictReport:
        """Synthesize all evidence into a verdict report with verification awareness."""
        logger.info("Synthesizing verdict report: evidence=%d", len(self.all_evidence))
        time_elapsed = self._get_elapsed_minutes()

        self._log("=" * 70, style="bold cyan")
        self._log(
            f"[VERDICT SYNTHESIS] Starting verdict synthesis with {len(self.all_evidence)} evidence items",
            style="bold cyan",
        )
        self._log("=" * 70, style="bold cyan")

        # Run batch verification on evidence that haven't had thorough verification.
        _streaming_methods = {"streaming", ""}
        needs_batch = [
            e
            for e in self.all_evidence
            if not e.verification_status or (e.verification_method or "") in _streaming_methods
        ]
        if needs_batch and self.verification_config.enable_batch_verification:
            self._log(
                f"[VERIFY] Running thorough verification on {len(needs_batch)} evidence items "
                "(this may take several minutes)...",
                style="yellow",
            )
            batch_result = await self.verification_pipeline.verify_batch(
                needs_batch, self.session_id
            )
            self.last_batch_verification = batch_result

            await self._apply_verification_results(batch_result, self.all_evidence)

        # Separate evidence by verification status
        verified_evidence = [e for e in self.all_evidence if e.verification_status == "verified"]
        flagged_evidence = [e for e in self.all_evidence if e.verification_status == "flagged"]
        rejected_evidence = [e for e in self.all_evidence if e.verification_status == "rejected"]
        other_evidence = [
            e
            for e in self.all_evidence
            if e.verification_status not in ["verified", "flagged", "rejected"]
        ]

        # Priority: verified > flagged > unverified > rejected
        # Weight by calibrated confidence
        priority_evidence = (
            sorted(verified_evidence, key=lambda e: e.confidence, reverse=True)
            + sorted(flagged_evidence, key=lambda e: e.confidence, reverse=True)
            + sorted(other_evidence, key=lambda e: e.confidence, reverse=True)
        )
        key_evidence = priority_evidence[:20]

        evidence_text = "\n".join(
            [
                f"- [{e.evidence_type.value}] {e.content} (verified: {e.verification_status or 'pending'}, confidence: {e.confidence:.0%})"
                for e in key_evidence
            ]
        )

        # Verification context for synthesis
        verification_context = ""
        if verified_evidence or flagged_evidence or rejected_evidence:
            verification_context = f"""
Verification Summary:
- High confidence (verified): {len(verified_evidence)} evidence items
- Medium confidence (flagged for review): {len(flagged_evidence)} evidence items
- Low confidence (rejected): {len(rejected_evidence)} evidence items
- Unverified: {len(other_evidence)} evidence items

Note: Prioritize verified evidence in your verdict. Flagged evidence may need additional context.
Rejected evidence ({len(rejected_evidence)}) has low confidence and should not drive primary conclusions.
"""

        prompt = f"""Synthesize all evidence into a final verdict for this claim.

Claim: {self.claim}
{verification_context}
Key Evidence (sorted by verification confidence):
{evidence_text}

Sub-Claims Investigated: {[t.topic for t in self.completed_topics]}
Sub-Claims Remaining: {[t.topic for t in self.topics_queue[:5]]}

Create:
1. A VERDICT: One of TRUE, MOSTLY_TRUE, MIXED, MOSTLY_FALSE, FALSE, or UNVERIFIABLE
2. A comprehensive summary (2-3 paragraphs) explaining the verdict
   - Base conclusions on verified/high-confidence evidence
   - Note supporting and contradicting evidence
   - Explain why the verdict was chosen
3. Quality assessment of the evidence (including verification rates)
4. Recommended next steps if more investigation is warranted

Be thorough and balanced. Note where evidence has lower confidence."""

        response = await self.call_claude(prompt, use_thinking=True)

        # Extract verdict from the response
        verdict = self._extract_verdict(response)

        return VerdictReport(
            summary=response,
            verdict=verdict,
            key_evidence=key_evidence,
            sub_claims_explored=[t.topic for t in self.completed_topics],
            sub_claims_remaining=[t.topic for t in self.topics_queue],
            quality_assessment="",
            recommended_next_steps=[],
            time_elapsed_minutes=time_elapsed,
            iterations_completed=self.state.iteration,
        )

    def _extract_verdict(self, response: str) -> str:
        """Extract verdict from the synthesis response."""
        response_upper = response.upper()
        verdicts = [
            "UNVERIFIABLE",
            "MOSTLY_TRUE", "MOSTLY TRUE", "MOSTLY-TRUE",
            "MOSTLY_FALSE", "MOSTLY FALSE", "MOSTLY-FALSE",
            "MIXED",
            "FALSE",
            "TRUE",
        ]
        for v in verdicts:
            if v in response_upper:
                # Normalize to underscore format
                return v.replace(" ", "_").replace("-", "_")
        return "UNVERIFIABLE"

    async def _run_parallel_initial_verification(self, claim: str, max_aspects: int = 3) -> None:
        """Run parallel initial evidence gathering to quickly build a broad evidence base.

        Decomposes the claim into distinct verification angles and gathers evidence
        in parallel using the intern pool.

        Args:
            claim: The main claim to verify
            max_aspects: Maximum number of parallel verification threads
        """
        if not self.intern_pool:
            return

        self._log("=" * 70, style="bold cyan")
        self._log("[PARALLEL VERIFICATION] Starting parallel initial evidence gathering", style="bold cyan")
        self._log("=" * 70, style="bold cyan")

        # Record in memory
        await self.memory.add_message(
            role="system",
            content=f"Starting parallel evidence gathering on claim: {claim}",
            metadata={"phase": "parallel_init"},
        )

        # Decompose claim into verification angles first
        aspects = await self.intern_pool._decompose_claim(
            claim=claim,
            llm_callback=self._kg_llm_callback,
            max_aspects=max_aspects,
        )

        self._log(f"[PARALLEL] Decomposed into {len(aspects)} verification angles:", style="cyan")
        for aspect in aspects:
            self._log(f"  * {aspect}", style="dim")

        # Create topics in database for tracking
        aspect_topics = []
        for aspect in aspects:
            topic = await self.db.create_topic(
                session_id=self.session_id,
                topic=aspect,
                depth=1,
                priority=9,
            )
            aspect_topics.append(topic)

        # Create directives from aspects
        directives = [
            VerificationDirective(
                action="search",
                topic=aspect,
                instructions=f"Gather evidence for and against this aspect of the claim: {aspect}",
                priority=8,
                max_searches=5,
            )
            for aspect in aspects
        ]

        # Check pause before long parallel operation
        if self._pause_requested:
            return

        # Execute in parallel
        result = await self.intern_pool.gather_evidence_parallel(directives, self.session_id)

        # Process results (with lock for thread safety)
        async with self._state_lock:
            self.all_evidence.extend(result.total_evidence)
            self.all_reports.extend(result.reports)

            # Mark topics as completed and track them
            for topic in aspect_topics:
                self.completed_topics.append(topic)

        # Update DB status outside lock
        for topic in aspect_topics:
            await self.db.update_topic_status(
                topic.id, "completed", len(result.total_evidence) // len(aspects)
            )

        # Update depth tracking
        self.current_depth = max(self.current_depth, 1)

        # Process all evidence into KG in real-time (fast mode)
        await self.kg_processor.process_evidence(result.total_evidence, self.session_id)

        # Store evidence summary in external memory for later retrieval
        if result.total_evidence:
            evidence_summary = "\n".join(
                [f"- {e.content[:200]}" for e in result.total_evidence[:20]]
            )
            await self.external_memory.store(
                session_id=self.session_id,
                content=f"Parallel verification evidence:\n{evidence_summary}",
                memory_type="finding",
                tags=["parallel", "initial"],
                metadata={"count": len(result.total_evidence)},
            )

        # Record completion in memory
        await self.memory.add_message(
            role="system",
            content=f"Parallel evidence gathering complete: {len(result.total_evidence)} evidence items from {result.total_searches} searches in {result.execution_time_seconds:.1f}s",
            metadata={"phase": "parallel_complete", "evidence_count": len(result.total_evidence)},
        )

        self._log("=" * 70, style="bold cyan")
        self._log(
            f"[PARALLEL VERIFICATION] Complete: {len(result.total_evidence)} evidence items, "
            f"{result.total_searches} searches, {result.execution_time_seconds:.1f}s",
            style="bold green",
        )
        if result.errors:
            self._log(f"  Errors: {len(result.errors)}", style="yellow")
        self._log("=" * 70, style="bold cyan")

    async def _run_parallel_topics(
        self, topics: list[SubClaim], max_parallel: int = 3
    ) -> None:
        """Run multiple queued sub-claims in parallel.

        Args:
            topics: List of sub-claims to verify in parallel
            max_parallel: Maximum number to run at once
        """
        if not self.intern_pool or not topics:
            return

        # Check pause before starting
        if self._pause_requested:
            return

        # Create directives from topics
        from ..models.findings import VerificationDirective

        directives = [
            VerificationDirective(
                action="search",
                topic=topic.topic,
                instructions=f"Gather evidence for and against this sub-claim: {topic.topic}",
                priority=topic.priority,
                max_searches=5,
            )
            for topic in topics[:max_parallel]
        ]

        self._log(f"[PARALLEL] Running {len(directives)} sub-claims in parallel", style="cyan")
        logger.info("Running parallel sub-claims: count=%d", len(directives))

        result = await self.intern_pool.gather_evidence_parallel(directives, self.session_id)

        # Process results (with lock for thread safety)
        async with self._state_lock:
            self.all_evidence.extend(result.total_evidence)
            self.all_reports.extend(result.reports)

            # Mark topics as completed
            for topic in topics[:max_parallel]:
                self.completed_topics.append(topic)
                self.current_depth = max(self.current_depth, topic.depth)

        # Update database outside lock
        evidence_per_topic = len(result.total_evidence) // max(len(topics[:max_parallel]), 1)
        for topic in topics[:max_parallel]:
            await self.db.update_topic_status(topic.id, "completed", evidence_per_topic)

        # Process evidence to KG
        await self.kg_processor.process_evidence(result.total_evidence, self.session_id)

        # Compress memory if needed
        await self.memory.maybe_compress()

    async def checkpoint_state(self, session: CheckSession) -> None:
        """Save Manager orchestration state to DB for pause/crash recovery."""
        elapsed = self._get_elapsed_minutes() * 60  # Convert to seconds
        session.elapsed_seconds = elapsed
        session.iteration_count = self.state.iteration
        session.phase = self._current_phase
        session.paused_at = datetime.now()
        session.status = "paused"
        await self.db.update_session(session)
        self._log(
            f"[CHECKPOINT] Saved state: elapsed={elapsed:.0f}s, "
            f"iteration={self.state.iteration}, phase={self._current_phase}"
        )

    async def _periodic_checkpoint(self, session: CheckSession) -> None:
        """Update elapsed time and iteration in DB (fire-and-forget)."""
        try:
            elapsed = self._get_elapsed_minutes() * 60
            session.elapsed_seconds = elapsed
            session.iteration_count = self.state.iteration
            session.phase = self._current_phase
            await self.db.update_session(session)
        except Exception:
            logger.debug("Periodic checkpoint update failed", exc_info=True)

    async def restore_state(self, session: CheckSession) -> None:
        """Rebuild Manager state from DB for resume after pause/crash."""
        self._log("[RESTORE] Rebuilding Manager state from database...")

        # Reload all evidence
        self.all_evidence = await self.db.get_session_findings(session.id)
        self._log(f"[RESTORE] Loaded {len(self.all_evidence)} evidence items")

        # Reset in_progress topics to pending (they were mid-execution)
        await self.db.reset_in_progress_topics(session.id)

        # Reconstruct topics from DB
        all_topics = await self.db.get_all_topics(session.id)
        self.topics_queue = [t for t in all_topics if t.status == "pending"]
        self.completed_topics = [t for t in all_topics if t.status == "completed"]
        self._log(
            f"[RESTORE] Queue: {len(self.topics_queue)} pending, "
            f"{len(self.completed_topics)} completed"
        )

        # Restore timing: set start_time so _get_elapsed_minutes() returns correct total
        from datetime import timedelta

        self.start_time = datetime.now() - timedelta(seconds=session.elapsed_seconds)

        # Restore iteration count
        self.state.iteration = session.iteration_count

        # Compute current depth from completed topics
        self.current_depth = max((t.depth for t in self.completed_topics), default=0)

        # Re-initialize decision logger
        await init_decision_logger(self.db)

        # Re-initialize memory context
        await self.memory.add_message(
            role="system",
            content=(
                f"Resumed verification session. "
                f"{len(self.all_evidence)} evidence items loaded, "
                f"{len(self.topics_queue)} sub-claims pending."
            ),
            metadata={"type": "resume", "session_id": session.id},
        )

        # Index existing evidence for retrieval
        if self.all_evidence:
            try:
                self.findings_retriever.add_findings(
                    findings=self.all_evidence,
                    session_id=session.id,
                )
            except Exception:
                logger.warning("Failed to index evidence during restore", exc_info=True)

        # KG auto-loads from its SQLite DB, just set session_id
        self.knowledge_graph.session_id = session.id

        # Clear pause flags
        self._pause_requested = False
        self.intern._pause_requested = False
        if self.intern_pool:
            self.intern_pool._pause_requested = False

        self._log("[RESTORE] State restoration complete")

    async def run_verification(
        self,
        claim: str,
        session_id: str,
        max_iterations: int = 5,
        use_parallel_init: bool = True,
        resume: bool = False,
        session: CheckSession | None = None,
    ) -> VerdictReport:
        """Run a complete claim verification session.

        Args:
            claim: The claim to verify
            session_id: Session ID for persistence (7-char hex)
            max_iterations: Number of manager ReAct loop iterations (controls depth)
            use_parallel_init: If True, start with parallel decomposition phase
            resume: If True, restore state from DB and continue
            session: Existing session object (required if resume=True)
        """
        self.claim = claim
        self.session_id = session_id
        self.knowledge_graph.session_id = session_id
        self.config.max_iterations = max_iterations
        self._current_phase = "init"

        # Initialize KG store async connection
        if self.kg_store._connection is None:
            await self.kg_store.connect()

        # Eagerly initialize external memory DB
        try:
            await self.external_memory._ensure_connected()
        except Exception:
            logger.debug("External memory initialization failed", exc_info=True)

        if resume and session:
            # Resume from checkpoint
            await self.restore_state(session)
            session.paused_at = None
            session.status = "running"
            await self.db.update_session(session)
            self._current_phase = "react_loop"
        else:
            # Fresh start
            self.start_time = datetime.now()

            # Initialize decision logger for audit trail
            await init_decision_logger(self.db)

            # Initialize memory for this session
            await self.memory.add_message(
                role="user", content=f"Claim to verify: {claim}", metadata={"session_id": session_id}
            )

            # Phase 1: Parallel initial evidence gathering (if enabled and pool available)
            if use_parallel_init and self.intern_pool:
                self._current_phase = "parallel_init"
                max_retries = 2
                for attempt in range(max_retries):
                    try:
                        await self._run_parallel_initial_verification(
                            claim, max_aspects=self.pool_size,
                        )
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            delay = (2 ** attempt) + random.uniform(0, 0.5)
                            logger.warning(
                                "Parallel initial verification attempt %d/%d failed: %s, "
                                "retrying in %.1fs",
                                attempt + 1, max_retries, e, delay,
                            )
                            self._log(
                                f"[PARALLEL] Attempt {attempt + 1} failed, "
                                f"retrying in {delay:.0f}s...",
                                style="yellow",
                            )
                            await asyncio.sleep(delay)
                        else:
                            logger.error(
                                "Parallel initial verification failed after %d attempts: %s",
                                max_retries, e, exc_info=True,
                            )
                            self._log(
                                f"[PARALLEL] Failed after {max_retries} attempts "
                                f"({type(e).__name__}), continuing sequentially",
                                style="yellow",
                            )

            # Phase 1.5: Search for authoritative fact-checks and primary sources
            if self.intern_pool and not self._pause_requested:
                try:
                    self._log(
                        "[AUTHORITATIVE] Searching for existing fact-checks and primary sources...",
                        style="bold cyan",
                    )
                    authoritative_directives = [
                        VerificationDirective(
                            action="search",
                            topic=f"fact check: {claim}",
                            instructions=(
                                "Focus on finding existing fact-checks from reputable organizations "
                                "(Snopes, PolitiFact, FactCheck.org, Full Fact, etc.). Also search for "
                                "primary sources, official government data, and authoritative records "
                                "that directly address this claim. Prioritize established verdicts "
                                "and original source material."
                            ),
                            priority=10,
                            max_searches=5,
                        ),
                    ]
                    authoritative_result = await self.intern_pool.gather_evidence_parallel(
                        authoritative_directives, session_id
                    )
                    async with self._state_lock:
                        self.all_evidence.extend(authoritative_result.total_evidence)
                        self.all_reports.extend(authoritative_result.reports)

                    # Process authoritative evidence into KG
                    await self.kg_processor.process_evidence(
                        authoritative_result.total_evidence, session_id
                    )

                    self._log(
                        f"[AUTHORITATIVE] Complete: {len(authoritative_result.total_evidence)} "
                        f"evidence items from authoritative source search",
                        style="bold green",
                    )
                except Exception as e:
                    logger.warning("Authoritative source search failed: %s", e, exc_info=True)
                    self._log(
                        f"[AUTHORITATIVE] Search failed ({type(e).__name__}), continuing",
                        style="yellow",
                    )

            # Initialize with the main claim as the first topic (if not enough evidence yet)
            if len(self.all_evidence) < 5:
                initial_topic = await self.db.create_topic(
                    session_id=session_id,
                    topic=claim,
                    depth=0,
                    priority=10,
                )
                async with self._state_lock:
                    self.topics_queue.append(initial_topic)

        context = {
            "claim": claim,
            "session_id": session_id,
        }

        # Phase 2: ReAct loop for deeper verification
        self._current_phase = "react_loop"

        # Load session for periodic checkpoints
        if not session:
            session = await self.db.get_session(session_id)

        result = await self.run(context, resume=resume)

        # Check if we paused
        if result.get("paused"):
            self._current_phase = "react_loop"
            await self.checkpoint_state(session)
            # Return partial report
            return VerdictReport(
                summary="Verification paused. Progress has been saved and can be resumed.",
                verdict="UNVERIFIABLE",
                key_evidence=self.all_evidence[:20],
                sub_claims_explored=[t.topic for t in self.completed_topics],
                sub_claims_remaining=[t.topic for t in self.topics_queue],
                quality_assessment="",
                recommended_next_steps=["Resume verification to continue"],
                time_elapsed_minutes=self._get_elapsed_minutes(),
                iterations_completed=self.state.iteration,
            )

        self._current_phase = "done"

        # Store final summary in external memory
        await self.external_memory.store(
            session_id=self.session_id,
            content=f"Verification completed on: {claim}\nTotal evidence: {len(self.all_evidence)}\nSub-claims explored: {len(self.completed_topics)}",
            memory_type="summary",
            tags=["final", "session_complete"],
            metadata={
                "evidence_count": len(self.all_evidence),
                "topics_count": len(self.completed_topics),
                "time_minutes": self._get_elapsed_minutes(),
            },
        )

        # Return the final report
        if "last_action" in result and result["last_action"].get("action") == "synthesize":
            return result["last_action"]["report"]

        # Generate final verdict report if not already done
        return await self._synthesize_report()

    async def cleanup(self) -> None:
        """Close persistent connections for KG store and external memory.

        Must be called when the Manager is no longer needed (e.g. harness exit).
        """
        try:
            await self.kg_store.close()
        except Exception:
            logger.debug("KG store close failed", exc_info=True)
        try:
            await self.external_memory.close()
        except Exception:
            logger.debug("External memory close failed", exc_info=True)

    async def reset(self, clear_knowledge_graph: bool = False, clear_memory: bool = False) -> None:
        """Reset the manager state.

        Args:
            clear_knowledge_graph: If True, also clear the knowledge graph data
            clear_memory: If True, also clear hybrid memory
        """
        self.claim = ""
        self.session_id = ""
        self.topics_queue = []
        self.completed_topics = []
        self.all_evidence = []
        self.all_reports = []
        self.current_depth = 0
        self.start_time = None
        self.state = type(self.state)()
        self.intern.reset()

        if self.intern_pool:
            self.intern_pool.reset_all()

        if clear_knowledge_graph:
            await self.kg_store.clear()

        if clear_memory:
            self.memory.clear()
            # Also clear the evidence retrieval index for fresh start
            try:
                self.findings_retriever.clear()
            except Exception:
                logger.debug("Failed to clear findings retriever", exc_info=True)

    def search_past_evidence(
        self,
        query: str,
        limit: int = 10,
        min_confidence: float = 0.5,
    ) -> list[dict]:
        """Search past verification sessions for relevant evidence.

        Uses hybrid semantic + lexical search for high-quality retrieval.

        Args:
            query: Search query
            limit: Maximum results
            min_confidence: Minimum confidence threshold

        Returns:
            List of dicts with evidence info and relevance scores
        """
        results = self.findings_retriever.search(
            query=query,
            limit=limit,
            min_confidence=min_confidence,
            use_reranker=True,  # Use reranker for best quality
        )

        return [
            {
                "content": r.finding.content,
                "evidence_type": r.finding.finding_type.value,
                "confidence": r.finding.confidence,
                "source_url": r.finding.source_url,
                "score": r.score,
                "reranked": r.reranked,
            }
            for r in results
        ]

    def get_retrieval_stats(self) -> dict:
        """Get statistics about the hybrid retrieval system."""
        return self.findings_retriever.stats()

    async def get_knowledge_graph_exports(self, output_dir: str = ".") -> dict:
        """Get knowledge graph visualizations and summaries for reports.

        Args:
            output_dir: Directory to save visualizations

        Returns:
            Dict with visualization paths and summary data
        """
        from ..knowledge import KnowledgeGraphVisualizer

        visualizer = KnowledgeGraphVisualizer(self.kg_store)

        exports = {
            "stats": await self.kg_store.get_stats(),
            "key_concepts": self.kg_query.get_key_concepts(10),
            "gaps": [g.to_dict() for g in await self.kg_query.identify_gaps()],
            "contradictions": await self.kg_query.get_contradictions(),
            "mermaid_diagram": visualizer.create_mermaid_diagram(max_nodes=20),
            "stats_card": await visualizer.create_summary_stats_card(),
        }

        return exports
