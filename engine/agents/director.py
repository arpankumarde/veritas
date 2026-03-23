"""Director agent - top-level interface for user interaction."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from ..costs.tracker import get_cost_tracker, reset_cost_tracker
from ..events import emit_synthesis
from ..interaction import InteractionConfig, UserInteraction
from ..logging_config import get_logger
from ..models.evidence import AgentRole, CheckSession, VerdictReport
from ..reports.writer import VerdictReportWriter
from ..storage.database import VeritasDatabase
from .base import AgentConfig, BaseAgent
from .intern import InternAgent
from .manager import ManagerAgent

logger = get_logger(__name__)


class DirectorAgent(BaseAgent):
    """The Director agent is the top-level interface that the user interacts with.

    Responsibilities:
    - Receive and interpret user claims to fact-check
    - Translate user claims into clear verification objectives
    - Manage check sessions (start, pause, resume, stop)
    - Report progress and verdicts to the user
    - Handle time limits and session management
    """

    def __init__(
        self,
        db: VeritasDatabase,
        config: AgentConfig | None = None,
        console: Console | None = None,
        interaction_config: InteractionConfig | None = None,
        owns_db: bool = False,
        max_depth: int = 5,
    ):
        super().__init__(AgentRole.DIRECTOR, db, config, console)
        self._owns_db = owns_db  # Only close db if we own it (not injected by caller)
        self.intern = InternAgent(db, config, console)

        # Set up interaction handler
        self.interaction_config = interaction_config or InteractionConfig()
        self.interaction = UserInteraction(
            config=self.interaction_config,
            console=self.console,
            llm_callback=self._interaction_llm_callback,
        )

        # Pass interaction to manager
        self.manager = ManagerAgent(
            db, self.intern, config, console,
            interaction=self.interaction,
            max_depth=max_depth,
        )
        self.current_session: CheckSession | None = None
        self._progress_task = None
        self._progress: Progress | None = None  # For pause/resume support

        # Wire up progress callbacks to interaction handler
        self.interaction.set_progress_callbacks(
            on_pause=self.pause_progress,
            on_resume=self.resume_progress,
        )

        # Input listener (set later, started after clarification)
        self._input_listener = None

    def set_input_listener(self, listener) -> None:
        """Set the input listener (will be started after clarification)."""
        self._input_listener = listener

    async def _start_input_listener(self) -> None:
        """Start the input listener (called after clarification is done)."""
        if self._input_listener:
            await self._input_listener.start()

    def pause_progress(self) -> None:
        """Pause the progress spinner (for interact mode)."""
        if self._progress:
            self._progress.stop()

    def resume_progress(self) -> None:
        """Resume the progress spinner (after interact mode)."""
        if self._progress:
            self._progress.start()

    async def _interaction_llm_callback(self, prompt: str) -> str:
        """LLM callback for interaction module (uses fast model)."""
        return await self.call_claude(prompt, model_override="haiku")

    @property
    def system_prompt(self) -> str:
        return """You are the Veritas Fact-Check Director. You interface with the user and oversee the entire claim verification operation.

RESPONSIBILITIES:
1. Receive claims from users and clarify ambiguities
2. Set clear verification objectives and success criteria
3. Monitor evidence gathering progress and quality
4. Provide meaningful updates to the user
5. Present final verdicts with supporting evidence

COMMUNICATION STYLE:
- Be professional, neutral, and balanced
- Present evidence from all sides before stating verdicts
- Be transparent about confidence levels and limitations
- Clearly distinguish between verified facts and uncertain claims

VERDICT PRESENTATION:
- Lead with the verdict (True/Mostly True/Mixed/Mostly False/False/Unverifiable)
- Support the verdict with the strongest evidence
- Acknowledge contradicting evidence
- Note confidence levels and any caveats
- Suggest areas for further investigation if needed"""

    async def think(self, context: dict[str, Any]) -> str:
        """Not used for Director - it's event-driven from user input."""
        return ""

    async def act(self, thought: str, context: dict[str, Any]) -> dict[str, Any]:
        """Not used for Director - it's event-driven from user input."""
        return {}

    async def observe(self, action_result: dict[str, Any]) -> str:
        """Not used for Director - it's event-driven from user input."""
        return ""

    def is_done(self, context: dict[str, Any]) -> bool:
        """Director is done when the session ends."""
        return context.get("session_ended", False)

    async def clarify_claim(self, claim: str) -> str:
        """Ask clarification questions before starting verification.

        Args:
            claim: The original claim to verify

        Returns:
            The enriched claim after clarification, or original if skipped
        """
        clarified = await self.interaction.clarify_claim(claim)

        if not clarified.skipped and clarified.clarifications:
            self.console.print()
            self.console.print(Panel(
                f"[bold]Original:[/bold] {clarified.original}\n\n"
                f"[bold]Refined:[/bold] {clarified.enriched_context}",
                title="[green]Claim Refined[/green]",
                border_style="green",
            ))
            self.console.print()

        return clarified.enriched_context

    async def start_verification(
        self,
        claim: str,
        max_iterations: int = 5,
        skip_clarification: bool = False,
        existing_session_id: str | None = None,
        resume: bool = False,
    ) -> VerdictReport:
        """Start a new verification session or resume a paused/crashed one.

        Args:
            claim: The claim to fact-check
            max_iterations: Number of manager ReAct loop iterations
            skip_clarification: Skip pre-verification clarification questions
            existing_session_id: Optional existing session ID to use (for UI/API)
            resume: If True, resume a paused or crashed session

        Returns:
            VerdictReport with evidence and verdict
        """
        # Reset interaction state and cost tracker
        self.interaction.reset()
        reset_cost_tracker()

        if resume and existing_session_id:
            # Resume flow: load session, skip clarification
            self.current_session = await self.db.get_session(existing_session_id)
            if not self.current_session:
                raise ValueError(f"Session {existing_session_id} not found")
            if self.current_session.status not in ("paused", "crashed"):
                raise ValueError(
                    f"Session {existing_session_id} is "
                    f"{self.current_session.status}, not paused/crashed"
                )
            effective_claim = self.current_session.goal
            max_iterations = self.current_session.max_iterations
        else:
            # Clarify claim if enabled (skip when using existing session from UI)
            if not skip_clarification and self.interaction_config.enable_clarification and not existing_session_id:
                effective_claim = await self.clarify_claim(claim)
            else:
                effective_claim = claim

        # Start input listener AFTER clarification is done
        await self._start_input_listener()

        if not resume:
            # Use existing session or create new one
            if existing_session_id:
                # Load existing session from database (created by API)
                self.current_session = await self.db.get_session(existing_session_id)
                if not self.current_session:
                    raise ValueError(f"Session {existing_session_id} not found")
            else:
                # Create new session (CLI flow)
                self.current_session = await self.db.create_session(
                    goal=effective_claim,
                    max_iterations=max_iterations,
                )

        # Set session ID on all agents for WebSocket events
        self.session_id = self.current_session.id
        self.manager.session_id = self.current_session.id
        self.intern.session_id = self.current_session.id

        self._log_header(effective_claim, max_iterations)

        # Run verification with progress display
        try:
            report = await self._run_with_progress(
                effective_claim, max_iterations, resume=resume
            )

            # Check if we paused (don't mark as completed)
            if self.current_session.status == "paused":
                self.console.print("\n[yellow]Verification paused. Progress saved.[/yellow]")
                return report

            # Update session
            self.current_session.status = "completed"
            self.current_session.ended_at = datetime.now()
            self.current_session.total_findings = len(report.key_evidence)
            self.current_session.phase = "done"
            await self.db.update_session(self.current_session)

            # Display results
            await self._display_report(report)

            # Auto-export all outputs
            output_path = await self.export_results()
            self.console.print(f"\n[bold green]Verification saved to: {output_path}/[/bold green]")

            return report

        except asyncio.CancelledError:
            logger.info("Verification cancelled by user: session=%s", self.current_session.id)
            self.console.print("\n[yellow]Verification interrupted by user[/yellow]")
            self.current_session.status = "interrupted"
            self.current_session.ended_at = datetime.now()
            await self.db.update_session(self.current_session)
            raise

        except Exception as e:
            logger.error(
                "Verification failed: session=%s, error=%s",
                self.current_session.id, e, exc_info=True,
            )
            self.console.print(f"\n[red]Error during verification: {e}[/red]")
            self.current_session.status = "error"
            self.current_session.ended_at = datetime.now()
            await self.db.update_session(self.current_session)
            raise

        finally:
            # Only close database if we own it (not injected by caller like VeritasHarness)
            if self._owns_db and self.db:
                await self.db.close()

    def pause_verification(self) -> None:
        """Request the verification to pause gracefully (saves state for resume)."""
        self.manager.pause()
        self.intern.pause()
        if self.manager.intern_pool:
            self.manager.intern_pool.pause()
        self._log("Pause requested - finishing current operation")

    async def _run_with_progress(
        self, claim: str, max_iterations: int, resume: bool = False
    ) -> VerdictReport:
        """Run verification with progress display."""
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=self.console,
            transient=False,
        )

        with self._progress as progress:
            label = "Resuming" if resume else "Verifying"
            task = progress.add_task(
                f"[cyan]{label}: {claim[:50]}...",
                total=None,
            )

            # Add callback to update progress
            def update_progress(agent, ctx):
                iteration = ctx.get("iteration", 0)
                evidence_count = len(self.manager.all_evidence)
                progress.update(
                    task,
                    description=f"[cyan]Iteration {iteration} | Evidence: {evidence_count}[/cyan]",
                )

            self.manager.add_callback(update_progress)

            # Run verification
            report = await self.manager.run_verification(
                claim=claim,
                session_id=self.current_session.id,
                max_iterations=max_iterations,
                resume=resume,
                session=self.current_session if resume else None,
            )

            # Check if paused
            if self.manager._pause_requested:
                progress.update(task, description="[yellow]Verification paused")
                # Refresh session from DB (checkpoint_state updated it)
                self.current_session = await self.db.get_session(self.current_session.id)
            else:
                progress.update(task, description="[green]Verification complete!")
            self._progress = None

            return report

    def _log_header(self, claim: str, max_iterations: int) -> None:
        """Log verification session header."""
        self.console.print()
        self.console.print(Panel(
            f"[bold]Claim:[/bold] {claim}\n"
            f"[bold]Iterations:[/bold] {max_iterations}\n"
            f"[bold]Session ID:[/bold] {self.current_session.id}",
            title="[bold blue]Veritas Fact Checker[/bold blue]",
            border_style="blue",
        ))
        self.console.print()

    async def _display_report(self, report: VerdictReport) -> None:
        """Display the final verdict report."""
        # Verdict display with color coding
        verdict_colors = {
            "TRUE": "bold green",
            "MOSTLY_TRUE": "green",
            "MIXED": "yellow",
            "MOSTLY_FALSE": "red",
            "FALSE": "bold red",
            "UNVERIFIABLE": "dim",
        }
        verdict_color = verdict_colors.get(report.verdict, "white")

        self.console.print()
        self.console.print(Panel(
            f"[{verdict_color}]VERDICT: {report.verdict}[/{verdict_color}]\n\n"
            f"{report.summary}",
            title="[bold green]Verification Verdict[/bold green]",
            border_style="green",
        ))

        # Evidence table
        if report.key_evidence:
            table = Table(title="Key Evidence", show_header=True, header_style="bold")
            table.add_column("Type", style="cyan", width=12)
            table.add_column("Evidence", style="white")
            table.add_column("Confidence", style="yellow", width=10)

            for evidence in report.key_evidence[:15]:
                table.add_row(
                    evidence.evidence_type.value.upper(),
                    evidence.content[:100] + "..." if len(evidence.content) > 100 else evidence.content,
                    f"{evidence.confidence:.0%}",
                )

            self.console.print(table)

        # Stats
        stats = await self.db.get_session_stats(self.current_session.id)
        self.console.print()
        self.console.print(Panel(
            f"[bold]Sub-Claims Explored:[/bold] {len(report.sub_claims_explored)}\n"
            f"[bold]Total Evidence:[/bold] {stats.get('total_evidence', 0)}\n"
            f"[bold]Unique Searches:[/bold] {stats.get('unique_searches', 0)}\n"
            f"[bold]Max Depth:[/bold] {stats.get('max_depth', 0)}\n"
            f"[bold]Time Used:[/bold] {report.time_elapsed_minutes:.1f} minutes\n"
            f"[bold]Iterations:[/bold] {report.iterations_completed}",
            title="[bold]Session Statistics[/bold]",
            border_style="dim",
        ))

        # Remaining sub-claims
        if report.sub_claims_remaining:
            self.console.print()
            self.console.print("[bold]Sub-claims for further investigation:[/bold]")
            for sc in report.sub_claims_remaining[:5]:
                self.console.print(f"  - {sc}")

        # Display cost summary
        self._display_costs()

    def _display_costs(self) -> None:
        """Display API cost summary."""
        cost_summary = get_cost_tracker().get_summary()

        # Build cost table
        table = Table(title="API Cost Estimate", show_header=True, header_style="bold")
        table.add_column("Model", style="cyan", width=10)
        table.add_column("Calls", justify="right", width=6)
        table.add_column("Input", justify="right", width=10)
        table.add_column("Output", justify="right", width=10)
        table.add_column("Thinking", justify="right", width=10)
        table.add_column("Cost", justify="right", style="green", width=10)

        # Add rows for each model (only if used)
        if cost_summary.sonnet_usage.calls > 0:
            table.add_row(
                "Sonnet 4.5",
                str(cost_summary.sonnet_usage.calls),
                f"{cost_summary.sonnet_usage.input_tokens:,}",
                f"{cost_summary.sonnet_usage.output_tokens:,}",
                f"{cost_summary.sonnet_usage.thinking_tokens:,}",
                f"${cost_summary.sonnet_cost:.4f}",
            )

        if cost_summary.opus_usage.calls > 0:
            table.add_row(
                "Opus 4.5",
                str(cost_summary.opus_usage.calls),
                f"{cost_summary.opus_usage.input_tokens:,}",
                f"{cost_summary.opus_usage.output_tokens:,}",
                f"{cost_summary.opus_usage.thinking_tokens:,}",
                f"${cost_summary.opus_cost:.4f}",
            )

        if cost_summary.haiku_usage.calls > 0:
            table.add_row(
                "Haiku 4.5",
                str(cost_summary.haiku_usage.calls),
                f"{cost_summary.haiku_usage.input_tokens:,}",
                f"{cost_summary.haiku_usage.output_tokens:,}",
                f"{cost_summary.haiku_usage.thinking_tokens:,}",
                f"${cost_summary.haiku_cost:.4f}",
            )

        # Add web searches row if any
        if cost_summary.web_searches > 0 or cost_summary.web_fetches > 0:
            table.add_row(
                "Web",
                f"{cost_summary.web_searches + cost_summary.web_fetches}",
                f"{cost_summary.web_searches} srch",
                f"{cost_summary.web_fetches} fetch",
                "-",
                f"${cost_summary.search_cost:.4f}",
            )

        # Add total row
        table.add_row(
            "[bold]TOTAL[/bold]",
            f"[bold]{cost_summary.total_calls}[/bold]",
            f"[bold]{cost_summary.total_input_tokens:,}[/bold]",
            f"[bold]{cost_summary.total_output_tokens - cost_summary.total_thinking_tokens:,}[/bold]",
            f"[bold]{cost_summary.total_thinking_tokens:,}[/bold]",
            f"[bold]${cost_summary.total_cost:.4f}[/bold]",
            style="bold",
        )

        self.console.print()
        self.console.print(table)
        self.console.print("[dim]Pricing: Opus $5/$25, Sonnet $3/$15, Haiku $1/$5 per MTok (input/output). Search: $0.01/search[/dim]")
        self.console.print("[dim]Note: Token counts are estimates (~4 chars/token). Thinking tokens billed as output.[/dim]")

    async def get_session_evidence(self) -> list:
        """Get all evidence from the current session."""
        if not self.current_session:
            return []
        return await self.db.get_session_findings(self.current_session.id)

    async def export_results(self) -> str:
        """Export all verification outputs to a dedicated folder.

        Creates: output/{slug}_{session_id}/
            - report.md       - Narrative report
            - evidence.json   - Structured data

        Returns:
            Path to the output directory
        """
        import json

        if not self.current_session:
            raise ValueError("No active session")

        evidence = await self.get_session_evidence()

        # Create output directory: output/{slug}_{session_id}/
        slug = self.current_session.slug or "verification"
        session_id = self.current_session.id
        output_dir = Path("output") / f"{slug}_{session_id}"
        output_dir.mkdir(parents=True, exist_ok=True)

        self.console.print(f"\n[bold cyan]Exporting verification to: {output_dir}/[/bold cyan]")

        # 1. Export JSON data
        json_output = {
            "session": {
                "id": self.current_session.id,
                "claim": self.current_session.goal,
                "slug": self.current_session.slug,
                "started_at": self.current_session.started_at.isoformat(),
                "ended_at": self.current_session.ended_at.isoformat() if self.current_session.ended_at else None,
                "status": self.current_session.status,
            },
            "evidence": [
                {
                    "content": e.content,
                    "type": e.evidence_type.value,
                    "source_url": e.source_url,
                    "confidence": e.confidence,
                    "search_query": e.search_query,
                }
                for e in evidence
            ],
            "sub_claims_explored": [t.topic for t in self.manager.completed_topics] if self.manager.completed_topics else [],
            "sub_claims_remaining": [t.topic for t in self.manager.topics_queue] if self.manager.topics_queue else [],
            "costs": get_cost_tracker().get_summary().to_dict(),
        }
        json_file = output_dir / "evidence.json"
        json_file.write_text(json.dumps(json_output, indent=2))
        self.console.print("  [dim]Saved evidence.json[/dim]")

        # 2. Export knowledge graph
        try:
            kg_exports = await self.manager.get_knowledge_graph_exports(str(output_dir))
            self.console.print(f"  [dim]Knowledge graph: {kg_exports.get('stats', {}).get('num_entities', 0)} entities, {kg_exports.get('stats', {}).get('num_relations', 0)} relations[/dim]")
        except Exception as e:
            logger.warning("Knowledge graph export failed: %s", e, exc_info=True)
            self.console.print(f"  [dim]Knowledge graph export skipped: {e}[/dim]")
            kg_exports = None

        # 3. Generate markdown report
        self.console.print("[bold cyan]Generating verification report...[/bold cyan]")
        self.console.print("[dim]This uses extended thinking to synthesize evidence into a narrative report.[/dim]\n")

        # Emit synthesis start event
        if self.session_id:
            await emit_synthesis(
                session_id=self.session_id,
                agent="director",
                message=f"Synthesizing {len(evidence)} evidence items into report...",
                progress=0
            )

        writer = VerdictReportWriter(model="opus")

        async def report_progress(message: str, progress: int) -> None:
            if self.session_id:
                await emit_synthesis(
                    session_id=self.session_id,
                    agent="director",
                    message=message,
                    progress=progress
                )

        sub_claims_explored = [t.topic for t in self.manager.completed_topics] if self.manager.completed_topics else []
        sub_claims_remaining = [t.topic for t in self.manager.topics_queue] if self.manager.topics_queue else []

        report = await writer.generate_report(
            session=self.current_session,
            findings=evidence,
            topics_explored=sub_claims_explored,
            topics_remaining=sub_claims_remaining,
            kg_exports=kg_exports,
            progress_callback=report_progress,
        )

        # Emit synthesis complete event
        if self.session_id:
            await emit_synthesis(
                session_id=self.session_id,
                agent="director",
                message="Report synthesis complete",
                progress=100
            )

        md_file = output_dir / "report.md"
        md_file.write_text(report)
        self.console.print("  [dim]Saved report.md[/dim]")

        return str(output_dir)

    def stop_verification(self) -> None:
        """Request the verification to stop gracefully."""
        self.manager.stop()
        self.intern.stop()
        self._log("Stop requested - finishing current operation")

    # Backward-compat alias
    def pause_research(self) -> None:
        """Alias for pause_verification (backward compat)."""
        return self.pause_verification()


class VeritasHarness:
    """Main harness for running fact-checking sessions.

    This is the primary entry point for running the hierarchical verification system.
    """

    def __init__(
        self,
        db_path: str = "veritas.db",
        interaction_config: InteractionConfig | None = None,
        max_depth: int = 5,
    ):
        self.db_path = db_path
        self.db: VeritasDatabase | None = None
        self.director: DirectorAgent | None = None
        self.console = Console()
        self.interaction_config = interaction_config
        self.max_depth = max_depth

    async def __aenter__(self):
        """Async context manager entry."""
        self.db = VeritasDatabase(self.db_path)
        await self.db.connect()
        self.director = DirectorAgent(
            self.db,
            console=self.console,
            interaction_config=self.interaction_config,
            max_depth=self.max_depth,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.director:
            await self.director.manager.cleanup()
        if self.db:
            await self.db.close()

    async def verify(
        self,
        claim: str,
        max_iterations: int = 5,
        existing_session_id: str | None = None,
        resume: bool = False,
    ) -> VerdictReport:
        """Run a fact-checking session.

        Args:
            claim: The claim to fact-check
            max_iterations: Number of manager ReAct loop iterations (default: 5)
            existing_session_id: Optional existing session ID (for UI/API)
            resume: If True, resume a paused/crashed session

        Returns:
            VerdictReport with evidence and verdict
        """
        if not self.director:
            raise RuntimeError("Harness not initialized - use async with")

        return await self.director.start_verification(
            claim,
            max_iterations,
            existing_session_id=existing_session_id,
            resume=resume,
        )

    # Backward-compat alias
    async def check(self, claim: str, max_iterations: int = 5, **kwargs) -> "VerdictReport":
        """Alias for verify (backward compat)."""
        return await self.verify(claim, max_iterations, **kwargs)

    def stop(self) -> None:
        """Stop the current verification session."""
        if self.director:
            self.director.stop_verification()
