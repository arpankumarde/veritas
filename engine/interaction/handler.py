"""User interaction handler for fact-checking sessions."""

import asyncio
import json
from collections.abc import Awaitable, Callable

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from ..logging_config import get_logger
from .config import InteractionConfig
from .models import (
    ClarificationQuestion,
    ClarifiedGoal,
    PendingQuestion,
    UserMessage,
)

logger = get_logger(__name__)


class UserInteraction:
    """Handles user interaction during fact-checking sessions.

    Provides three main features:
    1. Pre-verification clarification questions
    2. Async mid-verification questions with timeout
    3. User message queue for injecting guidance anytime
    """

    def __init__(
        self,
        config: InteractionConfig | None = None,
        console: Console | None = None,
        llm_callback: Callable[[str], Awaitable[str]] | None = None,
    ):
        """Initialize the interaction handler.

        Args:
            config: Interaction configuration
            console: Rich console for output
            llm_callback: Async function to call the LLM for generating questions
        """
        self.config = config or InteractionConfig()
        self.console = console or Console()
        self.llm_callback = llm_callback

        # State for async questions
        self._pending_question: PendingQuestion | None = None
        self._response_event = asyncio.Event()
        self._response_value: str | None = None

        # Message queue for user guidance
        self._message_queue: asyncio.Queue[UserMessage] = asyncio.Queue()

        # Track questions asked during session
        self._questions_asked: int = 0

        # Callbacks for pausing/resuming progress display
        self._on_pause: Callable[[], None] | None = None
        self._on_resume: Callable[[], None] | None = None

    def set_progress_callbacks(
        self,
        on_pause: Callable[[], None] | None = None,
        on_resume: Callable[[], None] | None = None,
    ) -> None:
        """Set callbacks for pausing/resuming the progress display.

        Args:
            on_pause: Called when interaction needs to pause the spinner
            on_resume: Called when interaction is done and spinner can resume
        """
        self._on_pause = on_pause
        self._on_resume = on_resume

    async def clarify_claim(self, claim: str) -> ClarifiedGoal:
        """Ask clarification questions before starting fact-checking.

        Args:
            claim: The original claim from the user

        Returns:
            ClarifiedGoal with enriched context from user answers
        """
        if not self.config.enable_clarification or self.config.autonomous_mode:
            return ClarifiedGoal(
                original=claim,
                enriched_context=claim,
                skipped=True,
            )

        # Generate clarification questions using LLM
        questions = await self._generate_clarification_questions(claim)

        if not questions:
            return ClarifiedGoal(
                original=claim,
                enriched_context=claim,
                skipped=True,
            )

        # Display intro
        self.console.print()
        self.console.print(Panel(
            "[bold]Before we begin, a few quick questions to focus the verification:[/bold]\n"
            "[dim]Press Enter to skip any question, or type 's' to skip all and start immediately.[/dim]",
            border_style="cyan",
        ))
        self.console.print()

        # Ask each question
        clarifications: dict[int, str] = {}

        for q in questions[:self.config.max_clarification_questions]:
            answer = await self._ask_clarification(q)

            if answer and answer.lower() == 's':
                # Skip remaining questions
                self.console.print("[dim]Skipping remaining questions...[/dim]")
                break

            if answer:
                clarifications[q.id] = answer

        # Generate enriched context from answers
        enriched = await self._enrich_claim(claim, questions, clarifications)

        return ClarifiedGoal(
            original=claim,
            clarifications=clarifications,
            enriched_context=enriched,
            skipped=False,
        )

    async def _generate_clarification_questions(
        self, claim: str
    ) -> list[ClarificationQuestion]:
        """Generate clarification questions using the LLM."""
        if not self.llm_callback:
            return []

        prompt = (
            "Given this claim to fact-check, generate 2-4 brief clarification "
            "questions that would help focus the verification.\n\n"
            f"Claim: {claim}\n\n"
            "Consider asking about:\n"
            "- Specific context or time period the claim refers to\n"
            "- Which aspect of the claim to prioritize\n"
            "- Known sources or references the user may have\n"
            "- Geographic or domain scope\n\n"
            "Keep questions brief and options concise."
        )

        clarify_schema = {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer"},
                                "question": {"type": "string"},
                                "options": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "category": {"type": "string"},
                            },
                            "required": ["id", "question"],
                        },
                    },
                },
                "required": ["questions"],
            },
        }

        json_prompt = (
            prompt
            + "\n\nRespond ONLY with a JSON object in this exact format, no other text:\n"
            '{"questions": [{"id": 1, "question": "...", "options": ["option1", "option2"], '
            '"category": "scope"}]}'
        )

        try:
            try:
                response = await self.llm_callback(
                    json_prompt, output_format=clarify_schema,
                )
            except TypeError:
                response = await self.llm_callback(json_prompt)

            if isinstance(response, dict):
                data = response.get("questions", [])
            else:
                # Try parsing as full JSON object first
                text = response.strip()
                try:
                    parsed = json.loads(text)
                    data = parsed.get("questions", [])
                except json.JSONDecodeError:
                    # Fallback: extract JSON array from response
                    start = text.find("[")
                    end = text.rfind("]") + 1
                    if start == -1 or end <= start:
                        return []
                    data = json.loads(text[start:end])

            return [ClarificationQuestion(**item) for item in data]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            logger.warning("Clarification question generation failed", exc_info=True)

        return []

    async def _ask_clarification(
        self, question: ClarificationQuestion
    ) -> str | None:
        """Ask a single clarification question."""
        # Format question with options if available
        prompt_text = f"[bold cyan]{question.question}[/bold cyan]"

        if question.options:
            options_text = " / ".join(question.options)
            prompt_text += f"\n[dim]Options: {options_text}[/dim]"

        self.console.print(prompt_text)

        # Get input with timeout
        try:
            loop = asyncio.get_event_loop()
            answer = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: Prompt.ask("[cyan]>[/cyan]", default="")),
                timeout=self.config.clarification_timeout,
            )
            return answer.strip() if answer else None
        except TimeoutError:
            logger.debug("Clarification question timed out")
            self.console.print("[dim]Timeout - skipping question[/dim]")
            return None

    async def _enrich_claim(
        self,
        original_claim: str,
        questions: list[ClarificationQuestion],
        answers: dict[int, str],
    ) -> str:
        """Generate an enriched claim description from clarification answers."""
        if not answers or not self.llm_callback:
            return original_claim

        # Format Q&A pairs
        qa_pairs = []
        for q in questions:
            if q.id in answers:
                qa_pairs.append(f"Q: {q.question}\nA: {answers[q.id]}")

        if not qa_pairs:
            return original_claim

        prompt = f"""Combine the original claim with the user's clarifications into an enriched, focused claim for verification.

Original Claim: {original_claim}

User Clarifications:
{chr(10).join(qa_pairs)}

Write a single enriched claim statement that incorporates these clarifications.
Keep it concise but complete. Do not add any preamble or explanation - just output the enriched claim."""

        try:
            enriched = await self.llm_callback(prompt)
            return enriched.strip()
        except Exception:
            logger.warning("Claim enrichment failed", exc_info=True)
            return original_claim

    async def ask_with_timeout(
        self,
        question: str,
        context: str = "",
        options: list[str] | None = None,
    ) -> str | None:
        """Ask a question during verification with a timeout.

        If the user doesn't respond within the timeout, returns None
        and verification continues autonomously.

        Args:
            question: The question to ask
            context: Optional context about why this question is being asked
            options: Optional list of suggested answers

        Returns:
            User's response, or None if timeout/skipped
        """
        if not self.config.enable_async_questions or self.config.autonomous_mode:
            return None

        if self._questions_asked >= self.config.max_questions_per_session:
            return None

        self._questions_asked += 1

        # Pause the progress spinner if callback is set
        if self._on_pause:
            self._on_pause()

        # Create pending question
        self._pending_question = PendingQuestion(
            text=question,
            context=context,
            options=options or [],
            timeout_seconds=self.config.question_timeout,
        )
        self._response_event.clear()
        self._response_value = None

        # Display question
        self.console.print()
        panel_content = f"[bold]{question}[/bold]"
        if context:
            panel_content += f"\n[dim]{context}[/dim]"
        if options:
            panel_content += f"\n[cyan]Options: {' / '.join(options)}[/cyan]"
        panel_content += f"\n[dim]({self.config.question_timeout}s timeout - verification will continue if no response)[/dim]"

        self.console.print(Panel(
            panel_content,
            title="[yellow]Question[/yellow]",
            border_style="yellow",
        ))

        try:
            # Get input directly with visible typing (since spinner is paused)
            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: Prompt.ask("[yellow]>[/yellow]", default="")
                ),
                timeout=self.config.question_timeout,
            )
            response = response.strip() if response else None
        except TimeoutError:
            self.console.print("[dim]No response - continuing verification...[/dim]")
            response = None

        self._pending_question = None

        # Resume the progress spinner
        if self._on_resume:
            self._on_resume()

        return response

    def respond(self, response: str) -> bool:
        """Handle a user response to a pending question.

        Called by the InputListener when user types during a pending question.

        Args:
            response: The user's response

        Returns:
            True if there was a pending question to respond to
        """
        if self._pending_question:
            self._response_value = response
            self._response_event.set()
            return True
        return False

    def has_pending_question(self) -> bool:
        """Check if there's a pending question waiting for response."""
        return self._pending_question is not None

    def inject_message(self, text: str) -> None:
        """Inject a guidance message into the queue.

        Messages will be picked up by the Manager agent on its next iteration.

        Args:
            text: The guidance message from the user
        """
        if not self.config.enable_message_queue or self.config.autonomous_mode:
            return

        message = UserMessage(content=text)
        self._message_queue.put_nowait(message)
        self.console.print("[bold green]Guidance queued[/bold green] [dim](will be used in next verification iteration)[/dim]")

    def get_pending_messages(self) -> list[UserMessage]:
        """Get all pending messages from the queue (non-blocking).

        Returns:
            List of user messages, empty if none pending
        """
        messages = []
        while True:
            try:
                msg = self._message_queue.get_nowait()
                msg.processed = True
                messages.append(msg)
            except asyncio.QueueEmpty:
                break
        return messages

    def reset(self) -> None:
        """Reset interaction state for a new session."""
        self._pending_question = None
        self._response_event.clear()
        self._response_value = None
        self._questions_asked = 0

        # Clear message queue
        while True:
            try:
                self._message_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
