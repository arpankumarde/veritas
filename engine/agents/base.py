"""Base agent class with ReAct loop implementation using Claude Agent SDK."""

import asyncio
import os
import random
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    query,
)
from rich.console import Console

from ..audit import DecisionType, get_decision_logger
from ..costs.tracker import get_cost_tracker
from ..events import emit_action, emit_agent_event, emit_error, emit_thinking
from ..logging_config import get_logger
from ..models.findings import AgentMessage, AgentRole
from ..storage.database import VeritasDatabase

logger = get_logger(__name__)


def _get_api_key() -> str | None:
    """Get the API key from Claude Code's config or environment."""
    # First check environment
    if api_key := os.environ.get("ANTHROPIC_API_KEY"):
        return api_key

    # Try Claude Code's get-api-key.sh script
    script_path = Path.home() / ".claude" / "get-api-key.sh"
    if script_path.exists():
        try:
            result = subprocess.run(
                ["bash", str(script_path)],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass

    return None


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    model: str = "sonnet"  # sonnet, opus, haiku
    max_turns: int = 10
    max_iterations: int = 100
    allowed_tools: list[str] = field(default_factory=lambda: ["WebSearch", "WebFetch"])
    max_thinking_tokens: int = 10000  # Extended thinking for deep reasoning


class ModelRouter:
    """Routes tasks to appropriate models based on complexity.

    Model selection strategy:
    - Haiku: Quick classification, simple extraction, yes/no decisions
    - Sonnet: Web search analysis, evidence extraction, moderate reasoning
    - Opus: Strategic planning, deep synthesis, complex reasoning
    """

    # Task to model mapping
    TASK_MODELS = {
        # Haiku tasks (fast, cheap)
        'classify': 'haiku',
        'extract_simple': 'haiku',
        'yes_no': 'haiku',
        'format': 'haiku',

        # Sonnet tasks (balanced)
        'search': 'sonnet',
        'extract_evidence': 'sonnet',
        'summarize': 'sonnet',
        'analyze_results': 'sonnet',
        'query_expansion': 'sonnet',

        # Opus tasks (deep reasoning)
        'strategic_planning': 'opus',
        'synthesis': 'opus',
        'critique': 'opus',
        'deep_analysis': 'opus',
        'verdict_writing': 'opus',
    }

    @classmethod
    def get_model_for_task(cls, task: str, default: str = 'sonnet') -> str:
        """Get the appropriate model for a task.

        Args:
            task: The task type
            default: Default model if task not found

        Returns:
            Model name (haiku, sonnet, or opus)
        """
        return cls.TASK_MODELS.get(task, default)

    @classmethod
    def should_use_thinking(cls, task: str) -> bool:
        """Determine if a task should use extended thinking.

        Extended thinking is best for:
        - Complex synthesis
        - Strategic planning
        - Deep analysis
        """
        thinking_tasks = {
            'strategic_planning',
            'synthesis',
            'critique',
            'deep_analysis',
            'verdict_writing',
        }
        return task in thinking_tasks


@dataclass
class AgentState:
    """Current state of an agent."""
    iteration: int = 0
    total_actions: int = 0
    last_action: str | None = None
    last_observation: str | None = None
    is_complete: bool = False
    error: str | None = None
    history: list[dict] = field(default_factory=list)


class BaseAgent(ABC):
    """Base class for all fact-checking agents implementing ReAct loop.

    ReAct = Reason + Act
    Uses Claude Agent SDK for all LLM calls.
    """

    def __init__(
        self,
        role: AgentRole,
        db: VeritasDatabase,
        config: AgentConfig | None = None,
        console: Console | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
    ):
        self.role = role
        self.agent_id = agent_id or role.value  # Unique identifier (e.g., "intern_0", "intern_1")
        self.db = db
        self.config = config or AgentConfig()
        self.console = console or Console()
        self.state = AgentState()
        self._stop_requested = False
        self._pause_requested = False
        self._callbacks: list[Callable] = []
        self.session_id = session_id  # For WebSocket events
        self._background_tasks: set[asyncio.Task] = set()  # Track fire-and-forget tasks
        self._consecutive_errors = 0
        self._max_consecutive_errors = 3  # Break after 3 consecutive failures

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System prompt defining the agent's role and behavior."""
        pass

    @abstractmethod
    async def think(self, context: dict[str, Any]) -> str:
        """Reason about the current state and decide what to do next."""
        pass

    @abstractmethod
    async def act(self, thought: str, context: dict[str, Any]) -> dict[str, Any]:
        """Execute an action based on the thought."""
        pass

    @abstractmethod
    async def observe(self, action_result: dict[str, Any]) -> str:
        """Process the result of an action."""
        pass

    @abstractmethod
    def is_done(self, context: dict[str, Any]) -> bool:
        """Check if the agent has completed its task."""
        pass

    async def run(self, initial_context: dict[str, Any], resume: bool = False) -> dict[str, Any]:
        """Run the ReAct loop until completion or max iterations.

        Args:
            initial_context: Context dict for the loop
            resume: If True, skip state reset (preserves iteration count etc.)
        """
        context = initial_context.copy()
        if not resume:
            self.state = AgentState()

        self._log(f"{'Resuming' if resume else 'Starting'} {self.role.value} agent")

        while (
            not self.is_done(context)
            and not self._stop_requested
            and not self._pause_requested
            and self.state.iteration < self.config.max_iterations
        ):
            self.state.iteration += 1
            logger.debug("ReAct iteration %d started for %s", self.state.iteration, self.role.value)
            self._log(f"Iteration {self.state.iteration}", style="dim")

            try:
                # Think
                thought = await self.think(context)
                self._log(f"[Think] {thought}")
                self.state.history.append({"type": "thought", "content": thought})

                # Emit thinking event
                if self.session_id:
                    await emit_thinking(
                        session_id=self.session_id,
                        agent=self.role.value,
                        thought=thought[:500]  # Truncate for display
                    )

                if self._stop_requested or self._pause_requested:
                    break

                # Act
                action_result = await self.act(thought, context)
                self.state.total_actions += 1
                self.state.last_action = str(action_result.get("action", "unknown"))
                self._log(f"[Act] {self.state.last_action}")
                self.state.history.append({"type": "action", "content": action_result})

                # Emit action event with search details when available
                if self.session_id:
                    action_details: dict[str, Any] = {"iteration": self.state.iteration}
                    if action_result.get("action") == "search":
                        action_details["query"] = str(action_result.get("query", ""))
                        action_details["queries_used"] = [
                            str(q) for q in action_result.get("queries_used", [])
                        ]
                        action_details["results_count"] = action_result.get("results_count", 0)
                        action_details["evidence_extracted"] = action_result.get("evidence_extracted", 0)
                        # Include top result titles (safe serialization)
                        results = action_result.get("results") or []
                        titles = []
                        for r in results[:5]:
                            try:
                                titles.append({
                                    "title": str(getattr(r, "title", "")),
                                    "url": str(getattr(r, "url", "")),
                                })
                            except Exception:
                                pass
                        if titles:
                            action_details["result_titles"] = titles
                    try:
                        await emit_action(
                            session_id=self.session_id,
                            agent=self.role.value,
                            action=self.state.last_action,
                            details=action_details,
                        )
                    except Exception as e:
                        logger.warning("Failed to emit action event: %s", e)

                if self._stop_requested or self._pause_requested:
                    break

                # Observe
                observation = await self.observe(action_result)
                self.state.last_observation = observation
                self._log(f"[Observe] {observation}")
                self.state.history.append({"type": "observation", "content": observation})

                # Update context
                context["last_thought"] = thought
                context["last_action"] = action_result
                context["last_observation"] = observation
                context["iteration"] = self.state.iteration

                # Fire callbacks
                for callback in self._callbacks:
                    await self._safe_callback(callback, context)

                # Reset consecutive error counter on success
                self._consecutive_errors = 0

            except Exception as e:
                self._consecutive_errors += 1
                logger.error("ReAct loop error in %s iteration %d: %s", self.role.value, self.state.iteration, e, exc_info=True)
                self.state.error = str(e)
                recoverable = self._consecutive_errors < self._max_consecutive_errors
                attempts = self._consecutive_errors
                max_err = self._max_consecutive_errors
                self._log(
                    f"[Error] {e} (attempt {attempts}/{max_err})",
                    style="red",
                )

                # Emit error event
                if self.session_id:
                    await emit_error(
                        session_id=self.session_id,
                        agent=self.role.value,
                        error=str(e),
                        recoverable=recoverable,
                    )

                if not recoverable:
                    self._log("Too many consecutive errors, stopping agent", style="red")
                    logger.critical("Agent %s stopped after %d consecutive errors", self.role.value, self._max_consecutive_errors)
                    break

                # Exponential backoff before retry
                backoff = (2 ** self._consecutive_errors) + random.uniform(0, 0.5)
                self._log(f"Retrying in {backoff:.1f}s...", style="yellow")
                await asyncio.sleep(backoff)

        if self._pause_requested:
            context["paused"] = True
            self._log("Agent paused")
        else:
            self.state.is_complete = self.is_done(context)
        status = "paused" if self._pause_requested else "completed"
        self._log(
            f"Agent {status}: iterations={self.state.iteration}, "
            f"actions={self.state.total_actions}"
        )
        logger.info("Agent %s finished: iterations=%d, actions=%d, error=%s", self.role.value, self.state.iteration, self.state.total_actions, self.state.error)

        return context

    def stop(self) -> None:
        """Request the agent to stop."""
        self._stop_requested = True
        self._log("Stop requested")

    def pause(self) -> None:
        """Request the agent to pause after the current iteration."""
        self._pause_requested = True
        self._log("Pause requested")

    def add_callback(self, callback: Callable) -> None:
        """Add a callback to be called after each iteration."""
        self._callbacks.append(callback)

    async def _safe_callback(self, callback: Callable, context: dict) -> None:
        """Safely execute a callback."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(self, context)
            else:
                callback(self, context)
        except Exception as e:
            self._log(f"Callback error: {e}", style="yellow")

    async def send_message(
        self,
        to_agent: AgentRole,
        message_type: str,
        content: str,
        session_id: str,
        metadata: dict | None = None,
    ) -> AgentMessage:
        """Send a message to another agent."""
        message = AgentMessage(
            session_id=session_id,
            from_agent=self.role,
            to_agent=to_agent,
            message_type=message_type,
            content=content,
            metadata=metadata or {},
        )
        return await self.db.save_message(message)

    @staticmethod
    def _build_env() -> dict[str, str]:
        """Build environment dict with API key for LLM calls."""
        env: dict[str, str] = {}
        if api_key := _get_api_key():
            env["ANTHROPIC_API_KEY"] = api_key
        return env

    def _track_llm_cost(self, model: str, prompt: str, response_text: str) -> None:
        """Track cost for an LLM call."""
        tracker = get_cost_tracker()
        tracker.track_call(
            model=model,
            input_text=prompt,
            output_text=response_text,
            system_prompt=self.system_prompt,
        )

    async def call_claude(
        self,
        prompt: str,
        tools: list[str] | None = None,
        use_thinking: bool = False,
        task_type: str | None = None,
        output_format: dict | None = None,
        model_override: str | None = None,
        system_prompt_override: str | None = None,
    ) -> str | dict | list | None:
        """Call Claude via the Agent SDK.

        Args:
            prompt: The prompt to send
            tools: Optional list of tools to enable (default: none for pure reasoning)
            use_thinking: Enable extended thinking for deep reasoning (Opus recommended)
            task_type: Optional task type for automatic model routing
            output_format: Optional JSON schema for structured output.
                Example: {"type": "json_schema", "schema": {"type": "object", ...}}
                When provided, returns parsed structured data instead of text.
            model_override: Explicit model to use, bypassing config and task routing.
            system_prompt_override: Override the agent's system prompt for this call.

        Returns:
            Text response (str), or parsed structured data (dict/list) if
            output_format is provided.
        """
        env = self._build_env()

        # Model selection priority: explicit override > task routing > config
        model = model_override or self.config.model
        if not model_override and task_type:
            model = ModelRouter.get_model_for_task(task_type, default=self.config.model)
            # Also check if we should use thinking for this task
            if not use_thinking and ModelRouter.should_use_thinking(task_type):
                use_thinking = True

        # Structured output needs 2 turns: one for the StructuredOutput
        # tool call, one for the tool result acknowledgement.
        max_turns = 2 if output_format else 1

        options = ClaudeAgentOptions(
            model=model,
            max_turns=max_turns,
            allowed_tools=tools or [],
            system_prompt=system_prompt_override or self.system_prompt,
            env=env,
            max_thinking_tokens=self.config.max_thinking_tokens if use_thinking else None,
            output_format=output_format,
        )

        response_text = ""
        structured_output = None

        # 5-minute timeout prevents indefinite hangs on stalled LLM calls
        async def _run_query():
            nonlocal response_text, structured_output
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            response_text += block.text
                        elif (
                            isinstance(block, ToolUseBlock)
                            and block.name == "StructuredOutput"
                        ):
                            structured_output = block.input
                elif isinstance(message, ResultMessage):
                    if message.structured_output is not None:
                        structured_output = message.structured_output

        try:
            await asyncio.wait_for(_run_query(), timeout=300)
        except asyncio.TimeoutError:
            self._log("LLM call timed out after 5 minutes", style="red")
            logger.error(
                "LLM call timed out after 5 minutes, model=%s, prompt_len=%d",
                model, len(prompt),
            )
            raise
        except Exception as e:
            self._log(f"Error calling Claude: {e}", style="red")
            logger.error(
                "LLM call failed: %s, model=%s, prompt_len=%d",
                e, model, len(prompt), exc_info=True,
            )
            raise

        self._track_llm_cost(model, prompt, response_text)

        # Return structured output if available and requested
        if output_format and structured_output is not None:
            logger.debug("LLM call OK: model=%s, response_len=%d", model, len(response_text))
            return structured_output

        logger.debug("LLM call OK: model=%s, response_len=%d", model, len(response_text))
        return response_text

    async def call_claude_with_tools(
        self,
        prompt: str,
        tools: list[str],
    ) -> tuple[str, list[dict]]:
        """Call Claude with tools enabled via the Agent SDK.

        Args:
            prompt: The prompt to send
            tools: List of tools to enable (e.g., ["WebSearch", "WebFetch"])

        Returns:
            Tuple of (text_response, tool_results)
        """
        from claude_agent_sdk import ToolResultBlock, ToolUseBlock

        env = self._build_env()

        options = ClaudeAgentOptions(
            model=self.config.model,
            max_turns=self.config.max_turns,
            allowed_tools=tools,
            system_prompt=self.system_prompt,
            env=env,
        )

        response_text = ""
        tool_results = []

        # 10-minute timeout for tool-use calls (tools like WebSearch take longer)
        async def _run_tool_query():
            nonlocal response_text
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            response_text += block.text
                        elif isinstance(block, ToolUseBlock):
                            tool_results.append({
                                "tool": block.name,
                                "id": block.id,
                                "input": block.input,
                            })
                        elif isinstance(block, ToolResultBlock):
                            for tr in tool_results:
                                if tr.get("id") == block.tool_use_id:
                                    tr["result"] = block.content
                                    tr["is_error"] = block.is_error

        try:
            await asyncio.wait_for(_run_tool_query(), timeout=600)
        except asyncio.TimeoutError:
            self._log("LLM tool call timed out after 10 minutes", style="red")
            logger.error(
                "LLM tool call timed out after 10 minutes, model=%s",
                self.config.model,
            )
            raise
        except Exception as e:
            self._log(f"Error calling Claude with tools: {e}", style="red")
            logger.error(
                "LLM tool call failed: %s, model=%s",
                e, self.config.model, exc_info=True,
            )
            raise

        self._track_llm_cost(self.config.model, prompt, response_text)

        # Track web searches and fetches
        cost_tracker = get_cost_tracker()
        for tr in tool_results:
            tool_name = tr.get("tool", "").lower()
            if "websearch" in tool_name or "web_search" in tool_name:
                cost_tracker.track_web_search()
            elif "webfetch" in tool_name or "web_fetch" in tool_name:
                cost_tracker.track_web_fetch()

        return response_text, tool_results

    def _log(self, message: str, style: str | None = None) -> None:
        """Log a message to the console."""
        logger.debug("[%s] %s", self.role.value.upper(), message)
        prefix = f"[{self.role.value.upper()}]"
        if style:
            self.console.print(f"{prefix} {message}", style=style)
        else:
            self.console.print(f"{prefix} {message}")

        # Optionally mirror logs to the UI as "system" events for verbosity
        if (
            self.session_id
            and os.environ.get("VERITAS_DISABLE_LOG_EVENTS") != "1"
        ):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            task = loop.create_task(
                emit_agent_event(
                    session_id=self.session_id,
                    event_type="system",
                    agent=self.role.value,
                    data={"message": message},
                )
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

    async def _log_decision(
        self,
        session_id: str,
        decision_type: DecisionType,
        decision_outcome: str,
        reasoning: str | None = None,
        inputs: dict | None = None,
        metrics: dict | None = None,
    ) -> None:
        """Log an agent decision for audit trail.

        This is a fire-and-forget async operation that won't block
        the main agent loop. Decisions are batched and written to DB
        in the background.

        Args:
            session_id: Current check session ID
            decision_type: Type of decision (from DecisionType enum)
            decision_outcome: The outcome/choice made
            reasoning: Agent's reasoning (truncated to 500 chars in DB)
            inputs: Key inputs that influenced the decision
            metrics: Metrics at decision time (e.g., time_remaining, evidence_count)
        """
        logger = get_decision_logger(self.db)
        if logger is None:
            return

        try:
            await logger.log_decision(
                session_id=session_id,
                agent_role=self.agent_id,  # Use unique agent_id instead of role
                decision_type=decision_type,
                decision_outcome=decision_outcome,
                reasoning=reasoning,
                inputs=inputs,
                metrics=metrics,
                iteration=self.state.iteration,
            )
        except Exception:
            # Don't let logging errors affect agent operation
            pass
