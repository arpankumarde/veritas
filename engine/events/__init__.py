"""
Event emission helpers for agents to send real-time updates to WebSocket clients.

Uses a registry pattern to decouple the agent layer from the API layer.
The API server registers its emitter at startup via register_emitter().
When running in CLI mode (no API server), events are proxied over HTTP.
"""

import asyncio
import os
from collections.abc import Callable, Coroutine
from typing import Any

from ..logging_config import get_logger

logger = get_logger(__name__)

EmitterCallback = Callable[[str, str, str, dict[str, Any]], Coroutine[Any, Any, None]]
SubscriberCountCallback = Callable[[str], int]

_registered_emitter: EmitterCallback | None = None
_registered_subscriber_count: SubscriberCountCallback | None = None


def register_emitter(
    emitter: EmitterCallback,
    subscriber_count: SubscriberCountCallback | None = None,
) -> None:
    """Register the event emitter callback."""
    global _registered_emitter, _registered_subscriber_count
    _registered_emitter = emitter
    _registered_subscriber_count = subscriber_count
    logger.info("Event emitter registered")


def _should_proxy_event(local_subscribers: int | None) -> bool:
    """Decide whether to forward events to the API server."""
    if os.environ.get("VERITAS_DISABLE_EVENT_PROXY") == "1":
        return False
    if os.environ.get("VERITAS_IN_API") == "1":
        return False
    if local_subscribers is not None and local_subscribers > 0:
        return False
    return True


async def _emit_remote_event(
    session_id: str, event_type: str, agent: str, data: dict[str, Any],
) -> None:
    """Forward event to the API server for WebSocket broadcasting."""
    base_url = os.environ.get("VERITAS_API_URL", "http://localhost:9090").rstrip("/")
    if not base_url:
        return
    try:
        import httpx
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.post(
                f"{base_url}/api/events/emit",
                json={
                    "session_id": session_id,
                    "event_type": event_type,
                    "agent": agent,
                    "data": data,
                },
            )
    except Exception:
        logger.debug("Remote event emission failed", exc_info=True)


async def emit_agent_event(
    session_id: str, event_type: str, agent: str, data: dict[str, Any],
) -> None:
    """Emit an event from an agent to all WebSocket subscribers."""
    local_subscribers: int | None = None

    if _registered_emitter is not None:
        try:
            try:
                asyncio.create_task(
                    _registered_emitter(session_id, event_type, agent, data)
                )
            except RuntimeError:
                await _registered_emitter(session_id, event_type, agent, data)

            if _registered_subscriber_count is not None:
                local_subscribers = _registered_subscriber_count(session_id)
        except Exception as e:
            logger.warning("Event emission error: %s", e, exc_info=True)

    if _should_proxy_event(local_subscribers):
        try:
            asyncio.create_task(
                _emit_remote_event(session_id, event_type, agent, data)
            )
        except RuntimeError:
            await _emit_remote_event(session_id, event_type, agent, data)


async def emit_thinking(session_id: str, agent: str, thought: str) -> None:
    await emit_agent_event(session_id, "thinking", agent, {"thought": thought})


async def emit_action(
    session_id: str, agent: str, action: str, details: dict[str, Any] | None = None,
) -> None:
    data: dict[str, Any] = {"action": action}
    if details:
        data.update(details)
    await emit_agent_event(session_id, "action", agent, data)


async def emit_evidence(
    session_id: str, agent: str, content: str,
    source: str | None = None, confidence: float | None = None,
) -> None:
    """Emit an evidence discovery event."""
    data: dict[str, Any] = {"content": content}
    if source:
        data["source"] = source
    if confidence is not None:
        data["confidence"] = confidence
    await emit_agent_event(session_id, "evidence", agent, data)


# Keep backward-compat alias
emit_finding = emit_evidence


async def emit_synthesis(
    session_id: str, agent: str, message: str, progress: int | None = None,
) -> None:
    data: dict[str, Any] = {"message": message}
    if progress is not None:
        data["progress"] = progress
    await emit_agent_event(session_id, "synthesis", agent, data)


async def emit_error(
    session_id: str, agent: str, error: str, recoverable: bool = True,
) -> None:
    await emit_agent_event(
        session_id, "error", agent, {"error": error, "recoverable": recoverable},
    )


async def emit_verdict(
    session_id: str, agent: str, verdict: str, confidence: float | None = None,
) -> None:
    """Emit a verdict determination event."""
    data: dict[str, Any] = {"verdict": verdict}
    if confidence is not None:
        data["confidence"] = confidence
    await emit_agent_event(session_id, "verdict", agent, data)
