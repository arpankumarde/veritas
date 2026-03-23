"""
Event emission routes.

Allows external processes (e.g., CLI) to forward agent events to the API
so connected WebSocket clients can receive them in real time.
"""
import json
from typing import Any

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from api.db import get_db
from api.events import emit_event, get_event_emitter

router = APIRouter(prefix="/api/events", tags=["events"])


class EventEmitRequest(BaseModel):
    """Request payload for emitting an agent event."""
    session_id: str = Field(..., min_length=1)
    event_type: str = Field(..., min_length=1)
    agent: str = Field(..., min_length=1)
    data: dict[str, Any] = Field(default_factory=dict)


@router.post("/emit")
async def emit_event_endpoint(payload: EventEmitRequest):
    """Emit an event to all WebSocket subscribers for a session."""
    await emit_event(
        session_id=payload.session_id,
        event_type=payload.event_type,
        agent=payload.agent,
        data=payload.data,
    )

    emitter = get_event_emitter()
    return {
        "status": "emitted",
        "session_id": payload.session_id,
        "event_type": payload.event_type,
        "subscribers": emitter.get_subscriber_count(payload.session_id),
    }


@router.get("/{session_id}")
async def list_session_events(
    session_id: str,
    limit: int = Query(default=500, ge=1, le=1000),
    order: str = "desc",
):
    """List persisted events for a session."""
    db = await get_db()
    rows = await db.list_events(session_id, limit=limit, order=order)

    events = []
    for row in rows:
        try:
            data = json.loads(row["data_json"]) if row["data_json"] else {}
        except Exception:
            data = {}
        events.append(
            {
                "session_id": row["session_id"],
                "event_type": row["event_type"],
                "agent": row["agent"],
                "timestamp": row["timestamp"],
                "data": data,
            }
        )

    return events
