"""
Check session routes.

Handles CRUD operations for check sessions.
"""
from fastapi import APIRouter, HTTPException, Query

from api.db import get_db
from api.models import CheckSessionCreate, CheckSessionResponse

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


def _session_response(s) -> CheckSessionResponse:
    """Build a CheckSessionResponse from a CheckSession."""
    return CheckSessionResponse(
        session_id=s.id,
        claim=s.claim,
        verdict=s.verdict,
        max_iterations=s.max_iterations,
        time_limit=s.time_limit_minutes,
        status=s.status,
        created_at=s.started_at,
        completed_at=s.ended_at,
        elapsed_seconds=s.elapsed_seconds,
        paused_at=s.paused_at,
        iteration_count=s.iteration_count,
    )


@router.post("/", response_model=CheckSessionResponse, status_code=201)
async def create_session(session: CheckSessionCreate):
    """
    Create a new check session in the database.

    This creates a session record that can be used for CLI or UI fact-checking.
    """
    db = await get_db()
    db_session = await db.create_session(session.claim, session.max_iterations)
    return _session_response(db_session)


@router.get("/", response_model=list[CheckSessionResponse])
async def list_sessions(limit: int = Query(default=100, ge=1, le=1000)):
    """List all check sessions from the database."""
    db = await get_db()
    sessions = await db.list_sessions(limit)
    return [_session_response(s) for s in sessions]


@router.get("/{session_id}", response_model=CheckSessionResponse)
async def get_session(session_id: str):
    """Get a specific check session from the database."""
    db = await get_db()
    session = await db.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    return _session_response(session)


@router.delete("/{session_id}", status_code=204)
async def delete_session(session_id: str):
    """Delete a check session from the database."""
    db = await get_db()
    deleted = await db.delete_session(session_id)

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")


@router.get("/{session_id}/stats")
async def get_session_stats(session_id: str):
    """Get aggregate stats for a session (evidence, sources, sub-claims)."""
    db = await get_db()
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return await db.get_session_stats(session_id)
