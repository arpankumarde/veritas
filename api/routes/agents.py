"""
Agent transparency API routes.

Serves agent decision logs, topic hierarchy, and agent status for the
Agent Transparency page.
"""
from fastapi import APIRouter, HTTPException, Query

from api.db import get_db

router = APIRouter(prefix="/api/sessions", tags=["agents"])


@router.get("/{session_id}/agents/decisions")
async def list_agent_decisions(
    session_id: str,
    limit: int = Query(default=200, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    """List agent decision logs -- reasoning traces and outcomes."""
    db = await get_db()
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return await db.list_agent_decisions(
        session_id=session_id,
        limit=limit,
        offset=offset,
    )


@router.get("/{session_id}/agents/hierarchy")
async def get_agent_hierarchy(session_id: str):
    """
    Get agent hierarchy with topic assignments and progress.

    Reconstructs the Director -> Manager -> Intern structure from
    topics and agent_decisions tables.
    """
    db = await get_db()
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get topics as proxy for agent work assignments
    topics = await db.list_topics(session_id=session_id, limit=200)

    # Get recent decisions grouped by role
    decisions = await db.list_agent_decisions(session_id=session_id, limit=100)

    # Build role summaries from decisions
    roles: dict[str, dict] = {}
    for d in decisions:
        role = d["agent_role"]
        if role not in roles:
            roles[role] = {
                "role": role,
                "decision_count": 0,
                "last_action": None,
                "last_action_at": None,
                "recent_decisions": [],
            }
        roles[role]["decision_count"] += 1
        if roles[role]["last_action"] is None:
            roles[role]["last_action"] = d["decision_outcome"]
            roles[role]["last_action_at"] = d["created_at"]
        if len(roles[role]["recent_decisions"]) < 5:
            roles[role]["recent_decisions"].append(d)

    # Topic progress
    total_topics = len(topics)
    completed_topics = sum(1 for t in topics if t["status"] == "completed")
    in_progress_topics = sum(1 for t in topics if t["status"] == "in_progress")

    return {
        "session_id": session_id,
        "roles": roles,
        "topics": topics,
        "progress": {
            "total_topics": total_topics,
            "completed": completed_topics,
            "in_progress": in_progress_topics,
            "pending": total_topics - completed_topics - in_progress_topics,
        },
    }
