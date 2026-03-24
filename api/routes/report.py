"""
Report endpoints for a session.
"""
from pathlib import Path

from fastapi import APIRouter, HTTPException

from api.db import get_db

router = APIRouter(prefix="/api/sessions", tags=["report"])


@router.get("/{session_id}/report")
async def get_report(session_id: str):
    """Return the verdict report markdown for a session, if available."""
    db = await get_db()
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    slug = session.slug or "fact_check"
    # Sanitize slug to prevent path traversal
    safe_slug = "".join(c for c in slug if c.isalnum() or c in "-_")
    safe_session_id = "".join(c for c in session_id if c.isalnum())
    output_dir = Path("output") / f"{safe_slug}_{safe_session_id}"

    # Ensure resolved path stays within the output directory
    base_output = Path("output").resolve()
    resolved = output_dir.resolve()
    if not str(resolved).startswith(str(base_output)):
        raise HTTPException(status_code=400, detail="Invalid session path")

    # Try both filenames (report.md is what the engine writes)
    report_path = resolved / "report.md"
    if not report_path.exists():
        report_path = resolved / "verdict.md"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Verdict report not found")

    return {
        "session_id": session_id,
        "report": report_path.read_text(),
        "path": str(report_path),
    }
