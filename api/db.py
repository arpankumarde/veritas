"""
Database service for API.

Thin wrapper around VeritasDatabase that adds API-specific query methods
(filtered listings, pagination, aggregated stats with joins).
"""
import json as _json

from engine.models.evidence import CheckSession
from engine.storage.database import VeritasDatabase


class APIDatabase:
    """Database wrapper for API endpoints.

    Uses VeritasDatabase via composition for connection pooling, schema
    management, and common CRUD operations.  Adds read-only query methods
    needed by the REST API (filtered listings, pagination, aggregated joins).
    """

    def __init__(self, db_path: str = "veritas.db"):
        self._core = VeritasDatabase(db_path)

    async def connect(self):
        """Connect to database (delegates to core pool)."""
        await self._core.connect()

    async def close(self):
        """Close database connections."""
        await self._core.close()

    # ------------------------------------------------------------------
    # Delegated common operations
    # ------------------------------------------------------------------

    async def get_session(self, session_id: str) -> CheckSession | None:
        return await self._core.get_session(session_id)

    async def create_session(self, claim: str, max_iterations: int = 5) -> CheckSession:
        return await self._core.create_session(claim, max_iterations)

    # ------------------------------------------------------------------
    # API-specific methods
    # ------------------------------------------------------------------

    async def list_sessions(self, limit: int = 100) -> list[CheckSession]:
        """List all check sessions, most recent first."""
        rows = await self._core._fetch_all(
            "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?",
            (limit,),
        )
        return [self._core._row_to_session(row) for row in rows]

    async def update_session_status(
        self, session_id: str, status: str, ended_at=None,
    ) -> bool:
        """Update session status and optionally set ended_at."""
        if ended_at:
            await self._core._write(
                "UPDATE sessions SET status = ?, ended_at = ? WHERE id = ?",
                (status, ended_at.isoformat(), session_id),
                op="update_session_status",
            )
        else:
            await self._core._write(
                "UPDATE sessions SET status = ? WHERE id = ?",
                (status, session_id),
                op="update_session_status",
            )
        return True

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and all related data atomically."""
        tables = [
            "evidence", "topics", "messages", "verification_results",
            "credibility_audit", "agent_decisions", "events",
        ]
        pool = self._core._check_pool()
        async with pool.acquire() as conn:
            try:
                # Acquire write lock upfront so all DELETEs run
                # in one atomic transaction
                await conn.execute("BEGIN IMMEDIATE")
                for table in tables:
                    await conn.execute(
                        f"DELETE FROM {table} WHERE session_id = ?",
                        (session_id,),
                    )
                cursor = await conn.execute(
                    "DELETE FROM sessions WHERE id = ?",
                    (session_id,),
                )
                await conn.execute("COMMIT")
                return cursor.rowcount > 0
            except Exception:
                await conn.execute("ROLLBACK")
                raise

    async def mark_crashed_sessions(self) -> int:
        """Mark sessions left as 'running' as 'crashed'. Returns count."""
        row = await self._core._fetch_one(
            "SELECT COUNT(*) as count FROM sessions WHERE status = 'running'"
        )
        count = row["count"] if row else 0
        if count > 0:
            await self._core._write(
                "UPDATE sessions SET status = 'crashed' WHERE status = 'running'",
                op="mark_crashed_sessions",
            )
        return count

    async def save_event(
        self,
        session_id: str,
        event_type: str,
        agent: str,
        timestamp: str,
        data_json: str,
        created_at: str,
    ) -> None:
        """Persist an emitted event for later playback."""
        await self._core._write(
            """INSERT INTO events (session_id, event_type, agent, timestamp, data_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, event_type, agent, timestamp, data_json, created_at),
            op="save_event",
        )

    async def list_events(self, session_id: str, limit: int = 500, order: str = "desc"):
        """List persisted events for a session."""
        order_sql = "DESC" if order.lower() == "desc" else "ASC"
        return await self._core._fetch_all(
            f"""SELECT session_id, event_type, agent, timestamp, data_json
            FROM events WHERE session_id = ?
            ORDER BY timestamp {order_sql} LIMIT ?""",
            (session_id, limit),
        )

    async def list_evidence(
        self,
        session_id: str,
        limit: int = 200,
        offset: int = 0,
        order: str = "desc",
        search: str | None = None,
        evidence_type: str | None = None,
        min_confidence: float | None = None,
        max_confidence: float | None = None,
    ):
        """List evidence with optional filters."""
        order_sql = "DESC" if order.lower() == "desc" else "ASC"
        params: list = [session_id]
        where = ["session_id = ?"]

        if search:
            where.append("(content LIKE ? OR source_url LIKE ? OR search_query LIKE ?)")
            like = f"%{search}%"
            params.extend([like, like, like])
        if evidence_type and evidence_type.lower() != "all":
            where.append("LOWER(evidence_type) = ?")
            params.append(evidence_type.lower())
        if min_confidence is not None:
            where.append("confidence >= ?")
            params.append(min_confidence)
        if max_confidence is not None:
            where.append("confidence <= ?")
            params.append(max_confidence)

        where_sql = " AND ".join(where)
        rows = await self._core._fetch_all(
            f"""SELECT id, session_id, content, evidence_type, source_url,
                confidence, search_query, created_at, verification_status,
                verification_method, kg_support_score
            FROM findings WHERE {where_sql}
            ORDER BY created_at {order_sql} LIMIT ? OFFSET ?""",
            (*params, limit, offset),
        )
        return [
            {
                "id": r["id"],
                "session_id": r["session_id"],
                "content": r["content"],
                "evidence_type": r["evidence_type"],
                "source_url": r["source_url"],
                "confidence": r["confidence"],
                "search_query": r["search_query"],
                "created_at": r["created_at"],
                "verification_status": r["verification_status"],
                "verification_method": r["verification_method"],
                "kg_support_score": r["kg_support_score"],
            }
            for r in rows
        ]

    async def list_sources(
        self, session_id: str, limit: int = 200, offset: int = 0,
    ):
        """List source index for a session."""
        rows = await self._core._fetch_all(
            """WITH source_stats AS (
                SELECT source_url, COUNT(*) as evidence_count,
                    AVG(confidence) as avg_confidence, MAX(created_at) as last_seen
                FROM findings
                WHERE session_id = ? AND source_url IS NOT NULL AND source_url != ''
                GROUP BY source_url
            ), latest_audit AS (
                SELECT url, MAX(created_at) as max_created_at
                FROM credibility_audit WHERE session_id = ? GROUP BY url
            )
            SELECT s.source_url, s.evidence_count, s.avg_confidence, s.last_seen,
                ca.domain, ca.final_score, ca.credibility_label
            FROM source_stats s
            LEFT JOIN latest_audit la ON la.url = s.source_url
            LEFT JOIN credibility_audit ca
              ON ca.url = la.url AND ca.created_at = la.max_created_at AND ca.session_id = ?
            ORDER BY s.evidence_count DESC, s.avg_confidence DESC
            LIMIT ? OFFSET ?""",
            (session_id, session_id, session_id, limit, offset),
        )

        results = []
        for row in rows:
            domain = row["domain"]
            final_score = row["final_score"]
            credibility_label = row["credibility_label"]

            results.append({
                "source_url": row["source_url"],
                "evidence_count": row["evidence_count"],
                "avg_confidence": row["avg_confidence"],
                "last_seen": row["last_seen"],
                "domain": domain,
                "final_score": final_score,
                "credibility_label": credibility_label,
            })
        return results

    async def list_verification_results(
        self, session_id: str, limit: int = 200, offset: int = 0,
    ):
        """List verification results joined with evidence content."""
        rows = await self._core._fetch_all(
            """SELECT
                vr.id, vr.session_id, vr.finding_id,
                vr.original_confidence, vr.verified_confidence,
                vr.verification_status, vr.verification_method,
                vr.consistency_score, vr.kg_support_score,
                vr.kg_entity_matches, vr.kg_supporting_relations,
                vr.critic_iterations, vr.corrections_made,
                vr.questions_asked, vr.external_verification_used,
                vr.contradictions, vr.verification_time_ms,
                vr.created_at, vr.error,
                e.content AS evidence_content, e.evidence_type,
                e.source_url, e.confidence AS current_confidence
            FROM verification_results vr
            LEFT JOIN evidence e ON e.id = vr.finding_id AND e.session_id = vr.session_id
            WHERE vr.session_id = ?
            ORDER BY vr.created_at DESC LIMIT ? OFFSET ?""",
            (session_id, limit, offset),
        )
        return [
            {
                "id": r["id"],
                "session_id": r["session_id"],
                "finding_id": r["finding_id"],
                "original_confidence": r["original_confidence"],
                "verified_confidence": r["verified_confidence"],
                "verification_status": r["verification_status"],
                "verification_method": r["verification_method"],
                "consistency_score": r["consistency_score"],
                "kg_support_score": r["kg_support_score"],
                "kg_entity_matches": r["kg_entity_matches"],
                "kg_supporting_relations": r["kg_supporting_relations"],
                "critic_iterations": r["critic_iterations"],
                "corrections_made": _json.loads(r["corrections_made"]) if r["corrections_made"] else None,
                "questions_asked": _json.loads(r["questions_asked"]) if r["questions_asked"] else None,
                "external_verification_used": bool(r["external_verification_used"]),
                "contradictions": _json.loads(r["contradictions"]) if r["contradictions"] else None,
                "verification_time_ms": r["verification_time_ms"],
                "created_at": r["created_at"],
                "error": r["error"],
                "evidence_content": r["evidence_content"],
                "evidence_type": r["evidence_type"],
                "source_url": r["source_url"],
                "current_confidence": r["current_confidence"],
            }
            for r in rows
        ]

    async def get_verification_stats(self, session_id: str) -> dict:
        """Get aggregate verification stats for a session."""
        row = await self._core._fetch_one(
            """SELECT
                COUNT(*) as total,
                SUM(CASE WHEN verification_status = 'verified' THEN 1 ELSE 0 END) as verified,
                SUM(CASE WHEN verification_status = 'flagged' THEN 1 ELSE 0 END) as flagged,
                SUM(CASE WHEN verification_status = 'rejected' THEN 1 ELSE 0 END) as rejected,
                SUM(CASE WHEN verification_status = 'pending' THEN 1 ELSE 0 END) as pending,
                AVG(verified_confidence) as avg_confidence,
                AVG(verification_time_ms) as avg_time_ms
            FROM verification_results WHERE session_id = ?""",
            (session_id,),
        )
        if not row or row["total"] == 0:
            return {
                "total": 0, "verified": 0, "flagged": 0,
                "rejected": 0, "pending": 0,
                "avg_confidence": None, "avg_time_ms": None,
            }
        return {
            "total": row["total"],
            "verified": row["verified"],
            "flagged": row["flagged"],
            "rejected": row["rejected"],
            "pending": row["pending"],
            "avg_confidence": row["avg_confidence"],
            "avg_time_ms": row["avg_time_ms"],
        }

    async def list_agent_decisions(
        self, session_id: str, limit: int = 200, offset: int = 0,
    ):
        """List agent decision logs for a session."""
        rows = await self._core._fetch_all(
            """SELECT id, session_id, agent_role, decision_type,
                decision_outcome, reasoning, inputs_json,
                metrics_json, iteration, created_at
            FROM agent_decisions WHERE session_id = ?
            ORDER BY created_at DESC LIMIT ? OFFSET ?""",
            (session_id, limit, offset),
        )
        return [
            {
                "id": r["id"],
                "session_id": r["session_id"],
                "agent_role": r["agent_role"],
                "decision_type": r["decision_type"],
                "decision_outcome": r["decision_outcome"],
                "reasoning": r["reasoning"],
                "inputs": _json.loads(r["inputs_json"]) if r["inputs_json"] else None,
                "metrics": _json.loads(r["metrics_json"]) if r["metrics_json"] else None,
                "iteration": r["iteration"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    async def list_topics(self, session_id: str, limit: int = 200):
        """List topic tree for a session."""
        rows = await self._core._fetch_all(
            """SELECT id, session_id, topic, parent_topic_id, depth,
                status, priority, assigned_at, completed_at, findings_count
            FROM topics WHERE session_id = ?
            ORDER BY depth ASC, priority DESC LIMIT ?""",
            (session_id, limit),
        )
        return [
            {
                "id": r["id"],
                "session_id": r["session_id"],
                "topic": r["topic"],
                "parent_topic_id": r["parent_topic_id"],
                "depth": r["depth"],
                "status": r["status"],
                "priority": r["priority"],
                "assigned_at": r["assigned_at"],
                "completed_at": r["completed_at"],
                "findings_count": r["findings_count"],
            }
            for r in rows
        ]

    async def get_session_slug(self, session_id: str) -> str | None:
        """Get the slug for a session ID."""
        row = await self._core._fetch_one(
            "SELECT slug FROM sessions WHERE id = ?", (session_id,)
        )
        return row["slug"] if row else None

    async def get_session_stats(self, session_id: str) -> dict:
        """Get aggregate stats for a session (evidence, sources, sub-claims)."""
        evidence_row = await self._core._fetch_one(
            "SELECT COUNT(*) as count FROM findings WHERE session_id = ?",
            (session_id,),
        )
        sources_row = await self._core._fetch_one(
            """SELECT COUNT(DISTINCT source_url) as count FROM findings
            WHERE session_id = ? AND source_url IS NOT NULL AND source_url != ''""",
            (session_id,),
        )
        topics_row = await self._core._fetch_one(
            "SELECT COUNT(*) as count FROM topics WHERE session_id = ?",
            (session_id,),
        )
        return {
            "evidence": evidence_row["count"] if evidence_row else 0,
            "sources": sources_row["count"] if sources_row else 0,
            "sub_claims": topics_row["count"] if topics_row else 0,
        }


# Global database instance
_db_instance: APIDatabase | None = None


async def get_db() -> APIDatabase:
    """Get or create the global database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = APIDatabase()
        await _db_instance.connect()
    return _db_instance


async def close_db():
    """Close the global database instance."""
    global _db_instance
    if _db_instance:
        await _db_instance.close()
        _db_instance = None
