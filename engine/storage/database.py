"""SQLite database for persisting fact-check state and evidence."""

import asyncio
import json
import re
import secrets
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import aiosqlite

from ..logging_config import get_logger
from ..models.evidence import (
    AgentMessage,
    AgentRole,
    CheckSession,
    Evidence,
    EvidenceType,
    SubClaim,
)

logger = get_logger(__name__)


def _generate_session_id() -> str:
    """Generate a unique 7-character hexadecimal session ID."""
    return secrets.token_hex(4)[:7]  # 7 hex chars


def _generate_slug(claim: str) -> str:
    """Generate a URL-friendly slug from the claim being checked.

    Examples:
        "The Earth is flat" -> "earth-flat"
        "COVID vaccines contain microchips" -> "covid-vaccines-contain-microchips"
    """
    # Remove common question words and punctuation
    text = claim.lower()
    text = re.sub(r"\?+$", "", text)

    # Remove stop words
    stop_words = {
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "and",
        "or",
        "but",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "what",
        "how",
        "why",
        "when",
        "where",
        "who",
        "which",
        "can",
        "that",
        "this",
        "it",
        "its",
        "they",
        "them",
        "their",
    }
    words = text.split()
    words = [w for w in words if w not in stop_words]

    # Take first 4-5 meaningful words
    words = words[:5]

    # Clean and join
    slug = "-".join(words)
    slug = re.sub(r"[^a-z0-9-]", "", slug)
    slug = re.sub(r"-+", "-", slug)
    slug = slug.strip("-")

    # Limit length
    if len(slug) > 50:
        slug = slug[:50].rsplit("-", 1)[0]

    return slug or "claim"


class _ConnectionPool:
    """Simple async SQLite connection pool with WAL mode."""

    def __init__(self, db_path: Path, size: int = 5):
        self._db_path = db_path
        self._size = size
        self._pool: asyncio.Queue[aiosqlite.Connection] = asyncio.Queue(maxsize=size)
        self._initialized = False

    async def init(self):
        """Create pool connections with WAL mode.

        Sets WAL on the first connection before opening the rest to avoid
        lock contention on a fresh database.
        """
        # First connection: set WAL mode (requires exclusive access)
        first = await aiosqlite.connect(self._db_path)
        first.row_factory = aiosqlite.Row
        await first.execute("PRAGMA busy_timeout=15000")
        await first.execute("PRAGMA journal_mode=WAL")
        await self._pool.put(first)

        # Remaining connections: WAL is already set at the file level
        for _ in range(self._size - 1):
            conn = await aiosqlite.connect(self._db_path)
            conn.row_factory = aiosqlite.Row
            await conn.execute("PRAGMA busy_timeout=15000")
            await self._pool.put(conn)
        self._initialized = True

    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool."""
        conn = await self._pool.get()
        healthy = True
        try:
            yield conn
        except Exception:
            # Connection may be in a bad state after an error; discard it
            healthy = False
            raise
        finally:
            if healthy:
                await self._pool.put(conn)
            else:
                # Discard the broken connection and create a fresh one
                try:
                    await conn.close()
                except Exception:
                    pass
                try:
                    new_conn = await aiosqlite.connect(self._db_path)
                    new_conn.row_factory = aiosqlite.Row
                    await new_conn.execute("PRAGMA busy_timeout=15000")
                    await new_conn.execute("PRAGMA journal_mode=WAL")
                    await self._pool.put(new_conn)
                except Exception:
                    pass  # Pool shrinks by one; next acquire will block until one returns

    async def close(self):
        """Close all pooled connections."""
        while not self._pool.empty():
            conn = await self._pool.get()
            await conn.close()
        self._initialized = False


class VeritasDatabase:
    """SQLite database manager for fact-check persistence."""

    def __init__(self, db_path: str = "veritas.db"):
        self.db_path = Path(db_path)
        self._pool: _ConnectionPool | None = None

    async def connect(self) -> None:
        """Connect to the database and create tables if needed."""
        if self._pool is not None and self._pool._initialized:
            return  # Already connected
        logger.info("Database connecting: %s", self.db_path)
        self._pool = _ConnectionPool(self.db_path)
        await self._pool.init()
        async with self._pool.acquire() as conn:
            await self._create_tables(conn)

    async def close(self) -> None:
        """Close all database connections."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    def _check_pool(self) -> _ConnectionPool:
        """Get the active connection pool.

        Raises RuntimeError if not connected.
        """
        if self._pool is None or not self._pool._initialized:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._pool

    # --- Low-level helpers to eliminate boilerplate ---

    async def _fetch_one(self, sql: str, params: tuple = ()) -> aiosqlite.Row | None:
        """Execute a read query and return a single row."""
        pool = self._check_pool()
        async with pool.acquire() as conn:
            cursor = await conn.execute(sql, params)
            return await cursor.fetchone()

    async def _fetch_all(self, sql: str, params: tuple = ()) -> list[aiosqlite.Row]:
        """Execute a read query and return all rows."""
        pool = self._check_pool()
        async with pool.acquire() as conn:
            cursor = await conn.execute(sql, params)
            return await cursor.fetchall()

    async def _write(self, sql: str, params: tuple = (), *, op: str = "") -> int:
        """Execute a write with commit/rollback. Returns lastrowid."""
        pool = self._check_pool()
        async with pool.acquire() as conn:
            try:
                cursor = await conn.execute(sql, params)
                await conn.commit()
                return cursor.lastrowid
            except Exception as e:
                await conn.rollback()
                logger.error("DB error in %s: %s", op or "write", e, exc_info=True)
                raise

    async def _write_many(self, sql: str, params_list: list, *, op: str = "") -> int:
        """Execute a batch write. Returns count of rows written."""
        pool = self._check_pool()
        async with pool.acquire() as conn:
            try:
                await conn.executemany(sql, params_list)
                await conn.commit()
                return len(params_list)
            except Exception as e:
                await conn.rollback()
                logger.error("DB error in %s: %s", op or "write_many", e, exc_info=True)
                raise

    # --- Row mappers ---

    @staticmethod
    def _row_to_session(row: aiosqlite.Row) -> CheckSession:
        return CheckSession(
            id=row["id"],
            claim=row["claim"],
            slug=row["slug"],
            verdict=row["verdict"] if "verdict" in row.keys() else None,
            max_iterations=row["max_iterations"] if "max_iterations" in row.keys() else 5,
            time_limit_minutes=(
                row["time_limit_minutes"] if "time_limit_minutes" in row.keys() else 0
            ),
            started_at=datetime.fromisoformat(row["started_at"]),
            ended_at=datetime.fromisoformat(row["ended_at"]) if row["ended_at"] else None,
            status=row["status"],
            total_findings=row["total_findings"],
            total_searches=row["total_searches"],
            depth_reached=row["depth_reached"],
            elapsed_seconds=row["elapsed_seconds"] or 0.0,
            paused_at=(
                datetime.fromisoformat(row["paused_at"]) if row["paused_at"] else None
            ),
            iteration_count=row["iteration_count"] or 0,
            phase=row["phase"] or "init",
        )

    @staticmethod
    def _row_to_subclaim(row: aiosqlite.Row) -> SubClaim:
        return SubClaim(
            id=row["id"],
            session_id=row["session_id"],
            topic=row["topic"],
            parent_topic_id=row["parent_topic_id"],
            depth=row["depth"],
            status=row["status"],
            priority=row["priority"],
        )

    @staticmethod
    def _row_to_evidence(row: aiosqlite.Row) -> Evidence:
        return Evidence(
            id=row["id"],
            session_id=row["session_id"],
            topic_id=row["topic_id"],
            content=row["content"],
            evidence_type=EvidenceType(row["evidence_type"]),
            source_url=row["source_url"],
            confidence=row["confidence"],
            search_query=row["search_query"],
            created_at=datetime.fromisoformat(row["created_at"]),
            validated_by_manager=bool(row["validated_by_manager"]),
            manager_notes=row["manager_notes"],
            verification_status=row["verification_status"],
            verification_method=row["verification_method"],
            kg_support_score=row["kg_support_score"],
            original_confidence=row["original_confidence"],
        )

    @staticmethod
    def _row_to_message(row: aiosqlite.Row) -> AgentMessage:
        return AgentMessage(
            id=row["id"],
            session_id=row["session_id"],
            from_agent=AgentRole(row["from_agent"]),
            to_agent=AgentRole(row["to_agent"]),
            message_type=row["message_type"],
            content=row["content"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    async def _create_tables(self, conn: aiosqlite.Connection) -> None:
        """Create database tables if they don't exist."""
        await conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                claim TEXT NOT NULL,
                slug TEXT,
                verdict TEXT,
                max_iterations INTEGER DEFAULT 5,
                time_limit_minutes INTEGER DEFAULT 0,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                status TEXT DEFAULT 'active',
                total_findings INTEGER DEFAULT 0,
                total_searches INTEGER DEFAULT 0,
                depth_reached INTEGER DEFAULT 0,
                elapsed_seconds REAL DEFAULT 0.0,
                paused_at TEXT,
                iteration_count INTEGER DEFAULT 0,
                phase TEXT DEFAULT 'init'
            );

            CREATE TABLE IF NOT EXISTS topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                topic TEXT NOT NULL,
                parent_topic_id INTEGER,
                depth INTEGER DEFAULT 0,
                status TEXT DEFAULT 'pending',
                priority INTEGER DEFAULT 5,
                assigned_at TEXT,
                completed_at TEXT,
                findings_count INTEGER DEFAULT 0,
                FOREIGN KEY (session_id) REFERENCES sessions(id),
                FOREIGN KEY (parent_topic_id) REFERENCES topics(id)
            );

            CREATE TABLE IF NOT EXISTS findings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                topic_id INTEGER,
                content TEXT NOT NULL,
                evidence_type TEXT NOT NULL,
                source_url TEXT,
                confidence REAL DEFAULT 0.8,
                search_query TEXT,
                created_at TEXT NOT NULL,
                validated_by_manager INTEGER DEFAULT 0,
                manager_notes TEXT,
                verification_status TEXT,
                verification_method TEXT,
                kg_support_score REAL DEFAULT 0.0,
                original_confidence REAL,
                FOREIGN KEY (session_id) REFERENCES sessions(id),
                FOREIGN KEY (topic_id) REFERENCES topics(id)
            );

            CREATE TABLE IF NOT EXISTS verification_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                finding_id INTEGER NOT NULL,
                original_confidence REAL,
                verified_confidence REAL,
                verification_status TEXT,
                verification_method TEXT,
                consistency_score REAL,
                kg_support_score REAL,
                kg_entity_matches INTEGER,
                kg_supporting_relations INTEGER,
                critic_iterations INTEGER,
                corrections_made TEXT,
                questions_asked TEXT,
                external_verification_used INTEGER DEFAULT 0,
                contradictions TEXT,
                verification_time_ms REAL,
                created_at TEXT NOT NULL,
                error TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id),
                FOREIGN KEY (finding_id) REFERENCES findings(id)
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                from_agent TEXT NOT NULL,
                to_agent TEXT NOT NULL,
                message_type TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE INDEX IF NOT EXISTS idx_findings_session ON findings(session_id);
            CREATE INDEX IF NOT EXISTS idx_topics_session ON topics(session_id);
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
            CREATE INDEX IF NOT EXISTS idx_verification_session ON verification_results(session_id);
            CREATE INDEX IF NOT EXISTS idx_verification_finding ON verification_results(finding_id);

            -- Audit trail tables for explainability
            CREATE TABLE IF NOT EXISTS credibility_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                finding_id TEXT,
                url TEXT NOT NULL,
                domain TEXT NOT NULL,
                final_score REAL NOT NULL,
                domain_authority_score REAL,
                recency_score REAL,
                source_type_score REAL,
                https_score REAL,
                path_depth_score REAL,
                credibility_label TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS agent_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                agent_role TEXT NOT NULL,
                decision_type TEXT NOT NULL,
                decision_outcome TEXT NOT NULL,
                reasoning TEXT,
                inputs_json TEXT,
                metrics_json TEXT,
                iteration INTEGER,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                agent TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                data_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE INDEX IF NOT EXISTS idx_credibility_session ON credibility_audit(session_id);
            CREATE INDEX IF NOT EXISTS idx_decisions_session ON agent_decisions(session_id);
            CREATE INDEX IF NOT EXISTS idx_decisions_type ON agent_decisions(decision_type);
            CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
            CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
        """)
        await conn.commit()

        # Migration: add max_iterations column for existing databases
        try:
            await conn.execute(
                "ALTER TABLE sessions ADD COLUMN max_iterations INTEGER DEFAULT 5"
            )
            await conn.commit()
        except Exception as e:
            # [HARDENED] ERR-002: Only suppress expected "duplicate column" errors
            if "duplicate column" in str(e).lower():
                pass
            else:
                raise

    # Session methods
    async def create_session(self, claim: str, max_iterations: int = 5) -> CheckSession:
        """Create a new fact-checking session with unique hex ID."""
        # Retry on ID collision (up to 3 attempts)
        last_err: Exception | None = None
        for _attempt in range(3):
            session_id = _generate_session_id()
            slug = _generate_slug(claim)
            now = datetime.now().isoformat()
            try:
                await self._write(
                    "INSERT INTO sessions (id, claim, slug, max_iterations, started_at, status) VALUES (?, ?, ?, ?, ?, 'active')",
                    (session_id, claim, slug, max_iterations, now),
                    op="create_session",
                )
                break
            except Exception as e:
                last_err = e
                if "unique" in str(e).lower() or "constraint" in str(e).lower():
                    continue
                raise
        else:
            raise last_err  # type: ignore[misc]
        logger.info("Session created: %s", session_id)
        return CheckSession(
            id=session_id,
            claim=claim,
            slug=slug,
            max_iterations=max_iterations,
            started_at=datetime.fromisoformat(now),
            status="active",
        )

    async def get_session(self, session_id: str) -> CheckSession | None:
        """Get a session by ID."""
        row = await self._fetch_one("SELECT * FROM sessions WHERE id = ?", (session_id,))
        return self._row_to_session(row) if row else None

    async def update_session(self, session: CheckSession) -> None:
        """Update a session."""
        await self._write(
            """
            UPDATE sessions SET
                status = ?, verdict = ?, ended_at = ?, total_findings = ?, total_searches = ?,
                depth_reached = ?, elapsed_seconds = ?, paused_at = ?,
                iteration_count = ?, phase = ?
            WHERE id = ?
            """,
            (
                session.status,
                session.verdict,
                session.ended_at.isoformat() if session.ended_at else None,
                session.total_findings,
                session.total_searches,
                session.depth_reached,
                session.elapsed_seconds,
                session.paused_at.isoformat() if session.paused_at else None,
                session.iteration_count,
                session.phase,
                session.id,
            ),
            op="update_session",
        )

    # Topic methods (SubClaim)
    async def create_topic(
        self,
        session_id: str,
        topic: str,
        parent_topic_id: int | None = None,
        depth: int = 0,
        priority: int = 5,
    ) -> SubClaim:
        """Create a new sub-claim to investigate."""
        lastrowid = await self._write(
            "INSERT INTO topics (session_id, topic, parent_topic_id, depth, priority) VALUES (?, ?, ?, ?, ?)",
            (session_id, topic, parent_topic_id, depth, priority),
            op="create_topic",
        )
        return SubClaim(
            id=lastrowid,
            session_id=session_id,
            topic=topic,
            parent_topic_id=parent_topic_id,
            depth=depth,
            priority=priority,
        )

    async def get_pending_topics(self, session_id: str, limit: int = 10) -> list[SubClaim]:
        """Get pending sub-claims ordered by priority."""
        rows = await self._fetch_all(
            """SELECT * FROM topics WHERE session_id = ? AND status = 'pending'
            ORDER BY priority DESC, depth ASC LIMIT ?""",
            (session_id, limit),
        )
        return [self._row_to_subclaim(row) for row in rows]

    async def update_topic_status(
        self, topic_id: int, status: str, findings_count: int = 0
    ) -> None:
        """Update sub-claim status."""
        completed_at = datetime.now().isoformat() if status == "completed" else None
        await self._write(
            "UPDATE topics SET status = ?, completed_at = ?, findings_count = ? WHERE id = ?",
            (status, completed_at, findings_count, topic_id),
            op="update_topic_status",
        )

    async def get_all_topics(self, session_id: str) -> list[SubClaim]:
        """Get all sub-claims for a session (for state reconstruction on resume)."""
        rows = await self._fetch_all(
            "SELECT * FROM topics WHERE session_id = ? ORDER BY priority DESC, depth ASC",
            (session_id,),
        )
        return [self._row_to_subclaim(row) for row in rows]

    async def reset_in_progress_topics(self, session_id: str) -> None:
        """Reset any in_progress sub-claims back to pending (for resume after pause/crash)."""
        await self._write(
            "UPDATE topics SET status = 'pending', assigned_at = NULL WHERE session_id = ? AND status = 'in_progress'",
            (session_id,),
            op="reset_in_progress_topics",
        )

    # Evidence methods
    async def save_evidence(self, evidence: Evidence, topic_id: int | None = None) -> Evidence:
        """Save a piece of evidence."""
        evidence.id = await self._write(
            """INSERT INTO findings (
                session_id, topic_id, content, evidence_type, source_url,
                confidence, search_query, created_at, validated_by_manager, manager_notes,
                verification_status, verification_method, kg_support_score, original_confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                evidence.session_id, topic_id, evidence.content, evidence.evidence_type.value,
                evidence.source_url, evidence.confidence, evidence.search_query,
                evidence.created_at.isoformat(), 1 if evidence.validated_by_manager else 0,
                evidence.manager_notes, evidence.verification_status, evidence.verification_method,
                evidence.kg_support_score, evidence.original_confidence,
            ),
            op="save_evidence",
        )
        return evidence

    async def update_evidence_verification(
        self,
        evidence_id: int,
        verification_status: str,
        verification_method: str,
        kg_support_score: float = 0.0,
        original_confidence: float | None = None,
        new_confidence: float | None = None,
    ) -> None:
        """Update an evidence item's verification status."""
        fields = [
            "verification_status = ?",
            "verification_method = ?",
            "kg_support_score = ?",
            "original_confidence = ?",
        ]
        params: list = [verification_status, verification_method, kg_support_score, original_confidence]
        if new_confidence is not None:
            fields.append("confidence = ?")
            params.append(new_confidence)
        params.append(evidence_id)
        await self._write(
            f"UPDATE findings SET {', '.join(fields)} WHERE id = ?",
            tuple(params),
            op="update_evidence_verification",
        )

    async def save_verification_result(
        self,
        session_id: str,
        finding_id: int,
        result_dict: dict,
    ) -> int:
        """Save a verification result."""
        return await self._write(
            """INSERT INTO verification_results (
                session_id, finding_id, original_confidence, verified_confidence,
                verification_status, verification_method, consistency_score,
                kg_support_score, kg_entity_matches, kg_supporting_relations,
                critic_iterations, corrections_made, questions_asked, external_verification_used,
                contradictions, verification_time_ms, created_at, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id, finding_id,
                result_dict.get("original_confidence"),
                result_dict.get("verified_confidence"),
                result_dict.get("verification_status"),
                result_dict.get("verification_method"),
                result_dict.get("consistency_score"),
                result_dict.get("kg_support_score"),
                result_dict.get("kg_entity_matches"),
                result_dict.get("kg_supporting_relations"),
                result_dict.get("critic_iterations"),
                json.dumps(result_dict.get("corrections_made", [])),
                json.dumps(result_dict.get("questions_asked", [])),
                1 if result_dict.get("external_verification_used") else 0,
                json.dumps(result_dict.get("contradictions", [])),
                result_dict.get("verification_time_ms", 0.0),
                datetime.now().isoformat(),
                result_dict.get("error"),
            ),
            op="save_verification_result",
        )

    async def get_verification_stats(self, session_id: str) -> dict:
        """Get verification statistics for a session."""
        row = await self._fetch_one(
            """SELECT
                COUNT(*) as total,
                SUM(CASE WHEN verification_status = 'verified' THEN 1 ELSE 0 END) as verified,
                SUM(CASE WHEN verification_status = 'flagged' THEN 1 ELSE 0 END) as flagged,
                SUM(CASE WHEN verification_status = 'rejected' THEN 1 ELSE 0 END) as rejected,
                SUM(CASE WHEN verification_status = 'skipped' THEN 1 ELSE 0 END) as skipped,
                AVG(original_confidence) as avg_original,
                AVG(verified_confidence) as avg_calibrated,
                AVG(verification_time_ms) as avg_time_ms
            FROM verification_results WHERE session_id = ?""",
            (session_id,),
        )

        if not row or row["total"] == 0:
            return {
                "total": 0,
                "verified": 0,
                "flagged": 0,
                "rejected": 0,
                "skipped": 0,
                "verification_rate": 0.0,
                "avg_original_confidence": 0.0,
                "avg_calibrated_confidence": 0.0,
                "avg_time_ms": 0.0,
            }

        total = row["total"]
        verified = row["verified"] or 0
        effective_total = total - (row["skipped"] or 0)
        verification_rate = verified / effective_total * 100 if effective_total > 0 else 0.0

        return {
            "total": total,
            "verified": verified,
            "flagged": row["flagged"] or 0,
            "rejected": row["rejected"] or 0,
            "skipped": row["skipped"] or 0,
            "verification_rate": verification_rate,
            "avg_original_confidence": row["avg_original"] or 0.0,
            "avg_calibrated_confidence": row["avg_calibrated"] or 0.0,
            "avg_time_ms": row["avg_time_ms"] or 0.0,
        }

    async def get_session_evidence(self, session_id: str) -> list[Evidence]:
        """Get all evidence for a session."""
        rows = await self._fetch_all(
            "SELECT * FROM findings WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        )
        return [self._row_to_evidence(row) for row in rows]

    # Backward-compat aliases (old name -> new name)
    async def save_finding(self, evidence: Evidence, topic_id: int | None = None) -> Evidence:
        """Alias for save_evidence (backward compat)."""
        return await self.save_evidence(evidence, topic_id)

    async def get_session_findings(self, session_id: str) -> list[Evidence]:
        """Alias for get_session_evidence (backward compat)."""
        return await self.get_session_evidence(session_id)

    # Message methods
    async def save_message(self, message: AgentMessage) -> AgentMessage:
        """Save an agent message."""
        message.id = await self._write(
            """INSERT INTO messages (
                session_id, from_agent, to_agent, message_type, content, metadata, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                message.session_id, message.from_agent.value, message.to_agent.value,
                message.message_type, message.content, json.dumps(message.metadata),
                message.created_at.isoformat(),
            ),
            op="save_message",
        )
        return message

    async def get_session_messages(
        self, session_id: str, agent_filter: AgentRole | None = None
    ) -> list[AgentMessage]:
        """Get messages for a session, optionally filtered by agent."""
        if agent_filter:
            rows = await self._fetch_all(
                """SELECT * FROM messages
                WHERE session_id = ? AND (from_agent = ? OR to_agent = ?)
                ORDER BY created_at""",
                (session_id, agent_filter.value, agent_filter.value),
            )
        else:
            rows = await self._fetch_all(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at",
                (session_id,),
            )
        return [self._row_to_message(row) for row in rows]

    # Stats methods
    async def get_session_stats(self, session_id: str) -> dict:
        """Get statistics for a session."""
        evidence_row = await self._fetch_one(
            """SELECT COUNT(*) as total_evidence, COUNT(DISTINCT search_query) as unique_searches,
            AVG(confidence) as avg_confidence FROM findings WHERE session_id = ?""",
            (session_id,),
        )
        topics_row = await self._fetch_one(
            """SELECT COUNT(*) as total_topics, MAX(depth) as max_depth,
            SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_topics
            FROM topics WHERE session_id = ?""",
            (session_id,),
        )
        return {
            "total_evidence": evidence_row["total_evidence"] or 0 if evidence_row else 0,
            "unique_searches": evidence_row["unique_searches"] or 0 if evidence_row else 0,
            "avg_confidence": evidence_row["avg_confidence"] or 0 if evidence_row else 0,
            "total_topics": topics_row["total_topics"] or 0 if topics_row else 0,
            "max_depth": topics_row["max_depth"] or 0 if topics_row else 0,
            "completed_topics": topics_row["completed_topics"] or 0 if topics_row else 0,
        }

    # Credibility Audit methods
    async def save_credibility_audit(
        self,
        session_id: str,
        finding_id: str | None,
        url: str,
        domain: str,
        final_score: float,
        domain_authority_score: float,
        recency_score: float,
        source_type_score: float,
        https_score: float,
        path_depth_score: float,
        credibility_label: str,
    ) -> int:
        """Save a credibility audit record."""
        return await self._write(
            """INSERT INTO credibility_audit (
                session_id, finding_id, url, domain, final_score,
                domain_authority_score, recency_score, source_type_score,
                https_score, path_depth_score, credibility_label, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id, finding_id, url, domain, final_score,
                domain_authority_score, recency_score, source_type_score,
                https_score, path_depth_score, credibility_label,
                datetime.now().isoformat(),
            ),
            op="save_credibility_audit",
        )

    async def get_credibility_audits(self, session_id: str) -> list[dict]:
        """Get all credibility audit records for a session."""
        rows = await self._fetch_all(
            "SELECT * FROM credibility_audit WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        )
        return [dict(row) for row in rows]

    # Agent Decision methods
    async def save_agent_decision(
        self,
        session_id: str,
        agent_role: str,
        decision_type: str,
        decision_outcome: str,
        reasoning: str | None = None,
        inputs_json: str | None = None,
        metrics_json: str | None = None,
        iteration: int | None = None,
    ) -> int:
        """Save an agent decision record."""
        return await self._write(
            """INSERT INTO agent_decisions (
                session_id, agent_role, decision_type, decision_outcome,
                reasoning, inputs_json, metrics_json, iteration, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id, agent_role, decision_type, decision_outcome,
                reasoning[:500] if reasoning else None,
                inputs_json, metrics_json, iteration,
                datetime.now().isoformat(),
            ),
            op="save_agent_decision",
        )

    async def save_agent_decisions_batch(
        self,
        decisions: list[dict],
    ) -> int:
        """Save multiple agent decisions in a batch."""
        if not decisions:
            return 0
        return await self._write_many(
            """INSERT INTO agent_decisions (
                session_id, agent_role, decision_type, decision_outcome,
                reasoning, inputs_json, metrics_json, iteration, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    d.get("session_id"), d.get("agent_role"),
                    d.get("decision_type"), d.get("decision_outcome"),
                    d.get("reasoning", "")[:500] if d.get("reasoning") else None,
                    d.get("inputs_json"), d.get("metrics_json"),
                    d.get("iteration"),
                    d.get("created_at", datetime.now().isoformat()),
                )
                for d in decisions
            ],
            op="save_agent_decisions_batch",
        )

    async def get_agent_decisions(
        self,
        session_id: str,
        agent_role: str | None = None,
        decision_type: str | None = None,
    ) -> list[dict]:
        """Get agent decisions for a session, optionally filtered."""
        query = "SELECT * FROM agent_decisions WHERE session_id = ?"
        params: list = [session_id]
        if agent_role:
            query += " AND agent_role = ?"
            params.append(agent_role)
        if decision_type:
            query += " AND decision_type = ?"
            params.append(decision_type)
        query += " ORDER BY created_at"
        rows = await self._fetch_all(query, tuple(params))
        return [dict(row) for row in rows]

    async def get_decision_stats(self, session_id: str) -> dict:
        """Get decision statistics for a session."""
        rows = await self._fetch_all(
            """SELECT agent_role, decision_type, COUNT(*) as count
            FROM agent_decisions WHERE session_id = ?
            GROUP BY agent_role, decision_type""",
            (session_id,),
        )

        stats = {"by_agent": {}, "by_type": {}, "total": 0}
        for row in rows:
            agent = row["agent_role"]
            dtype = row["decision_type"]
            count = row["count"]

            stats["by_agent"].setdefault(agent, 0)
            stats["by_agent"][agent] += count

            stats["by_type"].setdefault(dtype, 0)
            stats["by_type"][dtype] += count

            stats["total"] += count

        return stats
