"""External memory storage for evidence and context overflow."""

import asyncio
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import aiosqlite

from ..logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class StoredMemory:
    """A memory item stored externally."""
    id: str
    session_id: str
    content: str
    memory_type: str  # 'evidence', 'summary', 'context', 'decision'
    tags: list[str]
    created_at: datetime
    metadata: dict

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'session_id': self.session_id,
            'content': self.content,
            'memory_type': self.memory_type,
            'tags': self.tags,
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata,
        }


class ExternalMemoryStore:
    """External storage for long-term memory and context overflow.

    Based on Anthropic's approach of storing plan details and subagent
    outputs to filesystem/database when context approaches limits.

    Provides:
    - Persistent storage of evidence and context
    - Searchable archive by tags and content
    - Session-scoped memory management

    Requires explicit connect()/close() lifecycle, or lazy auto-connect
    via _ensure_connected().
    """

    def __init__(self, db_path: str = "memory.db"):
        """Initialize external memory store.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self._connection: aiosqlite.Connection | None = None
        self._connect_lock = asyncio.Lock()

    async def connect(self, _retries: int = 3) -> None:
        """Open persistent DB connection, create schema."""
        logger.info("External memory connecting: %s", self.db_path)
        for attempt in range(1, _retries + 1):
            self._connection = await aiosqlite.connect(self.db_path)
            try:
                await self._connection.execute("PRAGMA busy_timeout=10000")
                await self._connection.execute("PRAGMA journal_mode=WAL")
                await self._init_schema()
                return
            except Exception:
                await self._connection.close()
                self._connection = None
                if attempt < _retries:
                    logger.warning("External memory connect attempt %d failed, retrying...", attempt)
                    await asyncio.sleep(0.5 * attempt)
                else:
                    raise

    async def close(self) -> None:
        """Close the persistent DB connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None

    async def _ensure_connected(self) -> aiosqlite.Connection:
        """Return the persistent connection, lazily connecting if needed."""
        if self._connection is not None:
            return self._connection
        async with self._connect_lock:
            # Double-check after acquiring lock
            if self._connection is not None:
                return self._connection
            await self.connect()
        return self._connection

    async def _init_schema(self) -> None:
        """Initialize database schema on the persistent connection."""
        conn = self._connection
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                tags TEXT,
                created_at TEXT NOT NULL,
                metadata TEXT
            )
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memory_session
            ON memories(session_id)
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memory_type
            ON memories(memory_type)
        """)

        # Full-text search table
        await conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
            USING fts5(content, tags, tokenize='porter')
        """)

        await conn.commit()

    async def store(
        self,
        session_id: str,
        content: str,
        memory_type: str,
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Store content to external memory.

        Args:
            session_id: Current research session ID
            content: Content to store
            memory_type: Type of memory (evidence, summary, context, decision)
            tags: Optional tags for retrieval
            metadata: Optional metadata

        Returns:
            ID of stored memory
        """
        conn = await self._ensure_connected()

        memory_id = str(uuid.uuid4())[:8]
        tags = tags or []
        metadata = metadata or {}
        created_at = datetime.now()

        try:
            await conn.execute("""
                INSERT INTO memories
                    (id, session_id, content, memory_type,
                     tags, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                memory_id,
                session_id,
                content,
                memory_type,
                json.dumps(tags),
                created_at.isoformat(),
                json.dumps(metadata),
            ))

            # Use explicit rowid lookup instead of last_insert_rowid()
            await conn.execute("""
                INSERT INTO memories_fts (rowid, content, tags)
                VALUES (
                    (SELECT rowid FROM memories WHERE id = ?),
                    ?, ?
                )
            """, (memory_id, content, ' '.join(tags)))

            await conn.commit()
        except Exception:
            await conn.rollback()
            logger.error("External memory store error", exc_info=True)
            raise

        return memory_id

    # Synchronous wrapper for backward compatibility (runs in executor)
    def store_sync(
        self,
        session_id: str,
        content: str,
        memory_type: str,
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Synchronous store for backward compatibility.

        Note: Prefer using the async `store` method when possible.
        """
        import asyncio as _asyncio
        try:
            loop = _asyncio.get_running_loop()
            future = _asyncio.run_coroutine_threadsafe(
                self.store(
                    session_id, content, memory_type, tags, metadata
                ),
                loop,
            )
            return future.result(timeout=30)
        except RuntimeError:
            return _asyncio.run(
                self.store(
                    session_id, content, memory_type, tags, metadata
                )
            )

    async def search(
        self,
        query: str,
        session_id: str | None = None,
        memory_type: str | None = None,
        limit: int = 10,
    ) -> list[StoredMemory]:
        """Search external memory.

        Args:
            query: Search query (full-text search)
            session_id: Optional filter by session
            memory_type: Optional filter by type
            limit: Maximum results to return

        Returns:
            List of matching memories
        """
        conn = await self._ensure_connected()

        sql = """
            SELECT m.id, m.session_id, m.content,
                   m.memory_type, m.tags, m.created_at,
                   m.metadata
            FROM memories m
            JOIN memories_fts fts ON m.rowid = fts.rowid
            WHERE memories_fts MATCH ?
        """
        params: list = [query]

        if session_id is not None:
            sql += " AND m.session_id = ?"
            params.append(session_id)

        if memory_type is not None:
            sql += " AND m.memory_type = ?"
            params.append(memory_type)

        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)

        try:
            cursor = await conn.execute(sql, params)
            rows = await cursor.fetchall()
        except aiosqlite.OperationalError:
            # FTS query failed, fall back to LIKE
            sql = """
                SELECT id, session_id, content,
                       memory_type, tags, created_at, metadata
                FROM memories
                WHERE content LIKE ?
            """
            params = [f"%{query}%"]

            if session_id is not None:
                sql += " AND session_id = ?"
                params.append(session_id)

            if memory_type is not None:
                sql += " AND memory_type = ?"
                params.append(memory_type)

            sql += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            cursor = await conn.execute(sql, params)
            rows = await cursor.fetchall()

        return [self._row_to_memory(row) for row in rows]

    async def get_by_session(
        self,
        session_id: str,
        memory_type: str | None = None,
    ) -> list[StoredMemory]:
        """Get all memories for a session.

        Args:
            session_id: Session ID
            memory_type: Optional filter by type

        Returns:
            List of memories
        """
        conn = await self._ensure_connected()

        if memory_type:
            cursor = await conn.execute("""
                SELECT id, session_id, content,
                       memory_type, tags, created_at, metadata
                FROM memories
                WHERE session_id = ? AND memory_type = ?
                ORDER BY created_at
            """, (session_id, memory_type))
        else:
            cursor = await conn.execute("""
                SELECT id, session_id, content,
                       memory_type, tags, created_at, metadata
                FROM memories
                WHERE session_id = ?
                ORDER BY created_at
            """, (session_id,))

        rows = await cursor.fetchall()
        return [self._row_to_memory(row) for row in rows]

    async def get_recent(
        self,
        session_id: str,
        limit: int = 10,
    ) -> list[StoredMemory]:
        """Get most recent memories for a session.

        Args:
            session_id: Session ID
            limit: Maximum results

        Returns:
            List of recent memories
        """
        conn = await self._ensure_connected()

        cursor = await conn.execute("""
            SELECT id, session_id, content,
                   memory_type, tags, created_at, metadata
            FROM memories
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (session_id, limit))

        rows = await cursor.fetchall()
        return [self._row_to_memory(row) for row in rows]

    async def delete_session(self, session_id: str) -> int:
        """Delete all memories for a session.

        Args:
            session_id: Session to delete

        Returns:
            Number of memories deleted
        """
        conn = await self._ensure_connected()

        cursor = await conn.execute(
            "SELECT COUNT(*) FROM memories WHERE session_id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        count = row[0] if row else 0

        try:
            await conn.execute(
                "DELETE FROM memories WHERE session_id = ?",
                (session_id,),
            )
            await conn.commit()
        except Exception:
            await conn.rollback()
            logger.error("External memory delete error", exc_info=True)
            raise

        return count

    async def get_stats(self) -> dict:
        """Get memory store statistics."""
        conn = await self._ensure_connected()

        cursor = await conn.execute("SELECT COUNT(*) FROM memories")
        row = await cursor.fetchone()
        total = row[0] if row else 0

        cursor = await conn.execute("""
            SELECT memory_type, COUNT(*) as count
            FROM memories
            GROUP BY memory_type
        """)
        rows = await cursor.fetchall()
        by_type = {row[0]: row[1] for row in rows}

        cursor = await conn.execute(
            "SELECT COUNT(DISTINCT session_id) FROM memories"
        )
        row = await cursor.fetchone()
        sessions = row[0] if row else 0

        return {
            'total_memories': total,
            'by_type': by_type,
            'sessions': sessions,
        }

    @staticmethod
    def _row_to_memory(row) -> StoredMemory:
        """Convert a database row to a StoredMemory."""
        return StoredMemory(
            id=row[0],
            session_id=row[1],
            content=row[2],
            memory_type=row[3],
            tags=json.loads(row[4]) if row[4] else [],
            created_at=datetime.fromisoformat(row[5]),
            metadata=json.loads(row[6]) if row[6] else {},
        )
