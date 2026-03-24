"""
Knowledge graph query wrapper for API endpoints.

Reads from the veritas_kg.db SQLite database (synchronous) via asyncio executor.
"""
import asyncio
import json
import sqlite3
from functools import partial
from pathlib import Path
from typing import Any

# Default KG database path (relative to project root)
_DEFAULT_KG_PATH = Path("veritas_kg.db")


def _query_sync(db_path: str, sql: str, params: tuple = ()) -> list[dict]:
    """Execute a read-only query synchronously and return list of dicts."""
    conn = sqlite3.connect(db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        cursor = conn.execute(sql, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def _scalar_sync(db_path: str, sql: str, params: tuple = ()) -> Any:
    """Execute a query and return a single scalar value."""
    conn = sqlite3.connect(db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        cursor = conn.execute(sql, params)
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


class KnowledgeGraphAPI:
    """Async wrapper around the KG SQLite database."""

    def __init__(self, db_path: str | None = None):
        self.db_path = str(db_path or _DEFAULT_KG_PATH)

    def _db_exists(self) -> bool:
        return Path(self.db_path).exists()

    async def _run(self, fn, *args):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, partial(fn, self.db_path, *args))

    async def get_entities(self, session_id: str | None = None, entity_type: str | None = None, limit: int = 500) -> list[dict]:
        """Get all KG entities, optionally filtered by session and type."""
        if not self._db_exists():
            return []

        # Build query based on filters
        conditions = []
        params = []

        if session_id:
            conditions.append("(session_id = ? OR session_id IS NULL OR session_id = '')")
            params.append(session_id)

        if entity_type:
            conditions.append("entity_type = ?")
            params.append(entity_type)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        rows = await self._run(
            _query_sync,
            f"""
            SELECT id, name, entity_type, properties, session_id, created_at
            FROM kg_entities
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            tuple(params),
        )

        # Parse properties JSON
        for row in rows:
            try:
                if row.get("properties"):
                    row["properties"] = json.loads(row["properties"])
                else:
                    row["properties"] = {}
            except Exception:
                row["properties"] = {}
        return rows

    async def get_relations(self, session_id: str | None = None, limit: int = 1000) -> list[dict]:
        """Get all KG relations, optionally filtered by session."""
        if not self._db_exists():
            return []

        where_clause = "WHERE (r.session_id = ? OR r.session_id IS NULL OR r.session_id = '')" if session_id else ""
        params = (session_id, limit) if session_id else (limit,)

        rows = await self._run(
            _query_sync,
            f"""
            SELECT
                r.id, r.subject_id, r.predicate, r.object_id,
                r.source_id, r.confidence, r.properties, r.session_id, r.created_at,
                s.name AS subject_name, s.entity_type AS subject_type,
                o.name AS object_name, o.entity_type AS object_type
            FROM kg_relations r
            LEFT JOIN kg_entities s ON s.id = r.subject_id
            LEFT JOIN kg_entities o ON o.id = r.object_id
            {where_clause}
            ORDER BY r.created_at DESC
            LIMIT ?
            """,
            params,
        )
        for row in rows:
            try:
                if row.get("properties"):
                    row["properties"] = json.loads(row["properties"])
                else:
                    row["properties"] = {}
            except Exception:
                row["properties"] = {}
        return rows

    async def get_contradictions(self, session_id: str | None = None, limit: int = 100) -> list[dict]:
        """Get KG contradictions, optionally filtered by session."""
        if not self._db_exists():
            return []

        where_clause = "WHERE (session_id = ? OR session_id IS NULL OR session_id = '')" if session_id else ""
        params = (session_id, limit) if session_id else (limit,)

        return await self._run(
            _query_sync,
            f"""
            SELECT id, relation1_id, relation2_id, contradiction_type,
                   description, severity, resolution, resolved_at, session_id, created_at
            FROM kg_contradictions
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            params,
        )

    async def get_stats(self, session_id: str | None = None) -> dict:
        """Get KG statistics, optionally filtered by session."""
        if not self._db_exists():
            return {"entities": 0, "relations": 0, "contradictions": 0, "entity_types": {}}

        where = "WHERE (session_id = ? OR session_id IS NULL OR session_id = '')" if session_id else ""
        params = (session_id,) if session_id else ()

        entities_row = await self._run(
            _scalar_sync,
            f"SELECT COUNT(*) as count FROM kg_entities {where}",
            params,
        )
        relations_row = await self._run(
            _scalar_sync,
            f"SELECT COUNT(*) as count FROM kg_relations {where}",
            params,
        )
        contradictions_row = await self._run(
            _scalar_sync,
            f"SELECT COUNT(*) as count FROM kg_contradictions {where}",
            params,
        )

        # Entity type breakdown
        type_rows = await self._run(
            _query_sync,
            f"SELECT entity_type, COUNT(*) as count FROM kg_entities {where} GROUP BY entity_type ORDER BY count DESC",
            params,
        )
        entity_types = {row["entity_type"]: row["count"] for row in type_rows}

        return {
            "entities": entities_row["count"] if entities_row else 0,
            "relations": relations_row["count"] if relations_row else 0,
            "contradictions": contradictions_row["count"] if contradictions_row else 0,
            "entity_types": entity_types,
        }

    async def get_entity(self, entity_id: str) -> dict | None:
        """Get a single entity by ID with its relations."""
        if not self._db_exists():
            return None
        rows = await self._run(
            _query_sync,
            "SELECT id, name, entity_type, properties, created_at FROM kg_entities WHERE id = ?",
            (entity_id,),
        )
        if not rows:
            return None
        entity = rows[0]
        try:
            if entity.get("properties"):
                entity["properties"] = json.loads(entity["properties"])
            else:
                entity["properties"] = {}
        except Exception:
            entity["properties"] = {}

        # Get relations for this entity
        relations = await self._run(
            _query_sync,
            """
            SELECT r.id, r.subject_id, r.predicate, r.object_id, r.confidence,
                   s.name AS subject_name, o.name AS object_name
            FROM kg_relations r
            LEFT JOIN kg_entities s ON s.id = r.subject_id
            LEFT JOIN kg_entities o ON o.id = r.object_id
            WHERE r.subject_id = ? OR r.object_id = ?
            """,
            (entity_id, entity_id),
        )
        entity["relations"] = relations
        return entity


# Global instance
_kg_instance: KnowledgeGraphAPI | None = None


def get_kg() -> KnowledgeGraphAPI:
    """Get or create the global KG API instance."""
    global _kg_instance
    if _kg_instance is None:
        _kg_instance = KnowledgeGraphAPI()
    return _kg_instance
