"""Hybrid NetworkX + SQLite storage for knowledge graphs."""

import asyncio
import json
from datetime import datetime
from pathlib import Path

import aiosqlite

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

from ..logging_config import get_logger
from .models import Contradiction, Entity, Relation

logger = get_logger(__name__)


class HybridKnowledgeGraphStore:
    """Hybrid storage: NetworkX for in-memory graph algorithms, SQLite for persistence.

    This provides:
    - Fast graph queries via NetworkX (PageRank, centrality, paths)
    - Persistent storage via SQLite (async via aiosqlite)
    - Automatic sync between the two

    Requires explicit connect()/close() lifecycle.
    """

    def __init__(self, db_path: str = "knowledge_graph.db", session_id: str | None = None):
        self.db_path = Path(db_path)
        self.session_id = session_id

        if HAS_NETWORKX:
            self.graph = nx.DiGraph()
        else:
            self.graph = None  # Fallback mode

        self._connection: aiosqlite.Connection | None = None

    async def connect(self, _retries: int = 3):
        """Initialize async DB connection, create schema, and load graph."""
        logger.info("KG store connecting: %s", self.db_path)
        for attempt in range(1, _retries + 1):
            self._connection = await aiosqlite.connect(self.db_path)
            try:
                await self._connection.execute("PRAGMA busy_timeout=10000")
                await self._connection.execute("PRAGMA journal_mode=WAL")
                await self._init_db()
                await self._load_from_db()
                return
            except Exception:
                await self._connection.close()
                self._connection = None
                if attempt < _retries:
                    logger.warning("KG store connect attempt %d failed, retrying...", attempt)
                    await asyncio.sleep(0.5 * attempt)
                else:
                    raise

    async def close(self):
        """Close the async DB connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None

    async def _init_db(self):
        """Initialize SQLite schema."""
        conn = self._connection

        # Entities table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS kg_entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                properties TEXT,
                embedding BLOB,
                session_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Relations table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS kg_relations (
                id TEXT PRIMARY KEY,
                subject_id TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object_id TEXT NOT NULL,
                source_id TEXT,
                confidence REAL DEFAULT 1.0,
                properties TEXT,
                session_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (subject_id) REFERENCES kg_entities(id),
                FOREIGN KEY (object_id) REFERENCES kg_entities(id)
            )
        """)

        # Evidence/Sources table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS kg_evidence (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                source_url TEXT,
                source_title TEXT,
                credibility_score REAL,
                evidence_type TEXT,
                search_query TEXT,
                timestamp TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Contradictions table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS kg_contradictions (
                id TEXT PRIMARY KEY,
                relation1_id TEXT NOT NULL,
                relation2_id TEXT NOT NULL,
                contradiction_type TEXT NOT NULL,
                description TEXT,
                severity TEXT DEFAULT 'medium',
                resolution TEXT,
                resolved_at TIMESTAMP,
                session_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (relation1_id) REFERENCES kg_relations(id),
                FOREIGN KEY (relation2_id) REFERENCES kg_relations(id)
            )
        """)

        # Indexes for fast queries
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entity_type ON kg_entities(entity_type)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entity_name ON kg_entities(name)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_relation_predicate ON kg_relations(predicate)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_relation_subject ON kg_relations(subject_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_relation_object ON kg_relations(object_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entity_session ON kg_entities(session_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_relation_session ON kg_relations(session_id)"
        )

        await conn.commit()

    async def _load_from_db(self):
        """Load graph from SQLite into NetworkX.

        When session_id is set, only entities and relations belonging to that
        session are loaded, preventing cross-session contamination.  When
        session_id is None, all rows are loaded (backwards-compatible).
        """
        if not HAS_NETWORKX or self.graph is None:
            return

        conn = self._connection

        # Load entities as nodes (filtered by session_id when set)
        if self.session_id is not None:
            cursor = await conn.execute(
                "SELECT id, name, entity_type, properties FROM kg_entities WHERE session_id = ?",
                (self.session_id,),
            )
        else:
            cursor = await conn.execute(
                "SELECT id, name, entity_type, properties FROM kg_entities"
            )
        for row in await cursor.fetchall():
            entity_id, name, entity_type, properties = row
            props = json.loads(properties) if properties else {}
            self.graph.add_node(
                entity_id,
                name=name,
                entity_type=entity_type,
                **props
            )

        # Load relations as edges (filtered by session_id when set)
        if self.session_id is not None:
            cursor = await conn.execute("""
                SELECT id, subject_id, predicate, object_id, confidence, properties
                FROM kg_relations
                WHERE session_id = ?
            """, (self.session_id,))
        else:
            cursor = await conn.execute("""
                SELECT id, subject_id, predicate, object_id, confidence, properties
                FROM kg_relations
            """)
        for row in await cursor.fetchall():
            rel_id, subj, pred, obj, conf, properties = row
            props = json.loads(properties) if properties else {}
            if subj in self.graph and obj in self.graph:
                self.graph.add_edge(
                    subj, obj,
                    relation_id=rel_id,
                    predicate=pred,
                    confidence=conf,
                    **props
                )

    async def add_entity(self, entity: Entity, session_id: str | None = None) -> str:
        """Add entity to both NetworkX and SQLite."""
        conn = self._connection

        # [HARDENED] SEC-002: Use JSON instead of pickle for embedding serialization
        embedding_blob = (
            json.dumps(entity.embedding.tolist())
            if entity.embedding is not None
            else None
        )
        properties = json.dumps({
            'aliases': entity.aliases,
            'sources': entity.sources,
            'confidence': entity.confidence,
            **entity.properties,
        })

        try:
            await conn.execute("""
                INSERT OR REPLACE INTO kg_entities
                (id, name, entity_type, properties, embedding, session_id, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                entity.id, entity.name, entity.entity_type, properties,
                embedding_blob, session_id, datetime.now().isoformat(),
            ))

            # Add to NetworkX before committing so we can rollback on failure
            if HAS_NETWORKX and self.graph is not None:
                self.graph.add_node(
                    entity.id,
                    name=entity.name,
                    entity_type=entity.entity_type,
                    aliases=entity.aliases,
                    sources=entity.sources,
                    confidence=entity.confidence,
                    session_id=session_id,
                    **entity.properties
                )

            await conn.commit()
        except Exception:
            await conn.rollback()
            # Remove from NetworkX if it was added
            if HAS_NETWORKX and self.graph is not None and entity.id in self.graph:
                self.graph.remove_node(entity.id)
            raise

        return entity.id

    async def add_relation(
        self, relation: Relation, session_id: str | None = None,
    ) -> str:
        """Add relation to both NetworkX and SQLite."""
        conn = self._connection

        properties = json.dumps({
            'timestamp': relation.timestamp,
            **relation.properties,
        })

        try:
            await conn.execute("""
                INSERT INTO kg_relations
                (id, subject_id, predicate, object_id, source_id, confidence, properties, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                relation.id, relation.subject_id, relation.predicate,
                relation.object_id, relation.source_id, relation.confidence,
                properties, session_id
            ))

            # Add to NetworkX before committing so we can rollback on failure
            if HAS_NETWORKX and self.graph is not None:
                self.graph.add_edge(
                    relation.subject_id,
                    relation.object_id,
                    relation_id=relation.id,
                    predicate=relation.predicate,
                    confidence=relation.confidence,
                    source_id=relation.source_id,
                    session_id=session_id
                )

            await conn.commit()
        except Exception:
            await conn.rollback()
            # Remove edge from NetworkX if it was added
            if HAS_NETWORKX and self.graph is not None:
                try:
                    self.graph.remove_edge(relation.subject_id, relation.object_id)
                except Exception:
                    pass
            raise

        return relation.id

    async def add_contradiction(
        self, contradiction: Contradiction, session_id: str | None = None,
    ) -> str:
        """Record a detected contradiction."""
        conn = self._connection

        await conn.execute("""
            INSERT INTO kg_contradictions
            (id, relation1_id, relation2_id, contradiction_type, description, severity, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            contradiction.id,
            contradiction.relation1_id,
            contradiction.relation2_id,
            contradiction.contradiction_type,
            contradiction.description,
            contradiction.severity,
            session_id,
        ))
        await conn.commit()
        return contradiction.id

    async def get_entity(self, entity_id: str) -> Entity | None:
        """Get an entity by ID."""
        conn = self._connection

        cursor = await conn.execute(
            "SELECT id, name, entity_type, properties, embedding FROM kg_entities WHERE id = ?",
            (entity_id,)
        )
        row = await cursor.fetchone()

        if not row:
            return None

        entity_id, name, entity_type, properties, embedding_blob = row
        props = json.loads(properties) if properties else {}

        # [HARDENED] SEC-002: Use JSON instead of pickle for embedding deserialization
        embedding = None
        if embedding_blob:
            try:
                import numpy as np
                embedding = np.array(json.loads(embedding_blob))
            except (json.JSONDecodeError, TypeError, ValueError):
                logger.warning(
                    "KG store: corrupted embedding blob for entity %s",
                    entity_id, exc_info=True,
                )
                pass  # Corrupted or old-format blob -- skip embedding

        return Entity(
            id=entity_id,
            name=name,
            entity_type=entity_type,
            aliases=props.get('aliases', []),
            sources=props.get('sources', []),
            confidence=props.get('confidence', 1.0),
            embedding=embedding,
            properties={
                k: v for k, v in props.items()
                if k not in ['aliases', 'sources', 'confidence']
            },
        )

    async def query_by_entity_type(self, entity_type: str) -> list[dict]:
        """Query entities by type using SQLite."""
        conn = self._connection

        cursor = await conn.execute(
            "SELECT id, name, properties FROM kg_entities WHERE entity_type = ?",
            (entity_type,)
        )

        results = []
        for row in await cursor.fetchall():
            entity_id, name, properties = row
            props = json.loads(properties) if properties else {}
            results.append({
                'id': entity_id,
                'name': name,
                'type': entity_type,
                **props
            })
        return results

    async def get_entity_relations(self, entity_id: str) -> dict:
        """Get all relations for an entity."""
        if not HAS_NETWORKX or self.graph is None or entity_id not in self.graph:
            # Fallback to SQLite
            return await self._get_entity_relations_sql(entity_id)

        outgoing = []
        for _, target, data in self.graph.out_edges(entity_id, data=True):
            outgoing.append({
                'relation_id': data.get('relation_id', ''),
                'predicate': data.get('predicate', ''),
                'target_id': target,
                'target_name': self.graph.nodes[target].get('name', target),
                'confidence': data.get('confidence', 1.0)
            })

        incoming = []
        for source, _, data in self.graph.in_edges(entity_id, data=True):
            incoming.append({
                'relation_id': data.get('relation_id', ''),
                'predicate': data.get('predicate', ''),
                'source_id': source,
                'source_name': self.graph.nodes[source].get('name', source),
                'confidence': data.get('confidence', 1.0)
            })

        return {'outgoing': outgoing, 'incoming': incoming}

    async def _get_entity_relations_sql(self, entity_id: str) -> dict:
        """Get entity relations via SQLite (fallback)."""
        conn = self._connection

        # Outgoing relations
        cursor = await conn.execute("""
            SELECT r.id, r.predicate, r.object_id, e.name, r.confidence
            FROM kg_relations r
            JOIN kg_entities e ON r.object_id = e.id
            WHERE r.subject_id = ?
        """, (entity_id,))

        outgoing = [
            {
                'relation_id': row[0], 'predicate': row[1], 'target_id': row[2],
                'target_name': row[3], 'confidence': row[4],
            }
            for row in await cursor.fetchall()
        ]

        # Incoming relations
        cursor = await conn.execute("""
            SELECT r.id, r.predicate, r.subject_id, e.name, r.confidence
            FROM kg_relations r
            JOIN kg_entities e ON r.subject_id = e.id
            WHERE r.object_id = ?
        """, (entity_id,))

        incoming = [
            {
                'relation_id': row[0], 'predicate': row[1], 'source_id': row[2],
                'source_name': row[3], 'confidence': row[4],
            }
            for row in await cursor.fetchall()
        ]
        return {'outgoing': outgoing, 'incoming': incoming}

    async def get_unresolved_contradictions(self) -> list[dict]:
        """Get all unresolved contradictions."""
        conn = self._connection

        cursor = await conn.execute("""
            SELECT id, relation1_id, relation2_id, contradiction_type, description, severity
            FROM kg_contradictions
            WHERE resolved_at IS NULL
        """)

        contradictions = []
        for row in await cursor.fetchall():
            contradictions.append({
                'id': row[0],
                'relation1_id': row[1],
                'relation2_id': row[2],
                'type': row[3],
                'description': row[4],
                'severity': row[5],
            })
        return contradictions

    async def get_stats(self) -> dict:
        """Get knowledge graph statistics."""
        if HAS_NETWORKX and self.graph is not None:
            num_entities = self.graph.number_of_nodes()
            num_relations = self.graph.number_of_edges()
            num_components = (
                nx.number_weakly_connected_components(self.graph)
                if num_entities > 0 else 0
            )
            density = nx.density(self.graph) if num_entities > 1 else 0
        else:
            conn = self._connection
            cursor = await conn.execute("SELECT COUNT(*) FROM kg_entities")
            row = await cursor.fetchone()
            num_entities = row[0]
            cursor = await conn.execute("SELECT COUNT(*) FROM kg_relations")
            row = await cursor.fetchone()
            num_relations = row[0]
            num_components = 0
            density = 0

        # Get entity type counts
        type_counts = {}
        if HAS_NETWORKX and self.graph is not None:
            for _, data in self.graph.nodes(data=True):
                etype = data.get('entity_type', 'UNKNOWN')
                type_counts[etype] = type_counts.get(etype, 0) + 1

        return {
            'num_entities': num_entities,
            'num_relations': num_relations,
            'num_components': num_components,
            'density': density,
            'entity_types': type_counts,
        }

    async def clear(self):
        """Clear all data from the knowledge graph."""
        conn = self._connection

        await conn.execute("DELETE FROM kg_contradictions")
        await conn.execute("DELETE FROM kg_relations")
        await conn.execute("DELETE FROM kg_entities")
        await conn.execute("DELETE FROM kg_evidence")
        await conn.commit()

        if HAS_NETWORKX and self.graph is not None:
            self.graph.clear()
