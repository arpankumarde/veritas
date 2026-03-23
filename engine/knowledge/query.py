"""Manager query interface for knowledge graph analysis."""

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

from ..logging_config import get_logger
from .models import KnowledgeGap
from .store import HybridKnowledgeGraphStore

logger = get_logger(__name__)


class ManagerQueryInterface:
    """Interface for Manager agent to query knowledge graph.

    Answers questions like:
    - "What do I know about X?"
    - "What's missing?"
    - "What contradictions exist?"
    - "What are the key concepts?"
    """

    def __init__(self, kg_store: HybridKnowledgeGraphStore):
        self.store = kg_store

    async def what_do_i_know_about(self, topic: str) -> dict:
        """Get all knowledge related to a topic.

        Args:
            topic: The topic to query

        Returns:
            Dict with entities, claims, evidence, relations, and sources
        """
        if not HAS_NETWORKX or self.store.graph is None:
            return await self._what_do_i_know_sql(topic)

        # Find entities matching the topic
        matching_entities = []
        topic_lower = topic.lower()

        for node_id, data in self.store.graph.nodes(data=True):
            name = data.get('name', '').lower()
            aliases = [a.lower() for a in data.get('aliases', [])]

            if topic_lower in name or any(topic_lower in a for a in aliases):
                matching_entities.append(node_id)

        if not matching_entities:
            return {
                'found': False,
                'message': f"No knowledge about '{topic}' found yet.",
                'entities': [],
                'claims': [],
                'evidence': [],
                'relations': [],
                'sources': [],
            }

        # Gather all related information
        knowledge = {
            'found': True,
            'entities': [],
            'claims': [],
            'evidence': [],
            'relations': [],
            'sources': set()
        }

        for entity_id in matching_entities:
            data = self.store.graph.nodes[entity_id]
            entity_info = {
                'id': entity_id,
                'name': data.get('name'),
                'type': data.get('entity_type'),
                'confidence': data.get('confidence', 1.0)
            }

            if data.get('entity_type') == 'CLAIM':
                knowledge['claims'].append(entity_info)
            elif data.get('entity_type') == 'EVIDENCE':
                knowledge['evidence'].append(entity_info)
            else:
                knowledge['entities'].append(entity_info)

            # Get relations
            relations = await self.store.get_entity_relations(entity_id)
            knowledge['relations'].extend(relations['outgoing'])
            knowledge['relations'].extend(relations['incoming'])

            # Track sources
            for source in data.get('sources', []):
                knowledge['sources'].add(source)

        knowledge['sources'] = list(knowledge['sources'])
        knowledge['summary'] = self._generate_summary(knowledge)

        return knowledge

    async def _what_do_i_know_sql(self, topic: str) -> dict:
        """Fallback SQL implementation using the store's async connection."""
        conn = self.store._connection

        cursor = await conn.execute("""
            SELECT id, name, entity_type, properties
            FROM kg_entities
            WHERE name LIKE ? OR properties LIKE ?
        """, (f"%{topic}%", f"%{topic}%"))

        entities = []
        for row in await cursor.fetchall():
            entities.append({
                'id': row[0],
                'name': row[1],
                'type': row[2],
            })

        return {
            'found': len(entities) > 0,
            'entities': entities,
            'claims': [],
            'evidence': [],
            'relations': [],
            'sources': [],
            'summary': f"Found {len(entities)} related entities.",
        }

    async def identify_gaps(self) -> list[KnowledgeGap]:
        """Identify knowledge gaps using graph structure analysis.

        Uses:
        - Evidence count per claim
        - Disconnected components
        - Betweenness centrality
        - Entity type coverage
        """
        gaps = []

        if not HAS_NETWORKX or self.store.graph is None or len(self.store.graph) < 2:
            return gaps

        # 1. Find claims with insufficient evidence
        claims = await self.store.query_by_entity_type('CLAIM')
        for claim in claims:
            evidence_count = sum(
                1 for _, _, d in self.store.graph.in_edges(claim['id'], data=True)
                if d.get('predicate') in ['supports', 'evidence_for']
            )

            if evidence_count < 2:
                gaps.append(KnowledgeGap(
                    gap_type='insufficient_evidence',
                    entity=claim['name'],
                    current_count=evidence_count,
                    recommendation=f"Find more evidence for: {claim['name']}",
                    importance=0.8,
                ))

        # 2. Detect structural holes (disconnected clusters)
        if len(self.store.graph) > 5:
            components = list(nx.weakly_connected_components(self.store.graph))
            if len(components) > 1:
                component_labels = [
                    self._get_component_label(c) for c in components[:3]
                ]
                gaps.append(KnowledgeGap(
                    gap_type='disconnected_topics',
                    current_count=len(components),
                    recommendation=f"Research connections between: {', '.join(component_labels)}",
                    importance=0.7,
                ))

        # 3. Find bridging concepts (high betweenness, low degree)
        if len(self.store.graph) > 3:
            try:
                betweenness = nx.betweenness_centrality(self.store.graph)
                degree = dict(self.store.graph.degree())

                for node_id, bc in betweenness.items():
                    if bc > 0.2 and degree.get(node_id, 0) < 4:
                        data = self.store.graph.nodes[node_id]
                        gaps.append(KnowledgeGap(
                            gap_type='bridging_concept',
                            entity=data.get('name'),
                            recommendation=f"'{data.get('name')}' bridges important concepts. Explore it further.",
                            importance=bc,
                        ))
            except Exception:
                logger.debug("Betweenness centrality computation failed", exc_info=True)
                pass  # Graph too small or disconnected

        # 4. Find missing entity types
        type_counts = {}
        for _, data in self.store.graph.nodes(data=True):
            etype = data.get('entity_type', 'UNKNOWN')
            type_counts[etype] = type_counts.get(etype, 0) + 1

        expected_types = ['CLAIM', 'EVIDENCE', 'METHOD', 'METRIC']
        for etype in expected_types:
            if type_counts.get(etype, 0) < 2:
                gaps.append(KnowledgeGap(
                    gap_type='missing_entity_type',
                    entity_type=etype,
                    current_count=type_counts.get(etype, 0),
                    recommendation=f"Find more {etype.lower()}s in the research.",
                    importance=0.5,
                ))

        # Sort by importance
        gaps.sort(key=lambda g: g.importance, reverse=True)
        return gaps

    async def get_contradictions(self) -> list[dict]:
        """Get all unresolved contradictions."""
        contradictions = await self.store.get_unresolved_contradictions()

        result = []
        for c in contradictions:
            result.append({
                'id': c['id'],
                'type': c['type'],
                'description': c['description'],
                'severity': c['severity'],
                'recommendation': f"Resolve contradiction: {c['description']}",
            })

        return result

    def get_key_concepts(self, top_n: int = 10) -> list[dict]:
        """Get most important concepts by centrality."""
        if not HAS_NETWORKX or self.store.graph is None or len(self.store.graph) == 0:
            return []

        try:
            # Use PageRank for importance
            pagerank = nx.pagerank(self.store.graph)
        except Exception:
            logger.debug("PageRank computation failed", exc_info=True)
            return []

        # Sort by importance
        sorted_nodes = sorted(
            pagerank.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        key_concepts = []
        for node_id, score in sorted_nodes:
            data = self.store.graph.nodes[node_id]
            key_concepts.append({
                'id': node_id,
                'name': data.get('name'),
                'type': data.get('entity_type'),
                'importance': round(score, 4),
                'connections': self.store.graph.degree(node_id)
            })

        return key_concepts

    async def get_research_summary(self) -> str:
        """Generate a summary of current knowledge state for the Manager."""
        stats = await self.store.get_stats()

        summary_parts = []
        summary_parts.append("## Knowledge Graph Status")
        summary_parts.append(f"- Entities: {stats['num_entities']}")
        summary_parts.append(f"- Relations: {stats['num_relations']}")

        if stats.get('num_components', 0) > 1:
            summary_parts.append(f"- Disconnected clusters: {stats['num_components']}")

        # Key concepts
        key_concepts = self.get_key_concepts(5)
        if key_concepts:
            summary_parts.append("\n## Key Concepts")
            for c in key_concepts:
                summary_parts.append(f"- {c['name']} ({c['type']}, importance: {c['importance']})")

        # Gaps
        gaps = await self.identify_gaps()
        if gaps:
            summary_parts.append(f"\n## Knowledge Gaps ({len(gaps)} identified)")
            for g in gaps[:5]:
                summary_parts.append(f"- {g.recommendation}")

        # Contradictions
        contradictions = await self.get_contradictions()
        if contradictions:
            summary_parts.append(f"\n## Contradictions ({len(contradictions)} unresolved)")
            for c in contradictions[:3]:
                summary_parts.append(f"- {c['recommendation']}")

        return "\n".join(summary_parts)

    async def get_next_research_directions(self) -> list[str]:
        """Suggest next research directions based on graph analysis."""
        directions = []

        # From gaps
        gaps = await self.identify_gaps()
        for gap in gaps:
            directions.append(gap.recommendation)

        # From contradictions
        contradictions = await self.get_contradictions()
        for c in contradictions:
            directions.append(c['recommendation'])

        return directions[:10]

    def _get_component_label(self, component: set) -> str:
        """Get a label for a graph component."""
        if not component:
            return "empty"
        sample_id = next(iter(component))
        if HAS_NETWORKX and self.store.graph is not None and sample_id in self.store.graph:
            return self.store.graph.nodes[sample_id].get('name', str(sample_id))
        return str(sample_id)

    def _generate_summary(self, knowledge: dict) -> str:
        """Generate natural language summary of knowledge."""
        parts = []

        if knowledge['entities']:
            entity_names = [e['name'] for e in knowledge['entities'][:5]]
            parts.append(f"Related concepts: {', '.join(entity_names)}")

        if knowledge['claims']:
            parts.append(f"Claims found: {len(knowledge['claims'])}")

        if knowledge['evidence']:
            parts.append(f"Evidence pieces: {len(knowledge['evidence'])}")

        parts.append(f"From {len(knowledge['sources'])} sources")

        return ". ".join(parts) + "." if parts else "No information found."
