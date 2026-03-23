"""Knowledge graph visualization using Mermaid diagrams."""

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

from ..logging_config import get_logger
from .store import HybridKnowledgeGraphStore

logger = get_logger(__name__)


class KnowledgeGraphVisualizer:
    """Visualize knowledge graph using Mermaid diagrams and text stats."""

    def __init__(self, kg_store: HybridKnowledgeGraphStore):
        self.store = kg_store

    def create_mermaid_diagram(self, max_nodes: int = 30) -> str:
        """Generate Mermaid diagram syntax for embedding in reports.

        Args:
            max_nodes: Maximum number of nodes to include

        Returns:
            Mermaid diagram as a string
        """
        if not HAS_NETWORKX or self.store.graph is None or len(self.store.graph) == 0:
            return "```mermaid\ngraph TD\n    empty[No data yet]\n```"

        lines = ["```mermaid", "graph TD"]

        # Get most important nodes if too many
        if len(self.store.graph) > max_nodes:
            try:
                pagerank = nx.pagerank(self.store.graph)
                top_nodes = sorted(
                    pagerank.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:max_nodes]
                included_nodes = {n[0] for n in top_nodes}
            except Exception:
                logger.debug("PageRank failed for mermaid diagram, using first %d nodes", max_nodes, exc_info=True)
                included_nodes = set(list(self.store.graph.nodes())[:max_nodes])
        else:
            included_nodes = set(self.store.graph.nodes())

        # Add nodes with styling
        for node_id in included_nodes:
            data = self.store.graph.nodes[node_id]
            name = data.get('name', node_id).replace('"', "'").replace('\n', ' ')[:40]
            entity_type = data.get('entity_type', 'DEFAULT')

            # Make ID safe for Mermaid
            safe_id = self._make_safe_id(node_id)

            # Mermaid node syntax
            if entity_type == 'CLAIM':
                lines.append(f'    {safe_id}{{"{name}"}}')  # Diamond
            elif entity_type == 'EVIDENCE':
                lines.append(f'    {safe_id}(["{name}"])')  # Stadium
            elif entity_type == 'METHOD':
                lines.append(f'    {safe_id}[/"{name}"/]')  # Parallelogram
            else:
                lines.append(f'    {safe_id}["{name}"]')    # Rectangle

        # Add edges
        for u, v, data in self.store.graph.edges(data=True):
            if u in included_nodes and v in included_nodes:
                predicate = data.get('predicate', '').replace('"', "'")[:15]
                safe_u = self._make_safe_id(u)
                safe_v = self._make_safe_id(v)
                lines.append(f'    {safe_u} -->|{predicate}| {safe_v}')

        # Add styling
        lines.extend([
            "",
            "    classDef concept fill:#6366f1,color:white",
            "    classDef claim fill:#f59e0b,color:white",
            "    classDef evidence fill:#10b981,color:white",
            "    classDef method fill:#8b5cf6,color:white",
        ])

        lines.append("```")
        return "\n".join(lines)

    async def create_summary_stats_card(self) -> str:
        """Generate a text summary card for reports."""
        stats = await self.store.get_stats()

        # Count by type
        type_counts = stats.get('entity_types', {})

        card = f"""
+--------------------------------------------+
|          KNOWLEDGE GRAPH STATS             |
+--------------------------------------------+
|  Entities: {stats['num_entities']:<30}|
|  Relations: {stats['num_relations']:<29}|
|  Components: {stats.get('num_components', 0):<28}|
|  Density: {stats.get('density', 0):.3f}                           |
+--------------------------------------------+
|  By Type:                                  |"""

        for etype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            card += f"\n|    {etype}: {count:<33}|"

        card += "\n+--------------------------------------------+"
        return card

    def _make_safe_id(self, node_id: str) -> str:
        """Make an ID safe for Mermaid."""
        # Replace problematic characters
        safe = node_id.replace('-', '_').replace(' ', '_').replace('.', '_')
        # Ensure it starts with a letter
        if safe and not safe[0].isalpha():
            safe = 'n_' + safe
        return safe[:20]  # Limit length
