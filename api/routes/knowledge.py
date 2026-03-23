"""
Knowledge graph API routes.

Serves graph data for the Knowledge Graph visualization page.
"""
from fastapi import APIRouter, Query

from api.kg import get_kg

router = APIRouter(prefix="/api/sessions", tags=["knowledge"])


@router.get("/{session_id}/knowledge/graph")
async def get_knowledge_graph(
    session_id: str,
    entity_type: str | None = None,
    limit: int = Query(default=500, ge=1, le=1000),
):
    """Get entities and relations for graph visualization."""
    kg = get_kg()
    entities = await kg.get_entities(session_id=session_id, entity_type=entity_type, limit=limit)
    relations = await kg.get_relations(session_id=session_id, limit=limit * 2)
    contradictions = await kg.get_contradictions(session_id=session_id)

    return {
        "session_id": session_id,
        "entities": entities,
        "relations": relations,
        "contradictions": contradictions,
    }


@router.get("/{session_id}/knowledge/stats")
async def get_knowledge_stats(session_id: str):
    """Get KG statistics."""
    kg = get_kg()
    return await kg.get_stats(session_id=session_id)


@router.get("/{session_id}/knowledge/entity/{entity_id}")
async def get_entity_detail(session_id: str, entity_id: str):
    """Get a single entity with its relations."""
    kg = get_kg()
    entity = await kg.get_entity(entity_id)
    if not entity:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Entity not found")
    return entity
