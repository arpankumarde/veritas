"use client";

import { useState, useEffect, useRef, useMemo, useCallback } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";

/* ── Types ──────────────────────────────────────────────── */
interface KGEntity {
    id: string;
    name: string;
    entity_type: string;
    properties: Record<string, unknown>;
    created_at: string;
}

interface KGRelation {
    id: string;
    subject_id: string;
    predicate: string;
    object_id: string;
    confidence: number;
    subject_name?: string;
    subject_type?: string;
    object_name?: string;
    object_type?: string;
}

interface KGContradiction {
    id: string;
    relation1_id: string;
    relation2_id: string;
    contradiction_type: string;
    description: string;
    severity: string;
}

/* ── Entity style map ──────────────────────────────────── */
const entityTypeConfig: Record<string, { bg: string; border: string; dimBg: string; label: string }> = {
    CLAIM:        { bg: "#f59e0b", border: "#d97706", dimBg: "rgba(245,158,11,0.15)", label: "Claim" },
    CONCEPT:      { bg: "#a78bfa", border: "#8b5cf6", dimBg: "rgba(167,139,250,0.15)", label: "Concept" },
    SOURCE:       { bg: "#64748b", border: "#475569", dimBg: "rgba(100,116,139,0.15)", label: "Source" },
    EVIDENCE:     { bg: "#34d399", border: "#10b981", dimBg: "rgba(52,211,153,0.15)", label: "Evidence" },
    TECHNOLOGY:   { bg: "#22d3ee", border: "#06b6d4", dimBg: "rgba(34,211,238,0.15)", label: "Technology" },
    METHOD:       { bg: "#a78bfa", border: "#8b5cf6", dimBg: "rgba(167,139,250,0.15)", label: "Method" },
    METRIC:       { bg: "#fb7185", border: "#f43f5e", dimBg: "rgba(251,113,133,0.15)", label: "Metric" },
    PERSON:       { bg: "#fbbf24", border: "#f59e0b", dimBg: "rgba(251,191,36,0.15)", label: "Person" },
    ORGANIZATION: { bg: "#34d399", border: "#10b981", dimBg: "rgba(52,211,153,0.15)", label: "Organization" },
    QUOTE:        { bg: "#facc15", border: "#eab308", dimBg: "rgba(250,204,21,0.15)", label: "Quote" },
};
const defaultEntityCfg = { bg: "#64748b", border: "#475569", dimBg: "rgba(100,116,139,0.15)", label: "Other" };

const predicateColors: Record<string, string> = {
    supports: "#34d399",
    contradicts: "#fb7185",
    related: "#22d3ee",
    related_to: "#22d3ee",
    is_a: "#a78bfa",
    part_of: "#a78bfa",
    causes: "#facc15",
    cites: "#64748b",
    implements: "#a78bfa",
    outperforms: "#34d399",
    similar_to: "#22d3ee",
    alternative_to: "#fbbf24",
    authored_by: "#64748b",
    mentioned_in: "#64748b",
};
const defaultEdgeColor = "#1e2440";

/* ── Graph theme (observatory dark) ──────────────────────── */
const THEME = {
    fontColor: "#e2e8f0",
    fontBg: "rgba(8, 10, 18, 0.85)",
    edgeFontColor: "#94a3b8",
    pageBg: "#080a12",
    dimmedFont: "rgba(226,232,240,0.25)",
    dimmedEdge: "rgba(30,36,58,0.3)",
};

/* ── Degree centrality helper ──────────────────────────── */
function computeDegrees(entities: KGEntity[], relations: KGRelation[]): Map<string, number> {
    const deg = new Map<string, number>();
    entities.forEach(e => deg.set(e.id, 0));
    relations.forEach(r => {
        deg.set(r.subject_id, (deg.get(r.subject_id) || 0) + 1);
        deg.set(r.object_id, (deg.get(r.object_id) || 0) + 1);
    });
    return deg;
}

/* ── Main Component ────────────────────────────────────── */
export default function KnowledgeGraphPage() {
    const params = useParams();
    const sessionId = params.id as string;

    const [entities, setEntities] = useState<KGEntity[]>([]);
    const [relations, setRelations] = useState<KGRelation[]>([]);
    const [contradictions, setContradictions] = useState<KGContradiction[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState("");
    const [selectedId, setSelectedId] = useState<string | null>(null);
    const [nodeTypeFilter, setNodeTypeFilter] = useState<string>("all");
    const [confidenceFilter, setConfidenceFilter] = useState(0);
    const [searchQuery, setSearchQuery] = useState("");
    const [searchActive, setSearchActive] = useState(false);
    const [zoomLevel, setZoomLevel] = useState(100);
    const [stabilizing, setStabilizing] = useState(false);

    const containerRef = useRef<HTMLDivElement>(null);
    const networkRef = useRef<any>(null);
    const nodesDatasetRef = useRef<any>(null);
    const edgesDatasetRef = useRef<any>(null);
    const allNodesRef = useRef<Record<string, any>>({});
    const allEdgesRef = useRef<Record<string, any>>({});
    const highlightActiveRef = useRef(false);

    /* ── Fetch data ─────────────────────────────────────── */
    const fetchGraph = async () => {
        setLoading(true);
        setError("");
        try {
            const res = await fetch(`http://localhost:9090/api/sessions/${sessionId}/knowledge/graph?limit=500`);
            if (!res.ok) throw new Error("Failed to load knowledge graph");
            const data = await res.json();
            setEntities(data.entities || []);
            setRelations(data.relations || []);
            setContradictions(data.contradictions || []);
        } catch {
            setError("Unable to load knowledge graph data");
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => { fetchGraph(); }, [sessionId]);

    /* ── Filtering ──────────────────────────────────────── */
    const filteredEntities = useMemo(() => {
        return entities.filter(e => {
            if (nodeTypeFilter !== "all" && e.entity_type !== nodeTypeFilter) return false;
            if (confidenceFilter > 0) {
                const conf = (e.properties as Record<string, number>)?.confidence ?? 1;
                if (conf < confidenceFilter / 100) return false;
            }
            return true;
        });
    }, [entities, nodeTypeFilter, confidenceFilter]);

    const entityTypes = useMemo(() => {
        const types = new Set(entities.map(e => e.entity_type));
        return Array.from(types).sort();
    }, [entities]);

    const selectedNode = useMemo(() => {
        if (!selectedId) return null;
        return entities.find(e => e.id === selectedId) || null;
    }, [selectedId, entities]);

    const connectedEdges = useMemo(() => {
        if (!selectedId) return [];
        return relations.filter(e => e.subject_id === selectedId || e.object_id === selectedId);
    }, [selectedId, relations]);

    /* ── Rich detail computations for selected node ────── */
    const degrees = useMemo(() => computeDegrees(entities, relations), [entities, relations]);

    const selectedDetail = useMemo(() => {
        if (!selectedNode || !selectedId) return null;

        const props = selectedNode.properties as Record<string, unknown>;
        const confidence = typeof props?.confidence === "number" ? props.confidence : null;
        const description = typeof props?.description === "string" ? props.description : null;
        const aliases = Array.isArray(props?.aliases) ? (props.aliases as string[]).filter(Boolean) : [];
        const sources = Array.isArray(props?.sources) ? (props.sources as string[]) : [];

        // Degree info
        const inEdges = relations.filter(r => r.object_id === selectedId);
        const outEdges = relations.filter(r => r.subject_id === selectedId);
        const totalDegree = degrees.get(selectedId) || 0;

        // Group outgoing by predicate
        const outByPred: Record<string, { predicate: string; targets: { id: string; name: string; type: string }[] }> = {};
        outEdges.forEach(e => {
            const p = e.predicate;
            if (!outByPred[p]) outByPred[p] = { predicate: p, targets: [] };
            outByPred[p].targets.push({ id: e.object_id, name: e.object_name || "Unknown", type: e.object_type || "" });
        });

        // Group incoming by predicate
        const inByPred: Record<string, { predicate: string; sources: { id: string; name: string; type: string }[] }> = {};
        inEdges.forEach(e => {
            const p = e.predicate;
            if (!inByPred[p]) inByPred[p] = { predicate: p, sources: [] };
            inByPred[p].sources.push({ id: e.subject_id, name: e.subject_name || "Unknown", type: e.subject_type || "" });
        });

        // Neighbor types distribution
        const neighborTypes: Record<string, number> = {};
        connectedEdges.forEach(e => {
            const otherId = e.subject_id === selectedId ? e.object_id : e.subject_id;
            const otherType = e.subject_id === selectedId ? (e.object_type || "UNKNOWN") : (e.subject_type || "UNKNOWN");
            neighborTypes[otherType] = (neighborTypes[otherType] || 0) + 1;
        });

        // Average confidence of connected edges
        const edgeConfidences = connectedEdges.map(e => e.confidence).filter(c => c != null);
        const avgEdgeConfidence = edgeConfidences.length > 0 ? edgeConfidences.reduce((a, b) => a + b, 0) / edgeConfidences.length : null;

        // Contradictions involving this entity
        const entityContradictions = contradictions.filter(c => {
            return connectedEdges.some(e => e.id === c.relation1_id || e.id === c.relation2_id);
        });

        // Centrality percentile (how connected vs. all nodes)
        const allDegrees = Array.from(degrees.values()).sort((a, b) => a - b);
        const rank = allDegrees.filter(d => d <= totalDegree).length;
        const percentile = allDegrees.length > 0 ? Math.round((rank / allDegrees.length) * 100) : 0;

        return {
            confidence,
            description,
            aliases,
            sources,
            inEdges,
            outEdges,
            totalDegree,
            outByPred,
            inByPred,
            neighborTypes,
            avgEdgeConfidence,
            entityContradictions,
            percentile,
        };
    }, [selectedNode, selectedId, relations, connectedEdges, contradictions, degrees, entities]);

    /* ── Search ─────────────────────────────────────────── */
    const searchResults = useMemo(() => {
        if (!searchQuery.trim()) return [];
        const q = searchQuery.toLowerCase();
        return filteredEntities
            .filter(n => n.name.toLowerCase().includes(q))
            .slice(0, 8);
    }, [searchQuery, filteredEntities]);

    const centerOnNode = useCallback((nodeId: string) => {
        if (networkRef.current) {
            const pos = networkRef.current.getPositions([nodeId]);
            if (pos[nodeId]) {
                networkRef.current.moveTo({
                    position: pos[nodeId],
                    scale: 1.8,
                    animation: { duration: 600, easingFunction: "easeInOutQuad" as any }
                });
            }
            networkRef.current.selectNodes([nodeId], true);
            setSelectedId(nodeId);
            setSearchActive(false);
            setSearchQuery("");
        }
    }, []);

    const fitToView = useCallback(() => {
        if (networkRef.current) {
            networkRef.current.fit({
                animation: { duration: 600, easingFunction: "easeInOutQuad" as any },
                maxZoomLevel: 1.5
            });
        }
    }, []);

    /* ── Hover-to-dim ──────────────────────────────────── */
    const highlightConnected = useCallback((hoveredNodeId: string) => {
        if (!networkRef.current || !nodesDatasetRef.current || !edgesDatasetRef.current) return;

        highlightActiveRef.current = true;
        const connectedNodes = new Set<string>(networkRef.current.getConnectedNodes(hoveredNodeId));
        connectedNodes.add(hoveredNodeId);

        const connectedEdgeIds = new Set<string>(networkRef.current.getConnectedEdges(hoveredNodeId));

        // Dim non-connected nodes
        const nodeUpdates: any[] = [];
        const allNodes = allNodesRef.current;
        for (const nodeId in allNodes) {
            const node = allNodes[nodeId];
            if (connectedNodes.has(nodeId)) {
                nodeUpdates.push({
                    id: nodeId,
                    opacity: 1.0,
                    font: { color: THEME.fontColor, background: THEME.fontBg },
                });
            } else {
                nodeUpdates.push({
                    id: nodeId,
                    opacity: 0.15,
                    font: { color: THEME.dimmedFont, background: "transparent" },
                });
            }
        }
        nodesDatasetRef.current.update(nodeUpdates);

        // Dim non-connected edges
        const edgeUpdates: any[] = [];
        const allEdges = allEdgesRef.current;
        for (const edgeId in allEdges) {
            const edge = allEdges[edgeId];
            if (connectedEdgeIds.has(edgeId)) {
                const color = edge._originalColor || defaultEdgeColor;
                edgeUpdates.push({
                    id: edgeId,
                    color: { color, highlight: color, hover: color, opacity: 1.0 },
                    width: edge._originalWidth || 1.5,
                });
            } else {
                edgeUpdates.push({
                    id: edgeId,
                    color: { color: THEME.dimmedEdge, highlight: THEME.dimmedEdge, hover: THEME.dimmedEdge, opacity: 0.1 },
                    width: 0.5,
                });
            }
        }
        edgesDatasetRef.current.update(edgeUpdates);
    }, []);

    const unhighlightAll = useCallback(() => {
        if (!highlightActiveRef.current || !nodesDatasetRef.current || !edgesDatasetRef.current) return;
        highlightActiveRef.current = false;

        const nodeUpdates: any[] = [];
        const allNodes = allNodesRef.current;
        for (const nodeId in allNodes) {
            nodeUpdates.push({
                id: nodeId,
                opacity: 1.0,
                font: { color: THEME.fontColor, background: THEME.fontBg },
            });
        }
        nodesDatasetRef.current.update(nodeUpdates);

        const edgeUpdates: any[] = [];
        const allEdges = allEdgesRef.current;
        for (const edgeId in allEdges) {
            const edge = allEdges[edgeId];
            const color = edge._originalColor || defaultEdgeColor;
            edgeUpdates.push({
                id: edgeId,
                color: { color: color + "80", highlight: color, hover: color, opacity: 1.0 },
                width: edge._originalWidth || 1.5,
            });
        }
        edgesDatasetRef.current.update(edgeUpdates);
    }, []);

    /* ── Initialize vis-network ─────────────────────────── */
    useEffect(() => {
        if (!containerRef.current || loading || entities.length === 0) return;

        import("vis-network").then(({ Network }) => {
            import("vis-data").then(({ DataSet }) => {
                if (!containerRef.current) return;

                setStabilizing(true);

                // Compute degree centrality
                const degrees = computeDegrees(filteredEntities, relations);
                const maxDegree = Math.max(1, ...Array.from(degrees.values()));

                // Build filtered entity ID set
                const filteredIds = new Set(filteredEntities.map(e => e.id));

                // Prepare nodes with degree-based sizing
                const nodes = filteredEntities.map(e => {
                    const cfg = entityTypeConfig[e.entity_type] || defaultEntityCfg;
                    const degree = degrees.get(e.id) || 0;

                    // Size: 8 (isolated) to 40 (highest degree), logarithmic scale
                    const normalizedDeg = maxDegree > 0 ? degree / maxDegree : 0;
                    const size = 8 + Math.log(1 + normalizedDeg * 9) / Math.log(10) * 32;

                    // Progressive labels: only show labels for nodes with enough connections
                    const showLabel = degree >= 1;

                    const desc = typeof (e.properties as Record<string, unknown>)?.description === "string"
                        ? ((e.properties as Record<string, unknown>).description as string).slice(0, 80)
                        : null;

                    return {
                        id: e.id,
                        label: showLabel ? (e.name.length > 30 ? e.name.slice(0, 27) + "\u2026" : e.name) : undefined,
                        title: `${e.name}\n${e.entity_type} \u00b7 ${degree} connections${desc ? '\n' + desc : ''}`,
                        color: {
                            background: cfg.bg,
                            border: cfg.border,
                            highlight: { background: cfg.bg, border: "#e2e8f0" },
                            hover: { background: cfg.bg, border: "#e2e8f0" },
                        },
                        shape: "dot",
                        size,
                        mass: 1 + degree * 0.2,
                        font: {
                            color: THEME.fontColor,
                            size: Math.max(10, Math.min(16, 10 + degree * 0.5)),
                            face: "Outfit, system-ui, sans-serif",
                            background: THEME.fontBg,
                            strokeWidth: 0,
                        },
                        borderWidth: 2,
                        borderWidthSelected: 3,
                        opacity: 1.0,
                        _degree: degree,
                        _entityType: e.entity_type,
                        _fullName: e.name,
                    };
                });

                // Prepare edges
                const edges = relations
                    .filter(r => filteredIds.has(r.subject_id) && filteredIds.has(r.object_id))
                    .map(r => {
                        const color = predicateColors[r.predicate] || defaultEdgeColor;
                        const width = 1 + r.confidence * 1.5;
                        return {
                            id: r.id,
                            from: r.subject_id,
                            to: r.object_id,
                            label: r.predicate.replace(/_/g, " "),
                            title: r.predicate.replace(/_/g, " "),
                            color: {
                                color: color + "80",
                                highlight: color,
                                hover: color,
                            },
                            width,
                            arrows: { to: { enabled: true, scaleFactor: 0.5, type: "arrow" } },
                            arrowStrikethrough: false,
                            smooth: { enabled: true, type: "continuous", roundness: 0.5 },
                            dashes: r.predicate === "contradicts" ? [6, 4] : false,
                            _originalColor: color,
                            _originalWidth: width,
                        };
                    });

                const nodesDataset = new DataSet(nodes as any[]);
                const edgesDataset = new DataSet(edges as any[]);
                nodesDatasetRef.current = nodesDataset;
                edgesDatasetRef.current = edgesDataset;

                // Store references for hover manipulation
                const allNodesObj: Record<string, any> = {};
                nodes.forEach(n => { allNodesObj[n.id] = n; });
                allNodesRef.current = allNodesObj;

                const allEdgesObj: Record<string, any> = {};
                edges.forEach(e => { allEdgesObj[e.id] = e; });
                allEdgesRef.current = allEdgesObj;

                const options: any = {
                    nodes: {
                        borderWidth: 2,
                        borderWidthSelected: 3,
                        scaling: {
                            label: {
                                enabled: true,
                                min: 10,
                                max: 18,
                                maxVisible: 24,
                                drawThreshold: 6,
                            },
                        },
                    },
                    edges: {
                        font: {
                            size: 9,
                            color: "#64748b",
                            strokeWidth: 2,
                            strokeColor: "#080a12",
                            align: "middle",
                            background: "#080a12",
                        },
                        chosen: true,
                        selectionWidth: 1,
                        hoverWidth: 0.5,
                    },
                    physics: {
                        enabled: true,
                        solver: "forceAtlas2Based",
                        forceAtlas2Based: {
                            theta: 0.5,
                            gravitationalConstant: -120,
                            centralGravity: 0.008,
                            springLength: 160,
                            springConstant: 0.06,
                            damping: 0.45,
                            avoidOverlap: 0.3,
                        },
                        stabilization: {
                            enabled: true,
                            iterations: 800,
                            updateInterval: 25,
                        },
                        maxVelocity: 50,
                        minVelocity: 0.75,
                    },
                    interaction: {
                        hover: true,
                        tooltipDelay: 200,
                        hideEdgesOnDrag: false,
                        hideEdgesOnZoom: false,
                        dragNodes: true,
                        dragView: true,
                        zoomView: true,
                        multiselect: false,
                        navigationButtons: false,
                    },
                    layout: {
                        improvedLayout: true,
                        randomSeed: 42,
                    },
                };

                // Destroy existing
                if (networkRef.current) {
                    networkRef.current.destroy();
                }

                const network = new Network(containerRef.current!, { nodes: nodesDataset, edges: edgesDataset }, options);
                networkRef.current = network;

                // ── Events ──

                // Node selection
                network.on("selectNode", (params: any) => {
                    if (params.nodes.length > 0) setSelectedId(params.nodes[0]);
                });
                network.on("deselectNode", () => setSelectedId(null));

                // Zoom tracking
                network.on("zoom", () => {
                    const scale = network.getScale();
                    setZoomLevel(Math.round(scale * 100));
                });

                // Hover-to-dim
                network.on("hoverNode", (params: any) => {
                    highlightConnected(params.node);
                });
                network.on("blurNode", () => {
                    unhighlightAll();
                });

                // Drag: re-enable physics with progressive damping for natural deceleration
                let dampingRaf: number | null = null;
                let dampingTimer: NodeJS.Timeout;

                network.on("dragStart", () => {
                    // Cancel any ongoing wind-down
                    if (dampingRaf !== null) cancelAnimationFrame(dampingRaf);
                    clearTimeout(dampingTimer);

                    network.setOptions({
                        physics: {
                            enabled: true,
                            solver: "forceAtlas2Based",
                            forceAtlas2Based: {
                                gravitationalConstant: -80,
                                centralGravity: 0.005,
                                springLength: 160,
                                springConstant: 0.055,
                                damping: 0.4,
                                avoidOverlap: 0.3,
                            },
                            maxVelocity: 40,
                            minVelocity: 0.1,
                        },
                    });
                });

                network.on("dragEnd", () => {
                    if (dampingRaf !== null) cancelAnimationFrame(dampingRaf);
                    clearTimeout(dampingTimer);

                    // Progressively increase damping for smooth deceleration
                    let currentDamping = 0.4;
                    const targetDamping = 0.97;
                    const step = () => {
                        currentDamping += (targetDamping - currentDamping) * 0.03;
                        const progress = (currentDamping - 0.4) / (targetDamping - 0.4);
                        const maxVel = Math.max(1, 40 * Math.pow(1 - progress, 2));

                        network.setOptions({
                            physics: {
                                forceAtlas2Based: { damping: currentDamping },
                                maxVelocity: maxVel,
                            },
                        });

                        if (currentDamping < targetDamping - 0.005) {
                            dampingRaf = requestAnimationFrame(step);
                        } else {
                            dampingTimer = setTimeout(() => {
                                network.setOptions({ physics: { enabled: false } });
                            }, 500);
                        }
                    };
                    dampingRaf = requestAnimationFrame(step);
                });

                // After stabilization: disable physics, fit to view
                network.on("stabilizationIterationsDone", () => {
                    setStabilizing(false);
                    network.setOptions({ physics: { enabled: false } });
                    network.fit({
                        animation: { duration: 800, easingFunction: "easeInOutQuad" as any },
                        maxZoomLevel: 1.5,
                    });
                });
            });
        });

        return () => {
            if (networkRef.current) {
                networkRef.current.destroy();
                networkRef.current = null;
            }
        };
    }, [filteredEntities, relations, loading, highlightConnected, unhighlightAll]);

    /* ── Render ─────────────────────────────────────────── */
    return (
        <div className="min-h-screen flex flex-col bg-void">
            {/* Header */}
            <header className="border-b border-obs-border bg-surface/50 backdrop-blur-xl sticky top-0 z-20">
                <div className="max-w-full mx-auto px-6 py-3">
                    <div className="flex items-center gap-2 text-xs text-text-muted mb-1">
                        <Link href="/" className="hover:text-amber transition-colors">Sessions</Link>
                        <span className="material-symbols-outlined text-[12px]">chevron_right</span>
                        <Link href={`/check/${sessionId}`} className="hover:text-amber transition-colors">Check</Link>
                        <span className="material-symbols-outlined text-[12px]">chevron_right</span>
                        <span className="text-text-secondary">Knowledge Graph</span>
                    </div>
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <Link href={`/check/${sessionId}`} className="text-text-secondary hover:text-amber transition-colors">
                                <span className="material-symbols-outlined">arrow_back</span>
                            </Link>
                            <h1 className="text-lg font-display">Knowledge Graph</h1>
                            <span className="obs-badge badge-system">{filteredEntities.length} nodes</span>
                            <span className="obs-badge badge-system">{relations.length} edges</span>
                            {contradictions.length > 0 && (
                                <span className="obs-badge badge-error">{contradictions.length} contradictions</span>
                            )}
                        </div>
                        <div className="flex items-center gap-2">
                            {/* Search */}
                            <div className="relative">
                                <button
                                    onClick={() => setSearchActive(a => !a)}
                                    className="obs-btn btn-ghost text-xs px-2 py-1"
                                    title="Search entities"
                                >
                                    <span className="material-symbols-outlined text-sm">search</span>
                                </button>
                                {searchActive && (
                                    <div className="absolute right-0 top-full mt-1 w-72 z-50">
                                        <input
                                            autoFocus
                                            value={searchQuery}
                                            onChange={(e) => setSearchQuery(e.target.value)}
                                            placeholder="Search entities..."
                                            className="obs-input h-9 w-full text-sm"
                                            onKeyDown={(e) => {
                                                if (e.key === "Escape") { setSearchActive(false); setSearchQuery(""); }
                                                if (e.key === "Enter" && searchResults.length > 0) centerOnNode(searchResults[0].id);
                                            }}
                                        />
                                        {searchResults.length > 0 && (
                                            <div className="mt-1 bg-surface border border-obs-border rounded-lg max-h-64 overflow-y-auto" style={{ boxShadow: "var(--shadow-lg)" }}>
                                                {searchResults.map(n => {
                                                    const cfg = entityTypeConfig[n.entity_type] || defaultEntityCfg;
                                                    return (
                                                        <button
                                                            key={n.id}
                                                            onClick={() => centerOnNode(n.id)}
                                                            className="w-full text-left px-3 py-2 hover:bg-surface-hover transition-colors flex items-center gap-2 text-sm"
                                                        >
                                                            <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: cfg.bg }} />
                                                            <span className="text-text truncate">{n.name}</span>
                                                            <span className="text-xs text-text-muted ml-auto">{n.entity_type}</span>
                                                        </button>
                                                    );
                                                })}
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>

                            <button onClick={fitToView} className="obs-btn btn-ghost text-xs gap-1" title="Fit to view">
                                <span className="material-symbols-outlined text-sm">fit_screen</span>
                            </button>
                            <button onClick={fetchGraph} className="obs-btn btn-ghost text-xs gap-1">
                                <span className="material-symbols-outlined text-sm">refresh</span>
                            </button>
                        </div>
                    </div>
                </div>
            </header>

            {/* Glow line */}
            <div className="glow-line" />

            {/* Main Area */}
            {loading ? (
                <div className="flex-1 flex items-center justify-center">
                    <div className="flex items-center gap-3 text-text-secondary">
                        <span className="material-symbols-outlined animate-spin text-amber">progress_activity</span>
                        Loading knowledge graph...
                    </div>
                </div>
            ) : error ? (
                <div className="flex-1 flex items-center justify-center">
                    <div className="text-center">
                        <span className="material-symbols-outlined text-4xl text-text-muted mb-3 block">error</span>
                        <p className="text-sm text-text-secondary">{error}</p>
                        <button onClick={fetchGraph} className="obs-btn btn-ghost text-xs mt-4">Retry</button>
                    </div>
                </div>
            ) : entities.length === 0 ? (
                <div className="flex-1 flex items-center justify-center">
                    <div className="text-center">
                        <span className="material-symbols-outlined text-4xl text-text-muted mb-3 block">hub</span>
                        <p className="text-sm text-text-secondary">No knowledge graph data yet.</p>
                        <p className="text-xs text-text-muted mt-1">Entities will appear as the fact check processes evidence.</p>
                    </div>
                </div>
            ) : filteredEntities.length === 0 && entities.length > 0 ? (
                <div className="flex-1 flex items-center justify-center">
                    <div className="text-center">
                        <span className="material-symbols-outlined text-4xl text-text-muted mb-3 block">filter_alt_off</span>
                        <p className="text-sm text-text-secondary mb-1">All nodes hidden by current filters.</p>
                        <p className="text-xs text-text-muted mb-4">
                            {nodeTypeFilter !== "all" && `Type: ${nodeTypeFilter}`}
                            {nodeTypeFilter !== "all" && confidenceFilter > 0 && " \u00b7 "}
                            {confidenceFilter > 0 && `Min confidence: ${confidenceFilter}%`}
                        </p>
                        <button
                            onClick={() => { setNodeTypeFilter("all"); setConfidenceFilter(0); }}
                            className="obs-btn btn-primary text-xs"
                        >
                            <span className="material-symbols-outlined text-sm">restart_alt</span>
                            Reset Filters
                        </button>
                    </div>
                </div>
            ) : (
                <div className="flex-1 relative overflow-hidden">
                    {/* Vis-Network Container */}
                    <div ref={containerRef} className="absolute inset-0" style={{ background: THEME.pageBg }} />

                    {/* Stabilization overlay */}
                    {stabilizing && (
                        <div className="absolute inset-0 z-10 flex items-center justify-center" style={{ background: "rgba(8,10,18,0.7)" }}>
                            <div className="flex items-center gap-3 text-text-secondary text-sm">
                                <span className="material-symbols-outlined animate-spin text-amber">progress_activity</span>
                                Computing layout...
                            </div>
                        </div>
                    )}

                    {/* Detail Panel */}
                    {selectedNode && selectedDetail && (() => {
                        const cfg = entityTypeConfig[selectedNode.entity_type] || defaultEntityCfg;
                        const d = selectedDetail;
                        return (
                        <div className="absolute right-2 top-2 sm:right-4 sm:top-4 w-[calc(100vw-1rem)] sm:w-[22rem] bg-surface/95 backdrop-blur-xl border border-obs-border rounded-xl z-20 max-h-[calc(100%-4rem)] sm:max-h-[calc(100%-6rem)] overflow-y-auto animate-rise" style={{ boxShadow: "var(--shadow-lg)" }}>
                            {/* Header */}
                            <div className="p-5 pb-0">
                                <div className="flex items-start justify-between mb-2">
                                    <div className="flex items-start gap-2.5 min-w-0">
                                        <span className="w-3.5 h-3.5 rounded-full shrink-0 mt-0.5" style={{ backgroundColor: cfg.bg }} />
                                        <div className="min-w-0">
                                            <h3 className="font-semibold text-sm leading-snug break-words">{selectedNode.name}</h3>
                                            <div className="flex items-center gap-2 mt-1">
                                                <span className="obs-badge text-[10px]" style={{ backgroundColor: cfg.dimBg, color: cfg.bg, borderColor: cfg.bg + "40" }}>
                                                    {selectedNode.entity_type}
                                                </span>
                                                <span className="text-[10px] text-text-muted font-mono">{selectedNode.id.slice(0, 8)}</span>
                                            </div>
                                        </div>
                                    </div>
                                    <button onClick={() => { setSelectedId(null); if (networkRef.current) networkRef.current.unselectAll(); }} className="text-text-muted hover:text-text transition-colors shrink-0">
                                        <span className="material-symbols-outlined text-lg">close</span>
                                    </button>
                                </div>

                                {/* Description */}
                                {d.description && (
                                    <p className="text-xs text-text-secondary mt-2 leading-relaxed">{d.description}</p>
                                )}

                                {/* Aliases */}
                                {d.aliases.length > 0 && (
                                    <div className="mt-2 flex flex-wrap gap-1">
                                        {d.aliases.map((a, i) => (
                                            <span key={i} className="text-[10px] text-text-muted bg-surface-inset px-1.5 py-0.5 rounded">aka: {a}</span>
                                        ))}
                                    </div>
                                )}
                            </div>

                            {/* Stats row */}
                            <div className="grid grid-cols-4 gap-px bg-obs-border/30 mx-5 mt-4 rounded-lg overflow-hidden">
                                {[
                                    { label: "Degree", value: d.totalDegree, sub: `top ${100 - d.percentile}%` },
                                    { label: "In", value: d.inEdges.length },
                                    { label: "Out", value: d.outEdges.length },
                                    { label: "Confidence", value: d.confidence != null ? `${Math.round(d.confidence * 100)}%` : "\u2014" },
                                ].map((s, i) => (
                                    <div key={i} className="bg-surface-inset px-2 py-2 text-center">
                                        <div className="text-sm font-semibold text-text">{s.value}</div>
                                        <div className="text-[9px] text-text-muted uppercase tracking-wide">{s.label}</div>
                                        {s.sub && <div className="text-[9px] text-text-muted">{s.sub}</div>}
                                    </div>
                                ))}
                            </div>

                            {/* Confidence bar */}
                            {d.confidence != null && (
                                <div className="mx-5 mt-3">
                                    <div className="h-1.5 bg-surface-inset rounded-full overflow-hidden">
                                        <div
                                            className="h-full rounded-full transition-all duration-500"
                                            style={{
                                                width: `${d.confidence * 100}%`,
                                                backgroundColor: d.confidence >= 0.7 ? "#34d399" : d.confidence >= 0.4 ? "#facc15" : "#fb7185",
                                            }}
                                        />
                                    </div>
                                </div>
                            )}

                            {/* Neighbor type distribution */}
                            {Object.keys(d.neighborTypes).length > 0 && (
                                <div className="px-5 mt-4">
                                    <label className="text-[10px] text-text-muted uppercase tracking-widest font-bold block mb-2">Neighbor Types</label>
                                    <div className="flex flex-wrap gap-1.5">
                                        {Object.entries(d.neighborTypes).sort(([,a],[,b]) => b - a).map(([type, count]) => {
                                            const tc = entityTypeConfig[type] || defaultEntityCfg;
                                            return (
                                                <span key={type} className="inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded" style={{ backgroundColor: tc.dimBg, color: tc.bg }}>
                                                    <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: tc.bg }} />
                                                    {type.toLowerCase()} ({count})
                                                </span>
                                            );
                                        })}
                                    </div>
                                </div>
                            )}

                            {/* Contradictions */}
                            {d.entityContradictions.length > 0 && (
                                <div className="px-5 mt-4">
                                    <label className="text-[10px] text-rose uppercase tracking-widest font-bold block mb-2">
                                        <span className="material-symbols-outlined text-xs align-middle mr-1">warning</span>
                                        Contradictions ({d.entityContradictions.length})
                                    </label>
                                    <div className="space-y-1.5">
                                        {d.entityContradictions.map(c => (
                                            <div key={c.id} className="text-xs bg-rose-soft border border-rose/20 rounded-md p-2">
                                                <div className="font-medium text-rose">{c.contradiction_type.replace(/_/g, " ")}</div>
                                                <div className="text-text-secondary mt-0.5">{c.description}</div>
                                                <div className="text-[10px] text-text-muted mt-1">Severity: {c.severity}</div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {/* Outgoing Relations (grouped by predicate) */}
                            {Object.keys(d.outByPred).length > 0 && (
                                <div className="px-5 mt-4">
                                    <label className="text-[10px] text-text-muted uppercase tracking-widest font-bold block mb-2">
                                        <span className="material-symbols-outlined text-xs align-middle mr-0.5">arrow_forward</span>
                                        Outgoing ({d.outEdges.length})
                                    </label>
                                    <div className="space-y-2">
                                        {Object.values(d.outByPred).map(group => {
                                            const color = predicateColors[group.predicate] || defaultEdgeColor;
                                            return (
                                                <div key={group.predicate}>
                                                    <div className="flex items-center gap-1.5 mb-1">
                                                        <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: color }} />
                                                        <span className="text-[10px] font-mono text-text-muted uppercase">{group.predicate.replace(/_/g, " ")}</span>
                                                        <span className="text-[10px] text-text-muted">({group.targets.length})</span>
                                                    </div>
                                                    <div className="space-y-0.5 ml-3">
                                                        {group.targets.map(t => {
                                                            const tc = entityTypeConfig[t.type] || defaultEntityCfg;
                                                            return (
                                                                <button
                                                                    key={t.id}
                                                                    onClick={() => centerOnNode(t.id)}
                                                                    className="flex items-center gap-1.5 text-xs w-full text-left hover:bg-surface-hover px-1.5 py-1 rounded transition-colors"
                                                                >
                                                                    <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ backgroundColor: tc.bg }} />
                                                                    <span className="text-text-secondary truncate">{t.name}</span>
                                                                    <span className="text-[10px] text-text-muted ml-auto shrink-0">{t.type.toLowerCase()}</span>
                                                                </button>
                                                            );
                                                        })}
                                                    </div>
                                                </div>
                                            );
                                        })}
                                    </div>
                                </div>
                            )}

                            {/* Incoming Relations (grouped by predicate) */}
                            {Object.keys(d.inByPred).length > 0 && (
                                <div className="px-5 mt-4">
                                    <label className="text-[10px] text-text-muted uppercase tracking-widest font-bold block mb-2">
                                        <span className="material-symbols-outlined text-xs align-middle mr-0.5">arrow_back</span>
                                        Incoming ({d.inEdges.length})
                                    </label>
                                    <div className="space-y-2">
                                        {Object.values(d.inByPred).map(group => {
                                            const color = predicateColors[group.predicate] || defaultEdgeColor;
                                            return (
                                                <div key={group.predicate}>
                                                    <div className="flex items-center gap-1.5 mb-1">
                                                        <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: color }} />
                                                        <span className="text-[10px] font-mono text-text-muted uppercase">{group.predicate.replace(/_/g, " ")}</span>
                                                        <span className="text-[10px] text-text-muted">({group.sources.length})</span>
                                                    </div>
                                                    <div className="space-y-0.5 ml-3">
                                                        {group.sources.map(s => {
                                                            const sc = entityTypeConfig[s.type] || defaultEntityCfg;
                                                            return (
                                                                <button
                                                                    key={s.id}
                                                                    onClick={() => centerOnNode(s.id)}
                                                                    className="flex items-center gap-1.5 text-xs w-full text-left hover:bg-surface-hover px-1.5 py-1 rounded transition-colors"
                                                                >
                                                                    <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ backgroundColor: sc.bg }} />
                                                                    <span className="text-text-secondary truncate">{s.name}</span>
                                                                    <span className="text-[10px] text-text-muted ml-auto shrink-0">{s.type.toLowerCase()}</span>
                                                                </button>
                                                            );
                                                        })}
                                                    </div>
                                                </div>
                                            );
                                        })}
                                    </div>
                                </div>
                            )}

                            {/* Source references */}
                            {d.sources.length > 0 && (
                                <div className="px-5 mt-4">
                                    <label className="text-[10px] text-text-muted uppercase tracking-widest font-bold block mb-2">Source References</label>
                                    <div className="flex flex-wrap gap-1">
                                        {d.sources.map((src, i) => (
                                            <span key={i} className="text-[10px] font-mono bg-amber-soft text-amber px-1.5 py-0.5 rounded">
                                                evidence #{src}
                                            </span>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {/* Edge confidence */}
                            {d.avgEdgeConfidence != null && (
                                <div className="px-5 mt-4">
                                    <label className="text-[10px] text-text-muted uppercase tracking-widest font-bold block mb-1">Avg. Relation Confidence</label>
                                    <div className="flex items-center gap-2">
                                        <div className="flex-1 h-1.5 bg-surface-inset rounded-full overflow-hidden">
                                            <div
                                                className="h-full rounded-full"
                                                style={{
                                                    width: `${d.avgEdgeConfidence * 100}%`,
                                                    backgroundColor: d.avgEdgeConfidence >= 0.7 ? "#34d399" : d.avgEdgeConfidence >= 0.4 ? "#facc15" : "#fb7185",
                                                }}
                                            />
                                        </div>
                                        <span className="text-[10px] font-mono text-text-secondary">{Math.round(d.avgEdgeConfidence * 100)}%</span>
                                    </div>
                                </div>
                            )}

                            {/* Raw properties (collapsible for advanced users) */}
                            {selectedNode.properties && Object.keys(selectedNode.properties).filter(k => !["aliases", "sources", "confidence"].includes(k)).length > 0 && (
                                <details className="px-5 mt-4">
                                    <summary className="text-[10px] text-text-muted uppercase tracking-widest font-bold cursor-pointer hover:text-text-secondary transition-colors">
                                        Raw Properties
                                    </summary>
                                    <div className="space-y-1 mt-2">
                                        {Object.entries(selectedNode.properties)
                                            .filter(([k]) => !["aliases", "sources", "confidence"].includes(k))
                                            .map(([k, v]) => (
                                            <div key={k} className="text-xs flex justify-between gap-2">
                                                <span className="text-text-muted font-mono shrink-0">{k}</span>
                                                <span className="text-text-secondary truncate">{String(v)}</span>
                                            </div>
                                        ))}
                                    </div>
                                </details>
                            )}

                            {/* Metadata footer */}
                            <div className="px-5 py-3 mt-4 border-t border-obs-border/50 text-[10px] text-text-muted flex items-center justify-between">
                                <span>Created {new Date(selectedNode.created_at).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric", hour: "2-digit", minute: "2-digit" })}</span>
                                {connectedEdges.length === 0 && <span className="text-rose/60">Isolated node</span>}
                            </div>
                        </div>
                        );
                    })()}

                    {/* Bottom Controls Bar */}
                    <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-surface/90 backdrop-blur-xl border border-obs-border rounded-xl px-5 py-2.5 z-20 flex items-center gap-4 max-w-[92vw] overflow-x-auto" style={{ boxShadow: "var(--shadow-lg)" }}>
                        {/* Legend */}
                        <div className="flex items-center gap-3 text-[11px] shrink-0 border-r border-obs-border pr-4">
                            {Object.entries(entityTypeConfig).slice(0, 6).map(([key, cfg]) => (
                                <span key={key} className="flex items-center gap-1.5 text-text-secondary">
                                    <span className="w-2 h-2 rounded-full" style={{ backgroundColor: cfg.bg }} />
                                    <span>{cfg.label}</span>
                                </span>
                            ))}
                        </div>

                        {/* Node Type Filter */}
                        <div className="flex items-center gap-1.5 shrink-0 border-r border-obs-border pr-4">
                            <button
                                onClick={() => setNodeTypeFilter("all")}
                                className={`obs-chip text-[11px] ${nodeTypeFilter === "all" ? "chip-active" : ""}`}
                            >
                                All
                            </button>
                            {entityTypes.map(t => (
                                <button
                                    key={t}
                                    onClick={() => setNodeTypeFilter(prev => prev === t ? "all" : t)}
                                    className={`obs-chip text-[11px] ${nodeTypeFilter === t ? "chip-active" : ""}`}
                                >
                                    <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: (entityTypeConfig[t] || defaultEntityCfg).bg }} />
                                    {t.toLowerCase()}
                                </button>
                            ))}
                        </div>

                        {/* Confidence Slider */}
                        <div className="flex items-center gap-2 text-[11px] text-text-secondary shrink-0">
                            <span>Confidence</span>
                            <input
                                type="range"
                                min="0"
                                max="100"
                                step="5"
                                value={confidenceFilter}
                                onChange={(e) => setConfidenceFilter(parseInt(e.target.value))}
                                className="w-16 accent-amber"
                            />
                            <span className="font-mono w-8">{confidenceFilter}%</span>
                        </div>

                        {/* Reset Filters */}
                        {(nodeTypeFilter !== "all" || confidenceFilter > 0) && (
                            <button
                                onClick={() => { setNodeTypeFilter("all"); setConfidenceFilter(0); }}
                                className="obs-chip text-[11px] text-rose shrink-0"
                            >
                                <span className="material-symbols-outlined text-sm">restart_alt</span>
                                Reset
                            </button>
                        )}

                        {/* Zoom Level Indicator */}
                        <div className="text-[10px] text-text-secondary font-mono shrink-0 border-l border-obs-border pl-4">
                            {zoomLevel}%
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
