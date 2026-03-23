"use client";

import { useState, useEffect, useCallback } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";

/* ── types ─────────────────────────────────────────────── */
interface AgentDecision {
    id: number;
    agent_role: string;
    decision_type: string;
    decision_outcome: string;
    reasoning: string | null;
    inputs: Record<string, unknown> | null;
    metrics: Record<string, unknown> | null;
    iteration: number | null;
    created_at: string;
}

/* ── role config ───────────────────────────────────────── */
const roleConfig: Record<string, { icon: string; color: string; bgColor: string; label: string; description: string }> = {
    director: {
        icon: "military_tech",
        color: "text-violet",
        bgColor: "bg-violet/10",
        label: "Director",
        description: "Strategic planning and claim decomposition"
    },
    manager: {
        icon: "psychology",
        color: "text-amber",
        bgColor: "bg-amber/10",
        label: "Manager",
        description: "Tactical coordination and synthesis"
    },
    intern: {
        icon: "person_search",
        color: "text-emerald",
        bgColor: "bg-emerald/10",
        label: "Fact Check Intern",
        description: "Web search and evidence extraction"
    },
    researcher: {
        icon: "person_search",
        color: "text-emerald",
        bgColor: "bg-emerald/10",
        label: "Fact Check Intern",
        description: "Web search and evidence extraction"
    },
    research_intern: {
        icon: "person_search",
        color: "text-emerald",
        bgColor: "bg-emerald/10",
        label: "Fact Check Intern",
        description: "Web search and evidence extraction"
    },
};
const defaultRoleCfg = {
    icon: "smart_toy",
    color: "text-text-secondary",
    bgColor: "bg-text-secondary/10",
    label: "Agent",
    description: "System agent"
};

export default function AgentTransparencyPage() {
    const params = useParams();
    const sessionId = params.id as string;
    const [decisions, setDecisions] = useState<AgentDecision[]>([]);
    const [loading, setLoading] = useState(true);
    const [selectedAgent, setSelectedAgent] = useState<string | null>(null);

    const fetchData = useCallback(async () => {
        setLoading(true);
        try {
            const response = await fetch(`http://localhost:9090/api/sessions/${sessionId}/agents/decisions?limit=500`);
            if (response.ok) {
                const data = await response.json();
                setDecisions(data);
            }
        } catch (err) {
            console.error("Failed to fetch agent data:", err);
        } finally {
            setLoading(false);
        }
    }, [sessionId]);

    useEffect(() => { fetchData(); }, [fetchData]);

    // Group decisions by agent role
    const agentGroups: Record<string, AgentDecision[]> = {};
    decisions.forEach((d) => {
        const role = d.agent_role;
        if (!agentGroups[role]) agentGroups[role] = [];
        agentGroups[role].push(d);
    });

    // Get agent roles in hierarchy order
    const orderedAgents = Object.keys(agentGroups).sort((a, b) => {
        const levels: Record<string, number> = { director: 0, manager: 1, intern: 2, researcher: 2, research_intern: 2 };
        return (levels[a.toLowerCase()] || 3) - (levels[b.toLowerCase()] || 3);
    });

    // Stats
    const totalActions = decisions.length;
    const uniqueActionTypes = new Set(decisions.map(d => d.decision_type)).size;

    // Get timeline for selected agent or all
    const filteredDecisions = selectedAgent
        ? decisions.filter(d => d.agent_role === selectedAgent)
        : decisions;

    return (
        <div className="min-h-screen flex flex-col">
            {/* Header */}
            <header className="border-b border-obs-border bg-surface/40 backdrop-blur-xl sticky top-0 z-20">
                <div className="max-w-7xl mx-auto px-6 py-4">
                    <div className="flex items-center gap-2 text-[11px] text-text-muted font-mono mb-2">
                        <Link href="/" className="hover:text-amber transition-colors">Sessions</Link>
                        <span className="material-symbols-outlined text-[12px]">chevron_right</span>
                        <Link href={`/check/${sessionId}`} className="hover:text-amber transition-colors">Check</Link>
                        <span className="material-symbols-outlined text-[12px]">chevron_right</span>
                        <span className="text-text-secondary">Agent Activity</span>
                    </div>
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <Link href={`/check/${sessionId}`} className="text-text-secondary hover:text-amber transition-colors">
                                <span className="material-symbols-outlined">arrow_back</span>
                            </Link>
                            <div>
                                <h1 className="text-xl font-display font-semibold">Agent Transparency</h1>
                                <p className="text-[11px] text-text-muted font-mono uppercase tracking-widest">See what each agent did and why</p>
                            </div>
                        </div>
                        <button onClick={fetchData} className="obs-btn btn-ghost text-xs gap-1">
                            <span className="material-symbols-outlined text-sm">refresh</span>
                            Refresh
                        </button>
                    </div>
                </div>
            </header>

            <div className="glow-line" />

            {loading ? (
                <div className="flex-1 flex items-center justify-center">
                    <div className="flex items-center gap-3 text-text-secondary">
                        <span className="material-symbols-outlined animate-spin text-amber">progress_activity</span>
                        Loading agent activity...
                    </div>
                </div>
            ) : decisions.length === 0 ? (
                <div className="flex-1 flex items-center justify-center">
                    <div className="text-center">
                        <span className="material-symbols-outlined text-5xl text-text-muted mb-3 block">groups</span>
                        <p className="text-text-secondary">No agent activity yet</p>
                        <p className="text-xs text-text-muted mt-1">Actions will appear as the fact check progresses</p>
                    </div>
                </div>
            ) : (
                <main className="flex-1 max-w-7xl mx-auto w-full px-6 py-8">
                    {/* Overview Stats */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
                        <div className="obs-card group">
                            <div className="absolute inset-0 bg-gradient-to-br from-amber/5 to-transparent pointer-events-none" />
                            <div className="relative">
                                <div className="flex items-center justify-between mb-2">
                                    <span className="text-[11px] font-mono text-text-muted uppercase tracking-widest">Total Actions</span>
                                    <span className="material-symbols-outlined text-amber">analytics</span>
                                </div>
                                <p className="text-3xl font-bold font-mono text-text">{totalActions}</p>
                                <p className="text-xs text-text-muted mt-1">Actions taken by all agents</p>
                            </div>
                        </div>

                        <div className="obs-card group">
                            <div className="absolute inset-0 bg-gradient-to-br from-emerald/5 to-transparent pointer-events-none" />
                            <div className="relative">
                                <div className="flex items-center justify-between mb-2">
                                    <span className="text-[11px] font-mono text-text-muted uppercase tracking-widest">Active Agents</span>
                                    <span className="material-symbols-outlined text-emerald">groups</span>
                                </div>
                                <p className="text-3xl font-bold font-mono text-text">{orderedAgents.length}</p>
                                <p className="text-xs text-text-muted mt-1">Agents participating in fact check</p>
                            </div>
                        </div>

                        <div className="obs-card group">
                            <div className="absolute inset-0 bg-gradient-to-br from-gold/5 to-transparent pointer-events-none" />
                            <div className="relative">
                                <div className="flex items-center justify-between mb-2">
                                    <span className="text-[11px] font-mono text-text-muted uppercase tracking-widest">Action Types</span>
                                    <span className="material-symbols-outlined text-gold">category</span>
                                </div>
                                <p className="text-3xl font-bold font-mono text-text">{uniqueActionTypes}</p>
                                <p className="text-xs text-text-muted mt-1">Different types of actions performed</p>
                            </div>
                        </div>
                    </div>

                    {/* Agent Cards */}
                    <div className="mb-8">
                        <div className="flex items-center gap-3 mb-4">
                            <div className="w-1.5 h-1.5 rounded-full bg-text-muted" />
                            <h2 className="text-[11px] font-mono font-semibold text-text-muted uppercase tracking-widest">
                                Fact Check Team
                            </h2>
                        </div>
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                            {orderedAgents.map((agentRole) => {
                                const cfg = roleConfig[agentRole.toLowerCase()] || defaultRoleCfg;
                                const agentDecisions = agentGroups[agentRole];
                                const isSelected = selectedAgent === agentRole;

                                return (
                                    <button
                                        key={agentRole}
                                        onClick={() => setSelectedAgent(isSelected ? null : agentRole)}
                                        className={`obs-card text-left transition-all hover:shadow-lg cursor-pointer ${
                                            isSelected ? `ring-2 ${cfg.color.replace("text-", "ring-")} ${cfg.bgColor}` : ""
                                        }`}
                                    >
                                        <div className="flex items-start gap-3 mb-3">
                                            <div className={`w-12 h-12 rounded-xl ${cfg.bgColor} flex items-center justify-center shrink-0`}>
                                                <span className={`material-symbols-outlined text-xl ${cfg.color}`}>
                                                    {cfg.icon}
                                                </span>
                                            </div>
                                            <div className="flex-1 min-w-0">
                                                <h3 className="font-semibold text-text">{cfg.label}</h3>
                                                <p className="text-xs text-text-muted leading-relaxed">{cfg.description}</p>
                                            </div>
                                        </div>

                                        <div className="flex items-center justify-between text-xs">
                                            <div className="flex items-center gap-4">
                                                <div>
                                                    <span className="text-text-muted">Actions:</span>
                                                    <span className="ml-1 font-mono font-bold text-text">{agentDecisions.length}</span>
                                                </div>
                                                {agentDecisions[0] && (
                                                    <div>
                                                        <span className="text-text-muted">Last:</span>
                                                        <span className="ml-1 font-mono text-text-secondary">
                                                            {new Date(agentDecisions[0].created_at).toLocaleTimeString("en-US", {
                                                                hour: "2-digit",
                                                                minute: "2-digit"
                                                            })}
                                                        </span>
                                                    </div>
                                                )}
                                            </div>
                                            {isSelected && (
                                                <span className="obs-badge badge-action">Selected</span>
                                            )}
                                        </div>
                                    </button>
                                );
                            })}
                        </div>
                        {selectedAgent && (
                            <p className="text-xs text-text-muted mt-3 text-center">
                                Showing only actions from <span className="text-text font-semibold">{selectedAgent}</span>
                                <button onClick={() => setSelectedAgent(null)} className="ml-2 text-amber hover:underline">
                                    Clear filter
                                </button>
                            </p>
                        )}
                    </div>

                    {/* Activity Timeline */}
                    <div>
                        <div className="flex items-center gap-3 mb-4">
                            <div className="w-1.5 h-1.5 rounded-full bg-text-muted" />
                            <h2 className="text-[11px] font-mono font-semibold text-text-muted uppercase tracking-widest">
                                Activity Timeline ({filteredDecisions.length} actions)
                            </h2>
                        </div>
                        <div className="obs-card p-0 overflow-hidden">
                            <div className="max-h-[600px] overflow-y-auto">
                                <div className="divide-y divide-obs-border">
                                    {filteredDecisions.map((decision) => (
                                        <ActionCard key={decision.id} decision={decision} />
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>
                </main>
            )}
        </div>
    );
}

/* ── Action Card ────────────────────────────────────────── */
function ActionCard({ decision }: { decision: AgentDecision }) {
    const [expanded, setExpanded] = useState(false);
    const cfg = roleConfig[decision.agent_role.toLowerCase()] || defaultRoleCfg;

    // Parse action into human-readable text
    const action = getActionDescription(decision);
    const hasDetails = decision.reasoning || (decision.inputs && Object.keys(decision.inputs).length > 0) || (decision.metrics && Object.keys(decision.metrics).length > 0);

    return (
        <div className="p-4 hover:bg-surface/50 transition-colors">
            {/* Header */}
            <div className="flex items-start gap-3">
                <div className={`w-10 h-10 rounded-lg ${cfg.bgColor} flex items-center justify-center shrink-0`}>
                    <span className={`material-symbols-outlined text-lg ${cfg.color}`}>{cfg.icon}</span>
                </div>

                <div className="flex-1 min-w-0">
                    {/* Agent & Time */}
                    <div className="flex items-center gap-2 mb-1.5">
                        <span className={`text-sm font-semibold ${cfg.color}`}>{cfg.label}</span>
                        <span className="text-text-muted">&middot;</span>
                        <span className="text-xs text-text-muted font-mono">
                            {new Date(decision.created_at).toLocaleString("en-US", {
                                month: "short",
                                day: "numeric",
                                hour: "2-digit",
                                minute: "2-digit",
                                second: "2-digit"
                            })}
                        </span>
                        {decision.iteration !== null && (
                            <>
                                <span className="text-text-muted">&middot;</span>
                                <span className="text-xs text-text-muted">
                                    <span className="material-symbols-outlined text-xs align-middle mr-0.5">replay</span>
                                    Iteration {decision.iteration}
                                </span>
                            </>
                        )}
                    </div>

                    {/* Action Description */}
                    <p className="text-sm text-text leading-relaxed mb-2">{action}</p>

                    {/* Quick metrics */}
                    {decision.metrics && Object.keys(decision.metrics).length > 0 && (
                        <div className="flex flex-wrap gap-2 mb-2">
                            {Object.entries(decision.metrics).slice(0, 3).map(([key, value]) => (
                                <span key={key} className="inline-flex items-center gap-1 px-2 py-0.5 bg-surface-inset rounded-lg text-xs">
                                    <span className="text-text-muted">{key}:</span>
                                    <span className="text-text-secondary font-mono">{String(value)}</span>
                                </span>
                            ))}
                        </div>
                    )}

                    {/* Expand button */}
                    {hasDetails && (
                        <button
                            onClick={() => setExpanded(!expanded)}
                            className="text-xs text-text-muted hover:text-amber transition-colors flex items-center gap-1 mt-2"
                        >
                            <span className="material-symbols-outlined text-sm">
                                {expanded ? "expand_less" : "expand_more"}
                            </span>
                            {expanded ? "Hide details" : "Show details"}
                        </button>
                    )}

                    {/* Expanded Details */}
                    {expanded && hasDetails && (
                        <div className="mt-3 space-y-3 pl-4 border-l-2 border-obs-border">
                            {/* Reasoning */}
                            {decision.reasoning && (
                                <div>
                                    <p className="text-[11px] font-mono text-text-muted uppercase tracking-widest mb-1.5 flex items-center gap-1">
                                        <span className="material-symbols-outlined text-sm">psychology</span>
                                        Why this action was taken
                                    </p>
                                    <p className="text-sm text-text-secondary leading-relaxed italic bg-surface/30 rounded-xl p-3">
                                        &ldquo;{decision.reasoning}&rdquo;
                                    </p>
                                </div>
                            )}

                            {/* Inputs */}
                            {decision.inputs && Object.keys(decision.inputs).length > 0 && (
                                <div>
                                    <p className="text-[11px] font-mono text-text-muted uppercase tracking-widest mb-1.5 flex items-center gap-1">
                                        <span className="material-symbols-outlined text-sm">input</span>
                                        Input data used
                                    </p>
                                    <div className="bg-surface-inset border border-obs-border rounded-xl p-3 overflow-x-auto">
                                        <pre className="font-mono text-xs text-text-secondary">
                                            {JSON.stringify(decision.inputs, null, 2)}
                                        </pre>
                                    </div>
                                </div>
                            )}

                            {/* All Metrics */}
                            {decision.metrics && Object.keys(decision.metrics).length > 3 && (
                                <div>
                                    <p className="text-[11px] font-mono text-text-muted uppercase tracking-widest mb-1.5 flex items-center gap-1">
                                        <span className="material-symbols-outlined text-sm">monitoring</span>
                                        Performance metrics
                                    </p>
                                    <div className="grid grid-cols-2 gap-2">
                                        {Object.entries(decision.metrics).map(([key, value]) => (
                                            <div key={key} className="bg-surface/50 rounded-xl px-3 py-2">
                                                <p className="text-xs text-text-muted">{key}</p>
                                                <p className="text-sm text-text font-mono">{String(value)}</p>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

/* ── Helpers ────────────────────────────────────────────── */
function getActionDescription(decision: AgentDecision): string {
    const { decision_type, decision_outcome } = decision;

    const actionMap: Record<string, (outcome: string) => string> = {
        decompose_goal: (o) => `Decomposed claim → ${o}`,
        assign_topic: (o) => `Assigned new fact check topic: "${o}"`,
        critique_findings: (o) => `Critiqued evidence from fact check → ${o}`,
        web_search: (o) => `Searched the web for: "${o}"`,
        extract_finding: (o) => `Found: ${o}`,
        verify_finding: (o) => `Verified evidence → ${o}`,
        synthesis: (o) => `Synthesized results → ${o}`,
        directive_complete: (o) => `Completed directive → ${o}`,
        topic_complete: (o) => `Finished checking: "${o}"`,
        query_expansion: (o) => `Expanded search query → "${o}"`,
        fallback_search: (o) => `Tried alternative search → "${o}"`,
        deep_report_section: (o) => `Wrote report section → ${o}`,
        knowledge_graph_update: (o) => `Updated knowledge graph → ${o}`,
        contradiction_detected: (o) => `Detected contradiction → ${o}`,
        retrieve_context: (o) => `Retrieved context for fact check → ${o}`,
        route_tool: (o) => `Selected tool → ${o}`,
        plan: (o) => `Planned next steps → ${o}`,
    };

    const descFn = actionMap[decision_type.toLowerCase()];
    if (descFn) return descFn(decision_outcome);

    const humanType = decision_type
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');

    return `${humanType} → ${decision_outcome}`;
}
