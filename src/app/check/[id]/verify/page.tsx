"use client";

import { useState, useEffect, useCallback } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";

/* ── types ─────────────────────────────────────────────── */
interface VerificationQuestion {
    question: string;
    aspect: string;
    independent_answer: string | null;
    supports_original: boolean | null;
    confidence: number | null;
}

interface VerificationResult {
    id: number;
    finding_id: number;
    original_confidence: number | null;
    verified_confidence: number | null;
    verification_status: string | null;
    verification_method: string | null;
    consistency_score: number;
    kg_support_score: number;
    kg_entity_matches: number;
    kg_supporting_relations: number;
    critic_iterations: number;
    corrections_made: string[] | null;
    questions_asked: VerificationQuestion[] | null;
    external_verification_used: boolean;
    contradictions: unknown | null;
    verification_time_ms: number | null;
    created_at: string;
    error: string | null;
    finding_content: string | null;
    finding_type: string | null;
    source_url: string | null;
    current_confidence: number | null;
}

interface VerificationStats {
    total: number;
    verified: number;
    flagged: number;
    rejected: number;
    pending: number;
    avg_confidence: number | null;
    avg_time_ms: number | null;
}

/* ── pipeline stage config ──────────────────────────────── */
const pipelineStages = [
    { id: "finding", label: "Evidence", icon: "lightbulb", description: "Original evidence extracted from source" },
    { id: "questions", label: "Verification Questions", icon: "quiz", description: "Generate targeted questions to verify the claim" },
    { id: "evidence", label: "Evidence Search", icon: "search", description: "Search for corroborating or contradicting evidence" },
    { id: "scoring", label: "Confidence Score", icon: "speed", description: "Calculate final confidence based on evidence" },
];

function getStageFromResult(result: VerificationResult): number {
    const status = result.verification_status?.toLowerCase();
    if (status === "verified" || status === "flagged" || status === "rejected") return 4;
    if (status === "skipped") return 4;
    if (status === "processing" || status === "in_progress") return 2;
    return 0;
}

function mapStatus(status: string | null): string {
    if (!status) return "pending";
    const s = status.toLowerCase();
    if (s === "verified" || s === "confirmed") return "verified";
    if (s === "rejected" || s === "contradicted") return "contradicted";
    if (s === "flagged") return "flagged";
    if (s === "skipped") return "skipped";
    if (s === "processing" || s === "in_progress") return "processing";
    return "pending";
}

export default function VerificationPipelinePage() {
    const params = useParams();
    const sessionId = params.id as string;
    const [results, setResults] = useState<VerificationResult[]>([]);
    const [stats, setStats] = useState<VerificationStats | null>(null);
    const [loading, setLoading] = useState(true);
    const [selectedIdx, setSelectedIdx] = useState(0);
    const [sidebarOpen, setSidebarOpen] = useState(false);

    const fetchData = useCallback(async () => {
        setLoading(true);
        try {
            const [resResults, resStats] = await Promise.all([
                fetch(`http://localhost:9090/api/sessions/${sessionId}/verification/results?limit=200`),
                fetch(`http://localhost:9090/api/sessions/${sessionId}/verification/stats`),
            ]);
            if (resResults.ok) {
                const data = await resResults.json();
                setResults(data || []);
                setSelectedIdx(0);
            }
            if (resStats.ok) setStats(await resStats.json());
        } catch {
            // silently handle
        } finally {
            setLoading(false);
        }
    }, [sessionId]);

    useEffect(() => { fetchData(); }, [fetchData]);

    const selectedFinding = results[selectedIdx] || null;
    const activeStage = selectedFinding ? getStageFromResult(selectedFinding) : 0;

    const displayStats = stats || {
        total: results.length,
        verified: results.filter((r) => mapStatus(r.verification_status) === "verified").length,
        flagged: results.filter((r) => mapStatus(r.verification_status) === "flagged").length,
        rejected: results.filter((r) => mapStatus(r.verification_status) === "contradicted").length,
        pending: results.filter((r) => mapStatus(r.verification_status) === "pending").length,
        avg_confidence: null,
        avg_time_ms: null,
    };

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
                        <span className="text-text-secondary">Verification</span>
                    </div>
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <Link href={`/check/${sessionId}`} className="text-text-secondary hover:text-amber transition-colors">
                                <span className="material-symbols-outlined">arrow_back</span>
                            </Link>
                            <h1 className="text-xl font-display font-semibold">CoVe Verification Pipeline</h1>
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
                        Loading verification data...
                    </div>
                </div>
            ) : results.length === 0 ? (
                <div className="flex-1 flex items-center justify-center">
                    <div className="text-center">
                        <span className="material-symbols-outlined text-4xl text-text-muted mb-3 block">verified</span>
                        <p className="text-sm text-text-secondary">No verification results yet.</p>
                        <p className="text-xs text-text-muted mt-1">Evidence will be verified as it is discovered.</p>
                    </div>
                </div>
            ) : (
                <div className="flex-1 flex flex-col md:flex-row max-w-7xl mx-auto w-full overflow-hidden">
                    {/* Mobile sidebar toggle */}
                    <div className="md:hidden border-b border-obs-border p-3 flex items-center justify-between bg-surface/30">
                        <button
                            onClick={() => setSidebarOpen(!sidebarOpen)}
                            className="obs-btn btn-ghost text-xs gap-1"
                        >
                            <span className="material-symbols-outlined text-sm">{sidebarOpen ? "close" : "list"}</span>
                            {sidebarOpen ? "Hide" : "Show"} Results ({results.length})
                        </button>
                        {selectedFinding && (
                            <span className={`text-[10px] font-bold uppercase tracking-wider ${getStatusColor(mapStatus(selectedFinding.verification_status))}`}>
                                {mapStatus(selectedFinding.verification_status)}
                            </span>
                        )}
                    </div>

                    {/* Left Sidebar: Queue */}
                    <aside className={`${sidebarOpen ? "block" : "hidden"} md:block w-full md:w-72 shrink-0 border-r border-obs-border p-4 overflow-y-auto max-h-[50vh] md:max-h-[calc(100vh-180px)]`}>
                        <h3 className="text-[11px] font-mono font-semibold text-text-muted uppercase tracking-widest mb-4">
                            Results ({results.length})
                        </h3>
                        <div className="space-y-2">
                            {results.map((item, idx) => {
                                const isActive = selectedIdx === idx;
                                const status = mapStatus(item.verification_status);
                                const conf = item.verified_confidence ?? item.current_confidence;
                                return (
                                    <button
                                        key={item.id}
                                        onClick={() => { setSelectedIdx(idx); setSidebarOpen(false); }}
                                        className={`w-full text-left p-3 rounded-xl transition-all border cursor-pointer ${isActive
                                            ? "bg-surface-hover border-amber/30 ring-1 ring-amber/15"
                                            : "bg-surface-inset/60 border-obs-border hover:border-amber/20"
                                            }`}
                                    >
                                        <div className="flex items-center gap-2 mb-1.5">
                                            <StatusDot status={status} />
                                            <span className={`text-[10px] font-bold uppercase tracking-wider ${getStatusColor(status)}`}>
                                                {status}
                                            </span>
                                            {conf !== null && conf !== undefined && (
                                                <span className="text-[10px] font-mono text-text-muted ml-auto">
                                                    {Math.round(conf * 100)}%
                                                </span>
                                            )}
                                        </div>
                                        <p className="text-xs text-text-secondary line-clamp-2 leading-relaxed">
                                            {item.finding_content || `Evidence #${item.finding_id}`}
                                        </p>
                                    </button>
                                );
                            })}
                        </div>
                    </aside>

                    {/* Main Content */}
                    <main className="flex-1 p-4 md:p-8 space-y-8 overflow-y-auto max-h-[calc(100vh-180px)]">
                        {/* Stats Dashboard */}
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            <MiniStat icon="verified" label="Verified" value={displayStats.verified} color="text-emerald" />
                            <MiniStat icon="flag" label="Flagged" value={displayStats.flagged} color="text-gold" />
                            <MiniStat icon="dangerous" label="Rejected" value={displayStats.rejected} color="text-rose" />
                            <MiniStat icon="hourglass_empty" label="Pending" value={displayStats.pending} color="text-text-muted" />
                        </div>

                        {/* Avg confidence & time */}
                        {(displayStats.avg_confidence !== null || displayStats.avg_time_ms !== null) && (
                            <div className="flex items-center gap-6 text-xs text-text-muted">
                                {displayStats.avg_confidence !== null && (
                                    <span className="flex items-center gap-1">
                                        <span className="material-symbols-outlined text-sm">speed</span>
                                        Avg confidence: <span className="font-mono text-text-secondary">{Math.round(displayStats.avg_confidence * 100)}%</span>
                                    </span>
                                )}
                                {displayStats.avg_time_ms !== null && (
                                    <span className="flex items-center gap-1">
                                        <span className="material-symbols-outlined text-sm">timer</span>
                                        Avg time: <span className="font-mono text-text-secondary">{Math.round(displayStats.avg_time_ms)}ms</span>
                                    </span>
                                )}
                            </div>
                        )}

                        {/* Selected Finding */}
                        {selectedFinding && (
                            <>
                                <div className="obs-card">
                                    <div className="flex items-center gap-2 mb-2">
                                        <StatusDot status={mapStatus(selectedFinding.verification_status)} />
                                        <span className={`text-xs font-bold uppercase tracking-wider ${getStatusColor(mapStatus(selectedFinding.verification_status))}`}>
                                            {mapStatus(selectedFinding.verification_status)}
                                        </span>
                                        {selectedFinding.verification_method && (
                                            <span className="obs-badge badge-system ml-2">{selectedFinding.verification_method}</span>
                                        )}
                                    </div>
                                    <p className="text-sm text-text leading-relaxed">
                                        {selectedFinding.finding_content || `Evidence #${selectedFinding.finding_id}`}
                                    </p>
                                    {selectedFinding.source_url && (
                                        <a
                                            href={selectedFinding.source_url}
                                            target="_blank"
                                            rel="noreferrer"
                                            className="text-xs text-amber hover:underline mt-2 block truncate"
                                        >
                                            {selectedFinding.source_url}
                                        </a>
                                    )}
                                </div>

                                {/* Pipeline Flow */}
                                <div>
                                    <h3 className="text-[11px] font-mono font-semibold text-text-muted uppercase tracking-widest mb-6">
                                        Verification Pipeline
                                    </h3>
                                    <div className="flex items-start gap-0">
                                        {pipelineStages.map((stage, i) => {
                                            const isCompleted = i < activeStage;
                                            const isActive = i === activeStage && activeStage < 4;
                                            return (
                                                <div key={stage.id} className="flex items-start flex-1">
                                                    <div className={`flex-1 relative rounded-2xl p-5 border transition-all ${isActive
                                                        ? "bg-amber/10 border-amber/30 ring-1 ring-amber/15"
                                                        : isCompleted
                                                            ? "bg-emerald/5 border-emerald/30"
                                                            : "bg-surface-inset/60 border-obs-border"
                                                        }`}>
                                                        {isActive && (
                                                            <div className="absolute top-0 left-0 right-0 h-0.5 bg-gradient-to-r from-transparent via-amber to-transparent animate-flow" />
                                                        )}
                                                        <div className="flex items-center gap-2 mb-3">
                                                            <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${isCompleted ? "bg-emerald/10" : isActive ? "bg-amber/20" : "bg-obs-border"}`}>
                                                                <span className={`material-symbols-outlined text-base ${isCompleted ? "text-emerald" : isActive ? "text-amber" : "text-text-muted"}`}>
                                                                    {isCompleted ? "check_circle" : stage.icon}
                                                                </span>
                                                            </div>
                                                            <div>
                                                                <p className={`text-xs font-semibold ${isCompleted ? "text-emerald" : isActive ? "text-amber" : "text-text-muted"}`}>
                                                                    {stage.label}
                                                                </p>
                                                                {isActive && (
                                                                    <span className="text-[10px] text-amber animate-breathe">Processing...</span>
                                                                )}
                                                            </div>
                                                        </div>
                                                        <p className="text-xs text-text-muted leading-relaxed">{stage.description}</p>

                                                        {isCompleted && stage.id === "finding" && (
                                                            <div className="mt-3 text-xs text-emerald flex items-center gap-1">
                                                                <span className="material-symbols-outlined text-sm">check</span>
                                                                Evidence captured
                                                            </div>
                                                        )}
                                                    </div>

                                                    {i < pipelineStages.length - 1 && (
                                                        <div className="flex items-center pt-8 mx-2">
                                                            <div className={`w-8 h-0.5 rounded ${isCompleted ? "bg-emerald" : "bg-obs-border"}`} />
                                                            <span className={`material-symbols-outlined text-sm ${isCompleted ? "text-emerald" : "text-text-muted"}`}>
                                                                chevron_right
                                                            </span>
                                                        </div>
                                                    )}
                                                </div>
                                            );
                                        })}
                                    </div>
                                </div>

                                {/* Detail Metrics */}
                                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                    <MetricCard label="KG Support" value={selectedFinding.kg_support_score} format="pct" />
                                    <MetricCard label="Consistency" value={selectedFinding.consistency_score} format="pct" />
                                    <MetricCard label="KG Entities" value={selectedFinding.kg_entity_matches} format="num" />
                                    <MetricCard label="CRITIC Iterations" value={selectedFinding.critic_iterations} format="num" />
                                </div>

                                {/* Confidence Ring */}
                                {(selectedFinding.verified_confidence ?? selectedFinding.current_confidence) !== null && (() => {
                                    const conf = (selectedFinding.verified_confidence ?? selectedFinding.current_confidence) as number;
                                    return (
                                        <div className="obs-card flex items-center gap-8">
                                            <div className="relative w-24 h-24">
                                                <svg className="w-24 h-24" viewBox="0 0 100 100">
                                                    <circle cx="50" cy="50" r="42" fill="none" stroke="currentColor" strokeWidth="6" className="text-obs-border" />
                                                    <circle
                                                        cx="50" cy="50" r="42" fill="none" strokeWidth="6" strokeLinecap="round"
                                                        strokeDasharray={`${Math.round(conf * 264)} 264`}
                                                        transform="rotate(-90 50 50)"
                                                        className={conf >= 0.7 ? "text-emerald" : conf >= 0.4 ? "text-gold" : "text-rose"}
                                                        stroke="currentColor"
                                                    />
                                                </svg>
                                                <div className="absolute inset-0 flex items-center justify-center">
                                                    <span className="text-xl font-bold font-mono">{Math.round(conf * 100)}%</span>
                                                </div>
                                            </div>
                                            <div>
                                                <h4 className="font-semibold mb-1">
                                                    {selectedFinding.verified_confidence !== null ? "Verified" : "Original"} Confidence
                                                </h4>
                                                <p className="text-sm text-text-secondary">
                                                    {conf >= 0.7
                                                        ? "High confidence — evidence is well-supported."
                                                        : conf >= 0.4
                                                            ? "Moderate confidence — some supporting evidence found."
                                                            : "Low confidence — conflicting evidence or insufficient data."
                                                    }
                                                </p>
                                                {selectedFinding.original_confidence !== null && selectedFinding.verified_confidence !== null && (
                                                    <p className="text-xs text-text-muted mt-1">
                                                        Original: {Math.round(selectedFinding.original_confidence * 100)}% → Verified: {Math.round(selectedFinding.verified_confidence * 100)}%
                                                    </p>
                                                )}
                                            </div>
                                        </div>
                                    );
                                })()}

                                {/* Verification Questions & Answers */}
                                {selectedFinding.questions_asked && selectedFinding.questions_asked.length > 0 && (
                                    <div className="obs-card">
                                        <div className="flex items-center gap-2 mb-4">
                                            <span className="material-symbols-outlined text-amber">quiz</span>
                                            <h3 className="text-sm font-semibold">Verification Questions</h3>
                                            <span className="obs-badge badge-system ml-auto">{selectedFinding.questions_asked.length} questions</span>
                                        </div>
                                        <div className="space-y-4">
                                            {selectedFinding.questions_asked.map((q, idx) => (
                                                <div key={idx} className="border-l-4 border-amber/30 pl-4 py-2">
                                                    <div className="flex items-start gap-2 mb-2">
                                                        <span className="text-xs font-mono text-text-muted mt-0.5">Q{idx + 1}</span>
                                                        <div className="flex-1">
                                                            <p className="text-sm font-medium text-text mb-1">{q.question}</p>
                                                            <span className="obs-badge badge-system text-[10px]">{q.aspect}</span>
                                                        </div>
                                                    </div>

                                                    {q.independent_answer && (
                                                        <div className="mt-2 pl-8">
                                                            <p className="text-xs text-text-secondary mb-2">
                                                                <strong className="text-text-muted">Answer:</strong> {q.independent_answer}
                                                            </p>

                                                            <div className="flex items-center gap-3 text-xs">
                                                                {q.supports_original !== null && (
                                                                    <span className={`flex items-center gap-1 ${q.supports_original ? "text-emerald" : "text-rose"}`}>
                                                                        <span className="material-symbols-outlined text-sm">
                                                                            {q.supports_original ? "check_circle" : "cancel"}
                                                                        </span>
                                                                        {q.supports_original ? "Supports claim" : "Contradicts claim"}
                                                                    </span>
                                                                )}

                                                                {q.confidence !== null && (
                                                                    <span className="flex items-center gap-1 text-text-muted">
                                                                        <span className="material-symbols-outlined text-sm">speed</span>
                                                                        <span className="font-mono">{Math.round(q.confidence * 100)}%</span>
                                                                    </span>
                                                                )}
                                                            </div>
                                                        </div>
                                                    )}
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {/* CRITIC Corrections */}
                                {selectedFinding.corrections_made && selectedFinding.corrections_made.length > 0 && (
                                    <div className="obs-card border-gold/30 bg-gold/5">
                                        <div className="flex items-center gap-2 mb-3">
                                            <span className="material-symbols-outlined text-gold">edit_note</span>
                                            <h3 className="text-sm font-semibold">CRITIC Corrections</h3>
                                            <span className="obs-badge badge-system ml-auto">{selectedFinding.corrections_made.length} corrections</span>
                                        </div>
                                        <p className="text-xs text-text-muted mb-3">
                                            Self-critiqued and revised through {selectedFinding.critic_iterations} iteration{selectedFinding.critic_iterations !== 1 ? "s" : ""}
                                        </p>
                                        <ul className="space-y-2">
                                            {selectedFinding.corrections_made.map((correction, idx) => (
                                                <li key={idx} className="flex items-start gap-2 text-xs text-text-secondary">
                                                    <span className="material-symbols-outlined text-sm text-gold mt-0.5 shrink-0">arrow_right</span>
                                                    <span className="flex-1">{correction}</span>
                                                </li>
                                            ))}
                                        </ul>
                                    </div>
                                )}

                                {/* Error */}
                                {selectedFinding.error && (
                                    <div className="obs-card border-rose/30 bg-rose/5">
                                        <div className="flex items-center gap-2 text-rose text-xs font-semibold mb-1">
                                            <span className="material-symbols-outlined text-sm">error</span>
                                            Verification Error
                                        </div>
                                        <p className="text-xs text-text-secondary">{selectedFinding.error}</p>
                                    </div>
                                )}
                            </>
                        )}
                    </main>
                </div>
            )}
        </div>
    );
}

/* ── sub-components ─────────────────────────────────────── */
function StatusDot({ status }: { status: string }) {
    const colors: Record<string, string> = {
        verified: "bg-emerald",
        processing: "bg-amber animate-breathe",
        pending: "bg-text-muted",
        contradicted: "bg-rose",
        flagged: "bg-gold",
        skipped: "bg-amber-400",
    };
    return <span className={`w-2 h-2 rounded-full ${colors[status] || "bg-text-muted"}`} />;
}

function MiniStat({ icon, label, value, color }: { icon: string; label: string; value: number; color: string }) {
    return (
        <div className="obs-card py-4">
            <div className="flex items-center justify-between mb-1">
                <span className="text-[11px] font-mono text-text-muted uppercase tracking-widest">{label}</span>
                <span className={`material-symbols-outlined text-lg ${color}`}>{icon}</span>
            </div>
            <span className="font-mono text-2xl font-bold">{value}</span>
        </div>
    );
}

function MetricCard({ label, value, format }: { label: string; value: number | null; format: "pct" | "num" }) {
    const display = value === null || value === undefined
        ? "\u2014"
        : format === "pct"
            ? `${Math.round(value * 100)}%`
            : String(value);

    return (
        <div className="obs-card py-3 text-center">
            <p className="text-[11px] font-mono text-text-muted uppercase tracking-widest mb-1">{label}</p>
            <p className="font-mono text-lg font-bold">{display}</p>
        </div>
    );
}

function getStatusColor(status: string): string {
    switch (status) {
        case "verified": return "text-emerald";
        case "processing": return "text-amber";
        case "contradicted": return "text-rose";
        case "flagged": return "text-gold";
        case "skipped": return "text-amber-400";
        default: return "text-text-muted";
    }
}
