"use client";

import { useEffect, useMemo, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";

interface SourceIndexItem {
    source_url: string;
    findings_count: number;
    avg_confidence: number | null;
    last_seen: string;
    domain?: string | null;
    final_score?: number | null;
    credibility_label?: string | null;
}

export default function SourcesIndexPage() {
    const params = useParams();
    const sessionId = params.id as string;
    const [sources, setSources] = useState<SourceIndexItem[]>([]);
    const [loading, setLoading] = useState(true);
    const [search, setSearch] = useState("");
    const [sortBy, setSortBy] = useState<"last_seen" | "credibility" | "citations">("last_seen");
    const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");
    const [page, setPage] = useState(0);
    const perPage = 15;

    useEffect(() => {
        (async () => {
            setLoading(true);
            try {
                const response = await fetch(`http://localhost:9090/api/sessions/${sessionId}/sources?limit=500`);
                if (response.ok) {
                    const data: SourceIndexItem[] = await response.json();
                    setSources(data || []);
                }
            } catch {
                setSources([]);
            } finally {
                setLoading(false);
            }
        })();
    }, [sessionId]);

    const filtered = useMemo(() => {
        let result = [...sources];
        if (search.trim()) {
            const q = search.toLowerCase();
            result = result.filter(
                (s) =>
                    s.source_url.toLowerCase().includes(q) ||
                    (s.domain || "").toLowerCase().includes(q)
            );
        }
        result.sort((a, b) => {
            let cmp = 0;
            if (sortBy === "credibility") cmp = (a.final_score ?? 0) - (b.final_score ?? 0);
            else if (sortBy === "citations") cmp = a.findings_count - b.findings_count;
            else cmp = new Date(a.last_seen).getTime() - new Date(b.last_seen).getTime();
            return sortDir === "desc" ? -cmp : cmp;
        });
        return result;
    }, [sources, search, sortBy, sortDir]);

    const totalPages = Math.ceil(filtered.length / perPage);
    const pageData = filtered.slice(page * perPage, (page + 1) * perPage);

    const toggleSort = (col: typeof sortBy) => {
        if (sortBy === col) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
        else { setSortBy(col); setSortDir("desc"); }
        setPage(0);
    };

    const SortIcon = ({ col }: { col: typeof sortBy }) => {
        if (sortBy !== col) return <span className="material-symbols-outlined text-[14px] text-text-muted">unfold_more</span>;
        return <span className="material-symbols-outlined text-[14px] text-amber">{sortDir === "desc" ? "arrow_downward" : "arrow_upward"}</span>;
    };

    return (
        <div className="min-h-screen flex flex-col">
            {/* Header */}
            <header className="border-b border-obs-border bg-surface/40 backdrop-blur-xl sticky top-0 z-20">
                <div className="max-w-7xl mx-auto px-6 py-4">
                    <div className="flex items-center gap-2 text-xs text-text-muted mb-2">
                        <Link href="/" className="hover:text-amber transition-colors">Sessions</Link>
                        <span className="material-symbols-outlined text-[12px]">chevron_right</span>
                        <Link href={`/check/${sessionId}`} className="hover:text-amber transition-colors">Check</Link>
                        <span className="material-symbols-outlined text-[12px]">chevron_right</span>
                        <span className="text-text-secondary">Sources</span>
                    </div>
                    <div className="flex items-center gap-3">
                        <Link href={`/check/${sessionId}`} className="text-text-secondary hover:text-amber transition-colors">
                            <span className="material-symbols-outlined">arrow_back</span>
                        </Link>
                        <h1 className="text-xl font-display">Sources Index</h1>
                        <span className="obs-badge badge-system">{filtered.length} sources</span>
                    </div>
                </div>
            </header>

            {/* Glow line */}
            <div className="glow-line" />

            <main className="flex-1 max-w-7xl mx-auto w-full px-6 py-8">
                {/* Toolbar */}
                <div className="obs-card mb-6 py-4">
                    <div className="flex flex-wrap gap-3 items-center">
                        <div className="relative flex-1 min-w-[200px] max-w-md">
                            <span className="material-symbols-outlined absolute left-3 top-1/2 -translate-y-1/2 text-text-muted text-lg">search</span>
                            <input
                                value={search}
                                onChange={(e) => { setSearch(e.target.value); setPage(0); }}
                                placeholder="Search by URL or domain..."
                                className="obs-input h-9 w-full pl-10 text-sm"
                            />
                        </div>
                    </div>
                </div>

                {/* Table */}
                <div className="obs-card p-0 overflow-hidden">
                    {loading ? (
                        <div className="p-8 text-center text-text-secondary flex items-center justify-center gap-2">
                            <span className="material-symbols-outlined animate-spin text-amber">progress_activity</span>
                            Loading sources...
                        </div>
                    ) : filtered.length === 0 ? (
                        <div className="p-12 text-center">
                            <span className="material-symbols-outlined text-4xl text-text-muted mb-3 block">travel_explore</span>
                            <p className="text-sm text-text-secondary">No sources found.</p>
                        </div>
                    ) : (
                        <>
                            <div className="overflow-x-auto">
                                <table className="w-full text-sm">
                                    <thead>
                                        <tr className="border-b border-obs-border bg-surface/50 text-xs text-text-muted uppercase tracking-wider">
                                            <th className="text-left px-4 py-3 font-medium">Source URL</th>
                                            <th className="text-left px-4 py-3 font-medium cursor-pointer select-none" onClick={() => toggleSort("credibility")}>
                                                <span className="flex items-center gap-1">Credibility <SortIcon col="credibility" /></span>
                                            </th>
                                            <th className="text-left px-4 py-3 font-medium">Status</th>
                                            <th className="text-center px-4 py-3 font-medium cursor-pointer select-none" onClick={() => toggleSort("citations")}>
                                                <span className="flex items-center justify-center gap-1">Cited <SortIcon col="citations" /></span>
                                            </th>
                                            <th className="text-right px-4 py-3 font-medium cursor-pointer select-none" onClick={() => toggleSort("last_seen")}>
                                                <span className="flex items-center justify-end gap-1">Last Seen <SortIcon col="last_seen" /></span>
                                            </th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {pageData.map((source) => {
                                            const credScore = source.final_score;
                                            const credPct = credScore !== null && credScore !== undefined ? Math.round(credScore * 100) : null;
                                            const credColor = credPct !== null ? (credPct >= 70 ? "bg-emerald" : credPct >= 40 ? "bg-gold" : "bg-rose") : "bg-obs-border";

                                            return (
                                                <tr key={source.source_url} className="border-b border-obs-border hover:bg-surface-hover transition-colors">
                                                    <td className="px-4 py-3">
                                                        <a
                                                            href={source.source_url}
                                                            target="_blank"
                                                            rel="noreferrer"
                                                            className="text-amber hover:underline break-all line-clamp-1 text-sm"
                                                        >
                                                            {source.source_url}
                                                        </a>
                                                        {source.domain && (
                                                            <span className="text-xs text-text-muted block mt-0.5 font-mono">{source.domain}</span>
                                                        )}
                                                    </td>
                                                    <td className="px-4 py-3">
                                                        {credPct !== null ? (
                                                            <div className="flex items-center gap-2 min-w-[120px]">
                                                                <div className="flex-1 h-1.5 bg-obs-border rounded-full overflow-hidden">
                                                                    <div className={`h-full rounded-full ${credColor}`} style={{ width: `${credPct}%` }} />
                                                                </div>
                                                                <span className="text-xs font-mono text-text-secondary w-8 text-right">{credPct}%</span>
                                                            </div>
                                                        ) : (
                                                            <span className="text-xs text-text-muted">&mdash;</span>
                                                        )}
                                                    </td>
                                                    <td className="px-4 py-3">
                                                        {source.credibility_label ? (
                                                            <span className={`obs-badge ${getCredBadge(source.credibility_label)}`}>
                                                                {source.credibility_label}
                                                            </span>
                                                        ) : (
                                                            <span className="obs-badge badge-system">Unknown</span>
                                                        )}
                                                    </td>
                                                    <td className="px-4 py-3 text-center">
                                                        <span className="font-mono text-sm">{source.findings_count}</span>
                                                    </td>
                                                    <td className="px-4 py-3 text-right">
                                                        <span className="text-xs text-text-muted font-mono">{formatDate(source.last_seen)}</span>
                                                    </td>
                                                </tr>
                                            );
                                        })}
                                    </tbody>
                                </table>
                            </div>

                            {/* Pagination */}
                            {totalPages > 1 && (
                                <div className="flex items-center justify-between px-4 py-3 border-t border-obs-border bg-surface/30">
                                    <span className="text-xs text-text-muted">
                                        Showing {page * perPage + 1}&ndash;{Math.min((page + 1) * perPage, filtered.length)} of {filtered.length}
                                    </span>
                                    <div className="flex gap-1">
                                        <button
                                            onClick={() => setPage((p) => Math.max(0, p - 1))}
                                            disabled={page === 0}
                                            className="obs-btn btn-ghost text-xs px-2 py-1"
                                        >
                                            <span className="material-symbols-outlined text-sm">chevron_left</span>
                                        </button>
                                        {Array.from({ length: Math.min(totalPages, 5) }, (_, i) => {
                                            const p = page < 3 ? i : page - 2 + i;
                                            if (p >= totalPages) return null;
                                            return (
                                                <button
                                                    key={p}
                                                    onClick={() => setPage(p)}
                                                    className={`obs-btn text-xs px-3 py-1 ${p === page ? "btn-primary" : "btn-ghost"}`}
                                                >
                                                    {p + 1}
                                                </button>
                                            );
                                        })}
                                        <button
                                            onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                                            disabled={page >= totalPages - 1}
                                            className="obs-btn btn-ghost text-xs px-2 py-1"
                                        >
                                            <span className="material-symbols-outlined text-sm">chevron_right</span>
                                        </button>
                                    </div>
                                </div>
                            )}
                        </>
                    )}
                </div>
            </main>
        </div>
    );
}

function formatDate(value: string): string {
    return new Date(value).toLocaleString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

function getCredBadge(label: string): string {
    const l = label.toLowerCase();
    if (l.includes("high") || l.includes("trusted")) return "badge-success";
    if (l.includes("med") || l.includes("moderate")) return "badge-verify";
    return "badge-error";
}
