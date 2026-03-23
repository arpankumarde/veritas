"use client";

import { useEffect, useMemo, useState } from "react";

interface Evidence {
  id: number;
  session_id: string;
  content: string;
  finding_type: string;
  source_url?: string | null;
  confidence?: number | null;
  search_query?: string | null;
  created_at: string;
  verification_status?: string | null;
  verification_method?: string | null;
  kg_support_score?: number | null;
}

interface EvidenceBrowserProps {
  sessionId: string;
}

export default function EvidenceBrowser({ sessionId }: EvidenceBrowserProps) {
  const [evidence, setEvidence] = useState<Evidence[]>([]);
  const [loadingEvidence, setLoadingEvidence] = useState(true);
  const [error, setError] = useState("");
  const [search, setSearch] = useState("");
  const [evidenceType, setEvidenceType] = useState("all");
  const [minConfidence, setMinConfidence] = useState(0);
  const [order, setOrder] = useState<"desc" | "asc">("desc");
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [visibleCount, setVisibleCount] = useState(50);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoadingEvidence(true);
      setError("");
      try {
        const params = new URLSearchParams();
        params.set("limit", "500");
        params.set("order", order);
        if (search.trim()) params.set("search", search.trim());
        if (evidenceType !== "all") params.set("finding_type", evidenceType);
        if (minConfidence > 0) params.set("min_confidence", (minConfidence / 100).toFixed(2));
        const response = await fetch(`http://localhost:9090/api/sessions/${sessionId}/evidence?${params.toString()}`);
        if (!response.ok) throw new Error("Failed to load evidence");
        const data: Evidence[] = await response.json();
        if (!cancelled) setEvidence(data || []);
      } catch {
        if (!cancelled) setError("Unable to load evidence");
      } finally {
        if (!cancelled) setLoadingEvidence(false);
      }
    })();
    setVisibleCount(50);
    return () => { cancelled = true; };
  }, [sessionId, search, evidenceType, minConfidence, order]);

  const typeOptions = useMemo(() => {
    const types = new Set(evidence.map((f) => f.finding_type?.toLowerCase()));
    return ["all", ...Array.from(types).filter(Boolean)];
  }, [evidence]);

  const selectedEvidence = useMemo(() => {
    if (selectedId === null) return evidence[0] || null;
    return evidence.find((f) => f.id === selectedId) || null;
  }, [evidence, selectedId]);

  return (
    <div className="flex flex-col gap-4">
      {/* Filter Bar */}
      <div className="obs-card py-4">
        <div className="flex flex-wrap gap-3 items-center">
          <div className="relative flex-1 min-w-[200px] max-w-sm">
            <span className="material-symbols-outlined absolute left-3 top-1/2 -translate-y-1/2 text-text-muted text-lg">search</span>
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search evidence or sources..."
              className="obs-input h-9 w-full pl-10 text-sm"
            />
          </div>

          <div className="flex flex-wrap gap-2 items-center">
            {typeOptions.map((type) => (
              <button
                key={type}
                onClick={() => setEvidenceType(type)}
                className={`obs-chip ${evidenceType === type ? "chip-active" : ""}`}
              >
                {type === "all" ? "All types" : type}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-2 text-xs text-text-secondary ml-auto">
            <span>Confidence</span>
            <input
              type="range"
              min="0"
              max="100"
              step="5"
              value={minConfidence}
              onChange={(e) => setMinConfidence(parseInt(e.target.value))}
              className="w-20 accent-amber"
            />
            <span className="font-mono w-8">{minConfidence}%</span>
          </div>

          <select
            value={order}
            onChange={(e) => setOrder(e.target.value as "asc" | "desc")}
            className="obs-input h-9 text-sm"
          >
            <option value="desc">Newest first</option>
            <option value="asc">Oldest first</option>
          </select>
        </div>
      </div>

      {error && <div className="text-rose text-sm">{error}</div>}

      {/* Split Pane */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-4 h-[calc(100vh-14rem)]">
        {/* Left: Evidence List */}
        <div className="lg:col-span-2 obs-card p-0 flex flex-col overflow-hidden">
          <div className="px-4 py-3 border-b border-obs-border flex items-center justify-between bg-surface">
            <span className="text-xs font-mono text-text-muted uppercase tracking-wider">
              Latest Evidence
            </span>
            <span className="text-xs text-text-muted">
              {loadingEvidence ? "Loading..." : `${evidence.length} items`}
            </span>
          </div>
          <div className="flex-1 overflow-y-auto p-3 space-y-2">
            {loadingEvidence ? (
              <div className="text-sm text-text-secondary p-4">Loading evidence...</div>
            ) : evidence.length === 0 ? (
              <div className="text-sm text-text-secondary p-4 text-center">
                <span className="material-symbols-outlined text-3xl text-text-muted block mb-2">science</span>
                No evidence yet. Run a fact check session to populate this view.
              </div>
            ) : (
              <>
                {evidence.slice(0, visibleCount).map((item) => {
                  const isActive = item.id === (selectedEvidence?.id ?? evidence[0]?.id);
                  return (
                    <button
                      key={item.id}
                      onClick={() => setSelectedId(item.id)}
                      className={`w-full text-left flex flex-col gap-2 p-3 rounded-2xl transition-all border cursor-pointer ${isActive
                        ? "bg-surface-hover border-amber/40 ring-1 ring-amber/20"
                        : "bg-surface-inset/60 border-obs-border hover:border-amber/30"
                        }`}
                      style={isActive ? { boxShadow: "var(--shadow-md)" } : undefined}
                    >
                      <div className="flex items-center gap-2 text-xs">
                        <span className={`obs-badge ${getEvidenceBadge(item.finding_type)}`}>
                          {item.finding_type}
                        </span>
                        {typeof item.confidence === "number" && (
                          <ConfidenceRing value={item.confidence} />
                        )}
                      </div>
                      <p className="text-sm text-text line-clamp-2 leading-snug">{item.content}</p>
                      <span className="text-xs text-text-muted font-mono">{formatDate(item.created_at)}</span>
                    </button>
                  );
                })}
                {evidence.length > visibleCount && (
                  <button
                    type="button"
                    onClick={() => setVisibleCount((c) => c + 50)}
                    className="w-full text-center py-3 text-xs text-amber hover:text-amber/80 transition-colors border border-dashed border-amber/30 rounded-2xl hover:bg-amber/5"
                  >
                    Show more ({evidence.length - visibleCount} remaining)
                  </button>
                )}
              </>
            )}
          </div>
        </div>

        {/* Right: Detail View */}
        <div className="lg:col-span-3 obs-card p-0 flex flex-col overflow-hidden">
          {selectedEvidence ? (
            <>
              <div className="px-5 py-3 border-b border-obs-border bg-surface">
                <div className="flex items-center gap-2 mb-1.5">
                  <span className={`obs-badge ${getEvidenceBadge(selectedEvidence.finding_type)}`}>
                    {selectedEvidence.finding_type}
                  </span>
                  {typeof selectedEvidence.confidence === "number" && (
                    <span className="obs-badge badge-system">
                      {Math.round(selectedEvidence.confidence * 100)}% confidence
                    </span>
                  )}
                  <span className="text-xs text-text-muted font-mono ml-auto">{formatDate(selectedEvidence.created_at)}</span>
                </div>
              </div>
              <div className="flex-1 overflow-y-auto p-5 space-y-4">
                <div>
                  <h4 className="text-[10px] font-bold text-text-muted uppercase tracking-widest mb-1.5">Content</h4>
                  <p className="text-sm text-text leading-relaxed">{selectedEvidence.content}</p>
                </div>
                {selectedEvidence.source_url && (
                  <div>
                    <h4 className="text-[10px] font-bold text-text-muted uppercase tracking-widest mb-1.5">Source</h4>
                    <a
                      href={selectedEvidence.source_url}
                      target="_blank"
                      rel="noreferrer"
                      className="text-sm text-amber hover:underline break-all flex items-start gap-2 bg-surface-inset/50 p-2 rounded-lg border border-obs-border/50"
                    >
                      <span className="material-symbols-outlined text-sm mt-0.5 text-text-muted">open_in_new</span>
                      <span className="line-clamp-2">{selectedEvidence.source_url}</span>
                    </a>
                  </div>
                )}
                {selectedEvidence.search_query && (
                  <div>
                    <h4 className="text-[10px] font-bold text-text-muted uppercase tracking-widest mb-1.5">Search Query</h4>
                    <p className="text-xs text-text-secondary font-mono bg-surface-inset p-2 rounded-lg border border-obs-border/50">
                      {selectedEvidence.search_query}
                    </p>
                  </div>
                )}
              </div>
            </>
          ) : (
            <div className="flex-1 flex items-center justify-center text-text-muted">
              <div className="text-center">
                <span className="material-symbols-outlined text-4xl text-text-muted mb-2 block">description</span>
                <p className="text-sm">Select an evidence item to view details</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function ConfidenceRing({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color = pct >= 70 ? "text-emerald" : pct >= 40 ? "text-gold" : "text-rose";
  return (
    <span className={`text-xs font-mono font-medium ${color}`}>
      {pct}%
    </span>
  );
}

function formatDate(value: string): string {
  const date = new Date(value);
  return date.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function getEvidenceBadge(type?: string | null): string {
  const normalized = (type || "fact").toLowerCase();
  switch (normalized) {
    case "fact": return "badge-finding";
    case "insight": return "badge-thinking";
    case "question": return "badge-action";
    case "connection": return "badge-action";
    case "source": return "badge-system";
    default: return "badge-finding";
  }
}
