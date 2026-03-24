"use client";

import { useState } from "react";
import Link from "next/link";

interface PlagiarismMatch {
  passage: string;
  source_url: string;
  source_title: string;
  similarity: string;
}

interface PlagiarismResult {
  url: string;
  originality_score: number;
  verdict: string;
  summary: string;
  matches: PlagiarismMatch[];
  content_preview: string;
}

export default function PlagiarismCheckerPage() {
  const [url, setUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<PlagiarismResult | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!url.trim()) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const res = await fetch("http://localhost:9090/api/tools/check-plagiarism", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: url.trim() }),
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || "Analysis failed");
      }

      setResult(await res.json());
    } catch (err: any) {
      setError(err.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  const verdictConfig: Record<string, { label: string; color: string; bg: string; icon: string }> = {
    original: { label: "Original Content", color: "text-emerald", bg: "bg-emerald-soft", icon: "verified" },
    partially_plagiarized: { label: "Partially Plagiarized", color: "text-gold", bg: "bg-gold-soft", icon: "content_copy" },
    heavily_plagiarized: { label: "Heavily Plagiarized", color: "text-rose", bg: "bg-rose-soft", icon: "report" },
  };

  const similarityColors: Record<string, string> = {
    high: "text-rose bg-rose-soft border-rose/20",
    moderate: "text-gold bg-gold-soft border-gold/20",
    low: "text-text-muted bg-surface border-obs-border",
  };

  return (
    <div className="min-h-screen bg-white flex flex-col text-text font-sans">
      {/* Header */}
      <header className="border-b border-obs-border bg-white/80 backdrop-blur-md sticky top-0 z-20">
        <div className="max-w-4xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Link href="/dashboard" className="text-text-secondary hover:text-amber transition-colors">
              <span className="material-symbols-outlined">arrow_back</span>
            </Link>
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-lg bg-cyan-soft border border-cyan/10 flex items-center justify-center text-cyan">
                <span className="material-symbols-outlined text-lg">content_copy</span>
              </div>
              <div>
                <h1 className="text-base font-display font-bold tracking-tight text-text">Plagiarism Checker</h1>
                <p className="text-[9px] text-text-muted font-mono uppercase tracking-widest leading-none">Originality Analysis</p>
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-4xl mx-auto w-full px-6 py-12">
        {/* Input Section */}
        <div className="mb-12">
          <h2 className="text-2xl font-display font-semibold mb-2">Check for Plagiarism</h2>
          <p className="text-text-secondary text-sm mb-8">
            Paste a URL to any article or webpage. Veritas will extract key passages and search the web for matching content.
          </p>

          <form onSubmit={handleSubmit} className="flex gap-3">
            <div className="flex-1 relative">
              <span className="material-symbols-outlined absolute left-3.5 top-1/2 -translate-y-1/2 text-text-muted text-lg">link</span>
              <input
                type="url"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="https://example.com/article"
                className="w-full pl-10 pr-4 py-3 border border-obs-border rounded-lg text-sm focus:outline-none focus:border-amber focus:ring-1 focus:ring-amber/20 bg-white transition-colors"
                required
              />
            </div>
            <button
              type="submit"
              disabled={loading || !url.trim()}
              className="bg-amber hover:bg-amber-hover disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg px-6 py-3 text-sm font-bold uppercase tracking-wider transition-all flex items-center gap-2 shrink-0"
            >
              {loading ? (
                <span className="material-symbols-outlined animate-spin text-sm">progress_activity</span>
              ) : (
                <span className="material-symbols-outlined text-sm">plagiarism</span>
              )}
              {loading ? "Checking..." : "Check"}
            </button>
          </form>

          {error && (
            <div className="mt-4 p-4 bg-rose-soft border border-rose/20 rounded-lg flex items-start gap-3">
              <span className="material-symbols-outlined text-rose text-lg shrink-0">error</span>
              <p className="text-sm text-rose">{error}</p>
            </div>
          )}
        </div>

        {/* Loading State */}
        {loading && (
          <div className="border border-obs-border border-dashed rounded-lg p-16 text-center">
            <span className="material-symbols-outlined animate-spin text-amber text-4xl mb-4 block">progress_activity</span>
            <p className="text-sm font-medium text-text mb-1">Scraping content and searching for matches...</p>
            <p className="text-xs text-text-muted">This may take 20-30 seconds</p>
          </div>
        )}

        {/* Result */}
        {result && !loading && (() => {
          const vc = verdictConfig[result.verdict] || verdictConfig.original;
          return (
            <div className="space-y-6">
              {/* Verdict Card */}
              <div className="border border-obs-border rounded-lg overflow-hidden">
                <div className={`${vc.bg} px-8 py-8`}>
                  <div className="flex items-center gap-4">
                    <div className={`w-14 h-14 rounded-xl ${vc.bg} border border-current/10 flex items-center justify-center ${vc.color}`}>
                      <span className="material-symbols-outlined text-3xl">{vc.icon}</span>
                    </div>
                    <div>
                      <p className="text-[10px] font-mono font-bold text-text-muted uppercase tracking-[0.2em] mb-1">Verdict</p>
                      <h3 className={`text-2xl font-display font-bold ${vc.color}`}>{vc.label}</h3>
                    </div>
                    <div className="ml-auto text-right">
                      <p className="text-[10px] font-mono font-bold text-text-muted uppercase tracking-[0.2em] mb-1">Originality</p>
                      <p className="text-2xl font-display font-bold text-text">{Math.round(result.originality_score * 100)}%</p>
                    </div>
                  </div>
                </div>

                <div className="px-8 py-6 bg-white">
                  <p className="text-sm text-text-secondary leading-relaxed">{result.summary}</p>
                </div>

                {/* Originality Bar */}
                <div className="px-8 pb-6 bg-white">
                  <div className="h-2 bg-surface rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-700"
                      style={{
                        width: `${result.originality_score * 100}%`,
                        backgroundColor: result.originality_score >= 0.8 ? "#34d399"
                          : result.originality_score >= 0.5 ? "#facc15"
                          : "#fb7185",
                      }}
                    />
                  </div>
                  <div className="flex justify-between mt-1.5 text-[10px] font-mono text-text-muted">
                    <span>Plagiarized</span>
                    <span>Original</span>
                  </div>
                </div>
              </div>

              {/* Matches */}
              {result.matches.length > 0 && (
                <div className="border border-obs-border rounded-lg p-6 bg-white">
                  <h4 className="text-[11px] font-mono font-bold text-text-muted uppercase tracking-[0.2em] mb-4">
                    Matching Sources ({result.matches.length})
                  </h4>
                  <div className="space-y-3">
                    {result.matches.map((match, i) => (
                      <div key={i} className="border border-obs-border rounded-lg p-4 hover:border-amber/30 transition-colors">
                        <div className="flex items-start justify-between gap-4 mb-2">
                          <a
                            href={match.source_url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-sm font-medium text-text hover:text-amber transition-colors flex items-center gap-1.5 min-w-0"
                          >
                            <span className="material-symbols-outlined text-sm text-text-muted shrink-0">open_in_new</span>
                            <span className="truncate">{match.source_title || match.source_url}</span>
                          </a>
                          <span className={`text-[10px] font-mono font-bold uppercase tracking-wider px-2 py-0.5 rounded border shrink-0 ${similarityColors[match.similarity] || similarityColors.low}`}>
                            {match.similarity}
                          </span>
                        </div>
                        <p className="text-xs text-text-secondary bg-surface rounded p-2.5 font-mono leading-relaxed">
                          &ldquo;{match.passage}&rdquo;
                        </p>
                        <p className="text-[10px] text-text-muted mt-2 truncate">{match.source_url}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {result.matches.length === 0 && (
                <div className="border border-obs-border rounded-lg p-8 bg-white text-center">
                  <span className="material-symbols-outlined text-emerald text-3xl mb-3 block">check_circle</span>
                  <p className="text-sm text-text-secondary">No matching sources found. The content appears to be original.</p>
                </div>
              )}

              {/* Content Preview */}
              <div className="border border-obs-border rounded-lg p-6 bg-white">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="text-[11px] font-mono font-bold text-text-muted uppercase tracking-[0.2em]">Content Preview</h4>
                  <a href={result.url} target="_blank" rel="noopener noreferrer" className="text-xs text-amber hover:underline flex items-center gap-1">
                    <span className="material-symbols-outlined text-sm">open_in_new</span>
                    View Source
                  </a>
                </div>
                <p className="text-xs text-text-secondary font-mono leading-relaxed bg-surface p-4 rounded-lg whitespace-pre-wrap">{result.content_preview}</p>
              </div>
            </div>
          );
        })()}

        {/* Empty State */}
        {!result && !loading && !error && (
          <div className="border border-obs-border rounded-lg p-16 text-center bg-surface/30">
            <div className="w-16 h-16 rounded-xl bg-cyan-soft flex items-center justify-center mx-auto mb-6">
              <span className="material-symbols-outlined text-cyan text-3xl">content_copy</span>
            </div>
            <h3 className="text-lg font-display font-semibold mb-2">Paste a URL to get started</h3>
            <p className="text-sm text-text-secondary max-w-sm mx-auto">
              The plagiarism checker extracts key passages from the content and searches the web for matching text across millions of sources.
            </p>
          </div>
        )}
      </main>
    </div>
  );
}
