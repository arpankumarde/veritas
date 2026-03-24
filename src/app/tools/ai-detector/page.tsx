"use client";

import { useState } from "react";
import Link from "next/link";

type InputMode = "url" | "text";

interface DetectionResult {
  url: string;
  verdict: string;
  confidence: number;
  summary: string;
  indicators: string[];
  content_type: string;
  content_preview: string;
}

export default function AIDetectorPage() {
  const [mode, setMode] = useState<InputMode>("url");
  const [url, setUrl] = useState("");
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<DetectionResult | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (mode === "url" && !url.trim()) return;
    if (mode === "text" && !text.trim()) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const endpoint =
        mode === "url"
          ? "http://localhost:9090/api/tools/detect-ai"
          : "http://localhost:9090/api/tools/detect-ai-text";

      const body =
        mode === "url"
          ? { url: url.trim(), mode: "auto" }
          : { text: text.trim() };

      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
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

  const isSubmitDisabled =
    loading || (mode === "url" ? !url.trim() : !text.trim());

  const verdictConfig: Record<string, { label: string; color: string; bg: string; icon: string }> = {
    ai_generated: { label: "AI Generated", color: "text-rose", bg: "bg-rose-soft", icon: "smart_toy" },
    human_written: { label: "Human Written", color: "text-emerald", bg: "bg-emerald-soft", icon: "person" },
    mixed: { label: "Mixed Content", color: "text-gold", bg: "bg-gold-soft", icon: "compare" },
    inconclusive: { label: "Inconclusive", color: "text-text-muted", bg: "bg-surface", icon: "help" },
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
              <div className="w-8 h-8 rounded-lg bg-violet-soft border border-violet/10 flex items-center justify-center text-violet">
                <span className="material-symbols-outlined text-lg">smart_toy</span>
              </div>
              <div>
                <h1 className="text-base font-display font-bold tracking-tight text-text">AI Content Detector</h1>
                <p className="text-[9px] text-text-muted font-mono uppercase tracking-widest leading-none">Image & Text Analysis</p>
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-4xl mx-auto w-full px-6 py-12">
        {/* Input Section */}
        <div className="mb-12">
          <h2 className="text-2xl font-display font-semibold mb-2">Detect AI-Generated Content</h2>
          <p className="text-text-secondary text-sm mb-6">
            Paste a URL or text to analyze. Veritas will check for AI-generation patterns in writing style and structure.
          </p>

          {/* Mode Tabs */}
          <div className="flex gap-1 mb-6 bg-surface rounded-lg p-1 w-fit">
            <button
              type="button"
              onClick={() => { setMode("url"); setError(""); setResult(null); }}
              className={`flex items-center gap-1.5 px-4 py-2 rounded-md text-sm font-medium transition-all ${
                mode === "url"
                  ? "bg-white text-text shadow-sm"
                  : "text-text-muted hover:text-text"
              }`}
            >
              <span className="material-symbols-outlined text-sm">link</span>
              URL
            </button>
            <button
              type="button"
              onClick={() => { setMode("text"); setError(""); setResult(null); }}
              className={`flex items-center gap-1.5 px-4 py-2 rounded-md text-sm font-medium transition-all ${
                mode === "text"
                  ? "bg-white text-text shadow-sm"
                  : "text-text-muted hover:text-text"
              }`}
            >
              <span className="material-symbols-outlined text-sm">edit_note</span>
              Paste Text
            </button>
          </div>

          <form onSubmit={handleSubmit}>
            {mode === "url" ? (
              <div className="flex gap-3">
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
                  disabled={isSubmitDisabled}
                  className="bg-amber hover:bg-amber-hover disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg px-6 py-3 text-sm font-bold uppercase tracking-wider transition-all flex items-center gap-2 shrink-0"
                >
                  {loading ? (
                    <span className="material-symbols-outlined animate-spin text-sm">progress_activity</span>
                  ) : (
                    <span className="material-symbols-outlined text-sm">search</span>
                  )}
                  {loading ? "Analyzing..." : "Analyze"}
                </button>
              </div>
            ) : (
              <div className="space-y-3">
                <textarea
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder="Paste the text you want to analyze for AI-generated content..."
                  rows={8}
                  className="w-full px-4 py-3 border border-obs-border rounded-lg text-sm focus:outline-none focus:border-amber focus:ring-1 focus:ring-amber/20 bg-white transition-colors resize-y font-mono leading-relaxed"
                  required
                />
                <div className="flex items-center justify-between">
                  <p className="text-xs text-text-muted">
                    {text.length > 0 ? `${text.length.toLocaleString()} characters` : "Minimum ~50 characters recommended"}
                  </p>
                  <button
                    type="submit"
                    disabled={isSubmitDisabled}
                    className="bg-amber hover:bg-amber-hover disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg px-6 py-3 text-sm font-bold uppercase tracking-wider transition-all flex items-center gap-2 shrink-0"
                  >
                    {loading ? (
                      <span className="material-symbols-outlined animate-spin text-sm">progress_activity</span>
                    ) : (
                      <span className="material-symbols-outlined text-sm">search</span>
                    )}
                    {loading ? "Analyzing..." : "Analyze"}
                  </button>
                </div>
              </div>
            )}
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
            <p className="text-sm font-medium text-text mb-1">
              {mode === "url" ? "Scraping and analyzing content..." : "Analyzing text..."}
            </p>
            <p className="text-xs text-text-muted">
              {mode === "url" ? "This may take 10-20 seconds" : "This may take a few seconds"}
            </p>
          </div>
        )}

        {/* Result */}
        {result && !loading && (() => {
          const vc = verdictConfig[result.verdict] || verdictConfig.inconclusive;
          return (
            <div className="space-y-6">
              {/* Verdict Card */}
              <div className={`border border-obs-border rounded-lg overflow-hidden`}>
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
                      <p className="text-[10px] font-mono font-bold text-text-muted uppercase tracking-[0.2em] mb-1">Confidence</p>
                      <p className="text-2xl font-display font-bold text-text">{Math.round(result.confidence * 100)}%</p>
                    </div>
                  </div>
                </div>

                <div className="px-8 py-6 bg-white">
                  <p className="text-sm text-text-secondary leading-relaxed">{result.summary}</p>
                </div>

                {/* Confidence Bar */}
                <div className="px-8 pb-6 bg-white">
                  <div className="h-2 bg-surface rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-700"
                      style={{
                        width: `${result.confidence * 100}%`,
                        backgroundColor: result.verdict === "ai_generated" ? "#fb7185"
                          : result.verdict === "human_written" ? "#34d399"
                          : result.verdict === "mixed" ? "#facc15"
                          : "#94a3b8",
                      }}
                    />
                  </div>
                  <div className="flex justify-between mt-1.5 text-[10px] font-mono text-text-muted">
                    <span>Human</span>
                    <span>AI Generated</span>
                  </div>
                </div>
              </div>

              {/* Indicators */}
              {result.indicators.length > 0 && (
                <div className="border border-obs-border rounded-lg p-6 bg-white">
                  <h4 className="text-[11px] font-mono font-bold text-text-muted uppercase tracking-[0.2em] mb-4">Detection Indicators</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {result.indicators.map((ind, i) => (
                      <div key={i} className="flex items-start gap-2.5 p-3 bg-surface rounded-lg">
                        <span className={`material-symbols-outlined text-sm mt-0.5 ${vc.color}`}>arrow_right</span>
                        <p className="text-sm text-text-secondary">{ind}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Content Preview */}
              <div className="border border-obs-border rounded-lg p-6 bg-white">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="text-[11px] font-mono font-bold text-text-muted uppercase tracking-[0.2em]">Content Preview</h4>
                  {result.url !== "text://pasted" && (
                    <a href={result.url} target="_blank" rel="noopener noreferrer" className="text-xs text-amber hover:underline flex items-center gap-1">
                      <span className="material-symbols-outlined text-sm">open_in_new</span>
                      View Source
                    </a>
                  )}
                </div>
                <p className="text-xs text-text-secondary font-mono leading-relaxed bg-surface p-4 rounded-lg whitespace-pre-wrap">{result.content_preview}</p>
              </div>
            </div>
          );
        })()}

        {/* Empty State */}
        {!result && !loading && !error && (
          <div className="border border-obs-border rounded-lg p-16 text-center bg-surface/30">
            <div className="w-16 h-16 rounded-xl bg-violet-soft flex items-center justify-center mx-auto mb-6">
              <span className="material-symbols-outlined text-violet text-3xl">smart_toy</span>
            </div>
            <h3 className="text-lg font-display font-semibold mb-2">
              {mode === "url" ? "Paste a URL to get started" : "Paste text to get started"}
            </h3>
            <p className="text-sm text-text-secondary max-w-sm mx-auto">
              The detector analyzes writing patterns, sentence structure, and stylistic markers to determine if content was AI-generated.
            </p>
          </div>
        )}
      </main>
    </div>
  );
}
