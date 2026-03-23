"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

interface ReportPreviewProps {
  sessionId: string;
}

/** Convert heading text to a stable slug id. */
function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/\*\*/g, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/(^-|-$)/g, "");
}

export default function ReportPreview({ sessionId }: ReportPreviewProps) {
  const [report, setReport] = useState<string>("");
  const [path, setPath] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);
  const [format, setFormat] = useState<"md" | "pdf" | "json">("md");
  const [activeTocId, setActiveTocId] = useState<string | null>(null);

  const reportContainerRef = useRef<HTMLDivElement>(null);

  const [evidence, setEvidence] = useState<any[]>([]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoading(true);
      setError("");
      setCopied(false);
      try {
        // Fetch report
        const response = await fetch(`http://localhost:9090/api/sessions/${sessionId}/report`);
        if (!response.ok) {
          if (response.status === 404) throw new Error("Verdict report not generated yet");
          throw new Error("Failed to load verdict report");
        }
        const data = await response.json();

        // Fetch evidence for JSON export
        const evidenceResponse = await fetch(`http://localhost:9090/api/sessions/${sessionId}/evidence`);
        const evidenceData = evidenceResponse.ok ? await evidenceResponse.json() : [];

        if (!cancelled) {
          setReport(data.report || "");
          setPath(data.path || null);
          setEvidence(Array.isArray(evidenceData) ? evidenceData : evidenceData.findings || []);
        }
      } catch (err: unknown) {
        if (!cancelled) setError(err instanceof Error ? err.message : "Unable to load verdict report");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [sessionId]);

  const fileName = useMemo(() => {
    const base = `verdict_report_${sessionId}`;
    return format === "md" ? `${base}.md` : format === "json" ? `${base}.json` : `${base}.pdf`;
  }, [sessionId, format]);

  const tocItems = useMemo(() => {
    if (!report) return [];
    const lines = report.split("\n");
    const items: { level: number; text: string; id: string }[] = [];
    for (const line of lines) {
      const match = line.match(/^(#{1,3})\s+(.+)/);
      if (match) {
        const level = match[1].length;
        const text = match[2].replace(/\*\*/g, "").trim();
        const id = slugify(text);
        items.push({ level, text, id });
      }
    }
    return items;
  }, [report]);

  /** Scroll to a heading inside the report container. */
  const scrollToSection = useCallback((id: string) => {
    const container = reportContainerRef.current;
    if (!container) return;
    const target = container.querySelector(`[id="${CSS.escape(id)}"]`);
    if (target) {
      target.scrollIntoView({ behavior: "smooth", block: "start" });
      setActiveTocId(id);
    }
  }, []);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(report);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  };

  const handleDownload = async () => {
    if (format === "md") {
      const blob = new Blob([report], { type: "text/markdown" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = fileName;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    } else if (format === "json") {
      const jsonData = {
        session_id: sessionId,
        report_text: report,
        evidence: evidence,
        exported_at: new Date().toISOString(),
      };
      const blob = new Blob([JSON.stringify(jsonData, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = fileName;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    } else if (format === "pdf") {
      const printWindow = window.open("", "_blank");
      if (!printWindow) {
        alert("Please allow popups to export PDF");
        return;
      }

      // Simple HTML render of markdown for print
      let htmlBody = report
        .replace(/^### (.+)$/gm, '<h3>$1</h3>')
        .replace(/^## (.+)$/gm, '<h2>$1</h2>')
        .replace(/^# (.+)$/gm, '<h1>$1</h1>')
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        .replace(/^- (.+)$/gm, '<li>$1</li>')
        .replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>')
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>');
      htmlBody = `<p>${htmlBody}</p>`;

      const htmlContent = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Verdict Report - ${sessionId}</title>
<style>
@page { size: A4; margin: 0; }
* { box-sizing: border-box; }
body { font-family: Georgia, "Times New Roman", serif; font-size: 11pt; line-height: 1.65; color: #1a1a1a; margin: 0; padding: 0; }
article.paper { max-width: 210mm; margin: 0 auto; padding: 20mm 22mm; }
h1 { text-align: center; font-size: 20pt; font-weight: 700; line-height: 1.25; margin: 0 0 6pt 0; }
h2 { font-size: 14pt; font-weight: 700; margin: 26pt 0 8pt 0; padding-bottom: 3pt; border-bottom: 0.5pt solid #999; }
h3 { font-size: 12pt; font-weight: 700; margin: 18pt 0 6pt 0; }
p { margin: 0 0 8pt 0; text-align: justify; }
blockquote { margin: 10pt 0.4in; font-style: italic; color: #222; }
ul, ol { margin: 4pt 0 10pt 0; padding-left: 20pt; }
li { margin-bottom: 3pt; }
table { width: 100%; border-collapse: collapse; font-size: 9.5pt; margin: 12pt 0; }
th, td { border: 0.5pt solid #555; padding: 4pt 6pt; text-align: left; }
th { font-weight: 700; background: #f0f0f0; }
code { font-family: "Courier New", monospace; font-size: 9pt; background: #f4f4f4; padding: 1pt 3pt; }
pre { background: #f8f8f8; border: 0.5pt solid #ccc; padding: 8pt; font-size: 8.5pt; overflow-x: auto; }
hr { border: none; border-top: 0.5pt solid #bbb; margin: 18pt 0; }
a { color: #1a4480; text-decoration: underline; }
strong { font-weight: 700; }
em { font-style: italic; }
@media print { body { -webkit-print-color-adjust: exact; print-color-adjust: exact; } }
</style>
</head>
<body>
<article class="paper">
${htmlBody}
</article>
</body>
</html>`;

      printWindow.document.write(htmlContent);
      printWindow.document.close();
      printWindow.onload = () => {
        setTimeout(() => { printWindow.print(); }, 300);
      };
    }
  };

  /** Render markdown as simple HTML with heading IDs */
  const renderedHtml = useMemo(() => {
    if (!report) return "";
    let html = report;
    // Headings with IDs
    html = html.replace(/^### (.+)$/gm, (_, text) => {
      const id = slugify(text.replace(/\*\*/g, ""));
      return `<h3 id="${id}" class="text-lg font-semibold mt-5 mb-2 text-text scroll-mt-4">${text}</h3>`;
    });
    html = html.replace(/^## (.+)$/gm, (_, text) => {
      const id = slugify(text.replace(/\*\*/g, ""));
      return `<h2 id="${id}" class="text-xl font-display mt-6 mb-3 text-text scroll-mt-4">${text}</h2>`;
    });
    html = html.replace(/^# (.+)$/gm, (_, text) => {
      const id = slugify(text.replace(/\*\*/g, ""));
      return `<h1 id="${id}" class="text-2xl font-display mt-8 mb-4 text-text scroll-mt-4">${text}</h1>`;
    });
    // Bold and italic
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong class="font-semibold text-text">$1</strong>');
    html = html.replace(/\*(.+?)\*/g, '<em class="italic text-text-secondary">$1</em>');
    // Links
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer" class="text-amber hover:underline cursor-pointer">$1</a>');
    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code class="text-xs bg-surface-inset border border-obs-border rounded px-1.5 py-0.5 text-amber">$1</code>');
    // Blockquotes
    html = html.replace(/^> (.+)$/gm, '<blockquote class="border-l-4 border-amber/50 pl-6 italic text-text-secondary my-6 bg-amber/5 py-3 pr-4 rounded-r-lg">$1</blockquote>');
    // Horizontal rules
    html = html.replace(/^---$/gm, '<hr class="border-obs-border my-6">');
    // List items
    html = html.replace(/^- (.+)$/gm, '<li class="text-sm text-text leading-relaxed">$1</li>');
    // Wrap consecutive <li> in <ul>
    html = html.replace(/((<li[^>]*>.*?<\/li>\n?)+)/g, '<ul class="list-disc list-outside ml-6 text-sm text-text mb-4 space-y-2">$1</ul>');
    // Paragraphs (lines not already tagged)
    html = html.replace(/^(?!<[hubloa]|<li|<hr|<block)(.+)$/gm, '<p class="text-sm text-text/80 leading-relaxed mb-3">$1</p>');
    return html;
  }, [report]);

  return (
    <div className="flex flex-col lg:flex-row gap-4" style={{ minHeight: "36rem" }}>
      {/* Table of Contents Sidebar */}
      {tocItems.length > 0 && (
        <aside className="lg:w-64 shrink-0">
          <div className="obs-card sticky top-40">
            <h3 className="text-xs font-display font-normal text-text-muted uppercase tracking-wider mb-4">
              Table of Contents
            </h3>
            <nav className="space-y-1 max-h-96 overflow-y-auto scrollbar-hide">
              {tocItems.map((item, i) => (
                <button
                  key={i}
                  onClick={() => scrollToSection(item.id)}
                  className={`block w-full text-left text-sm truncate py-1 transition-colors hover:text-amber ${
                    activeTocId === item.id
                      ? "text-amber font-medium"
                      : item.level === 1
                        ? "text-text font-medium"
                        : item.level === 2
                          ? "text-text-secondary pl-4"
                          : "text-text-muted pl-8 text-xs"
                  }`}
                >
                  {item.text}
                </button>
              ))}
            </nav>
          </div>
        </aside>
      )}

      {/* Main Report Content */}
      <div className="flex-1 obs-card">
        {/* Header */}
        <div className="flex flex-wrap items-center justify-between gap-3 mb-6 pb-4 border-b border-obs-border">
          <div>
            <h3 className="text-lg font-display">Verdict Report</h3>
            <p className="text-xs text-text-muted mt-1">
              {path ? (
                <>Saved at: <span className="font-mono">{path}</span></>
              ) : (
                "Rendered from latest report"
              )}
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {/* Format Toggles */}
            <div className="flex bg-surface-inset rounded-lg p-0.5 border border-obs-border">
              {(["md", "pdf", "json"] as const).map((f) => (
                <button
                  key={f}
                  onClick={() => setFormat(f)}
                  className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${format === f
                      ? "bg-surface-hover text-amber"
                      : "text-text-muted hover:text-text"
                    }`}
                >
                  {f.toUpperCase()}
                </button>
              ))}
            </div>
            <button className="obs-btn btn-ghost text-sm" onClick={handleCopy} disabled={!report}>
              <span className="material-symbols-outlined text-base">
                {copied ? "check" : "content_copy"}
              </span>
              {copied ? "Copied" : "Copy"}
            </button>
            <button className="obs-btn btn-primary text-sm" onClick={handleDownload} disabled={!report}>
              <span className="material-symbols-outlined text-base">download</span>
              Download
            </button>
          </div>
        </div>

        {/* Content */}
        {loading ? (
          <div className="flex items-center gap-3 text-text-secondary">
            <span className="material-symbols-outlined animate-spin">progress_activity</span>
            <span className="text-sm">Loading verdict report...</span>
          </div>
        ) : error ? (
          <div className="text-center py-12">
            <span className="material-symbols-outlined text-4xl text-text-muted mb-3 block">article</span>
            <p className="text-sm text-rose mb-1">{error}</p>
            <p className="text-xs text-text-muted">Run a session to completion to generate a verdict report.</p>
          </div>
        ) : report.trim().length === 0 ? (
          <div className="text-center py-12">
            <span className="material-symbols-outlined text-4xl text-text-muted mb-3 block">draft</span>
            <p className="text-sm text-text-secondary">Verdict report is empty.</p>
            <p className="text-xs text-text-muted mt-1">Try rerunning the fact check or exporting evidence.</p>
          </div>
        ) : (
          <div
            ref={reportContainerRef}
            className="report-markdown max-h-[42rem] overflow-y-auto pr-2 scroll-smooth"
            dangerouslySetInnerHTML={{ __html: renderedHtml }}
          />
        )}
      </div>
    </div>
  );
}
