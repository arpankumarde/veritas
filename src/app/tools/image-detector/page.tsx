"use client";

import { useState, useRef, useCallback } from "react";
import Link from "next/link";

interface ImageDetectionResult {
  url: string;
  verdict: string;
  confidence: number;
  summary: string;
  indicators: string[];
  image_url: string;
}

type InputMode = "url" | "upload";

export default function ImageDetectorPage() {
  const [mode, setMode] = useState<InputMode>("upload");
  const [url, setUrl] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewSrc, setPreviewSrc] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<ImageDetectionResult | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) {
      setError("Please select an image file (JPG, PNG, WebP, GIF)");
      return;
    }
    if (file.size > 20 * 1024 * 1024) {
      setError("Image must be under 20 MB");
      return;
    }
    setSelectedFile(file);
    setError("");
    setPreviewSrc(URL.createObjectURL(file));
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const handleSubmitUrl = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!url.trim()) return;
    await analyze(() =>
      fetch("http://localhost:9090/api/tools/detect-ai-image", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: url.trim() }),
      })
    );
  };

  const handleSubmitUpload = async () => {
    if (!selectedFile) return;
    const formData = new FormData();
    formData.append("file", selectedFile);
    await analyze(() =>
      fetch("http://localhost:9090/api/tools/detect-ai-image-upload", {
        method: "POST",
        body: formData,
      })
    );
  };

  const analyze = async (fetchFn: () => Promise<Response>) => {
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const res = await fetchFn();
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || "Analysis failed");
      }
      const data = await res.json();
      // For uploads, use the local preview as the display image
      if (mode === "upload" && previewSrc) {
        data.image_url = previewSrc;
      }
      setResult(data);
    } catch (err: any) {
      setError(err.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  const clearFile = () => {
    setSelectedFile(null);
    if (previewSrc) URL.revokeObjectURL(previewSrc);
    setPreviewSrc(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const verdictConfig: Record<string, { label: string; color: string; bg: string; icon: string }> = {
    ai_generated: { label: "AI Generated", color: "text-rose", bg: "bg-rose-soft", icon: "auto_awesome" },
    human_created: { label: "Human Created", color: "text-emerald", bg: "bg-emerald-soft", icon: "photo_camera" },
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
              <div className="w-8 h-8 rounded-lg bg-rose-soft border border-rose/10 flex items-center justify-center text-rose">
                <span className="material-symbols-outlined text-lg">image_search</span>
              </div>
              <div>
                <h1 className="text-base font-display font-bold tracking-tight text-text">AI Image Detector</h1>
                <p className="text-[9px] text-text-muted font-mono uppercase tracking-widest leading-none">Visual Authenticity Analysis</p>
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-4xl mx-auto w-full px-6 py-12">
        {/* Input Section */}
        <div className="mb-12">
          <h2 className="text-2xl font-display font-semibold mb-2">Detect AI-Generated Images</h2>
          <p className="text-text-secondary text-sm mb-6">
            Upload an image or paste a URL. Veritas performs forensic metadata analysis — EXIF data, dimensions, color properties, embedded generation parameters — to determine if an image was AI-generated.
          </p>

          {/* Mode Tabs */}
          <div className="flex gap-1 p-1 bg-surface rounded-lg w-fit mb-6">
            <button
              onClick={() => setMode("upload")}
              className={`px-4 py-1.5 rounded-md text-xs font-bold uppercase tracking-wider transition-all ${
                mode === "upload" ? "bg-white text-text shadow-sm" : "text-text-muted hover:text-text"
              }`}
            >
              <span className="material-symbols-outlined text-sm align-middle mr-1">upload_file</span>
              Upload
            </button>
            <button
              onClick={() => setMode("url")}
              className={`px-4 py-1.5 rounded-md text-xs font-bold uppercase tracking-wider transition-all ${
                mode === "url" ? "bg-white text-text shadow-sm" : "text-text-muted hover:text-text"
              }`}
            >
              <span className="material-symbols-outlined text-sm align-middle mr-1">link</span>
              URL
            </button>
          </div>

          {/* URL Input */}
          {mode === "url" && (
            <form onSubmit={handleSubmitUrl} className="flex gap-3">
              <div className="flex-1 relative">
                <span className="material-symbols-outlined absolute left-3.5 top-1/2 -translate-y-1/2 text-text-muted text-lg">image</span>
                <input
                  type="url"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  placeholder="https://example.com/photo.jpg"
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
                  <span className="material-symbols-outlined text-sm">image_search</span>
                )}
                {loading ? "Analyzing..." : "Analyze"}
              </button>
            </form>
          )}

          {/* Upload Input */}
          {mode === "upload" && (
            <div className="space-y-4">
              {!selectedFile ? (
                <div
                  onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                  className={`border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-all ${
                    dragOver
                      ? "border-amber bg-amber-soft/30"
                      : "border-obs-border hover:border-amber/40 hover:bg-surface/30"
                  }`}
                >
                  <span className={`material-symbols-outlined text-4xl mb-3 block ${dragOver ? "text-amber" : "text-text-muted"}`}>
                    cloud_upload
                  </span>
                  <p className="text-sm font-medium text-text mb-1">
                    {dragOver ? "Drop image here" : "Drag & drop an image here"}
                  </p>
                  <p className="text-xs text-text-muted">or click to browse — JPG, PNG, WebP, GIF up to 20 MB</p>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) handleFile(file);
                    }}
                  />
                </div>
              ) : (
                <div className="border border-obs-border rounded-lg p-4 flex items-center gap-4">
                  {previewSrc && (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={previewSrc}
                      alt="Preview"
                      className="w-16 h-16 rounded-lg object-cover border border-obs-border"
                    />
                  )}
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-text truncate">{selectedFile.name}</p>
                    <p className="text-xs text-text-muted">{(selectedFile.size / 1024).toFixed(0)} KB — {selectedFile.type}</p>
                  </div>
                  <button onClick={clearFile} className="text-text-muted hover:text-rose transition-colors shrink-0" title="Remove">
                    <span className="material-symbols-outlined">close</span>
                  </button>
                  <button
                    onClick={handleSubmitUpload}
                    disabled={loading}
                    className="bg-amber hover:bg-amber-hover disabled:opacity-50 text-white rounded-lg px-6 py-2.5 text-sm font-bold uppercase tracking-wider transition-all flex items-center gap-2 shrink-0"
                  >
                    {loading ? (
                      <span className="material-symbols-outlined animate-spin text-sm">progress_activity</span>
                    ) : (
                      <span className="material-symbols-outlined text-sm">image_search</span>
                    )}
                    {loading ? "Analyzing..." : "Analyze"}
                  </button>
                </div>
              )}
            </div>
          )}

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
            <p className="text-sm font-medium text-text mb-1">Analyzing image...</p>
            <p className="text-xs text-text-muted">This may take 15-30 seconds</p>
          </div>
        )}

        {/* Result */}
        {result && !loading && (() => {
          const vc = verdictConfig[result.verdict] || verdictConfig.inconclusive;
          const showImage = result.image_url && !result.image_url.startsWith("upload://");
          const showUploadPreview = previewSrc && mode === "upload";
          const displaySrc = showUploadPreview ? previewSrc : showImage ? result.image_url : null;

          return (
            <div className="space-y-6">
              {/* Image Preview + Verdict */}
              <div className="border border-obs-border rounded-lg overflow-hidden">
                {displaySrc && (
                  <div className="bg-surface/50 p-6 flex items-center justify-center">
                    <div className="relative max-w-md w-full">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={displaySrc}
                        alt="Analyzed image"
                        className="w-full rounded-lg border border-obs-border shadow-md"
                        style={{ maxHeight: "400px", objectFit: "contain" }}
                      />
                      <div className={`absolute top-3 right-3 ${vc.bg} ${vc.color} px-3 py-1.5 rounded-lg text-xs font-bold uppercase tracking-wider flex items-center gap-1.5 border border-current/10 shadow-sm`}>
                        <span className="material-symbols-outlined text-sm">{vc.icon}</span>
                        {vc.label}
                      </div>
                    </div>
                  </div>
                )}

                {/* Verdict Details */}
                <div className={`${vc.bg} px-8 py-6`}>
                  <div className="flex items-center gap-4">
                    <div className={`w-12 h-12 rounded-xl ${vc.bg} border border-current/10 flex items-center justify-center ${vc.color}`}>
                      <span className="material-symbols-outlined text-2xl">{vc.icon}</span>
                    </div>
                    <div>
                      <p className="text-[10px] font-mono font-bold text-text-muted uppercase tracking-[0.2em] mb-1">Verdict</p>
                      <h3 className={`text-xl font-display font-bold ${vc.color}`}>{vc.label}</h3>
                    </div>
                    <div className="ml-auto text-right">
                      <p className="text-[10px] font-mono font-bold text-text-muted uppercase tracking-[0.2em] mb-1">Confidence</p>
                      <p className="text-xl font-display font-bold text-text">{Math.round(result.confidence * 100)}%</p>
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
                          : result.verdict === "human_created" ? "#34d399"
                          : "#94a3b8",
                      }}
                    />
                  </div>
                  <div className="flex justify-between mt-1.5 text-[10px] font-mono text-text-muted">
                    <span>Authentic</span>
                    <span>AI Generated</span>
                  </div>
                </div>
              </div>

              {/* Indicators */}
              {result.indicators.length > 0 && (
                <div className="border border-obs-border rounded-lg p-6 bg-white">
                  <h4 className="text-[11px] font-mono font-bold text-text-muted uppercase tracking-[0.2em] mb-4">Forensic Analysis</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {result.indicators.map((ind, i) => (
                      <div key={i} className="flex items-start gap-2.5 p-3 bg-surface rounded-lg">
                        <span className={`material-symbols-outlined text-sm mt-0.5 ${vc.color}`}>visibility</span>
                        <p className="text-sm text-text-secondary">{ind}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          );
        })()}

        {/* Empty State */}
        {!result && !loading && !error && !selectedFile && (
          <div className="border border-obs-border rounded-lg p-16 text-center bg-surface/30">
            <div className="w-16 h-16 rounded-xl bg-rose-soft flex items-center justify-center mx-auto mb-6">
              <span className="material-symbols-outlined text-rose text-3xl">image_search</span>
            </div>
            <h3 className="text-lg font-display font-semibold mb-2">Upload an image or paste a URL</h3>
            <p className="text-sm text-text-secondary max-w-sm mx-auto">
              The detector performs forensic analysis on image metadata — EXIF camera data, dimensions, PNG generation parameters, and color distribution — to determine if an image was AI-generated.
            </p>
          </div>
        )}
      </main>
    </div>
  );
}
