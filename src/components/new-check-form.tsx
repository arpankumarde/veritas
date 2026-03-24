"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

interface NewCheckFormProps {
  onClose: () => void;
  onSuccess: () => void;
}

interface ClarificationQuestion {
  question: string;
  options: string[];
  allow_multiple: boolean;
}

export default function NewCheckForm({ onClose, onSuccess }: NewCheckFormProps) {
  const router = useRouter();
  const [claim, setClaim] = useState("");
  const [maxIterations, setMaxIterations] = useState(1);
  const [enableClarification, setEnableClarification] = useState(true);
  const [enableMidQuestions, setEnableMidQuestions] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [showClarification, setShowClarification] = useState(false);
  const [questions, setQuestions] = useState<ClarificationQuestion[]>([]);
  const [answers, setAnswers] = useState<Record<string, string>>({});

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!claim.trim()) {
      setError("Please enter a claim to verify");
      return;
    }

    if (enableClarification && !showClarification) {
      await getClarificationQuestions();
      return;
    }

    if (showClarification && Object.keys(answers).length > 0) {
      await startWithEnrichedClaim();
    } else {
      await startCheck(claim);
    }
  };

  const getClarificationQuestions = async () => {
    setLoading(true);
    setError("");

    try {
      const response = await fetch("http://localhost:9090/api/checks/clarify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ claim: claim.trim(), max_questions: 4 }),
      });

      if (!response.ok) {
        throw new Error("Failed to generate questions");
      }

      const data = await response.json();
      const qs = data.questions || [];
      if (qs.length === 0) {
        await startCheck(claim.trim());
        return;
      }
      setQuestions(qs);
      setShowClarification(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to get clarification questions");
    } finally {
      setLoading(false);
    }
  };

  const startWithEnrichedClaim = async () => {
    setLoading(true);
    setError("");

    try {
      const enrichResponse = await fetch("http://localhost:9090/api/checks/enrich", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          claim: claim.trim(),
          questions: questions,
          answers: answers,
        }),
      });

      if (!enrichResponse.ok) {
        throw new Error("Failed to enrich claim");
      }

      const enrichData = await enrichResponse.json();
      const enrichedClaim = enrichData.enriched_claim;

      await startCheck(enrichedClaim);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start fact check");
      setLoading(false);
    }
  };

  const startCheck = async (checkClaim: string) => {
    try {
      const response = await fetch("http://localhost:9090/api/checks/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          claim: checkClaim,
          max_iterations: maxIterations,
          autonomous: !enableClarification && !enableMidQuestions,
          enable_mid_questions: enableMidQuestions,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to start fact check");
      }

      const data = await response.json();
      onSuccess();
      router.push(`/check/${data.session_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start fact check");
    } finally {
      setLoading(false);
    }
  };

  const depthLabel = maxIterations <= 3 ? "Standard" : maxIterations <= 7 ? "Detailed" : maxIterations <= 14 ? "Deep Audit" : "Absolute";

  return (
    <div className="bg-white border border-obs-border rounded-lg overflow-hidden relative shadow-2xl">
      <form onSubmit={handleSubmit} className="p-8 md:p-10 space-y-10">
        {/* Header */}
        <div className="space-y-2 border-b border-obs-border pb-6">
          <h2 className="text-xl md:text-2xl font-display font-semibold tracking-tight text-text uppercase">
            {showClarification ? "Audit Refinement" : "Initialize Audit"}
          </h2>
          <p className="text-[12px] font-mono text-text-secondary uppercase tracking-widest opacity-60">
            {showClarification
              ? "Refining verification scope via AI inquiry"
              : "Configuring multi-agent verification parameters"}
          </p>
        </div>

        {/* Close button */}
        <button
          type="button"
          onClick={onClose}
          className="absolute top-8 right-8 text-text-muted hover:text-amber transition-colors"
        >
          <span className="material-symbols-outlined">close</span>
        </button>

        {!showClarification ? (
          <>
            {/* Claim Input */}
            <div className="space-y-3">
              <label className="text-[10px] font-mono font-bold text-text-muted uppercase tracking-[0.2em]">
                Target Claim
              </label>
              <textarea
                value={claim}
                onChange={(e) => setClaim(e.target.value)}
                className="w-full min-h-[120px] bg-surface border border-obs-border text-base p-5 rounded-lg focus:outline-none focus:border-amber transition-colors text-text placeholder:text-text-muted/50"
                placeholder='e.g., "Company X reported record Q3 profits despite supply chain disruption"'
              />
            </div>

            {/* Iterations */}
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <label className="text-[10px] font-mono font-bold text-text-muted uppercase tracking-[0.2em]">
                  Audit Depth
                </label>
                <span className="text-amber font-mono text-[11px] font-bold">
                  {maxIterations} ITERATIONS
                </span>
              </div>
              <input
                type="range"
                min="1"
                max="20"
                step="1"
                value={maxIterations}
                onChange={(e) => setMaxIterations(parseInt(e.target.value))}
                className="w-full h-1 bg-obs-border rounded-full appearance-none cursor-pointer accent-amber"
              />
               <div className="flex justify-between text-[10px] text-text-muted font-mono uppercase tracking-widest">
                <span>Quick Scan</span>
                <span>Exhaustive</span>
              </div>
            </div>

            {/* Toggles */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="flex items-center justify-between p-4 bg-surface border border-obs-border rounded-lg">
                <div className="flex flex-col">
                  <span className="text-[11px] font-bold text-text uppercase tracking-wider">Clarification</span>
                  <span className="text-[9px] text-text-secondary uppercase tracking-widest">Pre-audit check</span>
                </div>
                <input
                  type="checkbox"
                  checked={enableClarification}
                  onChange={(e) => setEnableClarification(e.target.checked)}
                  className="w-4 h-4 accent-amber"
                />
              </div>
              <div className="flex items-center justify-between p-4 bg-surface border border-obs-border rounded-lg">
                <div className="flex flex-col">
                  <span className="text-[11px] font-bold text-text uppercase tracking-wider">Human in loop</span>
                  <span className="text-[9px] text-text-secondary uppercase tracking-widest">Mid-audit interrupt</span>
                </div>
                <input
                  type="checkbox"
                  checked={enableMidQuestions}
                  onChange={(e) => setEnableMidQuestions(e.target.checked)}
                  className="w-4 h-4 accent-amber"
                />
              </div>
            </div>
          </>
        ) : (
          <>
            {/* Clarification Questions */}
            <div className="space-y-8 bg-surface p-6 rounded-lg border border-obs-border">
              {questions.map((q, idx) => (
                <div key={idx} className="space-y-3">
                  <label className="block text-sm font-semibold text-text">
                    <span className="text-amber mr-2">{idx + 1}.</span> {q.question}
                  </label>
                  <input
                    type="text"
                    value={answers[idx.toString()] || ""}
                    onChange={(e) => setAnswers({ ...answers, [idx.toString()]: e.target.value })}
                    className="w-full bg-white border border-obs-border p-3 text-sm rounded focus:outline-none focus:border-amber transition-colors"
                    placeholder="Enter analytical response..."
                  />
                </div>
              ))}
              
              <button
                type="button"
                onClick={() => {
                  setShowClarification(false);
                  setQuestions([]);
                  setAnswers({});
                }}
                className="text-[10px] font-mono font-bold text-text-muted hover:text-amber uppercase tracking-[0.2em] transition-colors"
              >
                &larr; Back to claim
              </button>
            </div>
          </>
        )}

        {/* Error */}
        {error && (
          <div className="text-[11px] font-mono text-rose bg-rose-soft border border-rose/20 rounded-lg px-4 py-3 uppercase tracking-widest font-bold">
            [Error]: {error}
          </div>
        )}

        {/* Action Button */}
        <div className="pt-4">
          <button
            type="submit"
            disabled={loading || !claim.trim() || (showClarification && Object.keys(answers).length === 0)}
            className="w-full py-4 px-6 bg-amber hover:bg-amber-hover text-white font-bold text-[13px] rounded-lg transition-all duration-300 uppercase tracking-[0.2em] flex items-center justify-center gap-3 disabled:opacity-50 disabled:cursor-not-allowed shadow-none"
          >
            {loading ? (
              <>
                <span className="material-symbols-outlined animate-spin text-sm">progress_activity</span>
                {showClarification ? "Processing..." : "Analyzing..."}
              </>
            ) : (
              <>
                <span className="material-symbols-outlined text-[16px]">
                  {showClarification ? "settings_suggest" : (enableClarification ? "psychology_alt" : "policy")}
                </span>
                {showClarification ? "Start Verified Audit" : (enableClarification ? "Next Phase" : "Start Verified Audit")}
              </>
            )}
          </button>
          
          <div className="mt-6 flex justify-center text-[10px] font-mono text-text-muted uppercase tracking-[0.2em]">
            Mode: <span className="text-text font-bold ml-2">{depthLabel} Assessment</span>
          </div>
        </div>
      </form>
    </div>
  );
}
