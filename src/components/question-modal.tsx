"use client";

import { useState, useEffect } from "react";

interface QuestionModalProps {
  sessionId: string;
  questionId: string;
  question: string;
  context: string;
  options: string[];
  timeout: number;
  onSubmit: (response: string) => void;
  onTimeout: () => void;
}

export default function QuestionModal({
  sessionId,
  questionId,
  question,
  context,
  options,
  timeout,
  onSubmit,
  onTimeout,
}: QuestionModalProps) {
  const [response, setResponse] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [timeRemaining, setTimeRemaining] = useState(timeout);

  useEffect(() => {
    const interval = setInterval(() => {
      setTimeRemaining((prev) => {
        if (prev <= 1) {
          clearInterval(interval);
          onTimeout();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(interval);
  }, [timeout, onTimeout]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!response.trim() || submitting) return;

    setSubmitting(true);
    try {
      const res = await fetch(`http://localhost:9090/api/checks/${sessionId}/answer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question_id: questionId,
          response: response.trim(),
        }),
      });

      if (res.ok) {
        onSubmit(response.trim());
      } else {
        console.error("Failed to submit answer");
      }
    } catch (err) {
      console.error("Error submitting answer:", err);
    } finally {
      setSubmitting(false);
    }
  };

  const handleOptionClick = (option: string) => {
    setResponse(option);
  };

  const progress = (timeRemaining / timeout) * 100;
  const isLowTime = progress < 30;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm animate-fadeIn">
      <div className="bg-surface border border-amber/30 rounded-2xl max-w-2xl w-full mx-4 overflow-hidden animate-scale-up" style={{ boxShadow: "var(--shadow-lg)" }}>
        {/* Header */}
        <div className="bg-amber-softer border-b border-amber/20 px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-amber-soft flex items-center justify-center">
                <span className="material-symbols-outlined text-amber">help</span>
              </div>
              <div>
                <h2 className="text-lg font-display">Mid-Check Question</h2>
                <p className="text-xs text-text-secondary">The fact check agent needs your input</p>
              </div>
            </div>
            <div className="text-right">
              <div className="text-xs text-text-secondary mb-1">Time Remaining</div>
              <div className={`text-lg font-mono font-bold ${isLowTime ? "text-gold" : "text-amber"}`}>{timeRemaining}s</div>
            </div>
          </div>

          {/* Progress Bar */}
          <div className="mt-3 h-1 bg-obs-border rounded-full overflow-hidden">
            <div
              className={`h-full transition-all duration-1000 ease-linear ${isLowTime ? "bg-gold" : "bg-amber"}`}
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>

        {/* Content */}
        <form onSubmit={handleSubmit} className="p-6 space-y-6">
          {context && (
            <div className="bg-surface-inset/50 border border-obs-border rounded-xl p-4">
              <div className="flex items-start gap-2">
                <span className="material-symbols-outlined text-sm text-text-secondary mt-0.5">info</span>
                <div className="text-sm text-text-secondary">{context}</div>
              </div>
            </div>
          )}

          <div>
            <label className="block text-sm font-medium text-text-secondary mb-3">Question:</label>
            <p className="text-base text-text leading-relaxed">{question}</p>
          </div>

          {options && options.length > 0 && (
            <div className="space-y-2">
              <label className="block text-sm font-medium text-text-secondary mb-2">Suggested Options:</label>
              <div className="grid grid-cols-1 gap-2">
                {options.map((option, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => handleOptionClick(option)}
                    className={`text-left px-4 py-3 rounded-xl border transition-all ${
                      response === option
                        ? "bg-amber-soft border-amber text-text"
                        : "bg-surface-inset border-obs-border text-text-secondary hover:border-amber/50"
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <span className={`w-4 h-4 rounded-full border-2 flex-shrink-0 ${
                        response === option ? "border-amber bg-amber" : "border-text-muted"
                      }`} />
                      <span className="text-sm">{option}</span>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}

          <div>
            <label className="block text-sm font-medium text-text-secondary mb-2">
              {options && options.length > 0 ? "Or provide your own answer:" : "Your Answer:"}
            </label>
            <textarea
              value={response}
              onChange={(e) => setResponse(e.target.value)}
              className="obs-input w-full min-h-[100px] resize-none"
              placeholder="Type your response here..."
              disabled={submitting}
            />
          </div>

          <div className="flex items-center justify-between pt-2">
            <div className="text-xs text-text-muted">
              <span className="material-symbols-outlined text-sm align-middle mr-1">schedule</span>
              Verification will continue autonomously if no answer is provided
            </div>
            <button
              type="submit"
              disabled={!response.trim() || submitting}
              className="px-6 py-2.5 bg-amber hover:bg-amber-hover text-white font-semibold rounded-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
              style={{ boxShadow: "0 2px 8px rgb(var(--amber) / 0.25)" }}
            >
              {submitting ? (
                <>
                  <span className="material-symbols-outlined animate-spin text-lg">progress_activity</span>
                  Submitting...
                </>
              ) : (
                <>
                  <span className="material-symbols-outlined text-lg">send</span>
                  Submit Answer
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
