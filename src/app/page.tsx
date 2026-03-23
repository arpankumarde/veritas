"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import NewCheckForm from "@/components/new-check-form";

interface Session {
  session_id: string;
  goal: string;
  max_iterations: number;
  status: string;
  created_at: string;
  completed_at?: string | null;
  iteration_count?: number;
}

export default function Home() {
  const router = useRouter();
  const [sessions, setSessions] = useState<Session[]>([]);
  const [loading, setLoading] = useState(true);
  const [showNewCheck, setShowNewCheck] = useState(false);

  const fetchSessions = async () => {
    try {
      const response = await fetch("http://localhost:9090/api/sessions/?limit=50");
      if (response.ok) {
        const data = await response.json();
        setSessions(data);
      }
    } catch (err) {
      console.error("Failed to fetch sessions:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSessions();
  }, []);

  const activeStatuses = ["running", "pending", "paused", "crashed"];
  const activeSessions = sessions.filter((s) => activeStatuses.includes(s.status));
  const completedSessions = sessions.filter((s) => !activeStatuses.includes(s.status));

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="border-b border-obs-border bg-surface/40 backdrop-blur-xl sticky top-0 z-20">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="relative flex items-center justify-center w-10 h-10 rounded-xl bg-amber-soft border border-amber/20">
              <span className="material-symbols-outlined text-amber text-xl">verified</span>
              <div className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 rounded-full bg-amber animate-breathe" />
            </div>
            <div>
              <h1 className="text-lg font-display font-semibold tracking-tight text-text">Veritas</h1>
              <p className="text-[11px] text-text-muted font-mono uppercase tracking-widest">Fact Check Observatory</p>
            </div>
          </div>
          <button
            onClick={() => setShowNewCheck(true)}
            className="obs-btn btn-primary"
          >
            <span className="material-symbols-outlined text-lg">add_circle</span>
            New Fact Check
          </button>
        </div>
      </header>

      {/* Accent line */}
      <div className="glow-line" />

      {/* Main Content */}
      <main className="flex-1 max-w-7xl mx-auto w-full px-6 py-10">
        {/* Stats Overview */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5 mb-10 stagger-children">
          <div className="obs-card card-hover animate-rise group">
            <div className="absolute inset-0 bg-gradient-to-br from-amber/5 to-transparent pointer-events-none" />
            <div className="relative">
              <div className="flex justify-between items-start mb-4">
                <p className="text-[11px] font-mono font-medium text-text-muted uppercase tracking-widest">Total Sessions</p>
                <div className="w-8 h-8 rounded-lg bg-amber-soft flex items-center justify-center group-hover:bg-amber/15 transition-colors">
                  <span className="material-symbols-outlined text-amber text-base">folder_open</span>
                </div>
              </div>
              <p className="metric">{sessions.length}</p>
            </div>
          </div>
          <div className="obs-card card-hover animate-rise group">
            <div className="absolute inset-0 bg-gradient-to-br from-emerald/5 to-transparent pointer-events-none" />
            <div className="relative">
              <div className="flex justify-between items-start mb-4">
                <p className="text-[11px] font-mono font-medium text-text-muted uppercase tracking-widest">Active Now</p>
                <div className="w-8 h-8 rounded-lg bg-emerald-soft flex items-center justify-center group-hover:bg-emerald/15 transition-colors">
                  <span className="material-symbols-outlined text-emerald text-base">play_circle</span>
                </div>
              </div>
              <p className="font-mono text-3xl font-bold tracking-tighter text-emerald">{activeSessions.length}</p>
            </div>
          </div>
          <div className="obs-card card-hover animate-rise group">
            <div className="absolute inset-0 bg-gradient-to-br from-violet/5 to-transparent pointer-events-none" />
            <div className="relative">
              <div className="flex justify-between items-start mb-4">
                <p className="text-[11px] font-mono font-medium text-text-muted uppercase tracking-widest">Completed</p>
                <div className="w-8 h-8 rounded-lg bg-violet-soft flex items-center justify-center group-hover:bg-violet/15 transition-colors">
                  <span className="material-symbols-outlined text-violet text-base">check_circle</span>
                </div>
              </div>
              <p className="metric">{completedSessions.length}</p>
            </div>
          </div>
        </div>

        {/* Active Sessions */}
        {activeSessions.length > 0 && (
          <div className="mb-10">
            <div className="flex items-center gap-3 mb-5">
              <div className="w-1.5 h-1.5 rounded-full bg-emerald animate-breathe" />
              <h2 className="text-[11px] font-mono font-semibold text-text-muted uppercase tracking-widest">
                Active Checks
              </h2>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {activeSessions.map((session) => (
                <SessionCard key={session.session_id} session={session} onClick={() => router.push(`/check/${session.session_id}`)} />
              ))}
            </div>
          </div>
        )}

        {/* Session History */}
        <div>
          <div className="flex items-center gap-3 mb-5">
            <div className="w-1.5 h-1.5 rounded-full bg-text-muted" />
            <h2 className="text-[11px] font-mono font-semibold text-text-muted uppercase tracking-widest">
              {activeSessions.length > 0 ? "History" : "All Sessions"}
            </h2>
          </div>

          {loading ? (
            <div className="obs-card">
              <div className="flex items-center gap-3 text-text-secondary">
                <span className="material-symbols-outlined animate-spin text-amber">progress_activity</span>
                <span className="text-sm">Loading sessions...</span>
              </div>
            </div>
          ) : completedSessions.length === 0 && activeSessions.length === 0 ? (
            <div className="obs-card flex flex-col items-center justify-center py-20 text-center">
              <div className="w-20 h-20 rounded-2xl bg-amber-soft border border-amber/20 flex items-center justify-center mb-6">
                <span className="material-symbols-outlined text-amber text-4xl">verified</span>
              </div>
              <h3 className="text-xl font-display font-semibold mb-2">No fact checks yet</h3>
              <p className="text-sm text-text-secondary max-w-md mb-8">
                Launch your first fact check to verify claims with hierarchical evidence analysis and transparent AI reasoning.
              </p>
              <button
                onClick={() => setShowNewCheck(true)}
                className="obs-btn btn-primary text-base px-8 py-3"
              >
                <span className="material-symbols-outlined text-xl">rocket_launch</span>
                Start Fact Check
              </button>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {completedSessions.map((session) => (
                <SessionCard key={session.session_id} session={session} onClick={() => router.push(`/check/${session.session_id}`)} />
              ))}
            </div>
          )}
        </div>
      </main>

      {/* New Check Modal */}
      {showNewCheck && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-md p-4">
          <div className="w-full max-w-[640px]">
            <NewCheckForm
              onClose={() => setShowNewCheck(false)}
              onSuccess={() => {
                setShowNewCheck(false);
                fetchSessions();
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

function SessionCard({ session, onClick }: { session: Session; onClick: () => void }) {
  const isActive = session.status === "running" || session.status === "pending";
  const isPaused = session.status === "paused";
  const isCrashed = session.status === "crashed";
  const timeDisplay = isActive
    ? getElapsedTime(session.created_at)
    : getDuration(session.created_at, session.completed_at);

  const accentGradient = isCrashed
    ? "from-rose/40 via-rose to-rose/40"
    : isPaused
      ? "from-gold/40 via-gold to-gold/40"
      : "from-amber/40 via-amber to-amber/40";
  const statusTextColor = isCrashed ? "text-rose" : isPaused ? "text-gold" : isActive ? "text-emerald" : "text-text-muted";

  return (
    <button
      onClick={onClick}
      className="obs-card card-hover text-left w-full group relative overflow-hidden"
    >
      {(isActive || isPaused || isCrashed) && (
        <div className={`absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r ${accentGradient}`} />
      )}

      <div className="flex items-start justify-between gap-3 mb-4">
        <div className="flex items-center gap-2">
          {isActive ? (
            <span className="relative flex h-2.5 w-2.5">
              <span className="animate-breathe absolute inline-flex h-full w-full rounded-full bg-emerald opacity-75" />
              <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald" />
            </span>
          ) : isPaused || isCrashed ? (
            <span className={`status-dot ${isCrashed ? "bg-rose" : "bg-gold"}`} />
          ) : (
            <span className="status-dot status-dot-idle" />
          )}
          <span className={`text-[10px] font-mono font-bold uppercase tracking-wider ${statusTextColor}`}>
            {session.status}
          </span>
        </div>
        <span className="material-symbols-outlined text-text-muted group-hover:text-amber transition-colors text-lg">
          arrow_forward
        </span>
      </div>

      <h3 className="text-sm font-medium text-text group-hover:text-amber transition-colors line-clamp-2 mb-4 leading-snug">
        {session.goal}
      </h3>

      <div className="flex items-center justify-between text-[11px] text-text-muted font-mono">
        <span className="flex items-center gap-1.5">
          <span className="material-symbols-outlined text-[13px]">{isActive ? "schedule" : isPaused ? "pause" : isCrashed ? "warning" : "timer"}</span>
          {timeDisplay}
        </span>
        <span className="flex items-center gap-1.5">
          <span className="material-symbols-outlined text-[13px]">repeat</span>
          {session.iteration_count ?? 0}/{session.max_iterations}
        </span>
      </div>
    </button>
  );
}

function getElapsedTime(startDateString: string): string {
  const now = new Date();
  const start = new Date(startDateString);
  const diffMs = now.getTime() - start.getTime();
  const diffSecs = Math.floor(diffMs / 1000);
  const diffMins = Math.floor(diffSecs / 60);
  const diffHours = Math.floor(diffMins / 60);

  if (diffMins < 1) return "just started";
  if (diffMins < 60) return `${diffMins}m elapsed`;
  if (diffHours < 24) return `${diffHours}h ${diffMins % 60}m elapsed`;
  const diffDays = Math.floor(diffHours / 24);
  return `${diffDays}d ${diffHours % 24}h elapsed`;
}

function getDuration(startDateString: string, endDateString?: string | null): string {
  if (!endDateString) {
    const now = new Date();
    const start = new Date(startDateString);
    const diffMs = now.getTime() - start.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return start.toLocaleDateString("en-US", { month: "short", day: "numeric" });
  }

  const start = new Date(startDateString);
  const end = new Date(endDateString);
  const diffMs = end.getTime() - start.getTime();
  const diffSecs = Math.floor(diffMs / 1000);
  const diffMins = Math.floor(diffSecs / 60);
  const diffHours = Math.floor(diffMins / 60);

  if (diffSecs < 60) return `${diffSecs}s`;
  if (diffMins < 60) return `${diffMins}m ${diffSecs % 60}s`;
  return `${diffHours}h ${diffMins % 60}m`;
}
