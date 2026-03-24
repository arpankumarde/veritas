"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import NewCheckForm from "@/components/new-check-form";
import Link from "next/link";

interface Session {
  session_id: string;
  goal: string;
  max_iterations: number;
  status: string;
  created_at: string;
  completed_at?: string | null;
  iteration_count?: number;
}

export default function DashboardPage() {
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
    <div className="min-h-screen bg-white flex flex-col text-text font-sans">
      {/* Dashboard Header */}
      <header className="border-b border-obs-border bg-white/80 backdrop-blur-md sticky top-0 z-20">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-3 group">
            <div className="relative flex items-center justify-center w-8 h-8 rounded-lg bg-amber-soft border border-amber/10 group-hover:bg-amber-soft transition-colors text-amber">
               <span className="material-symbols-outlined text-lg font-bold">verified</span>
            </div>
            <div>
              <h1 className="text-base font-display font-bold tracking-tight text-text uppercase">Veritas</h1>
              <p className="text-[9px] text-text-muted font-mono uppercase tracking-widest leading-none">Command Center</p>
            </div>
          </Link>
          
          <div className="flex items-center gap-4">
            <button
              onClick={() => setShowNewCheck(true)}
              className="obs-btn btn-primary bg-amber hover:bg-amber-hover text-white rounded-lg px-5 py-2 text-[13px] font-bold uppercase tracking-wider transition-all shadow-none"
            >
              <span className="material-symbols-outlined text-sm">add</span>
              New Audit
            </button>
          </div>
        </div>
      </header>

      {/* Main Dashboard UI */}
      <main className="flex-1 max-w-7xl mx-auto w-full px-6 py-12">
        <div className="flex flex-col md:flex-row justify-between items-baseline mb-12 gap-6">
           <div>
             <h2 className="text-3xl font-display font-semibold mb-2">Fact Check Overview</h2>
             <p className="text-text-secondary text-sm">System Status: <span className="text-emerald font-bold uppercase tracking-widest text-[10px]">Operational</span></p>
           </div>
           
           <div className="flex gap-4 items-center">
              <div className="px-4 py-2 bg-surface border border-obs-border rounded-lg text-[11px] font-mono font-bold uppercase tracking-widest text-text-muted">
                 ID: {Math.random().toString(16).slice(2, 8).toUpperCase()}
              </div>
           </div>
        </div>

        {/* Stats Grid (Dify Style) */}
        <div className="grid grid-cols-1 md:grid-cols-3 bg-white border border-obs-border rounded-lg overflow-hidden mb-16">
          <div className="p-8 border-b md:border-b-0 md:border-r border-obs-border hover:bg-surface transition-colors">
            <p className="text-[10px] font-mono font-bold text-text-muted uppercase tracking-[0.2em] mb-4">Total Audits</p>
            <p className="font-display text-4xl font-bold text-text mb-1">{sessions.length}</p>
            <p className="text-[11px] text-text-secondary">Historical consensus record</p>
          </div>
          <div className="p-8 border-b md:border-b-0 md:border-r border-obs-border hover:bg-surface transition-colors">
            <p className="text-[10px] font-mono font-bold text-text-muted uppercase tracking-[0.2em] mb-4">Active Checks</p>
            <p className="font-display text-4xl font-bold text-amber mb-1">{activeSessions.length}</p>
            <div className="flex items-center gap-2">
               <span className="w-1.5 h-1.5 rounded-full bg-emerald animate-breathe" />
               <p className="text-[11px] text-text-secondary">AI processing active</p>
            </div>
          </div>
          <div className="p-8 hover:bg-surface transition-colors">
            <p className="text-[10px] font-mono font-bold text-text-muted uppercase tracking-[0.2em] mb-4">System Efficiency</p>
            <p className="font-display text-4xl font-bold text-text mb-1">98.4<span className="text-xl opacity-60 ml-0.5">%</span></p>
            <p className="text-[11px] text-text-secondary">Consensus accuracy metric</p>
          </div>
        </div>

        {/* Audit Sessions */}
        <div className="space-y-16">
          {/* Active Audits Section */}
          {activeSessions.length > 0 && (
            <div>
              <div className="flex items-center gap-4 mb-8">
                <h3 className="text-[12px] font-mono font-bold text-amber uppercase tracking-[0.3em]">Active Processing</h3>
                <div className="h-px flex-1 bg-obs-border opacity-50" />
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {activeSessions.map((session) => (
                  <SessionCard key={session.session_id} session={session} onClick={() => router.push(`/check/${session.session_id}`)} />
                ))}
              </div>
            </div>
          )}

          {/* History Audits Section */}
          <div>
            <div className="flex items-center gap-4 mb-8">
              <h3 className="text-[12px] font-mono font-bold text-text-muted uppercase tracking-[0.3em]">Audit History</h3>
              <div className="h-px flex-1 bg-obs-border opacity-50" />
            </div>

            {loading ? (
              <div className="p-12 border border-obs-border border-dashed rounded-lg text-center">
                 <span className="material-symbols-outlined animate-spin text-amber text-3xl mb-4">progress_activity</span>
                 <p className="text-sm font-mono text-text-secondary">Accessing audit logs...</p>
              </div>
            ) : completedSessions.length === 0 && activeSessions.length === 0 ? (
              <div className="py-20 flex flex-col items-center justify-center text-center border border-obs-border rounded-lg bg-surface">
                <div className="w-16 h-16 rounded-xl bg-amber-soft flex items-center justify-center mb-6">
                  <span className="material-symbols-outlined text-amber text-3xl">policy</span>
                </div>
                <h3 className="text-xl font-display font-semibold mb-2">No audits on record</h3>
                <p className="text-sm text-text-secondary max-w-sm mb-8">
                  Veritas is ready to perform its first claim verification. Initialize a new audit session below.
                </p>
                <button
                  onClick={() => setShowNewCheck(true)}
                  className="obs-btn btn-primary bg-amber hover:bg-amber-hover text-white rounded-lg px-8 py-3 text-sm font-bold shadow-none"
                >
                  Initialize New Audit
                </button>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {completedSessions.map((session) => (
                  <SessionCard key={session.session_id} session={session} onClick={() => router.push(`/check/${session.session_id}`)} />
                ))}
              </div>
            )}
          </div>
        </div>
      </main>

      {/* New Check Modal */}
      {showNewCheck && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-white/60 backdrop-blur-sm p-4 md:p-12 overflow-y-auto">
          <div className="w-full max-w-[640px] animate-scale-up shadow-2xl rounded-lg my-auto">
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

  const statusColor = isCrashed ? "bg-rose" : isPaused ? "bg-gold" : isActive ? "bg-amber" : "bg-text-muted opacity-40";
  const statusBorder = isCrashed ? "border-rose" : isPaused ? "border-gold" : isActive ? "border-amber" : "border-obs-border";
  const statusText = isCrashed ? "text-rose" : isPaused ? "text-gold" : isActive ? "text-amber" : "text-text-muted";

  return (
    <button
      onClick={onClick}
      className={`p-6 bg-white border ${statusBorder} hover:shadow-md transition-all text-left group relative overflow-hidden rounded-lg`}
    >
      <div className="flex items-center justify-between mb-8">
         <div className="flex items-center gap-2">
            <span className={`w-1.5 h-1.5 rounded-full ${statusColor} ${isActive ? 'animate-breathe' : ''}`} />
            <span className={`text-[10px] font-mono font-bold uppercase tracking-widest ${statusText}`}>
              {session.status}
            </span>
         </div>
         <span className="material-symbols-outlined text-[16px] text-text-muted group-hover:text-amber transition-colors">
            north_east
         </span>
      </div>

      <h3 className="text-sm font-semibold text-text mb-6 line-clamp-2 leading-relaxed min-h-[3rem]">
        {session.goal}
      </h3>

      <div className="flex items-center justify-between pt-4 border-t border-obs-border/30 text-[10px] font-mono text-text-secondary uppercase">
        <span className="flex items-center gap-2">
          <span className="material-symbols-outlined text-sm opacity-60">history</span>
          {timeDisplay}
        </span>
        <span className="flex items-center gap-2">
          {session.iteration_count ?? 0}/{session.max_iterations} <span className="opacity-40 tracking-normal italic">iters</span>
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

  if (diffMins < 1) return "initialized";
  if (diffMins < 60) return `${diffMins}m active`;
  return `${Math.floor(diffMins / 60)}h ${diffMins % 60}m`;
}

function getDuration(startDateString: string, endDateString?: string | null): string {
  if (!endDateString) {
    const start = new Date(startDateString);
    const now = new Date();
    const diffDays = Math.floor((now.getTime() - start.getTime()) / (1000 * 60 * 60 * 24));
    if (diffDays === 0) return "today";
    return `${diffDays}d ago`;
  }

  const start = new Date(startDateString);
  const end = new Date(endDateString);
  const diffMs = end.getTime() - start.getTime();
  const diffSecs = Math.floor(diffMs / 1000);
  const diffMins = Math.floor(diffSecs / 60);

  if (diffSecs < 60) return `${diffSecs}s`;
  if (diffMins < 60) return `${diffMins}m`;
  return `${Math.floor(diffMins / 60)}h ${diffMins % 60}m`;
}
