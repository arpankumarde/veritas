const API_BASE = "http://localhost:9090";

export interface Session {
  session_id: string;
  goal: string;
  claim?: string;
  status: string;
  verdict?: string;
  max_iterations: number;
  created_at: string;
  completed_at?: string | null;
  elapsed_seconds?: number;
  iteration_count?: number;
  paused_at?: string | null;
}

export interface Evidence {
  id: number;
  session_id: string;
  content: string;
  finding_type: string;
  evidence_type?: string;
  source_url?: string | null;
  confidence?: number | null;
  search_query?: string | null;
  created_at: string;
  verification_status?: string | null;
  verification_method?: string | null;
  kg_support_score?: number | null;
}

export interface VerificationStats {
  total: number;
  verified: number;
  flagged: number;
  rejected: number;
  unverified: number;
}

export interface Report {
  session_id: string;
  report: string;
  path?: string;
}

export interface StartCheckBody {
  claim: string;
  max_iterations: number;
  autonomous: boolean;
  enable_mid_questions?: boolean;
}

export interface StartCheckResponse {
  session_id: string;
  status: string;
  message?: string;
}

export interface WSEvent {
  event_type: string;
  agent: string;
  timestamp: string;
  data: Record<string, unknown>;
  session_id: string;
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });
  if (!res.ok) {
    throw new Error(`API error ${res.status}: ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export function getSessions(limit = 50): Promise<Session[]> {
  return apiFetch<Session[]>(`/api/sessions/?limit=${limit}`);
}

export function getSession(id: string): Promise<Session> {
  return apiFetch<Session>(`/api/sessions/${id}`);
}

export function getSessionStats(id: string): Promise<{ findings: number; sources: number; topics: number }> {
  return apiFetch(`/api/sessions/${id}/stats`);
}

export function getEvidence(id: string, params?: URLSearchParams): Promise<Evidence[]> {
  const qs = params ? `?${params.toString()}` : "";
  return apiFetch<Evidence[]>(`/api/sessions/${id}/evidence${qs}`);
}

export function getSources(id: string, limit = 500): Promise<any[]> {
  return apiFetch(`/api/sessions/${id}/sources?limit=${limit}`);
}

export function getReport(id: string): Promise<Report> {
  return apiFetch<Report>(`/api/sessions/${id}/report`);
}

export function getVerificationResults(id: string): Promise<any[]> {
  return apiFetch(`/api/sessions/${id}/verification/results`);
}

export function getVerificationStats(id: string): Promise<VerificationStats> {
  return apiFetch<VerificationStats>(`/api/sessions/${id}/verification/stats`);
}

export function getAgentDecisions(id: string): Promise<any[]> {
  return apiFetch(`/api/sessions/${id}/agents/decisions`);
}

export function getEvents(id: string, limit = 1000, order = "desc"): Promise<WSEvent[]> {
  return apiFetch(`/api/events/${id}?limit=${limit}&order=${order}`);
}

export function startCheck(body: StartCheckBody): Promise<StartCheckResponse> {
  return apiFetch<StartCheckResponse>("/api/checks/start", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function pauseCheck(id: string): Promise<any> {
  return apiFetch(`/api/checks/${id}/pause`, { method: "POST" });
}

export function resumeCheck(id: string): Promise<any> {
  return apiFetch(`/api/checks/${id}/resume`, { method: "POST" });
}

export function clarify(goal: string, maxQuestions = 4): Promise<{ questions: any[] }> {
  return apiFetch("/api/checks/clarify", {
    method: "POST",
    body: JSON.stringify({ goal, max_questions: maxQuestions }),
  });
}

export function enrich(goal: string, questions: any[], answers: Record<string, string>): Promise<{ enriched_goal: string }> {
  return apiFetch("/api/checks/enrich", {
    method: "POST",
    body: JSON.stringify({ goal, questions, answers }),
  });
}

// Legacy aliases
export const startResearch = startCheck;
export const getFindings = getEvidence;

export function getWSUrl(sessionId: string): string {
  return `ws://localhost:9090/ws/${sessionId}`;
}
