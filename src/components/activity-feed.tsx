"use client";

import { useState, useEffect, useRef, useMemo } from "react";
import { ResearchWebSocket, AgentEvent, ConnectionStatus } from "@/lib/websocket";

interface ActivityFeedProps {
  sessionId: string;
  sharedWs?: ResearchWebSocket | null;
}

export default function ActivityFeed({ sessionId, sharedWs }: ActivityFeedProps) {
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const [connStatus, setConnStatus] = useState<ConnectionStatus>("disconnected");
  const [connDetail, setConnDetail] = useState<string | undefined>();
  const wsRef = useRef<ResearchWebSocket | null>(null);
  const [query, setQuery] = useState("");
  const [selectedTypes, setSelectedTypes] = useState<string[]>([]);
  const [selectedAgents, setSelectedAgents] = useState<string[]>([]);
  const [groupSystemLogs, setGroupSystemLogs] = useState(true);
  const [compactView, setCompactView] = useState(false);
  const [expandedBlocks, setExpandedBlocks] = useState<Record<string, boolean>>({});
  const feedRef = useRef<HTMLDivElement | null>(null);
  const [unseenCount, setUnseenCount] = useState(0);
  const [isAtTop, setIsAtTop] = useState(true);
  const seenKeysRef = useRef<Set<string>>(new Set());
  const [historyLoadedAt, setHistoryLoadedAt] = useState<string | null>(null);
  const [historyStatus, setHistoryStatus] = useState<"idle" | "loading" | "loaded" | "error">("idle");
  const [historySource, setHistorySource] = useState<"api" | "cache" | null>(null);
  const scrollTopBeforeLoadRef = useRef<number | null>(null);
  const [expandedEvents, setExpandedEvents] = useState<Record<string, boolean>>({});

  useEffect(() => {
    // Use shared WebSocket from parent if available, otherwise create own
    const ownsWs = !sharedWs;
    const ws = sharedWs ?? new ResearchWebSocket(sessionId);

    let cancelled = false;
    const cacheKey = `activity_cache_${sessionId}`;
    try {
      const cachedRaw = localStorage.getItem(cacheKey);
      if (cachedRaw) {
        const cachedEvents: AgentEvent[] = JSON.parse(cachedRaw);
        if (Array.isArray(cachedEvents) && cachedEvents.length > 0) {
          setEvents((prev) => mergeEvents(prev, cachedEvents, seenKeysRef.current));
          setHistoryLoadedAt(new Date().toISOString());
          setHistoryStatus("loaded");
          setHistorySource("cache");
        }
      }
    } catch {
      // Ignore cache parse errors
    }

    (async () => {
      try {
        setHistoryStatus("loading");
        if (feedRef.current) {
          scrollTopBeforeLoadRef.current = feedRef.current.scrollTop;
        }
        const response = await fetch(
          `http://localhost:9090/api/events/${sessionId}?limit=1000&order=desc`
        );
        if (!response.ok) {
          setHistoryStatus("error");
          return;
        }
        const history: AgentEvent[] = await response.json();
        if (cancelled || !Array.isArray(history)) {
          return;
        }
        setEvents((prev) => mergeEvents(prev, history, seenKeysRef.current));
        setHistoryLoadedAt(new Date().toISOString());
        setHistoryStatus("loaded");
        setHistorySource("api");
      } catch {
        setHistoryStatus("error");
      }
    })();

    const unsubscribe = ws.onEvent((event) => {
      const key = getEventKey(event);
      if (seenKeysRef.current.has(key)) {
        return;
      }
      seenKeysRef.current.add(key);
      setEvents((prev) => [event, ...prev].slice(0, 2000));
    });

    if (ownsWs) {
      ws.connect();
    }
    wsRef.current = ws;

    const unsubStatus = ws.onStatusChange((status, detail) => {
      setConnStatus(status);
      setConnDetail(detail);
      setConnected(status === "connected");
    });

    const checkConnection = setInterval(() => {
      setConnected(ws.isConnected());
    }, 1000);

    return () => {
      cancelled = true;
      clearInterval(checkConnection);
      unsubscribe();
      unsubStatus();
      if (ownsWs) {
        ws.disconnect();
      }
    };
  }, [sessionId, sharedWs]);

  useEffect(() => {
    const cacheKey = `activity_cache_${sessionId}`;
    try {
      const snapshot = events.slice(0, 1000);
      localStorage.setItem(cacheKey, JSON.stringify(snapshot));
    } catch {
      // Ignore cache write errors
    }
  }, [events, sessionId]);

  const typeOptions = useMemo(() => {
    const base = ["thinking", "action", "finding", "synthesis", "error", "system"];
    const dynamic = Array.from(new Set(events.map((e) => e.event_type)));
    return Array.from(new Set([...base, ...dynamic]));
  }, [events]);

  const agentOptions = useMemo(() => {
    return Array.from(new Set(events.map((e) => e.agent))).sort();
  }, [events]);

  const filteredEvents = useMemo(() => {
    const q = query.trim().toLowerCase();
    return events.filter((event) => {
      if (selectedTypes.length > 0 && !selectedTypes.includes(event.event_type)) {
        return false;
      }
      if (selectedAgents.length > 0 && !selectedAgents.includes(event.agent)) {
        return false;
      }
      if (!q) {
        return true;
      }
      const haystack = getEventSearchText(event);
      return haystack.includes(q);
    });
  }, [events, query, selectedTypes, selectedAgents]);

  const displayItems = useMemo(() => {
    const ordered = [...filteredEvents].sort(
      (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
    );
    const items: DisplayItem[] = [];
    let currentBlock: LogBlock | null = null;
    let lastEventKey: string | null = null;
    let lastEventTime = 0;
    let blockCounter = 0;

    const flushBlock = () => {
      if (currentBlock) {
        items.push({ kind: "logBlock", block: currentBlock });
        currentBlock = null;
      }
    };

    for (const event of ordered) {
      if (groupSystemLogs && event.event_type === "system") {
        const message = extractSystemMessage(event);
        if (!message) {
          continue;
        }
        if (!currentBlock || !shouldGroupLog(currentBlock, event)) {
          flushBlock();
          currentBlock = createLogBlock(event, blockCounter++);
        }
        addLogLine(currentBlock, message);
        currentBlock.end = event.timestamp;
        continue;
      }

      flushBlock();

      const eventKey = `${event.event_type}|${event.agent}|${JSON.stringify(event.data || {})}`;
      const eventTime = new Date(event.timestamp).getTime();
      const lastItem = items[items.length - 1];
      if (
        lastItem?.kind === "event"
        && lastEventKey === eventKey
        && Math.abs(eventTime - lastEventTime) <= 1000
      ) {
        lastItem.repeat += 1;
        lastItem.event = event;
      } else {
        items.push({
          kind: "event",
          event,
          repeat: 1,
          phase: getEventPhase(event),
        });
        lastEventKey = eventKey;
      }
      lastEventTime = eventTime;
    }

    flushBlock();
    return items;
  }, [filteredEvents, groupSystemLogs]);

  useEffect(() => {
    if (isAtTop) {
      setUnseenCount(0);
      return;
    }
    setUnseenCount((prev) => prev + 1);
  }, [displayItems.length, isAtTop]);

  useEffect(() => {
    if (!feedRef.current) {
      return;
    }
    if (scrollTopBeforeLoadRef.current !== null) {
      feedRef.current.scrollTop = scrollTopBeforeLoadRef.current;
      scrollTopBeforeLoadRef.current = null;
    }
  }, [historyStatus]);

  const handleScroll = () => {
    const el = feedRef.current;
    if (!el) {
      return;
    }
    const nearTop = el.scrollTop <= 24;
    setIsAtTop(nearTop);
    if (nearTop) {
      setUnseenCount(0);
    }
  };

  const stats = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const event of filteredEvents) {
      counts[event.event_type] = (counts[event.event_type] || 0) + 1;
    }
    return counts;
  }, [filteredEvents]);

  const toggleType = (type: string) => {
    setSelectedTypes((prev) =>
      prev.includes(type) ? prev.filter((t) => t !== type) : [...prev, type]
    );
  };

  const toggleAgent = (agent: string) => {
    setSelectedAgents((prev) =>
      prev.includes(agent) ? prev.filter((a) => a !== agent) : [...prev, agent]
    );
  };

  const resetFilters = () => {
    setSelectedTypes([]);
    setSelectedAgents([]);
    setQuery("");
  };

  const toggleBlock = (id: string) => {
    setExpandedBlocks((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const toggleEvent = (id: string) => {
    setExpandedEvents((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  return (
    <div className="obs-card">
      {/* Header */}
      <div className="flex flex-col gap-3 mb-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <h3 className="text-lg font-semibold">Live Activity Feed</h3>
            <div className="flex items-center gap-2 text-xs text-text-secondary">
              <span>{filteredEvents.length} events</span>
              <span>&middot;</span>
              <span>{Object.keys(stats).length} types</span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <div
              className={`w-2 h-2 rounded-full ${
                connStatus === "connected" ? "bg-emerald animate-breathe"
                : connStatus === "reconnecting" ? "bg-gold animate-pulse"
                : "bg-text-muted"
              }`}
            />
            {/* <span className="text-sm text-text-secondary">
              {connStatus === "connected" ? "Connected"
               : connStatus === "reconnecting" ? "Reconnecting"
               : connStatus === "connecting" ? "Connecting"
               : "Disconnected"}
            </span> */}
          </div>
        </div>

        <div className="flex flex-wrap gap-2 items-center">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Filter logs (e.g., search, query, error, kg)"
            className="obs-input h-9 w-full md:w-64 text-sm"
          />
          <button
            type="button"
            className={`obs-chip ${groupSystemLogs ? "chip-active" : ""}`}
            onClick={() => setGroupSystemLogs((prev) => !prev)}
          >
            Group System Logs
          </button>
          <button
            type="button"
            className={`obs-chip ${compactView ? "chip-active" : ""}`}
            onClick={() => setCompactView((prev) => !prev)}
          >
            Compact View
          </button>
          <button
            type="button"
            className="obs-chip"
            onClick={resetFilters}
          >
            Reset Filters
          </button>
        </div>

        <div className="flex flex-wrap gap-2">
          {typeOptions.map((type) => (
            <button
              key={type}
              type="button"
              className={`obs-chip ${selectedTypes.includes(type) ? "chip-active" : ""}`}
              onClick={() => toggleType(type)}
            >
              {type}
              {stats[type] ? <span className="ml-1 text-xs text-text-muted">({stats[type]})</span> : null}
            </button>
          ))}
        </div>

        <div className="flex flex-wrap gap-2">
          {agentOptions.map((agent) => (
            <button
              key={agent}
              type="button"
              className={`obs-chip ${selectedAgents.includes(agent) ? "chip-active" : ""}`}
              onClick={() => toggleAgent(agent)}
            >
              {agent}
            </button>
          ))}
        </div>
      </div>

      {/* Connection Status Banner */}
      {connStatus === "reconnecting" && (
        <div className="mb-3 flex items-center gap-2 text-xs bg-gold/10 border border-gold/30 text-gold rounded-2xl px-4 py-2.5">
          <span className="material-symbols-outlined text-sm animate-spin">progress_activity</span>
          <span>Reconnecting{connDetail ? ` -- ${connDetail}` : "..."}</span>
        </div>
      )}
      {connStatus === "disconnected" && connDetail && (
        <div className="mb-3 flex items-center justify-between gap-2 text-xs bg-rose/10 border border-rose/30 text-rose rounded-2xl px-4 py-2.5">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-sm">cloud_off</span>
            <span>Disconnected -- {connDetail}. Live updates paused.</span>
          </div>
          <button
            type="button"
            className="obs-chip text-xs"
            onClick={() => wsRef.current?.connect()}
          >
            Retry
          </button>
        </div>
      )}

      {/* Events List */}
      {unseenCount > 0 ? (
        <button
          type="button"
          className="mb-3 w-full text-xs bg-amber-soft border border-amber/30 text-amber rounded-2xl py-2 hover:bg-amber/15 transition-colors"
          onClick={() => {
            if (feedRef.current) {
              feedRef.current.scrollTop = 0;
            }
            setUnseenCount(0);
          }}
        >
          {unseenCount} new update{unseenCount > 1 ? "s" : ""} -- Jump to latest
        </button>
      ) : null}
      {historyStatus === "loading" ? (
        <div className="mb-3 text-xs text-text-muted flex items-center gap-2">
          <span className="animate-breathe">Loading history...</span>
        </div>
      ) : null}
      {historyStatus === "loaded" && historyLoadedAt ? (
        <div className="mb-3 text-xs text-text-muted flex items-center gap-2">
          <span className="obs-badge badge-system">History loaded</span>
          <span>{formatTimestamp(historyLoadedAt)}</span>
          {historySource === "cache" ? (
            <span className="obs-badge badge-system">cached</span>
          ) : null}
        </div>
      ) : null}
      {historyStatus === "error" ? (
        <div className="mb-3 text-xs text-rose">
          Failed to load history. Live updates will continue.
        </div>
      ) : null}
      <div
        ref={feedRef}
        onScroll={handleScroll}
        className="space-y-2 max-h-[36rem] overflow-y-auto"
      >
        {displayItems.length === 0 ? (
          <div className="text-center py-8 text-text-secondary">
            <svg
              className="w-12 h-12 mx-auto mb-3 opacity-50"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M13 10V3L4 14h7v7l9-11h-7z"
              />
            </svg>
            <p className="text-sm">
              {historyStatus === "loading" ? "Loading history..." : "Waiting for events..."}
            </p>
            <p className="text-xs mt-1">
              Events will appear here when agents start working
            </p>
          </div>
        ) : (
          displayItems.map((item) => {
            if (item.kind === "logBlock") {
              const block = item.block;
              const expanded = expandedBlocks[block.id] ?? false;
              const phase = getPhaseForLogBlock(block);
              const previewLines = block.lines.slice(0, 3);
              return (
                <div
                  key={block.id}
                  className={`bg-surface-inset border border-obs-border rounded-2xl p-3 ${
                    compactView ? "text-xs" : "text-sm"
                  }`}
                >
                  <div className="flex items-start justify-between gap-3 mb-2">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="obs-badge badge-system">log batch</span>
                      <span className={`obs-badge ${phaseBadgeClass[phase]}`}>
                        {phase}
                      </span>
                      <span className="text-xs text-text-muted">
                        {block.agent}
                      </span>
                      <span className="text-xs text-text-muted">
                        {block.lines.length} lines
                      </span>
                    </div>
                    <button
                      type="button"
                      className="obs-chip"
                      onClick={() => toggleBlock(block.id)}
                    >
                      {expanded ? "Collapse" : "Expand"}
                    </button>
                  </div>

                  <div className="text-xs text-text-muted mb-2">
                    {formatTimestamp(block.start)}
                  </div>

                  {expanded ? (
                    <div className="space-y-1 font-mono text-xs">
                      {block.lines.map((line, idx) => (
                        <LogLineRow key={`${block.id}-${idx}`} line={line} />
                      ))}
                    </div>
                  ) : (
                    <div className="space-y-1 text-xs text-text-secondary">
                      {previewLines.map((line, idx) => (
                        <LogLineRow key={`${block.id}-preview-${idx}`} line={line} />
                      ))}
                      {block.lines.length > previewLines.length ? (
                        <div className="text-text-muted">
                          +{block.lines.length - previewLines.length} more lines
                        </div>
                      ) : null}
                    </div>
                  )}
                </div>
              );
            }

            const event = item.event;
            const eventId = getEventKey(event);
            const isExpanded = expandedEvents[eventId] ?? false;
            const phase = item.phase;
            return (
              <div
                key={`${event.timestamp}-${event.event_type}-${event.agent}`}
                className={`bg-surface-inset border border-obs-border rounded-2xl ${
                  compactView ? "p-2" : "p-3"
                } hover:border-amber/30 transition-colors`}
              >
                <div className="flex items-start justify-between gap-3 mb-2">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className={`obs-badge ${getBadgeClass(event.event_type)}`}>
                      {event.event_type}
                    </span>
                    <span className={`obs-badge ${phaseBadgeClass[phase]}`}>
                      {phase}
                    </span>
                    <span className="text-xs text-text-muted">
                      {event.agent}
                    </span>
                    {item.repeat > 1 ? (
                      <span className="obs-badge badge-system">x{item.repeat}</span>
                    ) : null}
                  </div>
                  <span className="text-xs text-text-muted">
                    {formatTimestamp(event.timestamp)}
                  </span>
                </div>

                <div className="text-sm text-text-secondary">
                  {renderEventData(event, {
                    expanded: isExpanded,
                    onToggle: () => toggleEvent(eventId),
                  })}
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

function getBadgeClass(eventType: string): string {
  switch (eventType) {
    case "thinking":
      return "badge-thinking";
    case "action":
      return "badge-action";
    case "finding":
      return "badge-finding";
    case "synthesis":
      return "badge-thinking";
    case "error":
      return "badge-error";
    case "system":
      return "badge-system";
    default:
      return "badge-thinking";
  }
}

function formatTimestamp(timestamp: string): string {
  const date = new Date(timestamp);
  return date.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function renderEventData(
  event: AgentEvent,
  opts: { expanded: boolean; onToggle: () => void }
): React.ReactNode {
  const data = event.data || {};
  const { expanded, onToggle } = opts;
  const maybeLongText = getEventPrimaryText(event);
  if (event.event_type === "finding") {
    return (
      <div className="space-y-2">
        {data.content ? <p>{data.content}</p> : null}
        <div className="flex flex-wrap gap-2 text-xs text-text-muted">
          {data.source ? (
            <a
              href={data.source}
              target="_blank"
              rel="noreferrer"
              className="text-amber hover:underline"
            >
              {truncateUrl(String(data.source))}
            </a>
          ) : null}
          {typeof data.confidence === "number" ? (
            <span className="obs-badge badge-system">
              Confidence {Math.round(data.confidence * 100)}%
            </span>
          ) : null}
        </div>
      </div>
    );
  }

  if (event.event_type === "thinking" && data.thought) {
    return renderExpandableText(maybeLongText, expanded, onToggle, "italic");
  }

  if (event.event_type === "action" && data.action) {
    const isSearch = data.action === "search";
    if (isSearch) {
      return (
        <div className="space-y-2">
          {data.query ? (
            <p className="font-medium">
              Search: <span className="text-amber">{data.query}</span>
            </p>
          ) : (
            <p className="font-medium">Search</p>
          )}
          {(data.results_count != null || data.evidence_extracted || data.iteration) ? (
            <div className="flex gap-3 text-xs text-text-muted">
              {data.results_count != null ? <span>{data.results_count} results</span> : null}
              {data.evidence_extracted ? (
                <span>{data.evidence_extracted} evidence extracted</span>
              ) : null}
              {data.iteration ? <span>Iteration {data.iteration}</span> : null}
            </div>
          ) : null}
          {Array.isArray(data.result_titles) && data.result_titles.length > 0 ? (
            <div className="mt-1 space-y-1.5 text-xs border-l-2 border-amber/30 pl-3">
              {data.result_titles.map((r: { title: string; url?: string }, i: number) => (
                <div key={i} className="flex items-start gap-2">
                  <span className="text-amber/60 shrink-0 font-mono">{i + 1}.</span>
                  <div className="min-w-0">
                    <p className="text-text font-medium">{r.title}</p>
                    {r.url ? (
                      <a
                        href={r.url}
                        target="_blank"
                        rel="noreferrer"
                        className="text-amber/70 hover:underline truncate block text-[11px]"
                      >
                        {truncateUrl(r.url, 60)}
                      </a>
                    ) : null}
                  </div>
                </div>
              ))}
            </div>
          ) : null}
          {Array.isArray(data.queries_used) && data.queries_used.length > 1 ? (
            <div className="text-[11px] text-text-muted">
              Also searched: {data.queries_used.slice(1).join(", ")}
            </div>
          ) : null}
        </div>
      );
    }
    const isDeepScrape = data.action === "deep_scrape";
    if (isDeepScrape) {
      return (
        <div className="space-y-2">
          <p className="font-medium">
            Deep Scraping <span className="text-amber">{data.urls_count} sources</span>
          </p>
          <div className="flex gap-3 text-xs text-text-muted flex-wrap">
            {data.platform_urls ? (
              <span>{data.platform_urls} platform URLs</span>
            ) : null}
            {Array.isArray(data.platforms) && data.platforms.length > 0 ? (
              <span className="flex gap-1 flex-wrap">
                {data.platforms.map((p: string) => (
                  <span key={p} className="obs-badge badge-action">{p}</span>
                ))}
              </span>
            ) : null}
          </div>
        </div>
      );
    }
    return (
      <div className="space-y-1">
        <p className="font-medium">{data.action}</p>
        {data.iteration ? (
          <p className="text-xs text-text-muted">Iteration {data.iteration}</p>
        ) : null}
      </div>
    );
  }

  if (event.event_type === "synthesis" && data.message) {
    return (
      <div className="space-y-1">
        <p>{data.message}</p>
        {typeof data.progress === "number" ? (
          <p className="text-xs text-text-muted">Progress {data.progress}%</p>
        ) : null}
      </div>
    );
  }

  if (event.event_type === "error" && data.error) {
    return <p className="text-rose">{data.error}</p>;
  }

  if (data.message) {
    return renderExpandableText(maybeLongText, expanded, onToggle);
  }

  if (data.content) {
    return renderExpandableText(maybeLongText, expanded, onToggle);
  }

  return (
    <pre className="text-xs font-mono overflow-x-auto">
      {JSON.stringify(data, null, 2)}
    </pre>
  );
}

type Phase =
  | "reasoning"
  | "action"
  | "search"
  | "finding"
  | "observe"
  | "knowledge"
  | "synthesis"
  | "error"
  | "system";

const phaseBadgeClass: Record<Phase, string> = {
  reasoning: "bg-violet/20 text-violet",
  action: "bg-amber/20 text-amber",
  search: "bg-amber/15 text-amber",
  finding: "bg-emerald/20 text-emerald",
  observe: "bg-gold/20 text-gold",
  knowledge: "bg-amber/15 text-amber",
  synthesis: "bg-violet/20 text-violet",
  error: "bg-rose/20 text-rose",
  system: "bg-obs-border/50 text-text-secondary",
};

type DisplayItem =
  | { kind: "event"; event: AgentEvent; repeat: number; phase: Phase }
  | { kind: "logBlock"; block: LogBlock };

type LogLine = { text: string; count: number; tone: LogTone };
type LogTone = "divider" | "label" | "url" | "meta" | "list" | "text";
type LogBlock = {
  id: string;
  agent: string;
  start: string;
  end: string;
  lines: LogLine[];
};

function getEventSearchText(event: AgentEvent): string {
  const data = event.data || {};
  return [
    event.event_type,
    event.agent,
    event.timestamp,
    JSON.stringify(data),
  ]
    .join(" ")
    .toLowerCase();
}

function getEventKey(event: AgentEvent): string {
  return [
    event.timestamp,
    event.event_type,
    event.agent,
    stableStringify(event.data || {}),
  ].join("|");
}

function mergeEvents(
  existing: AgentEvent[],
  incoming: AgentEvent[],
  seen: Set<string>
): AgentEvent[] {
  const merged = [...existing];
  for (const event of incoming) {
    const key = getEventKey(event);
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    merged.push(event);
  }
  return merged.slice(0, 2000);
}

function stableStringify(value: any): string {
  if (value === null || typeof value !== "object") {
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map((v) => stableStringify(v)).join(",")}]`;
  }
  const keys = Object.keys(value).sort();
  const entries = keys.map((k) => `"${k}":${stableStringify(value[k])}`);
  return `{${entries.join(",")}}`;
}

function getEventPhase(event: AgentEvent): Phase {
  switch (event.event_type) {
    case "thinking":
      return "reasoning";
    case "action":
      return "action";
    case "finding":
      return "finding";
    case "synthesis":
      return "synthesis";
    case "error":
      return "error";
    case "system": {
      const message = extractSystemMessage(event).toLowerCase();
      return getPhaseFromText(message);
    }
    default:
      return "system";
  }
}

function getPhaseFromText(message: string): Phase {
  if (message.includes("search") || message.includes("query")) return "search";
  if (message.includes("[observe]") || message.includes("observe")) return "observe";
  if (message.includes("kg") || message.includes("knowledge graph")) return "knowledge";
  if (message.includes("retrieval")) return "knowledge";
  if (message.includes("synth")) return "synthesis";
  return "system";
}

function extractSystemMessage(event: AgentEvent): string {
  const data = event.data || {};
  if (typeof data.message === "string") return data.message;
  if (typeof data.content === "string") return data.content;
  if (typeof data.text === "string") return data.text;
  return "";
}

function shouldGroupLog(block: LogBlock, event: AgentEvent): boolean {
  if (block.agent !== event.agent) return false;
  const blockEnd = new Date(block.end).getTime();
  const eventTime = new Date(event.timestamp).getTime();
  return Math.abs(eventTime - blockEnd) <= 2500;
}

function createLogBlock(event: AgentEvent, index: number): LogBlock {
  const timestamp = event.timestamp;
  return {
    id: `log-${event.agent}-${timestamp}-${index}`,
    agent: event.agent,
    start: timestamp,
    end: timestamp,
    lines: [],
  };
}

function addLogLine(block: LogBlock, message: string): void {
  const text = message.trim();
  if (!text) return;
  const tone = getLineTone(text);
  const last = block.lines[block.lines.length - 1];
  if (last && last.text === text) {
    last.count += 1;
    return;
  }
  block.lines.push({ text, count: 1, tone });
}

function getLineTone(text: string): LogTone {
  if (text.startsWith("\u2500")) return "divider";
  if (text.startsWith("URL:")) return "url";
  if (text.startsWith("Confidence:")) return "meta";
  if (text.match(/^\[\w+/)) return "label";
  if (text.match(/^\d+\./)) return "list";
  return "text";
}

function getPhaseForLogBlock(block: LogBlock): Phase {
  for (const line of block.lines) {
    const phase = getPhaseFromText(line.text.toLowerCase());
    if (phase !== "system") {
      return phase;
    }
  }
  return "system";
}

function LogLineRow({ line }: { line: LogLine }) {
  if (line.tone === "divider") {
    return <div className="h-px bg-obs-border my-2" />;
  }

  const base =
    line.tone === "label"
      ? "text-amber"
      : line.tone === "url"
      ? "text-amber"
      : line.tone === "meta"
      ? "text-text-secondary"
      : line.tone === "list"
      ? "text-text-secondary"
      : "text-text/80";

  return (
    <div className={`flex items-start gap-2 ${base}`}>
      <span className="flex-1 break-words">{line.text}</span>
      {line.count > 1 ? (
        <span className="obs-badge badge-system">x{line.count}</span>
      ) : null}
    </div>
  );
}

function getEventPrimaryText(event: AgentEvent): string {
  const data = event.data || {};
  if (typeof data.thought === "string") return data.thought;
  if (typeof data.message === "string") return data.message;
  if (typeof data.content === "string") return data.content;
  if (typeof data.error === "string") return data.error;
  if (typeof data.action === "string") return data.action;
  return JSON.stringify(data);
}

function renderExpandableText(
  text: string,
  expanded: boolean,
  onToggle: () => void,
  extraClass: string = ""
): React.ReactNode {
  const limit = 220;
  const shouldTruncate = text.length > limit;
  if (!shouldTruncate) {
    return <p className={extraClass}>{text}</p>;
  }
  return (
    <div className="space-y-2">
      <p className={extraClass}>
        {expanded ? text : `${text.slice(0, limit)}...`}
      </p>
      <button type="button" className="obs-chip" onClick={onToggle}>
        {expanded ? "Collapse" : "Expand"}
      </button>
    </div>
  );
}

function truncateUrl(url: string, maxLength: number = 48): string {
  if (url.length <= maxLength) {
    return url;
  }
  const start = url.slice(0, Math.max(0, maxLength - 12));
  const end = url.slice(-10);
  return `${start}\u2026${end}`;
}
