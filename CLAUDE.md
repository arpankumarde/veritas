# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture

Hierarchical multi-agent fact-checking system powered by Claude. Three agent tiers operate in a ReAct (Think-Act-Observe) loop:

- **Director** (`engine/agents/director.py`) — User-facing layer. Manages sessions, handles clarification, reports verdicts. Entry point is `VeritasHarness` (async context manager).
- **Manager** (`engine/agents/manager.py`) — Orchestration layer. Decomposes claims into sub-claims, creates `VerificationDirective` objects for interns, critically evaluates evidence, synthesizes final verdicts using Opus with extended thinking.
- **Parallel Intern Pool** (`engine/agents/parallel.py`, `engine/agents/intern.py`) — Evidence gathering. Multiple interns run concurrently via `asyncio.gather`, each executing web/academic searches and reporting `Evidence` objects back to the Manager.

### Data Flow

```
User claim → Director.start_verification()
  → Manager.run_verification() [ReAct loop, N iterations]
    → Each iteration: Manager creates VerificationDirective list
    → ParallelInternPool.gather_evidence_parallel(directives)
      → N InternAgents search in parallel (web + academic)
      → Evidence reported back via WebSocket events
    → Manager evaluates evidence, identifies gaps, queues next topics
  → Synthesis phase: KG construction → Opus verdict → VerdictReport
  → Output saved to output/{slug}_{session_id}/
```

### Model Routing

`ModelRouter` in `engine/agents/base.py` routes LLM calls by task type:

| Task | Model | Examples |
|------|-------|---------|
| Haiku | Quick classification | `classify`, `extract_simple`, `yes_no` |
| Sonnet | Search & extraction | `search`, `extract_evidence`, `summarize` |
| Opus | Deep reasoning | `strategic_planning`, `synthesis`, `verdict_writing` |

Extended thinking is enabled for Opus tasks.

### Key Subsystems

- **Knowledge Graph** (`engine/knowledge/`) — Real-time entity/relation extraction from evidence using NER + LLM. Stored in SQLite+NetworkX (`HybridKnowledgeGraphStore`). Enables contradiction detection and evidence corroboration scoring.
- **Hybrid Retrieval** (`engine/retrieval/`) — BGE embeddings + ChromaDB (semantic) combined with BM25 (lexical) via Reciprocal Rank Fusion, plus cross-encoder reranking.
- **Verification Pipeline** (`engine/verification/`) — Optional CoVe and CRITIC verification methods. Currently disabled in Manager for speed; relies on evidence quality + Opus synthesis.
- **Event System** (`engine/events/`) — `emit_thinking`, `emit_action`, `emit_finding`, `emit_synthesis` push real-time updates to the frontend via WebSocket.

## Development

```bash
# Python setup
pip install -e .
python -m spacy download en_core_web_sm

# Run CLI fact check
veritas "claim to verify" --iterations 5
veritas "claim" -n 3 --autonomous --no-clarify

# Launch API server (FastAPI on port 9090)
veritas ui
veritas ui --port 9090

# Frontend dev (Next.js on port 3004)
pnpm install
pnpm dev

# Lint
pnpm lint          # ESLint (Next.js core-web-vitals + TypeScript)
ruff check engine/  # Python linting
ruff format engine/ # Python formatting
```

### Environment Variables

Copy `.env.example` to `.env`. Required: `BRIGHT_DATA_API_TOKEN`. Optional: `ANTHROPIC_API_KEY` (falls back to CLI auth), `VERITAS_API_KEY` (API auth), `DATABASE_URL` (PostgreSQL for Prisma).

## Project Structure

- `engine/` — Python fact-checking engine (agents, tools, knowledge graph, retrieval, verification)
- `api/` — FastAPI backend (`api/server.py`). Routes in `api/routes/`. WebSocket at `/ws/events/{session_id}`
- `src/` — Next.js 16 frontend. App router in `src/app/`, components in `src/components/`
- `prisma/` — PostgreSQL schema (minimal, mostly scaffolding — primary data lives in SQLite via `engine/storage/database.py`)

### Frontend-Backend Communication

- **HTTP API** at `http://localhost:9090` — Sessions, evidence, reports, checks. Client in `src/lib/api.ts`
- **WebSocket** at `ws://localhost:9090/ws/{sessionId}` — Real-time agent events. Client class `ResearchWebSocket` in `src/lib/websocket.ts`
- Frontend pages: Landing (`/`), Dashboard (`/dashboard`), Check detail with tabs (`/check/[id]`), KG graph (`/check/[id]/graph`), Verification pipeline (`/check/[id]/verify`)

### Key Data Models

- **CheckSession** (`engine/models/evidence.py`) — 7-char hex ID, tracks status (active/running/paused/crashed/completed), iteration count, phase, elapsed time. Supports pause/resume.
- **Evidence** — Content, type (supporting/contradicting/contextual/source), source URL, confidence score, verification status, KG support score.
- **SubClaim** — Hierarchical decomposition with depth, priority, parent references.
- **VerificationDirective** — Manager→Intern instruction: action, topic, search queries, priority, depth.

## Conventions

- Python engine: `engine/` package, import as `engine.*`
- Env vars: `VERITAS_` prefix for app config
- Logs: `~/.veritas/veritas.log`
- CLI entry: `veritas` command (defined in `pyproject.toml` → `engine.main:app`, uses Typer)
- Python: Ruff with line-length 100, target py311, rules E/F/I/N/W/UP
- Frontend path alias: `@/*` → `./src/*`
- UI components: shadcn/ui (New York style) in `src/components/ui/`
- Build: hatchling (Python), pnpm (frontend)
- Async throughout: all agent code is async/await with asyncio
- State persistence: SQLite with WAL mode via `engine/storage/database.py` (aiosqlite). Full crash recovery via iteration checkpoints.
