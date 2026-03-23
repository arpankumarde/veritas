# Veritas - AI Fact Checking System

## Architecture
Hierarchical multi-agent system for automated claim verification:
- **Director** → User interface, session management
- **Manager** → Claim decomposition, verification strategy, verdict determination
- **Parallel Intern Pool** → Evidence gathering (web + academic search)

## Key Concepts
- **Claim**: Statement to verify (input)
- **Sub-claim**: Decomposed verifiable part of a claim
- **Evidence**: Gathered information (supporting, contradicting, contextual)
- **Verdict**: True / Mostly True / Mixed / Mostly False / False / Unverifiable

## Project Structure
- `engine/` - Python fact-checking engine (agents, verification, knowledge graph)
- `api/` - FastAPI backend with WebSocket support
- `src/` - Next.js frontend (shadcn UI components)
- `prisma/` - Database schema

## Development
```bash
# Install Python dependencies
pip install -e .
python -m spacy download en_core_web_sm

# Run CLI fact check
veritas "claim to verify" --iterations 5

# Launch web UI
veritas ui

# Frontend dev
pnpm dev
```

## Tech Stack
- Python: claude-agent-sdk, FastAPI, SQLite, NetworkX, ChromaDB
- Frontend: Next.js 16, React 19, Tailwind CSS 4, shadcn/ui
- Search: Bright Data SERP + Web Unlocker, Semantic Scholar, arXiv

## Conventions
- Python engine: `engine/` package, import as `engine.*`
- Env vars: `VERITAS_` prefix
- Logs: `~/.veritas/veritas.log`
- CLI: `veritas` command
