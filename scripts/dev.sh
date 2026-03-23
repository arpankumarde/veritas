#!/bin/bash
# Veritas - Full setup + launch script
# Usage: bash scripts/dev.sh
set -e

cd "$(dirname "$0")/.."
ROOT=$(pwd)

echo "========================================"
echo "  Veritas - AI Fact Checker"
echo "========================================"

# ── Cleanup ──────────────────────────────────
echo ""
echo "[1/6] Cleaning up..."

# Kill any existing servers on our ports
for port in 9090 3000; do
    pids=$(lsof -ti:$port 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "  Killing processes on port $port..."
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 1
    fi
done

# Remove stale venv if broken
if [ -d .venv ] && ! .venv/bin/python3 -c "import sys" 2>/dev/null; then
    echo "  Removing broken venv..."
    rm -rf .venv
fi

# ── Python venv ──────────────────────────────
echo ""
echo "[2/6] Setting up Python environment..."

if [ ! -d .venv ]; then
    python3 -m venv .venv
    echo "  Created .venv"
fi
source .venv/bin/activate

# ── Install Python deps ──────────────────────
echo ""
echo "[3/6] Installing Python dependencies..."

pip install -q --upgrade pip
pip install -e "." 2>&1 | tail -3
echo "  Engine installed"

# Install spacy model if missing
python3 -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null || {
    echo "  Downloading spaCy model..."
    python3 -m spacy download en_core_web_sm -q
}

# ── Node deps ────────────────────────────────
echo ""
echo "[4/6] Installing frontend dependencies..."

if [ ! -d node_modules ] || [ ! -d node_modules/.pnpm ]; then
    pnpm install --frozen-lockfile 2>/dev/null || pnpm install
else
    echo "  node_modules up to date"
fi

# ── Load env ─────────────────────────────────
echo ""
echo "[5/6] Loading environment..."

if [ -f .env ]; then
    set -a
    source .env
    set +a
    echo "  .env loaded"
else
    echo "  WARNING: No .env file found. Copy .env.example to .env"
fi

if [ -z "$BRIGHT_DATA_API_TOKEN" ]; then
    echo "  WARNING: BRIGHT_DATA_API_TOKEN not set. Web search will fail."
fi

# ── Launch servers ───────────────────────────
echo ""
echo "[6/6] Starting servers..."

# Start API server in background
echo "  Starting API server on :9090..."
.venv/bin/python3 -m api.server &
API_PID=$!

# Wait for API to be ready
for i in $(seq 1 15); do
    if curl -s http://localhost:9090/ > /dev/null 2>&1; then
        echo "  API server ready"
        break
    fi
    sleep 1
done

# Start Next.js frontend in background
echo "  Starting frontend on :3004..."
pnpm dev &
UI_PID=$!

# Wait for frontend
for i in $(seq 1 20); do
    if curl -s http://localhost:3004/ > /dev/null 2>&1; then
        echo "  Frontend ready"
        break
    fi
    sleep 1
done

echo ""
echo "========================================"
echo "  Veritas is running!"
echo ""
echo "  Frontend:  http://localhost:3004"
echo "  API:       http://localhost:9090"
echo "  API Docs:  http://localhost:9090/docs"
echo "========================================"
echo ""
echo "Press Ctrl+C to stop all servers"

# Trap Ctrl+C to kill both servers
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $API_PID $UI_PID 2>/dev/null
    wait $API_PID $UI_PID 2>/dev/null
    echo "Done."
    exit 0
}
trap cleanup SIGINT SIGTERM

# Wait for either process to exit
wait -n $API_PID $UI_PID 2>/dev/null
cleanup
