#!/usr/bin/env bash
set -euo pipefail

echo "=== Starting DualScore-EdgeProto FastAPI Service ==="
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "Listening on http://${HOST}:${PORT}..."
exec uvicorn src.api.app:app --host "${HOST}" --port "${PORT}" --reload
