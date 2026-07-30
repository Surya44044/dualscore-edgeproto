#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${1:-dualscore-edgeproto:local}"
PORT="${2:-8000}"

echo "=== Running Local Docker Container (${IMAGE_NAME} on port ${PORT}) ==="

if ! command -v docker &> /dev/null; then
    echo "[WARN] Docker CLI is not installed or available on this system."
    echo "[INFO] Skipping local Docker run."
    exit 0
fi

exec docker run --rm -p "${PORT}:${PORT}" -e PORT="${PORT}" "${IMAGE_NAME}"
