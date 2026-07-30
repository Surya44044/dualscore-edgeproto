#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${1:-dualscore-edgeproto:local}"

echo "=== Building Local Docker Image (${IMAGE_NAME}) ==="

if ! command -v docker &> /dev/null; then
    echo "[WARN] Docker CLI is not installed or available on this system."
    echo "[INFO] Skipping local Docker build."
    exit 0
fi

docker build -t "${IMAGE_NAME}" -f Dockerfile .
echo "[SUCCESS] Built Docker image: ${IMAGE_NAME}"
