#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${1:-dualscore-edgeproto-train:local}"

echo "=== Building Training Docker Image (${IMAGE_NAME}) ==="

if ! command -v docker &> /dev/null; then
    echo "[WARN] Docker CLI is not installed or available on this system."
    echo "[INFO] Skipping training Docker build."
    exit 0
fi

docker build -t "${IMAGE_NAME}" -f Dockerfile.train .
echo "[SUCCESS] Built Training Docker image: ${IMAGE_NAME}"
