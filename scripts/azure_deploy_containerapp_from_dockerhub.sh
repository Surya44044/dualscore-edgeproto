#!/usr/bin/env bash
set -euo pipefail

APP_NAME="${APP_NAME:-app-dualscore-edgeproto}"
ENV_NAME="${ENV_NAME:-env-dualscore-edgeproto}"
RESOURCE_GROUP="${RESOURCE_GROUP:-rg-dualscore-edgeproto}"
DOCKERHUB_IMAGE="${DOCKERHUB_IMAGE:-docker.io/YOUR_DOCKERHUB_USERNAME/dualscore-edgeproto:latest}"
PORT="${PORT:-8000}"

echo "=== Deploy Azure Container App from DockerHub ==="
echo "App Name       : ${APP_NAME}"
echo "Environment    : ${ENV_NAME}"
echo "Resource Group : ${RESOURCE_GROUP}"
echo "Docker Image   : ${DOCKERHUB_IMAGE}"
echo "Target Port    : ${PORT}"
echo "Command to run:"
echo "  az containerapp create --name \"${APP_NAME}\" --resource-group \"${RESOURCE_GROUP}\" --environment \"${ENV_NAME}\" --image \"${DOCKERHUB_IMAGE}\" --target-port ${PORT} --ingress external --cpu 0.25 --memory 0.5Gi"
echo ""

read -p "Do you want to execute this command now? (y/N): " confirm
if [[ "${confirm}" =~ ^[Yy]$ ]]; then
    az containerapp create \
      --name "${APP_NAME}" \
      --resource-group "${RESOURCE_GROUP}" \
      --environment "${ENV_NAME}" \
      --image "${DOCKERHUB_IMAGE}" \
      --target-port "${PORT}" \
      --ingress external \
      --cpu 0.25 \
      --memory 0.5Gi
    echo "[SUCCESS] Container App deployed."
else
    echo "[INFO] Command execution cancelled."
fi
