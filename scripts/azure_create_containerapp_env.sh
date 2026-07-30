#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-env-dualscore-edgeproto}"
RESOURCE_GROUP="${RESOURCE_GROUP:-rg-dualscore-edgeproto}"
LOCATION="${LOCATION:-eastus}"

echo "=== Azure Container Apps Environment Creator ==="
echo "Environment Name: ${ENV_NAME}"
echo "Resource Group  : ${RESOURCE_GROUP}"
echo "Location        : ${LOCATION}"
echo "Command to run:"
echo "  az containerapp env create --name \"${ENV_NAME}\" --resource-group \"${RESOURCE_GROUP}\" --location \"${LOCATION}\""
echo ""

read -p "Do you want to execute this command now? (y/N): " confirm
if [[ "${confirm}" =~ ^[Yy]$ ]]; then
    az containerapp env create --name "${ENV_NAME}" --resource-group "${RESOURCE_GROUP}" --location "${LOCATION}"
    echo "[SUCCESS] Container Apps Environment created."
else
    echo "[INFO] Command execution cancelled."
fi
