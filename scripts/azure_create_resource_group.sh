#!/usr/bin/env bash
set -euo pipefail

# Configuration Placeholders (Override via environment variables)
RESOURCE_GROUP="${RESOURCE_GROUP:-rg-dualscore-edgeproto}"
LOCATION="${LOCATION:-eastus}"

echo "=== Azure Resource Group Creator ==="
echo "Resource Group: ${RESOURCE_GROUP}"
echo "Location      : ${LOCATION}"
echo "Command to run:"
echo "  az group create --name \"${RESOURCE_GROUP}\" --location \"${LOCATION}\""
echo ""

read -p "Do you want to execute this command now? (y/N): " confirm
if [[ "${confirm}" =~ ^[Yy]$ ]]; then
    az group create --name "${RESOURCE_GROUP}" --location "${LOCATION}"
    echo "[SUCCESS] Resource Group created."
else
    echo "[INFO] Command execution cancelled."
fi
