#!/usr/bin/env bash
set -euo pipefail

RESOURCE_GROUP="${RESOURCE_GROUP:-rg-dualscore-edgeproto}"

echo "=== Azure Resource Group Deletion Helper ==="
echo "WARNING: This will permanently delete the Resource Group '${RESOURCE_GROUP}' and ALL resources contained within it!"
echo ""

read -p "Are you absolutely sure you want to delete '${RESOURCE_GROUP}'? Type 'DELETE' to confirm: " confirm
if [ "${confirm}" == "DELETE" ]; then
    echo "Deleting Resource Group ${RESOURCE_GROUP}..."
    az group delete --name "${RESOURCE_GROUP}" --yes --no-wait
    echo "[SUCCESS] Resource Group deletion requested."
else
    echo "[ABORTED] Deletion cancelled by user."
fi
