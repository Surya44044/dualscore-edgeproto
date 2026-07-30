#!/usr/bin/env bash
set -euo pipefail

echo "=== Running Non-Destructive Secret Scanner ==="

FOUND_SECRETS=0

# Patterns to scan for
PATTERNS=(
    "BEGIN (RSA|OPENSSH|EC|PGP) PRIVATE KEY"
    "azure_client_secret[[:space:]]*="
    "dockerhub_token[[:space:]]*="
    "client_secret[[:space:]]*:[[:space:]]*\"[^\"]+\""
    "\"subscriptionId\"[[:space:]]*:[[:space:]]*\"[a-f0-9-]{20,}\""
)

# Files/directories to exclude
EXCLUDES=(
    "--exclude-dir=.git"
    "--exclude-dir=.venv"
    "--exclude-dir=__pycache__"
    "--exclude-dir=.pytest_cache"
    "--exclude-dir=logs/antigravity_backups"
    "--exclude=check_no_secrets.sh"
    "--exclude=azure-credentials*.json"
    "--exclude=service-principal*.json"
)

for pattern in "${PATTERNS[@]}"; do
    if grep -rEI "${EXCLUDES[@]}" "${pattern}" . 2>/dev/null; then
        echo "[WARNING] Secret pattern match found: '${pattern}'"
        FOUND_SECRETS=1
    fi
done

# Check if credential JSON files are tracked in git
if git ls-files | grep -E "service-principal|azure-credentials|\.env" > /dev/null 2>&1; then
    echo "[ERROR] Tracked credential file detected in Git status!"
    FOUND_SECRETS=1
fi

if [ "${FOUND_SECRETS}" -eq 0 ]; then
    echo "[SUCCESS] Secret scan passed. No exposed credentials detected!"
else
    echo "[WARN] Potential secrets flagged above. Please inspect before pushing."
fi
