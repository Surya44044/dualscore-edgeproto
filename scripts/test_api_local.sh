#!/usr/bin/env bash
set -euo pipefail

echo "=== Testing FastAPI Endpoints Locally ==="

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python - << 'PY'
import sys
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

print("1. Testing GET /health...")
res = client.get("/health")
print("Response:", res.status_code, res.json())
assert res.status_code == 200, "Health check failed!"

print("\n2. Testing GET /metadata...")
res = client.get("/metadata")
print("Response:", res.status_code, res.json())
assert res.status_code == 200, "Metadata check failed!"

print("\n3. Testing POST /predict...")
payload = {
    "features": [0.05] * 48
}
res = client.post("/predict", json=payload)
print("Response:", res.status_code, res.json())
assert res.status_code == 200, "Single prediction failed!"

print("\n4. Testing POST /predict_batch...")
batch_payload = {
    "samples": [
        {"features": [0.05] * 48},
        {"features": [1.20] * 48}
    ]
}
res = client.post("/predict_batch", json=batch_payload)
print("Response:", res.status_code, res.json())
assert res.status_code == 200, "Batch prediction failed!"

print("\n5. Testing 48-feature validation error handling...")
bad_payload = {"features": [0.1] * 10}
res = client.post("/predict", json=bad_payload)
print("Response (Expected 422/400):", res.status_code)
assert res.status_code in (400, 422), "Validation check failed!"

print("\n[SUCCESS] All FastAPI local API tests passed successfully!")
PY
