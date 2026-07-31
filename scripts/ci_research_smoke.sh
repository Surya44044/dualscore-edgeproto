#!/usr/bin/env bash
set -euo pipefail

echo "=== CI Research Smoke Test: DualScore-EdgeProto ==="

echo ""
echo "1. Checking Python imports..."
python - << 'PY'
import sys
import numpy as np
import torch

from src.models.edgeproto import get_model
from src.api.app import app

print("Python:", sys.version)
print("NumPy:", np.__version__)
print("Torch:", torch.__version__)
print("FastAPI title:", app.title)
PY

echo ""
echo "2. Checking EdgeProto lightweight model creation and parameter count..."
python - << 'PY'
import torch
import yaml
from pathlib import Path
from src.models.edgeproto import get_model

config_path = Path("configs/rt_h1_arp_spoofing.yaml")
if not config_path.exists():
    config_path = Path("configs/default_cpu_config.yaml")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# RT-H1 known classes: Benign, DDoS, DoS, Recon
num_classes = 4

model = get_model(config, num_classes=num_classes)
params = sum(p.numel() for p in model.parameters())

print("EdgeProto params:", params)

# Our current EdgeProto is about 3.8k params.
# Keep loose upper bound so future minor changes do not break CI.
assert params < 30000, f"EdgeProto too large for lightweight claim: {params} params"

x = torch.randn(2, 4, 12)
with torch.no_grad():
    z = model.encoder(x)

print("Embedding shape:", tuple(z.shape))
assert tuple(z.shape) == (2, 32), f"Expected embedding shape (2,32), got {tuple(z.shape)}"

print("EdgeProto forward/embedding smoke test passed.")
PY

echo ""
echo "3. Checking API 48-feature validation..."
python - << 'PY'
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

r = client.get("/health")
print("/health:", r.status_code, r.json())
assert r.status_code == 200

r = client.get("/metadata")
print("/metadata:", r.status_code, r.json())
assert r.status_code == 200
assert r.json()["expected_features"] == 48

payload = {"features": [0.0] * 48}
r = client.post("/predict", json=payload)
print("/predict status:", r.status_code)
assert r.status_code == 200

bad_payload = {"features": [0.0] * 47}
r = client.post("/predict", json=bad_payload)
print("/predict bad status:", r.status_code)
assert r.status_code in (400, 422)

print("API validation smoke test passed.")
PY

echo ""
echo "4. Checking Dockerfile safety..."
if grep -qE "COPY data/|COPY models/|ADD data/|ADD models/" Dockerfile; then
  echo "ERROR: Dockerfile must not copy data/ or models/ into image."
  exit 1
fi

if grep -qE "\\[.*\\]\\(http|src.api.app\\]\\(" Dockerfile; then
  echo "ERROR: Dockerfile appears to contain markdown-corrupted URLs."
  exit 1
fi

echo "Dockerfile safety check passed."

echo ""
echo "5. Checking repository does not expose dataset folders in git tracking..."
if git ls-files | grep -qE "^(data/raw|data/processed|data/splits|models|results)/"; then
  echo "ERROR: data/models/results artifacts should not be tracked in Git."
  git ls-files | grep -E "^(data/raw|data/processed|data/splits|models|results)/"
  exit 1
fi

echo "Git artifact exclusion check passed."

echo ""
echo "=== CI Research Smoke Test PASSED ==="
