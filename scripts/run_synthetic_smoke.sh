#!/usr/bin/env bash
set -euo pipefail

echo "=== Running Synthetic Model Smoke Test ==="

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python - << 'PY'
import torch
import numpy as np
from src.models.edgeproto import get_model
from src.evaluation.evaluate_edgeproto_hybrid import SmallAutoencoder
from src.api.preprocess import preprocess_features, preprocess_batch_features

print("1. Instantiating EdgeProto Base (3,836 params)...")
config = {
    'model': {
        'num_blocks': 2,
        'in_channels': 4,
        'embedding_dim': 32,
        'hidden_dim': 16
    }
}
model = get_model(config, num_classes=4)
print("EdgeProto parameter count:", sum(p.numel() for p in model.parameters()))

print("\n2. Instantiating Autoencoder (11,520 params)...")
ae = SmallAutoencoder(input_dim=48)
print("Autoencoder parameter count:", sum(p.numel() for p in ae.parameters()))

print("\n3. Testing synthetic 48-feature tensor forward pass...")
x_raw = [0.05] * 48
x_tensor = preprocess_features(x_raw) # [1, 4, 12]
assert x_tensor.shape == (1, 4, 12), f"Unexpected shape {x_tensor.shape}"

emb = model.encoder(x_tensor)
assert emb.shape == (1, 32), f"Unexpected embedding shape {emb.shape}"

x_flat = x_tensor.view(1, -1)
recon = ae(x_flat)
assert recon.shape == (1, 48), f"Unexpected reconstruction shape {recon.shape}"

print("\n4. Testing batch preprocessing...")
batch_raw = [[0.05] * 48, [1.2] * 48]
batch_tensor = preprocess_batch_features(batch_raw)
assert batch_tensor.shape == (2, 4, 12), f"Unexpected batch shape {batch_tensor.shape}"

print("\n[SUCCESS] Synthetic smoke test completed cleanly with ZERO errors!")
PY
