#!/bin/bash
set -e

echo "=== EdgeProto-OSR End-to-End Pipeline on Synthetic Data ==="

echo "1. Generating Synthetic Data..."
python3 src/data/synthetic_generator.py --samples_per_class 1000 --output_dir data/raw

echo "2. Preprocessing Data..."
python3 src/data/preprocess.py --input data/raw/synthetic_ciciot2023.csv --output data/processed/synthetic_ciciot2023_log1p.csv

echo "3. Creating Splits for H1 (Mirai held out)..."
python3 src/data/make_splits.py --input data/processed/synthetic_ciciot2023_log1p.csv --config configs/h1_mirai.yaml --output_dir data/splits

echo "4. Training EdgeProto on H1..."
python3 src/training/train_edgeproto.py --config configs/h1_mirai.yaml --data_prefix data/splits/h1_mirai --output_dir models/h1_mirai

echo "5. Evaluating EdgeProto on H1..."
python3 src/evaluation/eval_open_set.py --config configs/h1_mirai.yaml --data_prefix data/splits/h1_mirai --model_path models/h1_mirai/best_edgeproto.pt --output_dir results/tables

echo "6. Quantizing EdgeProto (PTQ)..."
python3 src/deployment/quantize_ptq.py --config configs/h1_mirai.yaml --data_prefix data/splits/h1_mirai --model_path models/h1_mirai/best_edgeproto.pt --output_dir models/quantized

echo "7. Counting MACs..."
python3 src/deployment/count_macs.py --config configs/default_cpu_config.yaml

echo "=== Pipeline Complete ==="
