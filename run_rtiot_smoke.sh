#!/bin/bash
set -e

# Activate venv if not already activated
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    elif [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    fi
fi

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

mkdir -p logs
mkdir -p data/processed
mkdir -p data/splits
mkdir -p models/rt_h1_arp_spoofing
mkdir -p results/tables

CONFIG_FILE="configs/rt_h1_arp_spoofing.yaml"
PREFIX="rt_h1_arp_spoofing"
if [ -n "$EPOCHS_OVERRIDE" ]; then
    echo "Using EPOCHS_OVERRIDE=$EPOCHS_OVERRIDE"
    TEMP_CONFIG="configs/rt_h1_arp_spoofing_override.yaml"
    sed "s/epochs: 30/epochs: $EPOCHS_OVERRIDE/g" "$CONFIG_FILE" > "$TEMP_CONFIG"
    CONFIG_FILE="$TEMP_CONFIG"
    PREFIX="rt_h1_arp_spoofing_override"
fi

echo "1. Running prepare_rtiot2022.py..."
python src/data/prepare_rtiot2022.py \
    --input data/raw/RT-IoT2022/RT_IOT2022.csv \
    --output data/processed/rtiot2022_grouped_48.csv \
    --feature_mode conservative_no_ports \
    --exclude_tiny_bruteforce 1

echo "2. Running make_splits.py..."
python src/data/make_splits.py \
    --input data/processed/rtiot2022_grouped_48.csv \
    --config "$CONFIG_FILE" \
    --output_dir data/splits

echo "3. Running train_edgeproto.py..."
python src/training/train_edgeproto.py \
    --config "$CONFIG_FILE" \
    --data_prefix data/splits/$PREFIX \
    --output_dir models/rt_h1_arp_spoofing

echo "4. Running eval_open_set.py..."
python src/evaluation/eval_open_set.py \
    --config "$CONFIG_FILE" \
    --data_prefix data/splits/$PREFIX \
    --model_path models/rt_h1_arp_spoofing/best_edgeproto.pt \
    --output_dir results/tables

echo "5. Running count_macs.py..."
python src/deployment/count_macs.py \
    --config "$CONFIG_FILE"

if [ "$RUN_QUANT" = "1" ]; then
    echo "6. Running quantize_ptq.py..."
    mkdir -p models/quantized
    python src/deployment/quantize_ptq.py \
        --config "$CONFIG_FILE" \
        --data_prefix data/splits/$PREFIX \
        --model_path models/rt_h1_arp_spoofing/best_edgeproto.pt \
        --output_dir models/quantized
fi

if [ -n "$EPOCHS_OVERRIDE" ]; then
    rm -f configs/rt_h1_arp_spoofing_override.yaml
fi

echo "Smoke test completed successfully."
