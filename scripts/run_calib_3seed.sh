#!/bin/bash
# 3-seed calibration re-run: train per-seed models, eval hybrid, compute w=0.25 fusion
cd ~/dualscore-edgeproto
export PYTHONPATH=.
LOG=~/ciciot_calib.log
DATA=data/processed/ciciot2023/ciciot2023_sample_50k_processed.csv
SEEDS="42 123 2025"
EXPS="h1_mirai h2_recon h3_spoofing h4_web_bruteforce h5_dos h5b_dos_ddos"
mkdir -p models/calib results/tables/ciciot2023_50k/calib
echo "=== CALIB RUN START $(date) ===" >> "$LOG"
for S in $SEEDS; do
  for cfg in $EXPS; do
    tag="${cfg}_s${S}"
    OUTDIR="models/calib/${tag}"
    if [ -f "$OUTDIR/best_edgeproto.pt" ] && [ -f "results/tables/ciciot2023_50k/calib/${tag}_hybrid.json" ]; then
      echo "SKIP $tag (done)" >> "$LOG"; continue
    fi
    echo "===== CALIB $tag $(date) =====" >> "$LOG"
    python src/data/make_splits.py --input "$DATA" --config "configs/$cfg.yaml" --output_dir data/splits >> "$LOG" 2>&1 || { echo "FAIL $tag splits" >> "$LOG"; continue; }
    python scripts/train_edgeproto_calib.py --config "configs/$cfg.yaml" --data_prefix "data/splits/$cfg" --output_dir "$OUTDIR" --seed "$S" >> "$LOG" 2>&1 || { echo "FAIL $tag train" >> "$LOG"; continue; }
    rm -f "models/hybrid_autoencoders/${cfg}_ae_known.pt"
    python src/evaluation/evaluate_edgeproto_hybrid.py --prefix "$cfg" --config "configs/$cfg.yaml" --edgeproto_model "$OUTDIR/best_edgeproto.pt" >> "$LOG" 2>&1 || { echo "FAIL $tag hybrid" >> "$LOG"; continue; }
    cp "results/tables/hybrid/${cfg}_Hybrid-Mean-50-50_results.json" "results/tables/ciciot2023_50k/calib/${tag}_hybrid.json"
    echo "OK $tag $(date)" >> "$LOG"
  done
  echo "===== SEED $S COMPLETE =====" >> "$LOG"
done
echo "=== CALIB RUN END $(date) ===" >> "$LOG"
