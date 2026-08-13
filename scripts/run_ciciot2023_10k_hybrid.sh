#!/bin/bash
set -e
cd ~/dualscore-edgeproto
export PYTHONPATH=.
DATA=data/processed/ciciot2023/ciciot2023_sample_10k_processed.csv
declare -a CONFIGS=(h1_mirai h2_recon h3_spoofing h4_web_bruteforce h5_dos)
SUMMARY=results/tables/ciciot2023_10k/hybrid_10k_summary.csv
mkdir -p results/tables/ciciot2023_10k
echo "experiment,auroc,oscr" > "$SUMMARY"
for cfg in "${CONFIGS[@]}"; do
  echo "=============================================="
  echo "RUNNING $cfg (hybrid)"
  echo "=============================================="
  python src/evaluation/evaluate_edgeproto_hybrid.py \
    --prefix "data/splits_10k/$cfg" \
    --config "configs/$cfg.yaml" \
    --edgeproto_model "models/ciciot10k_${cfg}_base/best_edgeproto.pt" \
    --unknown "UNKNOWN" \
    --output results/tables/ciciot2023_10k/hybrid_${cfg}.json \
    2>&1 | tee -a ~/ciciot_10k_hybrid.log || echo "hybrid eval failed for $cfg (continuing)"
done
echo "=============================================="
echo "HYBRID PASS DONE"
