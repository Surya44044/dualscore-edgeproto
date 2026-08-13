#!/bin/bash
set -e
cd ~/dualscore-edgeproto
export PYTHONPATH=.
DATA=data/processed/ciciot2023/ciciot2023_smoke_2k_processed.csv
declare -a CONFIGS=(h2_recon h3_spoofing h4_web_bruteforce h5_dos)
SUMMARY=results/tables/ciciot2023_smoke/2k_all_summary.csv
echo "experiment,closed_acc,auroc,oscr" > "$SUMMARY"
for cfg in "${CONFIGS[@]}"; do
  echo "=============================================="
  echo "RUNNING $cfg"
  echo "=============================================="
  python src/data/make_splits.py --input "$DATA" --config "configs/$cfg.yaml" --output_dir data/splits
  python src/training/train_edgeproto.py --config "configs/$cfg.yaml" --data_prefix "data/splits/$cfg" --output_dir "models/ciciot_${cfg}_smoke" --seed 42
  python src/evaluation/eval_open_set.py --config "configs/$cfg.yaml" --data_prefix "data/splits/$cfg" --model_path "models/ciciot_${cfg}_smoke/best_edgeproto.pt" --output_dir results/tables/ciciot2023_smoke
  python scripts/append_summary.py "$cfg" "$SUMMARY"
done
echo "=============================================="
echo "ALL DONE. Summary:"
cat "$SUMMARY"
