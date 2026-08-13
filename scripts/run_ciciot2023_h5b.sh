#!/bin/bash
set -e
cd ~/dualscore-edgeproto
export PYTHONPATH=.
CFG=h5b_dos_ddos
DATA=data/processed/ciciot2023/ciciot2023_sample_10k_processed.csv
python src/data/make_splits.py --input "$DATA" --config configs/$CFG.yaml --output_dir data/splits
python src/training/train_edgeproto.py --config configs/$CFG.yaml --data_prefix data/splits/$CFG --output_dir models/ciciot10k_${CFG}_base --seed 42
python src/evaluation/eval_open_set.py --config configs/$CFG.yaml --data_prefix data/splits/$CFG --model_path models/ciciot10k_${CFG}_base/best_edgeproto.pt --output_dir results/tables/ciciot2023_10k
python src/evaluation/evaluate_edgeproto_hybrid.py --prefix $CFG --config configs/$CFG.yaml --edgeproto_model models/ciciot10k_${CFG}_base/best_edgeproto.pt
echo "H5B DONE"
