#!/bin/bash
set -e
cd ~/dualscore-edgeproto
export PYTHONPATH=.
DATA=data/processed/ciciot2023/ciciot2023_sample_10k_processed.csv
declare -a CONFIGS=(h1_mirai h2_recon h3_spoofing h4_web_bruteforce h5_dos)
SUMMARY=results/tables/ciciot2023_10k/base_10k_summary.csv
mkdir -p results/tables/ciciot2023_10k
echo "experiment,closed_acc,auroc,oscr" > "$SUMMARY"
for cfg in "${CONFIGS[@]}"; do
  echo "=============================================="
  echo "RUNNING $cfg (base)"
  echo "=============================================="
  python src/data/make_splits.py --input "$DATA" --config "configs/$cfg.yaml" --output_dir data/splits_10k
  python src/training/train_edgeproto.py --config "configs/$cfg.yaml" --data_prefix "data/splits_10k/$cfg" --output_dir "models/ciciot10k_${cfg}_base" --seed 42
  python src/evaluation/eval_open_set.py --config "configs/$cfg.yaml" --data_prefix "data/splits_10k/$cfg" --model_path "models/ciciot10k_${cfg}_base/best_edgeproto.pt" --output_dir results/tables/ciciot2023_10k
  python scripts/append_summary.py "$cfg" "$SUMMARY" || python -c "
import sys,json,glob
exp='$cfg'
j=json.load(open(glob.glob(f'results/tables/ciciot2023_10k/{exp}_results.json')[0]))
def s(o,n):
    out=None
    def w(o):
        nonlocal out
        if out is not None: return
        if isinstance(o,dict):
            for k,v in o.items():
                if out is not None: return
                if n in str(k).lower() and isinstance(v,(int,float)): out=v; return
                w(v)
        elif isinstance(o,list):
            for x in o: w(x)
    w(o); return out
acc=s(j,'acc'); au=s(j,'auroc'); os_=s(j,'oscr')
with open('$SUMMARY','a') as f: f.write(f'$cfg,{acc:.4f},{au:.4f},{os_:.4f}\n')
print(f'$cfg -> acc {acc:.4f} | AUROC {au:.4f} | OSCR {os_:.4f}')
"
done
echo "=============================================="
echo "ALL DONE. Summary:"
cat "$SUMMARY"
