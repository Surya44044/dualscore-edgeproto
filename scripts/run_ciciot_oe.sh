#!/bin/bash
# CICIoT2023 OE (Outlier Exposure) 3-seed runner - crash-tolerant, resumable
cd ~/dualscore-edgeproto
export PYTHONPATH=.
LOG=~/ciciot_oe.log
ERRLOG=~/ciciot_oe_errors.log
DATA=data/processed/ciciot2023/ciciot2023_sample_50k_processed.csv
OSCFILE=results/tables/ciciot2023_50k/oe_oscr.csv
SEEDIR=results/tables/ciciot2023_50k/oe_seeds
SEEDS="42 123 2025"
EXPS="h1_mirai h2_recon h3_spoofing h4_web_bruteforce h5_dos h5b_dos_ddos"
mkdir -p "$SEEDIR"
[ -f "$OSCFILE" ] || echo "exp,seed,rule,oscr" > "$OSCFILE"

echo "=== OE RUN START $(date) ===" >> "$LOG"
if [ ! -f "$DATA" ]; then echo "FATAL: $DATA missing" >> "$LOG"; exit 1; fi

ok=0; fail=0
for S in $SEEDS; do
  for cfg in $EXPS; do
    tag="${cfg}_s${S}"
    if [ -f "$SEEDIR/${tag}_base.json" ] && [ -f "$SEEDIR/${tag}_hybrid.json" ] && grep -q "^${cfg},${S},Hybrid-Mean-50-50," "$OSCFILE"; then
      echo "SKIP $tag (done)" >> "$LOG"; continue
    fi
    echo "===== OE START $tag $(date) =====" >> "$LOG"
    sf=0
    python src/data/make_splits.py --input "$DATA" --config "configs/$cfg.yaml" --output_dir data/splits >> "$LOG" 2>&1 || { echo "FAIL $tag splits" >> "$ERRLOG"; sf=1; }
    if [ $sf -eq 0 ]; then
      for f in data/splits/${cfg}_train.csv data/splits/${cfg}_val.csv data/splits/${cfg}_test_known.csv data/splits/${cfg}_test_unknown.csv; do
        [ -s "$f" ] || { echo "FAIL $tag file $f" >> "$ERRLOG"; sf=1; }
      done
    fi
    if [ $sf -eq 0 ]; then
      python scripts/train_edgeproto_oe.py --config "configs/$cfg.yaml" --data_prefix "data/splits/$cfg" --output_dir "models/oe_${cfg}_base" --seed "$S" --n_oe 20000 >> "$LOG" 2>&1 \
        || { echo "FAIL $tag oe_train" >> "$ERRLOG"; sf=1; }
      [ -s "models/oe_${cfg}_base/best_edgeproto.pt" ] || { echo "FAIL $tag oe_model" >> "$ERRLOG"; sf=1; }
    fi
    if [ $sf -eq 0 ]; then
      python src/evaluation/eval_open_set.py --config "configs/$cfg.yaml" --data_prefix "data/splits/$cfg" --model_path "models/oe_${cfg}_base/best_edgeproto.pt" --output_dir results/tables/ciciot2023_50k >> "$LOG" 2>&1 \
        || { echo "FAIL $tag eval_base" >> "$ERRLOG"; sf=1; }
      if [ $sf -eq 0 ]; then
        python scripts/validate_result.py "results/tables/ciciot2023_50k/${cfg}_results.json" >> "$LOG" 2>&1 \
          || { echo "FAIL $tag validate_base" >> "$ERRLOG"; sf=1; }
        [ $sf -eq 0 ] && cp "results/tables/ciciot2023_50k/${cfg}_results.json" "$SEEDIR/${tag}_base.json"
      fi
    fi
    if [ $sf -eq 0 ]; then
      rm -f "models/hybrid_autoencoders/${cfg}_ae_known.pt"
      python src/evaluation/evaluate_edgeproto_hybrid.py --prefix "$cfg" --config "configs/$cfg.yaml" --edgeproto_model "models/oe_${cfg}_base/best_edgeproto.pt" >> "$LOG" 2>&1 \
        || { echo "FAIL $tag hybrid" >> "$ERRLOG"; sf=1; }
      if [ $sf -eq 0 ]; then
        python scripts/validate_result.py "results/tables/hybrid/${cfg}_Hybrid-Mean-50-50_results.json" >> "$LOG" 2>&1 \
          || { echo "FAIL $tag validate_hybrid" >> "$ERRLOG"; sf=1; }
        [ $sf -eq 0 ] && cp "results/tables/hybrid/${cfg}_Hybrid-Mean-50-50_results.json" "$SEEDIR/${tag}_hybrid.json"
      fi
    fi
    if [ $sf -eq 0 ]; then
      python scripts/oscr_one.py "$cfg" "$S" "models/oe_${cfg}_base/best_edgeproto.pt" >> "$LOG" 2>&1 || { echo "FAIL $tag oscr" >> "$ERRLOG"; sf=1; }
      grep -q "^${cfg},${S},Hybrid-Mean-50-50," "$OSCFILE" || { echo "FAIL $tag oscr_row" >> "$ERRLOG"; sf=1; }
    fi
    if [ $sf -eq 0 ]; then echo "OK $tag $(date)" >> "$LOG"; ok=$((ok+1)); else fail=$((fail+1)); echo "===== FAILED $tag - continuing =====" >> "$LOG"; fi
  done
  echo "===== SEED $S COMPLETE $(date) =====" >> "$LOG"
done
echo "=== OE RUN END $(date) OK=$ok FAIL=$fail ===" >> "$LOG"
echo "=== ERRORS ===" >> "$LOG"; cat "$ERRLOG" >> "$LOG" 2>/dev/null || true
