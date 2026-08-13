#!/bin/bash
# CICIoT2023 3-seed runner v2 — crash-tolerant, resumable, fully logged
cd ~/dualscore-edgeproto
export PYTHONPATH=.
LOG=~/ciciot_10k_seeds.log
ERRLOG=~/ciciot_10k_seeds_errors.log
DATA=data/processed/ciciot2023/ciciot2023_sample_10k_processed.csv
OSCFILE=results/tables/ciciot2023_10k/seeds_oscr.csv
SEEDIR=results/tables/ciciot2023_10k/seeds
SEEDS="42 123 2025"
EXPS="h1_mirai h2_recon h3_spoofing h4_web_bruteforce h5_dos h5b_dos_ddos"
mkdir -p "$SEEDIR"
[ -f "$OSCFILE" ] || echo "exp,seed,rule,oscr" > "$OSCFILE"

echo "=== RUN START $(date) ===" >> "$LOG"

if [ ! -f "$DATA" ]; then
  echo "FATAL: $DATA missing at $(date)" >> "$LOG"
  exit 1
fi

ok_count=0
fail_count=0

for S in $SEEDS; do
  for cfg in $EXPS; do
    tag="${cfg}_s${S}"
    if [ -f "$SEEDIR/${tag}_base.json" ] && [ -f "$SEEDIR/${tag}_hybrid.json" ] && grep -q "^${cfg},${S},Hybrid-Mean-50-50," "$OSCFILE"; then
      echo "SKIP $tag (already complete)" >> "$LOG"
      continue
    fi
    echo "===== START $tag $(date) =====" >> "$LOG"
    step_fail=0

    python src/data/make_splits.py --input "$DATA" --config "configs/$cfg.yaml" --output_dir data/splits >> "$LOG" 2>&1 \
      || { echo "FAIL $tag make_splits $(date)" >> "$ERRLOG"; step_fail=1; }
    if [ $step_fail -eq 0 ]; then
      for f in "data/splits/${cfg}_train.csv" "data/splits/${cfg}_val.csv" "data/splits/${cfg}_test_known.csv" "data/splits/${cfg}_test_unknown.csv"; do
        [ -s "$f" ] || { echo "FAIL $tag splitfile $f $(date)" >> "$ERRLOG"; step_fail=1; }
      done
    fi

    if [ $step_fail -eq 0 ]; then
      python src/training/train_edgeproto.py --config "configs/$cfg.yaml" --data_prefix "data/splits/$cfg" --output_dir "models/ciciot10k_${cfg}_base" --seed "$S" >> "$LOG" 2>&1 \
        || { echo "FAIL $tag train $(date)" >> "$ERRLOG"; step_fail=1; }
      [ -s "models/ciciot10k_${cfg}_base/best_edgeproto.pt" ] \
        || { echo "FAIL $tag model_missing $(date)" >> "$ERRLOG"; step_fail=1; }
    fi

    if [ $step_fail -eq 0 ]; then
      python src/evaluation/eval_open_set.py --config "configs/$cfg.yaml" --data_prefix "data/splits/$cfg" --model_path "models/ciciot10k_${cfg}_base/best_edgeproto.pt" --output_dir results/tables/ciciot2023_10k >> "$LOG" 2>&1 \
        || { echo "FAIL $tag eval_base $(date)" >> "$ERRLOG"; step_fail=1; }
      if [ $step_fail -eq 0 ]; then
        python scripts/validate_result.py "results/tables/ciciot2023_10k/${cfg}_results.json" >> "$LOG" 2>&1 \
          || { echo "FAIL $tag validate_base $(date)" >> "$ERRLOG"; step_fail=1; }
        [ $step_fail -eq 0 ] && cp "results/tables/ciciot2023_10k/${cfg}_results.json" "$SEEDIR/${tag}_base.json"
      fi
    fi

    if [ $step_fail -eq 0 ]; then
      rm -f "models/hybrid_autoencoders/${cfg}_ae_known.pt"
      python src/evaluation/evaluate_edgeproto_hybrid.py --prefix "$cfg" --config "configs/$cfg.yaml" --edgeproto_model "models/ciciot10k_${cfg}_base/best_edgeproto.pt" >> "$LOG" 2>&1 \
        || { echo "FAIL $tag hybrid $(date)" >> "$ERRLOG"; step_fail=1; }
      if [ $step_fail -eq 0 ]; then
        python scripts/validate_result.py "results/tables/hybrid/${cfg}_Hybrid-Mean-50-50_results.json" >> "$LOG" 2>&1 \
          || { echo "FAIL $tag validate_hybrid $(date)" >> "$ERRLOG"; step_fail=1; }
        [ $step_fail -eq 0 ] && cp "results/tables/hybrid/${cfg}_Hybrid-Mean-50-50_results.json" "$SEEDIR/${tag}_hybrid.json"
      fi
    fi

    if [ $step_fail -eq 0 ]; then
      python scripts/oscr_one.py "$cfg" "$S" >> "$LOG" 2>&1 \
        || { echo "FAIL $tag oscr $(date)" >> "$ERRLOG"; step_fail=1; }
      grep -q "^${cfg},${S},Hybrid-Mean-50-50," "$OSCFILE" \
        || { echo "FAIL $tag oscr_missing_row $(date)" >> "$ERRLOG"; step_fail=1; }
    fi

    if [ $step_fail -eq 0 ]; then
      echo "OK $tag $(date)" >> "$LOG"
      ok_count=$((ok_count+1))
    else
      fail_count=$((fail_count+1))
      echo "===== FAILED $tag — continuing $(date) =====" >> "$LOG"
    fi
  done
  echo "===== SEED $S COMPLETE $(date) =====" >> "$LOG"
done

echo "=== RUN END $(date) OK=$ok_count FAIL=$fail_count ===" >> "$LOG"
echo "=== ERRORS ===" >> "$LOG"
cat "$ERRLOG" >> "$LOG" 2>/dev/null || true
