#!/usr/bin/env python3
"""CICIoT2023 Phase 1: discovery + class-balanced chunked sampler."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "ciciot2023" / "train.parquet"
OUT_DIR = ROOT / "data" / "processed" / "ciciot2023"
LABEL_COLS = ["Label", "Label_orig", "attack_class", "label"]

def discover(pf):
    print("=" * 70)
    print("DISCOVERY - CICIoT2023 train split")
    print("=" * 70)
    md = pf.metadata
    print(f"rows       : {md.num_rows:,}")
    print(f"columns    : {md.num_columns}")
    print(f"row groups : {pf.num_row_groups}")
    print(f"file size  : {RAW.stat().st_size / 1e9:.2f} GB")
    t0 = time.time()
    counts = {}
    for i in range(pf.num_row_groups):
        s = pf.read_row_group(i, columns=["attack_class"]).to_pandas()["attack_class"].value_counts(dropna=False)
        for k, v in s.items():
            counts[k] = counts.get(k, 0) + int(v)
    print(f"\nattack_class distribution (all row groups, {time.time()-t0:.1f}s):")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {str(k):<12} {v:>12,}  ({v/md.num_rows*100:.2f}%)")
    print("\nNaN/Inf estimate (first 3 row groups):")
    df = pd.concat([pf.read_row_group(i).to_pandas() for i in range(min(3, pf.num_row_groups))], ignore_index=True)
    numeric = [c for c in df.columns if c not in LABEL_COLS and pd.api.types.is_numeric_dtype(df[c])]
    bad = df[numeric].isna().sum() + np.isinf(df[numeric].to_numpy()).sum(axis=0)
    bad = bad[bad > 0]
    print(bad.to_string() if len(bad) else "  no NaN/Inf in checked rows")
    print(f"\nnumeric feature columns: {len(numeric)}")
    print(f"label columns: {LABEL_COLS}")

def build_balanced_sample(pf, n_per_class, seed):
    rng = np.random.default_rng(seed)
    needed, positions = {}, {}
    nan_label_rows = 0
    t0 = time.time()
    for i in range(pf.num_row_groups):
        ac = pf.read_row_group(i, columns=["attack_class"]).to_pandas()["attack_class"].to_numpy()
        nan_mask = pd.isna(ac)
        if nan_mask.any():
            nan_label_rows += int(nan_mask.sum())
            ac = ac[~nan_mask]
        for c in np.unique(ac):
            if c not in needed:
                needed[c] = n_per_class
                positions[c] = []
        for c in list(needed):
            if needed[c] <= 0:
                continue
            idx = np.where(ac == c)[0]
            if len(idx) == 0:
                continue
            take = min(len(idx), needed[c])
            chosen = rng.choice(idx, size=take, replace=False)
            positions[c].extend((i, int(p)) for p in chosen)
            needed[c] -= take
        if all(v <= 0 for v in needed.values()):
            print(f"  sampling done at row group {i} ({time.time()-t0:.1f}s)")
            break
    if nan_label_rows:
        print(f"  skipped empty-label rows: {nan_label_rows}")
    rg_needed = {}
    for c, lst in positions.items():
        for rg, pos in lst:
            rg_needed.setdefault(rg, []).append(pos)
    print(f"  reading {len(rg_needed)} row groups for selected rows ...")
    parts = []
    for rg in sorted(rg_needed):
        df = pf.read_row_group(rg).to_pandas()
        parts.append(df.iloc[rg_needed[rg]])
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="rows per attack_class")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="smoke_2k")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(str(RAW))
    discover(pf)
    print("\n" + "=" * 70)
    print(f"BUILDING BALANCED SAMPLE: {args.n} rows/class (seed={args.seed})")
    print("=" * 70)
    sample = build_balanced_sample(pf, args.n, args.seed)
    counts = sample["attack_class"].value_counts().sort_index()
    print(f"\nsampled rows total : {len(sample):,}")
    print("per class:")
    print(counts.to_string())
    print(f"binary label counts: {sample['label'].value_counts().to_dict()}")
    out_parquet = OUT_DIR / f"ciciot2023_{args.tag}_{args.n}perclass.parquet"
    sample.to_parquet(out_parquet, index=False)
    numeric_cols = [c for c in sample.columns if c not in LABEL_COLS and pd.api.types.is_numeric_dtype(sample[c])]
    meta = {
        "tag": args.tag, "n_per_class": args.n, "seed": args.seed,
        "total_rows": int(len(sample)), "source": str(RAW),
        "numeric_feature_columns": numeric_cols, "label_columns": LABEL_COLS,
        "per_class_counts": {str(k): int(v) for k, v in counts.items()},
    }
    out_meta = OUT_DIR / f"ciciot2023_{args.tag}_{args.n}perclass_meta.json"
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nsaved : {out_parquet.name}")
    print(f"saved : {out_meta.name}")

if __name__ == "__main__":
    main()
