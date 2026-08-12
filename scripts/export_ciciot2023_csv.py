#!/usr/bin/env python3
"""CICIoT2023: sampled parquet -> processed CSV (label = attack_class) for split/train pipeline."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABEL_COLS = ["Label", "Label_orig", "attack_class", "label"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="sampled parquet path")
    ap.add_argument("--tag", default="smoke_2k")
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    df["label"] = df["attack_class"].astype(str)
    num_cols = [c for c in df.columns if c not in LABEL_COLS and pd.api.types.is_numeric_dtype(df[c])]
    out_df = df[num_cols + ["label"]].astype({c: "float32" for c in num_cols})
    out = ROOT / "data" / "processed" / "ciciot2023" / f"ciciot2023_{args.tag}_processed.csv"
    out_df.to_csv(out, index=False)
    print(f"rows: {len(out_df):,}  features: {len(num_cols)}  -> {out.name}")
    print(out_df["label"].value_counts().to_string())

if __name__ == "__main__":
    main()
