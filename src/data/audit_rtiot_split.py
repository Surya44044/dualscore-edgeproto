import argparse
import os
import sys
import pandas as pd
import numpy as np

def check(condition, msg, fail_msg=None, exit_on_fail=True):
    if not condition:
        print(f"[FAIL] {fail_msg if fail_msg else msg}")
        if exit_on_fail:
            sys.exit(1)
    else:
        print(f"[PASS] {msg}")

def audit_split(prefix, known_classes, unknown_class):
    splits = ["train", "val", "test_known", "test_unknown"]
    files = {s: f"{prefix}_{s}.csv" for s in splits}
    
    # 1. Check existence
    for s, path in files.items():
        check(os.path.exists(path), f"Split file exists: {path}", exit_on_fail=True)
        
    dfs = {s: pd.read_csv(files[s]) for s in splits}
    
    for s, df in dfs.items():
        cols = df.columns.tolist()
        # 2. Exactly 48 numeric feature columns + label
        expected_cols = [f"feature_{i}" for i in range(48)] + ["label"]
        check(set(cols) == set(expected_cols) and len(cols) == 49,
              f"[{s}] Exactly 48 features + label", exit_on_fail=True)
              
        feat_cols = [c for c in cols if c != "label"]
        non_numeric = df[feat_cols].select_dtypes(exclude=[np.number]).columns.tolist()
        check(len(non_numeric) == 0, f"[{s}] All features are numeric", fail_msg=f"[{s}] Non-numeric: {non_numeric}", exit_on_fail=True)
        
        # 3. No NaN/Inf
        has_nan = df.isna().any().any()
        has_inf = np.isinf(df[feat_cols]).any().any()
        check(not has_nan and not has_inf, f"[{s}] No NaN or Inf found", exit_on_fail=True)
        
    # 4. Unknown class absent from train/val/test_known
    for s in ["train", "val", "test_known"]:
        unique_labels = dfs[s]['label'].unique()
        check(unknown_class not in unique_labels, f"[{s}] Unknown class ({unknown_class}) is absent.", exit_on_fail=True)
        
    # 5. test_unknown contains only the unknown class
    tu_labels = dfs["test_unknown"]['label'].unique()
    check(len(tu_labels) == 1 and tu_labels[0] == unknown_class, f"[test_unknown] contains ONLY {unknown_class}", exit_on_fail=True)
    
    # 6. train/val/test_known contain only the listed known classes
    known_set = set(known_classes)
    for s in ["train", "val", "test_known"]:
        unique_labels = set(dfs[s]['label'].unique())
        check(unique_labels.issubset(known_set), f"[{s}] contains only listed known classes", 
              fail_msg=f"[{s}] Found labels not in known_set: {unique_labels - known_set}", exit_on_fail=True)
              
    # 7. Print row counts and label counts
    print("\n=== Split Row Counts ===")
    for s in splits:
        print(f"  {s}: {len(dfs[s])}")
        
    print("\n=== Label Counts ===")
    for s in splits:
        print(f"\n[{s}]")
        print(dfs[s]['label'].value_counts())
        
    print("\n[SUCCESS] Split audit passed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--known", nargs="+", required=True)
    parser.add_argument("--unknown", required=True)
    args = parser.parse_args()
    
    audit_split(args.prefix, args.known, args.unknown)
