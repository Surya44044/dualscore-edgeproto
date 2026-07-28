import os
import sys
import pandas as pd
import numpy as np
import json

def check(condition, msg, fail_msg=None, exit_on_fail=True):
    if not condition:
        print(f"[FAIL] {fail_msg if fail_msg else msg}")
        if exit_on_fail:
            sys.exit(1)
    else:
        print(f"[PASS] {msg}")

def main():
    processed_csv = "data/processed/rtiot2022_grouped_48.csv"
    meta_json = "data/processed/rtiot2022_grouped_48_metadata.json"
    feat_txt = "data/processed/rtiot2022_selected_features.txt"
    
    # 1. Processed CSV exists
    check(os.path.exists(processed_csv), f"{processed_csv} exists.", exit_on_fail=True)
    df = pd.read_csv(processed_csv)
    cols = df.columns.tolist()
    
    # 2 & 3. Exactly 48 feature columns named feature_0...feature_47 plus label
    expected_cols = [f"feature_{i}" for i in range(48)] + ["label"]
    check(set(cols) == set(expected_cols) and len(cols) == 49, 
          "Exactly 48 feature columns (feature_0...feature_47) plus label.", 
          exit_on_fail=True)
    
    # 4. All feature columns are numeric
    feature_cols = [c for c in cols if c != "label"]
    non_numeric = df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()
    check(len(non_numeric) == 0, "All feature columns are numeric.", 
          fail_msg=f"Found non-numeric features: {non_numeric}", exit_on_fail=True)
    
    # 5. No NaN/Inf
    has_nan = df.isna().any().any()
    has_inf = np.isinf(df[feature_cols]).any().any()
    check(not has_nan and not has_inf, "No NaN/Inf found.", exit_on_fail=True)
    
    # 6. Metadata JSON exists and is valid
    check(os.path.exists(meta_json), f"{meta_json} exists and is valid.", exit_on_fail=False)
    metadata = {}
    if os.path.exists(meta_json):
        try:
            with open(meta_json, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"[FAIL] Metadata JSON is invalid: {e}")
            
    # 7. Selected feature list exists
    if not os.path.exists(feat_txt):
        print(f"[REPRODUCIBILITY FAIL] Selected feature list {feat_txt} is missing.")
    else:
        print(f"[PASS] Selected feature list {feat_txt} exists.")
        with open(feat_txt, 'r') as f:
            selected_features = [line.strip() for line in f.readlines() if line.strip()]
            
        # 8. No leakage columns
        leakage_cols = ["Attack_type", "label", "Unnamed: 0", "target", "category"]
        found_leakage = [c for c in leakage_cols if c in selected_features]
        check(len(found_leakage) == 0, "No obvious leakage columns in selected feature list.", 
              fail_msg=f"Leakage columns found: {found_leakage}", exit_on_fail=True)
              
        # 9. conservative_no_ports checks
        feature_mode = metadata.get("feature_mode", "")
        if feature_mode == "conservative_no_ports":
            banned_ports = ["id.orig_p", "id.resp_p"]
            found_ports = [c for c in banned_ports if c in selected_features]
            check(len(found_ports) == 0, "conservative_no_ports respected. No id.orig_p/id.resp_p found.", 
                  exit_on_fail=True)
                  
    # 10. Check RT-H1 split files
    splits = ["train", "val", "test_known", "test_unknown"]
    full_prefix = "rt_h1_arp_spoofing"
    override_prefix = "rt_h1_arp_spoofing_override"
    splits_dir = "data/splits"
    
    full_exists = all(os.path.exists(os.path.join(splits_dir, f"{full_prefix}_{s}.csv")) for s in splits)
    override_exists = all(os.path.exists(os.path.join(splits_dir, f"{override_prefix}_{s}.csv")) for s in splits)
    
    if override_exists and not full_exists:
        print("[WARNING] Only _override split files exist! These are not full-run files.")
        active_prefix = override_prefix
    elif full_exists:
        print("[PASS] Full RT-H1 split files exist.")
        active_prefix = full_prefix
    else:
        print("[FAIL] Neither full nor override splits completely exist.")
        sys.exit(0) # Exit nonzero only for data leakage or invalid schema, missing splits just stops split check
        
    df_train = pd.read_csv(os.path.join(splits_dir, f"{active_prefix}_train.csv"))
    df_val = pd.read_csv(os.path.join(splits_dir, f"{active_prefix}_val.csv"))
    df_tk = pd.read_csv(os.path.join(splits_dir, f"{active_prefix}_test_known.csv"))
    df_tu = pd.read_csv(os.path.join(splits_dir, f"{active_prefix}_test_unknown.csv"))
    
    # 12. Verify ARP_Spoofing appears only in test_unknown
    target_unknown = "ARP_Spoofing"
    check(target_unknown not in df_train['label'].unique(), f"{target_unknown} not in train.", exit_on_fail=False)
    check(target_unknown not in df_val['label'].unique(), f"{target_unknown} not in val.", exit_on_fail=False)
    check(target_unknown not in df_tk['label'].unique(), f"{target_unknown} not in test_known.", exit_on_fail=False)
    
    # 13. Verify test_unknown contains only ARP_Spoofing
    tu_unique = df_tu['label'].unique()
    check(len(tu_unique) == 1 and tu_unique[0] == target_unknown, f"test_unknown contains ONLY {target_unknown}.", exit_on_fail=False)
    
    # 14. Verify train/val/test_known contain only known labels
    known_labels = {"Benign", "DoS", "Recon", "DDoS"}
    train_unique = set(df_train['label'].unique())
    val_unique = set(df_val['label'].unique())
    tk_unique = set(df_tk['label'].unique())
    
    check(train_unique.issubset(known_labels), f"Train contains only known labels: {train_unique}", exit_on_fail=False)
    check(val_unique.issubset(known_labels), f"Val contains only known labels: {val_unique}", exit_on_fail=False)
    check(tk_unique.issubset(known_labels), f"Test Known contains only known labels: {tk_unique}", exit_on_fail=False)
    
    # 15. Print all split counts and label counts
    print(f"\n[Split Row Counts]")
    print(f"  Train: {len(df_train)}")
    print(f"  Val: {len(df_val)}")
    print(f"  Test Known: {len(df_tk)}")
    print(f"  Test Unknown: {len(df_tu)}")
    
    print("\n[Train Label Counts]")
    print(df_train['label'].value_counts())
    print("\n[Test Unknown Label Counts]")
    print(df_tu['label'].value_counts())

if __name__ == "__main__":
    main()
