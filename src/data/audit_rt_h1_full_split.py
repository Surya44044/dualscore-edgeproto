from pathlib import Path
import json
import numpy as np
import pandas as pd

PROCESSED = Path("data/processed/rtiot2022_grouped_48.csv")
META = Path("data/processed/rtiot2022_grouped_48_metadata.json")
FEATURES = Path("data/processed/rtiot2022_selected_features.txt")

PREFIX = "data/splits/rt_h1_arp_spoofing"

KNOWN = {"Benign", "DoS", "Recon", "DDoS"}
UNKNOWN = "ARP_Spoofing"

def fail(msg):
    print(f"[FAIL] {msg}")
    raise SystemExit(1)

def passed(msg):
    print(f"[PASS] {msg}")

def check_csv_schema():
    if not PROCESSED.exists():
        fail(f"{PROCESSED} missing")
    passed(f"{PROCESSED} exists")

    df = pd.read_csv(PROCESSED, nrows=5000)

    expected_cols = [f"feature_{i}" for i in range(48)] + ["label"]
    if list(df.columns) != expected_cols:
        print("Actual columns:", list(df.columns))
        fail("Processed CSV columns are not exactly feature_0..feature_47,label")
    passed("Processed CSV has exactly 48 features + label")

    feature_cols = [f"feature_{i}" for i in range(48)]

    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        fail(f"Non-numeric feature columns: {non_numeric}")
    passed("All feature columns numeric in sample")

    arr = df[feature_cols].to_numpy()
    if not np.isfinite(arr).all():
        fail("NaN/Inf detected in processed sample")
    passed("No NaN/Inf in processed sample")

def check_metadata_and_features():
    if not META.exists():
        fail(f"{META} missing")
    meta = json.loads(META.read_text())
    passed("Metadata JSON valid")

    if not FEATURES.exists():
        fail(f"{FEATURES} missing")

    features = FEATURES.read_text().splitlines()
    print("Selected feature count:", len(features))
    print("Selected features:")
    for f in features:
        print(" ", f)

    if len(features) != 48:
        fail("Selected feature list does not contain 48 lines")

    bad = {"Attack_type", "label", "Unnamed: 0", "target", "category"}
    bad_hits = [f for f in features if f in bad]
    if bad_hits:
        fail(f"Leakage features found: {bad_hits}")
    passed("No obvious label/index leakage features")

    if meta.get("feature_mode") == "conservative_no_ports":
        port_hits = [f for f in features if f in {"id.orig_p", "id.resp_p"}]
        if port_hits:
            fail(f"Port features found despite conservative_no_ports: {port_hits}")
        passed("conservative_no_ports respected: no id.orig_p/id.resp_p")

def read_split(name):
    path = Path(f"{PREFIX}_{name}.csv")
    if not path.exists():
        fail(f"Missing split: {path}")
    df = pd.read_csv(path)
    passed(f"{path} exists with {len(df)} rows")
    return df

def check_splits():
    train = read_split("train")
    val = read_split("val")
    test_known = read_split("test_known")
    test_unknown = read_split("test_unknown")

    splits = {
        "train": train,
        "val": val,
        "test_known": test_known,
        "test_unknown": test_unknown,
    }

    for name, df in splits.items():
        if "label" not in df.columns:
            fail(f"{name} missing label column")

    for name in ["train", "val", "test_known"]:
        labels = set(splits[name]["label"].unique())
        print(f"{name} labels:", labels)

        if UNKNOWN in labels:
            fail(f"{UNKNOWN} leaked into {name}")

        if not labels.issubset(KNOWN):
            fail(f"{name} has unexpected labels: {labels - KNOWN}")

        passed(f"{name} contains only known labels")

    unk_labels = set(test_unknown["label"].unique())
    print("test_unknown labels:", unk_labels)

    if unk_labels != {UNKNOWN}:
        fail(f"test_unknown is not exactly {UNKNOWN}")

    passed("test_unknown contains only held-out unknown class")

    print("\nSplit counts:")
    for name, df in splits.items():
        print(f"  {name}: {len(df)}")
        print(df["label"].value_counts().to_string())

def main():
    print("=" * 100)
    print("RT-H1 FULL SPLIT AUDIT")
    print("=" * 100)

    check_csv_schema()
    check_metadata_and_features()
    check_splits()

    print("=" * 100)
    print("[AUDIT PASS] RT-H1 full split audit passed.")
    print("=" * 100)

if __name__ == "__main__":
    main()
