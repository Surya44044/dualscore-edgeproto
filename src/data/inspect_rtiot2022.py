from pathlib import Path
import sys
import pandas as pd
from collections import Counter

POSSIBLE_LABEL_COLUMNS = [
    "Attack_type",
    "attack_type",
    "label",
    "Label",
    "class",
    "Class",
    "target",
    "Target",
    "category",
    "Category"
]

def safe_read_csv(path, nrows=5):
    try:
        return pd.read_csv(path, nrows=nrows)
    except UnicodeDecodeError:
        return pd.read_csv(path, nrows=nrows, encoding="latin1")

def find_label_column(columns):
    for c in POSSIBLE_LABEL_COLUMNS:
        if c in columns:
            return c
    return None

def count_rows_and_labels(csv_path, label_col=None, chunksize=25000):
    total_rows = 0
    label_counts = Counter()

    try:
        reader = pd.read_csv(csv_path, chunksize=chunksize)
    except UnicodeDecodeError:
        reader = pd.read_csv(csv_path, chunksize=chunksize, encoding="latin1")

    for i, chunk in enumerate(reader, 1):
        total_rows += len(chunk)

        if label_col and label_col in chunk.columns:
            vals = chunk[label_col].astype(str).fillna("MISSING")
            label_counts.update(vals.tolist())

        print(f"  processed chunk {i}, rows so far: {total_rows}")

    return total_rows, label_counts

def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/RT-IoT2022")
    root = root.resolve()

    print("=" * 100)
    print("RT-IoT2022 INSPECTION")
    print("=" * 100)
    print("Root:", root)

    if not root.exists():
        print("ERROR: folder does not exist")
        return

    csv_files = sorted(root.rglob("*.csv"))

    print("\nCSV files found:", len(csv_files))

    if not csv_files:
        print("No CSV files found. Listing all files:")
        for f in sorted(root.rglob("*")):
            if f.is_file():
                print("-", f)
        return

    for csv_path in csv_files:
        print("\n" + "=" * 100)
        print("CSV:", csv_path)
        print("Size MB:", round(csv_path.stat().st_size / (1024 * 1024), 2))

        sample = safe_read_csv(csv_path, nrows=5)

        print("\nShape of 5-row sample:", sample.shape)

        print("\nColumns:")
        for i, col in enumerate(sample.columns):
            print(f"  {i:02d}: {col}")

        label_col = find_label_column(sample.columns)

        print("\nDetected label column:", label_col)

        print("\nDtypes:")
        print(sample.dtypes.to_string())

        print("\nFirst 3 rows:")
        print(sample.head(3).to_string())

        print("\nCounting rows and labels safely in chunks...")
        total_rows, label_counts = count_rows_and_labels(csv_path, label_col=label_col)

        print("\nTOTAL ROWS:", total_rows)

        if label_col:
            print(f"\nLabel counts for column '{label_col}':")
            for label, count in label_counts.most_common():
                print(f"  {label}: {count}")
        else:
            print("\nWARNING: No label column detected automatically.")
            print("You must inspect column names manually and tell the assistant.")

    print("\n" + "=" * 100)
    print("INSPECTION COMPLETE")
    print("=" * 100)

if __name__ == "__main__":
    main()
