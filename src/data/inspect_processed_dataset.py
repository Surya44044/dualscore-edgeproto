import pandas as pd
import numpy as np
import argparse
import sys

def main(input_csv):
    print(f"Inspecting: {input_csv}")
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        sys.exit(1)
        
    cols = df.columns.tolist()
    if 'label' not in cols:
        print("ERROR: 'label' column is missing.")
        sys.exit(1)
        
    feature_cols = [c for c in cols if c != 'label']
    
    # Check count
    if len(feature_cols) != 48:
        print(f"ERROR: Expected exactly 48 feature columns, but found {len(feature_cols)}.")
        sys.exit(1)
    
    # Check numeric
    non_numeric = df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()
    if len(non_numeric) > 0:
        print(f"ERROR: Found non-numeric feature columns: {non_numeric}")
        sys.exit(1)
        
    print("SUCCESS: Exactly 48 numeric feature columns + label found.")
    
    # NaN / Inf Check
    has_nan = df.isna().any().any()
    has_inf = np.isinf(df[feature_cols]).any().any()
    
    if has_nan or has_inf:
        print("ERROR: NaN or Inf values found in the dataset.")
        sys.exit(1)
    else:
        print("SUCCESS: No NaN or Inf values found.")
        
    # Label counts
    print("\nLabel counts:")
    print(df['label'].value_counts())
    
    # Memory Usage
    mem_usage = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"\nMemory usage: {mem_usage:.2f} MB")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to processed CSV")
    args = parser.parse_args()
    
    main(args.input)
