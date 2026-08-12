import pandas as pd
import numpy as np
import os
import yaml
import argparse
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def make_splits(input_file, config_path, output_dir="data/splits"):
    print(f"Generating zero-day splits from {input_file} using {config_path}")
    os.makedirs(output_dir, exist_ok=True)
    
    config = load_config(config_path)
    held_out_classes = config['data']['held_out_classes']
    known_classes = config['data']['known_classes']
    
    # Load processed data
    df = pd.read_csv(input_file)
    
    if 'label' not in df.columns:
        raise ValueError("Error: Processed CSV is missing the 'label' column.")
        
    numeric_cols = [c for c in df.columns if c != 'label']
    non_numeric = df[numeric_cols].select_dtypes(exclude=[np.number]).columns.tolist()
    if len(non_numeric) > 0:
        raise ValueError(f"Error: Non-numeric features found in processed CSV: {non_numeric}")
    
    # Filter valid classes
    df = df[df['label'].isin(known_classes + held_out_classes)]
    
    # Separate into known and unknown
    df_known = df[df['label'].isin(known_classes)].copy()
    df_unknown = df[df['label'].isin(held_out_classes)].copy()
    
    # Train/Val/Test for known
    # Train: 70%, Val: 10%, Test: 20%
    train_df, temp_df = train_test_split(df_known, test_size=0.3, random_state=42, stratify=df_known['label'])
    val_df, test_known_df = train_test_split(temp_df, test_size=0.6666, random_state=42, stratify=temp_df['label'])
    
    test_unknown_df = df_unknown.copy()
    
    # Fit scaler on train only
    numeric_cols = [c for c in train_df.columns if c != 'label']
    scaler = StandardScaler()
    
    train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
    val_df[numeric_cols] = scaler.transform(val_df[numeric_cols])
    test_known_df[numeric_cols] = scaler.transform(test_known_df[numeric_cols])
    
    if len(test_unknown_df) > 0:
        test_unknown_df[numeric_cols] = scaler.transform(test_unknown_df[numeric_cols])
    # Pad numeric features to 48 (CICIoT2023 has 46); zero-pad AFTER scaling.
    TARGET_FEATURES = 48
    if len(numeric_cols) < TARGET_FEATURES:
        pad_cols = [f"pad_{i:02d}" for i in range(TARGET_FEATURES - len(numeric_cols))]
        for c in pad_cols:
            train_df[c] = 0.0
            val_df[c] = 0.0
            test_known_df[c] = 0.0
            if len(test_unknown_df) > 0:
                test_unknown_df[c] = 0.0
        numeric_cols = numeric_cols + pad_cols
    
    # Type cast back to float32
    train_df[numeric_cols] = train_df[numeric_cols].astype(np.float32)
    val_df[numeric_cols] = val_df[numeric_cols].astype(np.float32)
    test_known_df[numeric_cols] = test_known_df[numeric_cols].astype(np.float32)
    if len(test_unknown_df) > 0:
        test_unknown_df[numeric_cols] = test_unknown_df[numeric_cols].astype(np.float32)
    
    # Save outputs
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    out_prefix = os.path.join(output_dir, config_name)
    
    train_df.to_csv(f"{out_prefix}_train.csv", index=False)
    val_df.to_csv(f"{out_prefix}_val.csv", index=False)
    test_known_df.to_csv(f"{out_prefix}_test_known.csv", index=False)
    test_unknown_df.to_csv(f"{out_prefix}_test_unknown.csv", index=False)
    
    joblib.dump(scaler, f"{out_prefix}_scaler.pkl")
    
    print(f"Splits created for {config_name}:")
    print(f"  Train: {len(train_df)}")
    print(f"  Val: {len(val_df)}")
    print(f"  Test Known: {len(test_known_df)}")
    print(f"  Test Unknown: {len(test_unknown_df)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Processed input CSV")
    parser.add_argument("--config", type=str, required=True, help="Path to holdout config yaml")
    parser.add_argument("--output_dir", type=str, default="data/splits", help="Output directory")
    args = parser.parse_args()
    
    make_splits(args.input, args.config, args.output_dir)
