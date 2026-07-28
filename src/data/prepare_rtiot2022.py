import pandas as pd
import numpy as np
import os
import argparse
import json
import gc

LABEL_MAPPING = {
    'MQTT_Publish': 'Benign',
    'Thing_Speak': 'Benign',
    'Wipro_bulb': 'Benign',
    'DOS_SYN_Hping': 'DoS',
    'ARP_poisioning': 'ARP_Spoofing',
    'NMAP_UDP_SCAN': 'Recon',
    'NMAP_XMAS_TREE_SCAN': 'Recon',
    'NMAP_OS_DETECTION': 'Recon',
    'NMAP_TCP_scan': 'Recon',
    'NMAP_FIN_SCAN': 'Recon',
    'DDOS_Slowloris': 'DDoS',
    'Metasploit_Brute_Force_SSH': 'BruteForce'
}

def clean_data(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

def main(args):
    print(f"Reading dataset: {args.input}")
    df = pd.read_csv(args.input)
    
    # 1. Label counts before mapping
    if 'Attack_type' in df.columns:
        print("\nLabel counts BEFORE mapping:")
        print(df['Attack_type'].value_counts())
        
        # Apply mapping
        df['label'] = df['Attack_type'].map(LABEL_MAPPING)
        df.drop(columns=['Attack_type'], inplace=True)
        
        # Drop unmapped if any
        df.dropna(subset=['label'], inplace=True)
    
    if args.exclude_tiny_bruteforce == 1:
        df = df[df['label'] != 'BruteForce']
        
    print("\nLabel counts AFTER mapping:")
    print(df['label'].value_counts())
    
    # Drop index column
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)
        
    # Modes
    if args.feature_mode == 'conservative_no_ports':
        drop_cols = ['id.orig_p', 'id.resp_p', 'proto', 'service']
        df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors='ignore')
    elif args.feature_mode == 'full_numeric_encoded':
        # Factorize categorical
        for col in ['proto', 'service']:
            if col in df.columns:
                df[col] = pd.factorize(df[col])[0]
                
    df = clean_data(df)
    
    # Get numeric columns (excluding label)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # We need exactly 48 numeric columns.
    # Deterministically select first 48 or pad.
    selected_features = numeric_cols[:48]
    
    df_features = df[selected_features].copy()
    
    # Ensure float32
    df_features = df_features.astype(np.float32)
    
    # Apply log1p to non-negative heavy-tailed (we apply to all selected numeric since they are counts)
    # Ensure no negative values before log1p
    df_features = df_features.clip(lower=0.0)
    df_features = np.log1p(df_features)
    
    # Pad to 48 if necessary
    num_missing = 48 - len(selected_features)
    for i in range(num_missing):
        pad_col = f"pad_{i}"
        df_features[pad_col] = 0.0
        selected_features.append(pad_col)
        
    # Rename features
    rename_dict = {orig: f"feature_{i}" for i, orig in enumerate(selected_features)}
    df_features.rename(columns=rename_dict, inplace=True)
    
    # Combine with label
    df_features['label'] = df['label'].values
    
    # Free memory
    del df
    gc.collect()
    
    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_features.to_csv(args.output, index=False)
    
    print(f"\nSaved processed data to {args.output}")
    print(f"Shape: {df_features.shape}")
    
    # Metadata
    metadata = {
        "rows": len(df_features),
        "columns": len(df_features.columns),
        "feature_mode": args.feature_mode,
        "label_counts": df_features['label'].value_counts().to_dict(),
        "excluded_labels": ["BruteForce"] if args.exclude_tiny_bruteforce == 1 else [],
        "selected_features": selected_features,
        "feature_count": len(selected_features),
        "label_mapping": LABEL_MAPPING
    }
    meta_path = args.output.replace('.csv', '_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    # Selected features txt
    feat_txt_path = os.path.join(os.path.dirname(args.output), "rtiot2022_selected_features.txt")
    with open(feat_txt_path, 'w') as f:
        f.write("\n".join(selected_features))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--feature_mode", choices=['conservative_no_ports', 'full_numeric_encoded'], default='conservative_no_ports')
    parser.add_argument("--exclude_tiny_bruteforce", type=int, default=1)
    args = parser.parse_args()
    
    main(args)
