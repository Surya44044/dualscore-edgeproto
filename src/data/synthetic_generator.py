import pandas as pd
import numpy as np
import os
import argparse

def generate_synthetic_data(num_samples_per_class=2000, output_dir="data/raw"):
    os.makedirs(output_dir, exist_ok=True)
    
    classes = [
        "Benign", "DDoS", "DoS", "Recon", 
        "Web-based", "Brute Force", "Spoofing", "Mirai"
    ]
    
    # Generate 46 feature names
    features = [f"feature_{i}" for i in range(46)]
    
    data = []
    
    np.random.seed(42)
    
    for cls_idx, cls_name in enumerate(classes):
        # We simulate some distinct patterns by shifting the mean/std based on class index
        for _ in range(num_samples_per_class):
            row = []
            for f_idx in range(46):
                # First 20 features: timing/counts (exponential-like)
                if f_idx < 20:
                    val = np.random.exponential(scale=1.0 + cls_idx * 0.5)
                # Next 15 features: flag/ratios (uniform/normal bounded)
                elif f_idx < 35:
                    val = np.random.normal(loc=0.5 + cls_idx * 0.1, scale=0.1)
                    val = np.clip(val, 0, 1)
                # Remaining 11 features: packet length stats (normal)
                else:
                    val = np.random.normal(loc=100 + cls_idx * 10, scale=20)
                    val = max(0, val)
                
                row.append(val)
            
            row.append(cls_name)
            data.append(row)
            
    columns = features + ["label"]
    df = pd.DataFrame(data, columns=columns)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    out_file = os.path.join(output_dir, "synthetic_ciciot2023.csv")
    df.to_csv(out_file, index=False)
    print(f"Generated {len(df)} synthetic samples at {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_per_class", type=int, default=2000, help="Number of samples per class")
    parser.add_argument("--output_dir", type=str, default="data/raw", help="Output directory")
    args = parser.parse_args()
    
    generate_synthetic_data(args.samples_per_class, args.output_dir)
