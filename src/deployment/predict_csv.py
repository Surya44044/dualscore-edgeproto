import torch
import pandas as pd
import numpy as np
import argparse
import joblib

def predict(model_path, data_path, scaler_path, threshold):
    # Load scaler
    scaler = joblib.load(scaler_path)
    
    # Load data
    df = pd.read_csv(data_path)
    # Assuming CSV has 46 features and no label column, or we just drop label
    if 'label' in df.columns:
        features = df.drop(columns=['label'])
    else:
        features = df.copy()
        
    features = features.astype(np.float32)
    features = np.log1p(features)
    
    # pad
    num_features = features.shape[1]
    target_features = int(np.ceil(num_features / 4.0) * 4)
    pad_count = target_features - num_features
    for i in range(pad_count):
        features[f"pad_{i}"] = 0.0
        
    # Scale
    features = scaler.transform(features)
    features = features.astype(np.float32)
    
    # Reshape
    features = features.reshape(-1, 4, 12)
    tensor_x = torch.tensor(features)
    
    # Load model
    # Note: If it's TorchScript we use torch.jit.load
    try:
        model = torch.jit.load(model_path)
    except:
        # For simplicity in this demo, if it's not jit, we just fail gracefully
        print("This script expects a TorchScript exported model (e.g., from PTQ)")
        return
        
    model.eval()
    with torch.no_grad():
        _, dists = model(tensor_x)
        
    scores = -dists.numpy()
    max_scores = np.max(scores, axis=1)
    
    # Compare with threshold
    is_unknown = max_scores < threshold
    
    df['max_confidence'] = max_scores
    df['prediction'] = np.where(is_unknown, 'Unknown (Zero-Day)', 'Known')
    
    print(df[['max_confidence', 'prediction']].head(10))
    print(f"Total Unknown Detected: {is_unknown.sum()} / {len(is_unknown)}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--scaler", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=-10.0) # default example threshold
    args = parser.parse_args()
    
    predict(args.model, args.data, args.scaler, args.threshold)
