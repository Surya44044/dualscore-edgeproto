import pandas as pd
import numpy as np
import os
import gc
import argparse

def preprocess_chunks(input_file, output_file, chunksize=25000):
    """Reads data in chunks, applies log1p, pads features, and saves."""
    print(f"Preprocessing {input_file} in chunks of {chunksize}...")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    first_chunk = True
    
    for chunk_idx, df_chunk in enumerate(pd.read_csv(input_file, chunksize=chunksize)):
        # Identify numeric and label columns
        label_col = 'label'
        numeric_cols = [c for c in df_chunk.columns if c != label_col]
        
        # Drop rows with NaN or Inf
        df_chunk = df_chunk.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Apply float32 casting for memory safety
        df_chunk[numeric_cols] = df_chunk[numeric_cols].astype(np.float32)
        
        # Apply log1p scaling to handle heavy tails
        df_chunk[numeric_cols] = np.log1p(df_chunk[numeric_cols])
        
        # Pad to multiple of 4 (e.g., 46 -> 48)
        num_features = len(numeric_cols)
        target_features = int(np.ceil(num_features / 4.0) * 4)
        pad_count = target_features - num_features
        
        for i in range(pad_count):
            df_chunk[f"pad_{i}"] = 0.0
            
        new_numeric_cols = numeric_cols + [f"pad_{i}" for i in range(pad_count)]
        df_chunk[new_numeric_cols] = df_chunk[new_numeric_cols].astype(np.float32)
        
        # Save chunk
        mode = 'w' if first_chunk else 'a'
        header = first_chunk
        df_chunk.to_csv(output_file, mode=mode, header=header, index=False)
        
        first_chunk = False
        
        print(f"Processed chunk {chunk_idx+1}")
        
        # Explicit garbage collection
        del df_chunk
        gc.collect()

    print(f"Preprocessing complete. Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input raw CSV")
    parser.add_argument("--output", type=str, required=True, help="Output processed CSV")
    parser.add_argument("--chunksize", type=int, default=25000, help="Rows per chunk")
    args = parser.parse_args()
    
    preprocess_chunks(args.input, args.output, args.chunksize)
