import numpy as np
import torch
from typing import List

def preprocess_features(features_list: List[float]) -> torch.Tensor:
    """
    Validates and transforms 48 numeric features into PyTorch Tensor [1, 4, 12].
    """
    if len(features_list) != 48:
        raise ValueError(f"Feature vector must contain exactly 48 elements, got {len(features_list)}.")
        
    arr = np.array(features_list, dtype=np.float32)
    # Reshape to [1, 4, 12] for EdgeProto encoder
    tensor_3d = torch.tensor(arr.reshape(1, 4, 12))
    return tensor_3d

def preprocess_batch_features(batch_list: List[List[float]]) -> torch.Tensor:
    """
    Transforms batch of 48 numeric features into PyTorch Tensor [B, 4, 12].
    """
    arr_list = []
    for f in batch_list:
        if len(f) != 48:
            raise ValueError(f"Each sample in batch must contain exactly 48 elements, got {len(f)}.")
        arr_list.append(np.array(f, dtype=np.float32))
        
    arr_batch = np.stack(arr_list, axis=0) # [B, 48]
    tensor_3d = torch.tensor(arr_batch.reshape(-1, 4, 12))
    return tensor_3d
