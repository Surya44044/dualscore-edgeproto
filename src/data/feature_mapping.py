import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class IoTDataset(Dataset):
    def __init__(self, csv_file, label_map=None, is_train=True):
        self.df = pd.read_csv(csv_file)
        
        # Get labels and drop from features
        self.labels_str = self.df['label'].values
        numeric_cols = [c for c in self.df.columns if c != 'label']
        
        self.features = self.df[numeric_cols].values.astype(np.float32)
        
        # Reshape features to [B, Channels, Length] -> [B, 4, 12]
        # We assume 48 features
        actual_features = self.features.shape[1]
        assert actual_features == 48, f"Error: Expected exactly 48 numeric features, but got {actual_features}. Ensure preprocess/prepare script pads or truncates correctly."
        self.features = self.features.reshape(-1, 4, 12)
        
        # Build label map if not provided
        if label_map is None:
            unique_labels = sorted(list(set(self.labels_str)))
            self.label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}
        else:
            self.label_map = label_map
            
        # Map labels. Unknown labels get -1
        self.labels = np.array([self.label_map.get(lbl, -1) for lbl in self.labels_str], dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])

def get_dataloaders(train_csv, val_csv, test_known_csv, test_unknown_csv, batch_size=256, num_workers=0):
    # Create train set to build label map
    train_dataset = IoTDataset(train_csv, is_train=True)
    label_map = train_dataset.label_map
    
    val_dataset = IoTDataset(val_csv, label_map=label_map, is_train=False)
    test_known_dataset = IoTDataset(test_known_csv, label_map=label_map, is_train=False)
    
    # Unknown gets label -1
    test_unknown_dataset = IoTDataset(test_unknown_csv, label_map=label_map, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_known_loader = DataLoader(test_known_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_unknown_loader = DataLoader(test_unknown_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_known_loader, test_unknown_loader, label_map
