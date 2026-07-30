import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, IsolationForest

class MLP_MSP(nn.Module):
    def __init__(self, input_dim=48, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

class AutoencoderKnown(nn.Module):
    def __init__(self, input_dim=48):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
        
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

def get_random_forest():
    return RandomForestClassifier(
        n_estimators=50,
        max_depth=20,
        class_weight='balanced',
        n_jobs=1,
        random_state=42
    )

def get_isolation_forest(train_len):
    return IsolationForest(
        n_estimators=100,
        max_samples=min(10000, train_len),
        contamination='auto',
        n_jobs=1,
        random_state=42
    )
