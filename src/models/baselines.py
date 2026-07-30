import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest, RandomForestClassifier

class Autoencoder(nn.Module):
    def __init__(self, input_dim=48, hidden_dim=32, latent_dim=16):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(True)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x):
        # x is [B, 4, 12]. Flatten to [B, 48]
        x_flat = x.view(x.size(0), -1)
        z = self.encoder(x_flat)
        x_recon = self.decoder(z)
        
        # Reshape back if needed, but MSE loss can just use x_flat
        return x_recon

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=48, hidden_dim=32, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        logits = self.net(x_flat)
        return logits

def get_isolation_forest():
    # Basic isolation forest baseline
    return IsolationForest(n_estimators=100, contamination='auto', random_state=42)

def get_random_forest():
    return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
