import torch
import torch.nn as nn
import torch.nn.functional as F

class DSConv1d(nn.Module):
    """Depthwise-Separable 1D Convolution"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, 
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class EdgeProtoEncoder(nn.Module):
    def __init__(self, in_channels=4, embedding_dim=32):
        super().__init__()
        # Input shape: [B, 4, 12]
        self.conv1 = DSConv1d(in_channels, 16, kernel_size=3, padding=1) # [B, 16, 12]
        self.pool1 = nn.MaxPool1d(2) # [B, 16, 6]
        
        self.conv2 = DSConv1d(16, 32, kernel_size=3, padding=1) # [B, 32, 6]
        self.pool2 = nn.MaxPool1d(2) # [B, 32, 3]
        
        self.flatten = nn.Flatten()
        
        self.fc = nn.Linear(32 * 3, embedding_dim)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc(x)
        # L2 normalization for the embedding space
        x = F.normalize(x, p=2, dim=1)
        return x

class EdgeProto(nn.Module):
    def __init__(self, in_channels=4, embedding_dim=32, num_classes=None):
        super().__init__()
        self.encoder = EdgeProtoEncoder(in_channels, embedding_dim)
        
        # We can explicitly store prototypes if needed, but in standard Prototypical Nets 
        # they are computed dynamically during training. We define a buffer for evaluation.
        if num_classes is not None:
            self.register_buffer("prototypes", torch.zeros(num_classes, embedding_dim))
            self.register_buffer("thresholds", torch.zeros(num_classes))
            
    def compute_prototypes(self, embeddings, labels, num_classes):
        """Compute mean prototypes for each class in the batch."""
        prototypes = torch.zeros(num_classes, embeddings.size(1), device=embeddings.device)
        for c in range(num_classes):
            idx = (labels == c).nonzero(as_tuple=True)[0]
            if len(idx) > 0:
                prototypes[c] = embeddings[idx].mean(dim=0)
        # Normalize prototypes
        prototypes = F.normalize(prototypes, p=2, dim=1)
        return prototypes
        
    def forward(self, x, prototypes=None):
        """
        If prototypes are not provided, it just returns embeddings.
        If prototypes are provided, returns pairwise euclidean distances.
        """
        embeddings = self.encoder(x)
        
        if prototypes is None:
            if hasattr(self, "prototypes") and self.prototypes.abs().sum() > 0:
                prototypes = self.prototypes
            else:
                return embeddings
                
        # Compute distances: ||z - c||^2
        # d_ij = ||z_i||^2 + ||c_j||^2 - 2 * z_i^T c_j
        # Since they are L2 normalized, ||z_i||^2 = 1 and ||c_j||^2 = 1
        # So d_ij = 2 - 2 * z_i^T c_j
        
        z_dot_c = torch.matmul(embeddings, prototypes.t())
        distances = 2.0 - 2.0 * z_dot_c
        
        # To prevent small negative numbers due to numerical instability
        distances = torch.clamp(distances, min=0.0)
        
        return embeddings, distances

def get_model(config, num_classes=None):
    in_channels = config['model'].get('in_channels', 4)
    embedding_dim = config['model'].get('embedding_dim', 32)
    return EdgeProto(in_channels=in_channels, embedding_dim=embedding_dim, num_classes=num_classes)
