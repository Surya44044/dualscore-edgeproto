import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypeLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, distances, labels):
        """
        distances: [B, num_classes] (L2 squared distance)
        labels: [B]
        We want to minimize distance to true class, maximize to others.
        This is equivalent to CrossEntropy over negative distances.
        """
        logits = -distances / self.temperature
        return F.cross_entropy(logits, labels)

class SeparationLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, prototypes):
        """
        Push prototypes away from each other if they are closer than margin.
        prototypes: [num_classes, embedding_dim]
        """
        num_classes = prototypes.size(0)
        if num_classes < 2:
            return torch.tensor(0.0, device=prototypes.device)
            
        # compute pairwise distances
        z_dot_c = torch.matmul(prototypes, prototypes.t())
        dist = 2.0 - 2.0 * z_dot_c
        dist = torch.clamp(dist, min=0.0)
        
        loss = 0.0
        count = 0
        for i in range(num_classes):
            for j in range(i+1, num_classes):
                d = dist[i, j]
                if d < self.margin:
                    loss += (self.margin - d)
                count += 1
                
        if count > 0:
            loss /= count
            
        return loss
