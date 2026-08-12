import torch
import torch.optim as optim
import yaml
import argparse
import os
import gc
import numpy as np
import random
from src.data.feature_mapping import get_dataloaders
from src.models.edgeproto import get_model
from src.training.losses import PrototypeLoss, SeparationLoss

def load_config(config_path):
    from pathlib import Path as _P
    def _resolve(path):
        with open(path, 'r') as f:
            d = yaml.safe_load(f) or {}
        inc = d.pop('include', None)
        if inc:
            base = _resolve(str(_P(path).parent / inc))
            for k, v in d.items():
                if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                    base[k].update(v)
                else:
                    base[k] = v
            return base
        return d
    return _resolve(config_path)

def train(config_path, data_prefix, output_dir, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    config = load_config(config_path)
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cpu') # Forcing CPU based on constraints
    
    train_csv = f"{data_prefix}_train.csv"
    val_csv = f"{data_prefix}_val.csv"
    test_known_csv = f"{data_prefix}_test_known.csv"
    test_unknown_csv = f"{data_prefix}_test_unknown.csv"
    
    batch_size = config['training'].get('batch_size', 256)
    epochs = config['training'].get('epochs', 30)
    lr = config['training'].get('lr', 0.001)
    
    train_loader, val_loader, _, _, label_map = get_dataloaders(
        train_csv, val_csv, test_known_csv, test_unknown_csv, batch_size=batch_size
    )
    
    num_classes = len(label_map)
    model = get_model(config, num_classes=num_classes).to(device)
    
    proto_criterion = PrototypeLoss(temperature=config['training'].get('temperature', 0.5))
    sep_criterion = SeparationLoss(margin=config['training'].get('margin_sep', 0.5))
    lambda_sep = config['training'].get('lambda_sep', 0.01)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config['training'].get('weight_decay', 1e-4))
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            # Step 1: Forward pass to get embeddings
            embeddings = model.encoder(x)
            
            # Step 2: Compute prototypes for this batch
            prototypes = model.compute_prototypes(embeddings, y, num_classes)
            
            # Step 3: Compute distances to batch prototypes
            z_dot_c = torch.matmul(embeddings, prototypes.t())
            distances = 2.0 - 2.0 * z_dot_c
            distances = torch.clamp(distances, min=0.0)
            
            # Step 4: Compute losses
            loss_p = proto_criterion(distances, y)
            loss_s = sep_criterion(prototypes)
            
            loss = loss_p + lambda_sep * loss_s
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            # In validation, we might want to use a running average of prototypes,
            # but for simplicity in this script we compute batch prototypes or accumulate.
            # A rigorous way is to compute prototypes on the whole train set after epoch.
            all_train_embeds = []
            all_train_labels = []
            for tx, ty in train_loader:
                tx = tx.to(device)
                all_train_embeds.append(model.encoder(tx))
                all_train_labels.append(ty)
            
            all_train_embeds = torch.cat(all_train_embeds)
            all_train_labels = torch.cat(all_train_labels)
            
            global_prototypes = model.compute_prototypes(all_train_embeds, all_train_labels, num_classes)
            model.prototypes.data.copy_(global_prototypes)
            
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                _, v_dist = model(vx, prototypes=global_prototypes)
                loss_p = proto_criterion(v_dist, vy)
                val_loss += loss_p.item()
                
            val_loss /= len(val_loader)
            
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "best_edgeproto.pt"))
            
        # Free memory explicitly
        del all_train_embeds, all_train_labels
        gc.collect()
        
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--data_prefix", type=str, required=True, help="Prefix for splits (e.g. data/splits/h1_mirai)")
    parser.add_argument("--output_dir", type=str, default="models", help="Output directory for weights")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    train(args.config, args.data_prefix, args.output_dir, args.seed)
