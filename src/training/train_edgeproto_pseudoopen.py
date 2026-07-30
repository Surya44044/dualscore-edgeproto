import torch
import torch.optim as optim
import yaml
import argparse
import os
import gc
import random
from src.data.feature_mapping import get_dataloaders
from src.models.edgeproto import get_model
from src.training.losses import PrototypeLoss, SeparationLoss

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train(config_path, data_prefix, output_dir):
    config = load_config(config_path)
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cpu') # CPU only
    
    train_csv = f"{data_prefix}_train.csv"
    val_csv = f"{data_prefix}_val.csv"
    test_known_csv = f"{data_prefix}_test_known.csv"
    test_unknown_csv = f"{data_prefix}_test_unknown.csv"
    
    batch_size = config['training'].get('batch_size', 256)
    epochs = config['training'].get('epochs', 30)
    lr = config['training'].get('lr', 0.001)
    
    # Pseudo-open specific hyperparams
    pseudo_open_enabled = config['training'].get('pseudo_open_enabled', True)
    margin_open = config['training'].get('margin_open', 1.0)
    lambda_open = config['training'].get('lambda_open', 0.5)
    lambda_sep = config['training'].get('lambda_sep', 0.01)
    
    train_loader, val_loader, _, _, label_map = get_dataloaders(
        train_csv, val_csv, test_known_csv, test_unknown_csv, batch_size=batch_size
    )
    
    num_classes = len(label_map)
    model = get_model(config, num_classes=num_classes).to(device)
    
    proto_criterion = PrototypeLoss(temperature=config['training'].get('temperature', 0.5))
    sep_criterion = SeparationLoss(margin=config['training'].get('margin_sep', 0.5))
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config['training'].get('weight_decay', 0.0001))
    
    best_loss = float('inf')
    
    # Seed for deterministic random pseudo-open class selection
    rng = random.Random(42)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            # Step 1: Compute embeddings
            embeddings = model.encoder(x)
            
            # Step 2: Compute normal prototypes over all labels in batch
            prototypes = model.compute_prototypes(embeddings, y, num_classes)
            
            # Step 3: Compute closed-set prototype classification loss
            z_dot_c = torch.matmul(embeddings, prototypes.t())
            distances = 2.0 - 2.0 * z_dot_c
            distances = torch.clamp(distances, min=0.0)
            loss_closed = proto_criterion(distances, y)
            
            # Step 4: Compute prototype separation loss
            loss_sep = sep_criterion(prototypes)
            
            loss_open = torch.tensor(0.0, device=device)
            
            # Step 5: Select one pseudo-open label
            unique_labels = torch.unique(y).tolist()
            if pseudo_open_enabled and len(unique_labels) >= 2:
                po_label = rng.choice(unique_labels)
                
                # Step 6: Compute active prototypes excluding the pseudo-open class
                active_classes = [c for c in range(num_classes) if c != po_label]
                active_prototypes = prototypes[active_classes] # [num_active, emb_dim]
                
                # Step 7: For embeddings of the pseudo-open class, compute nearest distance to active prototypes
                po_mask = (y == po_label)
                po_embeddings = embeddings[po_mask]
                
                if len(po_embeddings) > 0:
                    po_dot_c = torch.matmul(po_embeddings, active_prototypes.t())
                    po_dist = 2.0 - 2.0 * po_dot_c
                    po_dist = torch.clamp(po_dist, min=0.0)
                    
                    d_min, _ = torch.min(po_dist, dim=1) # [N_po]
                    
                    # Step 8: Pseudo-open loss
                    # mean(max(0, margin_open - d_min))
                    loss_open = torch.mean(torch.clamp(margin_open - d_min, min=0.0))
            
            # Step 9: Total loss
            loss = loss_closed + lambda_sep * loss_sep + lambda_open * loss_open
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
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
            torch.save(model.state_dict(), os.path.join(output_dir, "best_edgeproto_pseudoopen.pt"))
            
        del all_train_embeds, all_train_labels
        gc.collect()
        
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--data_prefix", type=str, required=True, help="Prefix for splits")
    parser.add_argument("--output_dir", type=str, default="models", help="Output directory")
    args = parser.parse_args()
    
    train(args.config, args.data_prefix, args.output_dir)
