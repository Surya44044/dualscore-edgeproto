import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import yaml
import argparse
import os
import json
import time
from torch.utils.data import DataLoader

from src.data.feature_mapping import get_dataloaders, IoTDataset
from src.models.edgeproto import get_model
from src.evaluation.metrics import calculate_metrics
from src.evaluation.oscr import compute_oscr
from src.evaluation.open_set_metrics_extended import binary_open_set_metrics

class SmallAutoencoder(nn.Module):
    def __init__(self, input_dim=48):
        super().__init__()
        # Topology: 48 -> 64 -> 32 -> 16 -> 32 -> 64 -> 48
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        if x.ndim == 3:
            x = x.view(x.size(0), -1)
        code = self.encoder(x)
        recon = self.decoder(code)
        return recon

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_or_load_ae(train_loader, ae_model_path, device, input_dim=48, epochs=30, lr=0.001):
    os.makedirs(os.path.dirname(ae_model_path), exist_ok=True)
    ae = SmallAutoencoder(input_dim=input_dim).to(device)
    
    if os.path.exists(ae_model_path):
        print(f"Loading pre-trained Autoencoder from {ae_model_path}")
        ae.load_state_dict(torch.load(ae_model_path, map_location=device))
    else:
        print(f"Training Autoencoder for {epochs} epochs on known training data...")
        ae.train()
        optimizer = torch.optim.Adam(ae.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            count = 0
            for x, _ in train_loader:
                x = x.to(device)
                x_flat = x.view(x.size(0), -1)
                recon = ae(x_flat)
                loss = criterion(recon, x_flat)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * x.size(0)
                count += x.size(0)
                
            avg_loss = total_loss / max(count, 1)
            if epoch % 5 == 0 or epoch == epochs:
                print(f"AE Epoch {epoch}/{epochs} - Loss: {avg_loss:.6f}")
                
        torch.save(ae.state_dict(), ae_model_path)
        print(f"Saved Autoencoder to {ae_model_path}")
        
    ae.eval()
    return ae

def compute_prototype_distances(edgeproto_model, loader, device, prototypes):
    edgeproto_model.eval()
    all_dists = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            _, dists = edgeproto_model(x, prototypes=prototypes)
            all_dists.append(dists.cpu().numpy())
            all_labels.append(y.numpy())
            
    all_dists = np.concatenate(all_dists, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Prototype unknown score: distance to nearest prototype
    min_dists = np.min(all_dists, axis=1)
    preds = np.argmin(all_dists, axis=1)
    
    return min_dists, preds, all_labels, all_dists

def compute_ae_scores(ae, loader, device):
    ae.eval()
    all_mse = []
    
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            x_flat = x.view(x.size(0), -1)
            recon = ae(x_flat)
            mse = torch.mean((recon - x_flat) ** 2, dim=1).cpu().numpy()
            all_mse.append(mse)
            
    return np.concatenate(all_mse, axis=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--edgeproto_model", type=str, required=True)
    parser.add_argument("--unknown", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    data_prefix = f"data/splits/{args.prefix}"
    device = torch.device('cpu')

    train_csv = f"{data_prefix}_train.csv"
    val_csv = f"{data_prefix}_val.csv"
    test_known_csv = f"{data_prefix}_test_known.csv"
    test_unknown_csv = f"{data_prefix}_test_unknown.csv"

    train_loader, val_loader, test_known_loader, test_unknown_loader, label_map = get_dataloaders(
        train_csv, val_csv, test_known_csv, test_unknown_csv, batch_size=256, num_workers=0
    )

    num_classes = len(label_map)

    # Load EdgeProto
    edgeproto = get_model(config, num_classes=num_classes).to(device)
    edgeproto.load_state_dict(torch.load(args.edgeproto_model, map_location=device))
    edgeproto.eval()

    # Compute embeddings and prototypes on train set
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for x, y in train_loader:
            x = x.to(device)
            emb = edgeproto.encoder(x)
            all_embeddings.append(emb)
            all_labels.append(y)
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    prototypes = edgeproto.compute_prototypes(all_embeddings, all_labels, num_classes)

    # Train or load Autoencoder
    ae_path = f"models/hybrid_autoencoders/{args.prefix}_ae_known.pt"
    ae = train_or_load_ae(train_loader, ae_path, device)

    # Compute prototype unknown scores
    s_proto_val, _, _, _ = compute_prototype_distances(edgeproto, val_loader, device, prototypes)
    s_proto_tk, preds_tk, labels_tk, dists_tk = compute_prototype_distances(edgeproto, test_known_loader, device, prototypes)
    s_proto_tu, _, _, _ = compute_prototype_distances(edgeproto, test_unknown_loader, device, prototypes)

    # Compute AE unknown scores
    s_ae_val = compute_ae_scores(ae, val_loader, device)
    s_ae_tk = compute_ae_scores(ae, test_known_loader, device)
    s_ae_tu = compute_ae_scores(ae, test_unknown_loader, device)

    # Normalize using validation set known only
    mean_val_proto = np.mean(s_proto_val)
    std_val_proto = np.std(s_proto_val)
    if std_val_proto == 0:
        std_val_proto = 1e-8
        
    mean_val_ae = np.mean(s_ae_val)
    std_val_ae = np.std(s_ae_val)
    if std_val_ae == 0:
        std_val_ae = 1e-8

    z_proto_tk = (s_proto_tk - mean_val_proto) / std_val_proto
    z_proto_tu = (s_proto_tu - mean_val_proto) / std_val_proto

    z_ae_tk = (s_ae_tk - mean_val_ae) / std_val_ae
    z_ae_tu = (s_ae_tu - mean_val_ae) / std_val_ae

    # Fusion rules
    fusion_rules = {
        'Hybrid-Mean-50-50': lambda zp, zae: (zp + zae) / 2.0,
        'Hybrid-Max': lambda zp, zae: np.maximum(zp, zae),
        'Hybrid-Proto-0.75': lambda zp, zae: 0.75 * zp + 0.25 * zae,
        'Hybrid-AE-0.75': lambda zp, zae: 0.25 * zp + 0.75 * zae
    }

    # Known F1 Macro
    closed_metrics = calculate_metrics(labels_tk, preds_tk)
    known_f1_macro = float(closed_metrics['f1'])

    # OSCR prep: known scores (higher = more known = -s_proto_tk)
    max_conf_k = -s_proto_tk
    max_conf_u = -s_proto_tu
    correct_k = (preds_tk == labels_tk)
    oscr_val = float(compute_oscr(max_conf_k, max_conf_u, correct_k))

    # Parameter counts & Model Sizes
    edgeproto_params = sum(p.numel() for p in edgeproto.parameters() if p.requires_grad)
    ae_params = sum(p.numel() for p in ae.parameters() if p.requires_grad)
    params_numeric = edgeproto_params + ae_params
    rawweightkb_fp32 = float(params_numeric * 4 / 1024.0)

    edgeproto_filesize = os.path.getsize(args.edgeproto_model) / 1024.0
    ae_filesize = os.path.getsize(ae_path) / 1024.0
    savedmodelsize_kb = float(edgeproto_filesize + ae_filesize)

    # Benchmark inference time for 1000 samples
    dummy_x = torch.randn(1000, 4, 12, device=device)
    dummy_x_flat = dummy_x.view(1000, 48)
    # Warmup
    _ = edgeproto(dummy_x, prototypes=prototypes)
    _ = ae(dummy_x_flat)
    
    t0 = time.perf_counter()
    for _ in range(10):
        _ = edgeproto(dummy_x, prototypes=prototypes)
        _ = ae(dummy_x_flat)
    t1 = time.perf_counter()
    inf_time_1000 = float((t1 - t0) / 10.0)

    # Create directories
    os.makedirs("results/tables/hybrid", exist_ok=True)
    os.makedirs("results/scores/hybrid", exist_ok=True)

    for rule_name, rule_fn in fusion_rules.items():
        s_hybrid_tk = rule_fn(z_proto_tk, z_ae_tk)
        s_hybrid_tu = rule_fn(z_proto_tu, z_ae_tu)

        y_true_binary = np.concatenate([np.zeros(len(s_hybrid_tk)), np.ones(len(s_hybrid_tu))])
        y_score_binary = np.concatenate([s_hybrid_tk, s_hybrid_tu])

        # Compute extended open set metrics
        metrics = binary_open_set_metrics(y_true_binary, y_score_binary)

        result_data = {
            'prefix': args.prefix,
            'config name': os.path.splitext(os.path.basename(args.config))[0],
            'unknown_class': args.unknown,
            'model': rule_name,
            'knownf1macro': known_f1_macro,
            'AUROC': float(metrics['AUROC']) if metrics['AUROC'] is not None else 0.0,
            'AUPR_Out': float(metrics['AUPR_Out']) if metrics['AUPR_Out'] is not None else 0.0,
            'FPRat95_TPR': float(metrics['FPRat95_TPR']) if metrics['FPRat95_TPR'] is not None else 0.0,
            'TPRat1_FPR': float(metrics['TPRat1_FPR']) if metrics['TPRat1_FPR'] is not None else 0.0,
            'detection_error': float(metrics['detection_error']) if metrics['detection_error'] is not None else 0.0,
            'falsealarmsper10kat95tpr': float(metrics['falsealarmsper10kat95tpr']) if metrics['falsealarmsper10kat95tpr'] is not None else 0.0,
            'OSCR': oscr_val,
            'params_numeric': params_numeric,
            'rawweightkb_fp32': rawweightkb_fp32,
            'savedmodelsize_kb': savedmodelsize_kb,
            'inferencetime1000samplesseconds': inf_time_1000,
            'notes': f"Hybrid {rule_name} evaluation"
        }

        # Save JSON
        json_path = f"results/tables/hybrid/{args.prefix}_{rule_name}_results.json"
        with open(json_path, 'w') as f:
            json.dump(result_data, f, indent=4)
        print(f"Saved {json_path}")

        # Save score arrays
        npz_path = f"results/scores/hybrid/{args.prefix}_{rule_name}_scores.npz"
        np.savez(npz_path, y_true=y_true_binary, y_score=y_score_binary, score_k=s_hybrid_tk, score_u=s_hybrid_tu)
        print(f"Saved {npz_path}")

if __name__ == "__main__":
    main()
