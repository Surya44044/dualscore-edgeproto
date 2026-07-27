import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import yaml
import argparse
import os
import gc
import joblib
from src.data.feature_mapping import get_dataloaders
from src.models.baselines import Autoencoder, MLPClassifier, get_isolation_forest, get_random_forest

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_baselines(config_path, data_prefix, output_dir):
    config = load_config(config_path)
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cpu')
    
    train_csv = f"{data_prefix}_train.csv"
    val_csv = f"{data_prefix}_val.csv"
    
    # We also need pure DataFrames for sklearn
    df_train = pd.read_csv(train_csv)
    # df_val = pd.read_csv(val_csv)
    
    train_loader, val_loader, _, _, label_map = get_dataloaders(
        train_csv, val_csv, val_csv, val_csv, batch_size=config['training'].get('batch_size', 256)
    )
    
    num_classes = len(label_map)
    epochs = config['training'].get('epochs', 30)
    lr = config['training'].get('lr', 0.001)
    
    print("--- Training MLP Classifier ---")
    mlp = MLPClassifier(num_classes=num_classes).to(device)
    optimizer = optim.Adam(mlp.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_mlp_loss = float('inf')
    for epoch in range(epochs):
        mlp.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = mlp(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        mlp.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = mlp(x)
                loss = criterion(logits, y)
                val_loss += loss.item()
        
        if val_loss < best_mlp_loss:
            best_mlp_loss = val_loss
            torch.save(mlp.state_dict(), os.path.join(output_dir, "best_mlp.pt"))
            
    print("--- Training Autoencoder ---")
    ae = Autoencoder().to(device)
    optimizer = optim.Adam(ae.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_ae_loss = float('inf')
    for epoch in range(epochs):
        ae.train()
        train_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            x_flat = x.view(x.size(0), -1)
            optimizer.zero_grad()
            x_recon = ae(x)
            loss = criterion(x_recon, x_flat)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        ae.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                x_flat = x.view(x.size(0), -1)
                x_recon = ae(x)
                loss = criterion(x_recon, x_flat)
                val_loss += loss.item()
                
        if val_loss < best_ae_loss:
            best_ae_loss = val_loss
            torch.save(ae.state_dict(), os.path.join(output_dir, "best_ae.pt"))
            
    print("--- Training Random Forest ---")
    rf = get_random_forest()
    X_train = df_train.drop(columns=['label']).values
    y_train = df_train['label'].map(label_map).values
    rf.fit(X_train, y_train)
    joblib.dump(rf, os.path.join(output_dir, "rf_model.pkl"))
    
    print("--- Training Isolation Forest ---")
    iso = get_isolation_forest()
    iso.fit(X_train)
    joblib.dump(iso, os.path.join(output_dir, "iso_forest_model.pkl"))
    
    gc.collect()
    print("Baselines training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_prefix", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="models")
    args = parser.parse_args()
    
    train_baselines(args.config, args.data_prefix, args.output_dir)
