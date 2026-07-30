import os
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from src.baselines.rtiot_baseline_models import MLP_MSP, AutoencoderKnown, get_random_forest, get_isolation_forest
from src.evaluation.open_set_metrics_extended import binary_open_set_metrics, ensure_unknown_score_direction

# Set seed
torch.manual_seed(42)
np.random.seed(42)

def load_split(prefix, split):
    path = f"{prefix}_{split}.csv"
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    X = df.drop(columns=['label']).values.astype(np.float32)
    y = df['label'].values
    return X, y

def save_result(prefix, baseline, experiment,
                knownf1macro, metrics,
                params, rawweightkb, size_kb, train_time, infer_time, score_inverted, known_scores, unknown_scores):
    os.makedirs("results/tables/extended", exist_ok=True)
    os.makedirs("results/scores", exist_ok=True)
    
    res = {
        "experiment": experiment,
        "model": baseline,
        "model_family": 'classifier' if baseline in ['MLP-MSP', 'RandomForest-MSP'] else 'anomaly',
        "knownf1macro": knownf1macro,
        "params_numeric": params,
        "rawweightkb_fp32": rawweightkb,
        "savedmodelsize_kb": size_kb,
        "traintimeseconds": train_time,
        "inferencetime1000samplesseconds": infer_time,
        "scorewasinverted": score_inverted,
        "OSCR": None,
        "notes": "Baseline run"
    }
    res.update(metrics)
    
    baseline_slug = ""
    if baseline == "MLP-MSP": baseline_slug = "mlp"
    elif baseline == "RandomForest-MSP": baseline_slug = "rf"
    elif baseline == "IsolationForest": baseline_slug = "iforest"
    elif baseline == "Autoencoder-Known": baseline_slug = "ae"
    
    path = f"results/tables/extended/{experiment}_{baseline_slug}_extended.json"
    with open(path, "w") as f:
        json.dump(res, f, indent=4)
        
    out_npz = f"results/scores/{experiment}_{baseline_slug}_scores.npz"
    np.savez_compressed(out_npz, known_scores=known_scores, unknown_scores=unknown_scores)

def run_mlp(X_train, y_train, X_test_known, y_test_known, X_test_unknown, le, prefix, experiment):
    print("Running MLP-MSP...")
    num_classes = len(le.classes_)
    y_train_idx = le.transform(y_train)
    y_test_known_idx = le.transform(y_test_known)
    
    class_counts = np.bincount(y_train_idx)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    model = MLP_MSP(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train_idx))
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    
    start_time = time.time()
    model.train()
    for epoch in range(30):
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            out = model(X_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()
    train_time = time.time() - start_time
    
    model.eval()
    with torch.no_grad():
        out_known = model(torch.tensor(X_test_known))
        probs_known = torch.softmax(out_known, dim=1).numpy()
        preds_known = np.argmax(probs_known, axis=1)
        
        infer_start = time.time()
        model(torch.tensor(X_test_known[:1000]))
        infer_time = time.time() - infer_start
        
        out_unknown = model(torch.tensor(X_test_unknown))
        probs_unknown = torch.softmax(out_unknown, dim=1).numpy()
        
    prec, rec, f1, sup = precision_recall_fscore_support(y_test_known_idx, preds_known, average='macro', zero_division=0)
    
    scores_known = 1.0 - np.max(probs_known, axis=1)
    scores_unknown = 1.0 - np.max(probs_unknown, axis=1)
    
    known_adj, unknown_adj, inverted = ensure_unknown_score_direction(scores_known, scores_unknown, "MLP-MSP")
    y_true_unknown = [0]*len(known_adj) + [1]*len(unknown_adj)
    metrics = binary_open_set_metrics(y_true_unknown, known_adj + unknown_adj)
    
    params = sum(p.numel() for p in model.parameters())
    rawweightkb = (params * 4.0) / 1024.0
    
    # Save a temporary model to get saved size
    torch.save(model.state_dict(), 'temp_mlp.pt')
    saved_size = os.path.getsize('temp_mlp.pt') / 1024.0
    os.remove('temp_mlp.pt')
    
    save_result(prefix, "MLP-MSP", experiment,
                float(f1), metrics,
                params, rawweightkb, saved_size, train_time, infer_time, inverted, known_adj, unknown_adj)

def run_rf(X_train, y_train, X_test_known, y_test_known, X_test_unknown, le, prefix, experiment):
    print("Running RandomForest-MSP...")
    model = get_random_forest()
    y_train_idx = le.transform(y_train)
    y_test_known_idx = le.transform(y_test_known)
    
    start_time = time.time()
    model.fit(X_train, y_train_idx)
    train_time = time.time() - start_time
    
    probs_known = model.predict_proba(X_test_known)
    preds_known = np.argmax(probs_known, axis=1)
    
    infer_start = time.time()
    model.predict_proba(X_test_known[:1000])
    infer_time = time.time() - infer_start
    
    probs_unknown = model.predict_proba(X_test_unknown)
    
    prec, rec, f1, sup = precision_recall_fscore_support(y_test_known_idx, preds_known, average='macro', zero_division=0)
    
    scores_known = 1.0 - np.max(probs_known, axis=1)
    scores_unknown = 1.0 - np.max(probs_unknown, axis=1)
    
    known_adj, unknown_adj, inverted = ensure_unknown_score_direction(scores_known, scores_unknown, "RandomForest-MSP")
    y_true_unknown = [0]*len(known_adj) + [1]*len(unknown_adj)
    metrics = binary_open_set_metrics(y_true_unknown, known_adj + unknown_adj)
    
    import pickle
    model_size = len(pickle.dumps(model)) / 1024.0
    params = sum(tree.tree_.node_count for tree in model.estimators_)
    
    save_result(prefix, "RandomForest-MSP", experiment,
                float(f1), metrics,
                params, None, model_size, train_time, infer_time, inverted, known_adj, unknown_adj)

def run_if(X_train, X_test_known, X_test_unknown, prefix, experiment):
    print("Running IsolationForest...")
    model = get_isolation_forest(len(X_train))
    
    start_time = time.time()
    model.fit(X_train)
    train_time = time.time() - start_time
    
    infer_start = time.time()
    model.decision_function(X_test_known[:1000])
    infer_time = time.time() - infer_start
    
    # In IF, lower is anomalous, so we negate
    scores_known = -model.decision_function(X_test_known)
    scores_unknown = -model.decision_function(X_test_unknown)
    
    known_adj, unknown_adj, inverted = ensure_unknown_score_direction(scores_known, scores_unknown, "IsolationForest")
    y_true_unknown = [0]*len(known_adj) + [1]*len(unknown_adj)
    metrics = binary_open_set_metrics(y_true_unknown, known_adj + unknown_adj)
    
    import pickle
    model_size = len(pickle.dumps(model)) / 1024.0
    
    save_result(prefix, "IsolationForest", experiment,
                None, metrics,
                None, None, model_size, train_time, infer_time, inverted, known_adj, unknown_adj)

def run_ae(X_train, X_test_known, X_test_unknown, prefix, experiment):
    print("Running Autoencoder-Known...")
    model = AutoencoderKnown()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(X_train))
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    
    start_time = time.time()
    model.train()
    for epoch in range(30):
        for X_b, _ in train_loader:
            optimizer.zero_grad()
            out = model(X_b)
            loss = criterion(out, X_b)
            loss.backward()
            optimizer.step()
    train_time = time.time() - start_time
    
    model.eval()
    with torch.no_grad():
        x_k = torch.tensor(X_test_known)
        out_k = model(x_k)
        scores_known = torch.mean((x_k - out_k)**2, dim=1).numpy()
        
        infer_start = time.time()
        model(torch.tensor(X_test_known[:1000]))
        infer_time = time.time() - infer_start
        
        x_u = torch.tensor(X_test_unknown)
        out_u = model(x_u)
        scores_unknown = torch.mean((x_u - out_u)**2, dim=1).numpy()
        
    known_adj, unknown_adj, inverted = ensure_unknown_score_direction(scores_known, scores_unknown, "Autoencoder-Known")
    y_true_unknown = [0]*len(known_adj) + [1]*len(unknown_adj)
    metrics = binary_open_set_metrics(y_true_unknown, known_adj + unknown_adj)
    
    params = sum(p.numel() for p in model.parameters())
    rawweightkb = (params * 4.0) / 1024.0
    
    torch.save(model.state_dict(), 'temp_ae.pt')
    saved_size = os.path.getsize('temp_ae.pt') / 1024.0
    os.remove('temp_ae.pt')
    
    save_result(prefix, "Autoencoder-Known", experiment,
                None, metrics,
                params, rawweightkb, saved_size, train_time, infer_time, inverted, known_adj, unknown_adj)

def evaluate_baselines_extended(prefix, experiment):
    X_train, y_train = load_split(prefix, "train")
    X_test_known, y_test_known = load_split(prefix, "test_known")
    X_test_unknown, y_test_unknown = load_split(prefix, "test_unknown")
    
    le = LabelEncoder()
    le.fit(y_train)
    
    run_mlp(X_train, y_train, X_test_known, y_test_known, X_test_unknown, le, prefix, experiment)
    run_rf(X_train, y_train, X_test_known, y_test_known, X_test_unknown, le, prefix, experiment)
    run_if(X_train, X_test_known, X_test_unknown, prefix, experiment)
    run_ae(X_train, X_test_known, X_test_unknown, prefix, experiment)

if __name__ == '__main__':
    evaluate_baselines_extended("data/splits/rt_h1_arp_spoofing", "RT-H1")
    evaluate_baselines_extended("data/splits/rt_h2_recon", "RT-H2")
    evaluate_baselines_extended("data/splits/rt_h3_dos", "RT-H3")
