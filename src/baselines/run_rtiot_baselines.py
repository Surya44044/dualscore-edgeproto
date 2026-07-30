import argparse
import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from src.baselines.rtiot_baseline_models import MLP_MSP, AutoencoderKnown, get_random_forest, get_isolation_forest

# Set seed
torch.manual_seed(42)
np.random.seed(42)

def load_split(prefix, split):
    path = f"data/splits/{prefix}_{split}.csv"
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    X = df.drop(columns=['label']).values.astype(np.float32)
    y = df['label'].values
    return X, y

def save_result(prefix, baseline, known_classes, unknown_class, seed,
                acc, prec, rec, f1, auroc, oscr,
                params, size_kb, train_time, infer_time, per_class=None):
    os.makedirs("results/tables/baselines", exist_ok=True)
    res = {
        "dataset": "rtiot2022",
        "prefix": prefix,
        "baseline": baseline,
        "known_classes": ", ".join(known_classes),
        "unknown_class": unknown_class,
        "seed": seed,
        "known_accuracy": acc,
        "knownprecisionmacro": prec,
        "knownrecallmacro": rec,
        "knownf1macro": f1,
        "per_class": per_class,
        "AUROC": auroc,
        "OSCR": oscr,
        "params": params,
        "modelsizekb": size_kb,
        "traintimeseconds": train_time,
        "inferencetime1000samplesseconds": infer_time,
        "unknownscoredirection explanation": "higher score = unknown",
        "status": "success",
        "error": None
    }
    path = f"results/tables/baselines/{prefix}_{baseline}_results.json"
    with open(path, "w") as f:
        json.dump(res, f, indent=4)
        
def save_error(prefix, baseline, known_classes, unknown_class, error_msg):
    os.makedirs("results/tables/baselines", exist_ok=True)
    res = {
        "dataset": "rtiot2022",
        "prefix": prefix,
        "baseline": baseline,
        "known_classes": ", ".join(known_classes),
        "unknown_class": unknown_class,
        "status": "failed",
        "error": error_msg
    }
    path = f"results/tables/baselines/{prefix}_{baseline}_results.json"
    with open(path, "w") as f:
        json.dump(res, f, indent=4)

def run_mlp(X_train, y_train, X_test_known, y_test_known, X_test_unknown, le, prefix, known_classes, unknown_class):
    try:
        print("Running MLP-MSP...")
        num_classes = len(le.classes_)
        y_train_idx = le.transform(y_train)
        y_test_known_idx = le.transform(y_test_known)
        
        # Class weights
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
            
            # infer time on 1000
            infer_start = time.time()
            model(torch.tensor(X_test_known[:1000]))
            infer_time = time.time() - infer_start
            
            out_unknown = model(torch.tensor(X_test_unknown))
            probs_unknown = torch.softmax(out_unknown, dim=1).numpy()
            
        acc = float(accuracy_score(y_test_known_idx, preds_known))
        prec, rec, f1, sup = precision_recall_fscore_support(y_test_known_idx, preds_known, average='macro', zero_division=0)
        
        # Unknown score: 1 - max_softmax_prob
        scores_known = 1.0 - np.max(probs_known, axis=1)
        scores_unknown = 1.0 - np.max(probs_unknown, axis=1)
        
        y_true_binary = np.concatenate([np.zeros(len(scores_known)), np.ones(len(scores_unknown))])
        y_scores = np.concatenate([scores_known, scores_unknown])
        auroc = float(roc_auc_score(y_true_binary, y_scores))
        
        params = sum(p.numel() for p in model.parameters())
        model_size = sum(p.element_size() * p.numel() for p in model.parameters()) / 1024.0
        
        save_result(prefix, "MLP-MSP", known_classes, unknown_class, 42,
                    acc, float(prec), float(rec), float(f1), auroc, None,
                    params, model_size, train_time, infer_time)
        print("MLP-MSP complete.")
    except Exception as e:
        save_error(prefix, "MLP-MSP", known_classes, unknown_class, str(e))
        print(f"MLP-MSP failed: {e}")

def run_rf(X_train, y_train, X_test_known, y_test_known, X_test_unknown, le, prefix, known_classes, unknown_class):
    try:
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
        
        acc = float(accuracy_score(y_test_known_idx, preds_known))
        prec, rec, f1, sup = precision_recall_fscore_support(y_test_known_idx, preds_known, average='macro', zero_division=0)
        
        scores_known = 1.0 - np.max(probs_known, axis=1)
        scores_unknown = 1.0 - np.max(probs_unknown, axis=1)
        
        y_true_binary = np.concatenate([np.zeros(len(scores_known)), np.ones(len(scores_unknown))])
        y_scores = np.concatenate([scores_known, scores_unknown])
        auroc = float(roc_auc_score(y_true_binary, y_scores))
        
        # Estimate size using pickle
        import pickle
        model_size = len(pickle.dumps(model)) / 1024.0
        
        save_result(prefix, "RandomForest-MSP", known_classes, unknown_class, 42,
                    acc, float(prec), float(rec), float(f1), auroc, None,
                    None, model_size, train_time, infer_time)
        print("RandomForest-MSP complete.")
    except Exception as e:
        save_error(prefix, "RandomForest-MSP", known_classes, unknown_class, str(e))
        print(f"RF failed: {e}")

def run_if(X_train, X_test_known, X_test_unknown, prefix, known_classes, unknown_class):
    try:
        print("Running IsolationForest...")
        model = get_isolation_forest(len(X_train))
        
        start_time = time.time()
        model.fit(X_train)
        train_time = time.time() - start_time
        
        infer_start = time.time()
        model.decision_function(X_test_known[:1000])
        infer_time = time.time() - infer_start
        
        # Unknown score: -decision_function
        scores_known = -model.decision_function(X_test_known)
        scores_unknown = -model.decision_function(X_test_unknown)
        
        y_true_binary = np.concatenate([np.zeros(len(scores_known)), np.ones(len(scores_unknown))])
        y_scores = np.concatenate([scores_known, scores_unknown])
        auroc = float(roc_auc_score(y_true_binary, y_scores))
        
        import pickle
        model_size = len(pickle.dumps(model)) / 1024.0
        
        save_result(prefix, "IsolationForest", known_classes, unknown_class, 42,
                    None, None, None, None, auroc, None,
                    None, model_size, train_time, infer_time)
        print("IsolationForest complete.")
    except Exception as e:
        save_error(prefix, "IsolationForest", known_classes, unknown_class, str(e))
        print(f"IF failed: {e}")

def run_ae(X_train, X_test_known, X_test_unknown, prefix, known_classes, unknown_class):
    try:
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
            
        y_true_binary = np.concatenate([np.zeros(len(scores_known)), np.ones(len(scores_unknown))])
        y_scores = np.concatenate([scores_known, scores_unknown])
        auroc = float(roc_auc_score(y_true_binary, y_scores))
        
        params = sum(p.numel() for p in model.parameters())
        model_size = sum(p.element_size() * p.numel() for p in model.parameters()) / 1024.0
        
        save_result(prefix, "Autoencoder-Known", known_classes, unknown_class, 42,
                    None, None, None, None, auroc, None,
                    params, model_size, train_time, infer_time)
        print("Autoencoder complete.")
    except Exception as e:
        save_error(prefix, "Autoencoder-Known", known_classes, unknown_class, str(e))
        print(f"AE failed: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--unknown", required=True)
    parser.add_argument("--known", nargs='+', required=True)
    args = parser.parse_args()
    
    print(f"=== Running baselines for {args.prefix} ===")
    
    X_train, y_train = load_split(args.prefix, "train")
    X_test_known, y_test_known = load_split(args.prefix, "test_known")
    X_test_unknown, y_test_unknown = load_split(args.prefix, "test_unknown")
    
    if X_train is None:
        print(f"Splits for {args.prefix} not found.")
        return
        
    le = LabelEncoder()
    le.fit(y_train)
    
    run_mlp(X_train, y_train, X_test_known, y_test_known, X_test_unknown, le, args.prefix, args.known, args.unknown)
    run_rf(X_train, y_train, X_test_known, y_test_known, X_test_unknown, le, args.prefix, args.known, args.unknown)
    run_if(X_train, X_test_known, X_test_unknown, args.prefix, args.known, args.unknown)
    run_ae(X_train, X_test_known, X_test_unknown, args.prefix, args.known, args.unknown)

if __name__ == "__main__":
    main()
