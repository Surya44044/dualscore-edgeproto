import torch
import numpy as np
import pandas as pd
import yaml
import argparse
import os
import json
from src.data.feature_mapping import get_dataloaders
from src.models.edgeproto import get_model
from src.evaluation.metrics import calculate_metrics
from src.evaluation.oscr import compute_oscr

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_scores(model, loader, device, num_classes):
    model.eval()
    all_dists = []
    all_labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            _, dists = model(x)
            all_dists.append(dists.cpu().numpy())
            all_labels.append(y.numpy())
            
    all_dists = np.concatenate(all_dists, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Negative distance acts as a confidence score
    scores = -all_dists 
    return scores, all_labels

def evaluate(config_path, data_prefix, model_path, output_dir):
    config = load_config(config_path)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cpu')
    
    train_csv = f"{data_prefix}_train.csv"
    val_csv = f"{data_prefix}_val.csv"
    test_known_csv = f"{data_prefix}_test_known.csv"
    test_unknown_csv = f"{data_prefix}_test_unknown.csv"
    
    train_loader, val_loader, test_known_loader, test_unknown_loader, label_map = get_dataloaders(
        train_csv, val_csv, test_known_csv, test_unknown_csv, batch_size=256
    )
    
    num_classes = len(label_map)
    model = get_model(config, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    print("Computing evaluation scores...")
    val_scores, val_labels = get_scores(model, val_loader, device, num_classes)
    test_k_scores, test_k_labels = get_scores(model, test_known_loader, device, num_classes)
    
    has_unknown = os.path.exists(test_unknown_csv)
    if has_unknown:
        test_u_scores, _ = get_scores(model, test_unknown_loader, device, num_classes)
    
    # 1. Calibrate Thresholds on Validation Data
    # For each class, threshold is the 99th percentile of (negative) distance
    percentile = config['evaluation'].get('threshold_percentile', 99)
    thresholds = np.zeros(num_classes)
    for c in range(num_classes):
        c_scores = val_scores[val_labels == c, c]
        if len(c_scores) > 0:
            thresholds[c] = np.percentile(c_scores, 100 - percentile)
        else:
            thresholds[c] = -np.inf
            
    print("Thresholds:", thresholds)
    
    # 2. Closed-Set Evaluation
    preds_k = np.argmax(test_k_scores, axis=1)
    closed_set_metrics = calculate_metrics(test_k_labels, preds_k)
    print("Closed-Set Metrics:", closed_set_metrics)
    
    # 3. Open-Set Evaluation
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    dataset_name = config['data'].get('dataset', 'Unknown')
    
    results = {
        'dataset name': dataset_name,
        'config name': config_name,
        'known accuracy': float(closed_set_metrics['accuracy']),
        'known precision': float(closed_set_metrics['precision']),
        'known recall': float(closed_set_metrics['recall']),
        'known F1': float(closed_set_metrics['f1']),
        'per_class': closed_set_metrics['per_class'],
        'label_map': label_map
    }
    
    if has_unknown and len(test_u_scores) > 0:
        # Known vs Unknown detection
        # max_score across classes
        k_max_scores = np.max(test_k_scores, axis=1)
        u_max_scores = np.max(test_u_scores, axis=1)
        
        # Labels: 1 for Known, 0 for Unknown
        y_true_binary = np.concatenate([np.ones(len(k_max_scores)), np.zeros(len(u_max_scores))])
        y_scores_binary = np.concatenate([k_max_scores, u_max_scores])
        
        from sklearn.metrics import roc_auc_score
        auroc = roc_auc_score(y_true_binary, y_scores_binary)
        print(f"Open-Set Detection AUROC: {auroc:.4f}")
        
        # OSCR
        correct_k = (preds_k == test_k_labels)
        oscr = compute_oscr(k_max_scores, u_max_scores, correct_k)
        print(f"OSCR: {oscr:.4f}")
        
        results['AUROC'] = float(auroc)
        results['OSCR'] = float(oscr)
        
    out_file = os.path.join(output_dir, f"{config_name}_results.json")
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=4)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_prefix", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/tables")
    args = parser.parse_args()
    
    evaluate(args.config, args.data_prefix, args.model_path, args.output_dir)
