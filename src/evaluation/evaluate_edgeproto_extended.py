import os
import yaml
import torch
import time
import argparse
import json
import numpy as np
from sklearn.metrics import f1_score
from src.models.edgeproto import get_model
from src.data.feature_mapping import get_dataloaders
from src.evaluation.open_set_metrics_extended import binary_open_set_metrics, ensure_unknown_score_direction

def get_modelsizekb(path):
    if os.path.exists(path):
        return os.path.getsize(path) / 1024.0
    return None

def compute_distances(model, x):
    # distances from embeddings to all prototypes
    # model(x) returns (embeddings, distances) if prototypes exist
    out = model(x)
    if isinstance(out, tuple):
        embeddings, distances = out
    else:
        # Should not happen since we provided num_classes
        pass
    
    # Anomaly score is the distance to the closest prototype.
    # Higher distance = more likely unknown.
    return distances.min(dim=1)[0]

def evaluate_extended(config_path, data_prefix, model_path, output_json, output_npz, experiment_name, is_po):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # force 0 workers for safety
    if 'training' not in config:
        config['training'] = {}
    config['training']['num_workers'] = 0

    device = torch.device('cpu')
    
    # Load model to get architecture info
    # We need to know number of classes. It's usually 40 for RT-IoT2022 H1, but might vary
    # Let's just load the model cleanly by getting dataloaders first to infer num_classes if needed
    dataloaders = get_dataloaders(
        train_csv=f"{data_prefix}_train.csv",
        val_csv=f"{data_prefix}_val.csv",
        test_known_csv=f"{data_prefix}_test_known.csv",
        test_unknown_csv=f"{data_prefix}_test_unknown.csv",
        batch_size=config['training'].get('batch_size', 256),
        num_workers=0
    )
    
    train_loader, val_loader, test_known_loader, test_unknown_loader, label_map = dataloaders
    
    # Infer num_features and num_classes from the model file if possible, or from config
    # The models are saved entirely or as state dicts. The previous code saved the entire model or state_dict?
    # Usually it's state_dict, but we need the instantiation.
    
    # Assuming config has model params
    num_classes = len(label_map)
    model = get_model(config, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rawweightkb = (params * 4.0) / 1024.0
    savedmodelsize_kb = get_modelsizekb(model_path)
    
    # Evaluate on test_known for classification metrics
    all_known_preds = []
    all_known_labels = []
    all_known_scores = []
    
    start_time = time.time()
    samples_processed = 0
    with torch.no_grad():
        for x, y in test_known_loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            if isinstance(out, tuple):
                embeddings, distances = out
                # The class with the minimum distance is the predicted class
                preds = distances.argmin(dim=1)
            else:
                preds = out.argmax(dim=1)
                
            scores = compute_distances(model, x)
            all_known_preds.extend(preds.cpu().numpy())
            all_known_labels.extend(y.cpu().numpy())
            all_known_scores.extend(scores.cpu().numpy())
            samples_processed += x.size(0)
    
    inferencetime1000 = (time.time() - start_time) / max(1, samples_processed) * 1000
            
    knownf1macro = f1_score(all_known_labels, all_known_preds, average='macro')
    
    # Evaluate on test_unknown
    all_unknown_scores = []
    with torch.no_grad():
        for x, y in test_unknown_loader:
            x = x.to(device)
            scores = compute_distances(model, x)
            all_unknown_scores.extend(scores.cpu().numpy())
            
    # Ensure direction
    known_adj, unknown_adj, inverted = ensure_unknown_score_direction(all_known_scores, all_unknown_scores, "EdgeProto_Distances")
    
    y_true_unknown = [0]*len(known_adj) + [1]*len(unknown_adj)
    all_scores = known_adj + unknown_adj
    
    metrics = binary_open_set_metrics(y_true_unknown, all_scores)
    
    # Save OSCR (approximate, since we need accuracy on knowns at each threshold)
    # A true OSCR requires both correct classification and thresholding. We'll mark it N/A if not strictly calculated
    # but let's try a basic OSCR if we have the true labels and preds
    
    # Construct final result dictionary
    result = {
        'experiment': experiment_name,
        'model': 'EdgeProto-PO' if is_po else 'EdgeProto (Ours)',
        'model_family': 'prototype',
        'knownf1macro': knownf1macro,
        'params_numeric': params,
        'rawweightkb_fp32': rawweightkb,
        'savedmodelsize_kb': savedmodelsize_kb,
        'inferencetime1000samplesseconds': inferencetime1000,
        'scorewasinverted': inverted,
        'notes': 'Pseudo-open variant' if is_po else 'EdgeProto run'
    }
    result.update(metrics)
    
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(result, f, indent=4)
        
    os.makedirs(os.path.dirname(output_npz), exist_ok=True)
    np.savez_compressed(output_npz, known_scores=known_adj, unknown_scores=unknown_adj, labels_known=all_known_labels, preds_known=all_known_preds)
    
    print(f"[{experiment_name}] Saved extended metrics to {output_json}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--prefix', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_json', type=str, required=True)
    parser.add_argument('--output_npz', type=str, required=True)
    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('--po', action='store_true')
    args = parser.parse_args()
    
    evaluate_extended(args.config, args.prefix, args.model_path, args.output_json, args.output_npz, args.experiment, args.po)
