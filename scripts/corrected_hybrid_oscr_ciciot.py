#!/usr/bin/env python3
"""CICIoT2023: corrected hybrid OSCR computed from FUSED scores (audit method)."""
import sys, numpy as np, torch, csv
from src.data.feature_mapping import get_dataloaders
from src.models.edgeproto import get_model
from src.evaluation.oscr import compute_oscr
from src.evaluation.evaluate_edgeproto_hybrid import load_config, compute_prototype_distances

DEFAULT = ['h1_mirai','h2_recon','h3_spoofing','h4_web_bruteforce','h5_dos']
HOLDOUTS = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT
RULES = ['Hybrid-Mean-50-50','Hybrid-Max','Hybrid-Proto-0.75','Hybrid-AE-0.75']
device = torch.device('cpu')
CSV = 'results/tables/ciciot2023_10k/hybrid_oscr_corrected.csv'

print(f"{'exp':<20}{'fusion':<22}{'OSCR_corrected':>16}")
with open(CSV, 'a', newline='') as f:
    w = csv.writer(f)
    for exp in HOLDOUTS:
        config = load_config(f'configs/{exp}.yaml')
        train_loader, _, tk_loader, _, label_map = get_dataloaders(
            f'data/splits/{exp}_train.csv', f'data/splits/{exp}_val.csv',
            f'data/splits/{exp}_test_known.csv', f'data/splits/{exp}_test_unknown.csv',
            batch_size=256)
        model = get_model(config, num_classes=len(label_map)).to(device)
        model.load_state_dict(torch.load(f'models/ciciot10k_{exp}_base/best_edgeproto.pt', map_location=device))
        model.eval()
        embs, labs = [], []
        with torch.no_grad():
            for x, y in train_loader:
                embs.append(model.encoder(x)); labs.append(y)
        protos = model.compute_prototypes(torch.cat(embs), torch.cat(labs), len(label_map))
        _, preds_tk, labels_tk, _ = compute_prototype_distances(model, tk_loader, device, protos)
        correct_k = (preds_tk == labels_tk)
        for rule in RULES:
            d = np.load(f'results/scores/hybrid/{exp}_{rule}_scores.npz')
            oscr = float(compute_oscr(-d['score_k'], -d['score_u'], correct_k))
            print(f"{exp:<20}{rule:<22}{oscr:>16.4f}")
            w.writerow([exp, rule, f'{oscr:.4f}'])
