#!/usr/bin/env python3
"""Corrected OSCR for ONE experiment+seed, appended with seed tag (run right after hybrid)."""
import sys, os, numpy as np, torch, csv
from src.data.feature_mapping import get_dataloaders
from src.models.edgeproto import get_model
from src.evaluation.oscr import compute_oscr
from src.evaluation.evaluate_edgeproto_hybrid import load_config, compute_prototype_distances

exp, seed = sys.argv[1], sys.argv[2]
RULES = ['Hybrid-Mean-50-50','Hybrid-Max','Hybrid-Proto-0.75','Hybrid-AE-0.75']
device = torch.device('cpu')

need = [
    f'configs/{exp}.yaml',
    f'data/splits/{exp}_train.csv', f'data/splits/{exp}_val.csv',
    f'data/splits/{exp}_test_known.csv', f'data/splits/{exp}_test_unknown.csv',
    f'models/oe_{exp}_base/best_edgeproto.pt',
]
for p in need:
    if not os.path.exists(p):
        print(f'ERROR missing {p}'); sys.exit(2)
for rule in RULES:
    if not os.path.exists(f'results/scores/hybrid/{exp}_{rule}_scores.npz'):
        print(f'ERROR missing npz for {rule}'); sys.exit(2)

config = load_config(f'configs/{exp}.yaml')
train_loader, _, tk_loader, _, label_map = get_dataloaders(
    f'data/splits/{exp}_train.csv', f'data/splits/{exp}_val.csv',
    f'data/splits/{exp}_test_known.csv', f'data/splits/{exp}_test_unknown.csv', batch_size=256)
model = get_model(config, num_classes=len(label_map)).to(device)
model.load_state_dict(torch.load(f'models/oe_{exp}_base/best_edgeproto.pt', map_location=device))
model.eval()
embs, labs = [], []
with torch.no_grad():
    for x, y in train_loader:
        embs.append(model.encoder(x)); labs.append(y)
protos = model.compute_prototypes(torch.cat(embs), torch.cat(labs), len(label_map))
_, preds_tk, labels_tk, _ = compute_prototype_distances(model, tk_loader, device, protos)
correct_k = (preds_tk == labels_tk)
with open('results/tables/ciciot2023_50k/oe_oscr.csv', 'a', newline='') as f:
    w = csv.writer(f)
    for rule in RULES:
        d = np.load(f'results/scores/hybrid/{exp}_{rule}_scores.npz')
        o = float(compute_oscr(-d['score_k'], -d['score_u'], correct_k))
        w.writerow([exp, seed, rule, f'{o:.4f}'])
print(f'{exp} seed {seed} corrected OSCR done')
