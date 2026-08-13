#!/usr/bin/env python3
"""Variant A v2: per-class calibrated prototype score.
Score = min prototype distance (higher = more unknown).
Validates itself against the stored base AUROC before trusting output."""
import sys, json, numpy as np, torch
from src.data.feature_mapping import get_dataloaders
from src.models.edgeproto import get_model
from src.evaluation.evaluate_edgeproto_hybrid import load_config, compute_prototype_distances
from sklearn.metrics import roc_auc_score

exp = sys.argv[1] if len(sys.argv) > 1 else 'h5_dos'
seed = sys.argv[2] if len(sys.argv) > 2 else '2025'
device = torch.device('cpu')
config = load_config(f'configs/{exp}.yaml')
tr_loader, val_loader, tk_loader, tu_loader, label_map = get_dataloaders(
    f'data/splits/{exp}_train.csv', f'data/splits/{exp}_val.csv',
    f'data/splits/{exp}_test_known.csv', f'data/splits/{exp}_test_unknown.csv', batch_size=256)
model = get_model(config, num_classes=len(label_map)).to(device)
model.load_state_dict(torch.load(f'models/ciciot50k_{exp}_base/best_edgeproto.pt', map_location=device))
model.eval()

embs, labs = [], []
with torch.no_grad():
    for x, y in tr_loader:
        embs.append(model.encoder(x)); labs.append(y)
protos = model.compute_prototypes(torch.cat(embs), torch.cat(labs), len(label_map))

def dists(loader):
    d = []
    with torch.no_grad():
        for x, _ in loader:
            _, dd = model(x.to(device), prototypes=protos)
            d.append(dd.cpu().numpy())
    return np.concatenate(d)

Dv = dists(val_loader); Dk = dists(tk_loader); Du = dists(tu_loader)
mk = Dk.min(1); mu = Du.min(1)
y = np.r_[np.zeros(len(mk)), np.ones(len(mu))]
au_global = roc_auc_score(y, np.r_[mk, mu])

try:
    ref = json.load(open(f'results/tables/ciciot2023_50k/seeds/{exp}_s{seed}_base.json'))
    ref_au = ref.get('AUROC') or ref.get('open_set_auroc')
    print(f'  [check] stored={ref_au:.4f} computed_global={au_global:.4f} (must match!)')
except Exception as e:
    print('  [check] no stored ref:', e)

pred_k = Dk.argmin(1); pred_u = Du.argmin(1)
means, stds = {}, {}
for c in range(len(label_map)):
    vals = Dv[Dv.argmin(1) == c].min(1)
    means[c] = vals.mean() if len(vals) else 0.0
    stds[c] = vals.std() + 1e-8 if len(vals) else 1e-8
pk = (mk - np.array([means[c] for c in pred_k])) / np.array([stds[c] for c in pred_k])
pu = (mu - np.array([means[c] for c in pred_u])) / np.array([stds[c] for c in pred_u])
au_perclass = roc_auc_score(y, np.r_[pk, pu])

print(f'{exp} seed {seed}:')
print(f'  global min-dist AUROC : {au_global:.4f}')
print(f'  per-class z AUROC     : {au_perclass:.4f}')
