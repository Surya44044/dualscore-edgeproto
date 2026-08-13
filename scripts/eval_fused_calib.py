#!/usr/bin/env python3
"""Fused global+per-class calibration: score = w*global_z + (1-w)*perclass_z."""
import sys, json, numpy as np, torch
from src.data.feature_mapping import get_dataloaders
from src.models.edgeproto import get_model
from src.evaluation.evaluate_edgeproto_hybrid import load_config, compute_prototype_distances
from sklearn.metrics import roc_auc_score

def run(exp, seed, w):
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
    mk, mu = Dk.min(1), Du.min(1)
    y = np.r_[np.zeros(len(mk)), np.ones(len(mu))]
    # global z
    m0, s0 = mk.mean(), mk.std() + 1e-8
    zg = np.r_[(mk - m0) / s0, (mu - m0) / s0]
    # per-class z
    pred_k, pred_u = Dk.argmin(1), Du.argmin(1)
    means, stds = {}, {}
    for c in range(len(label_map)):
        vals = Dv[Dv.argmin(1) == c].min(1)
        means[c] = vals.mean() if len(vals) else 0.0
        stds[c] = vals.std() + 1e-8 if len(vals) else 1e-8
    zp = np.r_[(mk - np.array([means[c] for c in pred_k])) / np.array([stds[c] for c in pred_k]),
               (mu - np.array([means[c] for c in pred_u])) / np.array([stds[c] for c in pred_u])]
    fused = w * zg + (1 - w) * zp
    return roc_auc_score(y, fused)

for exp in ['h5_dos','h1_mirai','h2_recon']:
    print(f'--- {exp} seed 2025 ---')
    for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
        au = run(exp, '2025', w)
        print(f'  w={w:.2f}: AUROC={au:.4f}')
