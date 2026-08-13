#!/usr/bin/env python3
"""Adaptive calibration: per-input weight from prototype tightness.
score_i = a_i * global_z_i + (1-a_i) * perclass_z_i, a_i = sigmoid(k*(s_i - t)).
Tight class (small std) -> trust per-class; loose class -> trust global."""
import sys, numpy as np, torch
from src.data.feature_mapping import get_dataloaders
from src.models.edgeproto import get_model
from src.evaluation.evaluate_edgeproto_hybrid import load_config, compute_prototype_distances
from sklearn.metrics import roc_auc_score

def run(exp, seed, k=2.0, t=0.5):
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
    m0, s0 = mk.mean(), mk.std() + 1e-8
    zg = np.r_[(mk - m0) / s0, (mu - m0) / s0]
    pred_k, pred_u = Dk.argmin(1), Du.argmin(1)
    means, stds = {}, {}
    for c in range(len(label_map)):
        vals = Dv[Dv.argmin(1) == c].min(1)
        means[c] = vals.mean() if len(vals) else 0.0
        stds[c] = vals.std() + 1e-8 if len(vals) else 1e-8
    zp = np.r_[(mk - np.array([means[c] for c in pred_k])) / np.array([stds[c] for c in pred_k]),
               (mu - np.array([means[c] for c in pred_u])) / np.array([stds[c] for c in pred_u])]
    # per-input alpha from nearest prototype's validation std (normalized)
    sk = np.array([stds[c] for c in pred_k]); su = np.array([stds[c] for c in pred_u])
    sall = np.r_[sk, su]
    smax = sall.max()
    a = 1.0 / (1.0 + np.exp(-k * (sall / smax - t)))
    fused = a * zg + (1 - a) * zp
    return roc_auc_score(y, fused)

for exp in ['h5_dos','h1_mirai','h2_recon']:
    print(f'--- {exp} seed 2025 ---')
    for k, t in [(2.0, 0.5), (4.0, 0.5), (8.0, 0.5), (4.0, 0.3), (8.0, 0.3), (8.0, 0.7)]:
        try:
            au = run(exp, '2025', k, t)
            print(f'  k={k:.1f} t={t:.1f}: AUROC={au:.4f}')
        except Exception as e:
            print(f'  k={k:.1f} t={t:.1f}: ERR {e}')
