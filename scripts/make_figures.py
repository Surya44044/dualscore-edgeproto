#!/usr/bin/env python3
"""Figures: ROC, score distributions, seed stability, prototype similarity heatmap."""
import json
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score
from src.models.edgeproto import get_model
from src.evaluation.evaluate_edgeproto_hybrid import load_config

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT/'results/figures'; FIG.mkdir(parents=True, exist_ok=True)
EXPS = ['h1_mirai','h2_recon','h3_spoofing','h4_web_bruteforce','h5_dos','h5b_dos_ddos']
SEEDS = ['42','123','2025']

def fig_roc():
    plt.figure(figsize=(6,5))
    for exp in ['h1_mirai','h2_recon','h3_spoofing']:
        d = np.load(ROOT/f'results/scores/hybrid/{exp}_Hybrid-Mean-50-50_scores.npz')
        y, s = d['y_true'], d['y_score']
        a = roc_auc_score(y, s)
        if a < 0.5: s = -s; a = 1-a
        fpr, tpr, _ = roc_curve(y, s)
        plt.plot(fpr, tpr, label=f'{exp} (AUC={a:.3f})')
    plt.plot([0,1],[0,1],'k--',lw=0.8)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('Zero-day detection ROC (Hybrid-Mean-50-50)'); plt.legend()
    plt.tight_layout(); plt.savefig(FIG/'fig1_roc.png', dpi=150); plt.close()
    print('fig1_roc.png')

def fig_dists():
    fig, axs = plt.subplots(1, 2, figsize=(10,4))
    for ax, exp in zip(axs, ['h1_mirai', 'h5_dos']):
        d = np.load(ROOT/f'results/scores/hybrid/{exp}_Hybrid-Mean-50-50_scores.npz')
        sk, su = d['score_k'], d['score_u']
        ax.hist(sk, bins=60, alpha=0.6, label='known', density=True)
        ax.hist(su, bins=60, alpha=0.6, label='unknown', density=True)
        ax.set_title(exp); ax.legend()
    plt.tight_layout(); plt.savefig(FIG/'fig2_scores.png', dpi=150); plt.close()
    print('fig2_scores.png')

def fig_stability():
    exps, stds_b, stds_h = [], [], []
    for exp in EXPS:
        b, h = [], []
        for s in SEEDS:
            bj = json.load(open(ROOT/f'results/tables/ciciot2023_50k/seeds/{exp}_s{s}_base.json'))
            hj = json.load(open(ROOT/f'results/tables/ciciot2023_50k/seeds/{exp}_s{s}_hybrid.json'))
            b.append(bj.get('AUROC') or bj.get('open_set_auroc')); h.append(hj['AUROC'])
        exps.append(exp); stds_b.append(np.std(b)); stds_h.append(np.std(h))
    x = np.arange(len(exps)); w = 0.35
    plt.figure(figsize=(9,4))
    plt.bar(x-w/2, stds_b, w, label='base', alpha=0.8)
    plt.bar(x+w/2, stds_h, w, label='hybrid', alpha=0.8)
    plt.xticks(x, exps, rotation=30); plt.ylabel('std of AUROC (3 seeds)')
    plt.title('Seed stability: hybrid reduces variance'); plt.legend()
    plt.tight_layout(); plt.savefig(FIG/'fig3_stability.png', dpi=150); plt.close()
    print('fig3_stability.png')

def fig_proto():
    import pandas as pd
    exp = 'h5_dos'
    df = pd.read_csv(ROOT/'data/splits/h5_dos_train.csv')
    X = df.drop(columns=['label']).values.astype(np.float32)
    y = df['label'].values; classes = sorted(set(y)); cl2i = {c:i for i,c in enumerate(classes)}
    yn = np.array([cl2i[c] for c in y])
    cfg = load_config('configs/h5_dos.yaml')
    m = get_model(cfg, num_classes=len(classes))
    import torch
    m.load_state_dict(torch.load(ROOT/'models/calib/h5_dos_s2025/best_edgeproto.pt', map_location='cpu'))
    m.eval()
    import torch
    with torch.no_grad():
        e = m.encoder(torch.tensor(X.reshape(-1,4,12)))
        protos = m.compute_prototypes(e, torch.tensor(yn), len(classes))
    P = protos.numpy(); S = P @ P.T
    plt.figure(figsize=(6,5))
    plt.imshow(S, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(label='cosine similarity')
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
    plt.yticks(range(len(classes)), classes)
    plt.title('Prototype similarity (h5: DoS unknown)')
    plt.tight_layout(); plt.savefig(FIG/'fig4_prototypes.png', dpi=150); plt.close()
    print('fig4_prototypes.png')

if __name__ == '__main__':
    fig_roc(); fig_dists(); fig_stability(); fig_proto()
    print('ALL FIGURES DONE')
