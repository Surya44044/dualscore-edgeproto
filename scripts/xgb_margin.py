#!/usr/bin/env python3
"""XGB open-set score v2: top-2 probability margin (p1-p2) instead of max-prob."""
import json, numpy as np, pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from src.evaluation.oscr import compute_oscr

EXPS = ['h1_mirai','h2_recon','h3_spoofing','h4_web_bruteforce','h5_dos','h5b_dos_ddos']
OUT = Path('results/tables/ciciot2023_50k/baselines')

def load(fn):
    df = pd.read_csv(fn)
    return df.drop(columns=['label']).values.astype(np.float32), df['label'].values

for exp in EXPS:
    Xtr, ytr = load(f'data/splits/{exp}_train.csv')
    Xtk, ytk = load(f'data/splits/{exp}_test_known.csv')
    Xtu, _ = load(f'data/splits/{exp}_test_unknown.csv')
    classes = sorted(set(ytr)); cl2i = {c:i for i,c in enumerate(classes)}
    ytr_n = np.array([cl2i[c] for c in ytr]); ytk_n = np.array([cl2i[c] for c in ytk])
    sc = StandardScaler().fit(Xtr)
    Xtr, Xtk, Xtu = sc.transform(Xtr), sc.transform(Xtk), sc.transform(Xtu)
    m = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6, n_jobs=-1, random_state=42, eval_metric='logloss')
    m.fit(Xtr, ytr_n)
    pk = m.predict_proba(Xtk); pu = m.predict_proba(Xtu)
    margin_k = np.sort(pk, axis=1)[:, -1] - np.sort(pk, axis=1)[:, -2]
    margin_u = np.sort(pu, axis=1)[:, -1] - np.sort(pu, axis=1)[:, -2]
    preds_n = m.predict(Xtk); preds = np.array([classes[p] for p in preds_n])
    acc = accuracy_score(ytk, preds)
    au = roc_auc_score(np.r_[np.zeros(len(ytk)), np.ones(len(Xtu))], np.r_[margin_k, margin_u])
    oscr = float(compute_oscr(margin_k, margin_u, preds_n == ytk_n))
    json.dump({'exp': exp, 'model': 'XGB-margin', 'closed_acc': float(acc), 'AUROC': float(au), 'OSCR': oscr}, open(OUT / f'{exp}_XGB_margin.json','w'), indent=2)
    print(f"{exp:<20} XGB-margin acc={acc:.4f} AUROC={au:.4f} OSCR={oscr:.4f}", flush=True)
print("XGB margin done")
