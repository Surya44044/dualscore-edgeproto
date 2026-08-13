#!/usr/bin/env python3
"""XGBoost baseline on CICIoT2023 50k zero-day splits (numeric labels for XGB)."""
import json, time, csv
import numpy as np, pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from src.evaluation.oscr import compute_oscr

EXPS = ['h1_mirai','h2_recon','h3_spoofing','h4_web_bruteforce','h5_dos','h5b_dos_ddos']
OUT = Path('results/tables/ciciot2023_50k/baselines')
OUT.mkdir(parents=True, exist_ok=True)

def load(fn):
    df = pd.read_csv(fn)
    X = df.drop(columns=['label']).values.astype(np.float32)
    y = df['label'].values
    return X, y

rows = []
for exp in EXPS:
    Xtr, ytr = load(f'data/splits/{exp}_train.csv')
    Xtk, ytk = load(f'data/splits/{exp}_test_known.csv')
    Xtu, _ = load(f'data/splits/{exp}_test_unknown.csv')
    # numeric label map from train (sorted like get_dataloaders)
    classes = sorted(set(ytr))
    cl2i = {c: i for i, c in enumerate(classes)}
    ytr_n = np.array([cl2i[c] for c in ytr])
    ytk_n = np.array([cl2i[c] for c in ytk])
    sc = StandardScaler().fit(Xtr)
    Xtr, Xtk, Xtu = sc.transform(Xtr), sc.transform(Xtk), sc.transform(Xtu)
    m = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6, n_jobs=-1, random_state=42, eval_metric='logloss')
    t0 = time.time()
    m.fit(Xtr, ytr_n)
    ft = time.time() - t0
    t0 = time.time()
    preds_n = m.predict(Xtk)
    preds = np.array([classes[p] for p in preds_n])
    acc = accuracy_score(ytk, preds)
    known_k = m.predict_proba(Xtk).max(1)
    known_u = m.predict_proba(Xtu).max(1)
    pt = time.time() - t0
    y = np.r_[np.zeros(len(ytk)), np.ones(len(Xtu))]
    au = roc_auc_score(y, np.r_[known_k, known_u])
    oscr = float(compute_oscr(known_k, known_u, preds_n == ytk_n))
    rec = {'exp': exp, 'model': 'XGB', 'closed_acc': float(acc), 'AUROC': float(au), 'OSCR': float(oscr), 'train_s': round(ft,1), 'predict_s': round(pt,1)}
    rows.append(rec)
    with open(OUT / f'{exp}_XGB.json', 'w') as f:
        json.dump(rec, f, indent=2)
    print(f"{exp:<20} XGB  acc={acc:.4f} AUROC={au:.4f} OSCR={oscr:.4f} ({ft:.0f}s/{pt:.0f}s)", flush=True)

with open(OUT / 'xgb_summary.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['exp','model','closed_acc','AUROC','OSCR'])
    for r in rows:
        w.writerow([r['exp'], r['model'], f"{r['closed_acc']:.4f}", f"{r['AUROC']:.4f}", f"{r['OSCR']:.4f}"])
print('saved:', OUT / 'xgb_summary.csv', flush=True)
