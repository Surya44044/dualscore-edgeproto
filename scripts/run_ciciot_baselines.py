#!/usr/bin/env python3
"""CICIoT2023 baselines (RF/ET/LR/kNN) on zero-day splits - closed + open-set metrics."""
import sys, json, time, csv
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from src.evaluation.oscr import compute_oscr

TAG = sys.argv[1] if len(sys.argv) > 1 else '50k'
EXPS = ['h1_mirai','h2_recon','h3_spoofing','h4_web_bruteforce','h5_dos','h5b_dos_ddos']
OUT = Path(f'results/tables/ciciot2023_{TAG}/baselines')
OUT.mkdir(parents=True, exist_ok=True)
MODELS = {
    'RF': RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    'ET': ExtraTreesClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    'LR': LogisticRegression(max_iter=2000, random_state=42),
    'kNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
}

def load(fn):
    df = pd.read_csv(fn)
    y = df['label'].values
    X = df.drop(columns=['label']).values.astype(np.float32)
    return X, y

rows = []
for exp in EXPS:
    Xtr, ytr = load(f'data/splits/{exp}_train.csv')
    Xtk, ytk = load(f'data/splits/{exp}_test_known.csv')
    Xtu, ytu = load(f'data/splits/{exp}_test_unknown.csv')
    sc = StandardScaler().fit(Xtr)
    Xtr, Xtk, Xtu = sc.transform(Xtr), sc.transform(Xtk), sc.transform(Xtu)
    for name, model in MODELS.items():
        t0 = time.time()
        model.fit(Xtr, ytr)
        ft = time.time() - t0
        t0 = time.time()
        preds = model.predict(Xtk)
        acc = accuracy_score(ytk, preds)
        if hasattr(model, 'predict_proba'):
            known_k = model.predict_proba(Xtk).max(axis=1)
            known_u = model.predict_proba(Xtu).max(axis=1)
        else:
            dist_k, _ = model.kneighbors(Xtk, n_neighbors=1)
            dist_u, _ = model.kneighbors(Xtu, n_neighbors=1)
            known_k = -dist_k[:, 0]
            known_u = -dist_u[:, 0]
        pt = time.time() - t0
        y_open = np.concatenate([np.zeros(len(ytk)), np.ones(len(ytu))])
        s_open = np.concatenate([known_k, known_u])
        au = roc_auc_score(y_open, -s_open)
        correct_k = (preds == ytk)
        oscr = float(compute_oscr(known_k, known_u, correct_k))
        rec = {'exp': exp, 'model': name, 'closed_acc': float(acc), 'AUROC': float(au), 'OSCR': oscr, 'train_s': round(ft,1), 'predict_s': round(pt,1)}
        rows.append(rec)
        with open(OUT / f'{exp}_{name}.json', 'w') as f:
            json.dump(rec, f, indent=2)
        print(f"{exp:<20}{name:<5} acc={acc:.4f} AUROC={au:.4f} OSCR={oscr:.4f} ({ft:.0f}s/{pt:.0f}s)")
with open(OUT / 'baselines_summary.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['exp','model','closed_acc','AUROC','OSCR'])
    for r in rows:
        w.writerow([r['exp'], r['model'], f"{r['closed_acc']:.4f}", f"{r['AUROC']:.4f}", f"{r['OSCR']:.4f}"])
print('saved:', OUT / 'baselines_summary.csv')
