#!/usr/bin/env python3
"""CICIoT2023 additional experiments: dl_baseline | op_metrics | natural_ratio | mode_select"""
import os, sys, json, csv, subprocess
import numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score
from src.evaluation.oscr import compute_oscr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/processed/ciciot2023/ciciot2023_sample_50k_processed.csv'
EXPS = ['h1_mirai','h2_recon','h3_spoofing','h4_web_bruteforce','h5_dos','h5b_dos_ddos']
SEEDS = ['42','123','2025']
OUT = ROOT / 'results/tables/ciciot2023_50k/additional'
os.makedirs(OUT, exist_ok=True)

def load_split(exp, split):
    df = pd.read_csv(ROOT / f'data/splits/{exp}_{split}.csv')
    return df.drop(columns=['label']).values.astype(np.float32), df['label'].values

def regen_splits(exp):
    subprocess.run([sys.executable, 'src/data/make_splits.py', '--input', str(DATA),
                    '--config', f'configs/{exp}.yaml', '--output_dir', 'data/splits'], check=True)

class MLP(nn.Module):
    def __init__(self, nin, ncls):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nin, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, ncls))
    def forward(self, x): return self.net(x)

class CNN(nn.Module):
    def __init__(self, ncls):
        super().__init__()
        self.feats = nn.Sequential(nn.Conv1d(4,16,3,padding=1), nn.ReLU(), nn.MaxPool1d(2),
                                   nn.Conv1d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool1d(2))
        self.head = nn.Linear(32*3, ncls)
    def forward(self, x): return self.head(self.feats(x).flatten(1))

def fit_eval(name, make_model, exp):
    Xtr, ytr = load_split(exp, 'train'); Xva, yva = load_split(exp, 'val')
    Xtk, ytk = load_split(exp, 'test_known'); Xtu, _ = load_split(exp, 'test_unknown')
    classes = sorted(set(ytr)); cl2i = {c: i for i, c in enumerate(classes)}
    ytr_n = np.array([cl2i[c] for c in ytr]); yva_n = np.array([cl2i[c] for c in yva])
    ytk_n = np.array([cl2i[c] for c in ytk])
    R = (lambda a: a.reshape(-1,4,12)) if name == 'CNN' else (lambda a: a)
    Xtr_t, Xva_t, Xtk_t, Xtu_t = map(torch.tensor, (R(Xtr), R(Xva), R(Xtk), R(Xtu)))
    ytr_t, yva_t = torch.tensor(ytr_n), torch.tensor(yva_n)
    m = make_model(len(classes)); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss(); n = len(ytr_t); bs = 256
    best_acc, best_sd = -1.0, None
    for ep in range(30):
        m.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]; opt.zero_grad()
            loss = lossf(m(Xtr_t[idx]), ytr_t[idx]); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            acc = (m(Xva_t).argmax(1) == yva_t).float().mean().item()
        if acc > best_acc:
            best_acc = acc; best_sd = {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_sd); m.eval()
    with torch.no_grad():
        pk = torch.softmax(m(Xtk_t), 1).numpy(); pu = torch.softmax(m(Xtu_t), 1).numpy()
    conf_k, conf_u = pk.max(1), pu.max(1); preds = pk.argmax(1)
    acc = accuracy_score(ytk, np.array([classes[p] for p in preds]))
    au = roc_auc_score(np.r_[np.zeros(len(ytk)), np.ones(len(Xtu))], np.r_[-conf_k, -conf_u])
    oscr = float(compute_oscr(conf_k, conf_u, preds == ytk_n))
    rec = {'exp': exp, 'model': name, 'closed_acc': float(acc), 'AUROC': float(au), 'OSCR': oscr}
    with open(OUT / f'{exp}_{name}.json', 'w') as f: json.dump(rec, f, indent=2)
    print(f'{exp:<20} {name:<4} acc={acc:.4f} AUROC={au:.4f} OSCR={oscr:.4f}', flush=True)

def dl_baseline():
    for exp in EXPS:
        regen_splits(exp)
        fit_eval('MLP', lambda c: MLP(48, c), exp)
        fit_eval('CNN', lambda c: CNN(c), exp)

def op_metrics():
    rows = []
    for exp in EXPS:
        vals = {k: [] for k in ['AUPR_Out','TPRat1_FPR','FPRat95_TPR','detection_error']}
        for s in SEEDS:
            p = ROOT / f'results/tables/ciciot2023_50k/seeds/{exp}_s{s}_hybrid.json'
            if not p.exists(): continue
            j = json.load(open(p))
            for k in vals: vals[k].append(j.get(k, float('nan')))
        if len(vals['AUPR_Out']) == 3:
            rows.append([exp] + [f'{np.nanmean(vals[k]):.4f}+-{np.nanstd(vals[k]):.4f}' for k in vals])
    with open(OUT / 'operational_metrics.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['exp'] + list(vals.keys())); w.writerows(rows)
    for r in rows: print(r, flush=True)

def natural_ratio():
    import joblib, pyarrow.parquet as pq
    exp = 'h1_mirai'; regen_splits(exp)
    scaler = joblib.load(ROOT / 'data/splits/h1_mirai_scaler.pkl')
    hdr = pd.read_csv(DATA, nrows=0).columns.tolist(); feats = [c for c in hdr if c != 'label']
    pf = pq.ParquetFile(str(ROOT / 'data/raw/ciciot2023/train.parquet'))
    rng = np.random.default_rng(7)
    props = {'DDoS':0.728,'DoS':0.173,'Benign':0.0235,'Spoofing':0.0104,'Recon':0.0076,'Web-based':0.0005,'BruteForce':0.0003}
    n_known = 60000; quotas = {k: int(n_known*v) for k, v in props.items()}
    got = {k: 0 for k in quotas}; sel = {k: [] for k in quotas}
    for i in range(pf.num_row_groups):
        df = pf.read_row_group(i, columns=feats + ['attack_class']).to_pandas()
        for k in quotas:
            if got[k] >= quotas[k]: continue
            sub = df[df['attack_class'] == k]
            take = min(len(sub), quotas[k] - got[k])
            sel[k].append(sub.iloc[rng.choice(len(sub), take, replace=False)]); got[k] += take
        if all(got[k] >= quotas[k] for k in quotas): break
    known = pd.concat([pd.concat(v) for v in sel.values() if v])
    n_unk = int(n_known * 0.056 / 0.944); got_u = 0; unk_rows = []
    for i in range(pf.num_row_groups):
        df = pf.read_row_group(i, columns=feats + ['attack_class']).to_pandas()
        sub = df[df['attack_class'] == 'Mirai']
        take = min(len(sub), n_unk - got_u)
        unk_rows.append(sub.iloc[rng.choice(len(sub), take, replace=False)]); got_u += take
        if got_u >= n_unk: break
    unk = pd.concat(unk_rows)
    Xk_raw = known[feats].values.astype(np.float32)
    Xu_raw = unk[feats].values.astype(np.float32)
    Xk = scaler.transform(Xk_raw); Xu = scaler.transform(Xu_raw)
    Xk = np.hstack([Xk, np.zeros((len(Xk), 2), np.float32)]).reshape(-1,4,12)
    Xu = np.hstack([Xu, np.zeros((len(Xu), 2), np.float32)]).reshape(-1,4,12)
    Xtr, ytr = load_split(exp, 'train'); classes = sorted(set(ytr))
    cl2i = {c: i for i, c in enumerate(classes)}
    from src.models.edgeproto import get_model
    from src.evaluation.evaluate_edgeproto_hybrid import load_config, SmallAutoencoder
    cfg = load_config('configs/h1_mirai.yaml')
    m = get_model(cfg, num_classes=len(classes))
    m.load_state_dict(torch.load(ROOT/'models/calib/h1_mirai_s2025/best_edgeproto.pt', map_location='cpu')); m.eval()
    ae = SmallAutoencoder(48); ae.load_state_dict(torch.load(ROOT/'models/calib/h1_mirai_s2025_ae.pt', map_location='cpu')); ae.eval()
    with torch.no_grad():
        etr = m.encoder(torch.tensor(Xtr.reshape(-1,4,12)))
        protos = m.compute_prototypes(etr, torch.tensor(np.array([cl2i[c] for c in ytr])), len(classes))
        def dists(X):
            with torch.no_grad():
                _, d = m(torch.tensor(X), prototypes=protos)
            return d.numpy()
        def aes(X):
            with torch.no_grad():
                xf = torch.tensor(X.reshape(len(X),-1))
                return ((ae(xf)-xf)**2).mean(1).numpy()
    Dv, Dk, Du = dists(Xtr.reshape(-1,4,12)), dists(Xk), dists(Xu)
    Ev, Ek, Eu = aes(Xtr.reshape(-1,4,12)), aes(Xk), aes(Xu)
    m0, s0 = Dv.min(1).mean(), Dv.min(1).std()+1e-8
    me, se = Ev.mean(), Ev.std()+1e-8
    sk = ((Dk.min(1)-m0)/s0 + (Ek-me)/se)/2
    su = ((Du.min(1)-m0)/s0 + (Eu-me)/se)/2
    au = roc_auc_score(np.r_[np.zeros(len(Xk)), np.ones(len(Xu))], np.r_[sk, su])
    acc = accuracy_score(known['attack_class'].values, np.array([classes[p] for p in Dk.argmin(1)]))
    rec = {'exp': 'h1_natural_ratio', 'n_known': len(Xk), 'n_unknown': len(Xu), 'AUROC': float(au), 'closed_acc': float(acc)}
    json.dump(rec, open(OUT/'h1_natural_ratio.json','w'), indent=2)
    print('natural-ratio h1:', rec, flush=True)

def mode_select_one(exp):
    from src.models.edgeproto import get_model
    from src.evaluation.evaluate_edgeproto_hybrid import load_config, SmallAutoencoder
    Xtr, ytr = load_split(exp, 'train'); Xva, yva = load_split(exp, 'val')
    Xtk, ytk = load_split(exp, 'test_known'); Xtu, _ = load_split(exp, 'test_unknown')
    classes = sorted(set(ytr)); cl2i = {c: i for i, c in enumerate(classes)}
    ytr_n = np.array([cl2i[c] for c in ytr]); yva_n = np.array([cl2i[c] for c in yva])
    cfg = load_config(f'configs/{exp}.yaml')
    m = get_model(cfg, num_classes=len(classes))
    m.load_state_dict(torch.load(ROOT/f'models/calib/{exp}_s2025/best_edgeproto.pt', map_location='cpu')); m.eval()
    ae = SmallAutoencoder(48); ae.load_state_dict(torch.load(ROOT/f'models/calib/{exp}_s2025_ae.pt', map_location='cpu')); ae.eval()
    with torch.no_grad():
        etr = m.encoder(torch.tensor(Xtr.reshape(-1,4,12)))
        protos = m.compute_prototypes(etr, torch.tensor(ytr_n), len(classes))
    def dists(X):
        with torch.no_grad():
            _, d = m(torch.tensor(X.reshape(-1,4,12)), prototypes=protos); return d.numpy()
    def aes(X):
        with torch.no_grad():
            xf = torch.tensor(X.reshape(len(X),-1)); return ((ae(xf)-xf)**2).mean(1).numpy()
    Dv, Dk, Du = dists(Xva), dists(Xtk), dists(Xtu)
    Ev, Ek, Eu = aes(Xva), aes(Xtk), aes(Xtu)
    m0, s0 = Dv.min(1).mean(), Dv.min(1).std()+1e-8
    me, se = Ev.mean(), Ev.std()+1e-8
    predv = Dv.argmin(1); means, stds = {}, {}
    for c in range(len(classes)):
        vals = Dv[predv == c].min(1)
        means[c] = vals.mean() if len(vals) else 0.0
        stds[c] = vals.std()+1e-8 if len(vals) else 1e-8
    def pz(D):
        pr = D.argmin(1)
        return (D.min(1)-np.array([means[c] for c in pr]))/np.array([stds[c] for c in pr])
    za_v = (Ev-me)/se; std_v = ((Dv.min(1)-m0)/s0 + za_v)/2; cal_v = 0.25*pz(Dv)+0.75*za_v
    mode_of = {}
    for c in range(len(classes)):
        mask = yva_n == c
        y = np.r_[np.zeros((~mask).sum()), np.ones(mask.sum())]
        a1 = roc_auc_score(y, np.r_[std_v[~mask], std_v[mask]])
        a2 = roc_auc_score(y, np.r_[cal_v[~mask], cal_v[mask]])
        mode_of[c] = 'cal' if a2 >= a1 else 'std'
    zg_t = (np.r_[Dk, Du].min(1)-m0)/s0; za_t = (np.r_[Ek, Eu]-me)/se
    std_t = (zg_t+za_t)/2; cal_t = 0.25*np.r_[pz(Dk), pz(Du)]+0.75*za_t
    pr_t = np.r_[Dk, Du].argmin(1)
    sel = np.array([cal_t[i] if mode_of[pr_t[i]] == 'cal' else std_t[i] for i in range(len(pr_t))])
    y = np.r_[np.zeros(len(Dk)), np.ones(len(Du))]
    au = roc_auc_score(y, sel)
    preds = Dk.argmin(1); labels = np.array([cl2i[c] for c in ytk])
    oscr = float(compute_oscr(-sel[:len(Dk)], -sel[len(Dk):], preds == labels))
    modes = {classes[c]: mode_of[c] for c in range(len(classes))}
    json.dump({'exp': exp, 'AUROC': float(au), 'OSCR': oscr, 'modes': modes}, open(OUT/f'{exp}_mode_select.json','w'), indent=2)
    print(f'{exp:<20} mode_select AUROC={au:.4f} OSCR={oscr:.4f} | modes={modes}', flush=True)

def mode_select():
    for exp in EXPS:
        try: mode_select_one(exp)
        except Exception as e: print('FAIL', exp, e, flush=True)

if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'dl_baseline'
    {'dl_baseline': dl_baseline, 'op_metrics': op_metrics, 'natural_ratio': natural_ratio, 'mode_select': mode_select}[cmd]()
