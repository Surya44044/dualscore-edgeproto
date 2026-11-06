"""Natural-ratio zero-day evaluation for all holdouts (mirrors h1 recipe exactly)."""
import numpy as np, pandas as pd, pyarrow.parquet as pq, torch, json, sys
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
ROOT = Path('.')
DATA = 'data/processed/ciciot2023/ciciot2023_sample_50k_processed.csv'
OUT = ROOT/'results/tables/natural_ratio'; OUT.mkdir(parents=True, exist_ok=True)
N_KNOWN = 60000
SHARES = {'DDoS':0.728,'DoS':0.173,'Mirai':0.0564,'Benign':0.0235,'Spoofing':0.0104,'Recon':0.0076,'Web-based':0.0005,'BruteForce':0.0003}
PLAN = {
 'h1_mirai': {'unk': ['Mirai']},
 'h2_recon': {'unk': ['Recon']},
 'h3_spoofing': {'unk': ['Spoofing']},
 'h4_web_bruteforce': {'unk': ['Web-based','BruteForce']},
 'h5_dos': {'unk': ['DoS']},
}
def load_split(exp, split):
    df = pd.read_csv(ROOT/f'data/splits/{exp}_{split}.csv')
    lab = df.pop('label'); return df.values.astype(np.float32), lab

def sample(exp, unk_list, cap_unk=None):
    scaler = joblib.load(ROOT/f'data/splits/{exp}_scaler.pkl')
    hdr = pd.read_csv(DATA, nrows=0).columns.tolist(); feats = [c for c in hdr if c != 'label']
    pf = pq.ParquetFile(str(ROOT/'data/raw/ciciot2023/train.parquet'))
    rng = np.random.default_rng(7)
    quotas = {k: int(N_KNOWN*v) for k,v in SHARES.items() if k not in unk_list}
    p_unk = sum(SHARES[k] for k in unk_list)
    n_unk_target = cap_unk if cap_unk else int(N_KNOWN*p_unk/(1-p_unk))
    per_unk = {k: int(n_unk_target*SHARES[k]/p_unk) for k in unk_list}
    got = {k:0 for k in list(quotas)+list(per_unk)}; sel = {k:[] for k in list(quotas)+list(per_unk)}
    for i in range(pf.num_row_groups):
        df = pf.read_row_group(i, columns=feats+['attack_class']).to_pandas()
        for k in sel:
            if (k in quotas and got[k] >= quotas[k]) or (k in per_unk and got[k] >= per_unk[k]): continue
            need = (quotas.get(k, 10**9) if k in quotas else per_unk[k]) - got[k]
            sub = df[df['attack_class'] == k]
            if len(sub) == 0: continue
            take = min(len(sub), need)
            sel[k].append(sub.iloc[rng.choice(len(sub), take, replace=False)]); got[k] += take
        if all((k not in quotas or got[k] >= quotas[k]) and (k not in per_unk or got[k] >= per_unk[k]) for k in sel): break
    known = pd.concat([pd.concat(v) for v in [sel[k] for k in quotas] if v])
    unk = pd.concat([pd.concat(sel[k]) for k in unk_list if sel[k]])
    Xk_raw = known[feats].values.astype(np.float32); Xu_raw = unk[feats].values.astype(np.float32)
    Xk = scaler.transform(Xk_raw); Xu = scaler.transform(Xu_raw)
    Xk = np.hstack([Xk, np.zeros((len(Xk),2),np.float32)]).reshape(-1,4,12)
    Xu = np.hstack([Xu, np.zeros((len(Xu),2),np.float32)]).reshape(-1,4,12)
    return Xk, Xu, known

from src.models.edgeproto import get_model
from src.evaluation.evaluate_edgeproto_hybrid import load_config, SmallAutoencoder
def run(exp, tag=None, cap_unk=None):
    unk_list = PLAN[exp]['unk']
    Xk, Xu, known = sample(exp, unk_list, cap_unk)
    Xtr, ytr = load_split(exp, 'train')
    classes = sorted(set(ytr)); cl2i = {c:i for i,c in enumerate(classes)}
    cfg = load_config(f'configs/{exp}.yaml')
    m = get_model(cfg, num_classes=len(classes))
    m.load_state_dict(torch.load(ROOT/f'models/calib/{exp}_s2025/best_edgeproto.pt', map_location='cpu')); m.eval()
    ae = SmallAutoencoder(48); ae.load_state_dict(torch.load(ROOT/f'models/calib/{exp}_s2025_ae.pt', map_location='cpu')); ae.eval()
    with torch.no_grad():
        etr = m.encoder(torch.tensor(Xtr.reshape(-1,4,12)))
        protos = m.compute_prototypes(etr, torch.tensor(np.array([cl2i[c] for c in ytr])), len(classes))
        _, Dtr = m(torch.tensor(Xtr.reshape(-1,4,12)), prototypes=protos); Dtr = Dtr.numpy()
        _, Dk = m(torch.tensor(Xk), prototypes=protos); Dk = Dk.numpy()
        _, Du = m(torch.tensor(Xu), prototypes=protos); Du = Du.numpy()
        def aes(X):
            xf = torch.tensor(X.reshape(len(X),-1)); return ((ae(xf)-xf)**2).mean(1).numpy()
        Ef, Ek, Eu = aes(Xtr.reshape(-1,4,12)), aes(Xk), aes(Xu)
    m0,s0 = Dtr.min(1).mean(), Dtr.min(1).std()+1e-8; me,se = Ef.mean(), Ef.std()+1e-8
    sk = ((Dk.min(1)-m0)/s0 + (Ek-me)/se)/2
    su = ((Du.min(1)-m0)/s0 + (Eu-me)/se)/2
    au = float(roc_auc_score(np.r_[np.zeros(len(sk)), np.ones(len(su))], np.r_[sk, su]))
    acc = float(accuracy_score(known['attack_class'].values, np.array([classes[p] for p in Dk.argmin(1)])))
    rec = {'exp': f'{exp}_natural_ratio'+(f'_{tag}' if tag else ''), 'AUROC': au, 'closed_acc': acc,
           'n_known': int(len(sk)), 'n_unknown': int(len(su)), 'unknown': unk_list}
    json.dump(rec, open(OUT/f"{rec['exp']}.json",'w'), indent=2)
    print("NATRATIO", rec, flush=True)
    return rec

if __name__ == '__main__':
    recs = []
    for exp in ['h1_mirai','h2_recon','h3_spoofing','h4_web_bruteforce','h5_dos']:
        try: recs.append(run(exp))
        except Exception as e: print('FAIL', exp, repr(e)[:200], flush=True)
    try: recs.append(run('h4_web_bruteforce', tag='capped', cap_unk=1500))
    except Exception as e: print('FAIL h4capped', repr(e)[:200], flush=True)
    import csv
    with open(OUT/'natural_ratio_summary.csv','w',newline='') as f:
        w = csv.DictWriter(f, fieldnames=['exp','AUROC','closed_acc','n_known','n_unknown','unknown'])
        w.writeheader()
        for r in recs: w.writerow({'exp':r['exp'],'AUROC':round(r['AUROC'],4),'closed_acc':round(r['closed_acc'],4),'n_known':r['n_known'],'n_unknown':r['n_unknown'],'unknown':'|'.join(r['unknown'])})
    print("NATRATIO ALL DONE", len(recs), flush=True)
