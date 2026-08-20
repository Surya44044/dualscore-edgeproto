#!/usr/bin/env python3
"""3-seed w=0.25 fusion: per-seed calib models -> per-class proto z + AE z -> AUROC/OSCR."""
import json, csv, numpy as np, torch
from src.data.feature_mapping import get_dataloaders
from src.models.edgeproto import get_model
from src.evaluation.evaluate_edgeproto_hybrid import load_config, SmallAutoencoder
from src.evaluation.oscr import compute_oscr
from sklearn.metrics import roc_auc_score

EXPS = ['h1_mirai','h2_recon','h3_spoofing','h4_web_bruteforce','h5_dos','h5b_dos_ddos']
SEEDS = ['42','123','2025']
W = 0.25
dev = torch.device('cpu')
OUT = 'results/tables/ciciot2023_50k/calib/calib_fusion_3seed.csv'

def pz_scores(D, Dv, lm):
    predv = Dv.argmin(1); means = {}; stds = {}
    for c in range(len(lm)):
        vals = Dv[predv == c].min(1)
        means[c] = vals.mean() if len(vals) else 0.0
        stds[c] = vals.std() + 1e-8 if len(vals) else 1e-8
    pr = D.argmin(1)
    return (D.min(1) - np.array([means[c] for c in pr])) / np.array([stds[c] for c in pr])

with open(OUT, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['exp','seed','AUROC','OSCR'])
    for exp in EXPS:
        for seed in SEEDS:
            cfg = load_config(f'configs/{exp}.yaml')
            tr, val, tk, tu, lm = get_dataloaders(
                f'data/splits/{exp}_train.csv', f'data/splits/{exp}_val.csv',
                f'data/splits/{exp}_test_known.csv', f'data/splits/{exp}_test_unknown.csv', batch_size=256)
            m = get_model(cfg, num_classes=len(lm)).to(dev)
            m.load_state_dict(torch.load(f'models/calib/{exp}_s{seed}/best_edgeproto.pt', map_location=dev))
            m.eval()
            ae = SmallAutoencoder(input_dim=48).to(dev)
            ae.load_state_dict(torch.load(f'models/hybrid_autoencoders/{exp}_ae_known.pt', map_location=dev))
            ae.eval()
            emb, lab = [], []
            with torch.no_grad():
                for x, y in tr:
                    emb.append(m.encoder(x)); lab.append(y)
            protos = m.compute_prototypes(torch.cat(emb), torch.cat(lab), len(lm))
            def dists(loader):
                out = []
                with torch.no_grad():
                    for x, _ in loader:
                        _, dd = m(x.to(dev), prototypes=protos)
                        out.append(dd.cpu().numpy())
                return np.concatenate(out)
            def aes(loader):
                out = []
                with torch.no_grad():
                    for x, _ in loader:
                        xf = x.view(x.size(0), -1).to(dev)
                        out.append(((ae(xf) - xf) ** 2).mean(1).cpu().numpy())
                return np.concatenate(out)
            Dv, Dk, Du = dists(val), dists(tk), dists(tu)
            Ev, Ek, Eu = aes(val), aes(tk), aes(tu)
            me, se = Ev.mean(), Ev.std() + 1e-8
            zp_k, zp_u = pz_scores(Dk, Dv, lm), pz_scores(Du, Dv, lm)
            za_k, za_u = (Ek - me) / se, (Eu - me) / se
            sk = W * zp_k + (1 - W) * za_k
            su = W * zp_u + (1 - W) * za_u
            y = np.r_[np.zeros(len(sk)), np.ones(len(su))]
            au = roc_auc_score(y, np.r_[sk, su])
            preds = Dk.argmin(1); labels = np.array(tk.dataset.labels)
            oscr = float(compute_oscr(-sk, -su, preds == labels))
            w.writerow([exp, seed, f'{au:.4f}', f'{oscr:.4f}'])
            print(f'{exp:<20} seed {seed}: AUROC={au:.4f} OSCR={oscr:.4f}', flush=True)
print('saved:', OUT)
