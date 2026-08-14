#!/usr/bin/env python3
import sys, numpy as np, torch
from src.data.feature_mapping import get_dataloaders
from src.models.edgeproto import get_model
from src.evaluation.evaluate_edgeproto_hybrid import load_config, SmallAutoencoder
from sklearn.metrics import roc_auc_score

exp = sys.argv[1] if len(sys.argv) > 1 else 'h5_dos'
seed = sys.argv[2] if len(sys.argv) > 2 else '2025'
dev = torch.device('cpu')
cfg = load_config(f'configs/{exp}.yaml')
tr, val, tk, tu, lm = get_dataloaders(
    f'data/splits/{exp}_train.csv', f'data/splits/{exp}_val.csv',
    f'data/splits/{exp}_test_known.csv', f'data/splits/{exp}_test_unknown.csv', batch_size=256)
m = get_model(cfg, num_classes=len(lm)).to(dev)
m.load_state_dict(torch.load(f'models/oe_{exp}_base/best_edgeproto.pt', map_location=dev))
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
predv = Dv.argmin(1)
means, stds = {}, {}
for c in range(len(lm)):
    vals = Dv[predv == c].min(1)
    means[c] = vals.mean() if len(vals) else 0.0
    stds[c] = vals.std() + 1e-8 if len(vals) else 1e-8
def pz(D):
    pr = D.argmin(1)
    return (D.min(1) - np.array([means[c] for c in pr])) / np.array([stds[c] for c in pr])
zp = np.r_[pz(Dk), pz(Du)]
me, se = Ev.mean(), Ev.std() + 1e-8
zae = np.r_[(Ek - me) / se, (Eu - me) / se]
y = np.r_[np.zeros(len(Dk)), np.ones(len(Du))]
for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
    au = roc_auc_score(y, w * zp + (1 - w) * zae)
    print(f'{exp} w_proto={w:.2f} AUROC={au:.4f}')
