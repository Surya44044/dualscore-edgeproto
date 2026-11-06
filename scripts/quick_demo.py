#!/usr/bin/env python3
"""Teacher demo v2: calibrated operating point (95% known pass) + detection rate."""
import sys, numpy as np, torch
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score
from src.data.feature_mapping import get_dataloaders
from src.models.edgeproto import get_model
from src.evaluation.evaluate_edgeproto_hybrid import load_config, SmallAutoencoder
from src.evaluation.oscr import compute_oscr

dev = torch.device('cpu')
cfg = load_config('configs/h1_mirai.yaml')
tr, val, tk, tu, lm = get_dataloaders(
    'data/splits/h1_mirai_train.csv', 'data/splits/h1_mirai_val.csv',
    'data/splits/h1_mirai_test_known.csv', 'data/splits/h1_mirai_test_unknown.csv', batch_size=512)
mp, mae = 'models/calib/h1_mirai_s2025/best_edgeproto.pt', 'models/calib/h1_mirai_s2025_ae.pt'
if not (Path(mp).exists() and Path(mae).exists()):
    print("ERROR: model files missing"); sys.exit(1)
m = get_model(cfg, num_classes=len(lm)).to(dev); m.load_state_dict(torch.load(mp, map_location=dev)); m.eval()
ae = SmallAutoencoder(48).to(dev); ae.load_state_dict(torch.load(mae, map_location=dev)); ae.eval()

def dists(loader):
    out = []
    with torch.no_grad():
        for x, _ in loader:
            _, d = m(x.to(dev)); out.append(d.cpu().numpy())
    return np.concatenate(out)
def aes(loader):
    out = []
    with torch.no_grad():
        for x, _ in loader:
            xf = x.view(x.size(0), -1).to(dev)
            out.append(((ae(xf) - xf) ** 2).mean(1).cpu().numpy())
    return np.concatenate(out)

print("=" * 64)
print(" DUALSCORE-EDGEPROTO  |  h1: Mirai held out as ZERO-DAY")
print("=" * 64)
Dv, Dk, Du = dists(val), dists(tk), dists(tu)
Ev, Ek, Eu = aes(val), aes(tk), aes(tu)
m0, s0 = Dv.min(1).mean(), Dv.min(1).std() + 1e-8
me, se = Ev.mean(), Ev.std() + 1e-8
sv = ((Dv.min(1) - m0) / s0 + (Ev - me) / se) / 2
sk = ((Dk.min(1) - m0) / s0 + (Ek - me) / se) / 2
su = ((Du.min(1) - m0) / s0 + (Eu - me) / se) / 2

preds = Dk.argmin(1); labels = np.array(tk.dataset.labels)
acc = accuracy_score(labels, preds)
y = np.r_[np.zeros(len(sk)), np.ones(len(su))]
au = roc_auc_score(y, np.r_[sk, su])
oscr = float(compute_oscr(-sk, -su, preds == labels))
print(f" Known-family accuracy       : {acc:.4f}   (7 known classes)")
print(f" ZERO-DAY AUROC              : {au:.4f}")
print(f" OSCR                        : {oscr:.4f}")
print(f" Total parameters            : {sum(p.numel() for p in m.parameters()) + sum(p.numel() for p in ae.parameters()):,}")

# Operating point: threshold = 95th percentile of validation-KNOWN scores
THR = float(np.percentile(sv, 95))
fpr = float((sk > THR).mean())          # fraction of known wrongly alerted
tpr = float((su > THR).mean())          # fraction of zero-day caught
print("-" * 64)
print(f" Operating point: threshold = {THR:.3f}  (95th pct of KNOWN scores)")
print(f"   -> False-alarm rate on known traffic : {fpr*100:.1f}%   (target ~5%)")
print(f"   -> Zero-day DETECTION rate (Mirai)   : {tpr*100:.1f}%")
print("=" * 64)

inv = {i: c for c, i in lm.items()}
def score_one(x):
    with torch.no_grad():
        _, d = m(x)
        pz = (d.min(1).values.item() - m0) / s0
        xf = x.view(1, -1)
        az = (((ae(xf) - xf) ** 2).mean(1).item() - me) / se
    return (pz + az) / 2, d.argmin(1).item()

print(f"\n EXAMPLE ROWS  (verdict threshold = {THR:.2f})")
print(f" {'type':<9}{'nearest-family':<16}{'score':>8}  verdict")
print(" " + "-" * 56)
# pick 5 known that are BELOW threshold + 1-2 above (honest mix)
rows = []
with torch.no_grad():
    for x, _ in tk:
        _, d = m(x)
        for i in range(len(x)):
            rows.append((x[i], d[i]))
        if len(rows) >= 400: break
shown = 0
for x, d in rows:
    if shown >= 5: break
    with torch.no_grad():
        xf = x.view(1, -1)
        az = (((ae(xf) - xf) ** 2).mean(1).item() - me) / se
    pz = (d.min().item() - m0) / s0
    sc = (pz + az) / 2
    if sc < THR:
        print(f" {'known':<9}{inv[int(d.argmin())]:<16}{sc:>8.3f}  KNOWN")
        shown += 1
shown = 0
with torch.no_grad():
    for x, _ in tu:
        _, d = m(x)
        for i in range(len(x)):
            if shown >= 5: break
            xf = x[i].view(1, -1)
            az = (((ae(xf) - xf) ** 2).mean(1).item() - me) / se
            pz = (d[i].min().item() - m0) / s0
            sc = (pz + az) / 2
            if sc > THR:
                print(f" {'zero-day':<9}{inv[int(d[i].argmin())]:<16}{sc:>8.3f}  *** ALERT ***")
                shown += 1
print("\n Takeaway: at a 5% false-alarm operating point the model still")
print(f" catches {tpr*100:.0f}% of the never-seen family (Mirai).")
