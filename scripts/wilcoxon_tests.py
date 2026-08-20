#!/usr/bin/env python3
"""Wilcoxon signed-rank: base vs hybrid across 3 seeds (per dataset+metric)."""
import json, numpy as np
from scipy.stats import wilcoxon

EXPS = ['h1_mirai','h2_recon','h3_spoofing','h4_web_bruteforce','h5_dos','h5b_dos_ddos']
SEEDS = ['42','123','2025']

print("=== CICIoT 50k: base vs hybrid AUROC (3 seeds) ===")
for e in EXPS:
    base, hyb = [], []
    for s in SEEDS:
        b = json.load(open(f'results/tables/ciciot2023_50k/seeds/{e}_s{s}_base.json'))
        h = json.load(open(f'results/tables/ciciot2023_50k/seeds/{e}_s{s}_hybrid.json'))
        base.append(b.get('AUROC') or b.get('open_set_auroc'))
        hyb.append(h['AUROC'])
    d = np.array(hyb) - np.array(base)
    try:
        stat, p = wilcoxon(d)
        sig = 'SIGNIFICANT' if p < 0.05 else 'n.s.'
        print(f"{e:<20} base={np.mean(base):.3f} hyb={np.mean(hyb):.3f} diff={np.mean(d):+.3f} p={p:.3f} {sig}")
    except Exception as ex:
        print(f"{e:<20} wilcoxon unavailable (n too small or ties): {ex}")
