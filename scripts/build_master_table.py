#!/usr/bin/env python3
"""Final CICIoT2023 master table: 3-seed mean+/-std from saved per-seed JSONs."""
import json, csv, numpy as np
EXPS = ['h1_mirai','h2_recon','h3_spoofing','h4_web_bruteforce','h5_dos','h5b_dos_ddos']
SEEDS = ['42','123','2025']
oscr = {}
for line in open('results/tables/ciciot2023_50k/seeds_oscr.csv'):
    p = line.strip().split(',')
    if len(p) == 4 and p[2] == 'Hybrid-Mean-50-50':
        oscr[(p[0], p[1])] = float(p[3])
print(f"{'exp':<20}{'hybAUROC':>16}{'hybOSCR':>16}")
rows = []
for e in EXPS:
    hys = []
    for s in SEEDS:
        h = json.load(open(f'results/tables/ciciot2023_50k/seeds/{e}_s{s}_hybrid.json'))
        hys.append(h['AUROC'])
    os_ = [oscr[(e, s)] for s in SEEDS]
    def f(x): return f"{np.mean(x):.4f}+-{np.std(x):.4f}"
    print(f"{e:<20}{f(hys):>16}{f(os_):>16}")
    rows.append([e, f'{np.mean(hys):.4f}', f'{np.std(hys):.4f}', f'{np.mean(os_):.4f}', f'{np.std(os_):.4f}'])
with open('results/tables/ciciot2023_50k/master_table.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['exp','hybAUROC_mean','hybAUROC_std','hybOSCR_mean','hybOSCR_std'])
    w.writerows(rows)
print('saved: master_table.csv')
