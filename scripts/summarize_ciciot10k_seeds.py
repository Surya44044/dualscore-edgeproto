#!/usr/bin/env python3
import json, csv, numpy as np
EXPS = ['h1_mirai','h2_recon','h3_spoofing','h4_web_bruteforce','h5_dos','h5b_dos_ddos']
SEEDS = ['42','123','2025']
oscr = {}
with open('results/tables/ciciot2023_10k/seeds_oscr.csv') as f:
    for line in f:
        p = line.strip().split(',')
        if len(p) == 4 and p[0] in EXPS and p[2] == 'Hybrid-Mean-50-50':
            oscr[(p[0], p[1])] = float(p[3])
print(f"{'exp':<20}{'baseAU':>14}{'hybAU':>14}{'hybOSCR':>14}")
rows_out = []
for e in EXPS:
    baus, hyaus, hyos = [], [], []
    for s in SEEDS:
        b = json.load(open(f'results/tables/ciciot2023_10k/seeds/{e}_s{s}_base.json'))
        h = json.load(open(f'results/tables/ciciot2023_10k/seeds/{e}_s{s}_hybrid.json'))
        baus.append(b.get('AUROC') or b.get('open_set_auroc'))
        hyaus.append(h['AUROC'])
        hyos.append(oscr.get((e, s), float('nan')))
    def fmt(x):
        return f"{np.mean(x):.4f}+-{np.std(x):.4f}"
    print(f"{e:<20}{fmt(baus):>14}{fmt(hyaus):>14}{fmt(hyos):>14}")
    rows_out.append([e] + [f'{np.mean(x):.4f}' for x in (baus, hyaus, hyos)] + [f'{np.std(x):.4f}' for x in (baus, hyaus, hyos)])
with open('results/tables/ciciot2023_10k/seeds_summary.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['exp','baseAU_mean','hybAU_mean','hybOSCR_mean','baseAU_std','hybAU_std','hybOSCR_std'])
    w.writerows(rows_out)
print('saved: results/tables/ciciot2023_10k/seeds_summary.csv')
