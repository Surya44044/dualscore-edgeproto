#!/usr/bin/env python3
import json, csv, numpy as np
EXPS = ['h1_mirai','h2_recon','h3_spoofing','h4_web_bruteforce','h5_dos','h5b_dos_ddos']
SEEDS = ['42','123','2025']
print(f"{'exp':<20}{'baseAU':>14}{'hybAU':>14}")
rows = []
for e in EXPS:
    baus, hyaus = [], []
    for s in SEEDS:
        b = json.load(open(f'results/tables/ciciot2023_50k/oe_seeds/{e}_s{s}_base.json'))
        h = json.load(open(f'results/tables/ciciot2023_50k/oe_seeds/{e}_s{s}_hybrid.json'))
        baus.append(b.get('AUROC') or b.get('open_set_auroc'))
        hyaus.append(h['AUROC'])
    def f(x): return f"{np.mean(x):.4f}+-{np.std(x):.4f}"
    print(f"{e:<20}{f(baus):>14}{f(hyaus):>14}")
    rows.append([e, f'{np.mean(baus):.4f}', f'{np.std(baus):.4f}', f'{np.mean(hyaus):.4f}', f'{np.std(hyaus):.4f}'])
with open('results/tables/ciciot2023_50k/oe_summary.csv','w',newline='') as fp:
    w = csv.writer(fp)
    w.writerow(['exp','baseAU_mean','baseAU_std','hybAU_mean','hybAU_std'])
    w.writerows(rows)
print('saved: oe_summary.csv')
