#!/usr/bin/env python3
import json, csv
EXPS = ['h1_mirai','h2_recon','h3_spoofing','h4_web_bruteforce','h5_dos','h5b_dos_ddos']
oscr = {}
with open('results/tables/ciciot2023_10k/hybrid_oscr_corrected.csv') as f:
    for line in f:
        p = line.strip().split(',')
        if len(p) >= 3:
            oscr[(p[0], p[1])] = float(p[2])
rows = []
for e in EXPS:
    b = json.load(open(f'results/tables/ciciot2023_10k/{e}_results.json'))
    acc = b.get('known accuracy') or b.get('closed_set_metrics', {}).get('accuracy')
    base_au = b.get('AUROC') or b.get('open_set_auroc')
    base_os = b.get('OSCR') or b.get('oscr')
    h = json.load(open(f'results/tables/hybrid/{e}_Hybrid-Mean-50-50_results.json'))
    hy_au = h['AUROC']
    hy_os = oscr.get((e, 'Hybrid-Mean-50-50'), float('nan'))
    rows.append([e, f'{acc:.4f}', f'{base_au:.4f}', f'{base_os:.4f}', f'{hy_au:.4f}', f'{hy_os:.4f}'])
with open('results/tables/ciciot2023_10k/final_10k_table.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['experiment','known_acc','base_AUROC','base_OSCR','hybrid_AUROC','hybrid_OSCR_corrected'])
    w.writerows(rows)
print(f"{'experiment':<20}{'acc':>8}{'baseAU':>9}{'baseOSCR':>10}{'hybAU':>9}{'hybOSCR':>10}")
for r in rows:
    print(f"{r[0]:<20}{r[1]:>8}{r[2]:>9}{r[3]:>10}{r[4]:>9}{r[5]:>10}")
print('\nsaved: results/tables/ciciot2023_10k/final_10k_table.csv')
