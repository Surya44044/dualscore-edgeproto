import json, csv, numpy as np
EXPS = ['h1_mirai','h2_recon','h3_spoofing','h4_web_bruteforce','h5_dos','h5b_dos_ddos']
# non-OE hybrid (from master_table) + OE hybrid (from oe_summary)
non = { 'h1_mirai':(0.9276,0.0164),'h2_recon':(0.8923,0.0058),'h3_spoofing':(0.8222,0.0060),'h4_web_bruteforce':(0.8337,0.0066),'h5_dos':(0.3220,0.0086),'h5b_dos_ddos':(0.6447,0.0259) }
oe = {}
for line in open('results/tables/ciciot2023_50k/oe_summary.csv'):
    p = line.strip().split(',')
    if len(p) == 5 and p[0] in EXPS:
        oe[p[0]] = (float(p[3]), float(p[4]))
print(f"{'exp':<20}{'nonOE_hyb':>14}{'OE_hyb':>14}")
for e in EXPS:
    print(f"{e:<20}{non[e][0]:>10.4f}+-{non[e][1]:<5.4f}{oe[e][0]:>10.4f}+-{oe[e][1]:<5.4f}")
