#!/usr/bin/env python3
"""Append one experiment's metrics to a summary CSV (robust to JSON key names)."""
import sys, json, glob

def search(obj, name):
    out = None
    def walk(o):
        nonlocal out
        if out is not None:
            return
        if isinstance(o, dict):
            for k, v in o.items():
                if out is not None:
                    return
                if name in str(k).lower() and isinstance(v, (int, float)):
                    out = v
                    return
                walk(v)
        elif isinstance(o, list):
            for x in o:
                if out is not None:
                    return
                walk(x)
    walk(obj)
    return out

exp = sys.argv[1]
summary = sys.argv[2]
jsons = glob.glob(f"results/tables/ciciot2023_smoke/{exp}_results.json")
if not jsons:
    print(f"ERROR: no results json for {exp}")
    sys.exit(1)
j = json.load(open(jsons[0]))
acc = search(j, "acc")
auroc = search(j, "auroc")
oscr = search(j, "oscr")
if acc is None or auroc is None or oscr is None:
    print(f"ERROR: could not find metrics in {jsons[0]}")
    sys.exit(1)
with open(summary, "a") as f:
    f.write(f"{exp},{acc:.4f},{auroc:.4f},{oscr:.4f}\n")
print(f"{exp} -> acc {acc:.4f} | AUROC {auroc:.4f} | OSCR {oscr:.4f}")
