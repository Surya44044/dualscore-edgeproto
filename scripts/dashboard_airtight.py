#!/usr/bin/env python3
"""LIVE dashboard: airtight 3-seed calibration fusion (18 = 6 exp x 3 seeds)."""
import os, re, sys, time, subprocess, datetime
from pathlib import Path

HOME = Path.home()
LOG = HOME / "ciciot_airtight.log"
CSV = Path("results/tables/ciciot2023_50k/calib/calib_fusion_airtight.csv")
MODELS = Path("models/calib")

EXPS = ["h1_mirai","h2_recon","h3_spoofing","h4_web_bruteforce","h5_dos","h5b_dos_ddos"]
SEEDS = ["42","123","2025"]
TOTAL = len(EXPS) * len(SEEDS)
TS_RE = re.compile(r"(\w{3} \w{3} \d{1,2} \d{2}:\d{2}:\d{2}) UTC (\d{4})")

def parse_ts(s):
    m = TS_RE.search(s)
    if not m: return None
    try: return datetime.datetime.strptime(m.group(1)+" "+m.group(2), "%a %b %d %H:%M:%S %Y")
    except Exception: return None

def done_pairs():
    pairs = set()
    if CSV.exists():
        for line in CSV.read_text().splitlines()[1:]:
            p = line.split(",")
            if len(p) >= 4 and p[0] in EXPS and p[1] in SEEDS:
                pairs.add((p[0], p[1]))
    return pairs

def dur(sec):
    sec = int(sec); h, rem = divmod(sec, 3600); m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"

def proc_alive():
    try:
        out = subprocess.run(["pgrep","-f","calib_fusion_airtight"], capture_output=True, text=True).stdout.split()
        return out[0] if out else None
    except Exception:
        return None

def render():
    lines = []
    done = done_pairs()
    completed = len(done)
    pct = 100.0 * completed / TOTAL
    pid = proc_alive()

    # current experiment: last AE checkpoint file or last CSV row
    aexps = sorted(MODELS.glob("*_ae.pt")) if MODELS.exists() else []
    current = "preparing..."
    if aexps:
        last = aexps[-1].stem.replace("_ae","")
        current = last
    elif LOG.exists() and LOG.read_text(errors="ignore").strip():
        current = "starting first AE..."

    lines.append("=" * 62)
    lines.append("  AIRTIGHT FUSION  |  6 holdouts x 3 seeds  |  per-seed AEs")
    lines.append("=" * 62)
    bar_w = 28
    filled = int(round(bar_w * pct / 100))
    lines.append(f"  [{'#'*filled}{'-'*(bar_w-filled)}]  {pct:5.1f}%   ({completed}/{TOTAL})")
    lines.append(f"  RUN     : {'ALIVE (pid '+pid+')' if pid else 'STOPPED/FINISHED'}")
    lines.append(f"  CURRENT : {current}")

    # ETA from elapsed
    start = None
    if LOG.exists():
        for line in LOG.read_text(errors="ignore").splitlines():
            if not start:
                start = parse_ts(line)
    if start and completed > 0:
        el = (datetime.datetime.utcnow() - start).total_seconds()
        avg = el / completed
        rem = avg * (TOTAL - completed)
        fin = datetime.datetime.utcnow() + datetime.timedelta(seconds=rem)
        lines.append(f"  ELAPSED : {dur(el)}   AVG/EXP: {dur(avg)}")
        if completed < TOTAL:
            lines.append(f"  ETA     : ~{dur(rem)}  (finish ~{fin.strftime('%H:%M')} UTC)")
        else:
            lines.append("  ETA     : COMPLETE!")
    else:
        lines.append("  ELAPSED : starting...")

    lines.append("")
    lines.append("  GRID (rows=seeds, cols=holdouts)")
    lines.append("         " + "  ".join(f"{e[:9]:<11}" for e in EXPS))
    for s in SEEDS:
        row = f"  s{s:<6}"
        for e in EXPS:
            if (e, s) in done: row += "  [ OK ]     "
            elif (MODELS / f"{e}_s{s}_ae.pt").exists(): row += "  [AE ]     "
            elif f"{e}_s{s}" in current: row += "  [RUN]     "
            else: row += "  [   ]     "
        lines.append(row)

    lines.append("")
    lines.append("  - LOG TAIL (last 4):")
    if LOG.exists():
        for l in LOG.read_text(errors="ignore").splitlines()[-4:]:
            lines.append("    " + l[:95])
    lines.append("")
    lines.append(f"  [updated {datetime.datetime.utcnow().strftime('%H:%M:%S')} UTC | 3s | Ctrl+C exits view only]")
    return lines

def main():
    print("\033[?25l", end="", flush=True)
    try:
        while True:
            sys.stdout.write("\033[H\033[J")
            for l in render():
                print(l)
            sys.stdout.flush()
            time.sleep(3)
    except KeyboardInterrupt:
        pass
    finally:
        print("\033[?25h", end="", flush=True)
        print("\nDashboard closed. The run continues on the VM.")

if __name__ == "__main__":
    main()
