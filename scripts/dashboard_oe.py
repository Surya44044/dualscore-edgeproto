#!/usr/bin/env python3
"""LIVE dashboard for the OE 50k campaign (18 experiments = 6 holdouts x 3 seeds)."""
import os, re, sys, time, subprocess, datetime
from pathlib import Path

HOME = Path.home()
LOG = HOME / "ciciot_oe.log"
ERRLOG = HOME / "ciciot_oe_errors.log"
SEEDIR = Path("results/tables/ciciot2023_50k/oe_seeds")
OSCFILE = Path("results/tables/ciciot2023_50k/oe_oscr.csv")

EXPS = ["h1_mirai","h2_recon","h3_spoofing","h4_web_bruteforce","h5_dos","h5b_dos_ddos"]
SEEDS = ["42","123","2025"]
TOTAL = len(EXPS) * len(SEEDS)
TS_RE = re.compile(r"(\w{3} \w{3} \d{1,2} \d{2}:\d{2}:\d{2}) UTC (\d{4})")

def parse_ts(s):
    m = TS_RE.search(s)
    if not m: return None
    try: return datetime.datetime.strptime(m.group(1)+" "+m.group(2), "%a %b %d %H:%M:%S %Y")
    except Exception: return None

def is_done(tag):
    try:
        if not (SEEDIR / f"{tag}_base.json").exists(): return False
        if not (SEEDIR / f"{tag}_hybrid.json").exists(): return False
        return True
    except Exception: return False

def dur(sec):
    sec = int(sec); h, rem = divmod(sec, 3600); m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"

def proc_info():
    try:
        out = subprocess.run(["pgrep","-f","run_ciciot_oe.sh"], capture_output=True, text=True).stdout.split()
        if not out: return "STOPPED", "---", "---"
        pid = out[0]
        ps = subprocess.run(["ps","-p",pid,"-o","pcpu=,pmem=,etime="], capture_output=True, text=True).stdout.strip()
        return "ALIVE", pid, ps
    except Exception:
        return "?", "?", "?"

def render():
    lines = []
    if not LOG.exists():
        return ["Run has not started yet (no log)."]
    text = LOG.read_text(errors="ignore")
    raw = text.splitlines()
    done = [f"{e}_s{s}" for e in EXPS for s in SEEDS if is_done(f"{e}_s{s}")]
    completed = len(done)
    pct = 100.0 * completed / TOTAL

    st, pid, ps = proc_info()
    start = None; ok_times = []; starts = []; oks = set()
    for line in raw:
        if "=== OE RUN START " in line: start = parse_ts(line)
        if line.startswith("===== OE START "): starts.append(line.replace("===== OE START ","").split()[0])
        if line.startswith("OK "):
            oks.add(line.split()[1]); t = parse_ts(line)
            if t: ok_times.append(t)
    current = starts[-1] if starts else "?"
    if current in oks:
        pend = [x for x in starts if x not in oks]
        current = pend[-1] if pend else "finalizing..."
    elapsed = (datetime.datetime.utcnow() - start).total_seconds() if start else 0

    lines.append("=" * 66)
    lines.append("  OE CAMPAIGN  |  CICIoT2023 @50k  |  6 holdouts x 3 seeds")
    lines.append("=" * 66)
    bar_w = 30
    filled = int(round(bar_w * pct / 100))
    lines.append(f"  [{'#'*filled}{'-'*(bar_w-filled)}]  {pct:5.1f}%   ({completed}/{TOTAL})")
    lines.append(f"  RUN       : {st}  (pid {pid})")
    lines.append(f"  CURRENT   : {current}")
    if start:
        lines.append(f"  ELAPSED   : {dur(elapsed)}")
    avg = None
    if ok_times and start:
        avg = (ok_times[-1] - start).total_seconds() / max(completed, 1)
    elif len(ok_times) >= 2:
        avg = (ok_times[-1] - ok_times[0]).total_seconds() / (len(ok_times)-1)
    if avg and completed < TOTAL:
        rem = avg * (TOTAL - completed)
        fin = datetime.datetime.utcnow() + datetime.timedelta(seconds=rem)
        lines.append(f"  AVG/EXP   : {dur(avg)}   ETA: ~{dur(rem)}  (finish ~{fin.strftime('%H:%M')} UTC)")
    elif completed >= TOTAL:
        lines.append(f"  AVG/EXP   : {dur(avg) if avg else '?'}   STATUS: COMPLETE!")
    if ERRLOG.exists() and sum(1 for _ in ERRLOG.open(errors="ignore")) > 0:
        n = sum(1 for _ in ERRLOG.open(errors="ignore"))
        lines.append(f"  ! ERRORS  : {n} (see ~/ciciot_oe_errors.log - run continues)")
    lines.append("")
    lines.append("  GRID  (rows=seeds, cols=holdouts)")
    lines.append("         " + "  ".join(f"{e[:10]:<12}" for e in EXPS))
    for s in SEEDS:
        row = f"  s{s:<6}"
        for e in EXPS:
            tag = f"{e}_s{s}"
            if is_done(tag): row += "  [ OK ]      "
            elif tag in starts: row += "  [RUN ]      "
            else: row += "  [   ]      "
        lines.append(row)
    lines.append("")
    lines.append("  - LOG TAIL (last 5):")
    for l in raw[-5:]:
        lines.append("    " + l[:100])
    lines.append("")
    lines.append(f"  [updated {datetime.datetime.utcnow().strftime('%H:%M:%S')} UTC | refresh 3s | Ctrl+C exits view only]")
    return lines

def main():
    print("\033[?25l", end="", flush=True)  # hide cursor
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
