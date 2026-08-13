#!/usr/bin/env python3
"""CICIoT2023 3-seed run: LIVE progress + ETA (watch mode default)."""
import os, re, sys, time, argparse, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
LOG = Path.home() / "ciciot_10k_seeds.log"
ERRLOG = Path.home() / "ciciot_10k_seeds_errors.log"
SEEDIR = Path("results/tables/ciciot2023_10k/seeds")
OSCFILE = Path("results/tables/ciciot2023_10k/seeds_oscr.csv")

EXPS = ["h1_mirai","h2_recon","h3_spoofing","h4_web_bruteforce","h5_dos","h5b_dos_ddos"]
SEEDS = ["42","123","2025"]
TOTAL = len(EXPS) * len(SEEDS)

TS_RE = re.compile(r"(\w{3} \w{3} \d{1,2} \d{2}:\d{2}:\d{2}) UTC (\d{4})")

def parse_ts(line):
    m = TS_RE.search(line)
    if not m:
        return None
    try:
        return datetime.datetime.strptime(m.group(1) + " " + m.group(2), "%a %b %d %H:%M:%S %Y")
    except Exception:
        return None

def is_done(tag):
    try:
        if not (SEEDIR / f"{tag}_base.json").exists():
            return False
        if not (SEEDIR / f"{tag}_hybrid.json").exists():
            return False
        if OSCFILE.exists():
            exp, seed = tag.rsplit("_s", 1)
            for line in OSCFILE.read_text().splitlines():
                p = line.split(",")
                if len(p) == 4 and p[0] == exp and p[1] == seed and p[2] == "Hybrid-Mean-50-50":
                    return True
        return False
    except Exception:
        return False

def fmt_dur(sec):
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"

def bar(pct, width=24):
    filled = int(round(width * pct / 100.0))
    return "#" * filled + "-" * (width - filled)

def render():
    lines = []
    finished = False
    if not LOG.exists():
        return ["Run has not started yet (no log)."], False

    text = LOG.read_text(errors="ignore")
    raw = text.splitlines()
    done = [f"{e}_s{s}" for e in EXPS for s in SEEDS if is_done(f"{e}_s{s}")]
    completed = len(done)
    pct = 100.0 * completed / TOTAL

    lines.append("=" * 60)
    lines.append("CICIoT2023 3-SEED RUN - LIVE PROGRESS")
    lines.append("=" * 60)
    lines.append(f"[{bar(pct)}]  {pct:5.1f}%   ({completed}/{TOTAL} experiments done)")

    if "=== RUN END " in text:
        m = re.search(r"OK=(\d+) FAIL=(\d+)", text)
        lines.append("STATUS : FINISHED" + (f"   (OK={m.group(1)} FAIL={m.group(2)})" if m else ""))
        return lines, True

    start = None
    ok_times = []
    started_tags = []
    ok_tags = set()
    for line in raw:
        if "=== RUN START " in line:
            start = parse_ts(line)
        if line.startswith("===== START "):
            started_tags.append(line.replace("===== START ", "").split()[0])
        if line.startswith("OK "):
            ok_tags.add(line.split()[1])
            t = parse_ts(line)
            if t:
                ok_times.append(t)

    current = started_tags[-1] if started_tags else None
    if current in ok_tags:
        pending = [t for t in started_tags if t not in ok_tags]
        current = pending[-1] if pending else "preparing next experiment..."

    lines.append(f"STATUS : RUNNING")
    lines.append(f"CURRENT: {current}")
    if ERRLOG.exists() and sum(1 for _ in ERRLOG.open(errors="ignore")) > 0:
        lines.append(f"ERRORS : some logged - see ~/ciciot_10k_seeds_errors.log (run continues)")

    avg = None
    if ok_times and start:
        avg = (ok_times[-1] - start).total_seconds() / max(completed, 1)
    elif ok_times and len(ok_times) >= 2:
        avg = (ok_times[-1] - ok_times[0]).total_seconds() / (len(ok_times) - 1)

    if avg:
        remaining = TOTAL - completed
        eta = avg * remaining
        finish = datetime.datetime.utcnow() + datetime.timedelta(seconds=eta)
        lines.append(f"AVG    : {fmt_dur(avg)} per experiment")
        lines.append(f"REMAIN : ~{fmt_dur(eta)}   (finish ~{finish.strftime('%H:%M')} UTC = {finish.strftime('%H:%M')} UTC)")
    else:
        lines.append("REMAIN : estimating once the first experiments complete")
    return lines, False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="print once and exit")
    ap.add_argument("--interval", type=float, default=5.0, help="refresh seconds (default 5)")
    args = ap.parse_args()

    if args.once:
        lines, _ = render()
        print("\n".join(lines))
        return

    print("LIVE MODE  (Ctrl+C stops only this view - the VM run continues)")
    print("legend: done=#  pending=-   |   ETA uses rolling average of finished experiments\n")
    try:
        while True:
            lines, finished = render()
            sys.stdout.write("\033[H\033[J")
            print("\n".join(lines))
            print(f"[updated {datetime.datetime.utcnow().strftime('%H:%M:%S')} UTC | refresh {args.interval:.0f}s | Ctrl+C to exit]")
            sys.stdout.flush()
            if finished:
                print("[RUN FINISHED - keeping view open, Ctrl+C to exit]")
                time.sleep(30)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nLive view stopped. The VM run continues on its own.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("progress script error:", e)
