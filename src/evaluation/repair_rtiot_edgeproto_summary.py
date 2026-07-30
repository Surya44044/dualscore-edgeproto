import os
import json
import pandas as pd
import re
import glob

def parse_log_for_metrics(log_file):
    metrics = {"AUROC": None, "runtime_seconds": None, "maxramkb": None}
    if not os.path.exists(log_file):
        return metrics
    
    with open(log_file, "r") as f:
        content = f.read()
        
        # Parse AUROC
        auroc_match = re.search(r"Open-Set Detection AUROC:\s*([0-9.]+)", content)
        if auroc_match:
            metrics["AUROC"] = float(auroc_match.group(1))
            
        # Parse runtime: Elapsed (wall clock) time (h:mm:ss or m:ss): 2:38.25
        time_match = re.search(r"Elapsed \(wall clock\) time [^:]+:\s*([0-9:.]+)", content)
        if time_match:
            time_str = time_match.group(1)
            parts = time_str.split(':')
            if len(parts) == 3: # h:mm:ss
                metrics["runtime_seconds"] = float(parts[0])*3600 + float(parts[1])*60 + float(parts[2])
            elif len(parts) == 2: # m:ss
                metrics["runtime_seconds"] = float(parts[0])*60 + float(parts[1])
                
        # Parse RAM: Maximum resident set size (kbytes): 473848
        ram_match = re.search(r"Maximum resident set size \(kbytes\):\s*([0-9]+)", content)
        if ram_match:
            metrics["maxramkb"] = int(ram_match.group(1))
            
    return metrics

def main():
    experiments = [
        {
            "name": "RT-H1",
            "prefix": "rt_h1_arp_spoofing",
            "unknown_class": "ARP_Spoofing",
            "known_classes": "Benign, DDoS, DoS, Recon",
            "json_pattern": "results/tables/rt_h1_arp_spoofing_VALIDATED_clean_results.json",
            "log_pattern": "logs/day2_rtiot_h1_full_default_epochs.log"
        },
        {
            "name": "RT-H2",
            "prefix": "rt_h2_recon",
            "unknown_class": "Recon",
            "known_classes": "Benign, DoS, ARP_Spoofing, DDoS",
            "json_pattern": "results/tables/rt_h2_recon_results.json",
            "log_pattern": "logs/rt_h2_recon_clean_*.log"
        },
        {
            "name": "RT-H3",
            "prefix": "rt_h3_dos",
            "unknown_class": "DoS",
            "known_classes": "Benign, Recon, ARP_Spoofing, DDoS",
            "json_pattern": "results/tables/rt_h3_dos_results.json",
            "log_pattern": "logs/rt_h3_dos_clean_*.log"
        }
    ]
    
    rows = []
    
    for exp in experiments:
        # Find JSON
        json_files = glob.glob(exp["json_pattern"])
        if not json_files:
            print(f"Missing JSON for {exp['name']}")
            continue
        json_file = sorted(json_files)[0] # Take first match
        
        # Find log
        log_files = glob.glob(exp["log_pattern"])
        log_files = [f for f in log_files if not f.endswith("_audit.log") and not f.endswith("_verification.txt")]
        if not log_files:
            print(f"Missing log for {exp['name']}")
            log_file = ""
        else:
            log_file = sorted(log_files)[0]
            
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        log_metrics = parse_log_for_metrics(log_file) if log_file else {}
        
        auroc = data.get("AUROC")
        if auroc is None:
            auroc = log_metrics.get("AUROC")
            
        row = {
            "experiment": exp["name"],
            "known_classes": exp["known_classes"],
            "unknown_class": exp["unknown_class"],
            "known_accuracy": data.get("known accuracy"),
            "known_precision": data.get("known precision"),
            "known_recall": data.get("known recall"),
            "known_f1": data.get("known F1") if "known F1" in data else data.get("known_f1"),
            "AUROC": auroc,
            "OSCR": data.get("OSCR"),
            "params": "3.84k",
            "macs": "9.68 KMac",
            "runtime_seconds": log_metrics.get("runtime_seconds"),
            "maxramkb": log_metrics.get("maxramkb"),
            "result_json": json_file,
            "log_file": log_file
        }
        
        rows.append(row)
        
    df = pd.DataFrame(rows)
    
    csv_path = "results/tables/rtiot2022_edgeproto_single_run_summary_FIXED.csv"
    df.to_csv(csv_path, index=False)
    print(f"Created {csv_path}")
    
    md_path = "results/tables/rtiot2022_edgeproto_single_run_summary_FIXED.md"
    try:
        markdown_table = df.to_markdown(index=False)
    except ImportError:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tabulate"])
        markdown_table = df.to_markdown(index=False)
    
    with open(md_path, "w") as f:
        f.write(markdown_table)
    print(f"Created {md_path}")
    
    summary_md_path = "logs/DAY3_EDGEPROTO_H1_H2_H3_FIXED_SUMMARY.md"
    with open(summary_md_path, "w") as f:
        f.write("# EdgeProto Fixed Summary\n\n")
        f.write(markdown_table)
        f.write("\n")
    print(f"Created {summary_md_path}")

if __name__ == "__main__":
    main()
