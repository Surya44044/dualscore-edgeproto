import json
import os
import glob
import re
import pandas as pd

def parse_time_output(log_file):
    runtime = None
    max_ram = None
    if not os.path.exists(log_file):
        return runtime, max_ram
        
    with open(log_file, 'r') as f:
        content = f.read()
        
    # parse elapsed time: "Elapsed (wall clock) time (h:mm:ss or m:ss): 2:27.06"
    time_match = re.search(r"Elapsed \(wall clock\) time [^:]+: (\d+):([\d\.]+)", content)
    if time_match:
        mins = float(time_match.group(1))
        secs = float(time_match.group(2))
        runtime = mins * 60 + secs
        
    # parse max ram: "Maximum resident set size (kbytes): 472696"
    ram_match = re.search(r"Maximum resident set size \(kbytes\): (\d+)", content)
    if ram_match:
        max_ram = int(ram_match.group(1))
        
    return runtime, max_ram
    
def main():
    experiments = [
        {
            "name": "RT-H1",
            "json_pattern": "results/tables/rt_h1_arp_spoofing_results.json",
            "log_pattern": "logs/day2_rtiot_h1_full_default_epochs.log", # or similar
            "unknown_class": "ARP_Spoofing"
        },
        {
            "name": "RT-H2",
            "json_pattern": "results/tables/rt_h2_recon_results.json",
            "log_pattern": "logs/rt_h2_recon_clean_*.log",
            "unknown_class": "Recon"
        },
        {
            "name": "RT-H3",
            "json_pattern": "results/tables/rt_h3_dos_results.json",
            "log_pattern": "logs/rt_h3_dos_clean_*.log",
            "unknown_class": "DoS"
        }
    ]
    
    rows = []
    
    for exp in experiments:
        # Find JSON
        json_matches = glob.glob(exp["json_pattern"])
        if not json_matches:
            print(f"Skipping {exp['name']}: No JSON found.")
            continue
        json_file = json_matches[0]
        
        # Check for VALIDATED_clean_results.json explicitly for H1 if present
        if exp["name"] == "RT-H1":
            clean_h1 = "results/tables/rt_h1_arp_spoofing_VALIDATED_clean_results.json"
            if os.path.exists(clean_h1):
                json_file = clean_h1
                
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        # Find Log
        log_matches = glob.glob(exp["log_pattern"])
        log_matches = [m for m in log_matches if not m.endswith("_audit.log")] # Exclude audit logs
        log_file = log_matches[-1] if log_matches else None
        
        runtime, max_ram = parse_time_output(log_file) if log_file else (None, None)
        
        row = {
            "experiment": exp["name"],
            "known_classes": "",
            "unknown_class": exp["unknown_class"],
            "known_accuracy": data.get("known accuracy"),
            "known_precision": data.get("known precision"),
            "known_recall": data.get("known recall"),
            "known_f1": data.get("known F1"),
            "AUROC": data.get("Open-Set Detection AUROC"),
            "OSCR": data.get("OSCR"),
            "params": "3.84k",
            "macs": "9.68 KMac",
            "runtime_seconds": runtime,
            "max_ram_kb": max_ram,
            "result_json": json_file,
            "log_file": log_file
        }
        
        if exp["name"] == "RT-H1":
            row["known_classes"] = "Benign, DDoS, Recon, DoS"
        elif exp["name"] == "RT-H2":
            row["known_classes"] = "Benign, DDoS, DoS, ARP_Spoofing"
        elif exp["name"] == "RT-H3":
            row["known_classes"] = "Benign, DDoS, Recon, ARP_Spoofing"
            
        rows.append(row)
        
    df = pd.DataFrame(rows)
    csv_out = "results/tables/rtiot2022_edgeproto_single_run_summary.csv"
    df.to_csv(csv_out, index=False)
    print(f"Created {csv_out}")
    
    # Markdown
    md_out = "logs/DAY2_RTIOT_H1_H2_H3_SINGLE_RUN_SUMMARY.md"
    with open(md_out, 'w') as f:
        f.write("# RT-IoT2022 Holdout Experiments Single Run Summary\n\n")
        f.write("> **WARNING**: These are single-run RT-IoT2022 results. They are not final paper results until baselines, repeated seeds, quantization evaluation, and CICIoT2023 validation are completed.\n\n")
        
        # Write table
        f.write(df[['experiment', 'unknown_class', 'known_accuracy', 'known_f1', 'AUROC', 'OSCR', 'runtime_seconds', 'max_ram_kb']].to_markdown(index=False))
        f.write("\n")
        
    print(f"Created {md_out}")

if __name__ == "__main__":
    main()
