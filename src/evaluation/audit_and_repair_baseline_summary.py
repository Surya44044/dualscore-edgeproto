import os
import glob
import json
import pandas as pd
import re
import time
import torch
import yaml
from src.models.edgeproto import get_model
from src.data.feature_mapping import get_dataloaders

def get_modelsizekb(path):
    if os.path.exists(path):
        return os.path.getsize(path) / 1024.0
    return None

def parse_traintime(log_file):
    if not log_file or not os.path.exists(log_file):
        return None
    with open(log_file, 'r') as f:
        content = f.read()
    time_match = re.search(r"Elapsed \(wall clock\) time [^:]+:\s*([0-9:.]+)", content)
    if time_match:
        time_str = time_match.group(1)
        parts = time_str.split(':')
        if len(parts) == 3: # h:mm:ss
            return float(parts[0])*3600 + float(parts[1])*60 + float(parts[2])
        elif len(parts) == 2: # m:ss
            return float(parts[0])*60 + float(parts[1])
    return None

def benchmark_inference(prefix, config_path, model_path):
    if not os.path.exists(model_path):
        return None
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    test_known = f"data/splits/{prefix}_test_known.csv"
    if not os.path.exists(test_known):
        return None
        
    import pandas as pd
    import numpy as np
    
    df = pd.read_csv(test_known)
    X = df.drop(columns=['label']).values.astype(np.float32)
    # take 1000
    X = X[:1000]
    X = X.reshape(-1, 4, 12)
    
    # Needs to match num_classes used during training.
    # For H1: Benign, DDoS, DoS, Recon (4 known classes)
    # For H2: Benign, DoS, ARP_Spoofing, DDoS (4 known classes)
    # For H3: Benign, Recon, ARP_Spoofing, DDoS (4 known classes)
    # Let's just hardcode 4 since we know all our holdouts use 4 known classes.
    model = get_model(config, num_classes=4)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    t = torch.tensor(X)
    with torch.no_grad():
        start = time.time()
        model(t) # forward pass computes distances
        end = time.time()
        
    return end - start

def main():
    print("Task 1 - Baseline result audit and resource-field repair")
    csv_path = "results/tables/rtiot2022_edgeproto_vs_baselines.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        return
        
    df = pd.read_csv(csv_path)
    
    print("Verifying baseline JSONs...")
    baseline_jsons = glob.glob("results/tables/baselines/*_results.json")
    for b in baseline_jsons:
        with open(b, 'r') as f:
            data = json.load(f)
            assert "status" in data, f"Missing status in {b}"
            assert "AUROC" in data, f"Missing AUROC in {b}"
            assert "unknownscoredirection explanation" in data, f"Missing unknownscoredirection explanation in {b}"
            assert "traintimeseconds" in data, f"Missing traintimeseconds in {b}"
            assert "inferencetime1000samplesseconds" in data, f"Missing inferencetime1000samplesseconds in {b}"
            
    print("Verifying EdgeProto JSONs...")
    assert len(glob.glob("results/tables/rt_h1_arp_spoofing*results.json")) > 0
    assert len(glob.glob("results/tables/rt_h2_recon*results.json")) > 0
    assert len(glob.glob("results/tables/rt_h3_dos*results.json")) > 0
    
    print("Repairing EdgeProto fields...")
    # Fill EdgeProto fields
    for idx, row in df.iterrows():
        if row['model'] == 'EdgeProto (Ours)':
            exp = row['experiment']
            prefix = ""
            model_path = ""
            log_path = ""
            config_path = ""
            
            if exp == "RT-H1":
                prefix = "rt_h1_arp_spoofing"
                # H1 snapshot fix:
                snap = "models/rt_h1_arp_spoofing/best_edgeproto.pt"
                model_path = snap
                log_path = "logs/day2_rtiot_h1_full_default_epochs.log"
                config_path = "configs/rt_h1_arp_spoofing.yaml"
            elif exp == "RT-H2":
                prefix = "rt_h2_recon"
                model_path = "models/rt_h2_recon/best_edgeproto.pt"
                # Use glob to find the run-id file or log
                logs = glob.glob("logs/rt_h2_recon_clean_*.log")
                logs = [l for l in logs if "audit" not in l]
                if logs:
                    log_path = sorted(logs)[0]
                config_path = "configs/rt_h2_recon.yaml"
            elif exp == "RT-H3":
                prefix = "rt_h3_dos"
                model_path = "models/rt_h3_dos/best_edgeproto.pt"
                logs = glob.glob("logs/rt_h3_dos_clean_*.log")
                logs = [l for l in logs if "audit" not in l]
                if logs:
                    log_path = sorted(logs)[0]
                config_path = "configs/rt_h3_dos.yaml"
                
            msize = get_modelsizekb(model_path)
            ttime = parse_traintime(log_path)
            itime = benchmark_inference(prefix, config_path, model_path)
            
            df.at[idx, 'modelsizekb'] = msize if msize is not None else row['modelsizekb']
            df.at[idx, 'traintimeseconds'] = ttime if ttime is not None else row['traintimeseconds']
            df.at[idx, 'inferencetime1000samplesseconds'] = itime if itime is not None else row['inferencetime1000samplesseconds']

    out_csv = "results/tables/rtiot2022_edgeproto_vs_baselines_REPAIRED.csv"
    out_md = "results/tables/rtiot2022_edgeproto_vs_baselines_REPAIRED.md"
    
    df.to_csv(out_csv, index=False)
    try:
        md_table = df.to_markdown(index=False)
    except:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tabulate"])
        md_table = df.to_markdown(index=False)
        
    with open(out_md, 'w') as f:
        f.write(md_table)
        
    print(f"Repaired files saved to {out_csv} and {out_md}")
    
if __name__ == '__main__':
    main()
