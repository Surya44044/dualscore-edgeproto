import pandas as pd
import glob
import json
import os

def main():
    # Load EdgeProto FIXED summary
    edge_proto_csv = "results/tables/rtiot2022_edgeproto_single_run_summary_FIXED.csv"
    if not os.path.exists(edge_proto_csv):
        print(f"Error: {edge_proto_csv} not found.")
        return
        
    df_edge = pd.read_csv(edge_proto_csv)
    # EdgeProto CSV columns:
    # experiment,known_classes,unknown_class,known_accuracy,known_precision,known_recall,known_f1,AUROC,OSCR,params,macs,runtime_seconds,maxramkb,result_json,log_file
    
    # We want columns:
    # experiment, unknown_class, model, knownf1macro, AUROC, OSCR, params, modelsizekb, traintimeseconds, inferencetime1000samplesseconds, notes
    
    formatted_rows = []
    
    for _, row in df_edge.iterrows():
        formatted_rows.append({
            "experiment": row["experiment"],
            "unknown_class": row["unknown_class"],
            "model": "EdgeProto (Ours)",
            "knownf1macro": row.get("known_f1"),
            "AUROC": row.get("AUROC"),
            "OSCR": row.get("OSCR"),
            "params": row.get("params"),
            "modelsizekb": None, # or maxramkb
            "traintimeseconds": row.get("runtime_seconds"),
            "inferencetime1000samplesseconds": None,
            "notes": "EdgeProto run"
        })
        
    # Load baselines
    baseline_files = glob.glob("results/tables/baselines/*_results.json")
    for bfile in baseline_files:
        with open(bfile, "r") as f:
            data = json.load(f)
            
        if data.get("status") == "failed":
            print(f"Warning: {bfile} failed. Skipping or logging error.")
            formatted_rows.append({
                "experiment": data["prefix"],
                "unknown_class": data["unknown_class"],
                "model": data["baseline"],
                "knownf1macro": None,
                "AUROC": None,
                "OSCR": None,
                "params": None,
                "modelsizekb": None,
                "traintimeseconds": None,
                "inferencetime1000samplesseconds": None,
                "notes": f"FAILED: {data.get('error')}"
            })
            continue
            
        # Map prefix to experiment name
        exp_name = data["prefix"]
        if exp_name == "rt_h1_arp_spoofing":
            exp_name = "RT-H1"
        elif exp_name == "rt_h2_recon":
            exp_name = "RT-H2"
        elif exp_name == "rt_h3_dos":
            exp_name = "RT-H3"
            
        formatted_rows.append({
            "experiment": exp_name,
            "unknown_class": data["unknown_class"],
            "model": data["baseline"],
            "knownf1macro": data.get("knownf1macro"),
            "AUROC": data.get("AUROC"),
            "OSCR": data.get("OSCR"),
            "params": data.get("params"),
            "modelsizekb": data.get("modelsizekb"),
            "traintimeseconds": data.get("traintimeseconds"),
            "inferencetime1000samplesseconds": data.get("inferencetime1000samplesseconds"),
            "notes": "Baseline run"
        })
        
    df_all = pd.DataFrame(formatted_rows)
    
    # Sort by experiment then model
    df_all = df_all.sort_values(by=["experiment", "model"])
    
    csv_out = "results/tables/rtiot2022_edgeproto_vs_baselines.csv"
    df_all.to_csv(csv_out, index=False)
    print(f"Created {csv_out}")
    
    md_out = "results/tables/rtiot2022_edgeproto_vs_baselines.md"
    try:
        md_table = df_all.to_markdown(index=False)
    except Exception:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tabulate"])
        md_table = df_all.to_markdown(index=False)
        
    with open(md_out, "w") as f:
        f.write(md_table)
    print(f"Created {md_out}")
    
    summary_log = "logs/DAY3_RTIOT_BASELINE_COMPARISON_SUMMARY.md"
    with open(summary_log, "w") as f:
        f.write("# RT-IoT2022 EdgeProto vs Baselines\n\n")
        f.write(md_table)
        f.write("\n")
    print(f"Created {summary_log}")

if __name__ == "__main__":
    main()
