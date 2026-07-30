import os
import glob
import json
import pandas as pd

def format_number(val):
    if pd.isna(val): return ""
    if isinstance(val, float): return f"{val:.4f}"
    return str(val)

def main():
    json_files = glob.glob("results/tables/extended/*.json")
    rows = []
    
    for jf in json_files:
        with open(jf, 'r') as f:
            data = json.load(f)
            # Make sure we have unknown_class based on experiment
            if data['experiment'] == 'RT-H1': unknown = 'ARP_Spoofing'
            elif data['experiment'] == 'RT-H2': unknown = 'Recon'
            elif data['experiment'] == 'RT-H3': unknown = 'DoS'
            else: unknown = 'Unknown'
            
            row = {
                'experiment': data.get('experiment', ''),
                'unknown_class': unknown,
                'model': data.get('model', ''),
                'model_family': data.get('model_family', ''),
                'knownf1macro': data.get('knownf1macro', None),
                'AUROC': data.get('AUROC', None),
                'AUPR_Out': data.get('AUPR_Out', None),
                'FPRat95_TPR': data.get('FPRat95_TPR', None),
                'TPRat1_FPR': data.get('TPRat1_FPR', None),
                'OSCR': data.get('OSCR', None),
                'params_numeric': data.get('params_numeric', None),
                'rawweightkb_fp32': data.get('rawweightkb_fp32', None),
                'savedmodelsize_kb': data.get('savedmodelsize_kb', None),
                'traintimeseconds': data.get('traintimeseconds', None),
                'inferencetime1000samplesseconds': data.get('inferencetime1000samplesseconds', None),
                'scorewasinverted': data.get('scorewasinverted', False),
                'notes': data.get('notes', '')
            }
            rows.append(row)
            
    df = pd.DataFrame(rows)
    
    if len(df) == 0:
        print("No JSON files found in results/tables/extended/")
        return
        
    # Sort
    df = df.sort_values(by=['experiment', 'model'])
    
    csv_path = "results/tables/rtiot2022_publication_metrics_extended.csv"
    md_path = "results/tables/rtiot2022_publication_metrics_extended.md"
    summary_path = "logs/DAY4_RTIOT_PUBLICATION_METRICS_SUMMARY.md"
    
    df.to_csv(csv_path, index=False)
    
    df_md = df.copy()
    for col in df_md.columns:
        df_md[col] = df_md[col].apply(format_number)
    
    with open(md_path, 'w') as f:
        f.write(df_md.to_markdown(index=False))
        
    # Generate Interpretation Summary
    summary = ["# RT-IoT2022 Publication Metrics Summary\n"]
    
    for exp in sorted(df['experiment'].unique()):
        summary.append(f"## {exp}")
        subdf = df[df['experiment'] == exp].copy()
        
        # Best AUROC
        subdf_sorted = subdf.dropna(subset=['AUROC']).sort_values('AUROC', ascending=False)
        if not subdf_sorted.empty:
            summary.append("**AUROC Ranking:**")
            for i, row in subdf_sorted.iterrows():
                summary.append(f"- {row['model']}: {row['AUROC']:.4f}")
                
        # Best AUPR_Out
        subdf_sorted_aupr = subdf.dropna(subset=['AUPR_Out']).sort_values('AUPR_Out', ascending=False)
        if not subdf_sorted_aupr.empty:
            summary.append("\n**AUPR_Out Ranking:**")
            for i, row in subdf_sorted_aupr.iterrows():
                summary.append(f"- {row['model']}: {row['AUPR_Out']:.4f}")
                
        # FPRat95_TPR (lower is better)
        subdf_sorted_fpr = subdf.dropna(subset=['FPRat95_TPR']).sort_values('FPRat95_TPR', ascending=True)
        if not subdf_sorted_fpr.empty:
            summary.append("\n**FPRat95_TPR Ranking (Lower is Better):**")
            for i, row in subdf_sorted_fpr.iterrows():
                summary.append(f"- {row['model']}: {row['FPRat95_TPR']:.4f}")
                
        summary.append("")
        
    summary.append("## Interpretation Notes")
    summary.append("- Base EdgeProto is best AUROC on H1.")
    summary.append("- EdgeProto/PO is competitive on H2, with PO possibly best.")
    summary.append("- EdgeProto is weak on H3 compared to Autoencoder/RF/IF.")
    summary.append("- EdgeProto has very low params/raw weights and tiny inference cost, but MLP checkpoint may be slightly smaller due serialization.")
    summary.append("- Tree models have much larger serialized size and slower inference than neural tiny models.")
    
    with open(summary_path, 'w') as f:
        f.write("\n".join(summary))
        
    print(f"Summary written to {summary_path}")

if __name__ == '__main__':
    main()
