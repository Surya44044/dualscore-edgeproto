import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def main():
    os.makedirs("results/figures", exist_ok=True)
    df = pd.read_csv("results/tables/rtiot2022_edgeproto_vs_baselines.csv")
    
    # Clean up model names for the plot
    df_auroc = df.dropna(subset=['AUROC'])
    
    # 1. AUROC Comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_auroc, x='experiment', y='AUROC', hue='model')
    plt.title('Open-Set AUROC Comparison')
    plt.ylim(0, 1.0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('results/figures/rtiot2022_auroc_comparison.png')
    plt.close()
    
    # 2. Known F1 Comparison
    df_f1 = df.dropna(subset=['knownf1macro'])
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_f1, x='experiment', y='knownf1macro', hue='model')
    plt.title('Known Macro F1 Comparison')
    plt.ylim(0, 1.0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('results/figures/rtiot2022_known_f1_comparison.png')
    plt.close()
    
    # 3. Model Size vs AUROC
    # We will try to plot it if modelsizekb exists
    df_size = df.dropna(subset=['AUROC']).copy()
    
    # EdgeProto size approximation if missing
    def fix_size(row):
        if pd.isna(row['modelsizekb']) and "EdgeProto" in row['model']:
            # 3.84k parameters * 4 bytes roughly
            return 15.0
        return row['modelsizekb']
        
    df_size['modelsizekb'] = df_size.apply(fix_size, axis=1)
    df_size = df_size.dropna(subset=['modelsizekb'])
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_size, x='modelsizekb', y='AUROC', hue='model', style='experiment', s=150)
    plt.title('Model Size vs Open-Set AUROC')
    if df_size['modelsizekb'].max() > df_size['modelsizekb'].min() * 10:
        plt.xscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('results/figures/rtiot2022_model_size_vs_auroc.png')
    plt.close()
    
    print("Plots generated successfully in results/figures/")

if __name__ == '__main__':
    main()
