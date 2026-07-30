import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    csv_path = "results/tables/rtiot2022_hybrid_comparison.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} does not exist.")
        return
        
    df = pd.read_csv(csv_path)
    os.makedirs("results/figures", exist_ok=True)
    
    # Set style
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'font.size': 12, 'figure.autolayout': True})
    
    # 1. AUROC Comparison Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    df_auroc = df.dropna(subset=['AUROC'])
    sns.barplot(data=df_auroc, x='experiment', y='AUROC', hue='model', ax=ax)
    ax.set_title("RT-IoT2022 Open-Set AUROC Comparison Across All Models & Hybrid Rules", fontsize=14, fontweight='bold')
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("AUROC (Higher is Better)")
    ax.set_xlabel("Holdout Experiment")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    fig.savefig("results/figures/rtiot2022_hybrid_auroc_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved results/figures/rtiot2022_hybrid_auroc_comparison.png")
    
    # 2. AUPR_Out Comparison Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    df_aupr = df.dropna(subset=['AUPR_Out'])
    sns.barplot(data=df_aupr, x='experiment', y='AUPR_Out', hue='model', ax=ax)
    ax.set_title("RT-IoT2022 Open-Set AUPR (Unknown Positive) Comparison", fontsize=14, fontweight='bold')
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("AUPR_Out (Higher is Better)")
    ax.set_xlabel("Holdout Experiment")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    fig.savefig("results/figures/rtiot2022_hybrid_aupr_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved results/figures/rtiot2022_hybrid_aupr_comparison.png")
    
    # 3. FPRat95_TPR Comparison Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    df_fpr = df.dropna(subset=['FPRat95_TPR'])
    sns.barplot(data=df_fpr, x='experiment', y='FPRat95_TPR', hue='model', ax=ax)
    ax.set_title("RT-IoT2022 False Positive Rate at 95% TPR (Lower is Better)", fontsize=14, fontweight='bold')
    ax.set_ylabel("FPR @ 95% TPR (Lower is Better)")
    ax.set_xlabel("Holdout Experiment")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    fig.savefig("results/figures/rtiot2022_hybrid_fpr95_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved results/figures/rtiot2022_hybrid_fpr95_comparison.png")
    
    # 4. Resource Tradeoff Scatter Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    df_res = df.dropna(subset=['savedmodelsize_kb', 'AUROC'])
    
    sns.scatterplot(
        data=df_res, 
        x='savedmodelsize_kb', 
        y='AUROC', 
        hue='model', 
        style='experiment',
        s=120, 
        ax=ax
    )
    
    ax.set_title("Resource Tradeoff: Storage Size (KB) vs Open-Set AUROC", fontsize=14, fontweight='bold')
    ax.set_xlabel("Model Storage Size (KB, Log Scale)")
    ax.set_ylabel("AUROC")
    ax.set_xscale('log')
    ax.set_ylim(0.75, 1.02)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    fig.savefig("results/figures/rtiot2022_hybrid_resource_tradeoff.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved results/figures/rtiot2022_hybrid_resource_tradeoff.png")

if __name__ == "__main__":
    main()
