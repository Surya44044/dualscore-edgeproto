import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def main():
    os.makedirs("results/figures/extended", exist_ok=True)
    df = pd.read_csv("results/tables/rtiot2022_publication_metrics_extended.csv")
    
    df_auroc = df.dropna(subset=['AUROC'])
    
    # 1. AUROC Comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_auroc, x='experiment', y='AUROC', hue='model')
    plt.title('Extended Open-Set AUROC Comparison')
    plt.ylim(0, 1.0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('results/figures/extended/rtiot2022_auroc_comparison_extended.png')
    plt.close()
    
    # 2. AUPR_Out Comparison
    df_aupr = df.dropna(subset=['AUPR_Out'])
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_aupr, x='experiment', y='AUPR_Out', hue='model')
    plt.title('Extended Open-Set AUPR Comparison')
    plt.ylim(0, 1.0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('results/figures/extended/rtiot2022_aupr_comparison_extended.png')
    plt.close()
    
    # 3. FPRat95_TPR Comparison
    df_fpr = df.dropna(subset=['FPRat95_TPR'])
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_fpr, x='experiment', y='FPRat95_TPR', hue='model')
    plt.title('FPR at 95% TPR Comparison (Lower is Better)')
    plt.ylim(0, 1.0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('results/figures/extended/rtiot2022_fpr95_comparison_extended.png')
    plt.close()
    
    # 4. Model Size vs AUROC
    df_size = df.dropna(subset=['AUROC', 'savedmodelsize_kb']).copy()
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_size, x='savedmodelsize_kb', y='AUROC', hue='model', style='experiment', s=150)
    plt.title('Saved Model Size vs Open-Set AUROC')
    if len(df_size) > 0 and df_size['savedmodelsize_kb'].max() > df_size['savedmodelsize_kb'].min() * 10:
        plt.xscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('results/figures/extended/rtiot2022_model_size_vs_auroc_extended.png')
    plt.close()
    
    # 5. Inference Time vs AUROC
    df_infer = df.dropna(subset=['AUROC', 'inferencetime1000samplesseconds']).copy()
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_infer, x='inferencetime1000samplesseconds', y='AUROC', hue='model', style='experiment', s=150)
    plt.title('Inference Time (1k samples) vs Open-Set AUROC')
    if len(df_infer) > 0 and df_infer['inferencetime1000samplesseconds'].max() > df_infer['inferencetime1000samplesseconds'].min() * 10:
        plt.xscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('results/figures/extended/rtiot2022_infer_time_vs_auroc_extended.png')
    plt.close()
    
    print("Extended plots generated successfully in results/figures/extended/")

if __name__ == '__main__':
    main()
