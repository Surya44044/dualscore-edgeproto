import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    os.makedirs("results/figures", exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'font.size': 12, 'figure.autolayout': True})
    
    # 1. Repeated Seeds AUROC (Mean ± Std)
    full_seeds_csv = "results/tables/rtiot2022_repeated_seed_summary_FULL.csv"
    if os.path.exists(full_seeds_csv):
        df_seeds = pd.read_csv(full_seeds_csv)
        df_seeds_auroc = df_seeds[df_seeds['metric'] == 'AUROC']
        
        fig, ax = plt.subplots(figsize=(9, 5))
        exps = sorted(df_seeds_auroc['experiment'].unique())
        x = np.arange(len(exps))
        width = 0.35
        
        ep_means = [df_seeds_auroc[(df_seeds_auroc['experiment']==e) & (df_seeds_auroc['model']=='EdgeProto-Base')]['mean'].values[0] for e in exps]
        ep_stds = [df_seeds_auroc[(df_seeds_auroc['experiment']==e) & (df_seeds_auroc['model']=='EdgeProto-Base')]['std'].values[0] for e in exps]
        
        hy_means = [df_seeds_auroc[(df_seeds_auroc['experiment']==e) & (df_seeds_auroc['model']=='Hybrid-Mean-50-50')]['mean'].values[0] for e in exps]
        hy_stds = [df_seeds_auroc[(df_seeds_auroc['experiment']==e) & (df_seeds_auroc['model']=='Hybrid-Mean-50-50')]['std'].values[0] for e in exps]
        
        rects1 = ax.bar(x - width/2, ep_means, width, yerr=ep_stds, capsize=5, label='EdgeProto-Base', color='#4C72B0')
        rects2 = ax.bar(x + width/2, hy_means, width, yerr=hy_stds, capsize=5, label='Hybrid-Mean-50-50', color='#55A868')
        
        ax.set_ylabel('AUROC (Mean ± Std, 3 Seeds)')
        ax.set_title('RT-IoT2022 3-Seed Robustness: EdgeProto-Base vs Hybrid-Mean-50-50', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(exps)
        ax.set_ylim(0.80, 1.02)
        ax.legend()
        plt.tight_layout()
        fig.savefig("results/figures/rtiot2022_final_auroc_mean_seed.png", dpi=300)
        plt.close(fig)
        print("Saved results/figures/rtiot2022_final_auroc_mean_seed.png")

    # Load Audited Comparison
    comp_csv = "results/tables/rtiot2022_hybrid_comparison_METRIC_AUDITED.csv"
    if os.path.exists(comp_csv):
        df_comp = pd.read_csv(comp_csv)
        
        # 2. Single Run AUROC
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=df_comp, x='experiment', y='AUROC', hue='model', ax=ax)
        ax.set_title("RT-IoT2022 Open-Set AUROC Across All Evaluated Models", fontweight='bold')
        ax.set_ylim(0.75, 1.02)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        fig.savefig("results/figures/rtiot2022_final_auroc_single_run.png", dpi=300)
        plt.close(fig)
        print("Saved results/figures/rtiot2022_final_auroc_single_run.png")
        
        # 3. Single Run FPR95
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=df_comp, x='experiment', y='FPRat95_TPR', hue='model', ax=ax)
        ax.set_title("RT-IoT2022 False Positive Rate @ 95% TPR (Lower is Better)", fontweight='bold')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        fig.savefig("results/figures/rtiot2022_final_fpr95_single_run.png", dpi=300)
        plt.close(fig)
        print("Saved results/figures/rtiot2022_final_fpr95_single_run.png")
        
        # 4. Final Resource Tradeoff
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df_comp.dropna(subset=['savedmodelsize_kb', 'AUROC']), 
                        x='savedmodelsize_kb', y='AUROC', hue='model', style='experiment', s=140, ax=ax)
        ax.set_xscale('log')
        ax.set_title("Final Resource Tradeoff: Checkpoint Size (KB) vs AUROC", fontweight='bold')
        ax.set_xlabel("Saved Checkpoint Size (KB, Log Scale)")
        ax.set_ylim(0.75, 1.02)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        fig.savefig("results/figures/rtiot2022_final_resource_tradeoff.png", dpi=300)
        plt.close(fig)
        print("Saved results/figures/rtiot2022_final_resource_tradeoff.png")

    # 5. Latency Comparison Figure
    lat_csv = "results/tables/rtiot2022_latency_benchmark.csv"
    if os.path.exists(lat_csv):
        df_lat = pd.read_csv(lat_csv)
        df_bs1 = df_lat[df_lat['batch_size'] == 1]
        
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=df_bs1, x='experiment', y='ms_per_sample', hue='model', ax=ax)
        ax.set_title("Stream Inference Latency (Batch Size = 1, CPU Edge-Gateway)", fontweight='bold')
        ax.set_ylabel("Latency (ms / sample)")
        plt.legend()
        plt.tight_layout()
        fig.savefig("results/figures/rtiot2022_final_latency.png", dpi=300)
        plt.close(fig)
        print("Saved results/figures/rtiot2022_final_latency.png")

if __name__ == "__main__":
    main()
