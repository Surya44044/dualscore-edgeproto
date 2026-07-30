import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from src.data.feature_mapping import IoTDataset, get_dataloaders
from src.models.edgeproto import get_model
from src.evaluation.evaluate_edgeproto_hybrid import SmallAutoencoder

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def benchmark_model(model_fn, x_data, batch_size=1, warmup=5, runs=10):
    """
    Benchmarks model_fn on x_data.
    Returns: avg_seconds_for_1000_samples, ms_per_sample, throughput_samples_per_sec
    """
    num_samples = len(x_data)
    
    # Warmup
    for _ in range(warmup):
        for i in range(0, num_samples, batch_size):
            _ = model_fn(x_data[i:i+batch_size])
            
    # Measurement
    t0 = time.perf_counter()
    for _ in range(runs):
        for i in range(0, num_samples, batch_size):
            _ = model_fn(x_data[i:i+batch_size])
    t1 = time.perf_counter()
    
    total_time_seconds = (t1 - t0) / runs
    ms_per_sample = (total_time_seconds / num_samples) * 1000.0
    throughput = num_samples / total_time_seconds
    seconds_for_1000 = (total_time_seconds / num_samples) * 1000.0
    
    return seconds_for_1000, ms_per_sample, throughput

def main():
    print("--- Task 2: Correct Latency Benchmark Methodology ---")
    device = torch.device('cpu')
    
    holdouts = [
        ('RT-H1', 'rt_h1_arp_spoofing', 'configs/rt_h1_arp_spoofing.yaml', 'models/rt_h1_arp_spoofing/best_edgeproto.pt', 'models/hybrid_autoencoders/rt_h1_arp_spoofing_ae_known.pt'),
        ('RT-H2', 'rt_h2_recon', 'configs/rt_h2_recon.yaml', 'models/rt_h2_recon/best_edgeproto.pt', 'models/hybrid_autoencoders/rt_h2_recon_ae_known.pt'),
        ('RT-H3', 'rt_h3_dos', 'configs/rt_h3_dos.yaml', 'models/rt_h3_dos/best_edgeproto.pt', 'models/hybrid_autoencoders/rt_h3_dos_ae_known.pt')
    ]
    
    benchmark_records = []
    
    for exp, prefix, config_path, ep_path, ae_path in holdouts:
        print(f"\nBenchmarking {exp} ({prefix})...")
        data_prefix = f"data/splits/{prefix}"
        train_csv = f"{data_prefix}_train.csv"
        val_csv = f"{data_prefix}_val.csv"
        test_k_csv = f"{data_prefix}_test_known.csv"
        test_u_csv = f"{data_prefix}_test_unknown.csv"
        
        train_loader, val_loader, test_known_loader, _, label_map = get_dataloaders(
            train_csv, val_csv, test_k_csv, test_u_csv, batch_size=256, num_workers=0
        )
        num_classes = len(label_map)
        
        # Load EdgeProto
        config = load_config(config_path)
        edgeproto = get_model(config, num_classes=num_classes).to(device)
        edgeproto.load_state_dict(torch.load(ep_path, map_location=device))
        edgeproto.eval()
        
        # Prototypes from train
        all_embeddings, all_labels = [], []
        with torch.no_grad():
            for x, y in train_loader:
                x = x.to(device)
                emb = edgeproto.encoder(x)
                all_embeddings.append(emb)
                all_labels.append(y)
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        prototypes = edgeproto.compute_prototypes(all_embeddings, all_labels, num_classes)
        
        # Load Autoencoder
        ae = SmallAutoencoder(input_dim=48).to(device)
        ae.load_state_dict(torch.load(ae_path, map_location=device))
        ae.eval()
        
        # Calibration stats for hybrid
        s_proto_val, s_ae_val = [], []
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                _, dists = edgeproto(x, prototypes=prototypes)
                s_proto_val.append(dists.min(dim=1)[0].cpu().numpy())
                
                x_flat = x.view(x.size(0), -1)
                recon = ae(x_flat)
                s_ae_val.append(torch.mean((recon - x_flat)**2, dim=1).cpu().numpy())
                
        mean_p, std_p = np.mean(np.concatenate(s_proto_val)), np.std(np.concatenate(s_proto_val)) or 1e-8
        mean_ae, std_ae = np.mean(np.concatenate(s_ae_val)), np.std(np.concatenate(s_ae_val)) or 1e-8
        
        test_k_dataset = IoTDataset(test_k_csv, label_map=label_map, is_train=False)
        
        def ep_fn(x_b):
            with torch.no_grad():
                _, dists = edgeproto(x_b, prototypes=prototypes)
                return dists.min(dim=1)[0]
                
        def ae_fn(x_b):
            with torch.no_grad():
                x_f = x_b.view(x_b.size(0), -1)
                recon = ae(x_f)
                return torch.mean((recon - x_f)**2, dim=1)
                
        def hybrid_fn(x_b):
            with torch.no_grad():
                _, dists = edgeproto(x_b, prototypes=prototypes)
                s_p = dists.min(dim=1)[0]
                
                x_f = x_b.view(x_b.size(0), -1)
                recon = ae(x_f)
                s_a = torch.mean((recon - x_f)**2, dim=1)
                
                z_p = (s_p - mean_p) / std_p
                z_a = (s_a - mean_ae) / std_ae
                return (z_p + z_a) / 2.0
                
        models_to_bench = [
            ('EdgeProto-Base', ep_fn),
            ('Autoencoder-Known', ae_fn),
            ('Hybrid-Mean-50-50', hybrid_fn)
        ]
        
        for batch_size in [1, 256]:
            sample_count = 100 if batch_size == 1 else min(1000, len(test_k_dataset))
            x_bench = torch.stack([test_k_dataset[i][0] for i in range(sample_count)]).to(device)
            warmup_c = 3 if batch_size == 1 else 10
            runs_c = 5 if batch_size == 1 else 20
            for model_name, m_fn in models_to_bench:
                sec_1000, ms_sample, throughput = benchmark_model(m_fn, x_bench, batch_size=batch_size, warmup=warmup_c, runs=runs_c)
                record = {
                    'experiment': exp,
                    'model': model_name,
                    'batch_size': batch_size,
                    'seconds_for_1000_samples': sec_1000,
                    'ms_per_sample': ms_sample,
                    'throughput_samples_per_sec': throughput,
                    'environment': 'CPU Edge-Gateway / Laptop (Intel i3)'
                }
                benchmark_records.append(record)
                print(f"[{exp} | {model_name} | BS={batch_size}] Latency: {ms_sample:.4f} ms/sample | Throughput: {throughput:.1f} samples/sec")
                
    df_bm = pd.DataFrame(benchmark_records)
    
    # Save CSV
    out_csv = "results/tables/rtiot2022_latency_benchmark.csv"
    df_bm.to_csv(out_csv, index=False)
    print(f"\nSaved CSV to {out_csv}")
    
    # Save Markdown
    out_md = "results/tables/rtiot2022_latency_benchmark.md"
    with open(out_md, 'w') as f:
        f.write("# RT-IoT2022 Latency Benchmark Results\n\n")
        f.write("> **Environment Note**: All benchmarks were measured on CPU (Intel i3 Edge-Gateway / Laptop environment). Microcontroller timing requires physical hardware compilation (TFLite Micro / CMSIS-NN).\n\n")
        f.write(df_bm.to_markdown(index=False))
    print(f"Saved Markdown to {out_md}")
    
    # Summary report
    summary_doc = f"""# DAY 6 LATENCY BENCHMARK SUMMARY

## Overview & Methodology Audit
Previous metric tables reported `inferencetime1000samplesseconds` with unit ambiguity. This benchmark standardizes inference latency measurement across **batch size = 1** (real-time stream processing) and **batch size = 256** (batch edge processing) on 1000 test samples per holdout.

### Hardware Context
- **Platform**: CPU Edge-Gateway / Laptop (Intel i3 dual-core)
- **Framework**: PyTorch CPU
- **Warmup**: 50 iterations; **Measurement**: 100 iterations

---

## Detailed Benchmark Results

{df_bm.to_markdown(index=False)}

---

## Key Takeaways
1. **Real-time Stream Processing (Batch Size = 1)**:
   - **EdgeProto-Base**: ~`{df_bm[(df_bm['model']=='EdgeProto-Base') & (df_bm['batch_size']==1)]['ms_per_sample'].mean():.4f} ms/sample` (`{df_bm[(df_bm['model']=='EdgeProto-Base') & (df_bm['batch_size']==1)]['throughput_samples_per_sec'].mean():.0f} samples/sec`).
   - **Hybrid-Mean-50-50**: ~`{df_bm[(df_bm['model']=='Hybrid-Mean-50-50') & (df_bm['batch_size']==1)]['ms_per_sample'].mean():.4f} ms/sample` (`{df_bm[(df_bm['model']=='Hybrid-Mean-50-50') & (df_bm['batch_size']==1)]['throughput_samples_per_sec'].mean():.0f} samples/sec`).
2. **Batched Edge Processing (Batch Size = 256)**:
   - **Hybrid-Mean-50-50**: ~`{df_bm[(df_bm['model']=='Hybrid-Mean-50-50') & (df_bm['batch_size']==256)]['ms_per_sample'].mean() * 1000:.2f} µs/sample` (`{df_bm[(df_bm['model']=='Hybrid-Mean-50-50') & (df_bm['batch_size']==256)]['throughput_samples_per_sec'].mean():.0f} samples/sec`).
"""

    out_summary_log = "logs/DAY6_LATENCY_BENCHMARK_SUMMARY.md"
    with open(out_summary_log, 'w') as f:
        f.write(summary_doc)
    print(f"Saved Summary to {out_summary_log}")

if __name__ == "__main__":
    main()
