# Reproducibility Guide: DualScore-EdgeProto Research

This guide provides full instructions for reproducing the experimental results reported in the DualScore-EdgeProto / EdgeProto-OSR study.

---

## 🔬 Experimental Setup

### Environment Specifications
- **Operating System**: Ubuntu Linux (20.04/22.04 LTS)
- **CPU**: Intel i3 dual-core (CPU-only execution)
- **RAM**: 8 GB
- **Python Version**: 3.11+
- **PyTorch Version**: 2.3.1 (CPU build)

### Dataset
- **Dataset**: RT-IoT2022 dataset grouped into 48 numeric network flow features.
- **Leakage Prevention**: Selected feature set (`data/processed/rtiot2022_grouped_48_selected_features.txt`) excludes high-leakage fields (`Attack_type`, `label`, `Unnamed: 0`, `id.orig_p`, `id.resp_p`, `proto`, `service`).

---

## 🏃 Reproducing Experiments

### 1. Run Preflight Audit
Verify dataset integrity and feature mapping:
```bash
python run_day6_preflight.py
```

### 2. Evaluate Baseline Models
```bash
bash run_all_baselines.sh
```

### 3. Evaluate EdgeProto-Base & Pseudo-Open (PO)
```bash
bash run_all.sh
bash run_all_po.sh
```

### 4. Evaluate DualScore-Hybrid Models
```bash
python src/evaluation/evaluate_edgeproto_hybrid.py --prefix rt_h1_arp_spoofing --config configs/rt_h1_arp_spoofing.yaml --edgeproto_model models/rt_h1_arp_spoofing/best_edgeproto.pt --unknown ARP_Spoofing
python src/evaluation/evaluate_edgeproto_hybrid.py --prefix rt_h2_recon --config configs/rt_h2_recon.yaml --edgeproto_model models/rt_h2_recon/best_edgeproto.pt --unknown Recon
python src/evaluation/evaluate_edgeproto_hybrid.py --prefix rt_h3_dos --config configs/rt_h3_dos.yaml --edgeproto_model models/rt_h3_dos/best_edgeproto.pt --unknown DoS
```

### 5. Audit Open-Set Metrics (AUROC, AUPR, FPR95, OSCR)
```bash
python src/evaluation/audit_hybrid_metrics.py
```

### 6. Run 3-Seed Robustness Study
```bash
python run_day6_repeated_seeds.py
```

### 7. Run Latency & Quantization Benchmarks
```bash
python src/deployment/benchmark_final_models.py
python src/deployment/resource_quant_summary.py
```

---

## 📁 Key Research Output Manifest

- **Audited Metric Comparison**: `results/tables/rtiot2022_hybrid_comparison_METRIC_AUDITED.csv`
- **Full 3-Seed Aggregations**: `results/tables/rtiot2022_repeated_seed_summary_FULL.csv`
- **Latency Benchmark Table**: `results/tables/rtiot2022_latency_benchmark.csv`
- **Resource & Quantization Summary**: `results/tables/rtiot2022_resource_quant_summary.csv`
- **Publication Figures**: `results/figures/rtiot2022_final_*.png`
