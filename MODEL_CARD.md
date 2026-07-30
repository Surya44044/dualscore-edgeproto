# Model Card: DualScore-EdgeProto

## Model Details
- **Model Name**: DualScore-EdgeProto (Hybrid-Mean-50-50)
- **Model Version**: 1.0.0
- **Model Architecture**: Depthwise-Separable Convolutional Encoder + Metric Prototype Classifier combined with a 6-layer Fully-Connected Autoencoder.
- **Model Parameters**: 15,356 trainable parameters (EdgeProto: 3,836 + Autoencoder: 11,520).
- **Framework**: PyTorch (CPU-only build).
- **License**: MIT License.

---

## Intended Use
- **Primary Intended Use**: Real-time open-set zero-day network intrusion detection on edge gateways and IoT router devices.
- **Out-of-Scope Use Cases**: High-throughput core cloud routers processing multi-gigabit traffic without pre-filtering, or non-tabular un-normalized raw packet payload analysis.

---

## Architecture Specifications

### 1. EdgeProto Encoder
- Input: `[B, 4, 12]` reshaped from 48 numeric network flow features.
- Layer 1: Depthwise-Separable 1D Conv (in=4, out=16, k=3, p=1) + BatchNorm + ReLU + MaxPool1d(2)
- Layer 2: Depthwise-Separable 1D Conv (in=16, out=32, k=3, p=1) + BatchNorm + ReLU + MaxPool1d(2)
- Flatten + Linear(96, 32) + L2 Normalization.

### 2. Autoencoder Reconstruction Module
- Topology: `48 -> 64 -> 32 -> 16 -> 32 -> 64 -> 48`
- Activation: ReLU in intermediate layers, Linear output.

### 3. DualScore Fusion Rule (`Hybrid-Mean-50-50`)
- Prototype Distance Score: $s_{\text{proto}} = \min_{c} \| z - p_c \|^2$
- AE Reconstruction Score: $s_{\text{ae}} = \frac{1}{D} \sum_{d=1}^D (x_d - \hat{x}_d)^2$
- Normalization: $z_{\text{proto}} = \frac{s_{\text{proto}} - \mu_{\text{val,proto}}}{\sigma_{\text{val,proto}}}$, $z_{\text{ae}} = \frac{s_{\text{ae}} - \mu_{\text{val,ae}}}{\sigma_{\text{val,ae}}}$
- Combined Zero-Day Score: $s_{\text{hybrid}} = \frac{z_{\text{proto}} + z_{\text{ae}}}{2}$

---

## Performance Summary (RT-IoT2022 Dataset)

- **RT-H1 (ARP Spoofing)**: AUROC = `0.9925 ± 0.0005`, FPR@95%TPR = `0.0187`, OSCR = `0.9917`
- **RT-H2 (Reconnaissance)**: AUROC = `0.9850 ± 0.0030`, FPR@95%TPR = `0.1185`, OSCR = `0.9857`
- **RT-H3 (DoS Attack)**: AUROC = `0.9972 ± 0.0013`, FPR@95%TPR = `0.0078`, OSCR = `0.9362`
- **Inference Latency**: `0.68 - 0.81 ms / sample` stream (batch size = 1) on CPU.
