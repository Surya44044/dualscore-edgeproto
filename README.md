# DualScore-EdgeProto: Lightweight Open-Set Zero-Day IoT Intrusion Detection

[![CI/CD](https://github.com/USER/dualscore-edgeproto/actions/workflows/ci.yml/badge.svg)](https://github.com/USER/dualscore-edgeproto/actions/workflows/ci.yml)
[![Docker Image](https://img.shields.io/badge/docker-dualscore--edgeproto-blue)](https://hub.docker.com/)
[![Azure Deployment](https://img.shields.io/badge/azure-container--apps-0089D6)](https://azure.microsoft.com/en-us/products/container-apps/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

DualScore-EdgeProto is an ultra-lightweight Network Intrusion Detection System (NIDS) designed for open-set zero-day IoT intrusion detection. By combining normalized prototypical distance ($z_{\text{proto}}$) with autoencoder reconstruction error ($z_{\text{ae}}$), DualScore-EdgeProto achieves high open-set detection accuracy on zero-day attacks while maintaining an ultra-compact memory footprint suitable for edge gateways and microcontrollers.

---

## 📌 Key Highlights

- **Research Contribution**: DualScore-EdgeProto fuses prototype-based open-set recognition with autoencoder reconstruction error to solve out-of-distribution feature scale shifts (e.g., DoS attacks) without degrading known-class discrimination.
- **Ultra-Compact Footprint**: 
  - **EdgeProto-Base**: `3,836` parameters (~15 KB raw FP32 weights).
  - **DualScore-Hybrid**: `15,356` parameters (~60 KB raw FP32 weights, ~72.5 KB saved checkpoint).
- **Fast CPU Stream Inference**: ~0.68 ms per sample stream latency (~1,460 samples/sec) on dual-core CPU edge-gateways; ~5.4 µs per sample for batch size 256 (~186,000 samples/sec).
- **Production-Ready Deployment Roadmap**:
  $$\text{Local Git Repo} \longrightarrow \text{GitHub Actions CI/CD} \longrightarrow \text{DockerHub} \longrightarrow \text{Azure Container Apps}$$

---

## 📊 RT-IoT2022 Benchmark Results

| Holdout Experiment | Unknown Zero-Day Class | EdgeProto-Base AUROC | DualScore-Hybrid AUROC | Hybrid FPR @ 95% TPR | Audited OSCR |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **RT-H1** | ARP Spoofing | `0.9897` | **`0.9912 ± 0.0005`** | **`0.0179`** | **`0.9917`** |
| **RT-H2** | Reconnaissance | `0.9865` | **`0.9882 ± 0.0030`** | **`0.0826`** | **`0.9857`** |
| **RT-H3** | DoS Attack | `0.8991` | **`0.9988 ± 0.0013`** | **`0.0076`** | **`0.9362`** |

---

## 🚀 Deployment Architecture

```
                       +---------------------------+
                       |    GitHub Repository      |
                       +-------------+-------------+
                                     |
                                     v (push / PR)
                       +---------------------------+
                       |   GitHub Actions CI/CD    |
                       |  (Compile, Test, Docker)  |
                       +-------------+-------------+
                                     |
                                     v (push main)
                       +---------------------------+
                       |     DockerHub Registry    |
                       +-------------+-------------+
                                     |
                                     v (auto-deploy)
                       +---------------------------+
                       |   Azure Container Apps    |
                       |    (FastAPI Inference)    |
                       +---------------------------+
```

> ⚠️ **Note**: Azure Container Apps handles **inference API deployment only**. No raw or processed datasets are committed to Git, baked into Docker images, or deployed to production containers.

---

## 🛠️ Local Development & Fast API Usage

### 1. Installation & Environment Setup
```bash
git clone https://github.com/YOUR_USERNAME/edgeproto_osr_current.git
cd edgeproto_osr_current

# Activate virtual environment
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Local FastAPI Inference Service
```bash
bash scripts/run_api_local.sh
```
Or run directly via Makefile:
```bash
make api
```
Access the interactive OpenAPI Swagger UI at `http://127.0.0.1:8000/docs`.

### 3. Test Endpoints
```bash
bash scripts/test_api_local.sh
```
Example Single Sample Prediction:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d @examples/predict_request.json
```

---

## 🐳 Docker Deployment

### Build & Run Container Locally
```bash
docker build -t dualscore-edgeproto:local .
docker run --rm -p 8000:8000 dualscore-edgeproto:local
```
Or use the script/Makefile helper:
```bash
make docker-build
make docker-run
```

---

## ☁️ Azure Container Apps CI/CD

Deployment to Azure Container Apps is automated via GitHub Actions (`.github/workflows/dockerhub-azure-deploy.yml`).

### Required GitHub Secrets
- `DOCKERHUB_USERNAME`: Your DockerHub username.
- `DOCKERHUB_TOKEN`: Personal Access Token for DockerHub.
- `AZURE_CREDENTIALS`: Service Principal JSON output from Azure CLI.
- `AZURE_RESOURCE_GROUP`: Azure Resource Group name (e.g., `rg-dualscore-edgeproto`).
- `AZURE_CONTAINER_APP_NAME`: Container App name (e.g., `app-dualscore-edgeproto`).

For step-by-step Azure configuration, see [DEPLOY_AZURE.md](DEPLOY_AZURE.md) and [docs/GITHUB_DOCKER_AZURE_CICD.md](docs/GITHUB_DOCKER_AZURE_CICD.md).

---

## 📖 Documentation Sitemap

- [REPRODUCIBILITY.md](REPRODUCIBILITY.md) — Comprehensive guide to reproducing research experiments.
- [MODEL_CARD.md](MODEL_CARD.md) — Model card details, architecture specifications, and intended use.
- [DEPLOY_AZURE.md](DEPLOY_AZURE.md) — Complete guide to setting up Azure Container Apps.
- [docs/AZURE_STUDENT_FREE_SERVICES_GUIDE.md](docs/AZURE_STUDENT_FREE_SERVICES_GUIDE.md) — Guidance on Azure free/student tier service allocation.
- [docs/GITHUB_DOCKER_AZURE_CICD.md](docs/GITHUB_DOCKER_AZURE_CICD.md) — Complete CI/CD workflow explanation.
- [SECURITY.md](SECURITY.md) — Security policy and secret prevention guidelines.
- [CONTRIBUTING.md](CONTRIBUTING.md) — Contribution guidelines and coding conventions.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
