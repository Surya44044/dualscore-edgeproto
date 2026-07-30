# Complete Guide to GitHub, DockerHub & Azure Container Apps CI/CD Pipeline

This document explains the end-to-end automation roadmap for DualScore-EdgeProto deployment:

$$\text{Local Git Repo} \longrightarrow \text{GitHub Actions CI/CD} \longrightarrow \text{DockerHub} \longrightarrow \text{Azure Container Apps}$$

---

## 1. Local Development & Git Workflow

1. **Stage Code Changes**:
   ```bash
   git add .
   ```
   *Note*: `.gitignore` automatically prevents raw datasets (`data/`), virtual environments (`.venv/`), and secrets (`.env`, `azure-credentials*.json`) from being staged.

2. **Commit Code**:
   ```bash
   git commit -m "feat: Add FastAPI inference service and Azure CI/CD workflow"
   ```

3. **Push to Remote Main Branch**:
   ```bash
   git push origin main
   ```

---

## 2. GitHub Actions CI/CD Workflows

### A. `.github/workflows/ci.yml` (Continuous Integration)
- **Triggers**: On pull requests or commits to any branch.
- **Tasks**:
  - Sets up Python 3.11.
  - Compiles all Python source files (`python -m compileall -q src`).
  - Verifies FastAPI application imports (`from src.api.app import app`).
  - Runs automated secret scanner (`scripts/check_no_secrets.sh`).
  - Runs fast synthetic API smoke tests (`scripts/run_synthetic_smoke.sh`).
  - **No full dataset training** is conducted during CI to ensure fast build times.

### B. `.github/workflows/dockerhub-azure-deploy.yml` (Continuous Deployment)
- **Triggers**: On push to `main` branch or manual `workflow_dispatch`.
- **Tasks**:
  1. Runs CI verification steps.
  2. Builds CPU-optimized Docker image using production `Dockerfile`.
  3. Tags Docker image:
     - `docker.io/${DOCKERHUB_USERNAME}/dualscore-edgeproto:latest`
     - `docker.io/${DOCKERHUB_USERNAME}/dualscore-edgeproto:${GITHUB_SHA}`
  4. Authenticates with DockerHub using `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`.
  5. Pushes Docker images to DockerHub.
  6. Authenticates with Azure using `AZURE_CREDENTIALS` (Service Principal).
  7. Deploys / updates container image on **Azure Container Apps** (`AZURE_CONTAINER_APP_NAME`).

---

## 3. Docker Container Specifications

- **Base Image**: `python:3.11-slim` (lightweight Debian-based image).
- **CPU Only**: PyTorch CPU-only wheels installed without CUDA dependencies.
- **Port Exposure**: Exposes port `8000` (or dynamic `PORT` env var injected by Azure).
- **Entrypoint**: Uvicorn server running FastAPI `src.api.app:app`.
- **Zero Datasets**: Datasets are completely excluded via `.dockerignore`.

---

## 4. Azure Container Apps Architecture

- **Role**: Serves the public HTTP inference REST API (`/health`, `/metadata`, `/predict`, `/predict_batch`).
- **Resource Limits**: 0.25 - 0.5 vCPU, 0.5 - 1.0 GB RAM (fits well within Azure Free / Student tier limits).
- **Scaling**: 0 to 5 replicas (auto-scales to 0 when idle to save costs).
- **Demo Mode Safety**: If model weights are not loaded, FastAPI starts in `demo_mode=true` and clearly reports fallback behavior without faking predictions.
