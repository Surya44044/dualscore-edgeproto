# Azure Deployment Success Summary

## Project
DualScore-EdgeProto / EdgeProto-OSR

## Deployment Pipeline
Local repo → GitHub → GitHub Actions CI → DockerHub image build/push → Azure Container Apps deployment

## GitHub Repository
https://github.com/Surya44044/dualscore-edgeproto

## Azure Service
Azure Container Apps

## Container App
app-dualscore-edgeproto

## Resource Group
rg-dualscore-edgeproto

## Public API URL
https://app-dualscore-edgeproto.delightfuldesert-a9d820d6.centralindia.azurecontainerapps.io

## Verified Endpoints

### /health
Returned:
{"status":"ok","service":"dualscore-edgeproto-api","demo_mode":true,"version":"1.0.0"}

### /metadata
Returned model metadata with:
- model_name: DualScore-EdgeProto
- parameters: 15356
- expected_features: 48
- labels: Benign, DDoS, DoS, Recon
- demo_mode: true

## Important Note
The Azure deployment currently runs in explicit demo_mode=true because trained model artifacts are not packaged inside the Docker image. This deployment proves the reproducible containerized inference pipeline. Offline audited experiments remain the source of research performance claims.
