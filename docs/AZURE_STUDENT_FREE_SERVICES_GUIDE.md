# Azure Student & Free Tier Services Allocation Guide

This document provides strategic advice for leveraging Azure Free / Azure for Students credits efficiently without incurring unexpected charges.

---

## 🟢 RECOMMENDED & USEFUL SERVICES FOR THIS PROJECT

| Azure Service | Primary Use Case | Free Tier / Student Allowance | Best Practices |
| :--- | :--- | :--- | :--- |
| **Azure Container Apps** | **Inference API Deployment** | 180,000 vCPU-seconds / month + 2 Million requests | Set min replicas = 0 to scale to zero when idle. Ideal for FastAPI inference! |
| **DockerHub (External)** | **Primary Docker Image Registry** | Free public/private repository | Used as the primary registry to avoid paying for Azure Container Registry (ACR) storage. |
| **Linux Virtual Machine** | **CICIoT2023 Scaled Training** | 750 hours/month of B1s VM (Free Tier) | Use for heavy model training. Always stop/deallocate VM when not actively running experiments. |
| **Managed Disks** | **Temporary VM Storage** | 2x 64GB Premium SSDs | Store datasets during training on VM; delete unneeded snapshots. |
| **Azure Blob Storage** | **Artifacts & Checkpoint Storage** | 5 GB LRS Storage | Keep only small model checkpoints (`.pt`) and evaluation tables (`.csv`). Do not upload full datasets (>10GB). |
| **Azure Machine Learning** | **Structured Scaling Jobs** | Pay-as-you-go compute | Use compute clusters with auto-scale to 0 nodes when idle. |
| **Azure Cloud Shell** | **Browser CLI Management** | Free with storage mount | Convenient for executing `az` CLI commands directly in browser. |
| **Azure Monitor** | **Basic API Telemetry** | 5 GB ingestion / month | Track HTTP 200/500 response rates and latency metrics. |

---

## 🔴 SERVICES NOT NEEDED INITIALLY (AVOID TO PREVENT CHARGES)

To maximize your free credits, **do NOT provision** the following managed enterprise services:

- ❌ **Managed SQL Databases** (Azure SQL, MySQL, PostgreSQL) — Network flows are tabular files (`.csv`), not relational DB tables.
- ❌ **Cosmos DB** — No NoSQL document database required.
- ❌ **Azure Kubernetes Service (AKS)** — Way too heavy for lightweight microservices; Container Apps provides serverless container execution at 1/10th the cost.
- ❌ **Azure IoT Hub** — Simulated via REST API calls directly to Container Apps.
- ❌ **Service Fabric** — Enterprise microservice orchestrator; unnecessary.
- ❌ **API Management (APIM)** — FastAPI includes built-in OpenAPI documentation (`/docs`).
- ❌ **Cognitive / AI Services** (Vision, Speech, Translator) — Custom PyTorch model is used.
- ❌ **VPN Gateway & ExpressRoute** — Public ingress endpoints suffice.

---

## 💡 COST MANAGEMENT CHECKLIST

1. **Single Resource Group**: Keep all project assets inside `rg-dualscore-edgeproto`.
2. **Budget Alerts**: Set a $5.00 budget alert in the Azure Portal Cost Management tab.
3. **Delete Resource Group**: When your project demonstration is complete, run:
   ```bash
   az group delete --name rg-dualscore-edgeproto --yes --no-wait
   ```
