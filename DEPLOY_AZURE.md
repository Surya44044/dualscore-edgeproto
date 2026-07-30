# Step-by-Step Azure Container Apps Deployment Guide

This guide provides instructions for setting up Azure Container Apps for serving the DualScore-EdgeProto inference service.

---

## ⚠️ Important Budget & Cost Warnings

- **Budget Alerts**: Create a spending budget alert in the Azure Portal before provisioning resources.
- **Resource Cleanup**: Keep all project resources inside a single dedicated resource group (`rg-dualscore-edgeproto`). To delete everything after demonstration, run `az group delete -n rg-dualscore-edgeproto`.
- **Inference Only**: Azure Container Apps is designed exclusively for fast, lightweight HTTP REST API inference. **Do not upload datasets or deploy training jobs to Container Apps.**

---

## 🛠️ Step-by-Step Azure Setup

### Step 1: Install Azure CLI & Login
```bash
az login
```

### Step 2: Create Resource Group
```bash
az group create \
  --name rg-dualscore-edgeproto \
  --location eastus
```

### Step 3: Create Container Apps Environment
```bash
az containerapp env create \
  --name env-dualscore-edgeproto \
  --resource-group rg-dualscore-edgeproto \
  --location eastus
```

### Step 4: Create Container App from DockerHub Image
```bash
az containerapp create \
  --name app-dualscore-edgeproto \
  --resource-group rg-dualscore-edgeproto \
  --environment env-dualscore-edgeproto \
  --image docker.io/YOUR_DOCKERHUB_USERNAME/dualscore-edgeproto:latest \
  --target-port 8000 \
  --ingress external \
  --cpu 0.25 \
  --memory 0.5Gi
```

---

## 🔑 GitHub Actions Service Principal Configuration

To enable automated deployment from GitHub Actions:

### 1. Create Azure Service Principal
```bash
az ad sp create-for-rbac \
  --name "sp-dualscore-edgeproto" \
  --role contributor \
  --scopes /subscriptions/YOUR_SUBSCRIPTION_ID/resourceGroups/rg-dualscore-edgeproto \
  --sdk-auth
```

### 2. Configure GitHub Secrets
In your GitHub repository under **Settings > Secrets and variables > Actions**, add the following repository secrets:

- `DOCKERHUB_USERNAME`: Your DockerHub username.
- `DOCKERHUB_TOKEN`: DockerHub Personal Access Token (PAT).
- `AZURE_CREDENTIALS`: Paste the complete JSON output from `az ad sp create-for-rbac`.
- `AZURE_RESOURCE_GROUP`: `rg-dualscore-edgeproto`
- `AZURE_CONTAINER_APP_NAME`: `app-dualscore-edgeproto`

---

## 🧪 Testing the Live Azure Endpoint

Once GitHub Actions finishes deploying:

1. Retrieve the FQDN:
   ```bash
   az containerapp show \
     --name app-dualscore-edgeproto \
     --resource-group rg-dualscore-edgeproto \
     --query properties.configuration.ingress.fqdn -o tsv
   ```

2. Test Health Endpoint:
   ```bash
   curl https://<APP_URL>/health
   ```

3. Test Metadata Endpoint:
   ```bash
   curl https://<APP_URL>/metadata
   ```

4. Test Inference Endpoint:
   ```bash
   curl -X POST https://<APP_URL>/predict \
     -H "Content-Type: application/json" \
     -d @examples/predict_request.json
   ```
