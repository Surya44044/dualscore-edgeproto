# Azure Virtual Machine Runbook for Large CICIoT2023 Training

This runbook describes how to provision and use an Azure Linux VM (B-series / D-series) for large-scale training experiments.

---

## 1. Provision VM via Azure CLI
```bash
az vm create \
  --resource-group rg-dualscore-edgeproto \
  --name vm-dualscore-trainer \
  --image Ubuntu2204 \
  --size Standard_B2s \
  --admin-username azureuser \
  --generate-ssh-keys
```

## 2. Connect & Setup Environment
```bash
ssh azureuser@<VM_PUBLIC_IP>

# Clone local repo or pull code
git clone https://github.com/YOUR_USERNAME/edgeproto_osr_current.git
cd edgeproto_osr_current

# Setup Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Resource & Cleanup Guidelines
- Stop the VM when training is idle: `az vm stop --resource-group rg-dualscore-edgeproto --name vm-dualscore-trainer`
- Delete VM and associated disks when finished: `az vm delete --resource-group rg-dualscore-edgeproto --name vm-dualscore-trainer --yes`
