# Security Policy

## Reporting Security Vulnerabilities

If you discover a potential security vulnerability in DualScore-EdgeProto, please report it immediately by opening a confidential issue or emailing the project maintainers.

---

## Secret & Data Protection Guidelines

To prevent accidental leakage of sensitive credentials or dataset assets:

1. **Zero Secret Policy**:
   - **NEVER** commit API keys, service principal JSON files (`azure-credentials*.json`, `service-principal*.json`), SSH keys, DockerHub access tokens, or `.env` files to Git.
   - All credentials must be stored exclusively in **GitHub Repository Secrets** or Azure Key Vault.

2. **Zero Dataset Policy**:
   - Raw (`data/raw/`), processed (`data/processed/`), and split (`data/splits/`) datasets must remain in local uncommitted storage or secure blob storage.
   - Datasets must **NEVER** be copied into Docker images or uploaded to Azure Container Apps.

3. **Inference-Only Deployment**:
   - Production API containers on Azure Container Apps serve **inference endpoints only**.
   - No training operations run on public Container Apps endpoints.

4. **Secret Scanning**:
   - Pre-commit and CI/CD pipelines run non-destructive secret detection using `scripts/check_no_secrets.sh`.
