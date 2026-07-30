# Contributing to DualScore-EdgeProto

Thank you for contributing to DualScore-EdgeProto! Please adhere to the guidelines below when submitting pull requests or making modifications.

---

## Coding Standards & Conventions

1. **Python Style**:
   - Follow PEP 8 guidelines.
   - Run `python -m compileall -q src` before submitting code to ensure zero syntax errors.

2. **CPU-Only Constraints**:
   - Ensure all model layers and data pipelines maintain PyTorch CPU compatibility.
   - Set `num_workers=0` and keep batch size $\le 256$ for memory efficiency.

3. **No Secret or Data Leakage**:
   - Run `bash scripts/check_no_secrets.sh` prior to committing changes.
   - Do not add `.csv`, `.npz`, or `.pt` files outside permitted directories (`examples/` or `results/tables/`).

---

## Development Workflow

1. **Create Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Test Locally**:
   ```bash
   make ci-local
   ```

3. **Submit Pull Request**:
   - Push your branch to GitHub and open a PR against `main`.
   - CI will automatically verify syntax, run secret scans, and test API endpoints.
