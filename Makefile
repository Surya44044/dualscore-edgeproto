.PHONY: setup smoke api test-api docker-build docker-run docker-build-train ci-local secret-scan clean-cache help

SHELL := /bin/bash
VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

help:
	@echo "DualScore-EdgeProto Development Makefile"
	@echo "----------------------------------------"
	@echo "make setup              : Install dependencies in .venv"
	@echo "make smoke              : Run synthetic model smoke test"
	@echo "make api                : Run local FastAPI inference service"
	@echo "make test-api           : Test FastAPI endpoints locally"
	@echo "make docker-build       : Build local Docker inference image"
	@echo "make docker-run         : Run local Docker inference container"
	@echo "make docker-build-train : Build local Docker training image"
	@echo "make secret-scan        : Run non-destructive secret leakage scanner"
	@echo "make ci-local           : Run complete local CI check pipeline"
	@echo "make clean-cache        : Clean Python __pycache__ and build artifacts"

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

smoke:
	bash scripts/run_synthetic_smoke.sh

api:
	bash scripts/run_api_local.sh

test-api:
	bash scripts/test_api_local.sh

docker-build:
	bash scripts/docker_build_local.sh

docker-run:
	bash scripts/docker_run_local.sh

docker-build-train:
	bash scripts/docker_build_train.sh

secret-scan:
	bash scripts/check_no_secrets.sh

ci-local: secret-scan smoke test-api
	@echo "[SUCCESS] Local CI verification pipeline completed cleanly!"

clean-cache:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Cache cleared."
