# Azure Container Apps Specification

This directory holds the declarative YAML template for deploying the DualScore-EdgeProto inference service.

## Configuration Details
- **vCPU**: 0.25 vCPU
- **Memory**: 0.5 GiB
- **Port**: 8000
- **Ingress**: External HTTP Ingress enabled
- **Scaling Rules**: Min replicas = 0, Max replicas = 5
