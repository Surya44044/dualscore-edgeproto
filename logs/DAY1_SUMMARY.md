# Day 1 Summary — EdgeProto-OSR

## Machine
Ubuntu Linux, Intel i3 dual-core, 8GB RAM, CPU-only.

## Completed
- Copied Antigravity project into a safe folder.
- Created and activated Python virtual environment.
- Verified CPU-only PyTorch installation.
- Confirmed CUDA is unavailable, as expected.
- Inspected run_all.sh for dangerous commands.
- Generated synthetic CICIoT-like dataset.
- Preprocessed synthetic dataset using low-resource pipeline.
- Created H1 Mirai zero-day holdout split.
- Trained EdgeProto model on synthetic data.
- Evaluated closed-set and open-set metrics.
- Fixed Python import path issue using PYTHONPATH.
- Fixed missing training configuration issue.
- Fixed NumPy np.trapz compatibility issue.
- Fixed TorchScript quantized save failure using fallback checkpoint.
- Installed ptflops and measured MACs/parameters.
- Full run_all.sh synthetic smoke test completed successfully.

## Smoke-Test Results
Synthetic data only. These are NOT final research results.

EdgeProto:
- Parameters: approximately 3.84k
- Computational complexity: approximately 9.68 KMac/sample
- FP32 checkpoint size: approximately 22.96 KB
- INT8 fallback checkpoint size: approximately 17.80 KB

Synthetic H1 Mirai Holdout:
- Closed-set accuracy: approximately 0.934
- Open-set AUROC: approximately 0.492
- OSCR: approximately 0.481

## Important Integrity Note
Synthetic results are only for verifying that the software pipeline works. They must not be used as dissertation/paper performance claims. Final reported metrics must come from real datasets such as RT-IoT2022 and CICIoT2023.

## Next Step
Day 2: Download, inspect, and prepare RT-IoT2022 real dataset.
