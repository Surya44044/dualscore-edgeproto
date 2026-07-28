# Day 2 RT-H1 Validation Summary

## Dataset
RT-IoT2022

## Experiment
RT-H1: ARP_Spoofing held out as unknown / zero-day class.

## Known Classes
- Benign
- DDoS
- DoS
- Recon

## Unknown Class
- ARP_Spoofing

## Feature Mode
conservative_no_ports

## Feature Integrity
- 48 numeric traffic-statistical features.
- No Attack_type leakage.
- No label leakage.
- No Unnamed: 0 row-index leakage.
- No id.orig_p or id.resp_p port features.
- No NaN/Inf values.

## Split Counts
- Train: 80,731
- Validation: 11,535
- Test Known: 23,064
- Test Unknown: 7,750

## Clean Run ID
rt_h1_arp_spoofing_clean_20260728_120042

## Training
- Epochs: 30
- Hardware: CPU only
- Runtime: 2 min 25 sec
- Peak RAM: approximately 473,848 KB
- Exit status: 0

## EdgeProto Resource Metrics
- Parameters: approximately 3.84k
- MACs/sample: approximately 9.68 KMac

## Known-Class Metrics
- Accuracy: 0.9979188345
- Macro Precision: 0.9901596146
- Macro Recall: 0.9589071661
- Macro F1: 0.9731975015

## Open-Set Metrics
- AUROC: 0.9896531419
- OSCR: 0.9891338547

## Interpretation
This is a validated single-run RT-IoT2022 H1 result. It is not yet the final paper result because additional holdouts, baselines, repeated seeds, quantization evaluation, and CICIoT2023 experiments are still required.
