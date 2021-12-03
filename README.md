# Quantization for WBC Classification

## PTQ & QAT on ResNet18, ResNet34, ResNet50 and MobileNetV2
### ResNet18 Results Summary
<img src="https://gitlab.erc.monash.edu.au/dmeh0007/hematology/-/raw/8825f94ce00e9021dfdd1674fef7512273b061d5/Charts/Section-1-Panel-3-1la8vceuj.png" width="300" alt="Acc" />
<img src="https://gitlab.erc.monash.edu.au/dmeh0007/hematology/-/raw/8825f94ce00e9021dfdd1674fef7512273b061d5/Charts/Section-1-Panel-2-i4cwouhu7.png" width="300" alt="F1" />
<img src="https://gitlab.erc.monash.edu.au/dmeh0007/hematology/-/raw/8825f94ce00e9021dfdd1674fef7512273b061d5/Charts/Section-3-Panel-1-fy2s98k31.png" width="300" alt="acc_time" />


Full Results: https://wandb.ai/zarred/Baseline%20Training/reports/PTQ-QAT-Results--VmlldzoxMjY1MDE2

### Main Findings
- QAT provides limited advantage over PTQ for ResNet Models but recovers more predictive performance for MobileNetV2 because of the inclusion of depthwise seperable layers.
- ResNet18 with quantization seems to provide the best balance of accuracy to inference speed.
