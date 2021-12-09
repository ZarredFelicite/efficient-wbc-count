# White Blood Cell Detection & Classification

## Classification
placeholder discription

**Network saliency visualisation for ResNet18**
<img src="Media/saliency_vis.png" width="900" alt="saliency" />

WBC classified as Neutrophil with 99% confidence.
(Image hue has been shifted)

### Quantization for WBC Classification

#### PTQ & QAT on ResNet18, ResNet34, ResNet50 and MobileNetV2
##### ResNet18 Results Summary
<p float="left">
<img src="Charts/Section-1-Panel-3-1la8vceuj.png" width="275" alt="Acc"/>
<img src="Charts/Section-1-Panel-2-i4cwouhu7.png" width="275" alt="F1"/>
<img src="Charts/Section-3-Panel-1-fy2s98k31.png" width="275" alt="acc_time"/>
</p>

Full Results: https://wandb.ai/zarred/Baseline%20Training/reports/PTQ-QAT-Results--VmlldzoxMjY1MDE2

### Main Findings
- QAT provides limited advantage over PTQ for ResNet Models but recovers more predictive performance for MobileNetV2 because of the inclusion of depthwise seperable layers.
- ResNet18 with quantization seems to provide the best balance of accuracy to inference speed.
