# EGPDC-Seg

Official implementation of **Expert Guidance and Partially-Labeled Data Collaboration for Multi-Organ Segmentation**, published in *Neural Networks* (2025).

[Paper](https://doi.org/10.1016/j.neunet.2025.107396) | [PubMed](https://pubmed.ncbi.nlm.nih.gov/40132452/) | [Citation](#citation) | [Results](#key-results)

EGPDC-Seg is a federated framework for abdominal CT multi-organ segmentation that transfers multi-organ knowledge from fully labeled expert clients to partially labeled clients. Its reward-penalty loss enables effective learning from valuable single-organ annotations.

## Highlights

- Reuses cost-efficient single-organ annotations for multi-organ segmentation.
- Transfers multi-organ priors through an expert-guidance module (EGM).
- Learns effectively from partially labeled data with reward-penalty loss (RP loss).
- Supports privacy-preserving collaboration by exchanging model parameters across institutions.
- Evaluated on FLARE22, TCIA, AMOS, Synapse, and WORD.

## Method

EGPDC-Seg simulates three medical institutions as federated clients. Two clients use fully labeled multi-organ datasets, while one client uses a partially labeled dataset containing a verified pancreas annotation. The framework has two key components:

1. **Reward-penalty loss:** emphasizes verified organ annotations and promotes effective learning from partially labeled data.
2. **Expert-guidance module:** preserves the global model as an expert during local training and transfers its multi-organ knowledge to the partially labeled client.

The paper uses a weighted-average aggregation strategy. Its main experiments use 30 communication rounds, 20 local epochs, Adam with a learning rate of `2e-4`, a batch size of `8`, and inputs resized to `224 x 224`.

## Key Results

The following values are mean Dice scores reported in the published paper.

| Setting | Test dataset | Baseline with partial labels | + RP loss | + RP loss + EGM |
|---|---:|---:|---:|---:|
| AMOS partially labeled | Synapse | 54.20 | 74.89 | **80.85** |
| TCIA partially labeled | Synapse | 46.34 | 68.93 | **80.41** |
| FLARE22 partially labeled | Synapse | 50.56 | 68.73 | **80.40** |

The framework is also evaluated with different backbones. On the unseen Synapse dataset, mean Dice reaches `80.85` with U-Net, `83.06` with UNet++, `80.87` with TransUNet, and `83.10` with TransAttUnet.

## Datasets

The paper uses five public abdominal CT datasets:

- FLARE22
- TCIA
- AMOS
- Synapse
- WORD

Together, these datasets provide diverse multi-center evaluation settings for studying partially labeled abdominal CT segmentation and external generalization.

## Citation

```bibtex
@article{li2025expert,
  title     = {Expert Guidance and Partially-Labeled Data Collaboration for Multi-Organ Segmentation},
  author    = {Li, Li and Liu, Jianyi and Xiao, Hanguang and Zhou, Guanqun and Liu, Qiyuan and Zhang, Zhicheng},
  journal   = {Neural Networks},
  volume    = {187},
  pages     = {107396},
  year      = {2025},
  month     = jul,
  doi       = {10.1016/j.neunet.2025.107396},
  url       = {https://doi.org/10.1016/j.neunet.2025.107396},
  publisher = {Elsevier}
}
```
