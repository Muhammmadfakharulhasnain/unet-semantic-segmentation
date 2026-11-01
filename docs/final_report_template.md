# U-Net Semantic Segmentation on Oxford-IIIT Pet — Final Report

> Replace every bracketed field with information from your own run. Do not insert estimated or illustrative performance values.

## 1. Abstract
[Problem, method, dataset, actual main findings and one limitation. Write this after experiments finish.]

## 2. Research question
How effectively can a U-Net segment Oxford-IIIT Pet images into semantic regions, and how do data augmentation and loss-function choices affect segmentation quality?

## 3. Background
[Explain semantic segmentation, FCN, encoder-decoder models and U-Net.]

## 4. Dataset
- Dataset: Oxford-IIIT Pet
- Official split: [record exact split]
- Internal train/validation split: 85% / 15% of official trainval
- Image resolution: [from config]
- Classes: pet, background, boundary
- Preprocessing and augmentation: [describe]

## 5. Architecture
[Explain the convolution blocks, encoder, bottleneck, decoder, skip connections and output logits. Include tensor shapes and parameter count.]

## 6. Experimental setup
- Device: [CPU/GPU and exact model]
- Python/PyTorch/torchvision versions: [record]
- Optimizer: AdamW
- Learning rate: [record]
- Weight decay: [record]
- Batch size: [record]
- Epochs / early stopping: [record]
- Random seed: [record]

## 7. Results
Insert the actual contents of `results/tables/experiment_comparison.csv` here.

| Experiment | Augmentation | Loss | Pixel accuracy | Mean IoU | Mean Dice | Macro precision | Macro recall |
|---|---|---|---:|---:|---:|---:|---:|
| baseline_ce | No | Cross-Entropy | Not run | Not run | Not run | Not run | Not run |
| augmented_ce | Yes | Cross-Entropy | Not run | Not run | Not run | Not run | Not run |
| augmented_dice | Yes | Dice | Not run | Not run | Not run | Not run | Not run |
| augmented_combined | Yes | Combined | Not run | Not run | Not run | Not run | Not run |

The words “Not run” are placeholders, not experimental results. Replace them only with measured values.

## 8. Training dynamics
[Insert loss and mean-IoU plots. Discuss convergence, overfitting and early stopping.]

## 9. Per-class analysis
[Discuss pet, background and boundary metrics separately.]

## 10. Qualitative analysis and failure cases
[Insert prediction grids. Analyze at least ten difficult examples.]

## 11. Ablation and interpretation
[Compare augmentation and losses. State which differences are supported by the measured evidence.]

## 12. Limitations
[Small dataset, limited resolution, single seed, boundary ambiguity, compute budget and any observed implementation limitations.]

## 13. Conclusion and future work
[Summarize what was learned and propose a next experiment.]

## 14. References
Ronneberger, O., Fischer, P., and Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI 2015. https://arxiv.org/abs/1505.04597
