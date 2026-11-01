# Oxford-IIIT Pet — Dataset Validation Report

## 1. Dataset inventory

| Item | Count |
|---|---:|
| JPEG images | 7,390 |
| PNG segmentation masks | 7,390 |
| Matching image-mask pairs | 7,390 |
| Official trainval samples | 3,680 |
| Official test samples | 3,669 |
| Samples outside official splits | 41 |

## 2. Image-mask matching

- Unmatched images: 0
- Unmatched masks: 0

## 3. Official split validation

- Trainval/test overlap: 0
- Trainval missing images: 0
- Test missing images: 0
- Trainval missing masks: 0
- Test missing masks: 0

## 4. Mask inspection

Sample: Abyssinian_1.png

- Shape: 400 x 600
- Data type: uint8
- Unique labels: 1, 2, 3

| Label | Meaning | Pixels | Percentage |
|---|---|---:|---:|
| 1 | Pet | 22,938 | 9.56% |
| 2 | Background | 198,766 | 82.82% |
| 3 | Boundary | 18,296 | 7.62% |

## 5. Full official dataset quality check

| Check | Result |
|---|---:|
| Official samples checked | 7,349 |
| Dimension mismatches | 0 |
| Invalid-label masks | 0 |
| File-reading errors | 0 |

Status: PASSED

## 6. Visual inspection

- Original image inspected.
- Ground-truth mask visualized.
- Image-mask overlay generated.
- Sample visualization: results/figures/dataset_sample_visualization.png

## 7. Experimental protocol

Use the official trainval and test lists.

Do not include the 41 images outside the official splits in the experiments.

A validation subset must be created from trainval only. Keep the official test set untouched until final evaluation.
