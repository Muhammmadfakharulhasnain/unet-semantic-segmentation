# Paper Reading Notes — U-Net

## Paper

Ronneberger, O., Fischer, P., and Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI 2015.

Link: https://arxiv.org/abs/1505.04597

## Reading worksheet

### 1. Problem
What segmentation problem did the authors address?

### 2. Why previous methods were insufficient
What challenge motivated the architecture?

### 3. Main idea
Explain the U-shaped encoder-decoder and contracting/expanding paths.

### 4. Architecture
Draw and label:
- contracting path
- bottleneck
- expanding path
- skip connections
- final 1x1 convolution

### 5. Data and augmentation
What data did the paper use? What augmentation strategy did it use, and why?

### 6. Loss and optimization
Record the loss, optimizer, learning rate and training details reported by the paper.

### 7. Evaluation
What datasets and evaluation criteria were used?

### 8. Main results
Record only values you can verify directly in the paper. Do not transfer the paper's results to this project.

### 9. What I understood
Write a concise explanation without copying the abstract.

### 10. What I do not understand
List specific technical questions.

### 11. Connection to this implementation
What differs between the original paper and this educational PyTorch implementation?

### 12. Possible extension
What experiment could you run after establishing a reproducible baseline?
