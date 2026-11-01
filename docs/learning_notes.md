# Month 5 Learning Notes — Semantic Segmentation

## 1. Classification, detection and segmentation

- Image classification assigns one or more labels to a whole image.
- Object detection predicts object categories and bounding boxes.
- Semantic segmentation predicts a class for every pixel.

A segmentation model returns a tensor of class scores with shape `[B, C, H, W]`, where `B` is batch size, `C` is the number of semantic classes, and `H × W` is the output spatial resolution.

## 2. Semantic segmentation

Every pixel receives a semantic category. In this project the classes are pet, background and boundary. Semantic segmentation does not distinguish two separate objects of the same class as different object instances; that is the purpose of instance segmentation.

## 3. FCN

A Fully Convolutional Network replaces fully connected classification layers with convolutional operations so that spatial predictions can be produced. Learn the idea of preserving spatial layout and recovering resolution.

## 4. Encoder-decoder

The encoder progressively reduces spatial resolution and learns higher-level features. The decoder progressively increases spatial resolution to produce pixel-level predictions.

Downsampling increases receptive field and reduces computation, but it can lose fine spatial details.

## 5. U-Net

U-Net combines an encoder and decoder with skip connections. At each resolution, encoder features are concatenated with decoder features at the corresponding scale.

Why concatenate?
- Encoder features retain local spatial detail.
- Decoder features contain context from deeper layers.
- Combining them helps reconstruct boundaries and small structures.

The original U-Net paper was designed for biomedical image segmentation. This project uses the architecture as a learning baseline on Oxford-IIIT Pet, not as a direct reproduction of the paper's original experiments.

## 6. Output logits

The model returns raw logits. Do not apply softmax inside the model when using `CrossEntropyLoss`; PyTorch's cross-entropy combines log-softmax and negative log likelihood internally. For predictions, use `argmax` across the class dimension.

## 7. Data augmentation

A horizontal flip is applied jointly to the image and its mask. The exact same spatial transformation must be applied to both. Resizing masks uses nearest-neighbor interpolation because class IDs are categorical; bilinear interpolation would create invalid intermediate labels.

## 8. What to explain in your own words

1. Why is segmentation a dense prediction task?
2. Why do we need an encoder?
3. What information is lost during pooling?
4. How do skip connections help?
5. Why is mask resizing different from image resizing?
6. Why are logits returned instead of probabilities?
7. Why can pixel accuracy be misleading when classes are imbalanced?
8. Why should test data be used only once for final evaluation?
