# Convolutional Neural Networks (Mạng tích chập)

CNNs are specialized neural networks designed for data with spatial structure — primarily images. By exploiting local connectivity and weight sharing, they dramatically reduce parameters compared to fully-connected networks while learning hierarchical visual features.

---

## 1. Motivation: Why Convolutions for Images?

Consider a 224×224 RGB image. A fully-connected layer from input to 1000 hidden neurons would require `224 × 224 × 3 × 1000 ≈ 150 million` parameters — just for the first layer. This is:
- **Computationally expensive** to train.
- **Prone to overfitting** without enormous datasets.
- **Wasteful**: a cat's ear looks the same whether it appears top-left or bottom-right.

Convolutions address this by exploiting two key properties of natural images:
1. **Local structure**: Meaningful patterns (edges, textures) are local.
2. **Translation invariance**: A feature detector that works in one location should work everywhere.

A convolutional layer with a 3×3 kernel has only `3 × 3 × C_in × C_out` parameters, regardless of image size.

---

## 2. The Convolution Operation

### How a Kernel Slides Over the Input

A **kernel** (or filter) is a small matrix of learnable weights. It slides across the input feature map, computing a dot product at each position.

```
Input (5×5):              Kernel (3×3):          Output (3×3):
┌─────────────────┐       ┌─────────┐             ┌─────────┐
│ 1  2  3  0  1  │       │ 1  0 -1 │             │  ?  ?  ?│
│ 0  1  2  3  2  │  *    │ 1  0 -1 │    =        │  ?  ?  ?│
│ 1  0  1  2  0  │       │ 1  0 -1 │             │  ?  ?  ?│
│ 2  1  0  1  3  │       └─────────┘             └─────────┘
│ 1  2  3  1  0  │
└─────────────────┘

At position (0,0):
  sum([1·1 + 2·0 + 3·(-1) +
       0·1 + 1·0 + 2·(-1) +
       1·1 + 0·0 + 1·(-1)]) = 1 - 3 - 2 + 1 - 1 = -4
```

This particular kernel detects **vertical edges** (Sobel-like filter).

### Output Size Formula

```
Output size = floor((input_size + 2·padding - kernel_size) / stride) + 1
```

Examples with input 28×28, kernel 3×3:
| Padding | Stride | Output Size |
|---------|--------|-------------|
| 0 | 1 | 26×26 (valid) |
| 1 | 1 | 28×28 (same) |
| 0 | 2 | 13×13 |

### Stride

Stride controls how many pixels the kernel moves at each step:
- **Stride 1**: Dense, fine-grained feature maps.
- **Stride 2**: Halves spatial dimensions, reduces computation (alternative to pooling).

### Padding

- **Valid padding (p=0)**: No padding; output is smaller than input.
- **Same padding**: Add zeros around the border so output matches input size.

---

## 3. Pooling Layers

Pooling reduces spatial dimensions, making representations smaller and more robust to small translations.

### Max Pooling

```
Input (4×4):                 Max Pool 2×2, stride 2:
┌────────────────┐           ┌─────────┐
│ 1  3  2  4    │           │  3   4  │
│ 5  6  7  8    │    →      │  6   8  │  ←  wrong! see below
│ 3  2  1  0    │           │  6   9  │
│ 1  2  9  5    │           └─────────┘
└────────────────┘

Correct:
Top-left 2×2: max(1,3,5,6) = 6   Top-right 2×2: max(2,4,7,8) = 8
Bot-left 2×2: max(3,2,1,2) = 3   Bot-right 2×2: max(1,0,9,5) = 9

Output:
┌─────┐
│ 6  8│
│ 3  9│
└─────┘
```

- **Purpose**: Translation invariance — small shifts in input barely change the pooled output.
- Max pooling retains the strongest activation in each region.

### Average Pooling

Takes the **mean** of each pooling window instead of the maximum. Used in some architectures (e.g., global average pooling before the classifier head).

### Global Average Pooling (GAP)

Collapses each entire feature map to a single number (its mean). Used in modern CNNs (ResNet, GoogLeNet) to replace large fully-connected layers.

---

## 4. CNN Architecture Pattern

The classic building block of a CNN:

```
Input Image
    │
    ▼
[Conv Layer]     ← learns spatial features (kernels)
    │
    ▼
[Batch Norm]     ← optional: stabilizes training
    │
    ▼
[ReLU]           ← non-linearity
    │
    ▼
[Max Pool]       ← spatial downsampling
    │
    ▼
(repeat above N times)
    │
    ▼
[Flatten]        ← convert 2D feature maps to 1D vector
    │
    ▼
[Fully Connected]← combine learned features
    │
    ▼
[Softmax Output] ← class probabilities
```

Each successive Conv layer learns increasingly **abstract features**:
- **Layer 1**: edges, orientations, colors
- **Layer 2**: corners, simple shapes, textures
- **Layer 3+**: object parts (eyes, wheels), entire objects

---

## 5. Classic Architectures

### LeNet-5 (LeCun et al., 1998)
The pioneering CNN applied to handwritten digit recognition (MNIST). Architecture: two conv layers followed by three fully-connected layers. Used tanh activations and average pooling. Demonstrated that learned convolutional features outperform hand-crafted features.

### AlexNet (Krizhevsky et al., 2012)
Won ImageNet LSVRC 2012 by a large margin, igniting the deep learning revolution. Key innovations: ReLU activations, dropout for regularization, data augmentation, GPU training. 5 conv layers, 3 FC layers, ~60 million parameters.

### VGG (Simonyan & Zisserman, 2014)
Showed that depth matters more than kernel size. Used only 3×3 conv kernels throughout, stacked deeply (16–19 layers). Very simple and uniform architecture. Two stacked 3×3 convolutions have the same receptive field as one 5×5 kernel but fewer parameters and an extra non-linearity.

### ResNet (He et al., 2015)
Introduced **residual (skip) connections**: `output = F(x) + x`. This solved the degradation problem — adding more layers no longer hurts accuracy. Enabled networks 50–152+ layers deep. Won ImageNet 2015. ResNet50 remains one of the most widely used CNN backbones today.

---

## 6. Transfer Learning

Training a CNN from scratch requires millions of labeled images and days of GPU time. **Transfer learning** leverages a model pretrained on a large dataset (e.g., ImageNet with 1.2M images, 1000 classes):

### Approach

1. **Load pretrained model** (e.g., ResNet50 pretrained on ImageNet).
2. **Freeze backbone** (convolutional layers) — their weights remain fixed.
3. **Replace classifier head** — swap the final FC layer for one matching your number of classes.
4. **Fine-tune** — train only the new head (and optionally unfreeze later layers gradually).

### When to Freeze vs Fine-tune

| Scenario | Strategy |
|----------|----------|
| Small dataset, similar to ImageNet | Freeze all layers, train head only |
| Medium dataset, different domain | Freeze early layers, fine-tune later layers |
| Large dataset | Fine-tune entire network (lower LR for backbone) |

Transfer learning can reduce training time from days to minutes and significantly improve accuracy on small datasets.

---

## 7. Reference Links

- [CS231n: Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/convolutional-networks/)
- [PyTorch CIFAR-10 Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
