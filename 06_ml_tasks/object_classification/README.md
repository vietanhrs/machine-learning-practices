# Object Classification

Object classification is the task of assigning one or more categorical labels to an
input (typically an image). It is one of the most studied problems in computer vision
and a cornerstone of modern deep learning.

---

## 1. What is Object Classification?

Given an input image (or other data), object classification assigns it to one or more
predefined categories.

**Single-label classification:** One label per image.
```
Input: [image of a cat] → Label: "cat"
```

**Multi-label classification:** Multiple labels per image.
```
Input: [image of a dog on a beach] → Labels: {"dog", "beach", "outdoor"}
```

Object classification differs from:
- **Object detection:** Localize AND classify multiple objects with bounding boxes.
- **Image segmentation:** Assign a label to every pixel.
- **Object recognition:** More general term, sometimes used interchangeably with classification.

---

## 2. Feature Extraction

Before classification, meaningful features must be extracted from raw pixels.

### Hand-Crafted Features (Traditional ML)

**HOG (Histogram of Oriented Gradients):**
- Divides image into small cells; computes edge direction histogram per cell
- Captures shape and texture information
- Used in pedestrian detection (Dalal & Triggs, 2005)

**SIFT (Scale-Invariant Feature Transform):**
- Detects keypoints that are invariant to scale and rotation
- Produces a 128-dimensional descriptor per keypoint
- Excellent for matching features across views of the same object

**LBP (Local Binary Patterns):**
- Compares each pixel to its neighbors; encodes as a binary number
- Efficient for texture classification and face recognition

### Learned Features (Deep Learning)

Convolutional Neural Networks automatically learn hierarchical feature representations:
- **Early layers:** Detect edges, colors, gradients
- **Middle layers:** Detect textures, patterns, object parts
- **Later layers:** Detect high-level concepts (faces, wheels, eyes)

This **end-to-end learning** typically outperforms hand-crafted features, especially
when sufficient labeled data is available.

---

## 3. Classification Pipeline

A standard object classification pipeline:

```
Raw Image
    ↓  (resize, normalize, convert to tensor)
Preprocessing
    ↓
Feature Extraction   ← (CNN, HOG, SIFT, etc.)
    ↓
Classifier           ← (softmax layer, SVM, random forest, etc.)
    ↓
Predicted Label(s)
```

For deep learning, feature extraction and classification are trained jointly.
For traditional ML, they are separate stages.

---

## 4. CNN for Classification

A standard CNN classifier:

```
Input Image (H × W × C)
    ↓  Conv + ReLU + MaxPool  (×multiple)
Feature Map
    ↓  Flatten / Global Average Pooling
Feature Vector
    ↓  Fully Connected Layer(s)
Logits (one per class)
    ↓  Softmax
Class Probabilities
```

**Key design choices:**
- Number of convolutional layers and filters
- Pooling strategy (max pool, average pool, strided convolution)
- Normalization (BatchNorm, LayerNorm)
- Skip connections (ResNet) to enable very deep networks

Famous CNN architectures: LeNet (1998), AlexNet (2012), VGG (2014),
GoogLeNet/Inception (2014), ResNet (2015), EfficientNet (2019), ViT (2020).

---

## 5. Multi-label vs. Multi-class Classification

| Property            | Multi-class                            | Multi-label                               |
|--------------------|----------------------------------------|-------------------------------------------|
| Labels per sample  | Exactly one                            | Zero or more                              |
| Output layer       | Softmax (probabilities sum to 1)       | Sigmoid per class (independent)          |
| Loss function      | Cross-entropy                          | Binary cross-entropy (per class)          |
| Example            | Classify digit 0–9                     | Tag image: {dog, outdoor, running}       |
| Prediction         | argmax of softmax output               | threshold each sigmoid output at 0.5     |

**Key insight:** In multi-label classification, classes are treated **independently**.
A sigmoid activation is applied to each output neuron separately, so a sample can
have any number of active labels.

---

## 6. Common Benchmarks

| Dataset     | Classes | Images     | Description                              |
|-------------|---------|------------|------------------------------------------|
| MNIST       | 10      | 70,000     | Handwritten digits (28×28 grayscale)    |
| Fashion-MNIST | 10   | 70,000     | Clothing items (28×28 grayscale)        |
| CIFAR-10    | 10      | 60,000     | Natural images (32×32 color)            |
| CIFAR-100   | 100     | 60,000     | 100 fine-grained classes                |
| ImageNet    | 1,000   | 1.2M       | Large-scale visual recognition (224×224)|
| iNaturalist | 5,089+  | 2.7M       | Fine-grained species classification     |
| Open Images | 600     | 9M+        | Multi-label image classification        |

**ImageNet's impact:** The ImageNet Large Scale Visual Recognition Challenge (ILSVRC)
drove the deep learning revolution in computer vision. AlexNet's win in 2012 (top-5
error of 15.3% vs. 26.2% for the runner-up) triggered widespread adoption of deep CNNs.

---

## 7. Data Augmentation for Images

Data augmentation applies random transformations to training images to increase
effective dataset size and improve generalization.

### Common Augmentations

| Technique        | Description                                      | Notes                                        |
|------------------|--------------------------------------------------|----------------------------------------------|
| Horizontal Flip  | Mirror image left-right                          | Not valid for text or directional objects   |
| Vertical Flip    | Mirror image top-bottom                          | Rarely used for natural images              |
| Random Crop      | Crop a random subregion of the image             | Forces model to use local features          |
| Rotation         | Rotate by random angle                           | ±15° is common                              |
| Color Jitter     | Randomly change brightness/contrast/saturation  | Helps with lighting variation               |
| Gaussian Noise   | Add random noise to pixels                       | Encourages robustness                       |
| Cutout/Random Erase | Mask out a random rectangular region         | Simulates occlusion                         |

### Advanced Augmentations

- **Mixup:** Blend two images and their labels linearly: `x = λ*x1 + (1-λ)*x2`
- **CutMix:** Replace a random patch from one image with a patch from another
- **AutoAugment / RandAugment:** Automatically search for optimal augmentation policies

---

## 8. Evaluation

### Accuracy

```
Accuracy = (Number of correct predictions) / (Total predictions)
```

Simple but misleading when classes are imbalanced.

### Top-5 Accuracy

Count a prediction as correct if the true label is among the top-5 predictions.
Used in ImageNet evaluations where 1,000 classes make top-1 accuracy very stringent.

### Confusion Matrix

A K×K matrix where entry (i, j) = number of samples from class i predicted as class j.
Reveals per-class performance and common misclassification patterns.

### Per-class Metrics

Compute precision, recall, and F1 for each class separately — essential when
class frequencies are unequal (imbalanced datasets).

---

## 9. Common Pitfalls

### Class Imbalance

If training data has 9,000 images of class A and 100 of class B, the model will
learn to mostly predict class A. Solutions: oversampling, class-weighted loss,
or collecting more data for underrepresented classes.

### Distribution Shift

The model performs well on the training/test distribution but poorly when deployed
on real-world data that comes from a different distribution (different camera angle,
lighting, demographic, etc.).

### Shortcut Learning (Clever Hans Effect)

Named after a horse that appeared to count but was actually reading subtle human
cues — models can latch onto spurious correlations in the data.

**Example:** A model trained to detect cows learns to predict "cow" whenever it sees
a green pasture background. If evaluated on images of cows indoors, accuracy drops sharply.

**Diagnosis:** Saliency maps (Grad-CAM) can reveal which image regions the model uses.

**Prevention:** Diverse training data, causal modeling, adversarial training.

---

## 10. Reference Links

- [Papers with Code: Image Classification](https://paperswithcode.com/task/image-classification)
- [PyTorch Vision Datasets](https://pytorch.org/vision/stable/datasets.html)
- [ImageNet Challenge History](https://image-net.org/challenges/LSVRC/)
- [Deep Residual Learning (He et al., 2015 — ResNet)](https://arxiv.org/abs/1512.03385)
- [EfficientNet (Tan & Le, 2019)](https://arxiv.org/abs/1905.11946)
