# Quiz: Convolutional Neural Networks

Answer each question in your own words. For formula questions, show your derivation.

---

**Q1.** What does a single convolutional kernel (filter) do when it slides over an input feature map? Describe concretely: what computation is performed at each position, and what is stored in the output feature map?

---

**Q2.** Use the output size formula to answer: Given an input of size **32×32**, a kernel of size **5×5**, stride **2**, and padding **0**, what is the output spatial dimension? Show your calculation step by step.

```
Output size = floor((input_size + 2·padding - kernel_size) / stride) + 1
```

---

**Q3.** What is the difference between **"valid" padding** and **"same" padding**? When would you choose each? How much padding (in pixels) do you need to achieve "same" output size with a 3×3 kernel and stride 1?

---

**Q4.** Max pooling is said to provide **translation invariance**. Explain what translation invariance means in the context of image recognition and describe the mechanism by which max pooling helps achieve it. Give a concrete example.

---

**Q5.** A convolutional layer has the following configuration:
- Input: 3 channels (RGB)
- Kernel size: 3×3
- Number of filters: 64
- Bias: one per filter

How many **learnable parameters** does this layer have? Show your calculation using the formula:
```
Parameters = (kernel_height × kernel_width × C_in + 1) × C_out
```

---

**Q6.** What do **feature maps** represent at different depths of a CNN? Describe what features you would expect to find in feature maps from:
- The first convolutional layer
- A middle convolutional layer
- The last convolutional layer before the classifier

---

**Q7.** Suppose you process a 256×256 grayscale image with:
- Option A: A fully-connected layer with 1000 hidden units.
- Option B: A convolutional layer with 32 filters of size 3×3.

Calculate the number of parameters for each option. Why is Option B so much more efficient, and what assumption makes this possible?

---

**Q8.** Explain the role of **depth** in a CNN. If early layers detect edges and simple patterns, what do deeper layers detect? How does the concept of **receptive field** explain this hierarchy?

---

**Q9.** ResNet introduced **skip connections** (residual connections) to solve a problem called the **degradation problem**. Explain:
- (a) What is the degradation problem?
- (b) How does the residual formulation `H(x) = F(x) + x` help?
- (c) How does it also help with vanishing gradients during backpropagation?

---

**Q10.** You have a pretrained ResNet50 model and a small custom dataset of 500 images across 5 classes. Describe your transfer learning strategy:
- Which layers would you freeze? Why?
- How would you modify the model's output layer?
- What learning rate would you use, and why might it differ from training from scratch?

---

**Q11.** Describe the complete **CNN forward pass pipeline** for classifying an image, including: input dimensions, what happens at each layer type (Conv, BatchNorm, ReLU, Pool, Flatten, FC, Softmax), and how dimensions change.

---

**Q12.** Strided convolutions (stride > 1) and max pooling both reduce spatial dimensions. What is the advantage of using strided convolutions instead of separate pooling layers? Which modern architecture uses this approach?

---

**Q13.** Two VGG-style design choices are: (1) use many small 3×3 kernels rather than fewer large kernels, (2) double the number of filters after each pooling. Justify each choice: what is the benefit of small kernels over large ones? Why double the filters?

---

**Q14.** **Global Average Pooling (GAP)** replaces large fully-connected layers in modern CNNs. Describe what GAP does, how many parameters it adds, and what advantage it has over traditional FC layers in terms of regularization and spatial flexibility.

---

## Answers (Hidden — attempt questions before reading)

<details>
<summary>Click to reveal answer guidance</summary>

1. The kernel slides across the input. At each position, it computes a dot product between its weights and the corresponding patch of the input. The result (a scalar) is placed in the corresponding position of the output feature map. Multiple kernels produce multiple feature maps.

2. floor((32 + 0 - 5) / 2) + 1 = floor(27/2) + 1 = 13 + 1 = 14. Output: 14×14.

3. Valid: no padding, output smaller than input (no information added). Same: pad with zeros so output matches input size. For 3×3 kernel, stride 1: p = (3-1)/2 = 1 pixel.

4. Translation invariance: detector fires whether a feature is slightly shifted. Max pool takes the maximum over a region, so a nearby activation produces the same output. E.g., a cat eye shifted 1px still produces the same max within the 2×2 window.

5. (3 × 3 × 3 + 1) × 64 = (27 + 1) × 64 = 28 × 64 = 1,792 parameters.

6. Layer 1: edges, gradients, color blobs. Middle: corners, textures, simple shapes. Last: high-level object parts (eyes, wheels), class-discriminative features.

7. Option A: 256×256×1 × 1000 = 65,536,000 params. Option B: (3×3×1+1)×32 = 320 params. CNN assumes spatial locality and translation invariance — a feature detector should be the same everywhere.

8. Depth enables hierarchical composition: simple features → complex patterns. Receptive field grows with depth: a neuron in layer 3 "sees" a larger region of the input. Deeper layers have larger effective receptive fields.

9. (a) Adding more layers to a deep network worsens accuracy even on training data (not overfitting — just harder to optimize). (b) F(x) only needs to learn the residual; if the layer should be identity, F(x)=0 is easier to learn than fitting the full identity function. (c) The identity shortcut provides a direct gradient path bypassing multiple non-linear layers.

10. Freeze all convolutional backbone layers (they already know ImageNet features). Replace the 1000-class FC head with a 5-class FC layer. Use a low LR (e.g., 1e-3 for the head, freeze backbone). If validation improves, optionally unfreeze last few layers with even lower LR (e.g., 1e-4).

11. Input (3,224,224) → Conv: (64,222,222) → BN+ReLU: same → Pool: (64,111,111) → ... → Flatten: (vector) → FC → Softmax: (n_classes,).

12. Strided convolutions learn the downsampling function rather than using a fixed operation. They can be more flexible. Modern networks like ResNet use strided convolutions in residual blocks.

13. Two 3×3 kernels cover the same 5×5 receptive field as one 5×5 kernel but use fewer parameters (2×9 < 25) and have an extra non-linearity. Doubling filters compensates for halved spatial size to maintain representational capacity.

14. GAP averages each feature map to a single value → vector of size C_out. Zero additional parameters. No risk of overfitting from FC layers. Works on any spatial input size (spatially invariant).

</details>
