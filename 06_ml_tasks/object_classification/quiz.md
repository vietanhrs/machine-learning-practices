# Object Classification — Quiz

Test your understanding of object classification concepts, techniques, and pitfalls.

---

**Q1.** What is the key difference between **multi-class** and **multi-label** classification?

<details>
<summary>Answer</summary>

| Property           | Multi-class                                     | Multi-label                                      |
|--------------------|--------------------------------------------------|--------------------------------------------------|
| Labels per sample  | Exactly one label                               | Zero or more labels                              |
| Output activation  | Softmax (probabilities sum to 1)                | Sigmoid per class (independent, can sum to any) |
| Loss function      | Categorical cross-entropy                       | Binary cross-entropy per class                  |
| Example            | Classify a digit as 0, 1, 2, ..., 9            | Tag an image as {dog, outdoor, running}         |

In multi-class classification, classes are **mutually exclusive**. In multi-label,
a single image can simultaneously belong to multiple classes.

</details>

---

**Q2.** What does **top-5 accuracy** mean in image classification?

- A) The model must predict the correct label in fewer than 5 attempts
- B) A prediction is counted as correct if the true label is among the model's 5 highest-probability predictions
- C) The model is evaluated on only the 5 most common classes
- D) Accuracy is averaged across 5 different test splits

<details>
<summary>Answer</summary>

**B) A prediction is counted as correct if the true label is among the model's 5 highest-probability predictions.**

Top-5 accuracy is commonly used on ImageNet (1,000 classes) where distinguishing between
very similar categories (e.g., 120 dog breeds) makes top-1 accuracy very stringent.
Humans achieve roughly 95% top-5 accuracy on ImageNet; state-of-the-art models now
surpass human top-5 performance.

</details>

---

**Q3.** Explain how **data augmentation** helps improve model generalization. Why doesn't it help infinitely?

<details>
<summary>Answer</summary>

**How it helps:**

Data augmentation artificially increases the effective training set size by applying
random transformations (flips, crops, rotations, color jitter) to training images.
This forces the model to:
1. Be invariant to those transformations (a cat is still a cat when flipped)
2. Rely on the object's intrinsic features rather than spurious background/position cues
3. Not overfit to exact pixel values seen in training

**Why it doesn't help infinitely:**

1. **Domain mismatch:** Augmentations must be realistic. Aggressively rotating a digit 180°
   changes its class (6 becomes 9). Wrong augmentations introduce noise or label errors.
2. **Fundamental data scarcity:** Augmentation diversifies existing data but cannot create
   truly new examples of rare object poses, lighting conditions, or categories.
3. **Diminishing returns:** After enough diverse augmentations, the limiting factor
   becomes the model's capacity or the intrinsic difficulty of the task.

</details>

---

**Q4.** What is **shortcut learning** (the Clever Hans effect)? Give a concrete example from computer vision.

<details>
<summary>Answer</summary>

**Shortcut learning** occurs when a model achieves high performance on training/validation
data by learning **spurious correlations** (shortcuts) rather than the intended causal
features.

The name comes from "Clever Hans," a horse in early 1900s Germany that appeared to
solve arithmetic problems but was actually reading unconscious cues from its trainer.

**Computer vision example:**

A chest X-ray classifier trained to detect pneumonia learned to predict pneumonia
based on the **presence of a chest drain tube** (which is inserted in severe cases)
rather than the actual pneumonia pattern in the lung tissue. The model achieved high
accuracy on the original test set because the tube and pneumonia were correlated in the
training data — but the model failed completely on images of patients with drains
and no pneumonia.

**Other examples:**
- A "cow detector" that learned to recognize green pastures (most cow images showed cows outdoors)
- A skin cancer detector that learned to detect ruler marks in images (dermatologists use rulers
  on suspicious lesions, which are disproportionately represented in the positive class)

**Diagnosis and prevention:**
- Saliency maps / Grad-CAM to inspect what regions the model attends to
- Adversarial testing on images where the shortcut is absent
- Dataset curation to remove or balance spurious correlations

</details>

---

**Q5.** A model trained on images from one hospital achieves 92% accuracy. When deployed at a different hospital, it drops to 71%. What is this problem called, and what are two possible causes?

<details>
<summary>Answer</summary>

This is called **distribution shift** (also known as **domain shift** or **covariate shift**).

**Possible causes:**

1. **Equipment differences:** Different CT scanners, X-ray machines, or imaging protocols
   produce images with different contrast, resolution, or artifacts. The model learned
   features specific to the first hospital's equipment.

2. **Patient population differences:** The hospitals may serve different demographic groups
   with different disease prevalence, severity distributions, or comorbidities. If the
   second hospital treats sicker or younger patients, the class distribution differs.

3. **Preprocessing pipeline:** Different hospitals may apply different image preprocessing
   (normalization, DICOM processing), causing the input distribution to differ.

4. **Labeling practices:** Different radiologists may label the same finding differently
   (label shift), affecting what "ground truth" means.

**Solutions:** Fine-tuning on a small amount of target-domain data, domain adaptation
techniques, or training on diverse multi-site data from the start.

</details>

---

**Q6.** What is the difference between **HOG features** and **CNN features** for image classification?

<details>
<summary>Answer</summary>

| Aspect            | HOG (Histogram of Oriented Gradients)         | CNN Features                                    |
|-------------------|------------------------------------------------|--------------------------------------------------|
| How obtained      | Hand-crafted algorithm                         | Learned from data via backpropagation           |
| Invariances       | Manually designed (contrast normalization)     | Learned from data (e.g., translation via pooling)|
| Expressiveness    | Limited to gradient-based structure            | Hierarchical, abstract, task-specific           |
| Data requirement  | Works on small datasets                        | Requires large datasets (or transfer learning)  |
| Interpretability  | Relatively interpretable (edge orientations)   | Black box (later layers are hard to interpret)  |
| Flexibility       | Fixed once designed; not adaptable            | Adapts to any task via fine-tuning              |
| Performance       | Good for pedestrian detection, face recognition | State-of-the-art on most image tasks            |

HOG was the dominant feature for human detection (used in the original DPM and Viola-Jones
frameworks). CNNs replaced hand-crafted features as the default starting point after 2012.

</details>

---

**Q7.** Your training set has 10,000 images of class A and only 100 images of class B. What problems might this cause, and how would you address them?

<details>
<summary>Answer</summary>

**Problems from class imbalance:**

1. **Biased predictions:** The model learns to mostly predict class A (the majority class),
   achieving ~99% accuracy by rarely predicting B.
2. **Poor recall for minority class:** True positives for class B are rarely detected.
3. **Misleading evaluation:** Accuracy inflates the apparent performance.

**Solutions:**

1. **Class-weighted loss:** Assign a higher weight to class B in the loss function:
   `weight_B = n_total / (n_classes * n_B)`. Penalizes misclassifying class B more.

2. **Oversampling:** Duplicate or synthesize class B samples (SMOTE for tabular data;
   augmentation for images).

3. **Undersampling:** Randomly remove class A samples to balance the dataset (wastes data).

4. **Collect more data:** The most reliable solution — gather more class B images.

5. **Better metrics:** Evaluate with F1 score, AUC-ROC, or per-class accuracy rather
   than overall accuracy.

6. **Threshold tuning:** Adjust the classification threshold for class B to improve recall.

</details>

---

**Q8.** What was the significance of **ImageNet** and the ILSVRC competition for the field of deep learning?

<details>
<summary>Answer</summary>

**ImageNet** (Deng et al., 2009) is a large-scale image dataset with 1.2 million labeled
images across 1,000 categories, painstakingly assembled by crowdsourcing.

**The 2012 ILSVRC moment:**
When AlexNet (Krizhevsky, Sutskever, and Hinton) won the ILSVRC 2012 challenge with a
top-5 error of 15.3% — versus 26.2% for the next best entry — it demonstrated that
deep convolutional neural networks trained with GPUs dramatically outperformed all
hand-crafted approaches. This is widely considered the beginning of the modern deep
learning era in computer vision.

**Significance:**

1. **Benchmark:** Provided a standardized benchmark that enabled fair comparison of methods.
2. **Transfer learning:** Pre-trained ImageNet models became universal feature extractors —
   fine-tuning them on small datasets works remarkably well.
3. **GPU training:** Demonstrated the viability of GPU-accelerated deep learning at scale.
4. **Architecture innovation:** Drove rapid development of AlexNet → VGG → GoogLeNet →
   ResNet → EfficientNet → ViT.
5. **Dataset availability:** Made large-scale labeled data accessible to the research community.

</details>

---

**Q9.** Why is **transfer learning** so effective for image classification, especially with small datasets?

<details>
<summary>Answer</summary>

**Transfer learning** uses a model pre-trained on a large dataset (e.g., ImageNet) as a
starting point for a new task, rather than training from scratch.

**Why it works:**

1. **Universal visual features:** Early CNN layers learn edge detectors, color blobs, and
   texture detectors that are useful for virtually any image classification task.
   These features transfer well even across very different domains.

2. **Data efficiency:** Instead of needing millions of labeled examples to learn basic
   vision features, you can fine-tune with hundreds or thousands of target-domain images.

3. **Better initialization:** Starting from a well-trained model's weights avoids the
   "cold start" problem of random initialization, leading to faster convergence and
   often better final performance.

4. **Reduced compute:** Fine-tuning requires far fewer gradient updates than training
   from scratch.

**Common strategies:**
- **Feature extraction:** Freeze all pre-trained layers, only train the final classifier head.
- **Full fine-tuning:** Unfreeze all layers and fine-tune end-to-end with a small learning rate.
- **Gradual unfreezing:** Unfreeze layers from top to bottom progressively.

</details>

---

**Q10.** A classifier outputs probability [0.4, 0.35, 0.25] for classes [cat, dog, bird]. What is the predicted class, and what does it mean if this model is used for multi-label classification instead?

<details>
<summary>Answer</summary>

**For multi-class (softmax output):**

The predicted class is **"cat"** (highest probability, 0.4). The softmax ensures all
probabilities sum to 1. We select the argmax.

**If used for multi-label classification:**

These would be **independent sigmoid probabilities** (not softmax), where each value
represents P(label present) independently.

- cat: 0.4 → below 0.5 threshold → **not predicted**
- dog: 0.35 → below 0.5 threshold → **not predicted**
- bird: 0.25 → below 0.5 threshold → **not predicted**

Result: no labels predicted. The threshold (default 0.5) could be lowered if the
model is systematically underconfident. Crucially, in multi-label, all three labels
could be predicted simultaneously if their sigmoid outputs exceeded 0.5.

</details>

---

**Q11.** What is a **confusion matrix** and how do you interpret it for a 3-class classifier?

<details>
<summary>Answer</summary>

A **confusion matrix** is a K×K table (K = number of classes) where:
- Row i = true class i
- Column j = predicted class j
- Entry (i, j) = number of samples from class i predicted as class j

**Example (3 classes: cat, dog, bird):**

```
              Predicted
              Cat  Dog  Bird
Actual  Cat  [ 45    3    2 ]   ← 50 cats: 45 correct, 3 misclassified as dog, 2 as bird
        Dog  [  4   41    5 ]   ← 50 dogs: 41 correct, 4 as cat, 5 as bird
        Bird [  1    2   47 ]   ← 50 birds: 47 correct, 1 as cat, 2 as dog
```

**Interpretation:**
- **Diagonal** (45, 41, 47): Correct predictions (true positives for each class)
- **Off-diagonal** entries reveal which classes are confused with which others
- This model most often confuses dogs with birds (5 cases)

**Derived metrics per class:**
- Precision for cat = 45 / (45+4+1) = 45/50 = 0.90
- Recall for cat = 45 / (45+3+2) = 45/50 = 0.90

</details>

---

**Q12.** What is **BatchNorm** and why is it important for training deep CNNs?

<details>
<summary>Answer</summary>

**Batch Normalization (BatchNorm)** normalizes the activations of each layer across
the mini-batch to have zero mean and unit variance, then re-scales with learned
parameters γ and β:

```
x_norm = (x - mean_batch) / sqrt(var_batch + ε)
output  = γ * x_norm + β
```

**Why it is important:**

1. **Addresses internal covariate shift:** As weights update during training, the
   distribution of each layer's inputs shifts, requiring later layers to constantly
   re-adapt. BatchNorm stabilizes these distributions.

2. **Enables higher learning rates:** Without normalization, large learning rates
   cause gradients to explode or vanish. BatchNorm allows faster training.

3. **Acts as regularizer:** The noise from batch statistics has a mild regularizing
   effect, sometimes reducing the need for dropout.

4. **Enables very deep networks:** ResNet-style architectures would not train reliably
   to 50–100+ layers without BatchNorm.

**During inference:** BatchNorm uses running statistics (exponential moving averages
of batch mean and variance) accumulated during training, not the current batch.

</details>

---

**Q13.** Your model achieves 99% accuracy on the test set, but when you inspect the confusion matrix, you notice it never predicts class B (a rare defect class). What went wrong and how do you fix it?

<details>
<summary>Answer</summary>

**What went wrong:**

This is the classic **class imbalance trap**. If class B appears in only 1% of examples,
a model that always predicts "no defect" achieves 99% accuracy while being completely
useless for the actual task (detecting defects).

The model has learned to predict the majority class almost exclusively. The high accuracy
is entirely driven by correctly classifying the 99% majority.

**Evidence in the confusion matrix:**
- Row for class B: almost all entries in the "not B" column (false negatives)
- Column for class B: very few or zero entries (the model never predicts B)
- Class B recall ≈ 0%

**Fixes:**

1. **Use appropriate metrics:** F1 score, precision-recall curve, or AUC-PR for the
   defect class. Never report just accuracy on imbalanced data.

2. **Class-weighted loss:** `nn.CrossEntropyLoss(weight=torch.tensor([1.0, 99.0]))`
   penalizes missing class B 99x more than class A.

3. **Resampling:** Oversample class B (repeat defect images) or undersample class A.

4. **Threshold tuning:** Lower the decision threshold for class B — accept more false
   positives to increase recall (often preferred in safety-critical applications).

5. **Collect more defect samples:** The fundamental fix — gather more labeled defect images.

</details>
