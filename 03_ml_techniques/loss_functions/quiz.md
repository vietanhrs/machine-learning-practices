# Quiz: Loss Functions

Test your understanding of loss functions, their properties, and when to use each.

---

**Q1.** Why does MSE penalize large errors more than small errors? Is this always desirable?

<details>
<summary>Answer</summary>

**Why MSE penalizes large errors more:**

MSE computes the squared difference: `(y − ŷ)²`. The squaring operation means that as the error doubles, the loss quadruples. Mathematically:

- Error = 1 → Loss contribution = 1
- Error = 2 → Loss contribution = 4 (4×, not 2×)
- Error = 10 → Loss contribution = 100 (100×, not 10×)

The gradient of MSE with respect to `ŷ` is `2(ŷ − y)`, which grows linearly with the error. So the optimizer receives a stronger signal when errors are large — it is "more desperate" to reduce large mistakes.

**Is this always desirable?**

No. MSE is problematic when:

1. **Outliers are present**: A single mislabeled example or anomalous data point with a large error can dominate the total MSE and pull the model toward fitting that outlier at the expense of all other points.

2. **Heavy-tailed noise**: If the data-generating noise is not Gaussian (e.g., Laplacian or Cauchy), MSE is not the optimal loss.

**Alternative**: MAE treats all errors equally (linear penalty), making it robust to outliers. Huber loss provides a middle ground — quadratic for small errors, linear for large errors.

</details>

---

**Q2.** Why does binary cross-entropy work better than MSE for binary classification tasks?

<details>
<summary>Answer</summary>

**Reason 1 — Vanishing Gradients with MSE:**

For classification with a sigmoid output:
- MSE gradient w.r.t. the logit: `∂MSE/∂z = (ŷ − y) × ŷ × (1 − ŷ)`
- When the model is very wrong (e.g., predicts ŷ ≈ 0.99 for y = 0), `ŷ × (1 − ŷ) ≈ 0.0099` — the sigmoid saturates and the gradient nearly vanishes.
- The model learns very slowly even for grossly incorrect predictions.

With cross-entropy:
- BCE gradient w.r.t. the logit: `∂BCE/∂z = ŷ − y`
- When the model predicts ŷ = 0.99 for y = 0, the gradient is `0.99 − 0 = 0.99` — strong signal!
- No gradient vanishing regardless of sigmoid saturation.

**Reason 2 — Probabilistic Alignment:**

Binary cross-entropy is the negative log-likelihood under a Bernoulli distribution. Minimizing BCE is equivalent to maximum likelihood estimation of the true class probabilities. The loss is theoretically grounded for probabilistic classification.

MSE for classification has no probabilistic justification — it optimizes for squared distance from {0, 1}, not for calibrated probabilities.

**Reason 3 — Convexity:**

BCE with a logistic (sigmoid) output is convex in the logits. MSE with sigmoid is non-convex — it has more undesirable local minima.

</details>

---

**Q3.** What does the `delta` (δ) parameter in Huber loss control? What happens at the extremes (δ → 0 and δ → ∞)?

<details>
<summary>Answer</summary>

**δ controls the threshold between quadratic and linear behavior:**

```
L_δ(e) = {
    0.5 × e²             if |e| ≤ δ    (quadratic, smooth)
    δ × (|e| − 0.5δ)    if |e| > δ    (linear, robust)
}
```

- For |e| ≤ δ: loss is quadratic (like MSE) — smooth gradients, sensitive to errors
- For |e| > δ: loss is linear (like MAE) — constant gradient magnitude, robust to outliers

**As δ → 0 (very small):**
- Almost no errors fall in the quadratic region.
- The loss approaches MAE (pure linear penalty everywhere).
- Maximally robust to outliers.
- Gradient is nearly constant (±δ) for all errors.

**As δ → ∞ (very large):**
- Almost all errors fall in the quadratic region.
- The loss approaches MSE (pure quadratic penalty).
- Maximally sensitive to large errors (outliers dominate).

**Practical choice:**
- Set δ to the typical magnitude of expected residuals.
- If residuals are mostly in [−1, 1], δ = 1 works well.
- In object detection (bounding box regression), δ = 1 is the default ("Smooth L1").

</details>

---

**Q4.** When should you use focal loss? What problem does it solve that standard cross-entropy cannot?

<details>
<summary>Answer</summary>

**Problem: Extreme Class Imbalance**

Standard cross-entropy treats all examples equally. In severely imbalanced datasets (e.g., object detection where there are ~1000 background patches per object), the vast majority of training examples are easy negatives — the model quickly learns to classify them correctly with high confidence.

These easy examples dominate the total loss:
- 1000 easy negatives × small loss each = large cumulative loss
- 1 hard positive × large loss = small fraction of total

Result: The model is "distracted" by easy examples and never meaningfully learns to handle hard, informative examples.

**Focal Loss Solution:**

```
FL(p_t) = −α_t × (1 − p_t)^γ × log(p_t)
```

The modulating factor `(1 − p_t)^γ`:
- For easy examples (p_t → 1): `(1 − 0.99)^2 = 0.0001` — loss contribution near zero
- For hard examples (p_t → 0.5): `(1 − 0.5)^2 = 0.25` — loss contribution preserved

With `γ = 2`, easy examples receive 100× less weight than hard ones. The model automatically focuses training on the examples it finds most difficult.

**Use cases:**
- One-stage object detection (RetinaNet — the paper that introduced focal loss)
- Any severely class-imbalanced binary/multi-class classification
- Hard negative mining alternative (focal loss is continuous, not discrete)

</details>

---

**Q5.** Why must the loss function be differentiable? What happens if it is not, and how is this handled in practice?

<details>
<summary>Answer</summary>

**Why differentiability is required:**

Gradient descent requires computing `∂L/∂θ` — the gradient of the loss with respect to model parameters. This gradient is computed by backpropagation, which applies the chain rule through every operation in the computation graph. If the loss function is not differentiable, the chain rule breaks and no gradient can be computed.

**What happens with non-differentiable losses:**

- **Zero-gradient problem**: Losses like accuracy (a step function) have zero gradient almost everywhere. The optimizer sees a flat landscape and has no direction to move.
- **Undefined gradient**: At points of non-differentiability (like MAE at 0, hinge loss at the margin), the gradient is undefined.

**How non-differentiability is handled:**

1. **Subgradients**: For convex but non-smooth losses (MAE, hinge), a subgradient (a valid generalization of the gradient) is used. MAE subgradient at e=0 is often set to 0.

2. **Smooth approximations**: Replace non-differentiable metrics with smooth surrogates:
   - Accuracy → Cross-entropy
   - AUC → Approximate AUC losses
   - BLEU score → Cross-entropy on sequence tokens

3. **Straight-through estimator**: For discrete operations (like quantization), approximate the gradient through discrete steps as if they were identity functions.

4. **REINFORCE / policy gradient**: For non-differentiable reward in RL, use Monte Carlo gradient estimation.

</details>

---

**Q6.** Explain the mathematical relationship between maximum likelihood estimation (MLE) and cross-entropy minimization.

<details>
<summary>Answer</summary>

**They are equivalent — minimizing cross-entropy IS maximum likelihood estimation.**

**Setup:** We have a model `p(y|x; θ)` — the probability the model assigns to class `y` given input `x` with parameters `θ`.

**MLE objective:** Maximize the likelihood of the observed training data:
```
θ* = argmax Π P(yᵢ | xᵢ; θ)
   = argmax Σ log P(yᵢ | xᵢ; θ)    (log-likelihood, easier to work with)
```

**Cross-entropy:** The average negative log-probability assigned to the true class:
```
CE = −(1/N) Σ log P(yᵢ | xᵢ; θ)
```

**Connection:** Minimizing cross-entropy = maximizing the log-likelihood = MLE.

**For binary classification with sigmoid:**
- Model: `P(y=1|x; θ) = σ(θᵀx) = ŷ`
- Log-likelihood: `Σ [yᵢ log ŷᵢ + (1−yᵢ) log(1−ŷᵢ)]`
- Negative log-likelihood = Binary Cross-Entropy

**Implication:** When we train with cross-entropy, we are finding the maximum likelihood estimator of the parameters under the assumed probabilistic model. This gives cross-entropy a principled statistical interpretation — it is not just a convenient loss, it is the optimal loss under the Bernoulli/Categorical distributional assumption.

</details>

---

**Q7.** Explain the "margin" concept in hinge loss. Why does SVM produce sparse solutions (only support vectors matter)?

<details>
<summary>Answer</summary>

**Hinge Loss:**
```
L = max(0, 1 − y · ŷ)    where y ∈ {-1, +1}
```

**The Margin:**
`y · ŷ` is the "functional margin" — how far the prediction is on the correct side of the decision boundary.

- `y · ŷ > 1`: Correctly classified AND confident (margin > 1) → **zero loss**
- `y · ŷ = 0`: Exactly on the decision boundary → **loss = 1**
- `y · ŷ < 0`: Misclassified → **positive loss**

The margin of 1 means the model must not just predict the right class — it must predict it with sufficient confidence. Any correct, confident prediction is given zero loss.

**Why sparse solutions (support vectors)?**

The gradient of hinge loss is:
- `∂L/∂ŷ = 0` when `y · ŷ ≥ 1` (correctly classified with margin ≥ 1)
- `∂L/∂ŷ = −y` when `y · ŷ < 1`

Only examples with `y · ŷ < 1` (the **support vectors** — those near or on the wrong side of the decision boundary) contribute non-zero gradients. All confidently correct examples contribute zero gradient.

After training, the SVM decision boundary is defined only by the support vectors — examples that are closest to the boundary. This is the key geometric insight of SVMs: the margin is maximized by the support vectors, and all other training examples are irrelevant to the solution.

**Contrast with cross-entropy:** All examples always contribute non-zero gradient (log is never exactly zero for finite values), so all examples influence the decision boundary.

</details>

---

**Q8.** You are training a regression model and notice the training loss oscillates wildly despite using a small learning rate. MSE is the loss. What might be happening, and what alternative loss could help?

<details>
<summary>Answer</summary>

**Most Likely Cause: Outliers in the Training Data**

MSE squares residuals, so a single outlier with error = 10 contributes 100 to the loss, while 100 normal examples with error = 1 each contribute only 100 total. As mini-batches are sampled:

- Batches containing the outlier: gradient is dominated by the large squared error → large update
- Batches without the outlier: normal gradient → small update
- Result: wildly oscillating loss as the model alternates between fitting the outlier and fitting the normal data

**Additional causes:**
- Learning rate still too high (even if set low, MSE gradients are proportional to error magnitude)
- NaN/Inf values in the data
- Non-normalized target values (target range of 10,000 vs feature range of 0–1)

**Alternative Losses:**

1. **MAE** (`|y − ŷ|`): Gradient magnitude is constant (±1/N) regardless of error magnitude. A single outlier with large error contributes the same gradient as a normal example. More stable but less smooth.

2. **Huber Loss**: Quadratic near zero (smooth gradients), linear for large errors (outlier-robust). Best of both worlds. Recommended when you suspect outliers.

3. **Log-Cosh Loss**: `log(cosh(y − ŷ))` — approximately `0.5 × e²` for small e and `|e|` for large e. Smooth everywhere (unlike MAE) and outlier-robust.

**Practical steps:**
- First, inspect the data for outliers and NaN values
- Normalize targets to a reasonable range
- Switch to Huber loss with δ set to the expected residual scale

</details>

---

**Q9.** In multi-class classification with 5 classes, the model outputs softmax probabilities [0.1, 0.2, 0.05, 0.6, 0.05] for a sample with true class 3 (0-indexed). What is the categorical cross-entropy loss for this sample?

<details>
<summary>Answer</summary>

**Categorical Cross-Entropy for one sample:**
```
CCE = −Σₖ yₖ log(ŷₖ)
```

Since `y` is one-hot for class 3: `y = [0, 0, 0, 1, 0]`

Only the term for `k = 3` is non-zero:
```
CCE = −1 × log(ŷ₃) = −log(0.6)
```

```
CCE = −log(0.6) ≈ −(−0.5108) ≈ 0.5108
```

**Interpretation:**
- A perfect prediction (prob=1.0 for the true class) gives CCE = −log(1.0) = 0.
- This prediction (prob=0.6) gives CCE ≈ 0.51 — reasonable.
- A very wrong prediction (prob=0.01) would give CCE = −log(0.01) ≈ 4.6 — large loss.

**Key insight:** CCE only cares about the probability assigned to the **true class**. The distribution of probability among incorrect classes doesn't affect the loss (though softmax ensures they sum to 1 and thus influence each other implicitly).

</details>

---

**Q10.** Why is it incorrect to use accuracy as a training loss for a neural network, even though it is the most interpretable metric?

<details>
<summary>Answer</summary>

**Accuracy is not differentiable:**

Accuracy is defined as:
```
Accuracy = (1/N) Σᵢ 𝟙[argmax(ŷᵢ) = yᵢ]
```

The indicator function `𝟙[·]` is a step function — it is 0 or 1 with zero gradient almost everywhere (undefined at the discontinuity). Backpropagation cannot propagate through it.

**Consequence:** If the optimizer computes `∂Accuracy/∂θ = 0` for almost all parameter values, gradient descent has no direction to move in. No learning occurs.

**Piecewise constant:** A small change in model weights may change many individual predictions simultaneously or none at all — the gradient is either 0 or a discrete jump, not a smooth signal.

**What cross-entropy does instead:**

Cross-entropy is a smooth, differentiable surrogate for accuracy. Crucially:
- Minimizing cross-entropy encourages the model to assign high probability to the correct class.
- A model that assigns high probability to the correct class will also achieve high accuracy.
- The cross-entropy gradient provides useful information even when predictions are already correct (but not confident enough) — it encourages calibration.

**Lesson:** In ML, we often need to optimize a **differentiable proxy** of the true metric of interest. Choosing this proxy carefully is important: a poor proxy (like MSE for classification) may optimize the wrong thing.

</details>

---

**Q11.** How does contrastive loss work in a siamese network? What does each term in the loss contribute?

<details>
<summary>Answer</summary>

**Siamese Network:**
Two identical networks (shared weights) receive two inputs `x₁` and `x₂`, producing embeddings `f(x₁)` and `f(x₂)`. The goal is to learn an embedding space where:
- Similar pairs (y=1) have close embeddings.
- Dissimilar pairs (y=0) have distant embeddings.

**Contrastive Loss:**
```
L = y × d² + (1 − y) × max(margin − d, 0)²
```

Where `d = ||f(x₁) − f(x₂)||₂` is the Euclidean distance between embeddings.

**Term 1: `y × d²`** (Similar pairs, y=1)
- When `y = 1` (same class/similar): Loss = d²
- Penalizes the model for placing similar items far apart.
- Drives the distance toward 0 for positive pairs.

**Term 2: `(1 − y) × max(margin − d, 0)²`** (Dissimilar pairs, y=0)
- When `y = 0` (different class/dissimilar): Loss = `max(margin − d, 0)²`
- If `d ≥ margin`: Loss = 0 — the pair is already far enough apart.
- If `d < margin`: Loss = `(margin − d)²` — penalizes pairs that are too close.
- The `margin` hyperparameter defines the minimum required separation.

**Effect:**
- Positive pairs are pulled together.
- Negative pairs are pushed apart only if they are closer than `margin`.
- The model learns a meaningful metric space without needing class labels during inference.

**Applications:** Face verification (FaceNet), signature verification, one-shot learning.

</details>

---

**Q12.** Explain when you would design a custom loss function vs using a standard one. What are the risks?

<details>
<summary>Answer</summary>

**When to Use a Custom Loss:**

1. **Standard losses don't align with business objectives:** Example: In financial fraud detection, a false negative (missed fraud) may cost 100× more than a false positive. A standard cross-entropy treats them equally. A custom weighted loss `L = w_fn × L_fn + w_fp × L_fp` aligns optimization with true business cost.

2. **Task structure requires special objectives:** Metric learning (triplet loss), multi-task learning (weighted combination), sequence generation (CTC loss), generative models (adversarial loss).

3. **Domain-specific constraints:** Physics-informed neural networks add loss terms enforcing physical equations. Medical AI may penalize certain error types more severely.

4. **Standard metrics can't be optimized directly:** Optimizing BLEU (translation), NDCG (ranking), or mAP (detection) requires differentiable approximations as custom losses.

**Risks of Custom Losses:**

1. **Optimization instability:** Custom losses may have poor gradient properties (e.g., vanishing gradients, discontinuities) that make training unstable.

2. **Unintended mode collapse:** The model may find unexpected optima that minimize the custom loss without achieving the intended goal.

3. **Hard to debug:** When a standard loss fails, the literature has extensive debugging advice. Custom losses have none.

4. **Sensitivity to hyperparameters:** Custom loss terms often introduce additional hyperparameters (margin, weights, scaling) that require careful tuning.

5. **Theoretical justification:** Standard losses have statistical interpretations (MLE, MAP). Custom losses may lack these guarantees.

**Best practice:** Start with standard losses, identify failure modes, then design custom losses to address specific observed deficiencies — not preemptively.

</details>

---

*End of Quiz — 12 Questions*
