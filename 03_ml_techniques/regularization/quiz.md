# Quiz: Regularization

Test your understanding of regularization techniques and their effects on model training.

---

**Q1.** What is the effect of L1 regularization on model weights, and how does it differ from L2?

<details>
<summary>Answer</summary>

**L1 Regularization (Lasso):**
- Adds `λ × ||w||₁ = λ × Σ|wᵢ|` to the loss.
- Gradient contribution: `λ × sign(wᵢ)` — constant magnitude, changes direction at zero.
- **Produces sparse weights**: Many weights become exactly zero.
- Acts as automatic feature selection — irrelevant features are zeroed out.
- Equivalent to a Laplace prior on weights.

**L2 Regularization (Ridge):**
- Adds `λ × ||w||² = λ × Σwᵢ²` to the loss.
- Gradient contribution: `2λ × wᵢ` — proportional to weight magnitude.
- **Shrinks all weights toward zero** but rarely makes them exactly zero.
- Prefers solutions with many small weights rather than few large ones.
- Equivalent to a Gaussian prior on weights.

**Key Difference:**
- L1: weights either become exactly 0 or are shifted by a constant `λ` toward 0 (coordinate descent update). The constant shift can push small weights to exactly 0.
- L2: weights are multiplied by `(1 - 2αλ)` — they shrink proportionally but never reach 0 (unless the gradient also points to 0).

**When to use each:**
- L1: When you suspect many features are irrelevant and want a sparse, interpretable model.
- L2: When you believe all features contribute and want to prevent any single feature from dominating.

</details>

---

**Q2.** Explain geometrically why L1 regularization produces sparse solutions while L2 does not.

<details>
<summary>Answer</summary>

**Geometric Interpretation:**

Both L1 and L2 regularization can be interpreted as constrained optimization:

- **L1 constraint**: `||w||₁ ≤ t` — the feasible region is a **diamond** (L1 ball) with sharp corners at the coordinate axes.
- **L2 constraint**: `||w||₂ ≤ t` — the feasible region is a **sphere** (L2 ball) with no corners.

**Why L1 is sparse:**

The unconstrained loss function has elliptical contours in parameter space. As we expand the loss contours from the unconstrained minimum toward the constraint boundary, the contour first touches the boundary.

- For the **L1 diamond**: Elliptical contours are most likely to first touch the diamond at a **corner** (where some weight = 0). The corners are the "pointy" extremes of the diamond.
- For the **L2 sphere**: The contours touch the smooth sphere surface, where in general no weight is exactly zero.

In high dimensions, an L1 ball has exponentially more corners (2^d corners in d dimensions), making it overwhelmingly likely that the optimal constrained solution occurs at a corner — i.e., a sparse solution.

**Intuition**: A circle has no special points; you can touch it anywhere. A diamond has specific corner points that attract solutions with sparse coordinates.

</details>

---

**Q3.** What is the trade-off when choosing a high dropout rate vs a low dropout rate?

<details>
<summary>Answer</summary>

**Low Dropout Rate (p close to 0):**
- Few neurons are dropped per forward pass.
- Minimal regularization — model can still memorize training data.
- Training is fast and stable.
- Suitable when the model is not significantly overfitting.

**High Dropout Rate (p close to 1, e.g., 0.7-0.9):**
- Most neurons are dropped — the network operates with very few active units.
- Strong regularization — significantly reduces overfitting.
- **Downsides:**
  - Training is very slow (model sees much less information per step).
  - Effective model capacity is severely reduced — may lead to underfitting.
  - Gradients are very noisy — convergence becomes unreliable.
  - Training requires many more epochs to achieve the same loss.

**Typical Values:**
- `p = 0.2-0.3`: Mild regularization — convolutional layers or input layers.
- `p = 0.5`: Standard for fully connected layers (original paper recommendation).
- `p > 0.7`: Rarely used; usually too aggressive.

**Practical advice:**
- Start with p=0.5 for FC layers.
- If overfitting persists, increase; if training is too slow or underfit, decrease.
- Do not apply high dropout to the last layer before output.

</details>

---

**Q4.** Batch Normalization is often described as having a "regularization effect," but its primary purpose is different. Explain both its primary benefit and the regularization mechanism.

<details>
<summary>Answer</summary>

**Primary Benefit: Training Stability and Speed**

Batch Normalization was originally proposed to reduce **internal covariate shift** — the phenomenon where the distribution of inputs to each layer changes during training as the weights in previous layers are updated. This makes training deeper networks difficult because each layer must continuously adapt to the changing input distribution.

BN fixes this by normalizing each layer's inputs to have zero mean and unit variance (using batch statistics during training). This:
1. Allows using significantly higher learning rates without divergence.
2. Reduces sensitivity to weight initialization.
3. Enables effective training of very deep networks (50+ layers).
4. Accelerates convergence — often 2-10× faster training.

**Regularization Effect:**

BN introduces a form of noise into the training process:
- Batch statistics (mean and variance) are computed from the current mini-batch, not the full dataset.
- This is an approximation — the batch statistics are noisy estimates of the true statistics.
- Each example's normalization depends on the other examples in its batch → the model cannot memorize individual examples as easily.
- This noise acts as a form of regularization, similar to dropout.

**In practice:** With BN, you often need less or no dropout — BN provides sufficient regularization for many architectures (e.g., ResNets typically use BN without dropout in convolutional blocks).

**Important**: The regularization is a side effect, not the primary purpose. At test time, BN uses running statistics (computed during training), not batch statistics — so the noise disappears at inference.

</details>

---

**Q5.** When should training stop according to early stopping? How do you set the patience hyperparameter?

<details>
<summary>Answer</summary>

**When to Stop:**

Training should stop when the **validation loss stops improving for `patience` consecutive epochs** (or epochs where improvement is less than `min_delta`).

The intuition:
- Phase 1 (epochs 1-N): Validation loss decreases — the model is learning genuine patterns.
- Phase 2 (epochs N-M): Validation loss plateaus — the model has learned most of what it can generalize.
- Phase 3 (epochs M+): Validation loss increases — the model is overfitting (memorizing training noise).

Early stopping should trigger at the start of Phase 3, with `patience` controlling how long we wait to confirm the trend before stopping.

**Setting Patience:**

The patience value depends on:
1. **Learning rate and optimizer**: With a decaying learning rate, loss improvements may be slow — increase patience.
2. **Noise in validation metrics**: Small validation sets give noisy metrics — increase patience to avoid stopping prematurely.
3. **Training speed**: For fast-training models, a shorter patience is acceptable.
4. **Task complexity**: Complex tasks may have long plateaus before improving — increase patience.

**Common values:**
- `patience = 5-10`: Small datasets, fast training.
- `patience = 10-20`: Large datasets or slow convergence.
- `patience = 50+`: Fine-tuning large pretrained models (improvement can be slow).

**Best practice:**
- Always use `restore_best_weights=True` — revert to the best validation checkpoint when stopping.
- Monitor validation loss rather than validation accuracy (loss is smoother).
- Use `min_delta` to ignore trivially small improvements.

</details>

---

**Q6.** How is dropout related to ensemble learning? Why does this connection explain its effectiveness?

<details>
<summary>Answer</summary>

**The Ensemble Connection:**

Each forward pass through a network with dropout uses a different random subset of neurons — a different "thinned" network architecture. With N dropout-eligible neurons (each dropped with probability p), there are `2^N` possible thinned networks.

Over the course of training, each of these `2^N` thinned networks is trained on slightly different data (mini-batches) with slightly different architectures. The weights are **shared** across all these networks.

**At test time:** Using the full network with activations scaled by `(1-p)` is an approximation to taking the **geometric mean** of all `2^N` thinned networks' predictions. This is a form of model averaging (ensemble).

**Why ensembles work:** Different models trained independently tend to make different mistakes. When their predictions are averaged, errors cancel out and the correct signal reinforces. The ensemble has lower variance than any individual model.

**Why this explains dropout's effectiveness:**
1. **Reduced correlation**: Different thinned networks see different input patterns and make different predictions → less correlated errors → ensemble averaging is more effective.
2. **Prevents co-adaptation**: Neurons cannot rely on specific other neurons always being present → each learns more robust, independent features.
3. **Implicit regularization**: The model learns representations that work across many network configurations → inherently more generalizable.

**Caveats:** The approximation of test-time averaging is not exact (it's a geometric mean approximation, not the true ensemble mean). Monte Carlo Dropout (keeping dropout at test time and averaging over multiple passes) provides a more accurate ensemble estimate.

</details>

---

**Q7.** In Elastic Net regularization, what does the `alpha` parameter control? When would you set alpha close to 1 vs close to 0?

<details>
<summary>Answer</summary>

**Elastic Net Formula:**
```
L_elastic = L(θ) + λ × [alpha × ||w||₁ + (1 - alpha) × ||w||²]
```

**The `alpha` parameter controls the L1/L2 mix:**

- `alpha = 1.0`: Pure L1 (Lasso) — maximum sparsity
- `alpha = 0.0`: Pure L2 (Ridge) — no sparsity, maximum shrinkage
- `alpha = 0.5`: Equal blend — some sparsity + shrinkage

**When to set alpha close to 1 (more L1):**
- Many features are truly irrelevant — you want sparse feature selection.
- Features are mostly **uncorrelated** — Lasso can select between them cleanly.
- Interpretability is important — a sparse model is easier to explain.
- High-dimensional data (n_features >> n_samples).

**When to set alpha close to 0 (more L2):**
- Features are **highly correlated** — Lasso arbitrarily picks one correlated feature while zeroing others; Elastic Net (with more L2) includes all with reduced coefficients.
- You believe most features are relevant — you don't want aggressive zeroing.
- You need a smooth, stable optimization landscape.
- When pure Lasso gives unstable feature selection (changing slightly with data).

**Practical note:** Elastic Net was specifically designed to handle the "grouped selection" problem: when features are correlated, Lasso tends to pick one randomly from each group. Elastic Net with appropriate alpha selects the entire group.

</details>

---

**Q8.** What is the difference between weight decay and L2 regularization in Adam? Why does AdamW exist?

<details>
<summary>Answer</summary>

**For SGD: Equivalent**

With SGD, adding L2 to the loss is mathematically identical to weight decay:
- L2 in loss: gradient includes `+2λw`, so update: `w ← w - α(∇L + 2λw) = w(1 - 2αλ) - α∇L`
- Weight decay: `w ← w(1 - λ_wd) - α∇L`

Setting `λ_wd = 2αλ` makes them identical.

**For Adam: NOT Equivalent**

Adam normalizes each gradient component by its historical RMS value `√v̂`. When L2 is added to the loss:
- The regularization gradient `2λwᵢ` is also normalized: effective regularization ≈ `2λwᵢ / √v̂ᵢ`
- Parameters with large gradient history (`v̂ᵢ` large) get **less** regularization.
- Parameters with small gradient history (`v̂ᵢ` small) get **more** regularization.
- This is the **wrong** behavior — the intended uniform L2 penalty is distorted.

Frequently updated embedding weights (large `v̂`) receive almost no regularization. Rare parameters (small `v̂`) are over-regularized.

**AdamW Fix (Loshchilov & Hutter, 2019):**

AdamW applies weight decay directly to parameters, **bypassing** the adaptive scaling:
```
θ ← θ - α × m̂/(√v̂ + ε) - λ_wd × θ
```

The `λ_wd × θ` term is applied uniformly — every parameter gets the same proportional shrinkage regardless of its gradient history. This is the correct decoupled L2 regularization.

**Practical advice:** Always use AdamW for Transformers. In PyTorch: `torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)`.

</details>

---

**Q9.** A model trains well on the training set but generalizes poorly. List 5 regularization strategies you would try, and explain which you would try first and why.

<details>
<summary>Answer</summary>

**5 Regularization Strategies (in recommended order):**

**1. Early Stopping (try first)**
- Easiest to implement and costs nothing.
- Add a validation set, monitor validation loss, stop when it increases.
- No new hyperparameters to tune (just patience).
- Often reveals how much overfitting is occurring, guiding further decisions.

**2. Data Augmentation (try second)**
- One of the most effective techniques, especially for images and sequences.
- Effectively increases dataset size without collecting new data.
- Encodes domain inductive biases (translation invariance, etc.).
- No change to model architecture.

**3. Dropout (or increase existing dropout rate)**
- If using a neural network, add/increase dropout in fully connected layers.
- Start with p=0.3-0.5 for FC layers.
- Requires adjusting learning rate (dropout slows convergence).
- Very effective for large networks.

**4. L2 Regularization / Weight Decay**
- Add `weight_decay` to optimizer (use AdamW if using Adam).
- Start with values like 1e-4, 1e-3, 1e-2 and tune.
- Works for any parameterized model — linear, trees (implicitly), neural networks.
- Easy to tune with grid search over a log scale.

**5. Reduce Model Complexity**
- Fewer layers, smaller hidden dimensions, shallower trees.
- If data is limited, a smaller model may generalize better.
- Last resort — reducing capacity limits the model's ability to learn complex patterns.

**Why early stopping first:** It is free (no new hyperparameters), immediately informative (shows the severity of overfitting), and acts as a safety net while experimenting with other techniques. The others build on top of it.

</details>

---

**Q10.** Explain the concept of "internal covariate shift" and how Batch Normalization addresses it.

<details>
<summary>Answer</summary>

**Internal Covariate Shift:**

During neural network training, each layer's parameters change at every gradient step. As the weights of layer `k` change, the distribution of inputs to layer `k+1` also changes — even if the raw input distribution is fixed. This is called **internal covariate shift**.

Consequences:
1. Each layer must continuously adapt to changing input statistics → slower learning.
2. Activations can become saturated (too large → vanishing gradients) or die (too small → useless neurons).
3. Requires careful weight initialization and small learning rates.
4. Deeper networks suffer more — the effect compounds across layers.

**How Batch Normalization Addresses It:**

BN normalizes the inputs to each layer (or sublayer) to have approximately zero mean and unit variance, based on the current mini-batch statistics:

```
μ_B = mean(x_B)           (mini-batch mean)
σ²_B = var(x_B)           (mini-batch variance)
x̂ = (x - μ_B) / √(σ²_B + ε)    (normalize)
y = γ × x̂ + β             (scale and shift with learnable γ, β)
```

Regardless of how upstream weights change, the input distribution to the next layer is now approximately normalized. This:
1. Breaks the coupling between layers — each layer can optimize independently.
2. Allows higher learning rates without divergence.
3. Makes the optimization landscape smoother.
4. Reduces the importance of careful weight initialization.

**Note:** More recent research (e.g., Santurkar et al., 2018 "How Does Batch Normalization Help Optimization?") suggests BN's primary benefit may be **smoothing the optimization landscape** rather than strictly reducing covariate shift. The debate continues, but the empirical benefits are well-established.

</details>

---

**Q11.** What is the difference between explicit regularization (L1, L2, dropout) and implicit regularization? Give examples of implicit regularization.

<details>
<summary>Answer</summary>

**Explicit Regularization:**
Deliberate modifications to the loss function or network architecture designed specifically to reduce overfitting:
- L1/L2 penalty terms added to the loss.
- Dropout layers added to the architecture.
- Early stopping based on validation loss.

**Implicit Regularization:**
Regularization that emerges as a side effect of the optimization process or data handling — not an intentional design choice:

1. **Mini-batch SGD**: The noise in mini-batch gradient estimates acts as regularization. Models trained with mini-batch SGD find flatter minima than models trained with full-batch gradient descent, improving generalization.

2. **Learning rate schedule**: High initial learning rates can prevent the model from settling into sharp minima (which tend to overfit). Cosine annealing's high LR phases provide implicit regularization.

3. **Data augmentation**: Creates variation in training data, making it harder to memorize — implicit regularization without an explicit penalty term.

4. **Gradient clipping**: Limits the magnitude of updates, preventing extreme parameter values.

5. **Weight sharing**: CNNs share weights across spatial positions — a strong architectural prior that reduces effective model capacity.

6. **Architecture design**: Choosing certain activation functions (ReLU sparsity), pooling layers (information bottleneck), or skip connections can implicitly regularize.

7. **Batch normalization**: Its noise (batch statistics vs. true statistics) acts as implicit regularization.

**Key insight**: Implicit regularization is often more powerful than explicit methods because it is tailored to the specific optimization dynamics rather than being a fixed penalty term.

</details>

---

**Q12.** Your model achieves 99% training accuracy but 65% validation accuracy. How would you diagnose whether to use more data, more regularization, or a simpler model?

<details>
<summary>Answer</summary>

**Diagnosis Framework:**

The 34% gap between training (99%) and validation (65%) clearly indicates severe overfitting. The question is how to fix it most efficiently.

**Step 1: Plot Learning Curves (most informative)**

Vary training set size from 10% to 100%, record both training and validation accuracy:

- **Both curves plateau at similar (low) values**: High bias (underfitting) — model too simple. Need more complexity.
- **Large gap between train and val, val still improving**: Need more training data — the model can generalize but doesn't have enough data.
- **Large gap, val plateau is low and stable**: Need regularization — the model is memorizing rather than generalizing.

**Step 2: Check Data Quality**
- Are validation samples from the same distribution as training?
- Are there label errors in training data (leading to memorization of noise)?

**Step 3: Try Regularization First (cheapest)**

Before collecting more data or redesigning the model:
1. Add/increase dropout, add L2/weight decay.
2. Add early stopping.
3. Use data augmentation.

If regularization closes the gap to < 5%, the model capacity is appropriate.

**Step 4: Evaluate if More Data Helps**

If the learning curve shows val accuracy still rising at 100% of training data, more data will help. Collect more labeled examples or use semi-supervised/self-supervised learning.

**Step 5: Simplify the Model (last resort)**

If regularization doesn't help and more data is not available, reduce model capacity. But this often limits performance on complex tasks.

**Decision rule**: If training accuracy >> validation accuracy AND both are not limited by model capacity (i.e., model can fit complex patterns), regularization and/or more data is the answer. If even training accuracy is insufficient, the model needs more capacity.

</details>

---

**Q13.** How does batch normalization behave differently during training vs inference? What bug occurs if you forget to switch modes?

<details>
<summary>Answer</summary>

**During Training:**
BN uses **mini-batch statistics** (mean and variance computed from the current training batch):
- `μ_B = mean(x_batch)`, `σ²_B = var(x_batch)`
- Simultaneously updates **running statistics**: exponential moving averages of `μ` and `σ²` across all batches.

**During Inference:**
BN uses **running statistics** (accumulated during training):
- Fixed `μ_running` and `σ²_running` — not computed from the test batch.
- This ensures deterministic, stable predictions on single examples.

**Why the difference?**
- At training time, batch statistics work well because batches are reasonably large (32-512 examples).
- At inference time, the batch may contain just 1 example (real-time prediction) — batch statistics from a single sample are meaningless.
- Running statistics represent the true training data distribution, which is what we want for normalization.

**The Bug — Forgetting `model.eval()`:**

In PyTorch:
```python
model.train()  # BN uses batch stats (for training)
model.eval()   # BN uses running stats (for inference)
```

If you forget `model.eval()` during inference:
1. BN normalizes by batch statistics of your test batch instead of training running stats.
2. Predictions for different test batches will have different normalizations — predictions are **inconsistent and wrong**.
3. If evaluating batch_size=1: variance in denominator is 0 (or undefined) → NaN predictions.
4. Even with larger test batches, test distribution ≠ training distribution → wrong normalization → accuracy drops.

This is a very common, hard-to-debug mistake. Always call `model.eval()` before any prediction/evaluation.

</details>

---

**Q14.** What is the relationship between data augmentation and regularization? Does augmentation always help?

<details>
<summary>Answer</summary>

**Augmentation as Implicit Regularization:**

Data augmentation increases the apparent size and diversity of the training set by applying label-preserving transformations (flips, crops, noise) to existing examples. This:

1. **Prevents memorization**: The model cannot memorize specific pixel patterns — augmented versions vary. It must learn the underlying structure.
2. **Encodes invariances**: If you augment with horizontal flips, the model learns that horizontally flipped images have the same label → translation/flip invariance built in.
3. **Reduces variance**: Increased effective dataset size reduces the variance of the learned model.
4. **No explicit penalty**: Unlike L2 or dropout, augmentation doesn't change the loss function — it changes the training data distribution.

**Does augmentation always help?**

No — augmentation can hurt when:

1. **The augmentation breaks the task semantics**: Flipping a "6" produces a "9" in digit recognition — horizontal flip augmentation would confuse the model.
2. **The augmentation doesn't match the test distribution**: If test images are always upright but training uses heavy rotation augmentation, the model may generalize worse.
3. **The augmentation is too aggressive**: Very strong augmentation (high crop ratios, extreme color jitter) can destroy the signal in the data, making the task harder than it needs to be.
4. **The dataset is already large enough**: With millions of examples, augmentation may provide diminishing returns and only slow training.
5. **Wrong domain**: Text or tabular data augmentation is much harder to do correctly than image augmentation.

**Practical advice:** Always validate augmentation choices by comparing validation accuracy with and without augmentation. What works for natural images (flip, crop) may not work for medical images (where orientation matters) or satellite imagery (where scale matters differently).

</details>

---

*End of Quiz — 14 Questions*
