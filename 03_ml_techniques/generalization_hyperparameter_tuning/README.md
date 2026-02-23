# Generalization and Hyperparameter Tuning

A comprehensive guide to understanding model generalization, diagnosing bias-variance tradeoff, and systematically tuning hyperparameters.

---

## 1. Generalization

**Generalization** is the ability of a trained model to perform well on **new, unseen data** that was not part of the training set. It is the fundamental goal of machine learning — we do not care how well a model memorizes its training data, only how well it performs in the real world.

### Measuring Generalization

```
Generalization Gap = Test Error − Training Error
```

- **Small gap**: The model generalizes well.
- **Large gap (test >> train)**: The model overfits — it has memorized the training data but fails to generalize.
- **Both errors high**: The model underfits — it cannot even fit the training data well.

### Why Generalization is Hard

1. **Finite training data**: We can only observe a sample of the true data distribution.
2. **Model expressivity**: Complex models can fit any finite dataset perfectly (memorization).
3. **Distribution shift**: Test data may not perfectly match training data distribution.
4. **No free lunch theorem**: No single model generalizes best across all possible distributions.

---

## 2. Bias-Variance Tradeoff

### Mathematical Decomposition

For a regression task, the expected prediction error on a new point `x` can be decomposed as:

```
E[(y − f̂(x))²] = Bias(f̂(x))² + Variance(f̂(x)) + Noise
```

Where:

| Term | Definition | Meaning |
|------|-----------|---------|
| **Bias²** | `[E[f̂(x)] − f(x)]²` | Systematic error — how wrong is the average prediction? |
| **Variance** | `E[(f̂(x) − E[f̂(x)])²]` | Sensitivity to training data — how much do predictions vary? |
| **Noise** | `E[(y − f(x))²]` | Irreducible error from data noise |

### High Bias = Underfitting

- The model is **too simple** to capture the true relationship.
- Both training error and test error are high.
- Adding more training data helps very little — the model is inherently limited.
- Solution: Use a more complex model, add features, reduce regularization.

**Signs of underfitting:**
- Training loss is high and plateaus early.
- Learning curves: both train and val accuracy are below acceptable level.

### High Variance = Overfitting

- The model is **too complex** — it memorizes training data including noise.
- Training error is low but test error is high.
- Adding more training data helps significantly.
- Solution: Add regularization, reduce model complexity, get more data.

**Signs of overfitting:**
- Training loss is very low, validation loss is much higher.
- Learning curves: large gap between train and val accuracy.

### The Tradeoff

```
Model Complexity →

Simple model:   High Bias,    Low Variance  → Underfitting
Complex model:  Low Bias,     High Variance → Overfitting
Sweet spot:     Low Bias,     Low Variance  → Good generalization

Total Error = Bias² + Variance + Noise

As complexity increases:
    Bias² ↓↓ (better fit to training data)
    Variance ↑↑ (more sensitive to training set)
    Total Error is U-shaped — minimum at the optimal complexity
```

### Modern Perspective: Double Descent

For very large deep learning models (massively overparameterized), the classical U-shape may give way to a second descent of test error beyond the "interpolation threshold" — the so-called **double descent** phenomenon. Modern large models can generalize well even with more parameters than training examples, as long as training is regularized (dropout, weight decay, early stopping) and the models are trained with SGD.

---

## 3. Cross-Validation

### Why Not Just Train/Test Split?

A single train/test split has high variance — it depends heavily on which examples happen to land in the test set. With limited data, you need to use your data more efficiently.

### k-Fold Cross-Validation

Split data into `k` equal folds. Train on `k-1` folds, validate on the remaining fold. Rotate through all k folds.

```
Data:  [Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5]

Iter 1: Train=[2,3,4,5], Val=[1] → Score₁
Iter 2: Train=[1,3,4,5], Val=[2] → Score₂
Iter 3: Train=[1,2,4,5], Val=[3] → Score₃
Iter 4: Train=[1,2,3,5], Val=[4] → Score₄
Iter 5: Train=[1,2,3,4], Val=[5] → Score₅

Final score: mean(Score₁, ..., Score₅) ± std(Score₁, ..., Score₅)
```

**Benefits:**
- Every example is used for validation exactly once.
- Lower variance estimate of generalization performance.
- Standard: k=5 or k=10.

### Stratified k-Fold

Maintains class proportions in each fold — essential for imbalanced classification. If 10% of data is class A, each fold also has ~10% class A.

**Always use stratified k-fold for classification** unless data is balanced.

### Leave-One-Out CV (LOO-CV)

Special case of k-fold with k=N (one sample per fold). Each fold trains on N-1 examples, validates on 1.

- **Advantage**: Maximum use of training data.
- **Disadvantage**: Computationally expensive (N model fits), high variance for regression.
- **When useful**: Very small datasets (N < 30) where holding out a fold is too costly.

---

## 4. Hyperparameter Tuning

### What are Hyperparameters?

Hyperparameters are model settings that are **not learned** from training data but must be set before training:
- Learning rate, batch size, optimizer choice
- Network depth, width, dropout rate
- Regularization strength (λ), number of trees
- Kernel type (SVM), number of clusters (K-Means)

### Grid Search

**Exhaustively evaluate all combinations** of specified hyperparameter values.

```python
param_grid = {
    "n_estimators": [100, 200, 500],
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10]
}
# Total: 3 × 4 × 3 = 36 combinations × k folds = 36k model fits
```

**Pros**: Finds the best within the grid.
**Cons**: Exponential in the number of hyperparameters. Misses continuous optima between grid points.

### Random Search

**Sample hyperparameter combinations randomly** from specified distributions.

```python
param_distributions = {
    "n_estimators": [100, 200, ..., 1000],       # or scipy.stats.randint(100, 1000)
    "max_depth": [3, 5, ..., 20, None],
    "min_samples_split": [2, 3, ..., 20]
}
# Sample n_iter=50 random combinations
```

**Why Random Search beats Grid Search** (Bergstra & Bengio, 2012):
- When some hyperparameters matter more than others, random search assigns more effective combinations per dimension.
- Grid search wastes evaluations on low-importance dimensions (always the same value in a row/column).
- With the same compute budget, random search typically finds better configurations.

**Practical rule**: Use random search over grid search for > 2 hyperparameters.

### Bayesian Optimization

**Maintain a probabilistic model** of the objective function (typically a Gaussian Process or Tree-structured Parzen Estimator). Use it to select the most promising next hyperparameter combination.

```
1. Evaluate a few random configurations → initial observations
2. Fit a surrogate model to observations: f(hyperparams) → score
3. Use acquisition function to select the next hyperparameter set to evaluate:
   - Expected Improvement (EI)
   - Upper Confidence Bound (UCB)
4. Evaluate the selected configuration, update surrogate model
5. Repeat until budget exhausted
```

**Pros**: Sample-efficient — finds good configurations with fewer evaluations. Especially valuable when each evaluation is expensive (training a large neural network).

**Cons**: More complex to implement; overhead of surrogate model fitting. Less parallelizable.

**Libraries**: Optuna, Hyperopt, Scikit-Optimize (skopt), Ray Tune.

---

## 5. Model Selection

### The Validation Set Pitfall

If you repeatedly tune hyperparameters on the same validation set, you "overfit" to that set — the reported validation performance is optimistically biased. The model has been implicitly selected to look good on that specific set.

**Solution: Nested Cross-Validation**

```
Outer CV (5-fold): Estimate generalization performance
    └── Inner CV (3-fold): Select hyperparameters
```

```
For each outer fold (k=1..5):
    Train set (outer): 4/5 of data
    Test set (outer):  1/5 of data
    
    For each inner fold (k=1..3):
        Inner train:    2/3 of outer train
        Inner val:      1/3 of outer train
    
    Best hyperparams = argmax mean inner CV score
    Retrain with best hyperparams on full outer train
    Evaluate on outer test → record unbiased performance estimate
    
Final estimate: mean of 5 outer test scores
```

**When to use nested CV**: When you need an unbiased estimate of model performance after hyperparameter tuning. Especially important for publishing results or making high-stakes decisions.

---

## 6. Learning Curves

Learning curves show model performance (train and validation) as a function of training set size.

### Diagnosing with Learning Curves

```
High Bias (Underfitting):
    - Both train and val accuracy plateau at the same low value
    - Adding more data barely helps
    → Solution: More complex model, better features

High Variance (Overfitting):
    - Large gap between train (high) and val (low) accuracy
    - Val accuracy is still improving as data is added
    → Solution: More data, regularization, simpler model

Good Fit:
    - Small gap between train and val accuracy
    - Both improve with more data
    - Train and val curves converge
```

### Plotting Learning Curves

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    estimator, X, y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring='accuracy'
)
```

---

## 7. Reference Links

- [scikit-learn: Cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html)
- [scikit-learn: Grid Search and Hyperparameter Tuning](https://scikit-learn.org/stable/modules/grid_search.html)
