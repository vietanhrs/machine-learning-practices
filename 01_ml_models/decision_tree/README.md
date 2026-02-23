# Decision Trees

A decision tree is a **tree-structured model** that makes predictions by recursively partitioning the feature space into regions, each associated with a predicted class (classification) or value (regression).

---

## 1. What is a Decision Tree?

Each **internal node** tests a feature against a threshold (e.g., "age < 35?"). Each **branch** represents the outcome of that test. Each **leaf** holds a prediction (the majority class in its region for classification).

### Advantages

- Highly interpretable — you can follow the decision path.
- Requires little data preprocessing (no scaling needed).
- Handles both numerical and categorical features.
- Implicitly performs feature selection.

### Disadvantages

- Prone to overfitting (especially deep trees).
- Unstable — small changes in data can cause a completely different tree.
- Biased toward features with more levels.
- Not great for linear relationships.

---

## 2. Splitting Criteria

At each node, the algorithm searches for the feature and threshold that best separates the classes.

### Gini Impurity

Measures the probability of incorrectly classifying a randomly chosen element:

```
Gini(S) = 1 - Σ_c pₒ²
```

where `pₒ` is the proportion of class `c` in set `S`.

- Range: `[0, 0.5]` for binary classification.
- `Gini = 0` → pure node (all same class).
- `Gini = 0.5` → maximally impure (50/50 split).

### Entropy and Information Gain

Entropy measures the average uncertainty in a set:

```
H(S) = -Σ_c pₒ · log₂(pₒ)
```

Information Gain from splitting S on feature f at threshold t:

```
IG(S, f, t) = H(S) - [|S_left|/|S| · H(S_left) + |S_right|/|S| · H(S_right)]
```

- Higher IG → better split.
- The split is chosen to **maximize** IG (or equivalently, minimize the weighted child impurity).

### Gini vs. Entropy

| Property | Gini | Entropy |
|----------|------|---------|
| Computation | Faster (no log) | Slightly slower |
| Behavior | Favors larger partitions | More sensitive to class balance |
| In practice | Very similar results | Very similar results |

---

## 3. Tree Growing

### Recursive Splitting

1. At the root node, evaluate all possible (feature, threshold) splits.
2. Pick the split with the highest Information Gain (or lowest Gini).
3. Partition the data into left/right subsets.
4. Recurse on each subset until a stopping criterion is met.

### Stopping Criteria

| Criterion | Description |
|-----------|-------------|
| `max_depth` | Stop when the tree reaches a maximum depth |
| `min_samples_split` | Don't split a node with fewer than n samples |
| `min_samples_leaf` | Ensure each leaf has at least n samples |
| `min_impurity_decrease` | Only split if IG >= threshold |
| Pure node | Stop if all samples in the node have the same class |

---

## 4. Overfitting and Pruning

Deep trees memorize the training data and generalize poorly.

### Pre-pruning (Early Stopping)

Prevent the tree from growing too deep by setting `max_depth`, `min_samples_split`, or `min_samples_leaf` during training.

### Post-pruning

Grow the full tree, then remove subtrees that do not improve performance on a validation set. **Cost-Complexity Pruning** (α-pruning, `ccp_alpha` in sklearn) is the most principled approach:

```
R_α(T) = R(T) + α · |leaves(T)|
```

Larger α penalizes tree size more, producing simpler trees.

---

## 5. Random Forest

Random Forest is an **ensemble** of decision trees that reduces variance through two mechanisms:

### Bagging (Bootstrap Aggregating)

- Each tree is trained on a **bootstrap sample** of the training data (random sample with replacement, same size as original).
- Trees are **diverse** because they see slightly different training sets.
- Final prediction is the **majority vote** (classification) or **mean** (regression).

### Feature Subsampling (Random Subspace Method)

- At each split, only `m` randomly selected features are considered (typically `m = √n_features`).
- Decorrelates the trees — if one feature dominates, it doesn't appear in every tree.

### Why It Reduces Variance

An ensemble of many uncorrelated, low-bias trees has a variance ≈ `σ² / n_trees` (lower than a single tree). The wisdom of crowds effect: individual trees may be wrong, but the majority vote is often correct.

---

## 6. Gradient Boosting

Unlike bagging (parallel, independent trees), **gradient boosting** builds trees **sequentially**, with each tree correcting the errors of the previous ensemble.

### Core Idea

1. Start with a simple prediction (e.g., the mean of y).
2. Compute **residuals**: the difference between true labels and current predictions.
3. Fit a shallow decision tree to predict the residuals.
4. Update predictions: `ŷ ← ŷ + η · tree_prediction` (η = learning rate).
5. Repeat for `n_estimators` rounds.

### Intuition

Each tree is a small "correction" in the direction that reduces the loss most (gradient descent in function space).

### XGBoost Overview

XGBoost extends gradient boosting with:
- **Second-order gradients** (Newton's method) for faster convergence.
- **Regularization** (L1/L2) on leaf weights.
- **Column subsampling** (like Random Forest).
- Efficient handling of **missing values**.
- Out-of-core computation for datasets that don't fit in RAM.

---

## 7. Feature Importance

Decision trees and ensembles provide a natural measure of feature importance.

### Mean Decrease in Impurity (MDI)

```
importance(f) = Σ_{nodes split on f} (impurity_decrease × n_samples_at_node)
```

Normalized over all features to sum to 1. Can be misleading for high-cardinality features.

### Permutation Importance

Randomly shuffle feature `f` and measure the drop in model performance. More reliable than MDI, model-agnostic, but expensive.

---

## 8. Reference Links

- [Scikit-learn: Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
- [Wikipedia: Decision Tree Learning](https://en.wikipedia.org/wiki/Decision_tree_learning)
