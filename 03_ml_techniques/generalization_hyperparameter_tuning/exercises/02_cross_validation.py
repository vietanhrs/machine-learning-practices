"""
Exercise 02: Cross-Validation Strategies
==========================================
Implement k-fold cross-validation from scratch and compare multiple CV
strategies in terms of performance estimate variance and computational cost.

Learning objectives:
- Implement k-fold splitting from scratch
- Understand stratified k-fold for classification
- Compare holdout, k-fold, stratified, and LOO-CV
- Observe variance in performance estimates across strategies

Dependencies: numpy, scikit-learn, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# ---------------------------------------------------------------------------
# K-Fold from Scratch
# ---------------------------------------------------------------------------

def manual_kfold(X, y, k=5, shuffle=True, random_state=42):
    """
    Implement k-fold cross-validation split from scratch (without sklearn).

    Args:
        X (np.ndarray): Feature matrix, shape (N, D)
        y (np.ndarray): Labels, shape (N,)
        k (int): Number of folds
        shuffle (bool): Whether to shuffle indices before splitting
        random_state (int): Random seed for reproducibility

    Yields:
        tuple: (train_indices, val_indices) for each fold
               - train_indices: np.ndarray of training indices
               - val_indices: np.ndarray of validation indices

    Algorithm:
        1. Create array of all indices: [0, 1, 2, ..., N-1]
        2. Shuffle indices (if shuffle=True)
        3. Split indices into k roughly equal-sized chunks (folds)
        4. For each fold i:
           - val_indices = fold i
           - train_indices = all other folds combined
           - yield (train_indices, val_indices)

    Hints:
        - np.array_split(indices, k) splits array into k chunks (handles N not divisible by k)
        - To concatenate all train folds: np.concatenate([folds[j] for j in range(k) if j != i])
        - Use a Python generator (yield) — the function is called once per fold
        - Check: each index appears exactly once in val across all k folds
        - Check: each fold has approximately N/k examples

    Example:
        for train_idx, val_idx in manual_kfold(X, y, k=5):
            model.fit(X[train_idx], y[train_idx])
            score = model.score(X[val_idx], y[val_idx])
    """
    np.random.seed(random_state)
    N = len(y)

    # TODO: Create index array [0, 1, ..., N-1]
    # TODO: Shuffle indices if shuffle=True
    # TODO: Split into k chunks using np.array_split
    # TODO: For each fold i (0 to k-1):
    #   - val_indices = folds[i]
    #   - train_indices = concatenate all other folds
    #   - yield (train_indices, val_indices)
    pass


# ---------------------------------------------------------------------------
# Stratified K-Fold Evaluation
# ---------------------------------------------------------------------------

def stratified_kfold_eval(model, X, y, k=5):
    """
    Evaluate model using Stratified k-Fold CV (from sklearn).

    Stratified means each fold has approximately the same class proportion
    as the full dataset. This is important for imbalanced classification.

    Args:
        model: sklearn-compatible estimator with fit/predict methods
        X (np.ndarray): Feature matrix
        y (np.ndarray): Labels (should be categorical)
        k (int): Number of folds

    Returns:
        dict: {
            "scores": list of k accuracy scores (one per fold),
            "mean": float mean accuracy,
            "std": float std of accuracy
        }

    Hints:
        - Use StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        - For each fold: model.fit(X[train], y[train]), predict on X[val]
        - Compute accuracy_score(y[val], preds) for each fold
        - Important: Do NOT modify the original model — each fold fits a fresh copy
        - Use sklearn.base.clone(model) to get a fresh copy, or re-use the same estimator
          (sklearn's stateless design means fit() overwrites previous state)
        - Print: "Stratified {k}-Fold: {mean:.4f} ± {std:.4f}"
    """
    # TODO: Initialize StratifiedKFold
    # TODO: Initialize scores list
    # TODO: For each (train_idx, val_idx) fold:
    #   - model.fit(X[train_idx], y[train_idx])
    #   - preds = model.predict(X[val_idx])
    #   - score = accuracy_score(y[val_idx], preds)
    #   - Append to scores
    # TODO: Return {"scores": scores, "mean": mean, "std": std}
    pass


# ---------------------------------------------------------------------------
# Leave-One-Out CV
# ---------------------------------------------------------------------------

def leave_one_out_eval(model, X, y):
    """
    Evaluate model using Leave-One-Out Cross-Validation (LOO-CV).

    Each fold trains on N-1 examples and evaluates on 1 example.
    This is the most exhaustive CV strategy — N model fits total.

    When is LOO-CV useful?
    - Very small datasets (N < 30) where holding out a fold (20%) is too costly.
    - When you need maximum training data per fold.
    - When computational cost is acceptable (model trains fast).

    Args:
        model: sklearn-compatible estimator
        X (np.ndarray): Feature matrix
        y (np.ndarray): Labels

    Returns:
        dict: {
            "scores": list of N accuracy scores (each is 0 or 1 for classification),
            "mean": float mean accuracy,
            "std": float std of accuracy
        }

    Hints:
        - Use LeaveOneOut() from sklearn.model_selection
        - Each fold: val has 1 sample → accuracy is either 0.0 or 1.0
        - Accuracy of LOO-CV = fraction of correctly classified samples
        - Print estimated time or warning if N > 500 (LOO is slow for large N)
        - Print: "LOO-CV (N={N} fits): {mean:.4f} ± {std:.4f}"
    """
    # TODO: Initialize LeaveOneOut()
    # TODO: Check if N is large and print warning
    # TODO: For each (train_idx, val_idx) fold:
    #   - model.fit(X[train_idx], y[train_idx])
    #   - pred = model.predict(X[val_idx])
    #   - score = float(pred[0] == y[val_idx[0]])  # 0 or 1
    #   - Append to scores
    # TODO: Return {"scores": scores, "mean": mean, "std": std}
    pass


# ---------------------------------------------------------------------------
# Strategy Comparison
# ---------------------------------------------------------------------------

def compare_cv_strategies(model, X, y, k_fold=5, n_repeats=10):
    """
    Compare holdout, k-fold, stratified k-fold, and LOO-CV strategies.

    For holdout, repeat n_repeats times with different random seeds to show variance.

    Args:
        model: sklearn-compatible estimator
        X (np.ndarray): Feature matrix
        y (np.ndarray): Labels
        k_fold (int): k for k-fold and stratified CV
        n_repeats (int): Number of holdout repetitions to estimate variance

    Returns:
        dict: {strategy_name: {"mean": float, "std": float, "scores": list}}

    Strategies to compare:
    1. Holdout (70/30 split) — repeated n_repeats times with different seeds
    2. Manual k-Fold (from scratch using manual_kfold)
    3. Stratified k-Fold (using stratified_kfold_eval)
    4. LOO-CV (using leave_one_out_eval — only if N <= 300 for speed)

    Hints:
        - For Holdout:
            from sklearn.model_selection import train_test_split
            scores = []
            for seed in range(n_repeats):
                X_tr, X_v, y_tr, y_v = train_test_split(X, y, test_size=0.3,
                                                          stratify=y, random_state=seed)
                Fit model, predict, compute accuracy, append
        - For manual k-fold: use your manual_kfold generator
        - Print a comparison table
        - Plot boxplots of score distributions for each strategy
    """
    results = {}

    # TODO: Holdout strategy (repeated n_repeats times)

    # TODO: Manual k-fold strategy

    # TODO: Stratified k-fold strategy

    # TODO: LOO-CV (only run if len(X) <= 300 to avoid excessive time)

    # TODO: Print comparison table:
    # Strategy        | Mean     | Std      | N Scores
    # Holdout (10x)   | 0.9523   | 0.0234   | 10
    # Manual 5-Fold   | 0.9526   | 0.0128   | 5
    # Stratified 5-F  | 0.9561   | 0.0093   | 5
    # LOO-CV          | 0.9542   | 0.2093   | N

    # TODO: Plot boxplots showing score distribution variance
    pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    Compare cross-validation strategies on the breast cancer dataset.

    Steps:
    1. Load breast cancer dataset (~570 samples, binary classification)
    2. Standardize features
    3. Run all CV strategies
    4. Compare: mean accuracy, std, and variance in estimates
    5. Observe: k-fold has lower std than single holdout
    """
    print("=" * 60)
    print("Cross-Validation Strategy Comparison")
    print("=" * 60)

    # Load data
    data = load_breast_cancer()
    X, y = data.data, data.target
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, "
          f"{len(np.unique(y))} classes")
    print(f"Class distribution: {np.bincount(y)}")

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Model
    model = LogisticRegression(max_iter=1000, random_state=42)

    # TODO: Test manual_kfold first on a small example
    print("\n--- Testing manual_kfold ---")
    # for fold_num, (train_idx, val_idx) in enumerate(manual_kfold(X, y, k=5)):
    #     print(f"Fold {fold_num+1}: train={len(train_idx)}, val={len(val_idx)}")

    # TODO: Run full comparison
    # results = compare_cv_strategies(model, X, y, k_fold=5, n_repeats=10)

    print("\n--- Key Observations ---")
    print("1. Holdout (single): High std → high variance in estimate")
    print("2. Holdout (repeated): Lower std → but still high variance per single run")
    print("3. k-Fold: Lower std than holdout → more reliable estimate")
    print("4. Stratified k-Fold: Similar to k-fold but guaranteed class balance")
    print("5. LOO-CV: Very low bias, but std is high for classification (each score is 0/1)")


if __name__ == "__main__":
    main()
