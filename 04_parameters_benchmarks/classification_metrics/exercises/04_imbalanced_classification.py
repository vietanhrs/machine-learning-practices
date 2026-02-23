"""
Exercise 04 — Imbalanced Classification
========================================

Explore the challenges of imbalanced datasets and common mitigation strategies.

Learning objectives:
- Show that accuracy is misleading on imbalanced data.
- Find the optimal classification threshold on a validation set.
- Apply oversampling (SMOTE or random) to rebalance the training set.
- Compare model performance before and after resampling.

Run:
    python 04_imbalanced_classification.py

Dependencies:
    pip install imbalanced-learn   (for SMOTE; falls back to random if absent)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics as skmetrics


# ---------------------------------------------------------------------------
# Dataset generation (TODO)
# ---------------------------------------------------------------------------

def generate_imbalanced_dataset(
    n_samples: int = 2000,
    imbalance_ratio: float = 0.05,
    random_state: int = 42,
):
    """
    Generate a binary classification dataset with severe class imbalance.

    Parameters
    ----------
    n_samples       : int — total number of samples.
    imbalance_ratio : float — proportion of the minority class (e.g., 0.05 = 5%).
    random_state    : int

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples,) — binary labels {0, 1}

    Algorithm:
        Use sklearn.datasets.make_classification with:
            weights=[1 - imbalance_ratio, imbalance_ratio]
            n_features=10, n_informative=5, n_redundant=2
            random_state=random_state

    Hint: make_classification accepts a `weights` parameter that controls
          the proportion of samples in each class.
    """
    # TODO: implement
    pass


# ---------------------------------------------------------------------------
# Metric comparison on imbalanced data (TODO)
# ---------------------------------------------------------------------------

def compare_metrics_on_imbalanced(X: np.ndarray, y: np.ndarray) -> None:
    """
    Train a simple classifier and show how accuracy is misleading while
    F1, AUC, and MCC are informative.

    Steps:
        1. Split into train/val/test (60/20/20), stratified.
        2. Fit LogisticRegression on train.
        3. Predict on test with threshold=0.5.
        4. Also create a naive "always predict 0" baseline.
        5. Print a comparison table:
               Metric | Naive Baseline | Logistic Regression
               Accuracy, Precision, Recall, F1, AUC-ROC, MCC

    Hint: Use skmetrics.accuracy_score, precision_score, recall_score,
          f1_score, roc_auc_score, matthews_corrcoef.
          Show that the naive baseline achieves high accuracy but F1=MCC=0.
    """
    # TODO: implement
    pass


# ---------------------------------------------------------------------------
# Threshold tuning (TODO)
# ---------------------------------------------------------------------------

def threshold_tuning(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    metric: str = "f1",
) -> float:
    """
    Find the classification threshold that maximizes a given metric on the
    validation set.

    Parameters
    ----------
    model  : fitted sklearn model with predict_proba().
    X_val  : validation features.
    y_val  : validation labels.
    metric : str — one of 'f1', 'recall', 'precision', 'balanced_accuracy'.

    Returns
    -------
    best_threshold : float — the threshold in [0, 1] that maximizes `metric`.

    Algorithm:
        thresholds = np.linspace(0.01, 0.99, 200)
        For each t in thresholds:
            y_pred = (predict_proba(X_val)[:, 1] >= t).astype(int)
            score  = compute metric(y_val, y_pred)
        best_threshold = thresholds[argmax(scores)]

    Hint: Use if/elif to dispatch on the metric name:
        'f1'               → skmetrics.f1_score(zero_division=0)
        'recall'           → skmetrics.recall_score(zero_division=0)
        'precision'        → skmetrics.precision_score(zero_division=0)
        'balanced_accuracy'→ skmetrics.balanced_accuracy_score()
    """
    # TODO: implement
    pass


# ---------------------------------------------------------------------------
# Oversampling (TODO)
# ---------------------------------------------------------------------------

def oversample_smote(X: np.ndarray, y: np.ndarray, random_state: int = 42):
    """
    Oversample the minority class using SMOTE (or random oversampling as fallback).

    Parameters
    ----------
    X : np.ndarray of shape (n_train, n_features)
    y : np.ndarray of shape (n_train,)

    Returns
    -------
    X_resampled : np.ndarray
    y_resampled : np.ndarray

    Algorithm (preferred — SMOTE):
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=random_state)
        X_res, y_res = smote.fit_resample(X, y)

    Fallback (if imblearn is not installed — random oversampling):
        1. Find minority class indices.
        2. Calculate how many extra samples needed to balance.
        3. Sample with replacement from minority indices.
        4. Concatenate original data with oversampled minority.
        5. Shuffle and return.

    Hint: Use try/except ImportError to handle the optional imblearn dependency.
    """
    # TODO: implement
    pass


# ---------------------------------------------------------------------------
# Before/after comparison (TODO)
# ---------------------------------------------------------------------------

def evaluate_before_after_resampling(X: np.ndarray, y: np.ndarray) -> None:
    """
    Full pipeline: compare model performance before and after oversampling.

    Steps:
        1. Split into train (70%) and test (30%), stratified.
        2. Train LogisticRegression on original (imbalanced) train set.
        3. Oversample train set using oversample_smote().
        4. Train LogisticRegression on resampled train set.
        5. Evaluate BOTH models on the SAME original (unmodified) test set.
        6. Print comparison table showing:
               AUC-ROC, F1, Recall, Precision, MCC, Accuracy
           for both models.
        7. Optionally: apply threshold_tuning() to the oversampled model
           on a validation split before reporting test metrics.

    Hint: Only resample the TRAINING set. Never resample the test set.
          This is a critical best practice to avoid data leakage.
    """
    # TODO: implement
    pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    Full demonstration of imbalanced classification challenges and solutions.
    """
    print("=" * 65)
    print("IMBALANCED CLASSIFICATION — Full Pipeline")
    print("=" * 65)

    # --- Step 1: Generate dataset ---
    X, y = generate_imbalanced_dataset(n_samples=3000, imbalance_ratio=0.05)

    if X is not None:
        n_pos = y.sum()
        n_neg = (y == 0).sum()
        print(f"\nDataset: {len(y)} samples — {n_neg} negative ({n_neg/len(y):.1%}), "
              f"{n_pos} positive ({n_pos/len(y):.1%})")
    else:
        print("\nTODO: generate_imbalanced_dataset not implemented")
        return

    # --- Step 2: Show problem ---
    print("\n--- STEP 2: Metrics comparison (accuracy vs F1/AUC) ---")
    compare_metrics_on_imbalanced(X, y)

    # --- Step 3: Threshold tuning ---
    print("\n--- STEP 3: Threshold tuning ---")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X_tr, y_tr)

    for metric_name in ["f1", "recall", "balanced_accuracy"]:
        best_t = threshold_tuning(model, X_val, y_val, metric=metric_name)
        if best_t is not None:
            print(f"  Best threshold for {metric_name:>20}: {best_t:.4f}")
        else:
            print(f"  threshold_tuning({metric_name}) — TODO")

    # --- Step 4: Before/after resampling ---
    print("\n--- STEP 4: Before vs After oversampling ---")
    evaluate_before_after_resampling(X, y)

    print("\nDone. Key takeaways:")
    print("  1. Never use accuracy alone on imbalanced data.")
    print("  2. Threshold tuning can significantly improve recall without retraining.")
    print("  3. Oversampling (SMOTE) improves minority class detection.")
    print("  4. Always evaluate on the ORIGINAL unmodified test set.")


if __name__ == "__main__":
    main()
