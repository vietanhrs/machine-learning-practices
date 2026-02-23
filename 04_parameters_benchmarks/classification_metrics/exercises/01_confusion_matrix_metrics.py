"""
Exercise 01 — Confusion Matrix & Core Classification Metrics
=============================================================

Implement all standard binary classification metrics from scratch using only
NumPy. After implementing each function, compare your results to scikit-learn's
implementations in `print_metrics_report()`.

Learning objectives:
- Understand the raw counts (TP, TN, FP, FN) that underlie all metrics.
- See why accuracy fails on imbalanced data.
- Appreciate the tradeoff between precision and recall.
- Understand MCC as a robust single metric.

Run:
    python 01_confusion_matrix_metrics.py
"""

import numpy as np
from sklearn import metrics as skmetrics


# ---------------------------------------------------------------------------
# Helper: extract TP, TN, FP, FN
# ---------------------------------------------------------------------------

def _get_counts(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Extract the four confusion matrix counts for binary classification.

    Parameters
    ----------
    y_true : array of shape (n,) with values 0 or 1
    y_pred : array of shape (n,) with values 0 or 1

    Returns
    -------
    tp, tn, fp, fn : int
        True Positives, True Negatives, False Positives, False Negatives.

    Hint: Use boolean masking.
        tp = sum where y_true == 1 AND y_pred == 1
    """
    # TODO: implement using boolean array operations
    pass


# ---------------------------------------------------------------------------
# Core metrics (all TODO)
# ---------------------------------------------------------------------------

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute the 2x2 confusion matrix.

    Returns
    -------
    cm : np.ndarray of shape (2, 2)
        [[TN, FP],
         [FN, TP]]

    This matches sklearn's convention: rows = actual, columns = predicted,
    with label order [0, 1].

    Hint: Use _get_counts() and np.array().
    """
    # TODO: implement
    pass


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute accuracy = (TP + TN) / total.

    Hint: Simply compare arrays element-wise and take the mean.
    """
    # TODO: implement
    pass


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute precision = TP / (TP + FP).

    Edge case: if TP + FP == 0 (no positive predictions), return 0.0.

    Hint: Use _get_counts(). Watch out for zero-division.
    """
    # TODO: implement
    pass


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute recall (sensitivity) = TP / (TP + FN).

    Edge case: if TP + FN == 0 (no actual positives), return 0.0.

    Hint: Use _get_counts(). Watch out for zero-division.
    """
    # TODO: implement
    pass


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute F1 = 2 * precision * recall / (precision + recall).

    Edge case: if precision + recall == 0, return 0.0.

    Hint: Reuse the precision() and recall() functions you already wrote.
    """
    # TODO: implement
    pass


def f_beta_score(y_true: np.ndarray, y_pred: np.ndarray, beta: float) -> float:
    """
    Compute the generalized F-beta score.

    Formula:
        F_beta = (1 + beta²) * precision * recall
                 / (beta² * precision + recall)

    Parameters
    ----------
    beta : float
        beta > 1 → weights recall more (use when FN is costly).
        beta < 1 → weights precision more (use when FP is costly).
        beta = 1 → standard F1.

    Edge case: if the denominator is 0, return 0.0.

    Hint: Compute precision and recall first, then apply the formula.
    """
    # TODO: implement
    pass


def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute specificity (True Negative Rate) = TN / (TN + FP).

    This is recall applied to the negative class.

    Edge case: if TN + FP == 0 (no actual negatives), return 0.0.

    Hint: Use _get_counts().
    """
    # TODO: implement
    pass


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute balanced accuracy = (recall + specificity) / 2
                              = (TPR + TNR) / 2.

    This metric handles class imbalance by giving equal weight to both classes.

    Hint: Reuse recall() and specificity().
    """
    # TODO: implement
    pass


def mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the Matthews Correlation Coefficient (MCC).

    Formula:
        MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))

    Range: -1 (perfectly wrong) to 0 (random) to +1 (perfect).

    Edge case: if the denominator is 0, return 0.0.

    MCC uses all four confusion matrix cells, making it robust for imbalanced
    datasets. Unlike F1, it treats both classes symmetrically.

    Hint: Use _get_counts(). Use np.sqrt() for the square root.
          Cast values to float before squaring to avoid integer overflow.
    """
    # TODO: implement
    pass


# ---------------------------------------------------------------------------
# Report (PROVIDED — do not modify)
# ---------------------------------------------------------------------------

def print_metrics_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Print all metrics side-by-side with scikit-learn's values for verification.

    This function is fully implemented — use it to check your implementations.
    """
    print("=" * 60)
    print(f"{'Metric':<25} {'Yours':>10} {'sklearn':>10} {'Match':>7}")
    print("=" * 60)

    metrics_to_check = [
        ("Accuracy",          accuracy(y_true, y_pred),
                              skmetrics.accuracy_score(y_true, y_pred)),
        ("Precision",         precision(y_true, y_pred),
                              skmetrics.precision_score(y_true, y_pred, zero_division=0)),
        ("Recall",            recall(y_true, y_pred),
                              skmetrics.recall_score(y_true, y_pred, zero_division=0)),
        ("F1 Score",          f1_score(y_true, y_pred),
                              skmetrics.f1_score(y_true, y_pred, zero_division=0)),
        ("F2 Score",          f_beta_score(y_true, y_pred, beta=2),
                              skmetrics.fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        ("F0.5 Score",        f_beta_score(y_true, y_pred, beta=0.5),
                              skmetrics.fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)),
        ("Specificity",       specificity(y_true, y_pred),
                              skmetrics.recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        ("Balanced Accuracy", balanced_accuracy(y_true, y_pred),
                              skmetrics.balanced_accuracy_score(y_true, y_pred)),
        ("MCC",               mcc(y_true, y_pred),
                              skmetrics.matthews_corrcoef(y_true, y_pred)),
    ]

    for name, yours, sk in metrics_to_check:
        if yours is None:
            print(f"  {name:<23} {'TODO':>10} {sk:>10.4f} {'N/A':>7}")
        else:
            match = "OK" if abs(yours - sk) < 1e-6 else "FAIL"
            print(f"  {name:<23} {yours:>10.4f} {sk:>10.4f} {match:>7}")

    print("=" * 60)
    print("\nConfusion Matrix (yours):")
    cm = confusion_matrix(y_true, y_pred)
    if cm is not None:
        print(f"  TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
        print(f"  FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")
    else:
        print("  TODO: not implemented")

    print("\nConfusion Matrix (sklearn):")
    sk_cm = skmetrics.confusion_matrix(y_true, y_pred)
    print(f"  TN={sk_cm[0,0]:4d}  FP={sk_cm[0,1]:4d}")
    print(f"  FN={sk_cm[1,0]:4d}  TP={sk_cm[1,1]:4d}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    Demonstrate metrics on a binary classification problem with class imbalance.

    Dataset: 1000 samples, 90% class 0 (majority), 10% class 1 (minority).
    This simulates a realistic scenario such as fraud detection.

    Two models are compared:
    1. Majority Classifier: always predicts class 0.
    2. Realistic Classifier: simulates a model with real but imperfect signal.
    """
    np.random.seed(42)
    n = 1000
    imbalance_ratio = 0.10  # 10% positive class

    # Ground truth
    y_true = np.zeros(n, dtype=int)
    n_positive = int(n * imbalance_ratio)
    positive_indices = np.random.choice(n, size=n_positive, replace=False)
    y_true[positive_indices] = 1

    print(f"Dataset: {n} samples, {n_positive} positives ({100*imbalance_ratio:.0f}%)")
    print(f"Baseline (always 0) accuracy: {(y_true == 0).mean():.4f}\n")

    # --- Model 1: Majority Classifier (always predicts 0) ---
    print("=" * 60)
    print("MODEL 1: Majority Classifier (always predicts class 0)")
    print("=" * 60)
    y_pred_majority = np.zeros(n, dtype=int)
    print_metrics_report(y_true, y_pred_majority)

    # --- Model 2: Realistic Classifier ---
    print("\n" + "=" * 60)
    print("MODEL 2: Realistic Classifier (imperfect model)")
    print("=" * 60)
    # Simulate a model: catches 70% of positives, has 5% FP rate on negatives
    y_pred_realistic = np.zeros(n, dtype=int)
    for i in range(n):
        if y_true[i] == 1:
            # 70% chance to correctly detect a positive
            if np.random.rand() < 0.70:
                y_pred_realistic[i] = 1
        else:
            # 5% false positive rate on negatives
            if np.random.rand() < 0.05:
                y_pred_realistic[i] = 1
    print_metrics_report(y_true, y_pred_realistic)

    # --- Observation ---
    print("\n--- KEY OBSERVATION ---")
    print("Notice that the Majority Classifier achieves high accuracy but")
    print("has Recall=0, F1=0, and MCC=0 — it is completely useless!")
    print("The Realistic Classifier has lower accuracy but meaningful F1/MCC.")
    print("Always report multiple metrics on imbalanced datasets.")


if __name__ == "__main__":
    main()
