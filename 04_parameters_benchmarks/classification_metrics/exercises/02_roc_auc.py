"""
Exercise 02 — ROC Curve, AUC, and Precision-Recall Curve
=========================================================

Implement ROC and Precision-Recall curve computation from scratch.
Compare multiple classifiers visually.

Learning objectives:
- Understand that ROC/PR curves are produced by varying the threshold.
- Implement the trapezoidal rule for AUC computation.
- Know when to prefer PR curves over ROC curves (imbalanced data).
- Apply Youden's J statistic for optimal threshold selection.

Run:
    python 02_roc_auc.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics as skmetrics


# ---------------------------------------------------------------------------
# Core curve computations (TODO)
# ---------------------------------------------------------------------------

def compute_roc_curve(y_true: np.ndarray, y_scores: np.ndarray):
    """
    Compute the ROC curve by varying the classification threshold from 1 to 0.

    Parameters
    ----------
    y_true  : array of shape (n,) with binary labels {0, 1}.
    y_scores: array of shape (n,) with predicted probabilities for class 1.

    Returns
    -------
    fpr        : np.ndarray — False Positive Rate at each threshold.
    tpr        : np.ndarray — True Positive Rate (Recall) at each threshold.
    thresholds : np.ndarray — threshold values used.

    Algorithm:
        1. Sort all unique values in y_scores in descending order.
           Add 1.0 at the start (everything negative) and 0.0 at the end
           (everything positive) as sentinel thresholds — or simply use
           np.unique(y_scores)[::-1].
        2. For each threshold t:
               y_pred = (y_scores >= t).astype(int)
               Compute TP, FP, TN, FN from y_true and y_pred.
               TPR = TP / (TP + FN)    (handle zero-division → 0.0)
               FPR = FP / (FP + TN)    (handle zero-division → 0.0)
        3. Prepend point (FPR=0, TPR=0) and append (FPR=1, TPR=1) to close
           the curve.

    Hint: The curve should start at (0, 0) and end at (1, 1).
    """
    # TODO: implement
    pass


def compute_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """
    Compute the Area Under the ROC Curve using the trapezoidal rule.

    Formula (trapezoidal rule):
        AUC = sum over i of: (fpr[i+1] - fpr[i]) * (tpr[i+1] + tpr[i]) / 2

    This is equivalent to np.trapz(tpr, fpr).

    Parameters
    ----------
    fpr : np.ndarray — x-axis values (should be sorted ascending).
    tpr : np.ndarray — y-axis values.

    Returns
    -------
    auc : float — area under the curve.

    Hint: Use np.trapz(y=tpr, x=fpr) or implement manually with a loop.
          Make sure fpr is sorted ascending before computing.
    """
    # TODO: implement
    pass


def compute_pr_curve(y_true: np.ndarray, y_scores: np.ndarray):
    """
    Compute the Precision-Recall curve by varying the threshold.

    Parameters
    ----------
    y_true  : array of shape (n,) with binary labels {0, 1}.
    y_scores: array of shape (n,) with predicted probabilities for class 1.

    Returns
    -------
    precision_vals : np.ndarray — Precision at each threshold.
    recall_vals    : np.ndarray — Recall at each threshold.
    thresholds     : np.ndarray — threshold values used.

    Algorithm:
        1. Use the same threshold sweep as in compute_roc_curve.
        2. At each threshold t:
               y_pred = (y_scores >= t).astype(int)
               precision = TP / (TP + FP)   (if TP+FP == 0, set to 1.0)
               recall    = TP / (TP + FN)   (if TP+FN == 0, set to 0.0)
        3. The curve starts with high precision (strict threshold) and ends
           with high recall (lenient threshold).

    Hint: Sort thresholds in DESCENDING order so recall increases left-to-right.
          sklearn's convention is to include the point (recall=0, precision=1).
    """
    # TODO: implement
    pass


def find_optimal_threshold_youden(
    fpr: np.ndarray,
    tpr: np.ndarray,
    thresholds: np.ndarray,
) -> float:
    """
    Find the optimal classification threshold using Youden's J statistic.

    Youden's J = TPR - FPR = Sensitivity + Specificity - 1

    The optimal threshold maximizes J, i.e., the point on the ROC curve
    farthest above the diagonal.

    Parameters
    ----------
    fpr        : np.ndarray — False Positive Rate values from compute_roc_curve.
    tpr        : np.ndarray — True Positive Rate values from compute_roc_curve.
    thresholds : np.ndarray — corresponding threshold values.

    Returns
    -------
    optimal_threshold : float

    Hint:
        J = tpr - fpr
        optimal_idx = np.argmax(J)
        Be careful: the thresholds array may have one fewer element than
        fpr/tpr if sentinel points were added. Align them carefully.
    """
    # TODO: implement
    pass


def compare_models_roc(models: list, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Plot multiple ROC curves on the same axes for comparison.

    Parameters
    ----------
    models : list of tuples (name: str, fitted_model)
             Each model must have a predict_proba() method.
    X_test : np.ndarray
    y_test : np.ndarray

    Hint:
        For each (name, model) in models:
            y_scores = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = compute_roc_curve(y_test, y_scores)
            auc = compute_auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

        Add the diagonal (random classifier) as a dashed line.
        Add legend, labels, and title.
    """
    # TODO: implement
    pass


# ---------------------------------------------------------------------------
# Plotting (PROVIDED — do not modify)
# ---------------------------------------------------------------------------

def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    title: str = "ROC Curve",
    ax=None,
) -> None:
    """
    Plot a single ROC curve. Fully implemented — do not modify.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    ax.plot(fpr, tpr, color="steelblue", lw=2,
            label=f"ROC curve (AUC = {auc_score:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1,
            label="Random classifier (AUC = 0.5)")
    ax.fill_between(fpr, tpr, alpha=0.1, color="steelblue")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)


def plot_pr_curve(
    precision_vals: np.ndarray,
    recall_vals: np.ndarray,
    title: str = "Precision-Recall Curve",
    ax=None,
) -> None:
    """
    Plot a Precision-Recall curve. Fully implemented — do not modify.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    ax.plot(recall_vals, precision_vals, color="darkorange", lw=2,
            label="PR Curve")
    ax.fill_between(recall_vals, precision_vals, alpha=0.1, color="darkorange")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    Compare Logistic Regression, Random Forest, and a Dummy Classifier
    using ROC curves, AUC, and PR curves.

    The dataset is imbalanced (10% positive class) to demonstrate when
    PR curves are more informative than ROC curves.
    """
    # --- Data ---
    X, y = make_classification(
        n_samples=2000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        weights=[0.90, 0.10],   # 90/10 imbalance
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    # --- Models ---
    lr = LogisticRegression(max_iter=500, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    dummy = DummyClassifier(strategy="stratified", random_state=42)

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    dummy.fit(X_train, y_train)

    models = [
        ("Logistic Regression", lr),
        ("Random Forest", rf),
        ("Dummy (Random)", dummy),
    ]

    # --- Single ROC curve (Logistic Regression) ---
    lr_scores = lr.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = compute_roc_curve(y_test, lr_scores)
    auc_score = compute_auc(fpr, tpr)
    pr_precision, pr_recall, _ = compute_pr_curve(y_test, lr_scores)

    print("=== ROC / AUC Analysis (Logistic Regression) ===")
    if auc_score is not None:
        print(f"  Your AUC:    {auc_score:.4f}")
        sk_fpr, sk_tpr, _ = skmetrics.roc_curve(y_test, lr_scores)
        print(f"  sklearn AUC: {skmetrics.auc(sk_fpr, sk_tpr):.4f}")

    if thresholds is not None and fpr is not None and tpr is not None:
        opt_thresh = find_optimal_threshold_youden(fpr, tpr, thresholds)
        if opt_thresh is not None:
            print(f"  Optimal threshold (Youden's J): {opt_thresh:.4f}")

    # --- Plots ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    if fpr is not None and tpr is not None and auc_score is not None:
        plot_roc_curve(fpr, tpr, auc_score,
                       title="ROC Curve — Logistic Regression", ax=axes[0])
    else:
        axes[0].set_title("ROC Curve — TODO not implemented")

    if pr_precision is not None and pr_recall is not None:
        plot_pr_curve(pr_precision, pr_recall,
                      title="PR Curve — Logistic Regression", ax=axes[1])
    else:
        axes[1].set_title("PR Curve — TODO not implemented")

    plt.tight_layout()
    plt.savefig("roc_pr_curves.png", dpi=120, bbox_inches="tight")
    print("\nSaved: roc_pr_curves.png")

    # --- Multi-model comparison ---
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    compare_models_roc(models, X_test, y_test)
    plt.tight_layout()
    plt.savefig("multi_model_roc.png", dpi=120, bbox_inches="tight")
    print("Saved: multi_model_roc.png")

    plt.show()


if __name__ == "__main__":
    main()
