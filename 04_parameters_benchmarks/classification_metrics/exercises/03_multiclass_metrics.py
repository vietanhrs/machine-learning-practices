"""
Exercise 03 — Multiclass Classification Metrics
================================================

Extend binary classification metrics to the multiclass setting.
Implement macro, micro, and weighted averaging strategies from scratch.

Learning objectives:
- Understand how binary metrics generalize to K classes.
- Know when to use macro vs micro vs weighted averaging.
- Visualize a multiclass confusion matrix as a heatmap.

Run:
    python 03_multiclass_metrics.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics as skmetrics


# ---------------------------------------------------------------------------
# Multiclass confusion matrix (TODO)
# ---------------------------------------------------------------------------

def multiclass_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """
    Compute the K x K confusion matrix for multiclass classification.

    Parameters
    ----------
    y_true    : array of shape (n,) with integer class labels 0..K-1.
    y_pred    : array of shape (n,) with integer class labels 0..K-1.
    n_classes : int — number of classes K.

    Returns
    -------
    cm : np.ndarray of shape (K, K)
        cm[i, j] = number of samples where true class is i and predicted
        class is j.
        Diagonal elements are correct predictions.
        Off-diagonal elements are errors.

    Algorithm:
        Initialize cm = np.zeros((n_classes, n_classes), dtype=int)
        For each sample, increment cm[y_true[i], y_pred[i]] by 1.

    Hint: Avoid a Python loop over samples if you can.
          np.add.at(cm, (y_true, y_pred), 1) works efficiently.
          Or use: for true, pred in zip(y_true, y_pred): cm[true][pred] += 1
    """
    # TODO: implement
    pass


# ---------------------------------------------------------------------------
# Per-class metrics (TODO)
# ---------------------------------------------------------------------------

def per_class_precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
) -> dict:
    """
    Compute precision, recall, and F1 for each class in a one-vs-rest manner.

    Parameters
    ----------
    y_true    : array of shape (n,)
    y_pred    : array of shape (n,)
    n_classes : int

    Returns
    -------
    results : dict with keys 'precision', 'recall', 'f1', each mapping to
              a np.ndarray of shape (n_classes,).

    Algorithm for class k:
        TP_k = sum(y_true == k AND y_pred == k)
        FP_k = sum(y_true != k AND y_pred == k)
        FN_k = sum(y_true == k AND y_pred != k)
        TN_k = sum(y_true != k AND y_pred != k)

        precision_k = TP_k / (TP_k + FP_k)   [0 if denominator is 0]
        recall_k    = TP_k / (TP_k + FN_k)    [0 if denominator is 0]
        f1_k        = 2 * precision_k * recall_k / (precision_k + recall_k)
                                               [0 if denominator is 0]

    Hint: Loop over classes k = 0, 1, ..., n_classes - 1.
          Use boolean indexing for efficient computation.
    """
    # TODO: implement
    pass


# ---------------------------------------------------------------------------
# Averaging strategies (TODO)
# ---------------------------------------------------------------------------

def macro_average(per_class_scores: np.ndarray) -> float:
    """
    Compute the macro average of per-class scores.

    Macro average = simple unweighted mean over all classes.

    Formula:
        macro = sum(per_class_scores) / n_classes

    Parameters
    ----------
    per_class_scores : np.ndarray of shape (n_classes,)

    Returns
    -------
    macro : float

    Hint: Use np.mean().
    """
    # TODO: implement
    pass


def micro_average(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
) -> dict:
    """
    Compute micro-averaged precision, recall, and F1.

    Micro average: aggregate TP, FP, FN across ALL classes, then compute.

    Formula:
        TP_total = sum over k of TP_k
        FP_total = sum over k of FP_k
        FN_total = sum over k of FN_k

        micro_precision = TP_total / (TP_total + FP_total)
        micro_recall    = TP_total / (TP_total + FN_total)
        micro_f1        = 2 * micro_precision * micro_recall
                          / (micro_precision + micro_recall)

    Note: For single-label multiclass, micro F1 = accuracy.

    Returns
    -------
    dict with keys 'precision', 'recall', 'f1'.

    Hint: Reuse the loop from per_class_precision_recall_f1 to accumulate totals.
    """
    # TODO: implement
    pass


def weighted_average(
    per_class_scores: np.ndarray,
    class_counts: np.ndarray,
) -> float:
    """
    Compute the weighted average of per-class scores, weighted by class support.

    Formula:
        weighted = sum(per_class_scores * class_counts) / sum(class_counts)

    Parameters
    ----------
    per_class_scores : np.ndarray of shape (n_classes,) — metric for each class.
    class_counts     : np.ndarray of shape (n_classes,) — number of true samples
                       per class (support).

    Returns
    -------
    weighted : float

    Hint: Use np.average(per_class_scores, weights=class_counts).
    """
    # TODO: implement
    pass


# ---------------------------------------------------------------------------
# Heatmap visualization (PROVIDED — do not modify)
# ---------------------------------------------------------------------------

def plot_multiclass_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    title: str = "Confusion Matrix",
    ax=None,
) -> None:
    """
    Plot a confusion matrix as an annotated heatmap. Fully implemented.

    Parameters
    ----------
    cm          : np.ndarray of shape (K, K)
    class_names : list of str — class label names (length K)
    title       : str
    ax          : matplotlib Axes (optional)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names) - 1)))

    # Normalize for color scale but annotate with raw counts
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    sns.heatmap(
        cm_norm,
        annot=cm,           # show raw counts as annotations
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "Proportion"},
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    Evaluate a Logistic Regression classifier on the Iris dataset.
    Compare macro, micro, and weighted F1 scores.
    Show the confusion matrix as a heatmap.
    """
    # --- Load data ---
    iris = load_iris()
    X, y = iris.data, iris.target
    class_names = list(iris.target_names)
    n_classes = len(class_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    # --- Fit model ---
    model = LogisticRegression(max_iter=300, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # --- Confusion matrix ---
    cm = multiclass_confusion_matrix(y_test, y_pred, n_classes)
    class_counts = np.bincount(y_test, minlength=n_classes)

    # --- Per-class metrics ---
    per_class = per_class_precision_recall_f1(y_test, y_pred, n_classes)

    # --- Averaging ---
    macro_f1    = macro_average(per_class["f1"]) if per_class else None
    micro_vals  = micro_average(y_test, y_pred, n_classes)
    micro_f1    = micro_vals["f1"] if micro_vals else None
    weighted_f1 = weighted_average(per_class["f1"], class_counts) if per_class else None

    # --- Print results ---
    print("=" * 65)
    print(f"{'Metric':<30} {'Yours':>10} {'sklearn':>10} {'Match':>7}")
    print("=" * 65)

    sk_report = skmetrics.classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True
    )

    if per_class:
        for k, name in enumerate(class_names):
            yours_f1 = per_class["f1"][k]
            sk_f1 = sk_report[name]["f1-score"]
            match = "OK" if abs(yours_f1 - sk_f1) < 1e-6 else "FAIL"
            print(f"  F1 ({name:<12})          {yours_f1:>10.4f} {sk_f1:>10.4f} {match:>7}")

    print("-" * 65)

    def check(label, yours, sk_val):
        if yours is None:
            print(f"  {label:<28} {'TODO':>10} {sk_val:>10.4f} {'N/A':>7}")
        else:
            match = "OK" if abs(yours - sk_val) < 1e-6 else "FAIL"
            print(f"  {label:<28} {yours:>10.4f} {sk_val:>10.4f} {match:>7}")

    check("Macro F1",    macro_f1,    sk_report["macro avg"]["f1-score"])
    check("Micro F1",    micro_f1,    sk_report["accuracy"])
    check("Weighted F1", weighted_f1, sk_report["weighted avg"]["f1-score"])
    print("=" * 65)

    # --- Visualize ---
    fig, ax = plt.subplots(figsize=(7, 6))
    if cm is not None:
        plot_multiclass_confusion_matrix(
            cm, class_names,
            title=f"Iris Confusion Matrix (Logistic Regression)\n"
                  f"Macro F1={macro_f1:.3f}" if macro_f1 else "Iris Confusion Matrix",
            ax=ax,
        )
    else:
        ax.text(0.5, 0.5, "TODO: confusion_matrix not implemented",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Confusion Matrix")

    plt.tight_layout()
    plt.savefig("multiclass_confusion_matrix.png", dpi=120, bbox_inches="tight")
    print("\nSaved: multiclass_confusion_matrix.png")
    plt.show()


if __name__ == "__main__":
    main()
