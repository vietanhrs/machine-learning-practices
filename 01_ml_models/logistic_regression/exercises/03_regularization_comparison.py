"""
Exercise 03: Regularization Comparison — L1 vs L2
==================================================

Goal: Empirically observe how L1 (Lasso) and L2 (Ridge) regularization
      affect logistic regression coefficients as regularization strength varies.

Learning Objectives:
    - Understand the "regularization path" — how coefficients change with C.
    - Observe that L1 drives coefficients to exactly zero (sparse solution).
    - Observe that L2 shrinks all coefficients but rarely zeroes them out.
    - Learn when each regularization type is preferable.

Background:
    In sklearn's LogisticRegression, C = 1 / λ:
        - Small C  → strong regularization → simpler model
        - Large C  → weak regularization  → complex model (may overfit)

    L1 regularization objective:
        J_reg = J_BCE + λ Σ |wᵢ|

    L2 regularization objective:
        J_reg = J_BCE + (λ/2) Σ wᵢ²

Instructions:
    - Fill in every section marked with TODO.
    - Use sklearn's LogisticRegression (you do not need to implement GD here).
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# ---------------------------------------------------------------------------
# Train Models Across C Values
# ---------------------------------------------------------------------------

def train_with_regularization(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C_values: np.ndarray,
    penalty: str,
) -> np.ndarray:
    """
    Train logistic regression for each C value and collect coefficients.

    Parameters
    ----------
    X_train   : np.ndarray, shape (n_train, n_features)
    y_train   : np.ndarray, shape (n_train,)
    C_values  : np.ndarray — array of C values (regularization strengths)
    penalty   : str — 'l1' or 'l2'

    Returns
    -------
    np.ndarray, shape (len(C_values), n_features)
        Coefficient matrix where row i corresponds to C_values[i].

    TODO:
        - For each C in C_values:
            1. Instantiate LogisticRegression(
                   C=C,
                   penalty=penalty,
                   solver='liblinear',   # supports both l1 and l2
                   max_iter=1000,
                   random_state=42
               ).
            2. Fit on (X_train, y_train).
            3. Extract model.coef_[0] (shape: n_features).
            4. Store in a list.
        - Return np.array of all collected coefficient vectors.
    """
    # TODO: train models for each C value and collect coefficients
    raise NotImplementedError("Implement train_with_regularization()")


# ---------------------------------------------------------------------------
# Coefficient Path Plot (provided — do not modify)
# ---------------------------------------------------------------------------

def plot_coefficient_paths(
    C_values: np.ndarray,
    coef_matrix: np.ndarray,
    feature_names: list[str],
    penalty: str,
) -> None:
    """
    Plot how each feature's coefficient changes as C varies.

    A "regularization path" shows the trajectory of coefficients from
    strong regularization (small C, left) to weak regularization (large C, right).

    Parameters
    ----------
    C_values      : np.ndarray — array of C values (x-axis)
    coef_matrix   : np.ndarray, shape (n_C, n_features)
    feature_names : list[str]
    penalty       : str — used in the plot title
    """
    plt.figure(figsize=(10, 6))
    for i, name in enumerate(feature_names):
        plt.plot(np.log10(C_values), coef_matrix[:, i], label=name, linewidth=1.5)
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.xlabel("log₁₀(C)  [← stronger regularization | weaker regularization →]")
    plt.ylabel("Coefficient Value")
    plt.title(f"Regularization Path — {penalty.upper()} Penalty")
    plt.legend(loc="upper left", fontsize=7, ncol=2, framealpha=0.7)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    filename = f"coef_path_{penalty}.png"
    plt.savefig(filename, dpi=120)
    plt.show()
    print(f"Saved: {filename}")


# ---------------------------------------------------------------------------
# L1 vs L2 Comparison
# ---------------------------------------------------------------------------

def compare_l1_l2(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
) -> None:
    """
    Train both L1 and L2 regularized models across a range of C values,
    plot their coefficient paths, and compare sparsity and test accuracy.

    Parameters
    ----------
    X_train, X_test : np.ndarray — scaled features
    y_train, y_test : np.ndarray — labels
    feature_names   : list[str]

    TODO:
        1. Define C_values = np.logspace(-3, 3, 50) (50 values from 0.001 to 1000).
        2. For each penalty in ['l1', 'l2']:
           a. Call train_with_regularization() to get coef_matrix.
           b. Call plot_coefficient_paths() to visualise the path.
           c. For each C value, compute test accuracy and store it.
        3. Plot test accuracy vs. log10(C) for both L1 and L2 on the same axes.
           Title: "Test Accuracy vs. Regularization Strength"
        4. At the optimal C for each penalty (highest test accuracy), print:
           - Number of zero coefficients (|coef| < 1e-4) → sparsity for L1
           - Number of non-zero coefficients for L2
        5. Print a summary comparing sparsity and accuracy.

        Key insight to observe:
            L1 path: many coefficients collapse to exactly 0 at low C values.
            L2 path: coefficients shrink smoothly but remain non-zero.
    """
    # TODO: compare L1 and L2 paths, accuracy, and sparsity
    raise NotImplementedError("Implement compare_l1_l2()")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Load the breast cancer dataset and run the full regularization comparison.
    """
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = list(data.feature_names)

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    compare_l1_l2(X_train, X_test, y_train, y_test, feature_names)


if __name__ == "__main__":
    main()
