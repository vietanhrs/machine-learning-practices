"""
Exercise 03: Gradient Boosting from Scratch
============================================

Goal: Implement a gradient boosting classifier by sequentially fitting
      shallow decision trees to the residuals (pseudo-residuals) of the
      current ensemble, then compare with sklearn's GradientBoostingClassifier.

Learning Objectives:
    - Understand gradient boosting as iterative residual fitting.
    - Implement the sequential predict-residual-fit-update loop.
    - Understand why shallow trees (low max_depth) are preferred.
    - Compare training dynamics and feature importances with sklearn.

Background:
    For binary classification with log-loss, gradient boosting proceeds as:
        1. Initialize F₀ = log(p / (1-p)) where p = mean(y_train)
           (the log-odds of the base rate).
        2. For m = 1 to n_estimators:
           a. Compute pseudo-residuals (negative gradient of log-loss):
              r_m = y - sigmoid(F_{m-1})
           b. Fit a shallow decision tree h_m to (X, r_m).
           c. Update: F_m = F_{m-1} + learning_rate * h_m(X)
        3. Predict: ŷ = sigmoid(F_n) ≥ 0.5 → class 1.

Instructions:
    - Fill in every section marked with TODO.
    - Use sklearn's DecisionTreeRegressor as the base learner.
    - Do NOT use sklearn's GradientBoostingClassifier inside your class.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.ensemble import GradientBoostingClassifier as SklearnGBC
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid function.

    Parameters
    ----------
    z : np.ndarray

    Returns
    -------
    np.ndarray — values in (0, 1)

    TODO:
        - Clip z to [-500, 500] before computing exp to avoid overflow.
        - Return 1 / (1 + np.exp(-z)).
    """
    # TODO: implement numerically stable sigmoid
    raise NotImplementedError("Implement sigmoid()")


def compute_residuals(y_true: np.ndarray, y_pred_proba: np.ndarray) -> np.ndarray:
    """
    Compute pseudo-residuals (negative gradient of binary log-loss).

    For log-loss, the pseudo-residual at iteration m is simply:
        r = y_true - y_pred_proba

    This is the "error signal" that the next tree will try to correct.

    Parameters
    ----------
    y_true       : np.ndarray, shape (n,) — true binary labels (0 or 1)
    y_pred_proba : np.ndarray, shape (n,) — current predicted probabilities

    Returns
    -------
    np.ndarray, shape (n,) — residuals (error signal for the next tree)

    TODO:
        - Return y_true - y_pred_proba element-wise.
    """
    # TODO: compute and return pseudo-residuals
    raise NotImplementedError("Implement compute_residuals()")


# ---------------------------------------------------------------------------
# Gradient Boosting Classifier
# ---------------------------------------------------------------------------

class GradientBoostingClassifier:
    """
    Binary gradient boosting classifier trained by sequential residual fitting.

    Parameters
    ----------
    n_estimators  : int   — number of boosting rounds (trees)
    learning_rate : float — shrinkage factor applied to each tree's contribution
    max_depth     : int   — depth of each shallow regression tree
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

        self.trees_: list[DecisionTreeRegressor] = []
        self.F0_: float = 0.0          # initial log-odds
        self.train_loss_: list[float] = []  # log-loss per iteration

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingClassifier":
        """
        Train the gradient boosting model.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,) — binary labels (0 or 1)

        Returns
        -------
        self

        TODO:
            1. Compute base rate: p = mean(y).
               Initialize F0 = log(p / (1 - p))  [log-odds].
               Store as self.F0_.
               Initialize F = np.full(n_samples, self.F0_).

            2. For m in range(n_estimators):
               a. Convert current scores to probabilities: p_hat = sigmoid(F).
               b. Compute residuals: r = compute_residuals(y, p_hat).
               c. Fit a DecisionTreeRegressor(max_depth=self.max_depth, random_state=m)
                  on (X, r). This tree predicts the error signal.
               d. Update: F += self.learning_rate * tree.predict(X).
               e. Compute and store training log-loss:
                      loss = log_loss(y, sigmoid(F))
                  Append to self.train_loss_.
               f. Append the tree to self.trees_.

            3. Return self.

        Key insight:
            - A small learning_rate requires more trees but generalizes better
              (it's a regularization parameter — shrinkage).
            - max_depth=3 is common; deeper trees lead to faster overfitting.
        """
        # TODO: implement the gradient boosting training loop
        raise NotImplementedError("Implement GradientBoostingClassifier.fit()")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute predicted probability of class 1 for each sample.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,) — P(y=1 | x)

        TODO:
            1. Start with F = np.full(n_samples, self.F0_).
            2. For each tree in self.trees_:
                   F += self.learning_rate * tree.predict(X)
            3. Return sigmoid(F).
        """
        # TODO: aggregate tree predictions to produce probabilities
        raise NotImplementedError("Implement predict_proba()")

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary class labels.

        Parameters
        ----------
        X         : np.ndarray, shape (n_samples, n_features)
        threshold : float

        Returns
        -------
        np.ndarray, shape (n_samples,) — predicted labels (0 or 1)

        TODO:
            - Call predict_proba(X).
            - Return (proba >= threshold).astype(int).
        """
        # TODO: threshold probabilities into class labels
        raise NotImplementedError("Implement predict()")


# ---------------------------------------------------------------------------
# Effect of Tree Depth
# ---------------------------------------------------------------------------

def compare_boosting_depths(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    depths: list[int] = [1, 2, 3, 5],
) -> None:
    """
    Compare gradient boosting with different max_depth values.

    A depth-1 tree (stump) is the weakest learner; it only learns one rule.
    Deeper trees learn more complex patterns per round, but can overfit faster.

    Parameters
    ----------
    X_train, X_test : np.ndarray
    y_train, y_test : np.ndarray
    depths          : list[int] — max_depth values to compare

    TODO:
        - For each depth in depths:
            1. Train your GradientBoostingClassifier(max_depth=depth, n_estimators=200).
            2. Record test accuracy.
            3. Plot training loss over iterations.
        - Plot all training loss curves on one figure (one line per depth).
        - Print a summary table: depth | test accuracy | final training loss.
        - Save the figure as "gbm_depth_comparison.png".
    """
    # TODO: compare training dynamics and accuracy for different tree depths
    raise NotImplementedError("Implement compare_boosting_depths()")


# ---------------------------------------------------------------------------
# Feature Importance Plot (provided — do not modify)
# ---------------------------------------------------------------------------

def feature_importance_plot(
    trees: list[DecisionTreeRegressor],
    feature_names: list[str],
    title: str = "Gradient Boosting Feature Importances",
) -> None:
    """
    Plot aggregate feature importances by averaging across all trees.

    Parameters
    ----------
    trees         : list[DecisionTreeRegressor] — trained base learners
    feature_names : list[str]
    title         : str
    """
    if not trees:
        print("No trees found. Train the model first.")
        return

    importances = np.mean(
        [tree.feature_importances_ for tree in trees], axis=0
    )
    importances /= importances.sum()  # normalize
    sorted_idx = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 4))
    plt.bar(range(len(importances)), importances[sorted_idx], color="steelblue")
    plt.xticks(
        range(len(importances)),
        [feature_names[i] for i in sorted_idx],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    plt.title(title)
    plt.ylabel("Mean Importance (normalized)")
    plt.tight_layout()
    plt.savefig("gbm_feature_importances.png", dpi=120)
    plt.show()
    print("Saved: gbm_feature_importances.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Full gradient boosting demonstration:
        1. Generate and split data.
        2. Train your GBM and compare with sklearn.
        3. Plot training loss curve.
        4. Compare tree depths.
        5. Plot feature importances.
    """
    np.random.seed(42)

    X, y = make_classification(
        n_samples=800,
        n_features=15,
        n_informative=8,
        n_redundant=3,
        random_state=42,
    )
    feature_names = [f"Feature {i}" for i in range(X.shape[1])]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("=== Custom Gradient Boosting Classifier ===")
    gbm = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3)
    gbm.fit(X_train, y_train)
    preds = gbm.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Custom GBM Test Accuracy: {acc:.4f}")

    # Plot training loss
    plt.figure(figsize=(8, 4))
    plt.plot(gbm.train_loss_, color="darkorange", linewidth=2)
    plt.xlabel("Boosting Round")
    plt.ylabel("Log-Loss (training)")
    plt.title("Gradient Boosting — Training Loss per Round")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("gbm_train_loss.png", dpi=120)
    plt.show()

    print("\n=== Sklearn GradientBoostingClassifier ===")
    sk_gbm = SklearnGBC(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
    sk_gbm.fit(X_train, y_train)
    sk_acc = accuracy_score(y_test, sk_gbm.predict(X_test))
    print(f"Sklearn GBM Test Accuracy: {sk_acc:.4f}")

    print("\n=== Comparing Tree Depths ===")
    compare_boosting_depths(X_train, X_test, y_train, y_test, depths=[1, 2, 3, 5])

    print("\n=== Feature Importances ===")
    feature_importance_plot(gbm.trees_, feature_names)


if __name__ == "__main__":
    main()
