"""
Exercise 02: Random Forest from Scratch
========================================

Goal: Implement a Random Forest classifier by composing bootstrap sampling
      with multiple decision trees, then aggregate predictions via majority vote.

Learning Objectives:
    - Understand bootstrap sampling (sampling with replacement).
    - See how diversity among trees (different bootstrap samples + feature subsets)
      reduces variance compared to a single tree.
    - Implement majority vote aggregation.
    - Compute aggregate feature importances from the ensemble.

Instructions:
    - Fill in every section marked with TODO.
    - You may use sklearn's DecisionTreeClassifier as the base tree
      (no need to re-implement it from Exercise 01 here).
    - Do NOT use sklearn's RandomForestClassifier inside your RandomForest class.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier as SklearnRF


# ---------------------------------------------------------------------------
# Bootstrap Sampling
# ---------------------------------------------------------------------------

def bootstrap_sample(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Draw a bootstrap sample (random sample with replacement) from (X, y).

    The bootstrap sample has the same number of rows as the original dataset
    but is sampled WITH replacement, so some rows appear multiple times and
    others not at all (on average, ~63.2% of unique rows are included).

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)

    Returns
    -------
    (X_boot, y_boot) : tuple of np.ndarray — bootstrap sample

    TODO:
        - Use np.random.choice(n_samples, size=n_samples, replace=True)
          to generate random indices.
        - Index X and y with those indices and return.
    """
    # TODO: sample n_samples indices with replacement, return X[indices], y[indices]
    raise NotImplementedError("Implement bootstrap_sample()")


# ---------------------------------------------------------------------------
# Random Forest Classifier
# ---------------------------------------------------------------------------

class RandomForestClassifier:
    """
    Random Forest ensemble of decision trees.

    Each tree is trained on a bootstrap sample of the data.
    At each split, only max_features randomly chosen features are considered.

    Parameters
    ----------
    n_estimators : int   — number of trees in the forest
    max_depth    : int   — maximum depth of each tree (None = unlimited)
    max_features : str | int | None — number of features per split:
                         'sqrt' → √n_features (default for classification)
                         'log2' → log₂(n_features)
                         int    → fixed number
                         None   → all features
    random_state : int   — seed for reproducibility
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        max_features: str | int | None = "sqrt",
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.trees_: list[DecisionTreeClassifier] = []
        self.n_classes_: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        """
        Train the Random Forest.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        self

        TODO:
            1. Set random seed if provided (np.random.seed).
            2. Resolve max_features:
               - 'sqrt' → int(np.sqrt(n_features))
               - 'log2' → int(np.log2(n_features))
               - int    → use as-is
               - None   → n_features
            3. For each of n_estimators trees:
               a. Draw a bootstrap sample using bootstrap_sample().
               b. Instantiate DecisionTreeClassifier(
                      max_depth=self.max_depth,
                      max_features=resolved_max_features,
                      random_state=i  # different seed per tree
                  ).
               c. Fit the tree on the bootstrap sample.
               d. Append the tree to self.trees_.
            4. Store self.n_classes_ = len(np.unique(y)).
            5. Return self.
        """
        # TODO: train n_estimators trees on bootstrap samples
        raise NotImplementedError("Implement RandomForestClassifier.fit()")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels via majority vote across all trees.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,) — majority vote predictions

        TODO:
            1. Collect predictions from each tree: shape (n_estimators, n_samples).
               Hint: np.array([tree.predict(X) for tree in self.trees_])
            2. For each sample, find the majority class.
               Hint: iterate over columns or use scipy.stats.mode.
               Alternative: build a vote count matrix of shape
                            (n_samples, n_classes) and take argmax per row.
            3. Return the majority labels as a 1-D array.
        """
        # TODO: aggregate tree predictions via majority vote
        raise NotImplementedError("Implement predict()")

    def feature_importances_(self, X: np.ndarray) -> np.ndarray:
        """
        Compute mean feature importances across all trees.

        Parameters
        ----------
        X : np.ndarray — used only to infer n_features

        Returns
        -------
        np.ndarray, shape (n_features,) — normalized importances summing to 1

        TODO:
            - For each tree, retrieve tree.feature_importances_ (provided by sklearn's DT).
            - Average across all trees.
            - Normalize so they sum to 1 (they should already, but dividing by
              their sum makes it robust).
        """
        # TODO: average and normalize feature importances from all trees
        raise NotImplementedError("Implement feature_importances_()")


# ---------------------------------------------------------------------------
# Variance Reduction Demonstration
# ---------------------------------------------------------------------------

def compare_tree_vs_forest(
    X: np.ndarray,
    y: np.ndarray,
    n_repeats: int = 20,
    n_estimators: int = 100,
) -> None:
    """
    Demonstrate that Random Forest reduces variance vs. a single tree by
    repeatedly training both models on random train/test splits and plotting
    the distribution of test accuracies.

    Parameters
    ----------
    X            : np.ndarray
    y            : np.ndarray
    n_repeats    : int — number of independent train/test splits to run
    n_estimators : int — number of trees in the forest

    TODO:
        - Run n_repeats independent experiments:
            * Generate a random 80/20 train/test split.
            * Train a single DecisionTreeClassifier (max_depth=None) and record accuracy.
            * Train your RandomForestClassifier (n_estimators=n_estimators, max_depth=None)
              and record accuracy.
        - After all experiments:
            * Print mean and std of accuracies for both models.
            * Plot two histograms side by side (or overlapping):
                - Single tree accuracy distribution
                - Random Forest accuracy distribution
            * The forest histogram should be tighter (lower std = lower variance).
        - Save the figure as "tree_vs_forest_variance.png".
    """
    # TODO: run n_repeats experiments, compare accuracy distributions
    raise NotImplementedError("Implement compare_tree_vs_forest()")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Full Random Forest demonstration:
        1. Generate a synthetic classification dataset.
        2. Train your RandomForestClassifier.
        3. Compare accuracy with sklearn's RandomForestClassifier.
        4. Plot feature importances.
        5. Run variance comparison experiment.
    """
    np.random.seed(42)

    X, y = make_classification(
        n_samples=600,
        n_features=15,
        n_informative=8,
        n_redundant=3,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("=== Training Custom Random Forest ===")
    rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Custom RF Test Accuracy: {acc:.4f}")

    print("\n=== Training Sklearn Random Forest ===")
    skrf = SklearnRF(n_estimators=100, max_depth=None, random_state=42)
    skrf.fit(X_train, y_train)
    sk_preds = skrf.predict(X_test)
    sk_acc = accuracy_score(y_test, sk_preds)
    print(f"Sklearn RF Test Accuracy: {sk_acc:.4f}")

    print("\n=== Feature Importances ===")
    importances = rf.feature_importances_(X_train)
    feature_names = [f"Feature {i}" for i in range(X.shape[1])]
    sorted_idx = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 4))
    plt.bar(range(len(importances)), importances[sorted_idx])
    plt.xticks(range(len(importances)), [feature_names[i] for i in sorted_idx], rotation=45, ha="right")
    plt.title("Random Forest — Feature Importances")
    plt.tight_layout()
    plt.savefig("rf_feature_importances.png", dpi=120)
    plt.show()

    print("\n=== Variance Reduction Experiment ===")
    compare_tree_vs_forest(X, y, n_repeats=30, n_estimators=100)


if __name__ == "__main__":
    main()
