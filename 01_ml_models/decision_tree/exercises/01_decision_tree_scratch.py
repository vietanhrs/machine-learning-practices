"""
Exercise 01: Decision Tree Classifier from Scratch
===================================================

Goal: Implement a binary decision tree classifier using only NumPy,
      then compare it with sklearn's implementation.

Learning Objectives:
    - Implement Gini impurity and entropy from first principles.
    - Implement Information Gain and find the best split at each node.
    - Build the tree recursively using a Node data structure.
    - Traverse the tree to make predictions.

Instructions:
    - Fill in every section marked with TODO.
    - Do NOT use sklearn's DecisionTreeClassifier inside your implementation.
    - You MAY use sklearn in main() to generate data and compare results.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Any
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier as SklearnDT


# ---------------------------------------------------------------------------
# Impurity Functions
# ---------------------------------------------------------------------------

def gini_impurity(y: np.ndarray) -> float:
    """
    Compute the Gini impurity of a set of labels.

    Gini(S) = 1 - Σ_c p_c²

    where p_c = proportion of class c in S.

    Parameters
    ----------
    y : np.ndarray, shape (n,) — integer class labels

    Returns
    -------
    float — Gini impurity in [0, 0.5] for binary classification

    TODO:
        - If y is empty, return 0.0.
        - Compute the proportion of each class: p_c = count(c) / len(y).
        - Return 1 - sum(p_c²).
        Hint: np.unique with return_counts=True gives you class frequencies.
    """
    # TODO: implement Gini impurity
    raise NotImplementedError("Implement gini_impurity()")


def entropy(y: np.ndarray) -> float:
    """
    Compute the Shannon entropy of a set of labels.

    H(S) = -Σ_c p_c · log₂(p_c)

    Parameters
    ----------
    y : np.ndarray, shape (n,)

    Returns
    -------
    float — entropy in [0, log₂(n_classes)]

    TODO:
        - If y is empty, return 0.0.
        - Compute class proportions p_c.
        - Return -sum(p_c * log2(p_c)), skipping any p_c = 0 to avoid log(0).
        Hint: np.log2, filter out zeros with p[p > 0].
    """
    # TODO: implement entropy
    raise NotImplementedError("Implement entropy()")


def information_gain(
    y: np.ndarray,
    y_left: np.ndarray,
    y_right: np.ndarray,
    criterion: str = "gini",
) -> float:
    """
    Compute the Information Gain from splitting y into y_left and y_right.

    IG = impurity(y) - [|y_L|/|y| · impurity(y_L) + |y_R|/|y| · impurity(y_R)]

    Parameters
    ----------
    y        : np.ndarray — labels before the split (parent node)
    y_left   : np.ndarray — labels in the left child
    y_right  : np.ndarray — labels in the right child
    criterion: str — 'gini' or 'entropy'

    Returns
    -------
    float — Information Gain (always >= 0)

    TODO:
        - Select the impurity function based on criterion ('gini' → gini_impurity,
          'entropy' → entropy).
        - Compute parent impurity.
        - Compute the weighted average of child impurities.
        - Return parent - weighted_children.
    """
    # TODO: implement information gain using weighted child impurity
    raise NotImplementedError("Implement information_gain()")


# ---------------------------------------------------------------------------
# Best Split Finder
# ---------------------------------------------------------------------------

def best_split(
    X: np.ndarray,
    y: np.ndarray,
    criterion: str = "gini",
) -> tuple[int, float, float]:
    """
    Search all features and thresholds to find the split with highest IG.

    Parameters
    ----------
    X         : np.ndarray, shape (n_samples, n_features)
    y         : np.ndarray, shape (n_samples,)
    criterion : str — 'gini' or 'entropy'

    Returns
    -------
    (best_feature_index, best_threshold, best_ig) : tuple

    TODO:
        - Initialize best_ig = -1, best_feature = None, best_threshold = None.
        - For each feature index f in range(n_features):
            * Get the column X[:, f].
            * Find candidate thresholds: the unique sorted midpoints between
              consecutive distinct values.
              E.g., for values [1, 2, 4] → thresholds [1.5, 3.0].
              Hint: sort unique values, take (v[i] + v[i+1]) / 2 for each i.
            * For each threshold t:
                - Split: left where X[:,f] <= t, right where X[:,f] > t.
                - Skip if either split is empty.
                - Compute information_gain(y, y_left, y_right, criterion).
                - Update best if this IG is higher.
        - Return (best_feature, best_threshold, best_ig).
    """
    # TODO: exhaustive search for the best (feature, threshold) split
    raise NotImplementedError("Implement best_split()")


# ---------------------------------------------------------------------------
# Tree Node
# ---------------------------------------------------------------------------

@dataclass
class Node:
    """
    A single node in the decision tree.

    Attributes
    ----------
    feature_index : int | None   — feature index used for this split (None for leaves)
    threshold     : float | None — split threshold (None for leaves)
    left          : Node | None  — left subtree (X[f] <= threshold)
    right         : Node | None  — right subtree (X[f] > threshold)
    value         : Any          — predicted class label (only for leaf nodes)
    """
    feature_index: int | None = None
    threshold: float | None = None
    left: Any = None   # type: "Node | None"
    right: Any = None  # type: "Node | None"
    value: Any = None


# ---------------------------------------------------------------------------
# Decision Tree Classifier
# ---------------------------------------------------------------------------

class DecisionTreeClassifier:
    """
    Binary/multiclass decision tree classifier built by recursive splitting.

    Parameters
    ----------
    max_depth         : int  — maximum tree depth (None = unlimited)
    min_samples_split : int  — minimum samples needed to split a node
    criterion         : str  — 'gini' or 'entropy'
    """

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        criterion: str = "gini",
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root: Node | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        """
        Grow the decision tree.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        self

        TODO:
            - Call self._build_tree(X, y, depth=0) and store the result in self.root.
            - Return self.
        """
        # TODO: initiate recursive tree building, store root
        raise NotImplementedError("Implement fit()")

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        """
        Recursively build the tree by finding the best split and branching.

        Parameters
        ----------
        X     : np.ndarray — feature matrix at the current node
        y     : np.ndarray — labels at the current node
        depth : int        — current depth in the tree

        Returns
        -------
        Node — root of the subtree built from (X, y)

        TODO:
            Stopping conditions (return a leaf node):
                1. All labels in y are the same class.
                2. n_samples < min_samples_split.
                3. depth >= max_depth (if max_depth is set).
                4. best_split() returns IG = 0 (no beneficial split found).

            For a leaf node:
                - Set node.value = the most frequent class in y.
                  Hint: np.bincount(y).argmax() for integer labels.

            For an internal node:
                - Call best_split(X, y, self.criterion).
                - Split X and y on the best (feature, threshold).
                - Recursively call _build_tree on left and right subsets.
                - Return a Node(feature_index=..., threshold=..., left=..., right=...).
        """
        # TODO: implement recursive tree building with stopping conditions
        raise NotImplementedError("Implement _build_tree()")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for all samples in X.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,) — predicted integer class labels

        TODO:
            - Apply _traverse_tree(x, self.root) to each row of X.
            - Return the result as a NumPy array.
        """
        # TODO: predict labels for all rows of X
        raise NotImplementedError("Implement predict()")

    def _traverse_tree(self, x: np.ndarray, node: Node) -> int:
        """
        Traverse the tree from the given node and return a leaf prediction.

        Parameters
        ----------
        x    : np.ndarray, shape (n_features,) — single sample
        node : Node — current node

        Returns
        -------
        int — predicted class label

        TODO:
            - If node.value is not None → it is a leaf; return node.value.
            - Otherwise, compare x[node.feature_index] with node.threshold:
                * If <= threshold → recurse on node.left
                * Else            → recurse on node.right
        """
        # TODO: implement recursive tree traversal for a single sample
        raise NotImplementedError("Implement _traverse_tree()")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Train your DecisionTreeClassifier and compare with sklearn's.

    Steps:
        1. Generate a synthetic binary classification dataset.
        2. Split into train/test.
        3. Fit your tree (max_depth=5).
        4. Fit sklearn's DecisionTreeClassifier(max_depth=5).
        5. Compare test accuracy.
        6. Print a brief summary.

    Expected: your implementation should achieve accuracy within ~1-2% of sklearn.
    """
    np.random.seed(42)

    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training custom DecisionTreeClassifier ...")
    custom_tree = DecisionTreeClassifier(max_depth=5, criterion="gini")
    custom_tree.fit(X_train, y_train)
    custom_preds = custom_tree.predict(X_test)
    custom_acc = accuracy_score(y_test, custom_preds)
    print(f"Custom Tree Test Accuracy : {custom_acc:.4f}")

    print("\nTraining sklearn DecisionTreeClassifier ...")
    sklearn_tree = SklearnDT(max_depth=5, criterion="gini", random_state=42)
    sklearn_tree.fit(X_train, y_train)
    sklearn_preds = sklearn_tree.predict(X_test)
    sklearn_acc = accuracy_score(y_test, sklearn_preds)
    print(f"Sklearn Tree Test Accuracy: {sklearn_acc:.4f}")

    diff = abs(custom_acc - sklearn_acc)
    print(f"\nAccuracy difference: {diff:.4f}")
    if diff < 0.03:
        print("Your implementation matches sklearn closely!")
    else:
        print("There may be a bug — recheck your splitting logic.")


if __name__ == "__main__":
    main()
