"""
Exercise 02: Isolation Forest from Scratch
==========================================
Implement the Isolation Forest anomaly detection algorithm from scratch,
then compare it with scikit-learn's implementation.

Learning Goals:
    - Understand how random isolation trees are built
    - Implement path-length-based anomaly scoring
    - Understand the normalization by expected BST path length c(n)
    - Compare custom implementation with sklearn's IsolationForest

Reference:
    Liu, Fei Tony, Ting, Kai Ming, and Zhou, Zhi-Hua. 2008.
    "Isolation Forest." In Proceedings of ICDM.
    https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf

Requirements:
    pip install numpy scikit-learn matplotlib
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest as SKLearnIsolationForest
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Tree Node
# ---------------------------------------------------------------------------

@dataclass
class IsolationTreeNode:
    """
    A node in an Isolation Tree.

    Attributes:
        is_leaf:      True if this is a terminal node (sample isolated).
        size:         Number of samples at this node.
        split_feature: Index of the feature used for splitting (internal nodes only).
        split_value:   Threshold for the split (internal nodes only).
        left:         Left child node (samples where feature <= split_value).
        right:        Right child node (samples where feature > split_value).
    """
    is_leaf: bool = False
    size: int = 0
    split_feature: Optional[int] = None
    split_value: Optional[float] = None
    left: Optional["IsolationTreeNode"] = None
    right: Optional["IsolationTreeNode"] = None


# ---------------------------------------------------------------------------
# 1. Isolation Tree
# ---------------------------------------------------------------------------

class IsolationTree:
    """
    A single isolation tree: a random binary tree that isolates samples.
    """

    def __init__(self) -> None:
        self.root: Optional[IsolationTreeNode] = None

    def fit(self, X: np.ndarray, max_depth: int) -> "IsolationTree":
        """
        Recursively build the isolation tree on subsample X.

        Args:
            X:         2D array of shape (n_samples, n_features).
            max_depth: Maximum depth of the tree (stops early if reached).

        Returns:
            self.

        TODO:
            1. Call self._build(X, current_depth=0, max_depth=max_depth).
            2. Store the result in self.root.
            3. Return self.

        _build(X, current_depth, max_depth) → IsolationTreeNode:
            Base cases (return a leaf node):
                - len(X) <= 1: only one sample left → isolated.
                - current_depth >= max_depth: tree is too deep, stop.
                - All samples are identical (no valid split exists).
            Recursive case:
                1. Randomly select a feature index q (0 to n_features-1).
                2. Randomly select a split value p uniformly between
                   X[:, q].min() and X[:, q].max().
                3. Split: X_left = X[X[:, q] <= p], X_right = X[X[:, q] > p].
                4. If X_left or X_right is empty, return a leaf.
                5. Recursively build left and right children.
                6. Return an IsolationTreeNode with the split info.

        Hint:
            Create a helper method _build(self, X, depth, max_depth) → IsolationTreeNode.
        """
        raise NotImplementedError("TODO: implement IsolationTree.fit()")

    def path_length(self, x: np.ndarray, node: IsolationTreeNode, depth: int) -> float:
        """
        Compute the path length for a single sample x in the tree.

        The path length is the number of edges traversed from root to a leaf
        (or early termination) + an adjustment for the size of the leaf node
        (to account for the tree being cut short at max_depth).

        Args:
            x:     1D array of shape (n_features,) — a single sample.
            node:  Current tree node.
            depth: Current depth in the tree.

        Returns:
            Estimated path length (float).

        TODO:
            1. If node is a leaf or node.size <= 1:
                 return depth + 0  (just the depth reached)
            2. If node.size > 1 and is leaf (cut short by max_depth):
                 return depth + c(node.size)
                 where c(n) is the expected path length of a BST with n nodes
                 (see _expected_path_length() helper below).
            3. Otherwise (internal node):
                 If x[node.split_feature] <= node.split_value:
                     recurse on left child with depth+1
                 Else:
                     recurse on right child with depth+1.

        Hint:
            Leaf nodes created due to max_depth have size > 1.
            Leaf nodes created by isolating 1 sample have size == 1.
        """
        raise NotImplementedError("TODO: implement IsolationTree.path_length()")

    @staticmethod
    def _expected_path_length(n: int) -> float:
        """
        Compute the expected path length of an unsuccessful search in a
        Binary Search Tree (BST) with n nodes.

        Formula:
            c(n) = 2 * H(n-1) - 2*(n-1)/n    for n >= 2
            c(1) = 0
            c(0) = 0

        where H(n) = ln(n) + 0.5772156649 (Euler-Mascheroni constant)

        This is used to normalize anomaly scores.
        """
        if n <= 1:
            return 0.0
        EULER_MASCHERONI = 0.5772156649
        harmonic_n_minus_1 = math.log(n - 1) + EULER_MASCHERONI
        return 2.0 * harmonic_n_minus_1 - 2.0 * (n - 1) / n


# ---------------------------------------------------------------------------
# 2. Isolation Forest
# ---------------------------------------------------------------------------

class IsolationForest:
    """
    Isolation Forest: ensemble of Isolation Trees for anomaly detection.

    Args:
        n_trees:     Number of isolation trees to build.
        sample_size: Subsample size used to build each tree.
        random_seed: Seed for reproducibility.
    """

    def __init__(
        self,
        n_trees: int = 100,
        sample_size: int = 256,
        random_seed: int = 42,
    ) -> None:
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.random_seed = random_seed
        self.trees: List[IsolationTree] = []
        self.n_samples: int = 0

    def fit(self, X: np.ndarray) -> "IsolationForest":
        """
        Build an ensemble of isolation trees on the training data.

        Args:
            X: 2D array of shape (n_samples, n_features).

        Returns:
            self.

        TODO:
            1. Set self.n_samples = len(X).
            2. Set self.max_depth = ceil(log2(min(sample_size, n_samples))).
               This limits tree depth for efficiency; trees beyond this depth
               contribute little additional isolation information.
            3. Set random seeds: random.seed(self.random_seed), np.random.seed(...).
            4. For each tree (0 to n_trees - 1):
                 a. Draw a random subsample of size min(sample_size, n_samples)
                    WITHOUT replacement: idx = np.random.choice(n_samples, size, replace=False).
                 b. Build an IsolationTree on X[idx] with max_depth.
                 c. Append to self.trees.
            5. Return self.
        """
        raise NotImplementedError("TODO: implement IsolationForest.fit()")

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the anomaly score for each sample in X.

        Anomaly score: s(x, n) = 2^(-E[h(x)] / c(n))

        Where:
            E[h(x)] = average path length across all trees for sample x
            c(n)    = expected path length for a BST with n nodes (= self.n_samples)

        Args:
            X: 2D array of shape (n_test_samples, n_features).

        Returns:
            1D array of anomaly scores in [0, 1].
            - Scores close to 1 → very likely anomalous
            - Scores around 0.5 → normal
            - Scores close to 0 → not anomalous at all

        TODO:
            1. For each sample x in X:
                 a. Compute average path length across all trees:
                    avg_path = mean([tree.path_length(x, tree.root, depth=0)
                                     for tree in self.trees])
                 b. Compute score = 2^(-avg_path / c(self.n_samples))
                    where c(n) = IsolationTree._expected_path_length(n).
            2. Return np.array of scores.
        """
        raise NotImplementedError("TODO: implement IsolationForest.anomaly_score()")

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict whether each sample is an anomaly or normal.

        Args:
            X:         2D array of shape (n_samples, n_features).
            threshold: Anomaly score threshold. Scores above this → anomaly.

        Returns:
            1D array of predictions:
                -1 = anomaly
                 1 = normal

        TODO:
            1. Compute scores = self.anomaly_score(X).
            2. Return np.where(scores > threshold, -1, 1).
        """
        raise NotImplementedError("TODO: implement IsolationForest.predict()")


# ---------------------------------------------------------------------------
# 3. Compare with sklearn
# ---------------------------------------------------------------------------

def compare_with_sklearn(
    X: np.ndarray,
    y_true: np.ndarray,
    contamination: float = 0.1,
) -> None:
    """
    Compare custom Isolation Forest with sklearn's implementation.

    Args:
        X:             2D array of shape (n_samples, n_features).
        y_true:        Boolean array. True = anomaly.
        contamination: Expected proportion of anomalies.

    TODO:
        1. Train custom IsolationForest on X.
        2. Compute anomaly scores from custom model.
        3. Train sklearn IsolationForest on X with the same contamination.
        4. Compute anomaly scores from sklearn (use decision_function or score_samples).
        5. Compute AUC-ROC for both using roc_auc_score(y_true, scores).
        6. Print comparison table.
        7. Plot score distributions for normal vs anomalous samples from both models.

    Hint:
        sklearn's score_samples() returns negative scores where more negative = more anomalous.
        Negate them for consistent comparison: -clf.score_samples(X).
    """
    raise NotImplementedError("TODO: implement compare_with_sklearn()")


# ---------------------------------------------------------------------------
# Helper: Generate synthetic 2D dataset with anomalies
# ---------------------------------------------------------------------------

def generate_dataset(
    n_normal: int = 300,
    n_anomalies: int = 30,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a 2D dataset with two Gaussian clusters (normal) and
    uniformly scattered anomalies.

    Returns:
        X:       (n_normal + n_anomalies, 2) array.
        y_true:  Boolean array. True = anomaly.
    """
    rng = np.random.default_rng(seed)

    # Normal data: two Gaussian clusters
    X1 = rng.multivariate_normal([2, 2], [[0.5, 0.2], [0.2, 0.5]], n_normal // 2)
    X2 = rng.multivariate_normal([-2, -2], [[0.5, -0.1], [-0.1, 0.5]], n_normal - n_normal // 2)
    X_normal = np.vstack([X1, X2])

    # Anomalies: uniform scatter in outer region
    X_anomaly = rng.uniform(low=-8, high=8, size=(n_anomalies, 2))
    # Remove points accidentally close to the clusters
    dist1 = np.linalg.norm(X_anomaly - [2, 2], axis=1)
    dist2 = np.linalg.norm(X_anomaly - [-2, -2], axis=1)
    X_anomaly = X_anomaly[(dist1 > 3) & (dist2 > 3)]
    n_actual_anomalies = len(X_anomaly)

    X = np.vstack([X_normal, X_anomaly])
    y_true = np.zeros(len(X), dtype=bool)
    y_true[n_normal:] = True

    idx = rng.permutation(len(X))
    return X[idx], y_true[idx]


def visualize_anomaly_scores(
    X: np.ndarray,
    scores: np.ndarray,
    y_true: np.ndarray,
    title: str = "Isolation Forest Anomaly Scores",
) -> None:
    """
    Scatter plot colored by anomaly score.

    Args:
        X:      2D array of shape (n_samples, 2).
        scores: 1D anomaly score array.
        y_true: Boolean array of true labels (for overlay).
        title:  Plot title.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: colored by anomaly score
    sc = axes[0].scatter(X[:, 0], X[:, 1], c=scores, cmap="RdYlGn_r", s=30, alpha=0.8)
    plt.colorbar(sc, ax=axes[0], label="Anomaly Score")
    axes[0].set_title(f"{title}\n(Color = score; red = anomalous)")

    # Right: colored by ground truth
    axes[1].scatter(X[~y_true, 0], X[~y_true, 1], c="steelblue", s=30, alpha=0.6, label="Normal")
    axes[1].scatter(X[y_true, 0], X[y_true, 1], c="red", s=70, marker="x", label="True Anomaly")
    axes[1].set_title("Ground Truth Labels")
    axes[1].legend()

    plt.tight_layout()
    filename = title.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".png"
    plt.savefig(filename, dpi=100)
    plt.show()
    print(f"[Plot saved: {filename}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Demonstrate custom Isolation Forest on 2D data with anomalies.

    Steps:
        1. Generate dataset with two normal clusters + scattered anomalies.
        2. Train custom Isolation Forest.
        3. Compute anomaly scores and visualize.
        4. Compare with sklearn's IsolationForest.
    """
    print("=" * 60)
    print("Isolation Forest — Anomaly Detection")
    print("=" * 60)

    X, y_true = generate_dataset(n_normal=300, n_anomalies=30, seed=42)
    print(f"\nDataset: {len(X)} samples | {y_true.sum()} true anomalies")

    # --- Custom Isolation Forest ---
    print("\nTraining custom Isolation Forest...")
    clf = IsolationForest(n_trees=100, sample_size=256, random_seed=42)
    clf.fit(X)

    scores = clf.anomaly_score(X)
    predictions = clf.predict(X, threshold=0.6)

    auc = roc_auc_score(y_true.astype(int), scores)
    print(f"Custom IF AUC-ROC: {auc:.4f}")

    n_detected = (predictions == -1).sum()
    print(f"Flagged as anomaly: {n_detected} (true: {y_true.sum()})")

    visualize_anomaly_scores(X, scores, y_true, title="Custom Isolation Forest")

    # --- Comparison ---
    print("\nComparing with sklearn IsolationForest...")
    compare_with_sklearn(X, y_true, contamination=0.1)


if __name__ == "__main__":
    main()
