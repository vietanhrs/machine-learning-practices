"""
Exercise 01: K-Means Clustering from Scratch
=============================================

Goal: Implement the K-Means algorithm using only NumPy.

Learning Objectives:
    - Understand the iterative assign-update loop that drives K-Means.
    - Implement Euclidean distance, centroid initialization, cluster assignment,
      and centroid update from first principles.
    - Visualize the resulting clusters and compute inertia.

Instructions:
    - Fill in every section marked with TODO.
    - Do NOT use sklearn's KMeans inside your implementation.
    - You MAY use sklearn only in main() to generate synthetic data and to
      cross-check your inertia value.
    - Run the script and verify that your clusters look sensible on the plot.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


# ---------------------------------------------------------------------------
# Helper Function
# ---------------------------------------------------------------------------

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the Euclidean distance between two 1-D vectors a and b.

    Parameters
    ----------
    a : np.ndarray, shape (d,)
    b : np.ndarray, shape (d,)

    Returns
    -------
    float
        The straight-line distance ||a - b||.

    TODO:
        - Compute the element-wise difference between a and b.
        - Square each element.
        - Sum them.
        - Return the square root of the sum.
        Hint: np.sqrt and np.sum are your friends.
    """
    # TODO: implement Euclidean distance
    raise NotImplementedError("Implement euclidean_distance()")


# ---------------------------------------------------------------------------
# Step 1 – Initialize Centroids
# ---------------------------------------------------------------------------

def initialize_centroids(X: np.ndarray, k: int) -> np.ndarray:
    """
    Randomly select k data points from X as the initial centroids.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    k : int
        Number of clusters.

    Returns
    -------
    np.ndarray, shape (k, n_features)
        The initial centroid positions.

    TODO:
        - Use np.random.choice (without replacement) to pick k row indices.
        - Return those rows of X.
        Hint: You want k unique rows from X.
    """
    # TODO: randomly sample k rows from X (no replacement)
    raise NotImplementedError("Implement initialize_centroids()")


# ---------------------------------------------------------------------------
# Step 2 – Assign Each Point to the Nearest Centroid
# ---------------------------------------------------------------------------

def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Assign each data point to the nearest centroid.

    Parameters
    ----------
    X         : np.ndarray, shape (n_samples, n_features)
    centroids : np.ndarray, shape (k, n_features)

    Returns
    -------
    np.ndarray, shape (n_samples,)
        Integer array where labels[i] is the index of the nearest centroid
        for point X[i].

    TODO:
        - For each point in X, compute its distance to every centroid.
        - Assign the point to the index of the closest centroid.
        Hint: Use euclidean_distance() in a loop, or use np.linalg.norm
              with keepdims / broadcasting for a vectorised approach.
              np.argmin is useful for picking the nearest centroid.
    """
    # TODO: compute distances from each point to each centroid and assign labels
    raise NotImplementedError("Implement assign_clusters()")


# ---------------------------------------------------------------------------
# Step 3 – Update Centroids
# ---------------------------------------------------------------------------

def update_centroids(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """
    Recompute each centroid as the mean of points assigned to that cluster.

    Parameters
    ----------
    X      : np.ndarray, shape (n_samples, n_features)
    labels : np.ndarray, shape (n_samples,)
    k      : int

    Returns
    -------
    np.ndarray, shape (k, n_features)
        Updated centroid positions.

    TODO:
        - For each cluster index i in range(k), select all rows of X where
          labels == i.
        - Set centroid i to the mean of those rows (np.mean with axis=0).
        - Handle the edge case where a cluster has no assigned points
          (re-initialize that centroid randomly from X to avoid NaN).
    """
    # TODO: compute new centroids as the mean position of each cluster
    raise NotImplementedError("Implement update_centroids()")


# ---------------------------------------------------------------------------
# Full K-Means Loop
# ---------------------------------------------------------------------------

def kmeans(
    X: np.ndarray,
    k: int,
    max_iters: int = 100,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the K-Means algorithm on dataset X.

    Parameters
    ----------
    X         : np.ndarray, shape (n_samples, n_features)
    k         : int — number of clusters
    max_iters : int — maximum number of assign/update iterations
    tol       : float — stop early if centroid movement < tol

    Returns
    -------
    labels    : np.ndarray, shape (n_samples,) — final cluster assignments
    centroids : np.ndarray, shape (k, n_features) — final centroid positions

    TODO:
        1. Initialize centroids using initialize_centroids().
        2. Loop up to max_iters times:
           a. Assign each point to the nearest centroid (assign_clusters).
           b. Update centroids (update_centroids).
           c. Check for convergence: if the maximum shift of any centroid
              is less than tol, break early.
              Hint: np.linalg.norm on (new_centroids - old_centroids) per row,
              then np.max across all rows.
        3. Return the final labels and centroids.
    """
    # TODO: orchestrate the full K-Means loop
    raise NotImplementedError("Implement kmeans()")


# ---------------------------------------------------------------------------
# Plotting (provided — do not modify)
# ---------------------------------------------------------------------------

def plot_clusters(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    title: str = "K-Means Clustering",
) -> None:
    """
    Scatter plot of data points coloured by cluster with centroids marked.

    Parameters
    ----------
    X         : np.ndarray, shape (n_samples, 2)  — only first 2 dims plotted
    labels    : np.ndarray, shape (n_samples,)
    centroids : np.ndarray, shape (k, 2)
    title     : str
    """
    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=30, alpha=0.7)
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="X",
        s=200,
        c="black",
        zorder=5,
        label="Centroids",
    )
    plt.colorbar(scatter, label="Cluster")
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig("kmeans_result.png", dpi=120)
    plt.show()
    print("Plot saved to kmeans_result.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    End-to-end demonstration:
        1. Generate synthetic blobs.
        2. Run your K-Means implementation.
        3. Plot the result.
        4. Compute and print the final inertia (WCSS).

    TODO (in this function):
        - After obtaining labels and centroids from kmeans(), compute the
          inertia manually:
              inertia = Σ_i ||X[i] - centroids[labels[i]]||²
          Then print it.
        - (Optional) Compare your inertia with sklearn's KMeans inertia_
          to verify correctness.
    """
    np.random.seed(42)
    K = 4

    # Generate synthetic clustered data
    X, y_true = make_blobs(n_samples=300, centers=K, cluster_std=1.2, random_state=42)

    print(f"Dataset shape: {X.shape}")
    print(f"Running K-Means with k={K} ...")

    labels, centroids = kmeans(X, k=K, max_iters=100, tol=1e-4)

    print(f"Cluster sizes: {dict(zip(*np.unique(labels, return_counts=True)))}")

    # TODO: compute and print the inertia
    # inertia = ...
    # print(f"Inertia (WCSS): {inertia:.4f}")

    plot_clusters(X, labels, centroids)


if __name__ == "__main__":
    main()
