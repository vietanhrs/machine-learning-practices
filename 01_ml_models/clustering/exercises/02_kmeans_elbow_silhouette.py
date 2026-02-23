"""
Exercise 02: Choosing K — Elbow Method & Silhouette Analysis
=============================================================

Goal: Learn how to systematically select the number of clusters `k` using
      two complementary methods: the Elbow Method (inertia) and the
      Silhouette Score.

Learning Objectives:
    - Understand what inertia measures and why it always decreases with k.
    - Interpret the elbow curve to find the optimal k.
    - Understand the Silhouette score and how to use it for model selection.
    - Observe that the two methods can (and sometimes do) disagree.

Instructions:
    - Fill in every section marked with TODO.
    - The helper kmeans() from Exercise 01 is NOT available here — use
      sklearn.cluster.KMeans for the clustering step so you can focus
      on the evaluation logic.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs


# ---------------------------------------------------------------------------
# Inertia
# ---------------------------------------------------------------------------

def compute_inertia(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
) -> float:
    """
    Compute the Within-Cluster Sum of Squares (WCSS / inertia).

    Inertia = Σ_i ||X[i] - centroid[labels[i]]||²

    This measures how tightly packed each cluster is. Lower is better, but
    inertia always decreases as k increases, so it cannot be minimised alone.

    Parameters
    ----------
    X         : np.ndarray, shape (n_samples, n_features)
    labels    : np.ndarray, shape (n_samples,)  — cluster assignments
    centroids : np.ndarray, shape (k, n_features)

    Returns
    -------
    float — the total WCSS

    TODO:
        - For each point X[i], find its assigned centroid: centroids[labels[i]].
        - Compute the squared Euclidean distance between X[i] and that centroid.
        - Sum all squared distances.
        Hint: vectorised approach — compute (X - centroids[labels]) then
              np.sum of squared values.
    """
    # TODO: implement inertia computation
    raise NotImplementedError("Implement compute_inertia()")


# ---------------------------------------------------------------------------
# Elbow Method
# ---------------------------------------------------------------------------

def elbow_method(X: np.ndarray, k_range: range) -> list[float]:
    """
    Compute and plot inertia for each k in k_range (Elbow Method).

    The 'elbow' is the value of k beyond which adding more clusters provides
    diminishing returns — inertia drops slowly rather than steeply.

    Parameters
    ----------
    X       : np.ndarray, shape (n_samples, n_features)
    k_range : range  — e.g. range(2, 11)

    Returns
    -------
    list[float] — inertia values, one per k

    TODO:
        - For each k in k_range:
            1. Fit sklearn's KMeans(n_clusters=k, n_init=10, random_state=42).
            2. Extract labels and cluster_centers_.
            3. Call compute_inertia() (or use model.inertia_ as a check).
            4. Append to a list.
        - After the loop, plot k vs. inertia:
            * x-axis: k values
            * y-axis: inertia
            * title: "Elbow Method — Inertia vs. k"
            * mark each data point with a dot
        - Save the figure as "elbow_method.png".
        - Return the list of inertia values.
    """
    # TODO: implement elbow method — fit KMeans for each k, collect inertia, plot
    raise NotImplementedError("Implement elbow_method()")


# ---------------------------------------------------------------------------
# Silhouette Analysis
# ---------------------------------------------------------------------------

def silhouette_analysis(X: np.ndarray, k_range: range) -> list[float]:
    """
    Compute and plot the average Silhouette score for each k in k_range.

    The Silhouette score measures how similar a point is to its own cluster
    (cohesion) compared to other clusters (separation). Range: [-1, 1].
    A higher average score indicates better-defined clusters.

    Parameters
    ----------
    X       : np.ndarray, shape (n_samples, n_features)
    k_range : range — e.g. range(2, 11)

    Returns
    -------
    list[float] — silhouette scores, one per k

    TODO:
        - For each k in k_range:
            1. Fit KMeans(n_clusters=k, n_init=10, random_state=42).
            2. Get predicted labels.
            3. Compute sklearn.metrics.silhouette_score(X, labels).
               Note: silhouette_score requires at least 2 clusters and
               at least 2 distinct labels.
            4. Append score to a list.
        - Plot k vs. silhouette score:
            * title: "Silhouette Analysis"
            * draw a horizontal dashed line at score=0 for reference
        - Save as "silhouette_analysis.png".
        - Return the list of scores.
    """
    # TODO: implement silhouette analysis — fit KMeans, compute scores, plot
    raise NotImplementedError("Implement silhouette_analysis()")


# ---------------------------------------------------------------------------
# Side-by-Side Comparison
# ---------------------------------------------------------------------------

def compare_k_values(X: np.ndarray, k_range: range) -> None:
    """
    Run both the Elbow Method and Silhouette Analysis side by side and
    print a summary table.

    Parameters
    ----------
    X       : np.ndarray, shape (n_samples, n_features)
    k_range : range

    TODO:
        - Call elbow_method(X, k_range) and silhouette_analysis(X, k_range).
        - Create a side-by-side subplot (1 row, 2 columns) combining both plots.
        - Print a table to the console:
            k | Inertia | Silhouette
            ---------------------------
            2 | ...     | ...
            3 | ...     | ...
            ...
        - Identify and print:
            * The k with the lowest inertia rate-of-change (elbow suggestion).
            * The k with the highest silhouette score.
        Hint: The elbow can be approximated by finding the k where the
              second derivative of inertia is maximised (largest "kink").
    """
    # TODO: combine both analyses, print summary, identify best k by each metric
    raise NotImplementedError("Implement compare_k_values()")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Generate synthetic data and run the full k-selection analysis.
    """
    np.random.seed(0)

    # True number of clusters = 4
    X, y_true = make_blobs(
        n_samples=400,
        centers=4,
        cluster_std=1.0,
        random_state=0,
    )

    print("Dataset shape:", X.shape)
    print("Running k-selection analysis for k = 2..10 ...\n")

    k_range = range(2, 11)
    compare_k_values(X, k_range)


if __name__ == "__main__":
    main()
