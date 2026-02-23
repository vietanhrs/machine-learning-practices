"""
Exercise 03: DBSCAN Exploration
================================

Goal: Understand DBSCAN's behaviour across different parameter settings and
      dataset shapes, and contrast it with K-Means.

Learning Objectives:
    - Fit DBSCAN and interpret its output (cluster labels and noise).
    - Perform a grid search over (eps, min_samples) to see the effect of
      each hyperparameter.
    - Understand WHY DBSCAN outperforms K-Means on non-spherical data.

Background:
    DBSCAN assigns each point one of three roles:
        * Core point   — has >= min_samples neighbours within eps.
        * Border point — within eps of a core point but not core itself.
        * Noise point  — label = -1 in sklearn's output.
    The algorithm does NOT require specifying k in advance.

Instructions:
    - Fill in every section marked with TODO.
    - Use sklearn.cluster.DBSCAN throughout (no need to implement from scratch).
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Basic DBSCAN Wrapper
# ---------------------------------------------------------------------------

def run_dbscan(
    X: np.ndarray,
    eps: float,
    min_samples: int,
) -> np.ndarray:
    """
    Fit DBSCAN on X and return the cluster labels.

    Parameters
    ----------
    X          : np.ndarray, shape (n_samples, n_features)
    eps        : float — neighbourhood radius
    min_samples: int   — minimum points to form a core point

    Returns
    -------
    np.ndarray, shape (n_samples,)
        Cluster labels. Noise points are labelled -1.

    TODO:
        - Standardize X using StandardScaler (DBSCAN is distance-based,
          so scale matters).
        - Fit sklearn.cluster.DBSCAN(eps=eps, min_samples=min_samples).
        - Return the labels_ attribute.
    """
    # TODO: standardize, fit DBSCAN, return labels
    raise NotImplementedError("Implement run_dbscan()")


# ---------------------------------------------------------------------------
# Cluster & Noise Statistics
# ---------------------------------------------------------------------------

def count_clusters_and_noise(labels: np.ndarray) -> dict:
    """
    Summarise the DBSCAN output.

    Parameters
    ----------
    labels : np.ndarray — DBSCAN labels (noise = -1)

    Returns
    -------
    dict with keys:
        'n_clusters' : int — number of clusters found (excluding noise)
        'n_noise'    : int — number of noise points
        'cluster_sizes' : dict — {cluster_id: size} for non-noise clusters

    TODO:
        - Count unique labels excluding -1 to get n_clusters.
        - Count occurrences of -1 to get n_noise.
        - Build a dict of cluster sizes.
        Hint: np.unique with return_counts=True is useful.
    """
    # TODO: count clusters and noise from DBSCAN labels
    raise NotImplementedError("Implement count_clusters_and_noise()")


# ---------------------------------------------------------------------------
# Hyperparameter Grid Search
# ---------------------------------------------------------------------------

def tune_dbscan_params(
    X: np.ndarray,
    eps_values: list[float],
    min_samples_values: list[int],
) -> None:
    """
    Grid search over (eps, min_samples) and print a summary of results.

    For each combination, report:
        - Number of clusters found.
        - Number of noise points.
    Optionally, plot a heatmap of n_clusters across the grid.

    Parameters
    ----------
    X                  : np.ndarray
    eps_values         : list of float — eps values to try
    min_samples_values : list of int   — min_samples values to try

    TODO:
        - Nested loop over eps_values and min_samples_values.
        - For each pair, call run_dbscan() then count_clusters_and_noise().
        - Store results in a 2-D array (or dict).
        - Print a formatted table showing (eps, min_samples) → n_clusters, n_noise.
        - (Optional) Plot a heatmap using plt.imshow() or seaborn.heatmap().
        Hint: A large eps merges everything into one cluster. A tiny eps
              labels almost everything as noise. Balance is key.
    """
    # TODO: grid search over eps and min_samples, print results table
    raise NotImplementedError("Implement tune_dbscan_params()")


# ---------------------------------------------------------------------------
# K-Means vs. DBSCAN Comparison
# ---------------------------------------------------------------------------

def compare_kmeans_dbscan(
    X_blobs: np.ndarray,
    X_moons: np.ndarray,
    X_circles: np.ndarray,
) -> None:
    """
    Compare K-Means and DBSCAN visually on three dataset shapes.

    Why DBSCAN wins on non-spherical data:
        K-Means assumes clusters are convex and roughly equal in size.
        Its decision boundaries are linear (Voronoi cells).
        DBSCAN follows the density of points, so it can trace curved,
        elongated, or ring-shaped clusters that K-Means cannot capture.

    Parameters
    ----------
    X_blobs   : np.ndarray — isotropic Gaussian blobs (K-Means should do well)
    X_moons   : np.ndarray — two interleaved half-circles (K-Means fails)
    X_circles : np.ndarray — concentric circles (K-Means fails completely)

    TODO:
        - For each dataset (blobs, moons, circles):
            1. Run KMeans with an appropriate k.
            2. Run DBSCAN with tuned (eps, min_samples).
            3. Plot side by side: K-Means result | DBSCAN result.
        - Arrange all 6 subplots in a 3×2 grid.
        - Title each subplot with the dataset name and algorithm.
        - Save the figure as "kmeans_vs_dbscan.png".
        Hint:
            blobs   → k=3, eps=0.3, min_samples=5
            moons   → k=2, eps=0.15, min_samples=5
            circles → k=2, eps=0.15, min_samples=5
            Always standardize X before DBSCAN (done inside run_dbscan).
    """
    # TODO: compare K-Means and DBSCAN on three dataset shapes, plot 3x2 grid
    raise NotImplementedError("Implement compare_kmeans_dbscan()")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Full DBSCAN exploration pipeline.
    """
    np.random.seed(42)

    # Generate datasets
    X_blobs, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=42)
    X_moons, _ = make_moons(n_samples=300, noise=0.07, random_state=42)
    X_circles, _ = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)

    # --- Part 1: Basic DBSCAN ---
    print("=== Part 1: Basic DBSCAN on Blobs ===")
    labels = run_dbscan(X_blobs, eps=0.5, min_samples=5)
    stats = count_clusters_and_noise(labels)
    print(f"Clusters found: {stats['n_clusters']}")
    print(f"Noise points  : {stats['n_noise']}")
    print(f"Cluster sizes : {stats['cluster_sizes']}\n")

    # --- Part 2: Hyperparameter Tuning ---
    print("=== Part 2: Hyperparameter Grid Search on Moons ===")
    eps_values = [0.05, 0.1, 0.2, 0.4, 0.8]
    min_samples_values = [3, 5, 10]
    tune_dbscan_params(X_moons, eps_values, min_samples_values)

    # --- Part 3: Visual Comparison ---
    print("\n=== Part 3: K-Means vs. DBSCAN Comparison ===")
    compare_kmeans_dbscan(X_blobs, X_moons, X_circles)


if __name__ == "__main__":
    main()
