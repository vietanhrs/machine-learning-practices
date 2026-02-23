"""
Exercise 04: Hierarchical Clustering
=====================================

Goal: Explore agglomerative hierarchical clustering, interpret dendrograms,
      and compare the effect of different linkage methods.

Learning Objectives:
    - Build a linkage matrix using scipy.
    - Plot and interpret a dendrogram — understand what the height represents.
    - Cut a dendrogram at a chosen level to extract flat cluster assignments.
    - Compare single, complete, average, and Ward linkage visually.

Background:
    Hierarchical clustering builds a nested sequence of partitions.
    The LINKAGE MATRIX Z encodes each merge step:
        Z[i] = [cluster_a, cluster_b, distance, n_elements]
    The DENDROGRAM visualises this as a tree. The height at which two
    branches merge = the distance between those clusters (linkage-dependent).
    Cutting horizontally at height h gives clusters whose inter-cluster
    distance is >= h.

Instructions:
    - Fill in every section marked with TODO.
    - Use scipy.cluster.hierarchy throughout.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Step 1 — Build the Linkage Matrix
# ---------------------------------------------------------------------------

def compute_linkage_matrix(X: np.ndarray, method: str = "ward") -> np.ndarray:
    """
    Compute the linkage matrix for hierarchical clustering.

    Parameters
    ----------
    X      : np.ndarray, shape (n_samples, n_features)
    method : str — one of 'single', 'complete', 'average', 'ward'

    Returns
    -------
    np.ndarray, shape (n_samples-1, 4)
        The linkage matrix Z where each row encodes one merge step.
        Z[i] = [left_cluster, right_cluster, merge_distance, cluster_size]

    Linkage methods:
        single   — merge clusters that have the two closest individual points
                   (can produce "chaining" effect).
        complete — merge clusters based on their farthest pair of points
                   (produces compact, roughly equal clusters).
        average  — use the mean distance between all cross-cluster pairs.
        ward     — minimise the increase in within-cluster variance at each step
                   (most commonly recommended).

    TODO:
        - Standardize X with StandardScaler (distance-based method).
        - Call scipy.cluster.hierarchy.linkage(X_scaled, method=method).
        - Return the resulting linkage matrix Z.
    """
    # TODO: standardize X, compute and return the linkage matrix
    raise NotImplementedError("Implement compute_linkage_matrix()")


# ---------------------------------------------------------------------------
# Step 2 — Plot the Dendrogram
# ---------------------------------------------------------------------------

def plot_dendrogram(
    Z: np.ndarray,
    labels: list | None = None,
    title: str = "Dendrogram",
    max_display: int = 30,
) -> None:
    """
    Plot the dendrogram from a linkage matrix.

    Parameters
    ----------
    Z           : np.ndarray — linkage matrix from compute_linkage_matrix()
    labels      : list | None — optional leaf labels
    title       : str
    max_display : int — truncate to this many leaf nodes for readability

    Reading the dendrogram:
        - Each leaf at the bottom represents one data point (or one cluster
          if truncated).
        - The HEIGHT at which two branches join = the linkage distance between
          the two groups being merged.
        - A horizontal cut at height h produces clusters that are internally
          closer than h to each other.
        - Large vertical gaps in the dendrogram suggest natural cluster
          boundaries.

    TODO:
        - Call scipy.cluster.hierarchy.dendrogram() with:
            * Z as the first argument
            * truncate_mode='lastp', p=max_display (show top p merges)
            * labels=labels if provided
            * leaf_rotation=90 for readability
        - Set an appropriate figure size, title, x/y labels.
        - Save the figure as "dendrogram_{title}.png" (sanitise spaces).
        - Call plt.show().
    """
    # TODO: plot the dendrogram using scipy.cluster.hierarchy.dendrogram
    raise NotImplementedError("Implement plot_dendrogram()")


# ---------------------------------------------------------------------------
# Step 3 — Extract Flat Clusters by Cutting the Dendrogram
# ---------------------------------------------------------------------------

def cut_dendrogram(Z: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Cut the dendrogram to produce exactly n_clusters flat clusters.

    Parameters
    ----------
    Z          : np.ndarray — linkage matrix
    n_clusters : int — desired number of clusters

    Returns
    -------
    np.ndarray, shape (n_samples,)
        Integer cluster labels (1-indexed, as returned by fcluster).

    TODO:
        - Call scipy.cluster.hierarchy.fcluster(Z, t=n_clusters, criterion='maxclust').
        - Return the resulting label array.
        Hint: fcluster labels are 1-indexed (cluster 1, 2, ..., n_clusters).
              Subtract 1 if you prefer 0-indexed labels.
    """
    # TODO: cut the dendrogram and return flat cluster labels
    raise NotImplementedError("Implement cut_dendrogram()")


# ---------------------------------------------------------------------------
# Step 4 — Compare Linkage Methods
# ---------------------------------------------------------------------------

def compare_linkage_methods(X: np.ndarray, n_clusters: int = 3) -> None:
    """
    Fit hierarchical clustering with four linkage methods and compare results.

    Parameters
    ----------
    X          : np.ndarray, shape (n_samples, n_features)
    n_clusters : int — number of clusters to extract from each dendrogram

    What to observe:
        - Single linkage tends to chain: one large cluster absorbs points one
          by one. It is sensitive to outliers and noise.
        - Complete linkage creates compact, balanced clusters but can split
          large natural clusters.
        - Average linkage is a compromise between single and complete.
        - Ward linkage generally produces the most visually intuitive clusters
          when clusters are roughly convex and similar in size.

    TODO:
        - For each method in ['single', 'complete', 'average', 'ward']:
            1. Compute the linkage matrix.
            2. Plot the dendrogram (call plot_dendrogram).
            3. Extract flat cluster labels for n_clusters.
            4. Plot a scatter of X coloured by cluster label.
        - Arrange the scatter plots in a 2×2 grid for easy comparison.
        - Title each subplot with the linkage method name.
        - Save the comparison figure as "linkage_comparison.png".
    """
    # TODO: compare four linkage methods with dendrograms and scatter plots
    raise NotImplementedError("Implement compare_linkage_methods()")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Full hierarchical clustering pipeline:
        1. Generate synthetic data.
        2. Build and plot a dendrogram.
        3. Cut the dendrogram to extract cluster assignments.
        4. Compare all four linkage methods.
    """
    np.random.seed(7)

    X, y_true = make_blobs(n_samples=150, centers=3, cluster_std=1.0, random_state=7)

    print("=== Step 1: Compute Linkage Matrix (Ward) ===")
    Z_ward = compute_linkage_matrix(X, method="ward")
    print(f"Linkage matrix shape: {Z_ward.shape}")  # should be (n_samples-1, 4)

    print("\n=== Step 2: Plot Dendrogram ===")
    plot_dendrogram(Z_ward, title="Ward Linkage")

    print("\n=== Step 3: Cut Dendrogram into 3 Clusters ===")
    labels = cut_dendrogram(Z_ward, n_clusters=3)
    print(f"Unique cluster labels: {np.unique(labels)}")
    print(f"Cluster sizes: {dict(zip(*np.unique(labels, return_counts=True)))}")

    print("\n=== Step 4: Compare Linkage Methods ===")
    compare_linkage_methods(X, n_clusters=3)
    print("Done. Check saved figures.")


if __name__ == "__main__":
    main()
