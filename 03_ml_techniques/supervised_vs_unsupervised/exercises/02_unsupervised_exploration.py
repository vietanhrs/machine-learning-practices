"""
Exercise 02: Unsupervised Learning Exploration
===============================================
Explore structure in data without using labels during clustering or
dimensionality reduction. Visualize the discovered structure and evaluate
cluster quality using label-free metrics.

Learning objectives:
- Apply PCA and t-SNE for dimensionality reduction
- Perform K-Means clustering on reduced representations
- Evaluate clustering without ground truth labels
- Compare PCA vs t-SNE visualizations

Dependencies: scikit-learn, matplotlib, numpy
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Dimensionality Reduction
# ---------------------------------------------------------------------------

def reduce_dimensions_pca(X, n_components=2):
    """
    Reduce the dimensionality of X using PCA.

    Args:
        X (np.ndarray): Input feature matrix, shape (n_samples, n_features)
        n_components (int): Number of principal components to retain

    Returns:
        tuple: (X_reduced, pca_object)
            - X_reduced: shape (n_samples, n_components)
            - pca_object: fitted PCA instance (useful for explained variance ratio)

    Hints:
        - Standardize X first using StandardScaler (fit on X, transform X)
        - Initialize PCA(n_components=n_components, random_state=42)
        - Fit and transform in one step: pca.fit_transform(X_scaled)
        - Print the cumulative explained variance ratio
        - Return both the reduced data and the fitted PCA object
    """
    # TODO: Standardize the input data
    # TODO: Initialize and fit PCA
    # TODO: Print explained variance per component and cumulative variance
    # TODO: Return (X_reduced, pca)
    pass


def reduce_dimensions_tsne(X, n_components=2):
    """
    Reduce the dimensionality of X using t-SNE.

    t-SNE is excellent for visualization but is non-parametric (cannot
    transform new points) and is slower than PCA.

    Args:
        X (np.ndarray): Input feature matrix, shape (n_samples, n_features)
        n_components (int): Number of dimensions for embedding (usually 2 or 3)

    Returns:
        np.ndarray: X_reduced of shape (n_samples, n_components)

    Hints:
        - Standardize X first using StandardScaler
        - For large datasets, first apply PCA to ~50 dims before t-SNE for speed
        - Initialize TSNE(n_components=n_components, perplexity=30,
                          n_iter=1000, random_state=42)
        - Use tsne.fit_transform(X_scaled) — t-SNE has no separate transform()
        - Print a note about t-SNE limitations (non-parametric, slow)
    """
    # TODO: Standardize the input data
    # TODO: (Optional) Pre-reduce with PCA if n_features > 50
    # TODO: Initialize and fit t-SNE
    # TODO: Return X_reduced
    pass


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster_and_visualize(X, X_2d, true_labels=None, n_clusters=10, title="K-Means Clustering"):
    """
    Apply K-Means clustering on X and visualize the result using the 2D
    projection X_2d.

    Args:
        X (np.ndarray): Original or reduced feature matrix for clustering
        X_2d (np.ndarray): 2D embedding for visualization (from PCA or t-SNE)
        true_labels (np.ndarray or None): Ground truth labels for comparison
                                          (not used for clustering, only plotting)
        n_clusters (int): Number of K-Means clusters
        title (str): Plot title prefix

    Returns:
        np.ndarray: cluster_labels from K-Means

    Hints:
        - Initialize KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        - Fit on X (not X_2d — cluster in original/reduced space)
        - Create a 1×2 or 1×1 subplot figure
        - Left plot: color points by K-Means cluster_labels
        - Right plot (if true_labels is not None): color points by true_labels
        - Use plt.scatter with c=labels, cmap='tab10'
        - Add colorbar, title, and axis labels
    """
    # TODO: Initialize and fit KMeans on X
    # TODO: Create scatter plot(s) in 2D space
    # TODO: If true_labels is not None, create a second subplot for comparison
    # TODO: Add colorbar, labels, titles
    # TODO: plt.show()
    # TODO: Return cluster_labels
    pass


# ---------------------------------------------------------------------------
# Evaluation Without Labels
# ---------------------------------------------------------------------------

def evaluate_without_labels(X, labels):
    """
    Evaluate cluster quality using label-free intrinsic metrics.

    Args:
        X (np.ndarray): Feature matrix used for clustering
        labels (np.ndarray): Cluster assignments from K-Means

    Returns:
        dict: {"silhouette": float, "davies_bouldin": float}

    Metrics explained:
        - Silhouette Score: ranges from -1 to 1.
            Higher is better. Measures how similar a point is to its own
            cluster vs. neighboring clusters.
        - Davies-Bouldin Index: lower is better.
            Measures average similarity between each cluster and its most
            similar cluster. 0 = perfect clustering.

    Hints:
        - Use sklearn.metrics.silhouette_score(X, labels)
        - Use sklearn.metrics.davies_bouldin_score(X, labels)
        - Print both scores with interpretation
        - Handle edge case: if all points are in one cluster, return None values
    """
    # TODO: Compute silhouette score
    # TODO: Compute Davies-Bouldin index
    # TODO: Print scores with interpretation
    # TODO: Return {"silhouette": ..., "davies_bouldin": ...}
    pass


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_pca_tsne(X, true_labels):
    """
    Reduce X using both PCA and t-SNE, then visualize side-by-side.
    Color points by true_labels to see which method better separates classes.

    NOTE: true_labels are used ONLY for coloring the visualization.
    They are NOT used during dimensionality reduction or clustering.

    Args:
        X (np.ndarray): Input feature matrix
        true_labels (np.ndarray): Ground truth class labels (for coloring only)

    Hints:
        - Call reduce_dimensions_pca(X, n_components=2) — get X_pca
        - Call reduce_dimensions_tsne(X, n_components=2) — get X_tsne
        - Create a figure with 1 row, 2 columns
        - Left plot: scatter X_pca[:, 0] vs X_pca[:, 1], colored by true_labels
        - Right plot: scatter X_tsne[:, 0] vs X_tsne[:, 1], colored by true_labels
        - Use cmap='tab10' for consistent colors
        - Add titles "PCA" and "t-SNE", and a shared colorbar
        - Observe: t-SNE typically shows better class separation visually
    """
    # TODO: Apply PCA to get X_pca
    # TODO: Apply t-SNE to get X_tsne
    # TODO: Create side-by-side scatter plots colored by true_labels
    # TODO: Add titles, labels, colorbar
    # TODO: plt.show()
    pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    Explore the MNIST digits or iris dataset using unsupervised techniques.

    Steps:
    1. Load dataset (digits or iris)
    2. Reduce to 2D using PCA → visualize
    3. Reduce to 2D using t-SNE → visualize
    4. Apply K-Means clustering (without using labels)
    5. Evaluate clustering with silhouette and Davies-Bouldin scores
    6. Compare PCA vs t-SNE side-by-side
    7. Compare K-Means cluster labels vs true labels (as evaluation only)
    """
    print("=" * 60)
    print("Unsupervised Learning Exploration")
    print("=" * 60)

    # TODO: Load dataset
    # Hint: Use datasets.load_digits() for a richer dataset
    # digits = datasets.load_digits()
    # X, y = digits.data, digits.target
    # print(f"Dataset shape: {X.shape}, Classes: {np.unique(y)}")

    # TODO: Step 1 — Apply PCA, print explained variance
    # Hint: X_pca, pca = reduce_dimensions_pca(X, n_components=2)

    # TODO: Step 2 — Apply t-SNE
    # Hint: X_tsne = reduce_dimensions_tsne(X, n_components=2)

    # TODO: Step 3 — Cluster using K-Means (on PCA-reduced data)
    # Hint: n_clusters should match the number of true classes
    # cluster_labels = cluster_and_visualize(X_pca, X_pca, true_labels=y)

    # TODO: Step 4 — Evaluate clusters without labels
    # Hint: evaluate_without_labels(X_pca, cluster_labels)

    # TODO: Step 5 — Compare PCA vs t-SNE visualizations
    # Hint: compare_pca_tsne(X, y)

    # TODO: Step 6 — Also visualize t-SNE with cluster labels for comparison
    # Hint: cluster_and_visualize(X_tsne, X_tsne, true_labels=y, title="t-SNE")

    print("\nDone. Observe how well clusters align with true labels without using them during training.")


if __name__ == "__main__":
    main()
