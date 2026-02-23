# Clustering

Clustering is an **unsupervised learning** technique that groups data points so that points within the same group (cluster) are more similar to each other than to those in other groups. Because there are no labels, the algorithm discovers structure on its own.

---

## 1. What is Clustering?

**Definition:** Clustering partitions a dataset into subsets called clusters such that intra-cluster similarity is maximized and inter-cluster similarity is minimized.

### Common Use Cases

| Domain | Example |
|--------|---------|
| Marketing | Customer segmentation by purchasing behavior |
| Biology | Grouping genes with similar expression profiles |
| NLP | Topic modeling, document grouping |
| Anomaly Detection | Isolating noise points as outliers |
| Image Compression | Color quantization via pixel clustering |
| Geospatial | Identifying neighborhood regions |

Unlike supervised learning, clustering has **no ground truth labels** during training. Evaluation relies on internal metrics or domain knowledge.

---

## 2. K-Means Algorithm

K-Means partitions `n` observations into `k` clusters, where each observation belongs to the cluster with the nearest centroid.

### Algorithm Steps

1. **Initialize** `k` centroids randomly (or using K-Means++ for smarter initialization).
2. **Assign** each point to the nearest centroid (Euclidean distance by default).
3. **Update** each centroid to the mean of all points assigned to it.
4. **Repeat** steps 2–3 until centroids stop moving (convergence) or `max_iters` is reached.

### Objective Function (Inertia / WCSS)

K-Means minimizes **Within-Cluster Sum of Squares (WCSS)**:

```
J = Σ_i Σ_{x ∈ C_i} ||x - μ_i||²
```

where `μ_i` is the centroid of cluster `C_i`.

### Convergence

The algorithm is guaranteed to converge (WCSS is non-increasing) but may find a **local minimum**. Running multiple restarts (`n_init` in sklearn) mitigates this.

### Choosing K

**Elbow Method:**
- Plot inertia (WCSS) vs. `k`.
- The "elbow" — where inertia stops decreasing sharply — suggests the optimal `k`.
- Subjective; not always a clear elbow.

**Silhouette Score:**
- Measures how similar a point is to its own cluster vs. neighbouring clusters.
- Range: `-1` to `+1` (higher is better).
- Choose `k` that maximizes the average silhouette score.

```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

where `a(i)` = mean intra-cluster distance, `b(i)` = mean distance to nearest other cluster.

### Limitations of K-Means

- Requires specifying `k` in advance.
- Assumes spherical, equally-sized clusters.
- Sensitive to outliers (centroid shifts).
- Sensitive to feature scale — **always scale features first**.

---

## 3. DBSCAN

**Density-Based Spatial Clustering of Applications with Noise** groups points that are closely packed together and marks points in low-density regions as outliers.

### Key Parameters

| Parameter | Meaning |
|-----------|---------|
| `epsilon (ε)` | Radius of the neighborhood around a point |
| `min_samples` | Minimum number of points within ε to be a core point |

### Point Types

- **Core Point:** Has at least `min_samples` points within distance `ε` (including itself).
- **Border Point:** Within `ε` of a core point but does not itself have enough neighbors.
- **Noise Point:** Neither core nor border; treated as an outlier.

### Algorithm Steps

1. For each unvisited point, retrieve its ε-neighborhood.
2. If the point is a core point, start a new cluster and expand it.
3. Expansion: recursively add all density-reachable points.
4. Mark non-reachable points as noise.

### Advantages Over K-Means

- Does **not** require specifying the number of clusters.
- Can find clusters of **arbitrary shape** (moons, rings, blobs).
- Naturally handles **outliers/noise**.
- Works well when clusters have varying density (with careful tuning).

### Disadvantages

- Sensitive to `ε` and `min_samples` — tuning is non-trivial.
- Struggles with **high-dimensional** data (curse of dimensionality).
- Not ideal when clusters have very different densities.

---

## 4. Hierarchical Clustering

Hierarchical clustering builds a **tree of clusters** (dendrogram) that shows how clusters merge (agglomerative) or split (divisive) at each step.

### Agglomerative (Bottom-Up)

1. Start with each point as its own cluster.
2. Merge the two closest clusters.
3. Repeat until all points are in one cluster.

### Divisive (Top-Down)

1. Start with all points in one cluster.
2. Recursively split the least cohesive cluster.
3. Repeat until each point is its own cluster.

> Agglomerative is far more common in practice.

### Dendrogram

A dendrogram visualizes the hierarchy of merges. The **height** of a merge represents the distance between the two clusters. Cut the dendrogram at a height to obtain `k` clusters.

### Linkage Methods

| Method | Distance Between Clusters |
|--------|--------------------------|
| **Single** | Minimum distance between any two points across clusters |
| **Complete** | Maximum distance between any two points across clusters |
| **Average** | Mean distance between all pairs of points across clusters |
| **Ward** | Minimizes increase in total within-cluster variance after merge |

> **Ward linkage** generally produces the most compact, well-separated clusters and is the default recommendation.

---

## 5. Evaluation Metrics

Because clustering is unsupervised, we cannot use accuracy. Internal metrics assess cluster quality without labels.

### Silhouette Score

```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

- Range: `[-1, 1]`
- `+1`: point is well inside its cluster.
- `0`: point is on the boundary.
- `-1`: point may be assigned to the wrong cluster.
- Average over all points gives the overall score.

### Davies-Bouldin Index

```
DB = (1/k) Σ_i max_{j ≠ i} [ (s_i + s_j) / d(c_i, c_j) ]
```

- `s_i` = average distance of points in cluster `i` to centroid.
- `d(c_i, c_j)` = distance between centroids.
- **Lower is better** (more separation, more compactness).

### Calinski-Harabasz Index (Variance Ratio Criterion)

```
CH = [SSB / (k-1)] / [SSW / (n-k)]
```

- `SSB` = between-cluster scatter, `SSW` = within-cluster scatter.
- **Higher is better**.
- Fast to compute; tends to favor convex, well-separated clusters.

> None of these metrics require ground-truth labels — they are computed purely from the data and cluster assignments.

---

## 6. Choosing the Right Algorithm

| Criterion | K-Means | DBSCAN | Hierarchical |
|-----------|---------|--------|--------------|
| Number of clusters | Must specify `k` | Determined automatically | Determined by cutting dendrogram |
| Cluster shape | Spherical only | Arbitrary shape | Depends on linkage |
| Handles noise/outliers | No (outliers distort centroids) | Yes (noise label) | No |
| Scalability | Very fast — O(nkt) | Moderate — O(n log n) with index | Slow — O(n² log n) |
| Sensitivity to scale | Yes | Yes | Yes |
| Interpretability | Simple, easy to explain | Moderate | High (dendrogram) |
| Best for | Large datasets, known k | Irregular shapes, outliers | Small datasets, visualization |

---

## 7. Reference Links

- [Scikit-learn Clustering Documentation](https://scikit-learn.org/stable/modules/clustering.html)
- [Wikipedia: K-Means Clustering](https://en.wikipedia.org/wiki/K-means_clustering)
- [Wikipedia: DBSCAN](https://en.wikipedia.org/wiki/DBSCAN)
