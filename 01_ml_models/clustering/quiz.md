# Clustering — Quiz

Test your understanding of clustering concepts. Questions mix multiple-choice, true/false, and short-answer formats.

---

**Q1. (Multiple Choice)**
K-Means minimizes which objective function?

- A) Mean Absolute Error (MAE)
- B) Within-Cluster Sum of Squares (WCSS / Inertia)
- C) Between-Cluster Sum of Squares
- D) Binary Cross-Entropy

---

**Q2. (True/False)**
DBSCAN requires the user to specify the number of clusters before running the algorithm.

---

**Q3. (Multiple Choice)**
What is the Silhouette score range?

- A) 0 to 1
- B) -∞ to +∞
- C) -1 to +1
- D) 0 to 100

---

**Q4. (Short Answer)**
Explain what the Elbow Method is and why the "elbow" point is chosen as the optimal number of clusters.

---

**Q5. (Multiple Choice)**
How does DBSCAN classify a point that is within `ε` of a core point but does not itself have `min_samples` neighbors?

- A) Core point
- B) Noise point
- C) Border point
- D) Outlier point

---

**Q6. (True/False)**
Feature scaling (e.g., standardization) has no effect on K-Means clustering results.

---

**Q7. (Multiple Choice)**
Which scenario is best suited for DBSCAN rather than K-Means?

- A) Data is composed of 5 spherical, equally-sized Gaussian blobs.
- B) Data has clusters shaped like interlocking crescents with some noise points.
- C) You have 1 million data points and need fast clustering.
- D) You already know the exact number of clusters.

---

**Q8. (Short Answer)**
What is the time complexity of K-Means in terms of `n` (data points), `k` (clusters), and `t` (iterations)? Why is this considered efficient?

---

**Q9. (Multiple Choice)**
A Silhouette score of `-0.3` for a data point means:

- A) The point is perfectly placed in its cluster.
- B) The point is equidistant between two clusters.
- C) The point may be incorrectly assigned — it is closer to another cluster.
- D) There is no meaningful interpretation below 0.

---

**Q10. (True/False)**
Hierarchical clustering is generally more scalable to very large datasets than K-Means.

---

**Q11. (Multiple Choice)**
Which linkage method in hierarchical clustering minimizes the increase in total within-cluster variance at each merge step?

- A) Single linkage
- B) Complete linkage
- C) Average linkage
- D) Ward linkage

---

**Q12. (Short Answer)**
You are clustering customer data that contains extreme outliers in purchase amount. Describe one reason K-Means might perform poorly in this situation and suggest a better algorithm choice.

---

<details>
<summary><strong>Answer Key (click to expand)</strong></summary>

**A1.** B — K-Means minimizes Within-Cluster Sum of Squares (WCSS), also called inertia:
`J = Σ_i Σ_{x ∈ C_i} ||x - μ_i||²`

**A2.** False — DBSCAN does not require specifying the number of clusters. It determines clusters automatically based on `ε` (epsilon) and `min_samples`. The user specifies density parameters, not cluster count.

**A3.** C — The Silhouette score ranges from `-1` to `+1`. A score near `+1` indicates the point is well-clustered, near `0` indicates it lies on a cluster boundary, and near `-1` suggests it may be misassigned.

**A4.** The Elbow Method plots inertia (WCSS) against increasing values of `k`. As `k` increases, inertia always decreases, but the rate of decrease slows. The "elbow" is the point where adding another cluster provides diminishing returns — inertia drops steeply before the elbow and flattens after it. This elbow point balances model complexity with cluster quality.

**A5.** C — Border point. It is reachable from a core point (within ε of it) but does not have enough neighbors to be a core point itself. It is assigned to the same cluster as its core point neighbor.

**A6.** False — K-Means relies on Euclidean distance, so features with larger scales dominate the distance calculation. Features must be standardized (e.g., zero mean, unit variance) so that all features contribute equally.

**A7.** B — DBSCAN excels at finding clusters of arbitrary shape (like crescents or rings) and can label noise points, making it ideal for that scenario. K-Means assumes spherical clusters.

**A8.** K-Means has time complexity `O(n × k × t)` where `n` = number of points, `k` = number of clusters, `t` = number of iterations. It is efficient because typically `k << n` and `t` converges quickly in practice (often fewer than 100 iterations), making it linear in `n`.

**A9.** C — A negative Silhouette score means the point is, on average, closer to points in a neighboring cluster than to points in its own cluster. It is likely misassigned.

**A10.** False — Hierarchical (agglomerative) clustering has time complexity `O(n² log n)` and memory complexity `O(n²)`, making it much slower and more memory-intensive than K-Means `O(nkt)` for large datasets.

**A11.** D — Ward linkage. At each step, it merges the pair of clusters that results in the smallest increase in total within-cluster variance (WCSS). This tends to produce compact, roughly equal-sized clusters.

**A12.** K-Means computes cluster centroids as the mean of assigned points. Outliers in purchase amount pull centroids away from the true center of the majority of points, distorting cluster assignments. A better choice would be **DBSCAN**, which marks extreme outliers as noise points and is not affected by them during centroid computation (there are no centroids). Alternatively, **K-Medoids** uses the actual median point as the cluster representative, making it more robust to outliers.

</details>
