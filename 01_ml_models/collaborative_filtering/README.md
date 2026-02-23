# Collaborative Filtering

Collaborative Filtering (CF) is a technique for building **recommendation systems** by learning from patterns of user-item interactions. The core assumption is that users who agreed in the past will agree in the future, and that they will like similar items to those they have liked before.

---

## 1. What is Collaborative Filtering?

CF makes recommendations based solely on **historical interaction data** (ratings, clicks, purchases) — it does not need to know anything about item content or user demographics. This contrasts with **Content-Based Filtering**, which uses item features, and **Hybrid Methods**, which combine both.

### Types of Collaborative Filtering

| Type | Approach |
|------|----------|
| **Memory-Based** | Directly use the rating matrix to compute similarity (User-Based, Item-Based) |
| **Model-Based** | Learn a compact model from the rating matrix (Matrix Factorization, Neural CF) |

### When to Use CF

- You have a large user base with overlapping tastes.
- Items are hard to describe with features (e.g., movies, music).
- You have a dense enough rating matrix to compute meaningful similarities.

---

## 2. User-Based Collaborative Filtering

**Idea:** Find users who are similar to the target user and recommend items they liked.

### Steps

1. Represent each user as a vector of item ratings (NaN for unrated items).
2. Compute **cosine similarity** (or Pearson correlation) between all pairs of users.
3. For a target user `u` and target item `i`:
   - Find the `k` most similar users who have rated item `i`.
   - Predict `u`'s rating for `i` as a weighted average:

```
r̂(u, i) = Σ_{v ∈ N_k(u)} sim(u, v) · r(v, i)
           ─────────────────────────────────────
                   Σ_{v ∈ N_k(u)} |sim(u, v)|
```

### Cosine Similarity

```
cos(u, v) = (u · v) / (||u|| · ||v||)
```

When ratings are mean-centered per user, this is equivalent to Pearson correlation.

### Limitation

As the number of users grows, computing and storing all pairwise similarities becomes expensive. Also, user preferences can shift over time (the matrix is dynamic).

---

## 3. Item-Based Collaborative Filtering

**Idea:** Find items similar to items the target user has already rated, then recommend them.

### Steps

1. Represent each item as a vector of user ratings (across all users).
2. Compute similarity between all pairs of items.
3. For a target user `u` and target item `i`:
   - Find the `k` items most similar to `i` that `u` has already rated.
   - Predict the rating as a weighted average of `u`'s ratings for those items:

```
r̂(u, i) = Σ_{j ∈ N_k(i)} sim(i, j) · r(u, j)
           ─────────────────────────────────────
                   Σ_{j ∈ N_k(i)} |sim(i, j)|
```

### Why Item-Based is More Stable Than User-Based

- Items change less frequently than user preferences.
- The item-item similarity matrix can be precomputed offline and reused.
- The number of items is often smaller than the number of users (Amazon, Netflix: millions of users, but a finite catalog).

---

## 4. Matrix Factorization

Matrix Factorization (MF) is a **model-based** approach that decomposes the rating matrix into low-dimensional latent factor matrices.

### Idea

Approximate the `m × n` rating matrix `R` as the product of two lower-rank matrices:

```
R ≈ P × Qᵀ
```

- `P` : shape `(m_users, k)` — user latent factor matrix
- `Q` : shape `(n_items, k)` — item latent factor matrix
- `k` : number of latent factors (much smaller than m or n)

Each user is represented by a `k`-dimensional vector of latent factors, and each item similarly. The predicted rating for user `u` and item `i` is:

```
r̂(u, i) = p_u · q_i  (dot product)
```

### Latent Factors

Latent factors are not directly interpretable, but they capture abstract concepts like genre preferences, quality preferences, or niche interests. For example, in a movie recommender, one latent factor might correspond to "preference for action films" without being explicitly labeled.

### Training: Stochastic Gradient Descent (SGD)

Minimize the regularized reconstruction error over observed ratings:

```
L = Σ_{(u,i) observed} (r(u,i) - p_u · q_i)² + λ(||p_u||² + ||q_i||²)
```

Update rules for each observed (u, i):
```
e_ui = r(u,i) - p_u · q_i
p_u ← p_u + η · (e_ui · q_i - λ · p_u)
q_i ← q_i + η · (e_ui · p_u - λ · q_i)
```

### ALS (Alternating Least Squares)

Fix `Q` and solve for `P` in closed form, then fix `P` and solve for `Q`. Repeat. More parallelizable than SGD; preferred for very large, sparse matrices.

---

## 5. Cold Start Problem

A fundamental limitation of CF: it requires historical interaction data.

| Scenario | Problem |
|----------|---------|
| **New User** | No ratings → cannot compute similarity or latent factors |
| **New Item** | No ratings → item never gets recommended |
| **New System** | No data at all |

### Mitigation Strategies

- **Hybrid filtering:** Combine CF with content-based features for new users/items.
- **Onboarding questionnaire:** Ask new users to rate a few popular items.
- **Popularity-based defaults:** Recommend globally popular items to new users.
- **Side information:** Use user demographics or item metadata.
- **Transfer learning:** Use pre-trained embeddings from related domains.

---

## 6. Explicit vs. Implicit Feedback

| Type | Description | Examples |
|------|-------------|---------|
| **Explicit** | User directly expresses preference | Star ratings, thumbs up/down |
| **Implicit** | Inferred from behavior | Clicks, views, purchase history, play counts |

Implicit feedback is more abundant but noisier (a click does not mean the user liked the item). Special models like ALS with confidence weighting (Hu et al. 2008) handle implicit feedback.

---

## 7. Evaluation Metrics

### Rating Prediction Metrics

```
RMSE = √[ (1/n) Σ (r(u,i) - r̂(u,i))² ]
MAE  = (1/n) Σ |r(u,i) - r̂(u,i)|
```

Lower is better. RMSE penalizes large errors more than MAE.

### Ranking / Top-K Metrics

**Precision@K:** Of the top-K recommended items, what fraction are relevant?

```
Precision@K = |{relevant items} ∩ {top-K recommendations}| / K
```

**Recall@K:** Of all relevant items, what fraction appear in the top-K?

```
Recall@K = |{relevant items} ∩ {top-K recommendations}| / |{relevant items}|
```

Higher is better for both. There is a trade-off between Precision@K and Recall@K.

---

## 8. Reference Links

- [Wikipedia: Collaborative Filtering](https://en.wikipedia.org/wiki/Collaborative_filtering)
- [Surprise — Python Recommender System Library](https://surprise.readthedocs.io/en/stable/)
