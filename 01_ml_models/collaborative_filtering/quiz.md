# Collaborative Filtering — Quiz

Test your understanding of collaborative filtering and recommendation systems. Questions mix multiple-choice, true/false, and short-answer formats.

---

**Q1. (Multiple Choice)**
What is the key difference between user-based and item-based collaborative filtering?

- A) User-based uses content features; item-based uses ratings only.
- B) User-based computes similarity between users; item-based computes similarity between items.
- C) User-based is model-based; item-based is memory-based.
- D) Item-based requires explicit feedback; user-based works with implicit feedback.

---

**Q2. (Short Answer)**
Define the **cold start problem** in the context of collaborative filtering. Give one example each for a new user and a new item.

---

**Q3. (Multiple Choice)**
In matrix factorization, the rating matrix R (m users × n items) is approximated as:

- A) R ≈ P + Q where P and Q are diagonal matrices
- B) R ≈ P × Qᵀ where P is m × k and Q is n × k
- C) R ≈ P² where P is a square user-item matrix
- D) R ≈ sigmoid(U × V) where U and V are the user and item embeddings

---

**Q4. (True/False)**
Item-based collaborative filtering is generally more scalable than user-based collaborative filtering because item catalogs are typically smaller and more stable than user populations.

---

**Q5. (Multiple Choice)**
Which of the following is an example of **implicit feedback**?

- A) A user rates a movie 4 out of 5 stars.
- B) A user writes a review saying "I loved this product."
- C) A user streams a song 15 times this month.
- D) A user clicks "Dislike" on a video.

---

**Q6. (Short Answer)**
What do the **latent factors** in matrix factorization represent? Are they human-interpretable? Give an example of what a latent factor might capture in a movie recommendation system.

---

**Q7. (Multiple Choice)**
Precision@K is defined as:

- A) The proportion of all relevant items that appear in the top-K recommendations.
- B) The fraction of top-K recommended items that are actually relevant to the user.
- C) The mean squared error between true ratings and predicted ratings for top-K items.
- D) The number of relevant items in the top-K recommendation list.

---

**Q8. (True/False)**
In a typical real-world recommendation scenario (e.g., Netflix, Amazon), the user-item rating matrix is very dense — most users have rated most items.

---

**Q9. (Multiple Choice)**
Which of the following BEST describes the sparsity problem in collaborative filtering?

- A) The rating matrix has too many users compared to items.
- B) Most entries in the rating matrix are missing because users only rate a small fraction of all available items.
- C) The latent factor matrices P and Q have too few factors.
- D) Cosine similarity fails when feature vectors are orthogonal.

---

**Q10. (Short Answer)**
Explain how SVD (Singular Value Decomposition) relates to matrix factorization for recommendation systems. Why is direct SVD not straightforward when the rating matrix has missing values?

---

**Q11. (Multiple Choice)**
You have a recommendation system and want to measure how good your predictions of actual star ratings are. Which metric is most appropriate?

- A) Precision@K
- B) Recall@K
- C) Root Mean Squared Error (RMSE)
- D) F1 Score

---

**Q12. (Short Answer)**
A startup is launching a new music streaming platform. They have no user rating history yet but they have detailed metadata for each song (genre, artist, tempo, mood). Describe a strategy to handle recommendations on day one, when collaborative filtering would fail.

---

<details>
<summary><strong>Answer Key (click to expand)</strong></summary>

**A1.** B — User-based CF finds users with similar rating patterns to the target user and recommends items those similar users liked. Item-based CF identifies items similar to what the target user has already liked (based on how all users rate those items) and recommends similar items. Both are memory-based approaches that use the rating matrix directly.

**A2.** The cold start problem arises when a collaborative filtering system has insufficient interaction data to make recommendations.
- **New user example:** A newly registered user has not rated any items. There is no history to compare against other users, so the system cannot identify similar users or generate personalized recommendations.
- **New item example:** A newly added product has received no ratings from any user. The item never appears as a candidate in similarity computations, so it will never be recommended regardless of its actual quality.

**A3.** B — Matrix factorization decomposes the rating matrix R into two low-rank factor matrices: `R ≈ P × Qᵀ` where P (m × k) represents user latent factors and Q (n × k) represents item latent factors, with k much smaller than m or n. The predicted rating for user u and item i is the dot product of their respective latent vectors.

**A4.** True — User populations can be millions and change continuously (new users, shifting preferences). Item catalogs, while large, tend to be more stable and finite. The item-item similarity matrix can be precomputed offline once and updated incrementally when new items are added. Additionally, since there are typically fewer items than users, the item-item matrix is smaller and faster to compute.

**A5.** C — Streaming a song 15 times is **implicit feedback** — the user has not explicitly stated they like it, but their repeated listening behavior implies a preference. Options A, B, and D are all explicit feedback: the user deliberately communicated their opinion.

**A6.** Latent factors are low-dimensional continuous representations (embeddings) learned by the factorization algorithm. They are **not directly human-interpretable** — the algorithm optimizes them to minimize reconstruction error, not to align with meaningful human concepts. However, they implicitly capture underlying structure. In a movie recommender, one latent factor might correspond to a "preference for action films" (high values for action movies and action-loving users), another to "preference for drama," another to "art house vs. blockbuster taste," etc. — though these labels are never explicitly assigned.

**A7.** B — Precision@K measures the **fraction of recommended items (in the top K) that are actually relevant** to the user: `|{relevant} ∩ {top-K}| / K`. Option A describes Recall@K.

**A8.** False — Real-world rating matrices are extremely **sparse**. On Netflix, a user might rate 50-200 movies out of a catalog of 10,000+. On Amazon, a typical user interacts with a tiny fraction of millions of products. Sparsity ratios of 99%+ (fewer than 1% of entries observed) are common. This sparsity makes similarity computation challenging and amplifies the cold-start problem.

**A9.** B — The sparsity problem: most users rate only a small fraction of available items, leaving the vast majority of the rating matrix as missing values. This makes it difficult to find users (or items) with enough overlapping ratings to compute reliable similarity scores, and it limits the effectiveness of memory-based CF methods.

**A10.** Classic SVD factorizes a complete matrix R = UΣVᵀ. For recommendation, we want a truncated SVD using only the top-k singular values, giving `R ≈ U_k Σ_k V_kᵀ`. The problem is that SVD requires a **fully observed matrix** — it cannot handle missing values (NaN) directly. Imputing missing values with zeros or the mean before applying SVD introduces bias. Modern approaches use **Alternating Least Squares (ALS)** or **SGD** that optimize only over the observed entries, effectively computing an implicit factorization that handles missing data correctly.

**A11.** C — RMSE (Root Mean Squared Error) measures how close predicted numerical ratings are to actual star ratings. It is the standard metric for rating prediction tasks. Precision@K and Recall@K are used for ranking/top-N recommendation tasks, not for evaluating rating accuracy. F1 Score is for binary classification.

**A12.** On day one, collaborative filtering fails because there is no user-item interaction history. A practical strategy:
1. **Content-Based Filtering:** Use the song metadata (genre, artist, tempo, mood) to represent each song as a feature vector. Recommend songs similar in feature space to songs the user has expressed interest in (if any exists, e.g., from onboarding).
2. **Onboarding questionnaire:** Ask new users to rate or select a few songs/artists/genres they like. Use this to seed their profile.
3. **Popularity-based recommendations:** Recommend globally popular songs to all new users as a safe default.
4. **Hybrid transition:** As the user accumulates interactions, gradually shift weight from content-based to collaborative filtering. Once enough data exists, CF can take over.
This strategy ensures useful recommendations from day one and improves progressively as more data is collected.

</details>
