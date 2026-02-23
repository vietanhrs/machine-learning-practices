"""
Exercise 02: Item-Based Collaborative Filtering
================================================

Goal: Implement item-based collaborative filtering and compare its performance
      with user-based CF on the same synthetic dataset.

Learning Objectives:
    - Build an item-item similarity matrix (transposed perspective).
    - Predict ratings using similar items the user has already rated.
    - Generate a top-N recommendation list for a user.
    - Empirically compare user-based vs. item-based CF using RMSE.

Background:
    Item-Based CF workflow:
        1. Represent each item as a column vector of user ratings.
        2. Compute cosine similarity between every pair of items.
        3. For a target (user u, item i):
           - Find the k items most similar to i that user u has already rated.
           - Predict the rating as a similarity-weighted average of u's ratings
             for those similar items.

    Why item-based can be more stable:
        - Item catalogs change slowly; user preferences shift over time.
        - The item-item similarity matrix can be precomputed and cached.

Instructions:
    - Fill in every section marked with TODO.
    - Reuse the cosine_similarity() function from Exercise 01 (copy it here).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Similarity (copy / reimport from exercise 01)
# ---------------------------------------------------------------------------

def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """
    Compute cosine similarity between two 1-D vectors, ignoring NaN positions.

    Parameters
    ----------
    u, v : np.ndarray — rating vectors (NaN = unrated)

    Returns
    -------
    float — cosine similarity, or 0.0 if no common non-NaN entries exist

    TODO:
        - (Same as Exercise 01 — re-implement or copy your solution.)
        - Mask positions where both u and v are non-NaN.
        - Return dot(u_common, v_common) / (||u_common|| * ||v_common||).
        - Return 0.0 if mask is empty or either norm is zero.
    """
    # TODO: implement cosine_similarity (see Exercise 01 for hints)
    raise NotImplementedError("Implement cosine_similarity()")


# ---------------------------------------------------------------------------
# Item-Item Similarity
# ---------------------------------------------------------------------------

def compute_item_similarity_matrix(ratings: np.ndarray) -> np.ndarray:
    """
    Build the full item-item cosine similarity matrix.

    Each ITEM is represented by its column in the ratings matrix — a vector
    of all users' ratings for that item (NaN where not rated).

    Parameters
    ----------
    ratings : np.ndarray, shape (n_users, n_items) — NaN = unrated

    Returns
    -------
    np.ndarray, shape (n_items, n_items)
        Symmetric similarity matrix where sim[i, j] is the cosine similarity
        between item i's and item j's rating vectors.

    TODO:
        - Transpose the ratings matrix so rows = items, columns = users:
              R_T = ratings.T   (shape: n_items × n_users)
        - Compute pairwise cosine_similarity between all rows of R_T.
        - This is structurally identical to compute_user_similarity_matrix()
          from Exercise 01, but applied to items instead of users.
        Hint: for i in range(n_items): for j in range(i+1, n_items): ...
    """
    # TODO: transpose ratings, compute all pairwise item similarities
    raise NotImplementedError("Implement compute_item_similarity_matrix()")


# ---------------------------------------------------------------------------
# Rating Prediction (Item-Based)
# ---------------------------------------------------------------------------

def predict_rating_item_based(
    user_id: int,
    item_id: int,
    ratings: np.ndarray,
    item_similarity: np.ndarray,
    k: int = 5,
) -> float:
    """
    Predict user_id's rating for item_id using k most similar items
    that user_id has already rated.

    r̂(u, i) = Σ_{j ∈ N_k(i,u)} sim(i, j) · r(u, j)
               ─────────────────────────────────────
               Σ_{j ∈ N_k(i,u)} |sim(i, j)|

    where N_k(i, u) = k items most similar to i that user u has rated.

    Parameters
    ----------
    user_id         : int
    item_id         : int
    ratings         : np.ndarray, shape (n_users, n_items)
    item_similarity : np.ndarray, shape (n_items, n_items)
    k               : int — number of similar items to use

    Returns
    -------
    float — predicted rating, or np.nan if the user has rated no similar items

    TODO:
        1. Get the similarity row for item_id from item_similarity.
        2. Set self-similarity (sim[item_id]) to -inf.
        3. Find the k items with highest similarity.
        4. Filter: keep only those items that user_id has rated
           (ratings[user_id, j] is not NaN).
        5. If no valid items remain, return np.nan.
        6. Compute the weighted average rating.
    """
    # TODO: predict rating using top-k similar items the user has rated
    raise NotImplementedError("Implement predict_rating_item_based()")


# ---------------------------------------------------------------------------
# Recommendation List
# ---------------------------------------------------------------------------

def recommend_items(
    user_id: int,
    ratings: np.ndarray,
    item_similarity: np.ndarray,
    n: int = 10,
    k: int = 5,
) -> list[tuple[int, float]]:
    """
    Generate a top-N item recommendation list for user_id.

    Recommend only items the user has NOT yet rated.

    Parameters
    ----------
    user_id         : int
    ratings         : np.ndarray, shape (n_users, n_items)
    item_similarity : np.ndarray, shape (n_items, n_items)
    n               : int — number of recommendations to return
    k               : int — number of similar items used per prediction

    Returns
    -------
    list of (item_id, predicted_rating) sorted by predicted rating descending

    TODO:
        1. Identify items that user_id has NOT rated (ratings[user_id, :] is NaN).
        2. For each unrated item, call predict_rating_item_based().
        3. Collect all (item_id, predicted_rating) pairs where the prediction
           is not NaN.
        4. Sort by predicted_rating descending and return the top n.
    """
    # TODO: collect predicted ratings for unrated items and return top-n
    raise NotImplementedError("Implement recommend_items()")


# ---------------------------------------------------------------------------
# Comparison: User-Based vs. Item-Based
# ---------------------------------------------------------------------------

def compare_user_vs_item_based(
    ratings: np.ndarray,
    test_set: list[tuple[int, int, float]],
    k: int = 5,
) -> None:
    """
    Compute and compare RMSE of user-based vs. item-based CF on the same test set.

    Parameters
    ----------
    ratings  : np.ndarray — training rating matrix (NaN = unrated or held-out)
    test_set : list of (user_id, item_id, true_rating) triples
    k        : int — number of neighbours

    TODO:
        1. Compute the user-user similarity matrix.
        2. Compute the item-item similarity matrix.
        3. For each (user_id, item_id, true_rating) in test_set:
           a. Predict with user-based CF.
           b. Predict with item-based CF.
        4. Compute RMSE for each method.
        5. Print a comparison table:
               Method       | RMSE   | Coverage (% of test pairs with a prediction)
               ─────────────────────────────────────────────────────────────
               User-Based   | 0.xxx  | xx%
               Item-Based   | 0.xxx  | xx%
        Hint: Coverage = fraction of test triples where prediction != NaN.
    """
    # TODO: compute both similarity matrices, predict on test set, print RMSE comparison
    raise NotImplementedError("Implement compare_user_vs_item_based()")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Demonstrate item-based CF and compare with user-based CF.

    Setup: same synthetic 10-user × 20-item rating matrix as Exercise 01.
    """
    np.random.seed(42)
    n_users, n_items = 10, 20

    # Generate synthetic rating matrix
    ratings = np.random.uniform(1.0, 5.0, size=(n_users, n_items))
    mask = np.random.rand(n_users, n_items) < 0.4
    ratings[mask] = np.nan

    # Hold out test ratings
    ratings_train = ratings.copy()
    test_set = []
    observed = [(u, i) for u in range(n_users) for i in range(n_items)
                if not np.isnan(ratings[u, i])]
    hold_out_n = max(1, len(observed) // 10)
    hold_out_idx = np.random.choice(len(observed), size=hold_out_n, replace=False)
    for idx in hold_out_idx:
        u, i = observed[idx]
        test_set.append((u, i, ratings[u, i]))
        ratings_train[u, i] = np.nan

    print("=== Item-Based CF ===")
    print("Computing item-item similarity matrix ...")
    item_sim = compute_item_similarity_matrix(ratings_train)

    # Sample prediction
    sample_u, sample_i = 0, 5
    pred = predict_rating_item_based(sample_u, sample_i, ratings_train, item_sim, k=5)
    print(f"\nPredicted rating (item-based) for User {sample_u}, Item {sample_i}: {pred:.2f}")

    # Recommendation list for user 0
    print(f"\nTop-5 recommendations for User 0:")
    recs = recommend_items(0, ratings_train, item_sim, n=5, k=5)
    for item_id, pred_rating in recs:
        print(f"  Item {item_id:2d}: predicted rating = {pred_rating:.2f}")

    # Comparison
    print("\n=== User-Based vs. Item-Based RMSE Comparison ===")
    compare_user_vs_item_based(ratings_train, test_set, k=5)


if __name__ == "__main__":
    main()
