"""
Exercise 01: User-Based Collaborative Filtering
================================================

Goal: Implement user-based collaborative filtering from scratch using
      cosine similarity to find similar users and predict ratings.

Learning Objectives:
    - Compute cosine similarity between user rating vectors.
    - Build a user-user similarity matrix.
    - Predict ratings using a weighted average of similar users' ratings.
    - Evaluate predictions using RMSE.

Background:
    User-Based CF workflow:
        1. Represent each user as a vector of item ratings (NaN for unrated).
        2. Compute cosine similarity between every pair of users.
        3. For a target (user, item) pair:
           - Find the k users most similar to the target user who have rated
             the item.
           - Predict the rating as a similarity-weighted average.
        4. Evaluate on held-out ratings using RMSE.

Instructions:
    - Fill in every section marked with TODO.
    - The ratings matrix is a NumPy array of shape (n_users, n_items).
    - NaN represents an unrated entry.
    - Do NOT use any CF libraries (e.g., Surprise) in your implementation.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """
    Compute cosine similarity between two 1-D vectors, ignoring NaN positions.

    Only indices where BOTH u and v are non-NaN are used for the computation.
    This handles missing ratings correctly.

    Parameters
    ----------
    u : np.ndarray, shape (n_items,) — rating vector for user u
    v : np.ndarray, shape (n_items,) — rating vector for user v

    Returns
    -------
    float — cosine similarity in [-1, 1], or 0.0 if no common ratings exist

    TODO:
        1. Find indices where both u and v are non-NaN:
               mask = ~np.isnan(u) & ~np.isnan(v)
        2. If no common indices exist (mask.sum() == 0), return 0.0.
        3. Extract u_common = u[mask], v_common = v[mask].
        4. Compute cosine similarity:
               dot = np.dot(u_common, v_common)
               norm_u = np.linalg.norm(u_common)
               norm_v = np.linalg.norm(v_common)
               if norm_u == 0 or norm_v == 0: return 0.0
               return dot / (norm_u * norm_v)
    """
    # TODO: compute cosine similarity over non-NaN common items only
    raise NotImplementedError("Implement cosine_similarity()")


def compute_user_similarity_matrix(ratings: np.ndarray) -> np.ndarray:
    """
    Build the full user-user cosine similarity matrix.

    Parameters
    ----------
    ratings : np.ndarray, shape (n_users, n_items) — rating matrix (NaN = unrated)

    Returns
    -------
    np.ndarray, shape (n_users, n_users)
        Symmetric matrix where sim[i, j] = cosine_similarity(ratings[i], ratings[j]).
        Diagonal entries are 1.0 (a user is perfectly similar to themselves).

    TODO:
        - Initialize a (n_users, n_users) zero matrix.
        - Fill upper triangle by calling cosine_similarity() for each (i, j) pair.
        - Mirror the upper triangle to the lower triangle (the matrix is symmetric).
        - Set the diagonal to 1.0.
        Hint: for i in range(n_users): for j in range(i+1, n_users): ...
    """
    # TODO: compute all pairwise user similarities and return the symmetric matrix
    raise NotImplementedError("Implement compute_user_similarity_matrix()")


# ---------------------------------------------------------------------------
# Nearest Neighbours
# ---------------------------------------------------------------------------

def find_top_k_similar_users(
    user_id: int,
    similarity_matrix: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Return the indices of the k most similar users to user_id (excluding self).

    Parameters
    ----------
    user_id          : int — index of the target user
    similarity_matrix: np.ndarray, shape (n_users, n_users)
    k                : int — number of neighbours to return

    Returns
    -------
    np.ndarray, shape (k,) — indices of the k most similar users, sorted by
                              similarity (most similar first)

    TODO:
        - Get the similarity row for user_id.
        - Set self-similarity to -inf so the user does not select themselves.
        - Use np.argsort in descending order and take the first k indices.
        Hint: np.argsort(sims)[::-1][:k]
    """
    # TODO: return indices of k nearest neighbours (excluding self)
    raise NotImplementedError("Implement find_top_k_similar_users()")


# ---------------------------------------------------------------------------
# Rating Prediction
# ---------------------------------------------------------------------------

def predict_rating(
    user_id: int,
    item_id: int,
    ratings: np.ndarray,
    similarity_matrix: np.ndarray,
    k: int = 5,
) -> float:
    """
    Predict the rating of user_id for item_id using a weighted average of
    the k most similar users who have rated item_id.

    r̂(u, i) = Σ_{v ∈ N_k(u,i)} sim(u,v) · r(v,i)
               ─────────────────────────────────────
               Σ_{v ∈ N_k(u,i)} |sim(u,v)|

    Parameters
    ----------
    user_id          : int
    item_id          : int
    ratings          : np.ndarray, shape (n_users, n_items)
    similarity_matrix: np.ndarray, shape (n_users, n_users)
    k                : int — number of neighbours

    Returns
    -------
    float — predicted rating, or np.nan if no neighbour has rated item_id

    TODO:
        1. Call find_top_k_similar_users() to get k candidates.
        2. Filter: keep only neighbours who have actually rated item_id
           (ratings[v, item_id] is not NaN).
        3. If no valid neighbours, return np.nan.
        4. Compute the weighted average:
               numerator   = sum of sim(u,v) * r(v, item_id)
               denominator = sum of |sim(u,v)|
               if denominator == 0: return np.nan
               return numerator / denominator
    """
    # TODO: compute weighted average rating prediction from k neighbours
    raise NotImplementedError("Implement predict_rating()")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_rmse(
    ratings_test: list[tuple[int, int, float]],
    ratings_train: np.ndarray,
    similarity_matrix: np.ndarray,
    k: int = 5,
) -> float:
    """
    Compute RMSE on a set of held-out (user, item, true_rating) triples.

    RMSE = √[ (1/n) Σ (r_true - r̂)² ]

    Parameters
    ----------
    ratings_test     : list of (user_id, item_id, true_rating) triples
    ratings_train    : np.ndarray — training rating matrix (NaN = unrated)
    similarity_matrix: np.ndarray
    k                : int

    Returns
    -------
    float — RMSE over all test triples where a prediction was possible
            (skip triples where predict_rating returns np.nan)

    TODO:
        - For each (user_id, item_id, true_rating) in ratings_test:
            * Call predict_rating(user_id, item_id, ...).
            * If the result is not NaN, accumulate (true_rating - pred)².
        - Return sqrt(mean of squared errors).
        - Handle the edge case where no predictions could be made (return np.nan).
    """
    # TODO: iterate over test triples, compute and return RMSE
    raise NotImplementedError("Implement evaluate_rmse()")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Demonstrate user-based CF on a small synthetic ratings matrix.

    Setup:
        - 10 users, 20 items
        - Ratings in range [1.0, 5.0], with ~40% of entries missing (NaN)
        - Hold out 10% of known ratings for test evaluation

    Steps:
        1. Generate a synthetic ratings matrix.
        2. Hold out some ratings as a test set.
        3. Compute the user similarity matrix on the training data.
        4. Predict a few specific (user, item) pairs and print the results.
        5. Compute RMSE on the test set.
        6. Print the top-3 most similar users for user 0.
    """
    np.random.seed(42)
    n_users, n_items = 10, 20

    # Generate synthetic ratings: 1-5 stars, 40% missing
    ratings = np.random.uniform(1.0, 5.0, size=(n_users, n_items))
    mask = np.random.rand(n_users, n_items) < 0.4
    ratings[mask] = np.nan

    print(f"Ratings matrix shape: {ratings.shape}")
    print(f"Sparsity: {np.isnan(ratings).mean():.1%} missing\n")

    # Create a copy for training; hold out some observed ratings for test
    ratings_train = ratings.copy()
    test_set = []

    observed = [(u, i) for u in range(n_users) for i in range(n_items)
                if not np.isnan(ratings[u, i])]
    hold_out_n = max(1, len(observed) // 10)
    hold_out_idx = np.random.choice(len(observed), size=hold_out_n, replace=False)

    for idx in hold_out_idx:
        u, i = observed[idx]
        test_set.append((u, i, ratings[u, i]))
        ratings_train[u, i] = np.nan  # remove from training

    print(f"Training ratings : {(~np.isnan(ratings_train)).sum()}")
    print(f"Test ratings     : {len(test_set)}\n")

    # Compute similarity
    print("Computing user-user similarity matrix ...")
    sim_matrix = compute_user_similarity_matrix(ratings_train)

    # Show top-3 neighbours for user 0
    print("\nTop-3 most similar users to User 0:")
    neighbours = find_top_k_similar_users(0, sim_matrix, k=3)
    for nb in neighbours:
        print(f"  User {nb}: similarity = {sim_matrix[0, nb]:.4f}")

    # Predict a sample rating
    sample_u, sample_i = 0, 5
    pred = predict_rating(sample_u, sample_i, ratings_train, sim_matrix, k=5)
    true_val = ratings[sample_u, sample_i]
    print(f"\nPredicted rating for User {sample_u}, Item {sample_i}: {pred:.2f}")
    print(f"True rating (if known):                           {true_val}")

    # RMSE on test set
    rmse = evaluate_rmse(test_set, ratings_train, sim_matrix, k=5)
    print(f"\nTest RMSE (k=5): {rmse:.4f}")


if __name__ == "__main__":
    main()
