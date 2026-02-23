"""
Exercise 03: Matrix Factorization for Collaborative Filtering
==============================================================

Goal: Implement a matrix factorization model trained with Stochastic Gradient
      Descent (SGD) to reconstruct a sparse user-item rating matrix.

Learning Objectives:
    - Understand the latent factor decomposition R ≈ P × Qᵀ.
    - Implement SGD that updates only on OBSERVED (non-NaN) entries.
    - Include L2 regularization to prevent latent factors from growing unbounded.
    - Evaluate reconstruction quality using RMSE on held-out ratings.
    - Visualize training loss convergence.

Background:
    For each observed (user u, item i) rating r_{u,i}:
        Prediction: r̂_{u,i} = p_u · q_i  (dot product of latent vectors)
        Error:      e_{u,i} = r_{u,i} - r̂_{u,i}

    SGD Update (with L2 regularization λ):
        p_u ← p_u + η · (e_{u,i} · q_i − λ · p_u)
        q_i ← q_i + η · (e_{u,i} · p_u − λ · q_i)

    Loss (over all observed entries O):
        L = Σ_{(u,i)∈O} e_{u,i}² + λ (||p_u||² + ||q_i||²)

    n_factors (k) controls the rank of the approximation:
        - Too small: underfitting (cannot capture the complexity of preferences).
        - Too large:  overfitting (fits noise in the sparse training data).

Instructions:
    - Fill in every section marked with TODO.
    - Do NOT use any CF or matrix factorization libraries.
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_rmse(
    R_true: np.ndarray,
    R_pred: np.ndarray,
    mask: np.ndarray,
) -> float:
    """
    Compute RMSE only over entries specified by mask.

    Parameters
    ----------
    R_true : np.ndarray, shape (n_users, n_items) — true ratings
    R_pred : np.ndarray, shape (n_users, n_items) — predicted ratings
    mask   : np.ndarray, shape (n_users, n_items) — boolean; True = evaluate here

    Returns
    -------
    float — RMSE over masked entries

    TODO:
        - Compute squared errors: (R_true[mask] - R_pred[mask])²
        - Return the square root of the mean squared error.
        - Return np.nan if mask has no True entries.
    """
    # TODO: compute RMSE over entries where mask is True
    raise NotImplementedError("Implement compute_rmse()")


# ---------------------------------------------------------------------------
# Matrix Factorization Model
# ---------------------------------------------------------------------------

class MatrixFactorization:
    """
    Matrix Factorization via Stochastic Gradient Descent.

    Decomposes rating matrix R (m users × n items) into:
        R ≈ P × Qᵀ
    where P is (m × k) and Q is (n × k), with k = n_factors.

    Parameters
    ----------
    n_factors : int   — number of latent factors (rank of approximation)
    lr        : float — SGD learning rate (step size)
    reg       : float — L2 regularization strength λ
    n_epochs  : int   — number of full passes over all observed ratings
    """

    def __init__(
        self,
        n_factors: int = 10,
        lr: float = 0.01,
        reg: float = 0.1,
        n_epochs: int = 50,
    ) -> None:
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs

        self.P: np.ndarray | None = None   # user factor matrix (n_users, n_factors)
        self.Q: np.ndarray | None = None   # item factor matrix (n_items, n_factors)
        self.training_loss_curve: list[float] = []

    def fit(self, R: np.ndarray) -> "MatrixFactorization":
        """
        Train the model using SGD on observed entries of R.

        Parameters
        ----------
        R : np.ndarray, shape (n_users, n_items)
            Rating matrix with NaN for missing entries.

        Returns
        -------
        self

        TODO:
            1. Determine n_users and n_items from R.shape.

            2. Initialize P and Q with small random values:
               P = np.random.normal(0, 0.1, (n_users, n_factors))
               Q = np.random.normal(0, 0.1, (n_items, n_factors))

            3. Collect all observed (u, i) index pairs:
                   observed = [(u, i) for u in range(n_users)
                               for i in range(n_items)
                               if not np.isnan(R[u, i])]

            4. For each epoch in range(n_epochs):
               a. Shuffle the observed list to ensure stochastic updates.
               b. For each (u, i) in the shuffled list:
                    * Compute prediction: r_hat = P[u] @ Q[i]
                    * Compute error: e = R[u, i] - r_hat
                    * Update P and Q WITH regularization:
                          P[u] += lr * (e * Q[i] - reg * P[u])
                          Q[i] += lr * (e * P[u] - reg * Q[i])
                      IMPORTANT: compute e BEFORE updating P[u], then use
                      the OLD P[u] value in the Q[i] update. Or store a copy.
               c. Compute the total training loss over all observed entries:
                      predictions for all observed (u, i) using current P, Q
                      loss = Σ e² + reg * (||P||_F² + ||Q||_F²)
                  Append to self.training_loss_curve.

            5. Store final P and Q in self.P and self.Q.
            6. Return self.

        Tips:
            - A small learning_rate (0.005–0.02) usually works well.
            - More epochs = better fit but may overfit; monitor loss curve.
            - The Frobenius norm of a matrix A is np.sum(A**2).
        """
        # TODO: implement SGD matrix factorization training
        raise NotImplementedError("Implement MatrixFactorization.fit()")

    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict the rating for a single (user, item) pair.

        Parameters
        ----------
        user_id : int
        item_id : int

        Returns
        -------
        float — predicted rating = P[user_id] · Q[item_id]

        TODO:
            - Return the dot product of the user's and item's latent vectors.
            - Optionally clip to the valid rating range (e.g., [1, 5]).
        """
        # TODO: return dot product of user and item latent factors
        raise NotImplementedError("Implement predict()")

    def reconstruct(self) -> np.ndarray:
        """
        Reconstruct the full predicted rating matrix.

        Returns
        -------
        np.ndarray, shape (n_users, n_items) — full predicted rating matrix

        TODO:
            - Return the matrix product P @ Q.T.
            - This fills in ALL entries, including originally missing ones.
        """
        # TODO: return P @ Q.T as the full reconstructed rating matrix
        raise NotImplementedError("Implement reconstruct()")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Full matrix factorization demonstration:
        1. Generate a sparse rating matrix.
        2. Train the MatrixFactorization model.
        3. Plot the training loss curve.
        4. Compute train and test RMSE.
        5. Visualize the original vs. reconstructed rating matrix.
    """
    np.random.seed(42)
    n_users, n_items = 20, 30

    # Generate synthetic low-rank rating matrix (true rank = 5)
    # R_true = A @ B.T where A and B are (n, 5) matrices
    A = np.random.normal(0, 1, (n_users, 5))
    B = np.random.normal(0, 1, (n_items, 5))
    R_full = A @ B.T
    # Shift to rating scale [1, 5]
    R_full = 1 + 4 * (R_full - R_full.min()) / (R_full.max() - R_full.min())

    # Create sparse version: ~50% missing
    R_sparse = R_full.copy()
    missing_mask = np.random.rand(n_users, n_items) < 0.5
    R_sparse[missing_mask] = np.nan

    # Hold out 10% of observed entries for testing
    observed = [(u, i) for u in range(n_users) for i in range(n_items)
                if not np.isnan(R_sparse[u, i])]
    hold_out_n = max(1, len(observed) // 10)
    hold_out_idx = np.random.choice(len(observed), size=hold_out_n, replace=False)

    R_train = R_sparse.copy()
    test_mask = np.zeros((n_users, n_items), dtype=bool)
    for idx in hold_out_idx:
        u, i = observed[idx]
        R_train[u, i] = np.nan
        test_mask[u, i] = True

    train_mask = ~np.isnan(R_train)
    print(f"Training entries : {train_mask.sum()}")
    print(f"Test entries     : {test_mask.sum()}")
    print(f"Sparsity         : {np.isnan(R_train).mean():.1%}\n")

    # Train model
    print("Training Matrix Factorization (k=5, 100 epochs) ...")
    mf = MatrixFactorization(n_factors=5, lr=0.01, reg=0.05, n_epochs=100)
    mf.fit(R_train)

    # Training loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(mf.training_loss_curve, color="steelblue", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss (RMSE + Regularization)")
    plt.title("Matrix Factorization — Training Loss Curve")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("mf_training_loss.png", dpi=120)
    plt.show()
    print("Training loss curve saved to mf_training_loss.png\n")

    # Evaluate
    R_pred = mf.reconstruct()
    train_rmse = compute_rmse(R_full, R_pred, train_mask)
    test_rmse = compute_rmse(R_full, R_pred, test_mask)
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test  RMSE: {test_rmse:.4f}")

    # Visualize original (sparse) vs. reconstructed
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    im0 = axes[0].imshow(R_full, aspect="auto", cmap="RdYlGn", vmin=1, vmax=5)
    axes[0].set_title("True Rating Matrix (Full)")
    axes[0].set_xlabel("Items")
    axes[0].set_ylabel("Users")
    plt.colorbar(im0, ax=axes[0])

    R_visible = R_sparse.copy()
    im1 = axes[1].imshow(R_visible, aspect="auto", cmap="RdYlGn", vmin=1, vmax=5)
    axes[1].set_title("Observed Ratings (Sparse)")
    axes[1].set_xlabel("Items")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(R_pred, aspect="auto", cmap="RdYlGn", vmin=1, vmax=5)
    axes[2].set_title("Reconstructed Matrix (MF)")
    axes[2].set_xlabel("Items")
    plt.colorbar(im2, ax=axes[2])

    plt.suptitle("Matrix Factorization: Rating Matrix Reconstruction", fontsize=13)
    plt.tight_layout()
    plt.savefig("mf_reconstruction.png", dpi=120)
    plt.show()
    print("Reconstruction plot saved to mf_reconstruction.png")


if __name__ == "__main__":
    main()
