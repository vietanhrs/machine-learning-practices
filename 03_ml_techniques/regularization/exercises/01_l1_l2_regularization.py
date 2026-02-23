"""
Exercise 01: L1, L2, and Elastic Net Regularization
=====================================================
Implement and compare L1, L2, and Elastic Net regularization by training
linear regression models from scratch using gradient descent. Observe
how different penalties affect weight sparsity and shrinkage.

Learning objectives:
- Implement regularized loss functions and their gradients
- Observe L1 sparsity vs L2 shrinkage in practice
- Compare weight evolution under different regularization schemes

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Regularized Loss Functions
# ---------------------------------------------------------------------------

def ridge_loss(y_true, y_pred, weights, lambda_):
    """
    Ridge (L2) Loss: MSE + L2 penalty on weights.

    Formula:
        L_ridge = (1/N) * ||y_true - y_pred||² + lambda_ * ||w||²

    Args:
        y_true (np.ndarray): Ground truth values, shape (N,)
        y_pred (np.ndarray): Predicted values, shape (N,)
        weights (np.ndarray): Model weights (exclude bias), shape (D,)
        lambda_ (float): Regularization strength

    Returns:
        float: Ridge loss value

    Hints:
        - mse = np.mean((y_true - y_pred) ** 2)
        - l2_penalty = lambda_ * np.sum(weights ** 2)
        - Return mse + l2_penalty
        - Note: only regularize weights, not the bias term
    """
    # TODO: Compute MSE term
    # TODO: Compute L2 penalty (sum of squared weights)
    # TODO: Return MSE + lambda_ * L2 penalty
    pass


def ridge_gradient(X, y_true, weights, bias, lambda_):
    """
    Gradient of Ridge loss with respect to weights and bias.

    ∂L_ridge/∂w = (2/N) * Xᵀ(Xw + bias - y) + 2 * lambda_ * w
    ∂L_ridge/∂b = (2/N) * Σ(Xw + bias - y)

    Args:
        X (np.ndarray): Feature matrix, shape (N, D)
        y_true (np.ndarray): Ground truth, shape (N,)
        weights (np.ndarray): Current weights, shape (D,)
        bias (float): Current bias
        lambda_ (float): Regularization strength

    Returns:
        tuple: (grad_w, grad_b) — gradients for weights and bias

    Hints:
        - predictions = X @ weights + bias
        - residuals = predictions - y_true
        - grad_w = (2/N) * X.T @ residuals + 2 * lambda_ * weights
        - grad_b = (2/N) * np.sum(residuals)
        - Bias is NOT regularized (no lambda_ term in grad_b)
    """
    # TODO: Compute predictions
    # TODO: Compute residuals
    # TODO: Compute gradient for weights (include L2 penalty gradient)
    # TODO: Compute gradient for bias (no regularization)
    # TODO: Return (grad_w, grad_b)
    pass


def lasso_loss(y_true, y_pred, weights, lambda_):
    """
    Lasso (L1) Loss: MSE + L1 penalty on weights.

    Formula:
        L_lasso = (1/N) * ||y_true - y_pred||² + lambda_ * ||w||₁

    Args:
        y_true (np.ndarray): Ground truth values, shape (N,)
        y_pred (np.ndarray): Predicted values, shape (N,)
        weights (np.ndarray): Model weights (exclude bias), shape (D,)
        lambda_ (float): Regularization strength

    Returns:
        float: Lasso loss value

    Hints:
        - mse = np.mean((y_true - y_pred) ** 2)
        - l1_penalty = lambda_ * np.sum(np.abs(weights))
        - Return mse + l1_penalty
    """
    # TODO: Compute MSE term
    # TODO: Compute L1 penalty (sum of absolute weights)
    # TODO: Return MSE + lambda_ * L1 penalty
    pass


def lasso_gradient(X, y_true, weights, bias, lambda_):
    """
    Subgradient of Lasso loss with respect to weights and bias.

    ∂L_lasso/∂w ≈ (2/N) * Xᵀ(Xw + bias - y) + lambda_ * sign(w)
    ∂L_lasso/∂b = (2/N) * Σ(Xw + bias - y)

    Args:
        X (np.ndarray): Feature matrix, shape (N, D)
        y_true (np.ndarray): Ground truth, shape (N,)
        weights (np.ndarray): Current weights, shape (D,)
        bias (float): Current bias
        lambda_ (float): Regularization strength

    Returns:
        tuple: (grad_w, grad_b)

    Hints:
        - predictions = X @ weights + bias
        - residuals = predictions - y_true
        - grad_w = (2/N) * X.T @ residuals + lambda_ * np.sign(weights)
        - np.sign(0) = 0, so zero weights have no L1 gradient → they can stay at 0
        - grad_b = (2/N) * np.sum(residuals)
    """
    # TODO: Compute predictions and residuals
    # TODO: Compute weight gradient with L1 subgradient (np.sign)
    # TODO: Compute bias gradient (no regularization)
    # TODO: Return (grad_w, grad_b)
    pass


def elastic_net_loss(y_true, y_pred, weights, lambda_, alpha):
    """
    Elastic Net Loss: MSE + weighted combination of L1 and L2 penalties.

    Formula:
        L_elastic = (1/N)*||y - ŷ||² + lambda_*(alpha*||w||₁ + (1-alpha)*||w||²)

    Args:
        y_true (np.ndarray): Ground truth values, shape (N,)
        y_pred (np.ndarray): Predicted values, shape (N,)
        weights (np.ndarray): Model weights (exclude bias), shape (D,)
        lambda_ (float): Overall regularization strength
        alpha (float): L1/L2 mixing parameter in [0, 1]. alpha=1 → pure Lasso, alpha=0 → pure Ridge

    Returns:
        float: Elastic Net loss value

    Hints:
        - mse = np.mean((y_true - y_pred) ** 2)
        - l1_penalty = alpha * np.sum(np.abs(weights))
        - l2_penalty = (1 - alpha) * np.sum(weights ** 2)
        - Return mse + lambda_ * (l1_penalty + l2_penalty)
    """
    # TODO: Compute MSE term
    # TODO: Compute L1 penalty component
    # TODO: Compute L2 penalty component
    # TODO: Return MSE + lambda_ * (alpha * L1 + (1-alpha) * L2)
    pass


def elastic_net_gradient(X, y_true, weights, bias, lambda_, alpha):
    """
    Gradient of Elastic Net loss.

    ∂L/∂w = (2/N)*Xᵀ(Xw+b-y) + lambda_*(alpha*sign(w) + 2*(1-alpha)*w)
    ∂L/∂b = (2/N)*Σ(Xw+b-y)

    Args:
        X, y_true, weights, bias, lambda_, alpha: as above

    Returns:
        tuple: (grad_w, grad_b)

    Hints:
        - MSE gradient for weights: (2/N) * X.T @ (X @ weights + bias - y_true)
        - L1 gradient component: alpha * np.sign(weights)
        - L2 gradient component: 2 * (1 - alpha) * weights
        - Total: mse_grad + lambda_ * (l1_grad + l2_grad)
    """
    # TODO: Compute MSE gradient
    # TODO: Compute L1 gradient component (sign)
    # TODO: Compute L2 gradient component (2w)
    # TODO: Combine and return (grad_w, grad_b)
    pass


# ---------------------------------------------------------------------------
# Training with Regularization
# ---------------------------------------------------------------------------

def train_with_regularization(X, y, regularization_type, lambda_, lr=0.01, n_iters=500):
    """
    Train a linear regression model with specified regularization using gradient descent.

    Args:
        X (np.ndarray): Feature matrix, shape (N, D)
        y (np.ndarray): Target values, shape (N,)
        regularization_type (str): One of "l1", "l2", "elastic_net", "none"
        lambda_ (float): Regularization strength
        lr (float): Learning rate
        n_iters (int): Number of gradient descent iterations

    Returns:
        dict: {
            "weights": final weight vector,
            "bias": final bias,
            "loss_history": list of loss values,
            "weight_history": list of weight snapshots
        }

    Hints:
        - Initialize weights = np.zeros(X.shape[1]), bias = 0.0
        - Select appropriate gradient function based on regularization_type
        - For "elastic_net", use alpha=0.5 as default
        - Record loss and weights at every 10th iteration
        - Use ridge_gradient, lasso_gradient, or elastic_net_gradient
    """
    # TODO: Initialize weights and bias
    # TODO: Initialize loss_history and weight_history lists
    # TODO: Select gradient function based on regularization_type
    # TODO: For n_iters:
    #   - Compute gradient
    #   - Update weights and bias
    #   - Record loss every 10 iterations
    # TODO: Return results dict
    pass


# ---------------------------------------------------------------------------
# Visualization (Complete — Do Not Modify)
# ---------------------------------------------------------------------------

def plot_weight_evolution(lambda_values, weight_histories, feature_names=None, title="Weight Evolution"):
    """
    Plot how each weight changes as lambda increases, for multiple features.

    Args:
        lambda_values (list): Lambda values tested
        weight_histories (dict): {lambda_: final_weights}
        feature_names (list or None): Names for each feature
        title (str): Plot title
    """
    weight_matrix = np.array([weight_histories[l] for l in lambda_values])
    n_features = weight_matrix.shape[1]

    plt.figure(figsize=(12, 5))
    for i in range(n_features):
        label = feature_names[i] if feature_names else f"Feature {i}"
        plt.plot(lambda_values, weight_matrix[:, i], marker='o', label=label)

    plt.xlabel("Regularization Strength (λ)")
    plt.ylabel("Weight Value")
    plt.title(title)
    plt.xscale('log')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_regularization_effects(X, y, feature_names=None):
    """
    Train linear regression with L1, L2, and Elastic Net across multiple
    lambda values. Plot and compare the resulting weight vectors.

    Args:
        X (np.ndarray): Feature matrix, shape (N, D)
        y (np.ndarray): Target values, shape (N,)
        feature_names (list or None): Feature names for plotting

    Hints:
        - Test lambda_ values: np.logspace(-4, 1, 15)  [from 0.0001 to 10]
        - For each lambda_:
            - Train with L1, L2, ElasticNet regularization
            - Record final weights
        - Call plot_weight_evolution for each regularization type
        - Observe: L1 drives weights to exactly 0 (sparse); L2 shrinks but keeps all non-zero
        - Also print: how many weights are near-zero (< 1e-3) for each method at highest lambda_
    """
    # TODO: Define lambda values to test (log scale)
    # TODO: For each lambda, train L1, L2, and ElasticNet
    # TODO: Collect weight histories for plotting
    # TODO: Call plot_weight_evolution for each regularization type
    # TODO: Print sparsity comparison table
    pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    Demonstrate regularization effects on a linear regression problem
    with many features, some of which are irrelevant (noise).

    Steps:
    1. Generate data: y depends on only 5 out of 20 features
    2. Add 15 irrelevant noise features
    3. Train with L1, L2, ElasticNet across different lambda values
    4. Compare weight vectors — L1 should zero out noise features
    5. Plot weight evolution as lambda increases
    """
    print("=" * 60)
    print("L1, L2, and Elastic Net Regularization Comparison")
    print("=" * 60)

    np.random.seed(42)
    N = 200
    D_relevant = 5
    D_noise = 15
    D = D_relevant + D_noise

    # Generate features
    X = np.random.randn(N, D)

    # True weights: only first 5 features matter
    true_weights = np.array([3.0, -2.5, 1.5, -1.0, 2.0] + [0.0] * D_noise)
    y = X @ true_weights + 0.5 * np.random.randn(N)

    # Feature names
    feature_names = [f"Relevant_{i}" for i in range(D_relevant)] + \
                    [f"Noise_{i}" for i in range(D_noise)]

    print(f"Dataset: {N} samples, {D} features ({D_relevant} relevant, {D_noise} noise)")
    print(f"True non-zero weights: {true_weights[:D_relevant]}")

    # TODO: Call compare_regularization_effects(X, y, feature_names)

    # TODO: Also plot a single training run to show loss convergence
    # Hint:
    # for reg_type in ["none", "l1", "l2", "elastic_net"]:
    #     result = train_with_regularization(X, y, reg_type, lambda_=0.1)
    #     plt.plot(result["loss_history"], label=reg_type)
    # plt.show()

    print("\nDone. Observe that L1 zeros out noise features while L2 only shrinks them.")


if __name__ == "__main__":
    main()
