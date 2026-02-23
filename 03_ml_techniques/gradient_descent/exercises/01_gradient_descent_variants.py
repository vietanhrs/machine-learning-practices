"""
Exercise 01: Gradient Descent Variants
========================================
Implement and compare Batch GD, SGD, and Mini-Batch GD on a 2D quadratic
loss surface. Visualize convergence paths to build intuition about the
trade-offs between the variants.

Learning objectives:
- Implement all three gradient descent variants from scratch
- Understand how batch size affects convergence stability and speed
- Visualize optimization trajectories on a known loss surface

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


# ---------------------------------------------------------------------------
# Loss Function: 2D Quadratic (Simulates Linear Regression on Synthetic Data)
# ---------------------------------------------------------------------------

def f(theta, X, y):
    """
    Compute MSE loss for linear regression: L(θ) = (1/N) ||Xθ - y||²

    Args:
        theta (np.ndarray): Parameter vector, shape (n_features,)
        X (np.ndarray): Feature matrix, shape (n_samples, n_features)
        y (np.ndarray): Target vector, shape (n_samples,)

    Returns:
        float: Mean squared error loss

    Hints:
        - predictions = X @ theta
        - residuals = predictions - y
        - loss = np.mean(residuals ** 2)
    """
    # TODO: Compute predictions using X and theta
    # TODO: Compute residuals (predictions - y)
    # TODO: Return mean of squared residuals
    pass


def df(theta, X, y):
    """
    Compute gradient of MSE loss with respect to theta.
    ∇L(θ) = (2/N) · Xᵀ · (Xθ - y)

    Args:
        theta (np.ndarray): Parameter vector, shape (n_features,)
        X (np.ndarray): Feature matrix, shape (n_samples, n_features)
        y (np.ndarray): Target vector, shape (n_samples,)

    Returns:
        np.ndarray: Gradient vector, same shape as theta

    Hints:
        - residuals = X @ theta - y
        - gradient = (2 / len(y)) * X.T @ residuals
    """
    # TODO: Compute residuals
    # TODO: Compute and return the gradient
    pass


# ---------------------------------------------------------------------------
# Gradient Descent Variants
# ---------------------------------------------------------------------------

def batch_gd(X, y, lr=0.01, n_iters=100, theta_init=None):
    """
    Batch Gradient Descent: use the full dataset to compute each gradient.

    Args:
        X (np.ndarray): Feature matrix, shape (n_samples, n_features)
        y (np.ndarray): Target vector, shape (n_samples,)
        lr (float): Learning rate (step size)
        n_iters (int): Number of gradient steps
        theta_init (np.ndarray or None): Initial parameters. If None, use zeros.

    Returns:
        tuple: (theta_final, trajectory)
            - theta_final: shape (n_features,)
            - trajectory: list of theta values at each iteration, for plotting

    Hints:
        - Initialize theta: np.zeros(X.shape[1]) if theta_init is None
        - In each iteration: gradient = df(theta, X, y)
        - Update: theta = theta - lr * gradient
        - Append current theta to trajectory at each step
        - Optionally print loss every 10 iterations
    """
    # TODO: Initialize theta
    # TODO: Initialize empty trajectory list
    # TODO: Loop for n_iters:
    #   - Compute full-dataset gradient using df(theta, X, y)
    #   - Update theta
    #   - Append theta copy to trajectory
    # TODO: Return (theta, trajectory)
    pass


def sgd(X, y, lr=0.01, n_iters=100, theta_init=None):
    """
    Stochastic Gradient Descent: use a single random example per update.

    Args:
        X (np.ndarray): Feature matrix, shape (n_samples, n_features)
        y (np.ndarray): Target vector, shape (n_samples,)
        lr (float): Learning rate
        n_iters (int): Number of gradient steps (each uses 1 sample)
        theta_init (np.ndarray or None): Initial parameters

    Returns:
        tuple: (theta_final, trajectory)

    Hints:
        - In each iteration: randomly pick one index i from [0, N)
        - Extract X_i = X[i:i+1] (keep 2D shape) and y_i = y[i:i+1]
        - Compute gradient = df(theta, X_i, y_i)
        - Update theta
        - Observe: trajectory will be noisy (zigzag) compared to batch GD
    """
    # TODO: Initialize theta
    # TODO: Initialize trajectory
    # TODO: Loop for n_iters:
    #   - Pick random index i
    #   - Compute single-sample gradient
    #   - Update theta
    #   - Append to trajectory
    # TODO: Return (theta, trajectory)
    pass


def mini_batch_gd(X, y, lr=0.01, n_iters=100, batch_size=16, theta_init=None):
    """
    Mini-Batch Gradient Descent: use a random subset of B examples per update.

    Args:
        X (np.ndarray): Feature matrix, shape (n_samples, n_features)
        y (np.ndarray): Target vector, shape (n_samples,)
        lr (float): Learning rate
        n_iters (int): Number of gradient steps
        batch_size (int): Number of examples per mini-batch
        theta_init (np.ndarray or None): Initial parameters

    Returns:
        tuple: (theta_final, trajectory)

    Hints:
        - In each iteration: randomly sample batch_size indices without replacement
        - Indices: np.random.choice(len(y), size=batch_size, replace=False)
        - Extract X_batch = X[indices] and y_batch = y[indices]
        - Compute gradient on batch, update theta
        - Observe: smoother than SGD, more updates than BGD
    """
    # TODO: Initialize theta
    # TODO: Initialize trajectory
    # TODO: Loop for n_iters:
    #   - Sample batch_size random indices
    #   - Extract mini-batch
    #   - Compute mini-batch gradient
    #   - Update theta
    #   - Append to trajectory
    # TODO: Return (theta, trajectory)
    pass


# ---------------------------------------------------------------------------
# Visualization (Complete — Do Not Modify)
# ---------------------------------------------------------------------------

def visualize_trajectories(trajectories_dict, X, y, title="Gradient Descent Trajectories"):
    """
    Visualize optimization trajectories on the loss surface contour plot.

    Args:
        trajectories_dict (dict): {label: trajectory_list} where each
                                   trajectory_list is a list of theta arrays
        X (np.ndarray): Feature matrix (used to compute loss surface)
        y (np.ndarray): Target vector
        title (str): Plot title
    """
    # Create grid for contour plot
    all_thetas = np.concatenate([np.array(t) for t in trajectories_dict.values()])
    theta_min = all_thetas.min(axis=0) - 0.5
    theta_max = all_thetas.max(axis=0) + 0.5

    theta0_range = np.linspace(theta_min[0], theta_max[0], 100)
    theta1_range = np.linspace(theta_min[1], theta_max[1], 100)
    T0, T1 = np.meshgrid(theta0_range, theta1_range)

    Z = np.array([[f(np.array([t0, t1]), X, y)
                   for t0 in theta0_range]
                  for t1 in theta1_range])

    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(T0, T1, Z, levels=50, cmap='viridis', alpha=0.7)
    plt.colorbar(contour, ax=ax, label="Loss")

    colors = ['red', 'blue', 'green', 'orange']
    for (label, trajectory), color in zip(trajectories_dict.items(), colors):
        traj_array = np.array(trajectory)
        ax.plot(traj_array[:, 0], traj_array[:, 1],
                color=color, marker='o', markersize=3, linewidth=1.5, label=label)
        ax.plot(traj_array[0, 0], traj_array[0, 1],
                'o', color=color, markersize=8, markeredgecolor='black')

    ax.set_xlabel("θ₀ (Intercept)")
    ax.set_ylabel("θ₁ (Slope)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    Compare convergence paths of BGD, SGD, and Mini-Batch GD on a synthetic
    linear regression problem with a 2D quadratic loss surface.

    Steps:
    1. Generate synthetic data: y = 2.5 + 1.5*x + noise
    2. Run all three GD variants from the same starting point
    3. Visualize their trajectories on the loss surface contour
    4. Compare final loss values and convergence speed
    """
    print("=" * 60)
    print("Gradient Descent Variants Comparison")
    print("=" * 60)

    # Generate synthetic data
    np.random.seed(42)
    N = 100
    x = np.random.randn(N)
    y = 2.5 + 1.5 * x + 0.5 * np.random.randn(N)
    X = np.column_stack([np.ones(N), x])  # Add bias column

    # Starting point (far from optimum)
    theta_init = np.array([-2.0, -2.0])

    # TODO: Run Batch GD
    # Hint: theta_bgd, traj_bgd = batch_gd(X, y, lr=0.1, n_iters=50, theta_init=theta_init.copy())

    # TODO: Run SGD
    # Hint: theta_sgd, traj_sgd = sgd(X, y, lr=0.1, n_iters=50, theta_init=theta_init.copy())

    # TODO: Run Mini-Batch GD (batch_size=16)
    # Hint: theta_mb, traj_mb = mini_batch_gd(X, y, lr=0.1, n_iters=50, batch_size=16, theta_init=theta_init.copy())

    # TODO: Print final theta and loss for each variant
    # Hint: Compare to true theta [2.5, 1.5]
    print(f"True parameters: θ₀=2.5, θ₁=1.5")
    # print(f"BGD final theta: {theta_bgd}, loss: {f(theta_bgd, X, y):.6f}")
    # print(f"SGD final theta: {theta_sgd}, loss: {f(theta_sgd, X, y):.6f}")
    # print(f"Mini-Batch final theta: {theta_mb}, loss: {f(theta_mb, X, y):.6f}")

    # TODO: Visualize trajectories
    # trajectories = {
    #     "Batch GD": traj_bgd,
    #     "SGD": traj_sgd,
    #     "Mini-Batch GD (B=16)": traj_mb,
    # }
    # visualize_trajectories(trajectories, X, y)


if __name__ == "__main__":
    main()
