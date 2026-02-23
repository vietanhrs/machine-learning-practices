"""
Exercise 02: Optimizers From Scratch
======================================
Implement SGD, Momentum, AdaGrad, RMSProp, and Adam optimizers from scratch.
Train a logistic regression model with each optimizer and compare convergence.

Learning objectives:
- Understand the mathematical mechanics of each optimizer
- Implement bias correction in Adam
- Compare convergence speed and stability across optimizers

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Utility: Logistic Regression Forward Pass & Gradient
# ---------------------------------------------------------------------------

def sigmoid(z):
    """Numerically stable sigmoid function."""
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))


def compute_loss_and_grad(params, X, y):
    """
    Compute binary cross-entropy loss and gradient for logistic regression.

    Args:
        params (np.ndarray): Weight vector, shape (n_features,) — includes bias as params[0]
        X (np.ndarray): Feature matrix with bias column, shape (n_samples, n_features)
        y (np.ndarray): Binary labels {0, 1}, shape (n_samples,)

    Returns:
        tuple: (loss, gradient)
            - loss (float): Binary cross-entropy
            - gradient (np.ndarray): Same shape as params
    """
    logits = X @ params
    probs = sigmoid(logits)
    eps = 1e-15
    loss = -np.mean(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps))
    gradient = X.T @ (probs - y) / len(y)
    return loss, gradient


# ---------------------------------------------------------------------------
# Optimizer Classes
# ---------------------------------------------------------------------------

class SGDOptimizer:
    """
    Vanilla Stochastic Gradient Descent.

    Update rule:
        θ ← θ − lr × g
    """

    def __init__(self):
        pass

    def step(self, params, grads, lr=0.01):
        """
        Perform one SGD parameter update.

        Args:
            params (np.ndarray): Current parameters
            grads (np.ndarray): Gradient at current parameters
            lr (float): Learning rate

        Returns:
            np.ndarray: Updated parameters

        Hints:
            - Simple subtraction: params - lr * grads
            - This is the baseline — no gradient history, no adaptation
        """
        # TODO: Compute and return updated parameters
        pass


class MomentumOptimizer:
    """
    SGD with Momentum.

    Maintains an exponential moving average (velocity) of past gradients:
        v ← beta × v + (1 − beta) × g
        θ ← θ − lr × v

    Args:
        beta (float): Momentum coefficient (default 0.9)
    """

    def __init__(self, beta=0.9):
        self.beta = beta
        self.v = None  # Velocity vector (initialized on first step)

    def step(self, params, grads, lr=0.01):
        """
        Perform one momentum parameter update.

        Args:
            params (np.ndarray): Current parameters
            grads (np.ndarray): Gradient at current parameters
            lr (float): Learning rate

        Returns:
            np.ndarray: Updated parameters

        Hints:
            - Initialize self.v = np.zeros_like(params) if self.v is None
            - Update velocity: self.v = self.beta * self.v + (1 - self.beta) * grads
            - Update params: params - lr * self.v
        """
        # TODO: Initialize velocity on first call
        # TODO: Update velocity with exponential moving average
        # TODO: Return updated parameters
        pass


class AdaGradOptimizer:
    """
    AdaGrad: Adaptive Gradient Algorithm.

    Accumulates sum of squared gradients to adapt per-parameter learning rates:
        G ← G + g²          (element-wise)
        θ ← θ − (lr / √(G + ε)) × g

    Note: G grows monotonically — learning rate shrinks to zero over time.
    """

    def __init__(self, eps=1e-8):
        self.eps = eps
        self.G = None  # Accumulated squared gradients

    def step(self, params, grads, lr=0.01):
        """
        Perform one AdaGrad parameter update.

        Args:
            params (np.ndarray): Current parameters
            grads (np.ndarray): Current gradients
            lr (float): Global learning rate

        Returns:
            np.ndarray: Updated parameters

        Hints:
            - Initialize self.G = np.zeros_like(params) on first call
            - Accumulate: self.G += grads ** 2   (element-wise addition)
            - Compute adaptive lr: lr / np.sqrt(self.G + self.eps)
            - Return params - adaptive_lr * grads
        """
        # TODO: Initialize G on first call
        # TODO: Accumulate squared gradients
        # TODO: Compute adaptive learning rate per parameter
        # TODO: Return updated parameters
        pass


class RMSPropOptimizer:
    """
    RMSProp: Root Mean Square Propagation.

    Uses exponential moving average of squared gradients (fixes AdaGrad decay):
        E[g²] ← beta × E[g²] + (1 − beta) × g²
        θ ← θ − (lr / √(E[g²] + ε)) × g

    Args:
        beta (float): Decay rate for squared gradient EMA (default 0.9)
        eps (float): Numerical stability constant (default 1e-8)
    """

    def __init__(self, beta=0.9, eps=1e-8):
        self.beta = beta
        self.eps = eps
        self.E_g2 = None  # Exponential moving average of squared gradients

    def step(self, params, grads, lr=0.01):
        """
        Perform one RMSProp parameter update.

        Args:
            params (np.ndarray): Current parameters
            grads (np.ndarray): Current gradients
            lr (float): Learning rate

        Returns:
            np.ndarray: Updated parameters

        Hints:
            - Initialize self.E_g2 = np.zeros_like(params) on first call
            - Update EMA: self.E_g2 = self.beta * self.E_g2 + (1 - self.beta) * grads**2
            - Adaptive lr: lr / np.sqrt(self.E_g2 + self.eps)
            - Unlike AdaGrad, E_g2 doesn't grow to infinity (beta discounts old values)
        """
        # TODO: Initialize E_g2 on first call
        # TODO: Update exponential moving average of squared gradients
        # TODO: Compute adaptive learning rate
        # TODO: Return updated parameters
        pass


class AdamOptimizer:
    """
    Adam: Adaptive Moment Estimation.

    Combines momentum (first moment) and RMSProp (second moment) with bias correction:
        m ← beta1 × m + (1 − beta1) × g         (first moment — EMA of gradients)
        v ← beta2 × v + (1 − beta2) × g²        (second moment — EMA of squared grads)
        m̂ = m / (1 − beta1^t)                   (bias-corrected first moment)
        v̂ = v / (1 − beta2^t)                   (bias-corrected second moment)
        θ ← θ − lr × m̂ / (√v̂ + ε)

    Args:
        beta1 (float): First moment decay (default 0.9)
        beta2 (float): Second moment decay (default 0.999)
        eps (float): Numerical stability (default 1e-8)
    """

    def __init__(self, beta1=0.9, beta2=0.999, eps=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None    # First moment (mean of gradients)
        self.v = None    # Second moment (variance of gradients)
        self.t = 0       # Time step (for bias correction)

    def step(self, params, grads, lr=0.001):
        """
        Perform one Adam parameter update with bias correction.

        Args:
            params (np.ndarray): Current parameters
            grads (np.ndarray): Current gradients
            lr (float): Learning rate (default 0.001 for Adam)

        Returns:
            np.ndarray: Updated parameters

        Hints:
            - Initialize self.m and self.v as zeros on first call
            - Increment self.t += 1 at each step
            - Update first moment:  self.m = beta1 * self.m + (1 - beta1) * grads
            - Update second moment: self.v = beta2 * self.v + (1 - beta2) * grads**2
            - Bias correction (IMPORTANT):
                m_hat = self.m / (1 - self.beta1 ** self.t)
                v_hat = self.v / (1 - self.beta2 ** self.t)
            - Update: params - lr * m_hat / (np.sqrt(v_hat) + self.eps)
        """
        # TODO: Initialize m and v on first call
        # TODO: Increment time step
        # TODO: Update first and second moments
        # TODO: Apply bias correction
        # TODO: Compute and return updated parameters
        pass


# ---------------------------------------------------------------------------
# Training and Comparison
# ---------------------------------------------------------------------------

def compare_optimizers(X, y, n_iters=200, lr=0.01):
    """
    Train logistic regression with each optimizer and compare convergence.

    Args:
        X (np.ndarray): Feature matrix (with bias column), shape (n_samples, n_features)
        y (np.ndarray): Binary labels, shape (n_samples,)
        n_iters (int): Number of gradient steps per optimizer
        lr (float): Learning rate (same for all, except Adam which uses 0.001)

    Returns:
        dict: {optimizer_name: loss_history_list}

    Hints:
        - Initialize all optimizers with the same random parameters (use same seed)
        - For each optimizer, run n_iters steps:
            loss, grad = compute_loss_and_grad(params, X, y)
            params = optimizer.step(params, grad, lr=lr)
            Append loss to history
        - Use lr=0.001 for Adam (its standard default)
        - Collect loss history for each optimizer
        - Plot loss curves after training
    """
    # TODO: Define optimizers to compare
    # optimizers = {
    #     "SGD": SGDOptimizer(),
    #     "Momentum": MomentumOptimizer(),
    #     "AdaGrad": AdaGradOptimizer(),
    #     "RMSProp": RMSPropOptimizer(),
    #     "Adam": AdamOptimizer(),
    # }

    # TODO: For each optimizer:
    #   - Initialize params with same random seed
    #   - Run training loop
    #   - Record loss at each step

    # TODO: Plot all loss curves on one figure
    # Hint:
    #   plt.figure(figsize=(10, 6))
    #   for name, history in loss_histories.items():
    #       plt.plot(history, label=name)
    #   plt.xlabel("Iteration")
    #   plt.ylabel("Loss")
    #   plt.title("Optimizer Comparison — Logistic Regression")
    #   plt.legend()
    #   plt.grid(True)
    #   plt.show()

    # TODO: Return loss_histories dict
    pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    Generate a binary classification dataset and compare all 5 optimizers
    on logistic regression training.

    Steps:
    1. Generate synthetic binary classification data
    2. Standardize features
    3. Add bias column
    4. Run compare_optimizers()
    5. Print final loss for each optimizer
    """
    print("=" * 60)
    print("Optimizer Comparison from Scratch")
    print("=" * 60)

    # Generate data
    np.random.seed(42)
    X_raw, y = make_classification(
        n_samples=500, n_features=10, n_informative=6,
        n_redundant=2, random_state=42
    )

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Add bias column
    X = np.column_stack([np.ones(len(y)), X_scaled])

    # TODO: Call compare_optimizers(X, y, n_iters=300, lr=0.1)
    # Hint: Adam converges fastest initially; SGD is slowest but stable

    print("\nExpected: Adam should show fastest initial convergence.")
    print("SGD is slowest; Momentum and RMSProp are intermediate.")


if __name__ == "__main__":
    main()
