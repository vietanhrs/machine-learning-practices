"""
Exercise 03: Activation Functions Deep Dive
===========================================
Explore six activation functions: Sigmoid, Tanh, ReLU, Leaky ReLU, ELU, Softmax.
Visualize their shapes, derivatives, and understand their practical implications.

Learning goals:
- Understand the saturation problem (sigmoid/tanh).
- See how vanishing gradients accumulate across layers.
- Compare convergence speed of networks with different activations.

References:
    - Clevert et al. "ELU" (2016): https://arxiv.org/abs/1511.07289
    - Maas et al. "Leaky ReLU": https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Activation Functions
# ---------------------------------------------------------------------------

def sigmoid(z):
    """
    Sigmoid: σ(z) = 1 / (1 + e^(-z))   Range: (0, 1)

    Args:
        z (np.ndarray): Input array.

    Returns:
        np.ndarray: Sigmoid output.

    TODO:
        Implement sigmoid with numerical stability (clip z to [-500, 500]).
    """
    # TODO: implement sigmoid
    raise NotImplementedError("Implement sigmoid(z)")


def tanh(z):
    """
    Hyperbolic tangent: tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
    Range: (-1, 1).  Zero-centered (advantage over sigmoid).

    Args:
        z (np.ndarray): Input array.

    Returns:
        np.ndarray: Tanh output.

    TODO:
        Implement tanh.
        Hint: np.tanh(z) is available, but also implementable from scratch.
    """
    # TODO: implement tanh
    raise NotImplementedError("Implement tanh(z)")


def relu(z):
    """
    Rectified Linear Unit: ReLU(z) = max(0, z)   Range: [0, ∞)

    Args:
        z (np.ndarray): Input array.

    Returns:
        np.ndarray: ReLU output.

    TODO:
        Implement ReLU.
    """
    # TODO: implement relu
    raise NotImplementedError("Implement relu(z)")


def leaky_relu(z, alpha=0.01):
    """
    Leaky ReLU: f(z) = z if z > 0, else alpha * z   Range: (-∞, ∞)

    Fixes the dying ReLU problem by allowing a small gradient for z < 0.

    Args:
        z (np.ndarray): Input array.
        alpha (float): Negative slope coefficient (default 0.01).

    Returns:
        np.ndarray: Leaky ReLU output.

    TODO:
        Implement Leaky ReLU.
        Hint: np.where(z > 0, z, alpha * z)
    """
    # TODO: implement leaky_relu
    raise NotImplementedError("Implement leaky_relu(z, alpha)")


def elu(z, alpha=1.0):
    """
    Exponential Linear Unit: f(z) = z if z > 0, else alpha * (e^z - 1)
    Range: (-alpha, ∞). Smooth, zero-centered in expectation.

    Args:
        z (np.ndarray): Input array.
        alpha (float): Scale for negative saturation (default 1.0).

    Returns:
        np.ndarray: ELU output.

    TODO:
        Implement ELU.
        Hint: np.where(z > 0, z, alpha * (np.exp(z) - 1))
    """
    # TODO: implement elu
    raise NotImplementedError("Implement elu(z, alpha)")


def softmax(z):
    """
    Softmax: softmax(z_i) = e^(z_i) / sum_j(e^(z_j))
    Converts a vector of logits into a probability distribution.

    Args:
        z (np.ndarray): Logit vector, shape (n_classes,) or (batch, n_classes).

    Returns:
        np.ndarray: Probability distribution, same shape as z.

    TODO:
        Implement numerically stable softmax.
        Hint: Subtract max before exp (works along axis=-1).
    """
    # TODO: implement softmax
    raise NotImplementedError("Implement softmax(z)")


# ---------------------------------------------------------------------------
# Plot Activations and Derivatives
# ---------------------------------------------------------------------------

def plot_activations_and_derivatives():
    """
    Plot each activation function alongside its derivative.

    This scaffold sets up the plotting grid.
    Students fill in the derivative computations.

    Layout: 2 rows × 3 columns
        Row 1: activation functions
        Row 2: their derivatives
    """
    z = np.linspace(-4, 4, 400)

    activations = {
        "Sigmoid": sigmoid(z),
        "Tanh": tanh(z),
        "ReLU": relu(z),
        "Leaky ReLU": leaky_relu(z, alpha=0.1),
        "ELU": elu(z, alpha=1.0),
    }

    # TODO: Compute derivatives for each activation function.
    # Hint: Use the analytical formulas:
    #   sigmoid':   s * (1 - s)  where s = sigmoid(z)
    #   tanh':      1 - tanh(z)^2
    #   relu':      (z > 0).astype(float)
    #   leaky_relu': np.where(z > 0, 1, 0.1)
    #   elu':        np.where(z > 0, 1, elu(z, 1.0) + 1.0)

    derivatives = {
        "Sigmoid": None,    # TODO: sigmoid derivative
        "Tanh": None,       # TODO: tanh derivative
        "ReLU": None,       # TODO: relu derivative
        "Leaky ReLU": None, # TODO: leaky relu derivative (alpha=0.1)
        "ELU": None,        # TODO: elu derivative
    }

    fig, axes = plt.subplots(2, 5, figsize=(18, 6))
    colors = ["steelblue", "darkorange", "green", "red", "purple"]

    for i, (name, color) in enumerate(zip(activations.keys(), colors)):
        # Activation
        axes[0, i].plot(z, activations[name], color=color, linewidth=2)
        axes[0, i].axhline(0, color='k', linewidth=0.5)
        axes[0, i].axvline(0, color='k', linewidth=0.5)
        axes[0, i].set_title(name, fontsize=11)
        axes[0, i].set_ylim(-1.5, 2.0)
        axes[0, i].grid(True, alpha=0.3)

        # Derivative
        if derivatives[name] is not None:
            axes[1, i].plot(z, derivatives[name], color=color, linewidth=2,
                            linestyle='--')
        else:
            axes[1, i].text(0.5, 0.5, "TODO:\nimplement\nderivative",
                            ha='center', va='center', transform=axes[1, i].transAxes,
                            fontsize=9, color='gray')
        axes[1, i].axhline(0, color='k', linewidth=0.5)
        axes[1, i].axvline(0, color='k', linewidth=0.5)
        axes[1, i].set_title(f"{name}'", fontsize=11)
        axes[1, i].grid(True, alpha=0.3)

    axes[0, 0].set_ylabel("f(z)", fontsize=12)
    axes[1, 0].set_ylabel("f'(z)", fontsize=12)
    fig.suptitle("Activation Functions and Their Derivatives", fontsize=14)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Vanishing Gradient Demonstration
# ---------------------------------------------------------------------------

def demonstrate_vanishing_gradient(n_layers=10):
    """
    Show how sigmoid gradients shrink exponentially through deep networks.

    Simulate passing a gradient signal backward through n_layers of sigmoid neurons,
    each with a random activation value.

    Args:
        n_layers (int): Number of layers to simulate.

    TODO:
        1. Generate a random activation value a for each layer
           from sigmoid's output range: a = np.random.uniform(0.1, 0.9).
        2. Compute the local gradient: grad = a * (1 - a)  (sigmoid derivative).
        3. Multiply all local gradients together to get the accumulated gradient.
        4. Record the cumulative gradient after each layer.
        5. Plot cumulative gradient vs layer depth.

    Hint:
        gradient = 1.0
        gradient_history = []
        for l in range(n_layers):
            a = np.random.uniform(0.1, 0.9)
            local_grad = ...   # sigmoid derivative at a
            gradient *= local_grad
            gradient_history.append(gradient)
    """
    # TODO: implement vanishing gradient demonstration
    raise NotImplementedError("Implement demonstrate_vanishing_gradient(n_layers)")


# ---------------------------------------------------------------------------
# Compare Training Speed
# ---------------------------------------------------------------------------

def compare_training_speed(X, y, n_epochs=100, lr=0.01):
    """
    Train a small 2-hidden-layer network with different activation functions
    and compare their loss curves.

    Network architecture: [n_features → 32 → 32 → 1] with sigmoid output.

    Args:
        X (np.ndarray): Input features, shape (n_samples, n_features).
        y (np.ndarray): Binary labels (0 or 1), shape (n_samples,).
        n_epochs (int): Number of training epochs.
        lr (float): Learning rate.

    TODO:
        For each activation in ['sigmoid', 'tanh', 'relu', 'leaky_relu']:
            1. Initialize weights (He init for relu/leaky_relu, Xavier otherwise).
            2. Run a simple forward + backward + update loop.
            3. Record the BCE loss per epoch.
        Plot all four loss curves on the same axes.

    Hint:
        - You can reuse the MLP class from 02_mlp_from_scratch.py, or
          implement a minimal training loop here directly.
        - This is intentionally open-ended: focus on comparing the curves.
        - Expected result: ReLU/Leaky ReLU converge faster than Sigmoid.
    """
    # TODO: implement training comparison
    raise NotImplementedError("Implement compare_training_speed(X, y)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    np.random.seed(42)

    print("1. Plotting activation functions and derivatives...")
    plot_activations_and_derivatives()

    print("\n2. Demonstrating vanishing gradient through sigmoid layers...")
    demonstrate_vanishing_gradient(n_layers=15)

    print("\n3. Comparing training speed of different activations...")
    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=5,
        random_state=42
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    compare_training_speed(X, y, n_epochs=150, lr=0.05)

    print("\nObservation: ReLU-based activations typically converge faster")
    print("because they do not saturate and maintain larger gradient magnitudes.")

    # Softmax demo
    print("\n4. Softmax demonstration:")
    logits = np.array([2.0, 1.0, 0.1])
    probs = softmax(logits)
    print(f"  Logits:         {logits}")
    print(f"  Softmax output: {probs}")
    print(f"  Sum of probs:   {probs.sum():.6f}  (should be 1.0)")


if __name__ == "__main__":
    main()
