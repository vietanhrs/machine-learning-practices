"""
Exercise 01: The Perceptron
===========================
The perceptron is the simplest neural network — a single neuron with a step activation.
It is the historical starting point of deep learning (Rosenblatt, 1958).

Learning goals:
- Understand the perceptron learning rule.
- Observe convergence on linearly separable data.
- Understand *why* it fails on XOR (non-linearly separable data).

Perceptron Convergence Theorem:
    If the training data is linearly separable, the perceptron learning rule
    is guaranteed to converge to a solution in a finite number of steps.
    If the data is NOT linearly separable, the algorithm will never converge.
    This fundamental limitation motivated the development of multi-layer networks.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# ---------------------------------------------------------------------------
# Activation Function
# ---------------------------------------------------------------------------

def step_function(z):
    """
    Binary step activation function.

    Returns 1 if z >= 0, else 0.

    Args:
        z (float or np.ndarray): Pre-activation value(s).

    Returns:
        int or np.ndarray: Binary output.

    TODO:
        Implement the step function.
        Hint: np.where(condition, value_if_true, value_if_false) works
              on both scalars and arrays.
    """
    # TODO: implement step function
    raise NotImplementedError("Implement step_function(z)")


# ---------------------------------------------------------------------------
# Perceptron Class
# ---------------------------------------------------------------------------

class Perceptron:
    """
    A single-layer perceptron classifier.

    The perceptron learning rule:
        For each training sample (x, y):
            y_hat = predict(x)
            error  = y - y_hat        # either -1, 0, or +1
            w     += lr * error * x
            b     += lr * error

    Attributes:
        n_features (int): Number of input features.
        lr (float): Learning rate.
        weights (np.ndarray): Weight vector, shape (n_features,).
        bias (float): Scalar bias term.
        errors_per_epoch (list): Number of misclassifications per epoch.
    """

    def __init__(self, n_features, lr=0.01):
        """
        Initialize perceptron weights and bias to zero.

        Args:
            n_features (int): Dimensionality of input features.
            lr (float): Learning rate for weight updates.
        """
        self.n_features = n_features
        self.lr = lr
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.errors_per_epoch = []

    def predict(self, X):
        """
        Compute binary predictions for input X.

        Args:
            X (np.ndarray): Input data, shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels (0 or 1), shape (n_samples,).

        TODO:
            1. Compute the linear combination: z = X @ self.weights + self.bias
            2. Apply step_function to z and return the result.
        """
        # TODO: implement predict
        raise NotImplementedError("Implement Perceptron.predict(X)")

    def fit(self, X, y, n_epochs=100):
        """
        Train the perceptron using the perceptron learning rule.

        For each epoch:
            - Iterate over every training sample.
            - Compute prediction y_hat.
            - Update weights: w += lr * (y - y_hat) * x
            - Update bias:    b += lr * (y - y_hat)
            - Count misclassifications and store in self.errors_per_epoch.

        Args:
            X (np.ndarray): Training data, shape (n_samples, n_features).
            y (np.ndarray): Binary labels (0 or 1), shape (n_samples,).
            n_epochs (int): Number of full passes through the dataset.

        TODO:
            Implement the training loop.
            Hint:
                - Loop over epochs, then over samples.
                - An error only occurs when y != y_hat.
                - Use self.errors_per_epoch.append(n_errors) each epoch.
                - Early stopping: break if n_errors == 0 (converged).
        """
        # TODO: implement training loop
        raise NotImplementedError("Implement Perceptron.fit(X, y, n_epochs)")

    def plot_decision_boundary(self, X, y, title="Perceptron Decision Boundary"):
        """
        Plot the decision boundary learned by the perceptron.

        This function is complete — no changes needed.

        Args:
            X (np.ndarray): Input data, shape (n_samples, 2).
            y (np.ndarray): Binary labels, shape (n_samples,).
            title (str): Plot title.
        """
        assert X.shape[1] == 2, "plot_decision_boundary only works for 2D input."

        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        h = 0.02

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict(grid).reshape(xx.shape)

        cmap_bg = ListedColormap(["#FFAAAA", "#AAAAFF"])
        cmap_pt = ListedColormap(["#FF0000", "#0000FF"])

        plt.figure(figsize=(8, 5))
        plt.contourf(xx, yy, Z, cmap=cmap_bg, alpha=0.5)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_pt, edgecolors="k", s=60)
        plt.title(title)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    np.random.seed(42)

    # ------------------------------------------------------------------
    # Dataset 1: Linearly separable — AND gate with noise
    # The perceptron SHOULD converge on this data.
    # ------------------------------------------------------------------
    print("=" * 55)
    print("Dataset 1: Linearly Separable (AND-like)")
    print("=" * 55)

    # Two clusters well-separated in 2D
    X_pos = np.random.randn(50, 2) + np.array([2.0, 2.0])   # class 1
    X_neg = np.random.randn(50, 2) + np.array([-2.0, -2.0]) # class 0
    X_linear = np.vstack([X_pos, X_neg])
    y_linear = np.hstack([np.ones(50), np.zeros(50)]).astype(int)

    # Shuffle
    idx = np.random.permutation(len(y_linear))
    X_linear, y_linear = X_linear[idx], y_linear[idx]

    perceptron_linear = Perceptron(n_features=2, lr=0.1)
    perceptron_linear.fit(X_linear, y_linear, n_epochs=50)

    print(f"Errors per epoch: {perceptron_linear.errors_per_epoch}")
    final_acc = np.mean(perceptron_linear.predict(X_linear) == y_linear)
    print(f"Final training accuracy: {final_acc:.2%}")

    perceptron_linear.plot_decision_boundary(
        X_linear, y_linear, title="Perceptron — Linearly Separable Data"
    )

    # Plot convergence
    plt.figure(figsize=(7, 4))
    plt.plot(perceptron_linear.errors_per_epoch, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Misclassifications")
    plt.title("Perceptron Convergence (Linearly Separable Data)")
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # Dataset 2: XOR — NOT linearly separable
    # The perceptron CANNOT solve this — demonstrate failure.
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("Dataset 2: XOR (Not Linearly Separable)")
    print("XOR cannot be solved by a single perceptron.")
    print("=" * 55)

    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y_xor = np.array([0, 1, 1, 0])

    # Add noise to make it a multi-sample problem
    X_xor_noisy = np.vstack([X_xor + np.random.randn(*X_xor.shape) * 0.1
                              for _ in range(25)])
    y_xor_noisy = np.tile(y_xor, 25)

    perceptron_xor = Perceptron(n_features=2, lr=0.1)
    perceptron_xor.fit(X_xor_noisy, y_xor_noisy, n_epochs=100)

    final_acc_xor = np.mean(perceptron_xor.predict(X_xor_noisy) == y_xor_noisy)
    print(f"Final training accuracy on XOR: {final_acc_xor:.2%}")
    print("(Should be ~50% — no better than random; the perceptron fails.)")

    perceptron_xor.plot_decision_boundary(
        X_xor_noisy, y_xor_noisy,
        title="Perceptron — XOR Data (Fails to Separate)"
    )

    plt.figure(figsize=(7, 4))
    plt.plot(perceptron_xor.errors_per_epoch, color="red", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Misclassifications")
    plt.title("Perceptron — XOR (Errors Never Reach Zero)")
    plt.tight_layout()
    plt.show()

    print("\nConclusion: A single perceptron can only learn linearly separable")
    print("functions. XOR requires at least one hidden layer (MLP).")
    print("See exercise 02_mlp_from_scratch.py for the solution.")


if __name__ == "__main__":
    main()
