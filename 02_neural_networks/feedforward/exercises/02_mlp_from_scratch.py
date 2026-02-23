"""
Exercise 02: Multi-Layer Perceptron from Scratch
=================================================
Build a fully-connected neural network using only NumPy.
This exercise is the core of understanding deep learning:
you implement forward pass, backpropagation, and gradient descent by hand.

Learning goals:
- Understand matrix-form forward pass.
- Implement backpropagation via the chain rule.
- See how a non-linear decision boundary emerges from hidden layers.

References:
    - CS231n Notes: https://cs231n.github.io/neural-networks-case-study/
    - Nielsen "Neural Networks and Deep Learning": http://neuralnetworksanddeeplearning.com/
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap


# ---------------------------------------------------------------------------
# Activation Functions
# ---------------------------------------------------------------------------

def relu(z):
    """
    Rectified Linear Unit activation.

    Args:
        z (np.ndarray): Pre-activation array, any shape.

    Returns:
        np.ndarray: max(0, z), element-wise.

    TODO:
        Implement ReLU.
        Hint: np.maximum(0, z) works cleanly on arrays.
    """
    # TODO: implement relu
    raise NotImplementedError("Implement relu(z)")


def relu_derivative(z):
    """
    Derivative of ReLU with respect to z.

    Returns 1 where z > 0, else 0.
    (Technically undefined at z=0; convention is 0.)

    Args:
        z (np.ndarray): Pre-activation array (same shape as used in forward pass).

    Returns:
        np.ndarray: Element-wise derivative.

    TODO:
        Implement relu_derivative.
        Hint: (z > 0).astype(float)
    """
    # TODO: implement relu_derivative
    raise NotImplementedError("Implement relu_derivative(z)")


def sigmoid(z):
    """
    Sigmoid activation function.

    σ(z) = 1 / (1 + e^(-z))

    Args:
        z (np.ndarray): Pre-activation array.

    Returns:
        np.ndarray: Values in (0, 1).

    TODO:
        Implement sigmoid.
        Hint: Use np.clip(z, -500, 500) before computing exp to avoid overflow.
    """
    # TODO: implement sigmoid
    raise NotImplementedError("Implement sigmoid(z)")


def softmax(z):
    """
    Softmax activation for multi-class output.

    softmax(z)_i = exp(z_i) / sum_j(exp(z_j))

    Applied row-wise when z has shape (batch_size, n_classes).

    Args:
        z (np.ndarray): Logits, shape (batch_size, n_classes).

    Returns:
        np.ndarray: Probability distribution over classes, same shape.

    TODO:
        Implement softmax in a numerically stable way.
        Hint:
            - Subtract max(z, axis=1, keepdims=True) before exp to avoid overflow.
            - Divide by sum(exp, axis=1, keepdims=True).
    """
    # TODO: implement softmax (numerically stable)
    raise NotImplementedError("Implement softmax(z)")


# ---------------------------------------------------------------------------
# MLP Class
# ---------------------------------------------------------------------------

class MLP:
    """
    Multi-Layer Perceptron (fully-connected neural network) — NumPy only.

    Architecture: variable depth, configurable activation for hidden layers.
    Output layer uses sigmoid for binary classification.

    Attributes:
        layer_sizes (list[int]): List of layer sizes including input and output.
            Example: [2, 64, 32, 1] → input:2, hidden:64, hidden:32, output:1
        activation (str): Activation for hidden layers ('relu' or 'sigmoid').
        weights (list[np.ndarray]): Weight matrices per layer.
        biases (list[np.ndarray]): Bias vectors per layer.
        cache (dict): Stores Z and A values for each layer during forward pass.
        loss_history (list[float]): Training loss per epoch.
    """

    def __init__(self, layer_sizes, activation='relu'):
        """
        Initialize MLP architecture.

        Args:
            layer_sizes (list[int]): Sizes of each layer [n_input, n_h1, ..., n_output].
            activation (str): Hidden layer activation ('relu' or 'sigmoid').
        """
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.weights = []
        self.biases = []
        self.cache = {}
        self.gradients = {}
        self.loss_history = []
        self.initialize_weights()

    def initialize_weights(self):
        """
        Initialize weight matrices and bias vectors for all layers.

        Use He initialization for ReLU:
            W ~ N(0, sqrt(2 / n_in))

        Use Xavier initialization for Sigmoid/Tanh:
            W ~ N(0, sqrt(1 / n_in))

        Biases initialized to zero.

        TODO:
            For each pair of adjacent layers (n_in, n_out):
                1. Create weight matrix W of shape (n_in, n_out).
                2. Scale appropriately based on self.activation.
                3. Create bias vector b of shape (1, n_out) filled with zeros.
                4. Append W to self.weights, b to self.biases.

        Hint:
            for i in range(len(self.layer_sizes) - 1):
                n_in  = self.layer_sizes[i]
                n_out = self.layer_sizes[i + 1]
                ...
        """
        # TODO: implement weight initialization
        raise NotImplementedError("Implement MLP.initialize_weights()")

    def _activation_fn(self, z):
        """Apply hidden layer activation (internal helper)."""
        if self.activation == 'relu':
            return relu(z)
        elif self.activation == 'sigmoid':
            return sigmoid(z)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def _activation_derivative(self, z):
        """Derivative of hidden layer activation (internal helper)."""
        if self.activation == 'relu':
            return relu_derivative(z)
        elif self.activation == 'sigmoid':
            s = sigmoid(z)
            return s * (1 - s)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def forward(self, X):
        """
        Perform the forward pass through all layers.

        For each hidden layer l (1-indexed):
            Z[l] = A[l-1] @ W[l] + b[l]
            A[l] = activation(Z[l])

        For the output layer:
            Z[L] = A[L-1] @ W[L] + b[L]
            A[L] = sigmoid(Z[L])       ← binary classification

        Store all Z and A values in self.cache for use in backprop.

        Args:
            X (np.ndarray): Input data, shape (batch_size, n_features).

        Returns:
            np.ndarray: Output predictions, shape (batch_size, n_output).

        TODO:
            1. Set self.cache['A0'] = X  (input is the first "activation").
            2. Loop through each layer (use self.weights, self.biases):
                a. Compute Z = prev_A @ W + b
                b. Store Z in cache as self.cache[f'Z{l}']
                c. Apply activation (hidden layers: _activation_fn, output: sigmoid)
                d. Store A in cache as self.cache[f'A{l}']
            3. Return the final layer's output.

        Note: Layer indices in cache are 1-based (Z1, A1, Z2, A2, ..., ZL, AL).
        """
        # TODO: implement forward pass
        raise NotImplementedError("Implement MLP.forward(X)")

    def backward(self, X, y):
        """
        Compute gradients via backpropagation.

        Uses binary cross-entropy loss:
            L = -[y log(a) + (1-y) log(1-a)]

        Gradient at output (sigmoid + BCE):
            dA_L = -(y / A_L) + (1 - y) / (1 - A_L)
            dZ_L = A_L - y   (simplification for sigmoid + BCE)

        Then propagate backward through each layer:
            dW[l] = A[l-1].T @ dZ[l]  / m
            db[l] = mean(dZ[l], axis=0)
            dA[l-1] = dZ[l] @ W[l].T
            dZ[l-1] = dA[l-1] * activation_derivative(Z[l-1])

        Stores gradients in self.gradients as 'dW{l}' and 'db{l}'.

        Args:
            X (np.ndarray): Input data (same as used in forward pass), shape (m, n_features).
            y (np.ndarray): True labels, shape (m, 1) or (m,).

        TODO:
            1. Get number of samples m = X.shape[0].
            2. Retrieve the number of layers L.
            3. Compute dZ for the output layer: dZ = A_L - y (reshape y if needed).
            4. Loop backward from L down to 1:
                a. Compute dW = (A[l-1].T @ dZ) / m
                b. Compute db = mean(dZ, axis=0, keepdims=True)
                c. Store in self.gradients
                d. If not the first layer: compute dA_prev = dZ @ W[l].T
                   then dZ = dA_prev * _activation_derivative(Z[l-1])

        Hint: Use self.cache[f'A{l}'] and self.cache[f'Z{l}'] from the forward pass.
        """
        # TODO: implement backpropagation
        raise NotImplementedError("Implement MLP.backward(X, y)")

    def update_weights(self, lr):
        """
        Apply gradient descent weight updates.

        W[l] -= lr * dW[l]
        b[l] -= lr * db[l]

        Args:
            lr (float): Learning rate.

        TODO:
            Loop over all layers and update self.weights[i] and self.biases[i]
            using the gradients stored in self.gradients.

        Hint: Layer l is 1-indexed in gradients ('dW1', 'db1', ...), but
              self.weights is 0-indexed. So layer l maps to self.weights[l-1].
        """
        # TODO: implement weight updates
        raise NotImplementedError("Implement MLP.update_weights(lr)")

    def fit(self, X, y, lr=0.01, n_epochs=100, batch_size=32):
        """
        Train the MLP using mini-batch gradient descent.

        For each epoch:
            1. Shuffle the training data.
            2. Split into mini-batches of size batch_size.
            3. For each mini-batch:
                a. Forward pass.
                b. Backward pass.
                c. Update weights.
            4. Compute and store the full-dataset loss in self.loss_history.
            5. Print loss every 10 epochs.

        Binary cross-entropy loss:
            L = -mean(y * log(a + eps) + (1 - y) * log(1 - a + eps))

        Args:
            X (np.ndarray): Training data, shape (n_samples, n_features).
            y (np.ndarray): Binary labels (0 or 1), shape (n_samples,).
            lr (float): Learning rate.
            n_epochs (int): Number of training epochs.
            batch_size (int): Mini-batch size.

        TODO:
            Implement the training loop described above.
            Hint:
                - Use np.random.permutation(n_samples) to shuffle indices.
                - Clip predictions before computing log: np.clip(a, 1e-8, 1-1e-8).
                - Reshape y to (n_samples, 1) at the start if needed.
        """
        # TODO: implement training loop
        raise NotImplementedError("Implement MLP.fit(X, y, lr, n_epochs, batch_size)")

    def predict(self, X):
        """
        Generate binary class predictions.

        Args:
            X (np.ndarray): Input data, shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels (0 or 1), shape (n_samples,).

        TODO:
            1. Run forward pass to get probabilities.
            2. Apply threshold: class 1 if probability >= 0.5, else class 0.
            3. Return as integer array.
        """
        # TODO: implement predict
        raise NotImplementedError("Implement MLP.predict(X)")

    def plot_loss(self):
        """
        Plot the training loss curve over epochs.
        This function is complete — no changes needed.
        """
        if not self.loss_history:
            print("No loss history found. Call fit() first.")
            return
        plt.figure(figsize=(8, 4))
        plt.plot(self.loss_history, label="Training Loss", color="steelblue")
        plt.xlabel("Epoch")
        plt.ylabel("Binary Cross-Entropy Loss")
        plt.title("MLP Training Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Helper: Plot Decision Boundary
# ---------------------------------------------------------------------------

def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    """
    Plot the 2D decision boundary for a trained MLP.
    Works for any model with a .predict(X) method.
    """
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)

    cmap_bg = ListedColormap(["#FFAAAA", "#AAAAFF"])
    cmap_pt = ListedColormap(["#FF0000", "#0000FF"])

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_bg, alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_pt, edgecolors="k", s=40)
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
    # Dataset: make_moons — two interleaving half-circles (non-linear)
    # A single perceptron cannot separate these; MLP can.
    # ------------------------------------------------------------------
    X, y = make_moons(n_samples=500, noise=0.2, random_state=42)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Dataset: make_moons")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Test samples:     {X_test.shape[0]}")

    # ------------------------------------------------------------------
    # Train MLP
    # Architecture: [2 → 64 → 32 → 1]
    # ------------------------------------------------------------------
    mlp = MLP(layer_sizes=[2, 64, 32, 1], activation='relu')

    print("\nTraining MLP [2 → 64 → 32 → 1] with ReLU hidden layers...")
    mlp.fit(X_train, y_train, lr=0.05, n_epochs=200, batch_size=32)

    # Evaluate
    train_acc = np.mean(mlp.predict(X_train) == y_train)
    test_acc  = np.mean(mlp.predict(X_test)  == y_test)
    print(f"\nTraining Accuracy: {train_acc:.2%}")
    print(f"Test Accuracy:     {test_acc:.2%}")

    # ------------------------------------------------------------------
    # Visualize
    # ------------------------------------------------------------------
    mlp.plot_loss()

    plot_decision_boundary(
        mlp, X_train, y_train,
        title="MLP — Non-Linear Decision Boundary on make_moons (Train)"
    )
    plot_decision_boundary(
        mlp, X_test, y_test,
        title="MLP — Non-Linear Decision Boundary on make_moons (Test)"
    )


if __name__ == "__main__":
    main()
