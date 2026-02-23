"""
Exercise 02: Multiclass Logistic Regression with Softmax
=========================================================

Goal: Extend binary logistic regression to multiple classes using the
      Softmax function and categorical cross-entropy loss.

Learning Objectives:
    - Implement the Softmax function and understand its role.
    - Implement one-hot encoding.
    - Implement categorical cross-entropy loss.
    - Train a multiclass logistic regression model from scratch.
    - Evaluate on the Iris dataset and visualize results via a confusion matrix.

Instructions:
    - Fill in every section marked with TODO.
    - Do NOT use sklearn's LogisticRegression inside the class.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


# ---------------------------------------------------------------------------
# Softmax Activation
# ---------------------------------------------------------------------------

def softmax(z: np.ndarray) -> np.ndarray:
    """
    Compute the Softmax function row-wise for a 2-D input.

    Softmax converts raw class scores into a probability distribution:
        P(y=c | x) = exp(z_c) / Σⱼ exp(z_j)

    All output values are in (0, 1) and each row sums to 1.

    Parameters
    ----------
    z : np.ndarray, shape (n_samples, n_classes) — raw class scores (logits)

    Returns
    -------
    np.ndarray, shape (n_samples, n_classes) — probability matrix

    TODO:
        - Subtract the row-wise maximum from z before exponentiating.
          This prevents numerical overflow (exp of large numbers → ∞).
          Hint: z - np.max(z, axis=1, keepdims=True)
        - Compute exp of the shifted z.
        - Divide each row by the sum of that row.
          Hint: sum with axis=1, keepdims=True.
    """
    # TODO: implement numerically stable Softmax
    raise NotImplementedError("Implement softmax()")


# ---------------------------------------------------------------------------
# Loss Function
# ---------------------------------------------------------------------------

def cross_entropy_loss(
    y_true_onehot: np.ndarray,
    y_pred_proba: np.ndarray,
) -> float:
    """
    Compute the categorical cross-entropy loss.

    L = -(1/n) Σᵢ Σ_c y_{i,c} · log(ŷ_{i,c})

    Parameters
    ----------
    y_true_onehot : np.ndarray, shape (n_samples, n_classes) — one-hot labels
    y_pred_proba  : np.ndarray, shape (n_samples, n_classes) — Softmax output

    Returns
    -------
    float — scalar loss

    TODO:
        - Add epsilon (1e-15) to y_pred_proba to avoid log(0).
        - Compute element-wise: y_true_onehot * log(y_pred_proba).
        - Sum across classes (axis=1), then take the negative mean across samples.
    """
    # TODO: implement categorical cross-entropy
    raise NotImplementedError("Implement cross_entropy_loss()")


# ---------------------------------------------------------------------------
# One-Hot Encoding
# ---------------------------------------------------------------------------

def one_hot_encode(y: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert integer class labels to a one-hot encoded matrix.

    Example: y = [0, 2, 1], num_classes=3
             → [[1, 0, 0],
                [0, 0, 1],
                [0, 1, 0]]

    Parameters
    ----------
    y           : np.ndarray, shape (n_samples,) — integer labels
    num_classes : int

    Returns
    -------
    np.ndarray, shape (n_samples, num_classes) — one-hot matrix

    TODO:
        - Create a zero matrix of shape (len(y), num_classes).
        - For each row i, set column y[i] to 1.
        Hint: np.eye(num_classes)[y] achieves this in one line.
    """
    # TODO: implement one-hot encoding
    raise NotImplementedError("Implement one_hot_encode()")


# ---------------------------------------------------------------------------
# Multiclass Logistic Regression (Softmax Regression)
# ---------------------------------------------------------------------------

class MulticlassLogisticRegression:
    """
    Multinomial (Softmax) logistic regression trained with gradient descent.

    Parameters
    ----------
    lr      : float — learning rate
    n_iters : int   — number of gradient descent iterations
    """

    def __init__(self, lr: float = 0.1, n_iters: int = 500) -> None:
        self.lr = lr
        self.n_iters = n_iters
        self.W: np.ndarray | None = None   # shape (n_features, n_classes)
        self.b: np.ndarray | None = None   # shape (n_classes,)
        self.loss_history: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MulticlassLogisticRegression":
        """
        Train using gradient descent with Softmax output.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,) — integer class labels 0..C-1

        Returns
        -------
        self

        TODO:
            1. Determine n_features and n_classes.
            2. Initialize W as zeros of shape (n_features, n_classes).
               Initialize b as zeros of shape (n_classes,).
            3. One-hot encode y → Y_onehot.
            4. For each iteration:
               a. Compute logits: Z = X @ W + b  (shape: n_samples × n_classes)
               b. Compute probabilities: P = softmax(Z)
               c. Compute gradients:
                    dW = (1/n) * Xᵀ @ (P - Y_onehot)
                    db = (1/n) * sum(P - Y_onehot, axis=0)
               d. Update: W -= lr * dW, b -= lr * db
               e. Compute loss using cross_entropy_loss() and store it.
            5. Return self.
        """
        # TODO: implement multiclass gradient descent
        raise NotImplementedError("Implement MulticlassLogisticRegression.fit()")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute class probability matrix.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)

        TODO:
            - Compute Z = X @ W + b.
            - Return softmax(Z).
        """
        # TODO: implement predict_proba using softmax
        raise NotImplementedError("Implement predict_proba()")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict integer class labels by taking argmax of probabilities.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,) — predicted class indices

        TODO:
            - Call predict_proba(X).
            - Return np.argmax along axis=1.
        """
        # TODO: implement predict using argmax
        raise NotImplementedError("Implement predict()")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Train Softmax Logistic Regression on the Iris dataset and evaluate it.

    Steps:
        1. Load the Iris dataset (3 classes, 4 features).
        2. Train/test split (80/20, stratified).
        3. Standardize features.
        4. Train MulticlassLogisticRegression.
        5. Print accuracy on train and test sets.
        6. Plot the confusion matrix.
        7. Plot the training loss curve.

    TODO:
        - Instantiate, fit, predict, compute accuracy.
        - Use ConfusionMatrixDisplay.from_predictions() to plot the matrix.
        - Plot the loss curve from model.loss_history.
    """
    data = load_iris()
    X, y = data.data, data.target
    class_names = data.target_names

    print(f"Iris dataset: {X.shape[0]} samples, {X.shape[1]} features, "
          f"{len(class_names)} classes")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # TODO: train model, evaluate, plot confusion matrix and loss curve
    raise NotImplementedError("Complete main()")


if __name__ == "__main__":
    main()
