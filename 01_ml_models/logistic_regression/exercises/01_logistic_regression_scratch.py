"""
Exercise 01: Logistic Regression from Scratch
==============================================

Goal: Implement binary logistic regression using gradient descent, without
      using sklearn's LogisticRegression.

Learning Objectives:
    - Implement the sigmoid function and understand its role.
    - Implement binary cross-entropy loss.
    - Implement gradient descent for logistic regression.
    - Optionally extend to L2 (Ridge) regularization.
    - Evaluate on a real dataset (breast cancer).

Instructions:
    - Fill in every section marked with TODO.
    - Do NOT use sklearn's LogisticRegression inside the class.
    - You MAY use sklearn for dataset loading, train/test split, and metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


# ---------------------------------------------------------------------------
# Activation Function
# ---------------------------------------------------------------------------

def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid (logistic) function element-wise.

    σ(z) = 1 / (1 + e^{-z})

    Parameters
    ----------
    z : np.ndarray — linear combination wᵀx + b (any shape)

    Returns
    -------
    np.ndarray — same shape as z, values in (0, 1)

    TODO:
        - Compute 1 / (1 + np.exp(-z)).
        - Be careful with numerical overflow: for very negative z,
          np.exp(-z) → ∞. Use np.clip(z, -500, 500) to avoid this.
    """
    # TODO: implement sigmoid with numerical stability
    raise NotImplementedError("Implement sigmoid()")


# ---------------------------------------------------------------------------
# Loss Function
# ---------------------------------------------------------------------------

def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the binary cross-entropy (log-loss) averaged over all samples.

    L = -(1/n) Σ [y · log(ŷ) + (1-y) · log(1-ŷ)]

    Parameters
    ----------
    y_true : np.ndarray, shape (n,) — ground truth binary labels (0 or 1)
    y_pred : np.ndarray, shape (n,) — predicted probabilities in (0, 1)

    Returns
    -------
    float — scalar loss value

    TODO:
        - Add a small epsilon (1e-15) to y_pred and (1 - y_pred) inside the
          log to prevent log(0) → -∞.
        - Compute the element-wise formula.
        - Return the mean (average over n samples).
    """
    # TODO: implement binary cross-entropy loss
    raise NotImplementedError("Implement binary_cross_entropy()")


# ---------------------------------------------------------------------------
# Logistic Regression Classifier
# ---------------------------------------------------------------------------

class LogisticRegression:
    """
    Binary Logistic Regression trained with gradient descent.

    Parameters
    ----------
    lr             : float — learning rate (step size for gradient descent)
    n_iters        : int   — maximum number of gradient descent iterations
    regularization : str | None — 'l2', 'l1', or None
    lambda_        : float — regularization strength (used when regularization is set)
    """

    def __init__(
        self,
        lr: float = 0.01,
        n_iters: int = 1000,
        regularization: str | None = None,
        lambda_: float = 0.1,
    ) -> None:
        self.lr = lr
        self.n_iters = n_iters
        self.regularization = regularization
        self.lambda_ = lambda_

        # These will be set during fit()
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.loss_history: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """
        Train the model using gradient descent.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,) — binary labels 0 or 1

        Returns
        -------
        self

        TODO:
            1. Initialize self.weights as zeros of shape (n_features,).
               Initialize self.bias = 0.0.
            2. For each iteration:
               a. Compute the linear prediction: z = X @ w + b
               b. Apply sigmoid to get probabilities: ŷ = sigmoid(z)
               c. Compute gradients:
                    dw = (1/n) * Xᵀ @ (ŷ - y)
                    db = (1/n) * sum(ŷ - y)
               d. If regularization == 'l2':
                    add (lambda_ / n) * w to dw (gradient of L2 penalty).
                    Note: do NOT regularize the bias.
               e. If regularization == 'l1':
                    add (lambda_ / n) * sign(w) to dw.
               f. Update: w -= lr * dw, b -= lr * db
               g. Compute and store the loss in self.loss_history.
            3. Return self.
        """
        # TODO: implement gradient descent training loop
        raise NotImplementedError("Implement LogisticRegression.fit()")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute predicted probabilities for class 1.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,) — P(y=1 | x) for each sample

        TODO:
            - Compute z = X @ self.weights + self.bias
            - Return sigmoid(z)
        """
        # TODO: compute and return predicted probabilities
        raise NotImplementedError("Implement predict_proba()")

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary class labels.

        Parameters
        ----------
        X         : np.ndarray, shape (n_samples, n_features)
        threshold : float — probability threshold for class 1 (default 0.5)

        Returns
        -------
        np.ndarray, shape (n_samples,) — predicted labels (0 or 1)

        TODO:
            - Call predict_proba(X).
            - Return (probabilities >= threshold).astype(int)
        """
        # TODO: threshold probabilities into class labels
        raise NotImplementedError("Implement predict()")

    def plot_loss_curve(self) -> None:
        """
        Plot the training loss over iterations. (Provided — do not modify.)
        """
        if not self.loss_history:
            print("No loss history found. Run fit() first.")
            return
        plt.figure(figsize=(8, 4))
        plt.plot(self.loss_history, color="royalblue", linewidth=2)
        plt.xlabel("Iteration")
        plt.ylabel("Binary Cross-Entropy Loss")
        plt.title("Logistic Regression — Training Loss Curve")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("lr_loss_curve.png", dpi=120)
        plt.show()
        print("Loss curve saved to lr_loss_curve.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Train and evaluate logistic regression on the breast cancer dataset.

    Steps:
        1. Load the breast cancer dataset.
        2. Train/test split (80/20, stratified).
        3. Standardize features.
        4. Train your LogisticRegression.
        5. Evaluate accuracy on the test set.
        6. Plot the loss curve.
        7. (Optional) Compare with sklearn's LogisticRegression.

    TODO:
        - Instantiate LogisticRegression with lr=0.1, n_iters=500.
        - Call fit(), predict(), and accuracy_score().
        - Print the classification report.
        - Call plot_loss_curve().
    """
    print("Loading breast cancer dataset ...")
    data = load_breast_cancer()
    X, y = data.data, data.target

    print(f"Dataset shape: {X.shape}, classes: {data.target_names}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # TODO: instantiate, fit, predict, evaluate
    raise NotImplementedError("Complete main()")


if __name__ == "__main__":
    main()
