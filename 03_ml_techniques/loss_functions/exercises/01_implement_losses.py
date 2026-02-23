"""
Exercise 01: Implement Loss Functions from Scratch
====================================================
Implement all major loss functions using only NumPy. Compare their behavior
on regression and classification tasks, especially under outlier conditions.

Learning objectives:
- Understand the mathematical formulas for each loss function
- Observe different outlier sensitivity profiles
- Implement numerical stability tricks (epsilon clipping)

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Regression Losses
# ---------------------------------------------------------------------------

def mse_loss(y_true, y_pred):
    """
    Mean Squared Error: L = (1/N) Σ (y_true - y_pred)²

    Args:
        y_true (np.ndarray): Ground truth values, shape (N,)
        y_pred (np.ndarray): Predicted values, shape (N,)

    Returns:
        float: Scalar MSE loss

    Hints:
        - Compute residuals = y_true - y_pred
        - Square element-wise: residuals ** 2
        - Return np.mean(squared_residuals)
        - Sensitive to outliers — squaring amplifies large errors
    """
    # TODO: Compute squared differences between y_true and y_pred
    # TODO: Return the mean of squared differences
    pass


def mae_loss(y_true, y_pred):
    """
    Mean Absolute Error: L = (1/N) Σ |y_true - y_pred|

    Args:
        y_true (np.ndarray): Ground truth values, shape (N,)
        y_pred (np.ndarray): Predicted values, shape (N,)

    Returns:
        float: Scalar MAE loss

    Hints:
        - Compute absolute residuals = np.abs(y_true - y_pred)
        - Return np.mean(absolute_residuals)
        - Robust to outliers — linear penalty regardless of error magnitude
    """
    # TODO: Compute absolute differences
    # TODO: Return the mean of absolute differences
    pass


def huber_loss(y_true, y_pred, delta=1.0):
    """
    Huber Loss: Quadratic for small errors, linear for large errors.

    Formula:
        L(e) = 0.5 * e²                    if |e| ≤ delta
        L(e) = delta * (|e| - 0.5*delta)   if |e| > delta

    Args:
        y_true (np.ndarray): Ground truth values, shape (N,)
        y_pred (np.ndarray): Predicted values, shape (N,)
        delta (float): Threshold separating quadratic and linear regimes (default 1.0)

    Returns:
        float: Scalar Huber loss (mean over all samples)

    Hints:
        - residuals = y_true - y_pred
        - abs_residuals = np.abs(residuals)
        - Use np.where(condition, quadratic_value, linear_value) for element-wise selection
        - quadratic = 0.5 * residuals ** 2
        - linear    = delta * (abs_residuals - 0.5 * delta)
        - Apply np.where(abs_residuals <= delta, quadratic, linear)
        - Return np.mean(element_wise_losses)
    """
    # TODO: Compute residuals
    # TODO: Compute element-wise Huber loss using np.where
    # TODO: Return mean Huber loss
    pass


# ---------------------------------------------------------------------------
# Classification Losses
# ---------------------------------------------------------------------------

def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """
    Binary Cross-Entropy: L = -(1/N) Σ [y*log(ŷ) + (1-y)*log(1-ŷ)]

    Args:
        y_true (np.ndarray): Binary labels {0, 1}, shape (N,)
        y_pred (np.ndarray): Predicted probabilities in [0, 1], shape (N,)
        eps (float): Small constant for numerical stability (avoids log(0))

    Returns:
        float: Scalar binary cross-entropy loss

    Hints:
        - IMPORTANT: Clip y_pred to [eps, 1-eps] before computing log:
          y_pred = np.clip(y_pred, eps, 1 - eps)
        - Compute per-sample loss:
          loss_i = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        - Return np.mean(loss_i)
        - Without clipping: log(0) = -inf → NaN in loss
    """
    # TODO: Clip predictions to avoid log(0)
    # TODO: Compute binary cross-entropy per sample
    # TODO: Return mean loss
    pass


def categorical_cross_entropy(y_true_onehot, y_pred_proba, eps=1e-15):
    """
    Categorical Cross-Entropy: L = -(1/N) Σᵢ Σₖ y_ik * log(ŷ_ik)

    Simplifies to: L = -(1/N) Σᵢ log(ŷᵢ[true_class])

    Args:
        y_true_onehot (np.ndarray): One-hot encoded labels, shape (N, K)
        y_pred_proba (np.ndarray): Softmax probabilities, shape (N, K)
        eps (float): Small constant for numerical stability

    Returns:
        float: Scalar categorical cross-entropy loss

    Hints:
        - Clip y_pred_proba: np.clip(y_pred_proba, eps, 1.0)
        - Compute element-wise: y_true_onehot * np.log(y_pred_proba_clipped)
        - Sum over classes (axis=1), then negate and take mean
        - Equivalently: -np.mean(np.sum(y_true_onehot * np.log(y_pred_clipped), axis=1))
    """
    # TODO: Clip predicted probabilities
    # TODO: Compute element-wise product of true labels and log-probabilities
    # TODO: Sum over classes, negate, and return mean
    pass


def hinge_loss(y_true, y_pred):
    """
    Hinge Loss (for SVM-style classification):
        L = (1/N) Σ max(0, 1 - y * ŷ)

    Args:
        y_true (np.ndarray): Labels in {-1, +1}, shape (N,)
        y_pred (np.ndarray): Raw model scores (not probabilities), shape (N,)

    Returns:
        float: Scalar hinge loss

    Hints:
        - margins = y_true * y_pred   (functional margin)
        - per_sample_loss = np.maximum(0, 1 - margins)
        - Return np.mean(per_sample_loss)
        - Note: y_true must be in {-1, +1} (not {0, 1})
        - Zero loss when margins >= 1 (correct AND confident)
    """
    # TODO: Compute functional margins (y_true * y_pred)
    # TODO: Compute per-sample hinge loss using np.maximum
    # TODO: Return mean hinge loss
    pass


def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25, eps=1e-15):
    """
    Focal Loss: Down-weights easy examples to focus training on hard ones.
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Where:
        p_t = y_pred       if y_true = 1
        p_t = 1 - y_pred   if y_true = 0
        alpha_t = alpha    if y_true = 1
        alpha_t = 1-alpha  if y_true = 0

    Args:
        y_true (np.ndarray): Binary labels {0, 1}, shape (N,)
        y_pred (np.ndarray): Predicted probabilities [0, 1], shape (N,)
        gamma (float): Focusing parameter (default 2.0). Higher = more focus on hard examples.
        alpha (float): Class weight for positive class (default 0.25)
        eps (float): Numerical stability

    Returns:
        float: Scalar focal loss

    Hints:
        - Clip y_pred: np.clip(y_pred, eps, 1 - eps)
        - Compute p_t: np.where(y_true == 1, y_pred, 1 - y_pred)
        - Compute alpha_t: np.where(y_true == 1, alpha, 1 - alpha)
        - Modulating factor: (1 - p_t) ** gamma
        - Per-sample focal loss: -alpha_t * modulating_factor * np.log(p_t)
        - Return np.mean(focal_loss_per_sample)
        - When gamma=0: reduces to weighted binary cross-entropy
    """
    # TODO: Clip predictions
    # TODO: Compute p_t (probability assigned to the true class)
    # TODO: Compute alpha_t (class-specific weight)
    # TODO: Compute modulating factor (1 - p_t)^gamma
    # TODO: Compute and return mean focal loss
    pass


# ---------------------------------------------------------------------------
# Comparison Utility
# ---------------------------------------------------------------------------

def compare_losses_on_outliers(y_true, y_pred_normal, y_pred_outlier):
    """
    Compare regression loss functions on predictions with and without outliers.

    Args:
        y_true (np.ndarray): Ground truth values
        y_pred_normal (np.ndarray): Predictions close to y_true
        y_pred_outlier (np.ndarray): Predictions with at least one large outlier error

    Hints:
        - Compute MSE, MAE, Huber for both y_pred_normal and y_pred_outlier
        - Print a comparison table showing how each loss changes with outliers
        - MSE should show the largest increase with outliers
        - MAE should show the smallest (linear) increase
        - Huber should be in between
        - Also visualize loss curves as a function of single error magnitude:
          errors = np.linspace(-5, 5, 200)
          mse_curve = errors ** 2 / 2
          mae_curve = np.abs(errors)
          huber_curve = [huber_loss(np.array([0.0]), np.array([e])) for e in errors]
    """
    # TODO: Compute and print losses for normal predictions
    # TODO: Compute and print losses for outlier predictions
    # TODO: Print comparison table
    # TODO: Plot loss function curves for MSE, MAE, Huber as a function of error magnitude
    pass


# ---------------------------------------------------------------------------
# Loss Curve Visualization
# ---------------------------------------------------------------------------

def plot_loss_curves():
    """
    Plot loss function curves as a function of error magnitude.
    Shows how each loss responds to errors of different sizes.

    This function is complete — study the shape of each curve.
    """
    errors = np.linspace(-4, 4, 400)
    mse_vals = 0.5 * errors ** 2
    mae_vals = np.abs(errors)

    # Huber: quadratic inside [-1, 1], linear outside
    huber_vals = np.where(
        np.abs(errors) <= 1.0,
        0.5 * errors ** 2,
        1.0 * (np.abs(errors) - 0.5)
    )

    plt.figure(figsize=(10, 5))
    plt.plot(errors, mse_vals, label="0.5 × MSE", linewidth=2, color='red')
    plt.plot(errors, mae_vals, label="MAE", linewidth=2, color='blue')
    plt.plot(errors, huber_vals, label="Huber (δ=1)", linewidth=2, color='green', linestyle='--')
    plt.axvline(x=-1, color='gray', linestyle=':', alpha=0.7, label="δ = ±1")
    plt.axvline(x=1, color='gray', linestyle=':', alpha=0.7)
    plt.xlabel("Prediction Error (y_true - y_pred)")
    plt.ylabel("Loss Value")
    plt.title("Regression Loss Functions: Response to Error Magnitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    Demonstrate all loss functions on sample data.

    Steps:
    1. Create sample regression data (y_true, y_pred_good, y_pred_outlier)
    2. Compute and display all regression losses
    3. Create sample classification data
    4. Compute and display all classification losses
    5. Plot loss curves
    6. Compare loss sensitivity to outliers
    """
    print("=" * 60)
    print("Loss Functions Implementation and Comparison")
    print("=" * 60)

    np.random.seed(42)

    # Regression losses
    print("\n--- Regression Losses ---")
    y_true_reg = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    # TODO: Create y_pred_good (close to y_true) and y_pred_outlier (one large error)
    # Hint: y_pred_good = y_true_reg + np.random.randn(5) * 0.1
    # Hint: y_pred_outlier = y_pred_good.copy(); y_pred_outlier[2] = 100.0

    # TODO: Compute and print MSE, MAE, Huber for both prediction sets

    # Classification losses
    print("\n--- Classification Losses ---")

    # TODO: Binary cross-entropy
    # y_true_binary = np.array([1, 0, 1, 1, 0])
    # y_pred_binary = np.array([0.9, 0.1, 0.8, 0.3, 0.2])

    # TODO: Categorical cross-entropy (5 samples, 3 classes)
    # y_true_cat = np.zeros((5, 3)); y_true_cat[np.arange(5), [0, 1, 2, 0, 1]] = 1
    # y_pred_cat = np.array([[0.7, 0.2, 0.1], ...])

    # TODO: Hinge loss (convert binary to {-1, +1})
    # y_true_hinge = np.where(y_true_binary == 1, 1, -1)
    # y_pred_scores = np.array([1.5, -0.8, 1.2, -0.3, -1.1])  # raw scores

    # TODO: Focal loss
    # y_pred_imbalanced = np.array([0.95, 0.05, 0.90, 0.6, 0.1])  # mostly confident

    # Plot loss curves
    print("\n--- Plotting Loss Curves ---")
    plot_loss_curves()

    # TODO: Call compare_losses_on_outliers
    # compare_losses_on_outliers(y_true_reg, y_pred_good, y_pred_outlier)


if __name__ == "__main__":
    main()
