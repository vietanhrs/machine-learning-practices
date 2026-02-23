"""
Exercise 01: Statistical Anomaly Detection
==========================================
Implement classical statistical anomaly detection methods:
z-score, IQR, moving average, and Mahalanobis distance.
Evaluate and visualize each detector on synthetic time series data.

Learning Goals:
    - Understand the assumptions behind each statistical method
    - Implement univariate and multivariate anomaly detection
    - Evaluate detectors using precision, recall, and F1 score
    - Visualize anomalies in time series and multivariate data

Requirements:
    pip install numpy scipy matplotlib scikit-learn
"""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import f1_score, precision_score, recall_score


# ---------------------------------------------------------------------------
# 1. Z-Score Anomaly Detection
# ---------------------------------------------------------------------------

def zscore_anomaly(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """
    Flag data points as anomalies if their z-score exceeds the threshold.

    Args:
        data:      1D numpy array of observations.
        threshold: Z-score magnitude above which a point is flagged.

    Returns:
        Boolean array of the same shape as data.
        True = anomaly, False = normal.

    TODO:
        1. Compute the mean (mu) and standard deviation (sigma) of data.
        2. Compute z_scores = (data - mu) / sigma.
        3. Return np.abs(z_scores) > threshold.

    Note:
        If sigma == 0 (all values identical), return an array of all False
        to avoid division by zero.
    """
    raise NotImplementedError("TODO: implement zscore_anomaly()")


# ---------------------------------------------------------------------------
# 2. IQR Anomaly Detection
# ---------------------------------------------------------------------------

def iqr_anomaly(data: np.ndarray, multiplier: float = 1.5) -> np.ndarray:
    """
    Flag data points as anomalies using the IQR (interquartile range) method.

    Args:
        data:       1D numpy array of observations.
        multiplier: How many IQRs beyond Q1/Q3 to set the fences.
                    Standard value is 1.5 (Tukey's fences).

    Returns:
        Boolean array. True = anomaly, False = normal.

    TODO:
        1. Compute Q1 = np.percentile(data, 25) and Q3 = np.percentile(data, 75).
        2. Compute IQR = Q3 - Q1.
        3. Lower fence = Q1 - multiplier * IQR.
        4. Upper fence = Q3 + multiplier * IQR.
        5. Return (data < lower_fence) | (data > upper_fence).
    """
    raise NotImplementedError("TODO: implement iqr_anomaly()")


# ---------------------------------------------------------------------------
# 3. Moving Average Anomaly Detection
# ---------------------------------------------------------------------------

def moving_average_anomaly(
    time_series: np.ndarray,
    window: int,
    n_std: float = 3.0,
) -> np.ndarray:
    """
    Flag time series points that deviate more than n_std standard deviations
    from the rolling mean within a sliding window.

    Args:
        time_series: 1D numpy array representing a time series.
        window:      Size of the rolling window.
        n_std:       Number of standard deviations to use as the threshold.

    Returns:
        Boolean array. True = anomaly, False = normal.

    TODO:
        1. Compute rolling_mean and rolling_std using a sliding window of size `window`.
           For each index i:
               window_data = time_series[max(0, i-window+1) : i+1]
               rolling_mean[i] = np.mean(window_data)
               rolling_std[i]  = np.std(window_data)
           (Or use np.convolve / pandas for efficiency — but implement from scratch first.)
        2. Compute deviation = np.abs(time_series - rolling_mean).
        3. Return deviation > n_std * rolling_std.

    Hint:
        Handle the edge case at the beginning where fewer than `window` points
        are available — just use whatever data is available.
        Also handle where rolling_std is 0 to avoid division issues.
    """
    raise NotImplementedError("TODO: implement moving_average_anomaly()")


# ---------------------------------------------------------------------------
# 4. Mahalanobis Distance Anomaly Detection
# ---------------------------------------------------------------------------

def mahalanobis_anomaly(
    X: np.ndarray,
    threshold_percentile: float = 97.5,
) -> np.ndarray:
    """
    Flag multivariate anomalies using Mahalanobis distance.

    Mahalanobis distance is the multivariate generalization of z-score.
    It accounts for correlations between features and scales by covariance.

    Args:
        X:                    2D numpy array of shape (n_samples, n_features).
        threshold_percentile: Percentile of Mahalanobis distances to use as the
                              anomaly threshold. Points above this are anomalies.

    Returns:
        Boolean array of shape (n_samples,). True = anomaly.

    TODO:
        1. Compute the mean vector: mu = np.mean(X, axis=0).
        2. Compute the covariance matrix: cov = np.cov(X.T).
        3. Compute the inverse covariance: cov_inv = np.linalg.inv(cov).
           Handle near-singular matrices with np.linalg.pinv() if needed.
        4. For each sample x_i, compute Mahalanobis distance:
               d_i = sqrt((x_i - mu) @ cov_inv @ (x_i - mu).T)
           Or use scipy.spatial.distance.mahalanobis(x_i, mu, cov_inv).
        5. Compute threshold = np.percentile(distances, threshold_percentile).
        6. Return distances > threshold.

    Note:
        Under a multivariate Gaussian, the squared Mahalanobis distance follows
        a chi-squared distribution with n_features degrees of freedom.
        A common threshold is chi2.ppf(0.975, df=n_features).
    """
    raise NotImplementedError("TODO: implement mahalanobis_anomaly()")


# ---------------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------------

def evaluate_detector(
    y_true: np.ndarray,
    anomaly_mask: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Evaluate an anomaly detector against ground truth labels.

    Args:
        y_true:       Boolean or integer array. True/1 = actual anomaly.
        anomaly_mask: Boolean array from detector. True/1 = predicted anomaly.

    Returns:
        Tuple of (precision, recall, f1_score).

    TODO:
        1. Compute precision = TP / (TP + FP).
           Where TP = sum(y_true & anomaly_mask), FP = sum(~y_true & anomaly_mask).
        2. Compute recall = TP / (TP + FN).
           Where FN = sum(y_true & ~anomaly_mask).
        3. Compute F1 = 2 * precision * recall / (precision + recall).
           Handle zero denominator cases (return 0.0 for those metrics).
        4. Print a summary and return (precision, recall, f1).

    Hint:
        sklearn.metrics.precision_score, recall_score, f1_score can verify
        your results: precision_score(y_true.astype(int), anomaly_mask.astype(int))
    """
    raise NotImplementedError("TODO: implement evaluate_detector()")


# ---------------------------------------------------------------------------
# 6. Visualization (provided — complete implementation)
# ---------------------------------------------------------------------------

def plot_anomalies(
    data: np.ndarray,
    anomaly_mask: np.ndarray,
    title: str = "Anomaly Detection",
    figsize: Tuple[int, int] = (14, 4),
) -> None:
    """
    Plot a time series with detected anomalies highlighted in red.

    Args:
        data:         1D array of observations (the time series values).
        anomaly_mask: Boolean array. True = anomaly.
        title:        Plot title.
        figsize:      Figure size (width, height).
    """
    fig, ax = plt.subplots(figsize=figsize)
    t = np.arange(len(data))

    # Normal points in blue
    ax.plot(t, data, color="steelblue", linewidth=1.0, label="Normal")

    # Anomalous points as red dots
    if anomaly_mask.any():
        ax.scatter(
            t[anomaly_mask],
            data[anomaly_mask],
            color="red",
            zorder=5,
            s=50,
            label=f"Anomaly (n={anomaly_mask.sum()})",
        )

    ax.set_title(title)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Value")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}.png", dpi=100)
    plt.show()
    print(f"[Plot saved: {title.lower().replace(' ', '_')}.png]")


# ---------------------------------------------------------------------------
# Helper: Generate synthetic time series with anomalies
# ---------------------------------------------------------------------------

def generate_synthetic_time_series(
    n_points: int = 500,
    n_anomalies: int = 20,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic time series with injected point anomalies.

    Returns:
        data:    1D array of values (normal + anomalies injected).
        y_true:  Boolean array. True at anomaly positions.
    """
    rng = np.random.default_rng(seed)

    # Base signal: sinusoidal trend + Gaussian noise
    t = np.linspace(0, 4 * np.pi, n_points)
    data = np.sin(t) + 0.5 * np.sin(2 * t) + rng.normal(0, 0.2, n_points)

    # Inject anomalies
    y_true = np.zeros(n_points, dtype=bool)
    anomaly_indices = rng.choice(n_points, n_anomalies, replace=False)
    y_true[anomaly_indices] = True
    # Shift anomalies by ±3–5 std deviations
    data[anomaly_indices] += rng.choice([-1, 1], n_anomalies) * rng.uniform(3, 5, n_anomalies)

    return data, y_true


def generate_multivariate_data(
    n_normal: int = 300,
    n_anomalies: int = 20,
    n_features: int = 2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate 2D Gaussian normal data with injected anomalies.

    Returns:
        X:       Array of shape (n_normal + n_anomalies, n_features).
        y_true:  Boolean array. True for anomalous rows.
    """
    rng = np.random.default_rng(seed)

    # Normal data: correlated Gaussian
    cov = np.array([[1.0, 0.7], [0.7, 1.0]])
    X_normal = rng.multivariate_normal(mean=[0, 0], cov=cov, size=n_normal)

    # Anomalies: scattered far from center
    X_anomaly = rng.uniform(low=4, high=7, size=(n_anomalies, n_features))
    X_anomaly *= rng.choice([-1, 1], size=X_anomaly.shape)

    X = np.vstack([X_normal, X_anomaly])
    y_true = np.zeros(len(X), dtype=bool)
    y_true[n_normal:] = True

    # Shuffle
    idx = rng.permutation(len(X))
    return X[idx], y_true[idx]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Run all anomaly detectors on synthetic data and compare results.

    Steps:
        1. Generate a synthetic univariate time series with injected anomalies.
        2. Apply z-score, IQR, and moving average detectors.
        3. Evaluate and compare precision, recall, F1 for each.
        4. Generate multivariate data and apply Mahalanobis distance detection.
        5. Plot results.
    """
    print("=" * 65)
    print("Statistical Anomaly Detection")
    print("=" * 65)

    # --- Univariate time series ---
    data, y_true = generate_synthetic_time_series(n_points=500, n_anomalies=20)
    print(f"\nTime series: {len(data)} points | {y_true.sum()} true anomalies\n")

    detectors = {
        "Z-score (threshold=3)":         zscore_anomaly(data, threshold=3.0),
        "IQR (multiplier=1.5)":          iqr_anomaly(data, multiplier=1.5),
        "Moving Average (window=20)":    moving_average_anomaly(data, window=20, n_std=3.0),
    }

    print(f"{'Detector':<35} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Flagged':>10}")
    print("-" * 75)
    for name, mask in detectors.items():
        prec, rec, f1 = evaluate_detector(y_true, mask)
        print(f"{name:<35} {prec:>10.3f} {rec:>10.3f} {f1:>10.3f} {mask.sum():>10}")
        plot_anomalies(data, mask, title=name)

    # --- Multivariate: Mahalanobis ---
    print("\n--- Multivariate Mahalanobis Distance ---")
    X_multi, y_multi = generate_multivariate_data(n_normal=300, n_anomalies=20)
    maha_mask = mahalanobis_anomaly(X_multi, threshold_percentile=97.5)
    prec, rec, f1 = evaluate_detector(y_multi, maha_mask)
    print(f"Mahalanobis: Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")

    # Scatter plot for multivariate
    fig, ax = plt.subplots(figsize=(7, 6))
    normal = X_multi[~maha_mask]
    anomalous = X_multi[maha_mask]
    ax.scatter(normal[:, 0], normal[:, 1], c="steelblue", alpha=0.6, label="Normal")
    ax.scatter(anomalous[:, 0], anomalous[:, 1], c="red", s=70, label=f"Anomaly (n={maha_mask.sum()})")
    ax.set_title("Mahalanobis Anomaly Detection (2D)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("mahalanobis_anomaly_2d.png", dpi=100)
    plt.show()
    print("[Plot saved: mahalanobis_anomaly_2d.png]")


if __name__ == "__main__":
    main()
