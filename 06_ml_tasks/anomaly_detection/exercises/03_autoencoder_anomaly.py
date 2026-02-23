"""
Exercise 03: Autoencoder-Based Anomaly Detection
=================================================
Train an autoencoder on normal data only. At inference time, flag samples
with high reconstruction error as anomalies — the model cannot reconstruct
patterns it has never seen.

Learning Goals:
    - Build and train a dense autoencoder in PyTorch
    - Understand the "train only on normal, test on mixed" paradigm
    - Set anomaly thresholds using reconstruction error percentiles
    - Visualize reconstruction error distributions

Requirements:
    pip install torch numpy matplotlib scikit-learn
"""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# 1. Autoencoder Model
# ---------------------------------------------------------------------------

class Autoencoder(nn.Module):
    """
    Fully-connected (dense) autoencoder for tabular data.

    Architecture:
        Encoder: input_dim → hidden → encoding_dim  (with ReLU activations)
        Decoder: encoding_dim → hidden → input_dim  (with ReLU + Sigmoid/Identity)

    Args:
        input_dim:    Dimensionality of the input features.
        encoding_dim: Size of the bottleneck (latent) representation.
        hidden_dims:  List of hidden layer sizes in the encoder
                      (decoder mirrors this in reverse).
    """

    def __init__(
        self,
        input_dim: int,
        encoding_dim: int = 8,
        hidden_dims: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]

        # TODO:
        #   Build self.encoder as an nn.Sequential:
        #       input_dim → hidden_dims[0] → ReLU
        #       hidden_dims[0] → hidden_dims[1] → ReLU
        #       ...
        #       hidden_dims[-1] → encoding_dim  (no activation on bottleneck)
        #
        #   Build self.decoder as an nn.Sequential (mirror of encoder):
        #       encoding_dim → hidden_dims[-1] → ReLU
        #       ...
        #       hidden_dims[0] → input_dim  (use Sigmoid for [0,1] data or Identity)
        #
        #   Hint: use a loop to build layers dynamically:
        #       encoder_layers = []
        #       dims = [input_dim] + hidden_dims + [encoding_dim]
        #       for i in range(len(dims) - 1):
        #           encoder_layers.append(nn.Linear(dims[i], dims[i+1]))
        #           if i < len(dims) - 2:
        #               encoder_layers.append(nn.ReLU())
        #       self.encoder = nn.Sequential(*encoder_layers)
        raise NotImplementedError("TODO: implement Autoencoder.__init__()")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode and then decode.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Reconstructed tensor of shape (batch_size, input_dim).

        TODO:
            1. Pass x through self.encoder to get latent = self.encoder(x).
            2. Pass latent through self.decoder to get reconstruction.
            3. Return reconstruction.
        """
        raise NotImplementedError("TODO: implement Autoencoder.forward()")

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample mean squared reconstruction error.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            1D tensor of shape (batch_size,) with per-sample MSE.

        TODO:
            1. Get reconstruction = self.forward(x).
            2. Compute squared differences: diff = (x - reconstruction) ** 2.
            3. Average across feature dimension: error = diff.mean(dim=1).
            4. Return error (shape: [batch_size]).
        """
        raise NotImplementedError("TODO: implement Autoencoder.reconstruction_error()")


# ---------------------------------------------------------------------------
# 2. Training
# ---------------------------------------------------------------------------

def train_autoencoder(
    model: Autoencoder,
    X_normal: np.ndarray,
    n_epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> List[float]:
    """
    Train the autoencoder using only normal (non-anomalous) data.

    The model learns to reconstruct normal patterns. It will have high
    reconstruction error on anomalies because it never learned their structure.

    Args:
        model:     The Autoencoder model.
        X_normal:  2D numpy array of shape (n_normal_samples, input_dim).
                   Contains ONLY normal (non-anomalous) training samples.
        n_epochs:  Number of training epochs.
        lr:        Learning rate for Adam optimizer.
        batch_size: Mini-batch size.
        device:    Torch device. Defaults to auto-detect CPU/GPU.

    Returns:
        List of average training loss per epoch.

    TODO:
        1. Set device. Move model to device.
        2. Convert X_normal to a FloatTensor and create DataLoader.
        3. Define optimizer = torch.optim.Adam(model.parameters(), lr=lr).
        4. Define criterion = nn.MSELoss().
        5. For each epoch:
             a. model.train()
             b. For each batch x_batch:
                  - Move x_batch to device.
                  - Forward pass: x_reconstructed = model(x_batch).
                  - Compute loss = criterion(x_reconstructed, x_batch).
                  - Backward + optimizer step.
             c. Record and print average epoch loss.
        6. Return epoch loss list.

    Note:
        Unlike classification, the target IS the input (x → encode → decode → x).
        This is why MSELoss(reconstruction, input) is the right loss.
    """
    raise NotImplementedError("TODO: implement train_autoencoder()")


# ---------------------------------------------------------------------------
# 3. Anomaly Detection
# ---------------------------------------------------------------------------

def detect_anomalies(
    model: Autoencoder,
    X: np.ndarray,
    threshold_percentile: float = 95.0,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Detect anomalies by thresholding reconstruction error.

    Args:
        model:                Trained Autoencoder.
        X:                    2D numpy array of shape (n_samples, input_dim).
                              Can be a mix of normal and anomalous samples.
        threshold_percentile: Percentile of reconstruction errors to use as
                              the anomaly threshold (e.g., 95 means the top
                              5% of errors are flagged as anomalies).
        device:               Torch device.

    Returns:
        errors:       1D numpy array of per-sample reconstruction errors.
        anomaly_mask: Boolean array. True = predicted anomaly.
        threshold:    The computed error threshold value.

    TODO:
        1. Set model to eval mode.
        2. Convert X to a FloatTensor and move to device.
        3. With torch.no_grad():
               errors = model.reconstruction_error(X_tensor).cpu().numpy()
        4. threshold = np.percentile(errors, threshold_percentile).
        5. anomaly_mask = errors > threshold.
        6. Return (errors, anomaly_mask, threshold).
    """
    raise NotImplementedError("TODO: implement detect_anomalies()")


# ---------------------------------------------------------------------------
# 4. Visualization (provided — complete implementation)
# ---------------------------------------------------------------------------

def plot_reconstruction_errors(
    errors_normal: np.ndarray,
    errors_anomalous: np.ndarray,
    threshold: Optional[float] = None,
) -> None:
    """
    Plot histogram of reconstruction errors for normal vs. anomalous samples.

    Args:
        errors_normal:    Reconstruction errors for true normal samples.
        errors_anomalous: Reconstruction errors for true anomalous samples.
        threshold:        Optional threshold line to draw.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    bins = np.linspace(
        min(errors_normal.min(), errors_anomalous.min()),
        max(errors_normal.max(), errors_anomalous.max()),
        50,
    )

    ax.hist(errors_normal, bins=bins, alpha=0.6, color="steelblue",
            label=f"Normal (n={len(errors_normal)})", density=True)
    ax.hist(errors_anomalous, bins=bins, alpha=0.6, color="tomato",
            label=f"Anomalous (n={len(errors_anomalous)})", density=True)

    if threshold is not None:
        ax.axvline(threshold, color="black", linestyle="--", linewidth=1.8,
                   label=f"Threshold = {threshold:.4f}")

    ax.set_xlabel("Reconstruction Error (MSE)")
    ax.set_ylabel("Density")
    ax.set_title("Reconstruction Error Distribution: Normal vs. Anomalous")
    ax.legend()
    plt.tight_layout()
    plt.savefig("reconstruction_error_distribution.png", dpi=100)
    plt.show()
    print("[Plot saved: reconstruction_error_distribution.png]")


def visualize_reconstructions(
    model: Autoencoder,
    X_sample: np.ndarray,
    device: Optional[torch.device] = None,
    n_show: int = 8,
    feature_names: Optional[List[str]] = None,
) -> None:
    """
    Visualize input vs. reconstruction for a sample of data points.

    Args:
        model:         Trained Autoencoder.
        X_sample:      Small array of shape (n_samples, input_dim) to visualize.
        device:        Torch device.
        n_show:        Number of samples to display.
        feature_names: Optional list of feature names for x-axis labels.
    """
    if device is None:
        device = torch.device("cpu")

    model.eval()
    X_tensor = torch.FloatTensor(X_sample[:n_show]).to(device)

    with torch.no_grad():
        reconstructed = model(X_tensor).cpu().numpy()

    n_show = min(n_show, len(X_sample))
    fig, axes = plt.subplots(n_show, 1, figsize=(10, n_show * 2))
    if n_show == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        x_i = X_sample[i]
        r_i = reconstructed[i]
        x_axis = np.arange(len(x_i))

        ax.bar(x_axis - 0.2, x_i, 0.4, label="Original", color="steelblue", alpha=0.8)
        ax.bar(x_axis + 0.2, r_i, 0.4, label="Reconstructed", color="tomato", alpha=0.8)

        mse = np.mean((x_i - r_i) ** 2)
        ax.set_title(f"Sample {i+1} | Reconstruction MSE = {mse:.5f}")
        if feature_names:
            ax.set_xticks(x_axis)
            ax.set_xticklabels(feature_names, rotation=45, ha="right")
        if i == 0:
            ax.legend()

    plt.tight_layout()
    plt.savefig("reconstruction_visualizations.png", dpi=100)
    plt.show()
    print("[Plot saved: reconstruction_visualizations.png]")


# ---------------------------------------------------------------------------
# Helper: Generate synthetic data
# ---------------------------------------------------------------------------

def generate_data(
    n_normal_train: int = 500,
    n_normal_test: int = 100,
    n_anomalies_test: int = 50,
    n_features: int = 10,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic tabular data: normal samples from a Gaussian,
    anomalies from a different distribution.

    Returns:
        X_train_normal:  Normal samples for training only.
        X_test:          Mix of normal and anomalous samples.
        y_test:          Boolean labels (True = anomaly) for test set.
        feature_names:   List of feature name strings.
    """
    rng = np.random.default_rng(seed)

    # Normal: zero-mean, some correlations
    cov = np.eye(n_features) * 0.5 + 0.5
    mean_normal = np.zeros(n_features)
    X_train_normal = rng.multivariate_normal(mean_normal, cov, n_normal_train).astype(np.float32)
    X_test_normal = rng.multivariate_normal(mean_normal, cov, n_normal_test).astype(np.float32)

    # Anomalies: shifted mean + different covariance
    mean_anomaly = rng.uniform(3, 6, n_features)
    X_test_anomaly = rng.multivariate_normal(mean_anomaly, np.eye(n_features), n_anomalies_test).astype(np.float32)

    # Normalize training data to [~0, ~1] range using train stats
    train_min = X_train_normal.min(axis=0)
    train_max = X_train_normal.max(axis=0)
    X_train_normal = (X_train_normal - train_min) / (train_max - train_min + 1e-8)
    X_test_normal  = (X_test_normal  - train_min) / (train_max - train_min + 1e-8)
    X_test_anomaly = (X_test_anomaly - train_min) / (train_max - train_min + 1e-8)

    # Combine test set
    X_test = np.vstack([X_test_normal, X_test_anomaly])
    y_test = np.zeros(len(X_test), dtype=bool)
    y_test[n_normal_test:] = True

    idx = rng.permutation(len(X_test))
    X_test, y_test = X_test[idx], y_test[idx]

    feature_names = [f"feat_{i}" for i in range(n_features)]
    return X_train_normal, X_test, y_test, feature_names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Train an autoencoder on normal data and detect anomalies in the test set.

    Steps:
        1. Generate synthetic normal training data and mixed test data.
        2. Build and train the autoencoder.
        3. Compute reconstruction errors on the test set.
        4. Set threshold at 95th percentile.
        5. Evaluate precision, recall, F1 against true labels.
        6. Visualize error distributions and sample reconstructions.
    """
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data ---
    X_train, X_test, y_test, feature_names = generate_data(
        n_normal_train=500, n_normal_test=100, n_anomalies_test=50, n_features=10
    )
    print(f"\nTraining samples (normal only): {len(X_train)}")
    print(f"Test samples: {len(X_test)} ({y_test.sum()} anomalies)")

    # --- Model ---
    model = Autoencoder(input_dim=10, encoding_dim=4, hidden_dims=[32, 16])
    print(f"\nModel architecture:\n{model}")

    # --- Train ---
    losses = train_autoencoder(model, X_train, n_epochs=60, lr=1e-3, batch_size=64, device=device)

    # --- Detect ---
    errors, anomaly_mask, threshold = detect_anomalies(model, X_test, threshold_percentile=95.0, device=device)
    print(f"\nAnomaly threshold (95th percentile): {threshold:.5f}")
    print(f"Flagged as anomaly: {anomaly_mask.sum()} (true: {y_test.sum()})")

    # --- Evaluate ---
    tp = (y_test & anomaly_mask).sum()
    fp = (~y_test & anomaly_mask).sum()
    fn = (y_test & ~anomaly_mask).sum()
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    print(f"\nPrecision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}")

    # --- Visualize ---
    errors_normal    = errors[~y_test]
    errors_anomalous = errors[y_test]
    plot_reconstruction_errors(errors_normal, errors_anomalous, threshold=threshold)

    # Show reconstructions for a few normal and anomalous samples
    print("\n--- Normal sample reconstructions ---")
    visualize_reconstructions(model, X_test[~y_test][:4], device=device,
                               n_show=4, feature_names=feature_names)

    print("\n--- Anomalous sample reconstructions ---")
    visualize_reconstructions(model, X_test[y_test][:4], device=device,
                               n_show=4, feature_names=feature_names)


if __name__ == "__main__":
    main()
