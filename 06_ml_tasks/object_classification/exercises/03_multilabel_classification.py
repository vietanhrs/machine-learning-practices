"""
Exercise 03: Multi-Label Classification
========================================
Implement multi-label classification where each sample can simultaneously
belong to multiple classes. Unlike multi-class, classes are treated independently
using sigmoid activations and binary cross-entropy loss.

Learning Goals:
    - Understand the difference between multi-class (softmax) and multi-label (sigmoid)
    - Implement a multi-label neural network in PyTorch
    - Compute multi-label-specific metrics (Hamming loss, subset accuracy, per-label F1)
    - Visualize label co-occurrence patterns

Requirements:
    pip install torch numpy scikit-learn matplotlib
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, hamming_loss
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# 1. Sigmoid Function
# ---------------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Compute the element-wise sigmoid function.

    Args:
        x: Numpy array of any shape.

    Returns:
        Array of the same shape with values in (0, 1).

    TODO:
        Implement: σ(x) = 1 / (1 + exp(-x))

        Handle numerical overflow for very negative x:
            Use np.clip(x, -500, 500) before computing exp, or use:
            np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
    """
    raise NotImplementedError("TODO: implement sigmoid()")


# ---------------------------------------------------------------------------
# 2. Multi-Label Neural Network
# ---------------------------------------------------------------------------

class MultiLabelClassifier(nn.Module):
    """
    Fully-connected neural network for multi-label classification.

    Architecture:
        Linear(input_dim, 128) → ReLU → Dropout
        Linear(128, 64) → ReLU → Dropout
        Linear(64, n_labels) → (no activation — raw logits)

    Note:
        During inference, apply torch.sigmoid() to get per-label probabilities.
        The final layer has NO activation — sigmoid is applied externally or
        inside the loss function (nn.BCEWithLogitsLoss is numerically stable).

    Args:
        input_dim: Number of input features.
        n_labels:  Number of output labels (one sigmoid per label).
        dropout:   Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        n_labels: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        # TODO:
        #   Build a Sequential network:
        #       nn.Linear(input_dim, 128) → nn.ReLU() → nn.Dropout(dropout)
        #       nn.Linear(128, 64) → nn.ReLU() → nn.Dropout(dropout)
        #       nn.Linear(64, n_labels)   ← NO activation here
        #   Store as self.network = nn.Sequential(...)
        raise NotImplementedError("TODO: implement MultiLabelClassifier.__init__()")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Float tensor of shape (batch_size, input_dim).

        Returns:
            Raw logits of shape (batch_size, n_labels).
            (Apply sigmoid externally to get probabilities.)

        TODO:
            Return self.network(x).
        """
        raise NotImplementedError("TODO: implement MultiLabelClassifier.forward()")


# ---------------------------------------------------------------------------
# 3. Binary Cross-Entropy Loss for Multi-Label
# ---------------------------------------------------------------------------

def binary_cross_entropy_multilabel(
    y_true: torch.Tensor,
    logits: torch.Tensor,
) -> torch.Tensor:
    """
    Compute numerically stable binary cross-entropy loss for multi-label classification.

    Args:
        y_true:  Float tensor of shape (batch_size, n_labels). Values in {0.0, 1.0}.
        logits:  Raw logit tensor of shape (batch_size, n_labels) — NOT probabilities.

    Returns:
        Scalar loss value (mean BCE over all labels and batch).

    TODO:
        Use nn.BCEWithLogitsLoss()(logits, y_true).

        BCEWithLogitsLoss applies sigmoid internally and computes:
            BCE = -[y * log(σ(x)) + (1-y) * log(1 - σ(x))]
        It is numerically more stable than applying sigmoid then BCELoss.

    Note:
        Do NOT apply sigmoid to logits before passing them here.
        BCEWithLogitsLoss expects raw logits.
    """
    raise NotImplementedError("TODO: implement binary_cross_entropy_multilabel()")


# ---------------------------------------------------------------------------
# 4. Threshold Predictions
# ---------------------------------------------------------------------------

def threshold_predictions(
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Convert predicted probabilities to binary multi-label predictions.

    Args:
        y_pred_proba: Float array of shape (n_samples, n_labels).
                      Values in [0, 1] (sigmoid outputs).
        threshold:    Decision threshold. Probabilities above this are set to 1.

    Returns:
        Binary integer array of shape (n_samples, n_labels). Values in {0, 1}.

    TODO:
        Return (y_pred_proba >= threshold).astype(int).
    """
    raise NotImplementedError("TODO: implement threshold_predictions()")


# ---------------------------------------------------------------------------
# 5. Multi-Label Metrics
# ---------------------------------------------------------------------------

def multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute standard multi-label classification metrics.

    Args:
        y_true:      Binary int array of shape (n_samples, n_labels). Ground truth.
        y_pred:      Binary int array of shape (n_samples, n_labels). Predictions.
        label_names: Optional list of label name strings for per-label report.

    Returns:
        Dictionary with metric names → values:
            - "hamming_loss":    Fraction of label/sample pairs that are incorrectly labeled
            - "subset_accuracy": Fraction of samples where ALL labels are predicted correctly
            - "macro_f1":        F1 score averaged uniformly across labels
            - "micro_f1":        F1 score computed globally across all label instances
            - Per-label F1 scores if label_names is provided

    TODO:
        1. Hamming loss:
               hamming = hamming_loss(y_true, y_pred)
               (or compute manually: np.mean(y_true != y_pred))
        2. Subset accuracy (exact match ratio):
               subset_acc = np.mean(np.all(y_true == y_pred, axis=1))
        3. Macro F1 (average per-label F1, unweighted):
               macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        4. Micro F1 (globally across all labels):
               micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
        5. Per-label F1 if label_names given:
               per_label = f1_score(y_true, y_pred, average=None, zero_division=0)
        6. Print a formatted report.
        7. Return the metrics dict.
    """
    raise NotImplementedError("TODO: implement multilabel_metrics()")


# ---------------------------------------------------------------------------
# 6. Plot Label Co-occurrence (provided — complete implementation)
# ---------------------------------------------------------------------------

def plot_label_co_occurrence(
    y_true: np.ndarray,
    label_names: List[str],
) -> None:
    """
    Plot a heatmap showing how frequently pairs of labels co-occur in the dataset.

    Args:
        y_true:      Binary int array of shape (n_samples, n_labels).
        label_names: List of label names for axis ticks.
    """
    n_labels = y_true.shape[1]
    co_occurrence = np.zeros((n_labels, n_labels), dtype=int)

    for i in range(n_labels):
        for j in range(n_labels):
            co_occurrence[i, j] = np.sum(y_true[:, i] & y_true[:, j])

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(co_occurrence, cmap="Blues")
    plt.colorbar(im, ax=ax, label="Co-occurrence count")

    ax.set_xticks(range(n_labels))
    ax.set_yticks(range(n_labels))
    ax.set_xticklabels(label_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(label_names, fontsize=9)

    # Annotate cells
    for i in range(n_labels):
        for j in range(n_labels):
            ax.text(j, i, str(co_occurrence[i, j]),
                    ha="center", va="center", fontsize=8,
                    color="white" if co_occurrence[i, j] > co_occurrence.max() * 0.6 else "black")

    ax.set_title("Label Co-occurrence Matrix")
    plt.tight_layout()
    plt.savefig("label_co_occurrence.png", dpi=100)
    plt.show()
    print("[Plot saved: label_co_occurrence.png]")


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_multilabel_model(
    model: MultiLabelClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_epochs: int = 40,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> List[float]:
    """
    Train the multi-label classifier.

    Args:
        model:      MultiLabelClassifier instance.
        X_train:    Float array of shape (n_samples, n_features).
        y_train:    Binary float array of shape (n_samples, n_labels).
        n_epochs:   Number of training epochs.
        lr:         Learning rate.
        batch_size: Mini-batch size.
        device:     Torch device.

    Returns:
        List of average training losses per epoch.

    Note (complete implementation provided):
        This function is fully implemented for you to focus effort on the model
        and metrics functions above. Study it to understand the training loop.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    X_tensor = torch.FloatTensor(X_train).to(device)
    y_tensor = torch.FloatTensor(y_train).to(device)
    dataset  = TensorDataset(X_tensor, y_tensor)
    loader   = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    epoch_losses = []
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            logits = model(x_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        epoch_losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1:3d}/{n_epochs}]  Loss: {avg_loss:.4f}")

    return epoch_losses


def predict_multilabel(
    model: MultiLabelClassifier,
    X: np.ndarray,
    threshold: float = 0.5,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate multi-label predictions for the given data.

    Returns:
        probas: Float array of sigmoid probabilities, shape (n_samples, n_labels).
        preds:  Binary int array of predictions, shape (n_samples, n_labels).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    X_tensor = torch.FloatTensor(X).to(device)
    with torch.no_grad():
        logits = model(X_tensor)
        probas = torch.sigmoid(logits).cpu().numpy()

    preds = threshold_predictions(probas, threshold=threshold)
    return probas, preds


# ---------------------------------------------------------------------------
# Helper: Generate synthetic multi-label dataset
# ---------------------------------------------------------------------------

def generate_multilabel_dataset(
    n_samples: int = 1000,
    n_features: int = 20,
    n_labels: int = 5,
    label_density: float = 0.3,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Generate a synthetic multi-label dataset with correlated labels.

    Returns:
        X:           Float array (n_samples, n_features).
        y:           Binary int array (n_samples, n_labels).
        label_names: List of label name strings.
    """
    rng = np.random.default_rng(seed)

    # Feature matrix from a Gaussian
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)

    # Labels: each label is a threshold on a linear combination of features
    W = rng.standard_normal((n_features, n_labels))
    logits = X @ W + rng.standard_normal((n_samples, n_labels)) * 0.5
    # Convert to binary using sigmoid + threshold
    probs = 1 / (1 + np.exp(-logits))
    y = (probs > (1 - label_density)).astype(np.int32)

    # Add some label correlations (label 0 and 1 often co-occur)
    co_occur_mask = y[:, 0] == 1
    y[co_occur_mask, 1] = np.where(rng.random(co_occur_mask.sum()) < 0.6, 1, y[co_occur_mask, 1])

    label_names = [f"Category_{chr(65+i)}" for i in range(n_labels)]
    return X, y, label_names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Train and evaluate a multi-label classifier on a synthetic dataset.

    Steps:
        1. Generate synthetic multi-label data.
        2. Show label distribution and co-occurrence.
        3. Train MultiLabelClassifier.
        4. Evaluate with multi-label metrics.
        5. Show per-label performance comparison.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Multi-Label Classification")
    print("=" * 60)

    # --- Data ---
    X, y, label_names = generate_multilabel_dataset(
        n_samples=1000, n_features=20, n_labels=5, label_density=0.3
    )

    # Train/test split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"\nDataset: {len(X)} samples | {len(label_names)} labels")
    print(f"Label names: {label_names}")
    print("\nLabel frequencies:")
    for i, name in enumerate(label_names):
        freq = y[:, i].mean()
        print(f"  {name}: {freq:.3f} ({int(y[:, i].sum())} positive examples)")

    # --- Label co-occurrence ---
    plot_label_co_occurrence(y, label_names)

    # --- Model ---
    model = MultiLabelClassifier(input_dim=20, n_labels=5, dropout=0.3)
    print(f"\nModel:\n{model}")

    # --- Train ---
    losses = train_multilabel_model(
        model, X_train, y_train.astype(np.float32),
        n_epochs=50, lr=1e-3, batch_size=64, device=device
    )

    # --- Predict ---
    probas, preds = predict_multilabel(model, X_test, threshold=0.5, device=device)

    # --- Metrics ---
    print("\n--- Multi-Label Evaluation Metrics ---")
    metrics = multilabel_metrics(y_test, preds, label_names=label_names)

    # --- Sigmoid sanity check ---
    print("\n--- Sigmoid Sanity Check ---")
    test_inputs = np.array([-10.0, -2.0, 0.0, 2.0, 10.0])
    print(f"sigmoid({test_inputs}) = {sigmoid(test_inputs).round(4)}")

    # --- Threshold sensitivity ---
    print("\n--- Effect of Decision Threshold ---")
    for thresh in [0.3, 0.5, 0.7]:
        preds_t = threshold_predictions(probas, threshold=thresh)
        hl = hamming_loss(y_test, preds_t)
        macro_f1 = f1_score(y_test, preds_t, average='macro', zero_division=0)
        print(f"  Threshold={thresh}  Hamming={hl:.3f}  Macro-F1={macro_f1:.3f}")


if __name__ == "__main__":
    main()
