"""
Exercise 02: Dropout and Batch Normalization
=============================================
Implement neural networks with and without dropout/batch normalization.
Train on a dataset designed to cause overfitting and compare regularization
effects using training and validation loss curves.

Learning objectives:
- Implement Dropout and BatchNorm in PyTorch
- Correctly use model.train() and model.eval()
- Observe overfitting without regularization
- Compare dropout vs batch normalization effects

Dependencies: torch, numpy, matplotlib, scikit-learn
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Network Architectures
# ---------------------------------------------------------------------------

class NetworkBaseline(nn.Module):
    """
    A 3-layer MLP with NO regularization.

    This network is designed to overfit on small datasets:
    - Large hidden dimensions relative to dataset size
    - No dropout, no batch normalization

    Architecture: input → 256 → 128 → 64 → 1 (binary classification)

    Args:
        input_dim (int): Number of input features
    """

    def __init__(self, input_dim):
        super().__init__()
        # TODO: Define self.network using nn.Sequential with:
        #   - nn.Linear(input_dim, 256)
        #   - nn.ReLU()
        #   - nn.Linear(256, 128)
        #   - nn.ReLU()
        #   - nn.Linear(128, 64)
        #   - nn.ReLU()
        #   - nn.Linear(64, 1)
        # No dropout, no batch normalization
        pass

    def forward(self, x):
        # TODO: Pass x through self.network and squeeze the last dimension
        # Return logits (shape: N,) for BCEWithLogitsLoss
        pass


class NetworkWithDropout(nn.Module):
    """
    A 3-layer MLP WITH Dropout regularization.

    Dropout is applied after each ReLU activation (before the next linear layer).
    IMPORTANT:
    - During training (model.train()): dropout is ACTIVE — randomly zeros neurons
    - During evaluation (model.eval()): dropout is INACTIVE — full network

    Architecture: input → 256 → Dropout(p) → 128 → Dropout(p) → 64 → Dropout(p) → 1

    Args:
        input_dim (int): Number of input features
        dropout_rate (float): Probability of zeroing a neuron (default 0.5)
    """

    def __init__(self, input_dim, dropout_rate=0.5):
        super().__init__()
        # TODO: Define self.network using nn.Sequential with:
        #   - nn.Linear(input_dim, 256)
        #   - nn.ReLU()
        #   - nn.Dropout(p=dropout_rate)    ← add after activation
        #   - nn.Linear(256, 128)
        #   - nn.ReLU()
        #   - nn.Dropout(p=dropout_rate)
        #   - nn.Linear(128, 64)
        #   - nn.ReLU()
        #   - nn.Dropout(p=dropout_rate)
        #   - nn.Linear(64, 1)
        pass

    def forward(self, x):
        # TODO: Pass through self.network and squeeze
        pass


class NetworkWithBatchNorm(nn.Module):
    """
    A 3-layer MLP WITH Batch Normalization.

    BatchNorm is applied between linear layers and activation functions.
    Convention: Linear → BatchNorm → Activation

    IMPORTANT: BatchNorm behavior differs between train() and eval() modes:
    - model.train(): uses mini-batch statistics
    - model.eval(): uses running statistics (accumulated during training)

    Architecture: input → 256 → BN → ReLU → 128 → BN → ReLU → 64 → BN → ReLU → 1

    Args:
        input_dim (int): Number of input features
    """

    def __init__(self, input_dim):
        super().__init__()
        # TODO: Define self.network using nn.Sequential with:
        #   - nn.Linear(input_dim, 256)
        #   - nn.BatchNorm1d(256)    ← before activation
        #   - nn.ReLU()
        #   - nn.Linear(256, 128)
        #   - nn.BatchNorm1d(128)
        #   - nn.ReLU()
        #   - nn.Linear(128, 64)
        #   - nn.BatchNorm1d(64)
        #   - nn.ReLU()
        #   - nn.Linear(64, 1)
        pass

    def forward(self, x):
        # TODO: Pass through self.network and squeeze
        pass


# ---------------------------------------------------------------------------
# Training and Evaluation
# ---------------------------------------------------------------------------

def train_and_compare(X_train, y_train, X_val, y_val, networks_dict, n_epochs=100, lr=0.001):
    """
    Train multiple networks and record their train/validation loss curves.

    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels {0, 1}
        X_val (np.ndarray): Validation features
        y_val (np.ndarray): Validation labels {0, 1}
        networks_dict (dict): {name: nn.Module} — networks to train
        n_epochs (int): Training epochs
        lr (float): Learning rate

    Returns:
        dict: {name: {"train_loss": [...], "val_loss": [...]}}

    Hints:
        - Convert numpy arrays to tensors ONCE before the training loop:
          X_train_t = torch.FloatTensor(X_train)
          y_train_t = torch.FloatTensor(y_train)
          X_val_t = torch.FloatTensor(X_val)
          y_val_t = torch.FloatTensor(y_val)
        - Use nn.BCEWithLogitsLoss() as the loss function
        - For each network (model), training loop:
            1. model.train()        ← enable dropout/batch stats
            2. optimizer.zero_grad()
            3. logits = model(X_train_t)
            4. loss = criterion(logits, y_train_t)
            5. loss.backward()
            6. optimizer.step()
            7. model.eval()         ← disable dropout, use running stats
            8. with torch.no_grad():
                val_logits = model(X_val_t)
                val_loss = criterion(val_logits, y_val_t)
        - Record train_loss and val_loss each epoch
        - IMPORTANT: Always call model.train() before training step
                     and model.eval() before evaluation
    """
    results = {name: {"train_loss": [], "val_loss": []} for name in networks_dict}

    # TODO: Convert data to tensors
    # TODO: Initialize loss function: nn.BCEWithLogitsLoss()
    # TODO: For each network in networks_dict:
    #   - Initialize Adam optimizer
    #   - For each epoch:
    #       - model.train() → forward → backward → step → record train loss
    #       - model.eval() → with torch.no_grad(): evaluate → record val loss
    # TODO: Return results dict
    pass


# ---------------------------------------------------------------------------
# Visualization (Complete — Do Not Modify)
# ---------------------------------------------------------------------------

def plot_train_val_curves(results):
    """
    Plot training and validation loss curves for all networks.

    Args:
        results (dict): {name: {"train_loss": [...], "val_loss": [...]}}
    """
    n_networks = len(results)
    fig, axes = plt.subplots(1, n_networks, figsize=(6 * n_networks, 4), sharey=True)
    if n_networks == 1:
        axes = [axes]

    colors = {'train': 'blue', 'val': 'red'}

    for ax, (name, metrics) in zip(axes, results.items()):
        ax.plot(metrics["train_loss"], color=colors['train'],
                linewidth=2, label="Train Loss")
        ax.plot(metrics["val_loss"], color=colors['val'],
                linewidth=2, label="Val Loss", linestyle='--')

        final_train = metrics["train_loss"][-1]
        final_val = metrics["val_loss"][-1]
        gap = final_val - final_train

        ax.set_title(f"{name}\nFinal: Train={final_train:.3f}, Val={final_val:.3f}\nGap={gap:.3f}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Regularization Comparison: Train vs Validation Loss", fontsize=14)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    Compare three networks on an overfitting-prone setup:
    1. Baseline (no regularization) — should overfit
    2. With Dropout — should generalize better
    3. With Batch Normalization — should generalize better + faster convergence

    Steps:
    1. Create small dataset (500 samples, 20 features) — easy to overfit
    2. Train all three networks
    3. Plot training/validation curves
    4. Compare final train-val gap (smaller gap = less overfitting)
    """
    print("=" * 60)
    print("Dropout vs Batch Normalization vs Baseline")
    print("=" * 60)
    torch.manual_seed(42)
    np.random.seed(42)

    # Create dataset — intentionally small to induce overfitting
    X_raw, y = make_classification(
        n_samples=500, n_features=20, n_informative=10,
        n_redundant=5, random_state=42
    )

    # Split: 300 train, 200 val
    X_train, X_val, y_train, y_val = train_test_split(
        X_raw, y, test_size=0.4, random_state=42
    )

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    input_dim = X_train.shape[1]
    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")

    # TODO: Initialize all three networks
    # networks = {
    #     "Baseline (No Reg)": NetworkBaseline(input_dim),
    #     "Dropout (p=0.5)": NetworkWithDropout(input_dim, dropout_rate=0.5),
    #     "Batch Norm": NetworkWithBatchNorm(input_dim),
    # }

    # TODO: Train all networks and get results
    # results = train_and_compare(X_train, y_train, X_val, y_val, networks, n_epochs=150)

    # TODO: Plot results
    # plot_train_val_curves(results)

    # TODO: Print summary: train-val gap for each network
    # Hint: smaller gap = better regularization
    print("\nExpected observations:")
    print("- Baseline: Large train-val gap (overfitting)")
    print("- Dropout: Smaller gap, smoother val curve")
    print("- BatchNorm: Faster convergence, good generalization")


if __name__ == "__main__":
    main()
