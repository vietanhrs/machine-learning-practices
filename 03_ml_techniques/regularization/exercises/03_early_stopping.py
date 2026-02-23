"""
Exercise 03: Early Stopping
============================
Implement an EarlyStopping callback class that monitors validation loss
and stops training when the model begins to overfit. Demonstrate its
effect on a model intentionally designed to overfit.

Learning objectives:
- Implement early stopping with patience and min_delta
- Save and restore best model weights
- Visualize training progress with the stopping point marked

Dependencies: torch, numpy, matplotlib, scikit-learn
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Early Stopping Callback
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Monitor validation loss and stop training when it stops improving.

    Usage:
        early_stopping = EarlyStopping(patience=10, min_delta=1e-4)
        for epoch in range(max_epochs):
            train_one_epoch(model)
            val_loss = evaluate(model, val_data)
            early_stopping(val_loss, model)
            if early_stopping.should_stop:
                print(f"Stopped at epoch {epoch}")
                break
        model.load_state_dict(early_stopping.best_weights)  # Restore best

    Args:
        patience (int): Number of epochs to wait for improvement before stopping.
                        If no improvement for `patience` epochs, stop training.
        min_delta (float): Minimum change in val_loss to count as an improvement.
                           Improvements smaller than min_delta are ignored.
        restore_best_weights (bool): If True, restore model to best checkpoint on stop.
    """

    def __init__(self, patience=10, min_delta=1e-4, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        # TODO: Initialize tracking variables:
        # self.best_loss = float('inf')      # Best validation loss seen so far
        # self.patience_counter = 0          # Epochs since last improvement
        # self._should_stop = False          # Flag to signal training to stop
        # self.best_weights = None           # Saved model state dict (best checkpoint)
        # self.best_epoch = 0               # Epoch number of best checkpoint
        pass

    def __call__(self, val_loss, model):
        """
        Update early stopping state based on current validation loss.

        Called at the end of each epoch with the current validation loss.

        Args:
            val_loss (float): Current epoch's validation loss
            model (nn.Module): The model being trained (for saving weights)

        Logic:
            - If val_loss improved by more than min_delta below best_loss:
                - Update best_loss = val_loss
                - Reset patience_counter = 0
                - Save model weights: copy.deepcopy(model.state_dict())
                - Update best_epoch = current epoch
            - Else (no significant improvement):
                - Increment patience_counter
                - If patience_counter >= patience:
                    - Set _should_stop = True
                    - If restore_best_weights: restore model.load_state_dict(best_weights)

        Hints:
            - Use copy.deepcopy(model.state_dict()) to save a snapshot of weights
            - When restoring: model.load_state_dict(self.best_weights)
            - self.best_epoch needs to be tracked externally (pass epoch number or track internally)
        """
        # TODO: Check if val_loss improved significantly (val_loss < best_loss - min_delta)
        # TODO: If improved: update best_loss, reset counter, save weights, record epoch
        # TODO: Else: increment counter; if counter >= patience: set stop flag, optionally restore
        pass

    @property
    def should_stop(self):
        """
        Returns True if training should be stopped.

        Returns:
            bool: True when patience is exhausted and training should stop.

        Hints:
            - Simply return self._should_stop
        """
        # TODO: Return self._should_stop
        pass


# ---------------------------------------------------------------------------
# Simple Overfitting Model
# ---------------------------------------------------------------------------

class OverfittingModel(nn.Module):
    """A large MLP designed to overfit on small datasets."""

    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Training with Early Stopping
# ---------------------------------------------------------------------------

def train_with_early_stopping(model, X_train, y_train, X_val, y_val,
                               max_epochs=500, lr=0.001, patience=15):
    """
    Train a model with early stopping based on validation loss.

    Args:
        model (nn.Module): The model to train
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels {0, 1}
        X_val (np.ndarray): Validation features
        y_val (np.ndarray): Validation labels {0, 1}
        max_epochs (int): Maximum training epochs (early stopping may end earlier)
        lr (float): Learning rate
        patience (int): Early stopping patience

    Returns:
        dict: {
            "train_losses": list of training losses,
            "val_losses": list of validation losses,
            "stop_epoch": epoch at which training stopped (or max_epochs if not stopped),
            "best_epoch": epoch with best validation loss
        }

    Hints:
        - Convert data to tensors at the start
        - Initialize: criterion = nn.BCEWithLogitsLoss()
        - Initialize: optimizer = optim.Adam(model.parameters(), lr=lr)
        - Initialize: early_stopping = EarlyStopping(patience=patience, min_delta=1e-4)
        - In each epoch:
            1. model.train() → forward → backward → optimizer.step()
            2. model.eval() → with torch.no_grad(): compute val_loss
            3. early_stopping(val_loss.item(), model)
            4. if early_stopping.should_stop: break
        - After training: print message with stop epoch and best epoch
        - Return the results dict
    """
    # TODO: Convert X_train, y_train, X_val, y_val to torch.FloatTensor
    # TODO: Initialize criterion, optimizer, early_stopping
    # TODO: Initialize train_losses and val_losses lists
    # TODO: Training loop with early stopping check
    # TODO: Return results dict with train_losses, val_losses, stop_epoch, best_epoch
    pass


# ---------------------------------------------------------------------------
# Visualization (Complete — Do Not Modify)
# ---------------------------------------------------------------------------

def plot_early_stopping_demo(train_losses, val_losses, stop_epoch, best_epoch):
    """
    Plot training and validation loss curves with early stopping annotations.

    Args:
        train_losses (list): Training loss per epoch
        val_losses (list): Validation loss per epoch
        stop_epoch (int): Epoch where training stopped
        best_epoch (int): Epoch with best validation loss
    """
    epochs = range(len(train_losses))

    plt.figure(figsize=(12, 5))
    plt.plot(epochs, train_losses, color='blue', linewidth=2, label='Training Loss')
    plt.plot(epochs, val_losses, color='red', linewidth=2,
             linestyle='--', label='Validation Loss')

    # Mark the best epoch
    plt.axvline(x=best_epoch, color='green', linestyle='--', linewidth=2,
                label=f'Best Epoch: {best_epoch} (val={val_losses[best_epoch]:.4f})')
    plt.plot(best_epoch, val_losses[best_epoch], 'g*', markersize=15)

    # Mark the stop epoch (if early stopping triggered)
    if stop_epoch < len(train_losses) - 1:
        plt.axvline(x=stop_epoch, color='orange', linestyle=':', linewidth=2,
                    label=f'Stop Epoch: {stop_epoch}')
        plt.axvspan(best_epoch, stop_epoch, alpha=0.1, color='yellow', label='Patience Window')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Early Stopping Demonstration\n"
              f"Training stopped at epoch {stop_epoch}, best weights from epoch {best_epoch}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Summary statistics
    print(f"\n--- Early Stopping Summary ---")
    print(f"Epochs trained:       {stop_epoch + 1}")
    print(f"Best epoch:           {best_epoch}")
    print(f"Best val loss:        {val_losses[best_epoch]:.4f}")
    print(f"Final train loss:     {train_losses[stop_epoch]:.4f}")
    print(f"Train-val gap (best): {val_losses[best_epoch] - train_losses[best_epoch]:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    Demonstrate early stopping on a model that would severely overfit without it.

    Steps:
    1. Create a small dataset (300 training, 200 validation samples)
    2. Train OverfittingModel WITHOUT early stopping (max epochs)
    3. Train OverfittingModel WITH early stopping
    4. Compare: how many epochs were trained, what is the val loss
    5. Plot both training curves with early stopping marker
    """
    print("=" * 60)
    print("Early Stopping Demonstration")
    print("=" * 60)
    torch.manual_seed(42)
    np.random.seed(42)

    # Create small dataset to induce overfitting
    X_raw, y = make_classification(
        n_samples=500, n_features=20, n_informative=8,
        n_redundant=6, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_raw, y, test_size=0.4, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    # -----------------------------------------------------------------------
    # Without Early Stopping: Train for all 300 epochs
    # -----------------------------------------------------------------------
    print("\n--- Training WITHOUT Early Stopping (300 epochs) ---")
    # TODO: Initialize OverfittingModel(input_dim=X_train.shape[1])
    # TODO: Train WITHOUT early stopping (use max_epochs=300, large patience)
    # OR manually write a training loop without early stopping
    # results_no_es = train_with_early_stopping(model, ..., max_epochs=300, patience=999)

    # -----------------------------------------------------------------------
    # With Early Stopping
    # -----------------------------------------------------------------------
    print("\n--- Training WITH Early Stopping (patience=15) ---")
    # TODO: Initialize a fresh OverfittingModel
    # TODO: Call train_with_early_stopping with patience=15
    # results_es = train_with_early_stopping(model_es, X_train, y_train, X_val, y_val,
    #                                         max_epochs=300, patience=15)

    # -----------------------------------------------------------------------
    # Visualization and Comparison
    # -----------------------------------------------------------------------
    # TODO: Plot early stopping results
    # plot_early_stopping_demo(
    #     results_es["train_losses"],
    #     results_es["val_losses"],
    #     results_es["stop_epoch"],
    #     results_es["best_epoch"]
    # )

    # TODO: Compare final validation loss: with vs without early stopping
    print("\nExpected: With early stopping, the best val loss is lower and")
    print("training stops well before epoch 300 (while val loss is still good).")


if __name__ == "__main__":
    main()
