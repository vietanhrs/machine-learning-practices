"""
Exercise 04: MLP with PyTorch
==============================
Rebuild the MLP using PyTorch's nn.Module API.
PyTorch handles automatic differentiation (autograd) — you focus on
architecture design, the training loop structure, and best practices.

Learning goals:
- Use nn.Sequential to build MLPs declaratively.
- Write a proper PyTorch training loop (optimizer, loss, zero_grad, backward, step).
- Implement train/validation split, evaluation metrics, and early stopping.
- Plot training curves.

References:
    - PyTorch nn.Sequential: https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
    - PyTorch Datasets & DataLoaders: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Model Builder
# ---------------------------------------------------------------------------

def build_mlp(input_size, hidden_sizes, output_size, activation='relu'):
    """
    Build a fully-connected MLP using nn.Sequential.

    Architecture:
        Linear(input_size, hidden_sizes[0]) → Activation
        Linear(hidden_sizes[0], hidden_sizes[1]) → Activation
        ...
        Linear(hidden_sizes[-1], output_size)   ← no activation (logits)

    Args:
        input_size (int): Number of input features.
        hidden_sizes (list[int]): Number of neurons in each hidden layer.
        output_size (int): Number of output neurons.
        activation (str): 'relu', 'tanh', or 'sigmoid'.

    Returns:
        nn.Sequential: The constructed MLP model.

    TODO:
        1. Create a list of layers (Python list).
        2. Add the first linear layer: nn.Linear(input_size, hidden_sizes[0]).
        3. Add the chosen activation function.
        4. For each subsequent hidden layer, add nn.Linear + activation.
        5. Add the final output layer: nn.Linear(hidden_sizes[-1], output_size).
           (No activation — use raw logits with CrossEntropyLoss or BCEWithLogitsLoss.)
        6. Return nn.Sequential(*layers).

    Hint:
        activation_fn = {'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid()}[activation]
        Be careful to use separate activation instances or re-instantiate for each layer.
    """
    # TODO: implement build_mlp
    raise NotImplementedError("Implement build_mlp(...)")


# ---------------------------------------------------------------------------
# Training and Evaluation
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion):
    """
    Run one full training epoch over the DataLoader.

    Steps for each mini-batch:
        1. Move data to the correct device.
        2. Zero out accumulated gradients: optimizer.zero_grad()
        3. Forward pass: outputs = model(X_batch)
        4. Compute loss: loss = criterion(outputs, y_batch)
        5. Backward pass: loss.backward()
        6. Update weights: optimizer.step()
        7. Accumulate total loss and correct predictions.

    Args:
        model (nn.Module): The MLP model.
        loader (DataLoader): Training data loader.
        optimizer: PyTorch optimizer (e.g., Adam).
        criterion: Loss function (e.g., CrossEntropyLoss).

    Returns:
        tuple[float, float]: (average_loss, accuracy) for the epoch.

    TODO:
        Implement the training loop described above.
        Hint:
            - model.train() before the loop.
            - For multi-class: predicted = outputs.argmax(dim=1)
            - Accumulate: total_loss += loss.item() * len(X_batch)
                          correct    += (predicted == y_batch).sum().item()
            - Divide by total samples at the end.
    """
    # TODO: implement train_epoch
    raise NotImplementedError("Implement train_epoch(...)")


def evaluate(model, loader, criterion):
    """
    Evaluate the model on a DataLoader (validation or test set).

    Similar to train_epoch but:
    - No gradient computation (use torch.no_grad()).
    - No optimizer step.
    - Model in eval mode: model.eval()

    Args:
        model (nn.Module): The MLP model.
        loader (DataLoader): Validation/test data loader.
        criterion: Loss function.

    Returns:
        tuple[float, float]: (average_loss, accuracy).

    TODO:
        Implement evaluation loop inside a torch.no_grad() context.
        Pattern:
            model.eval()
            with torch.no_grad():
                for X_batch, y_batch in loader:
                    ...
    """
    # TODO: implement evaluate
    raise NotImplementedError("Implement evaluate(...)")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    """
    Plot training and validation loss and accuracy curves side by side.
    This function is complete — no changes needed.

    Args:
        train_losses (list[float]): Training loss per epoch.
        val_losses (list[float]): Validation loss per epoch.
        train_accs (list[float]): Training accuracy per epoch.
        val_accs (list[float]): Validation accuracy per epoch.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, label="Train Loss", color="steelblue")
    ax1.plot(epochs, val_losses, label="Val Loss", color="darkorange", linestyle='--')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accs, label="Train Acc", color="steelblue")
    ax2.plot(epochs, val_accs, label="Val Acc", color="darkorange", linestyle='--')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curves")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("PyTorch MLP Training Curves", fontsize=13)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Dataset: make_classification (sklearn) → PyTorch TensorDataset
    # ------------------------------------------------------------------
    X_np, y_np = make_classification(
        n_samples=2000, n_features=20, n_informative=10,
        n_classes=3, n_clusters_per_class=1, random_state=42
    )
    scaler = StandardScaler()
    X_np = scaler.fit_transform(X_np).astype(np.float32)
    y_np = y_np.astype(np.int64)

    X_tensor = torch.from_numpy(X_np)
    y_tensor = torch.from_numpy(y_np)

    dataset = TensorDataset(X_tensor, y_tensor)

    # 70 / 15 / 15 split
    n_total = len(dataset)
    n_train = int(0.70 * n_total)
    n_val   = int(0.15 * n_total)
    n_test  = n_total - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)

    print(f"Dataset: {n_train} train / {n_val} val / {n_test} test samples")

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    model = build_mlp(
        input_size=20,
        hidden_sizes=[128, 64, 32],
        output_size=3,        # 3 classes
        activation='relu'
    ).to(device)

    print(f"\nModel architecture:\n{model}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # Training loop with early stopping stub
    # ------------------------------------------------------------------
    n_epochs   = 100
    patience   = 10  # early stopping patience

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_state = None

    for epoch in range(1, n_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}")

        # ------------------------------------------------------------------
        # Early Stopping Logic
        # TODO:
        #   1. If val_loss improved (is less than best_val_loss):
        #       - Update best_val_loss.
        #       - Save model state: best_state = {k: v.clone() for k, v in model.state_dict().items()}
        #       - Reset epochs_no_improve to 0.
        #   2. Else:
        #       - Increment epochs_no_improve.
        #       - If epochs_no_improve >= patience: print a message and break.
        # ------------------------------------------------------------------
        # TODO: implement early stopping
        pass  # remove this line when implementing

    # Restore best weights if early stopping triggered
    if best_state is not None:
        model.load_state_dict(best_state)
        print("\nRestored best model weights.")

    # ------------------------------------------------------------------
    # Final evaluation on test set
    # ------------------------------------------------------------------
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.3f}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)


if __name__ == "__main__":
    main()
