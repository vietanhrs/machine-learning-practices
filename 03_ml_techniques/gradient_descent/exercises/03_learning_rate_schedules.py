"""
Exercise 03: Learning Rate Schedules
======================================
Implement common learning rate schedules and compare their effect on
training convergence. Visualize the schedule curves and training loss curves
to understand how each schedule affects optimization.

Learning objectives:
- Understand why fixed learning rates are suboptimal
- Implement step decay, exponential decay, cosine annealing, and warm-up
- Compare training curves across different schedules

Dependencies: numpy, matplotlib, scikit-learn
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Utility: Sigmoid and Logistic Regression Loss
# ---------------------------------------------------------------------------

def sigmoid(z):
    """Numerically stable sigmoid."""
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))


def logistic_loss_and_grad(params, X, y):
    """Compute binary cross-entropy loss and gradient."""
    probs = sigmoid(X @ params)
    eps = 1e-15
    loss = -np.mean(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps))
    grad = X.T @ (probs - y) / len(y)
    return loss, grad


# ---------------------------------------------------------------------------
# Learning Rate Schedules
# ---------------------------------------------------------------------------

def step_decay(initial_lr, epoch, drop=0.5, epochs_drop=10):
    """
    Step Decay: multiply learning rate by `drop` every `epochs_drop` epochs.

    Formula:
        lr(epoch) = initial_lr × drop^(floor(epoch / epochs_drop))

    Args:
        initial_lr (float): Starting learning rate
        epoch (int): Current epoch (0-indexed)
        drop (float): Multiplicative factor applied every epochs_drop (default 0.5 = halve)
        epochs_drop (int): How often to apply the drop (default 10)

    Returns:
        float: Learning rate for the current epoch

    Hints:
        - Use np.floor(epoch / epochs_drop) to get how many drops have occurred
        - Multiply initial_lr by drop raised to the floor value
        - Example: epoch=25, drop=0.5, epochs_drop=10 → lr = initial_lr × 0.5² = initial_lr × 0.25
    """
    # TODO: Compute number of drops applied so far (floor division)
    # TODO: Return initial_lr × drop^(n_drops)
    pass


def exponential_decay(initial_lr, epoch, decay_rate=0.95):
    """
    Exponential Decay: continuously reduce learning rate by an exponential factor.

    Formula:
        lr(epoch) = initial_lr × decay_rate^epoch

    Args:
        initial_lr (float): Starting learning rate
        epoch (int): Current epoch (0-indexed)
        decay_rate (float): Multiplicative decay per epoch (default 0.95)

    Returns:
        float: Learning rate for the current epoch

    Hints:
        - Return initial_lr × decay_rate^epoch
        - With decay_rate=0.95: after 20 epochs, lr ≈ initial_lr × 0.36
        - Smoother than step decay — no sudden drops
    """
    # TODO: Return initial_lr × (decay_rate ** epoch)
    pass


def cosine_annealing(initial_lr, epoch, T_max, lr_min=0.0):
    """
    Cosine Annealing: smoothly reduce learning rate following a cosine curve.

    Formula:
        lr(epoch) = lr_min + 0.5 × (initial_lr − lr_min) × (1 + cos(π × epoch / T_max))

    Args:
        initial_lr (float): Maximum (starting) learning rate
        epoch (int): Current epoch (0-indexed)
        T_max (int): Half-period of the cosine schedule (total epochs to reach lr_min)
        lr_min (float): Minimum learning rate (default 0.0)

    Returns:
        float: Learning rate for the current epoch

    Hints:
        - Use np.cos(np.pi * epoch / T_max)
        - At epoch=0:     lr = initial_lr   (cos(0) = 1)
        - At epoch=T_max: lr = lr_min       (cos(π) = -1)
        - Smooth and often achieves better final accuracy than step decay
    """
    # TODO: Implement cosine annealing formula
    # TODO: Return lr_min + 0.5 × (initial_lr - lr_min) × (1 + cos(π × epoch / T_max))
    pass


def warmup_cosine(initial_lr, epoch, warmup_epochs, total_epochs, lr_min=0.0):
    """
    Linear Warm-Up followed by Cosine Annealing.

    During warm-up (epoch < warmup_epochs):
        lr = initial_lr × (epoch / warmup_epochs)    [linear increase from 0 to initial_lr]

    After warm-up (epoch >= warmup_epochs):
        lr = cosine_annealing applied to remaining epochs

    Args:
        initial_lr (float): Target learning rate after warm-up
        epoch (int): Current epoch (0-indexed)
        warmup_epochs (int): Number of epochs for linear warm-up
        total_epochs (int): Total training epochs
        lr_min (float): Minimum learning rate after cosine annealing

    Returns:
        float: Learning rate for the current epoch

    Hints:
        - If epoch < warmup_epochs: return initial_lr * (epoch / warmup_epochs)
          (Special case: if warmup_epochs == 0, skip warm-up)
        - Else: apply cosine_annealing with adjusted epoch and T_max
          adjusted_epoch = epoch - warmup_epochs
          T_max = total_epochs - warmup_epochs
          return cosine_annealing(initial_lr, adjusted_epoch, T_max, lr_min)
        - Note: at epoch=0 during warm-up, lr=0 (or very small)
    """
    # TODO: Handle warm-up phase (linear increase)
    # TODO: Handle post-warm-up phase (cosine annealing)
    pass


# ---------------------------------------------------------------------------
# Visualization (Complete — Do Not Modify)
# ---------------------------------------------------------------------------

def plot_schedules(schedules_dict, n_epochs=100, initial_lr=0.1):
    """
    Plot multiple learning rate schedules over n_epochs.

    Args:
        schedules_dict (dict): {name: schedule_fn} where schedule_fn(epoch) -> lr
        n_epochs (int): Number of epochs to plot
        initial_lr (float): Initial learning rate for reference line
    """
    epochs = np.arange(n_epochs)
    fig, ax = plt.subplots(figsize=(12, 5))

    for name, fn in schedules_dict.items():
        lrs = [fn(e) for e in epochs]
        ax.plot(epochs, lrs, linewidth=2, label=name, marker='')

    ax.axhline(y=initial_lr, color='gray', linestyle='--', alpha=0.5, label=f"Fixed LR={initial_lr}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedules Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Training with Schedule
# ---------------------------------------------------------------------------

def train_with_schedule(X, y, schedule_fn, n_epochs=100, batch_size=32):
    """
    Train logistic regression using mini-batch SGD with a learning rate schedule.

    Args:
        X (np.ndarray): Feature matrix with bias column
        y (np.ndarray): Binary labels
        schedule_fn (callable): Function that maps epoch → learning rate
        n_epochs (int): Number of training epochs
        batch_size (int): Mini-batch size

    Returns:
        list: Loss values recorded at the end of each epoch

    Hints:
        - Initialize params with np.random.randn(X.shape[1]) × 0.01
        - In each epoch:
            1. Get current lr: lr = schedule_fn(epoch)
            2. Shuffle data: indices = np.random.permutation(len(y))
            3. Loop over mini-batches:
                X_batch = X[indices[start:end]]
                y_batch = y[indices[start:end]]
                loss, grad = logistic_loss_and_grad(params, X_batch, y_batch)
                params = params - lr * grad
            4. Record epoch loss (full dataset loss for monitoring)
        - Return list of epoch losses
    """
    # TODO: Initialize parameters
    # TODO: Initialize loss history list
    # TODO: For each epoch:
    #   - Get current learning rate from schedule_fn(epoch)
    #   - Shuffle training data
    #   - Iterate over mini-batches
    #   - Compute gradient and update parameters
    #   - Compute and record epoch loss on full dataset
    # TODO: Return loss history
    pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    Compare training curves for different learning rate schedules.

    Steps:
    1. Generate binary classification data
    2. Define schedules: fixed, step decay, exponential, cosine, warmup+cosine
    3. Plot all schedule curves
    4. Train logistic regression with each schedule
    5. Plot training loss curves side-by-side
    """
    print("=" * 60)
    print("Learning Rate Schedules Comparison")
    print("=" * 60)

    # Generate data
    np.random.seed(42)
    X_raw, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10,
        n_redundant=5, random_state=42
    )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X = np.column_stack([np.ones(len(y)), X_scaled])

    initial_lr = 0.1
    n_epochs = 100

    # TODO: Define schedule functions (use lambda to bind initial_lr and n_epochs)
    # Example:
    # schedules = {
    #     "Fixed LR":          lambda e: initial_lr,
    #     "Step Decay":        lambda e: step_decay(initial_lr, e, drop=0.5, epochs_drop=20),
    #     "Exponential Decay": lambda e: exponential_decay(initial_lr, e, decay_rate=0.97),
    #     "Cosine Annealing":  lambda e: cosine_annealing(initial_lr, e, T_max=n_epochs),
    #     "Warmup + Cosine":   lambda e: warmup_cosine(initial_lr, e, warmup_epochs=10, total_epochs=n_epochs),
    # }

    # TODO: Plot schedule curves
    # Hint: plot_schedules(schedules, n_epochs=n_epochs, initial_lr=initial_lr)

    # TODO: Train with each schedule and record loss histories
    # Hint:
    # loss_histories = {}
    # for name, schedule_fn in schedules.items():
    #     print(f"Training with {name}...")
    #     history = train_with_schedule(X, y, schedule_fn, n_epochs=n_epochs)
    #     loss_histories[name] = history

    # TODO: Plot training loss curves
    # Hint:
    # plt.figure(figsize=(12, 5))
    # for name, history in loss_histories.items():
    #     plt.plot(history, label=name)
    # plt.xlabel("Epoch")
    # plt.ylabel("Training Loss")
    # plt.title("Training Loss with Different LR Schedules")
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.show()

    print("\nDone. Compare how quickly each schedule converges and final loss achieved.")


if __name__ == "__main__":
    main()
