"""
Exercise 03: Time Series Forecasting with LSTM
===============================================
Build an LSTM model to forecast future values of a time series using
a sliding window approach. Evaluate with MAE and RMSE.

Learning goals:
- Create overlapping input-output windows from sequential data.
- Train an LSTM as a regression model (output: continuous values).
- Evaluate and visualize time series predictions.

References:
    - PyTorch LSTM: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------------

def create_sequences(data, seq_len):
    """
    Create sliding window (input, target) pairs from a 1D time series.

    For a series [x_0, x_1, x_2, ..., x_N]:
    - Input:  [x_t, x_{t+1}, ..., x_{t+seq_len-1}]   (seq_len values)
    - Target: x_{t+seq_len}                            (one step ahead)

    Args:
        data (np.ndarray): 1D time series, shape (N,).
        seq_len (int): Length of input window.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - X: Input sequences, shape (N - seq_len, seq_len, 1).
            - y: Target values,   shape (N - seq_len, 1).

    TODO:
        1. Initialize empty lists X_list and y_list.
        2. Loop i from 0 to len(data) - seq_len - 1:
               X_list.append(data[i : i + seq_len])
               y_list.append(data[i + seq_len])
        3. Convert to numpy arrays.
        4. Reshape X to (n_samples, seq_len, 1)  ← LSTM expects (batch, seq, features).
        5. Reshape y to (n_samples, 1).
        6. Return X, y.
    """
    # TODO: implement create_sequences
    raise NotImplementedError("Implement create_sequences(data, seq_len)")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TimeSeriesLSTM(nn.Module):
    """
    LSTM model for single-step time series forecasting.

    Architecture:
        LSTM (input_size=1, hidden=hidden_size, n_layers)
        → last hidden state
        → Linear(hidden_size, 1)  ← predict one future value

    Args:
        hidden_size (int): LSTM hidden state size.
        n_layers (int): Number of stacked LSTM layers.
        dropout (float): Dropout between LSTM layers (if n_layers > 1).
    """

    def __init__(self, hidden_size=64, n_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers    = n_layers

        # TODO:
        #   Define self.lstm = nn.LSTM(
        #       input_size=1,
        #       hidden_size=hidden_size,
        #       num_layers=n_layers,
        #       dropout=dropout if n_layers > 1 else 0,
        #       batch_first=True
        #   )
        raise NotImplementedError("Define self.lstm in TimeSeriesLSTM.__init__")

        # TODO:
        #   Define self.fc = nn.Linear(hidden_size, 1)
        raise NotImplementedError("Define self.fc in TimeSeriesLSTM.__init__")

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input sequences, shape (batch, seq_len, 1).

        Returns:
            torch.Tensor: Forecasted values, shape (batch, 1).

        TODO:
            1. Pass x through self.lstm to get (output, (h_n, c_n)).
            2. Extract the last hidden state of the last layer:
                   h_last = h_n[-1]   # shape: (batch, hidden_size)
            3. Pass through self.fc to get prediction, shape (batch, 1).
            4. Return predictions.
        """
        # TODO: implement forward pass
        raise NotImplementedError("Implement TimeSeriesLSTM.forward(x)")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_forecaster(model, X_train, y_train, n_epochs, lr, batch_size=32,
                     device=None):
    """
    Train the LSTM forecaster using MSE loss.

    Args:
        model (nn.Module): TimeSeriesLSTM.
        X_train (torch.Tensor): Training inputs, shape (n, seq_len, 1).
        y_train (torch.Tensor): Training targets, shape (n, 1).
        n_epochs (int): Number of epochs.
        lr (float): Learning rate.
        batch_size (int): Mini-batch size.
        device: torch.device.

    Returns:
        list[float]: Training loss per epoch.

    TODO:
        1. Create a DataLoader from TensorDataset(X_train, y_train).
        2. Define criterion = nn.MSELoss().
        3. Define optimizer = optim.Adam(model.parameters(), lr=lr).
        4. For each epoch:
               a. model.train()
               b. Iterate DataLoader: zero_grad, forward, loss, backward, step.
               c. Track epoch average loss.
               d. Print every 10 epochs.
        5. Return loss history.
    """
    # TODO: implement train_forecaster
    raise NotImplementedError("Implement train_forecaster(...)")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_forecast(model, X_test, y_test, device):
    """
    Evaluate the forecaster on test data.

    Compute:
        MAE  = mean(|y_true - y_pred|)
        RMSE = sqrt(mean((y_true - y_pred)^2))

    Args:
        model (nn.Module): Trained TimeSeriesLSTM.
        X_test (torch.Tensor): Test inputs, shape (n, seq_len, 1).
        y_test (torch.Tensor): True future values, shape (n, 1).
        device: torch.device.

    Returns:
        tuple[float, float, np.ndarray]: (MAE, RMSE, predictions_array)

    TODO:
        1. model.eval(), torch.no_grad().
        2. y_pred = model(X_test.to(device)).cpu().numpy()
        3. y_true = y_test.numpy()
        4. Compute MAE and RMSE.
        5. Print results.
        6. Return (MAE, RMSE, y_pred.flatten()).
    """
    # TODO: implement evaluate_forecast
    raise NotImplementedError("Implement evaluate_forecast(...)")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_predictions(y_true, y_pred, title="Time Series Forecast"):
    """
    Plot the true vs predicted time series.
    This function is complete — no changes needed.

    Args:
        y_true (np.ndarray): Ground truth values, shape (n,).
        y_pred (np.ndarray): Model predictions, shape (n,).
        title (str): Plot title.
    """
    plt.figure(figsize=(12, 5))
    time = np.arange(len(y_true))

    plt.plot(time, y_true, label="True", color="steelblue", linewidth=1.5)
    plt.plot(time, y_pred, label="Predicted", color="darkorange",
             linewidth=1.5, linestyle="--")
    plt.fill_between(time,
                     y_true - np.abs(y_true - y_pred),
                     y_true + np.abs(y_true - y_pred),
                     alpha=0.15, color="darkorange", label="Error band")

    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
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
    # Generate synthetic time series: noisy sinusoidal + trend
    # ------------------------------------------------------------------
    N = 1000
    t = np.linspace(0, 8 * np.pi, N)

    # Signal: multiple frequencies + trend + noise
    signal = (np.sin(t)
              + 0.5 * np.sin(2.3 * t + 1.0)
              + 0.2 * np.cos(4.1 * t)
              + 0.05 * t            # slow upward trend
              + np.random.randn(N) * 0.1)

    # Normalize to [-1, 1]
    signal = (signal - signal.mean()) / signal.std()

    plt.figure(figsize=(12, 3))
    plt.plot(t, signal, linewidth=0.8, color='steelblue')
    plt.title("Synthetic Time Series (Noisy Sinusoid + Trend)")
    plt.xlabel("t")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # Create sliding window sequences
    # ------------------------------------------------------------------
    SEQ_LEN = 30

    X, y = create_sequences(signal, seq_len=SEQ_LEN)
    print(f"Sequences: X shape = {X.shape}, y shape = {y.shape}")

    # Train/test split (80/20 — no shuffle for time series!)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
    y_test_t  = torch.tensor(y_test,  dtype=torch.float32)

    print(f"Train: {X_train_t.shape}, Test: {X_test_t.shape}")

    # ------------------------------------------------------------------
    # Build and train model
    # ------------------------------------------------------------------
    model = TimeSeriesLSTM(hidden_size=64, n_layers=2, dropout=0.2).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    loss_history = train_forecaster(
        model, X_train_t, y_train_t,
        n_epochs=50, lr=1e-3, batch_size=64, device=device
    )

    # Plot training loss
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history, color='steelblue')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('LSTM Forecaster Training Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    mae, rmse, y_pred = evaluate_forecast(model, X_test_t, y_test_t, device)
    print(f"\nTest MAE:  {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")

    # ------------------------------------------------------------------
    # Visualize predictions (first 200 test steps)
    # ------------------------------------------------------------------
    n_show = min(200, len(y_pred))
    plot_predictions(
        y_test[:n_show, 0],
        y_pred[:n_show],
        title=f"LSTM Time Series Forecast (MAE={mae:.4f}, RMSE={rmse:.4f})"
    )

    # ------------------------------------------------------------------
    # Ablation: compare different sequence lengths
    # ------------------------------------------------------------------
    print("\nAblation: effect of sequence length on RMSE...")
    for seq in [5, 15, 30, 50]:
        X_abl, y_abl = create_sequences(signal, seq_len=seq)
        sp   = int(0.8 * len(X_abl))
        Xtr  = torch.tensor(X_abl[:sp], dtype=torch.float32)
        ytr  = torch.tensor(y_abl[:sp], dtype=torch.float32)
        Xte  = torch.tensor(X_abl[sp:], dtype=torch.float32)
        yte  = torch.tensor(y_abl[sp:], dtype=torch.float32)

        m_abl = TimeSeriesLSTM(hidden_size=32, n_layers=1).to(device)
        train_forecaster(m_abl, Xtr, ytr, n_epochs=20, lr=1e-3, device=device)
        _, rmse_abl, _ = evaluate_forecast(m_abl, Xte, yte, device)
        print(f"  seq_len={seq:3d} → RMSE = {rmse_abl:.4f}")


if __name__ == "__main__":
    main()
