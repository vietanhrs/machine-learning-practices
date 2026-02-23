"""
Exercise 02: Custom Loss Functions in PyTorch
==============================================
Implement custom loss functions as PyTorch nn.Module subclasses.
This exercise covers losses for imbalanced classification and metric learning.

Learning objectives:
- Extend nn.Module to create custom loss functions
- Implement weighted cross-entropy for class imbalance
- Implement contrastive and triplet losses for metric learning
- Use custom losses in a training loop

Dependencies: torch, numpy, matplotlib, scikit-learn
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Custom Loss Modules
# ---------------------------------------------------------------------------

class WeightedCrossEntropyLoss(nn.Module):
    """
    Binary Cross-Entropy with per-class weights to handle class imbalance.

    When positive examples are rare (e.g., 5% of data), standard BCE
    can be minimized by always predicting negative. Weighting the positive
    class more heavily forces the model to learn the minority class.

    Args:
        pos_weight (float): Weight multiplier for positive class (y=1).
                           Set to (n_negative / n_positive) for balanced training.

    Usage:
        loss_fn = WeightedCrossEntropyLoss(pos_weight=10.0)
        loss = loss_fn(logits, labels)
    """

    def __init__(self, pos_weight=1.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        """
        Compute weighted binary cross-entropy.

        Args:
            logits (torch.Tensor): Raw model outputs (before sigmoid), shape (N,)
            targets (torch.Tensor): Binary labels {0, 1}, shape (N,)

        Returns:
            torch.Tensor: Scalar loss value

        Hints:
            - Use nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.pos_weight]))
              This combines sigmoid + BCE in a numerically stable way.
            - OR manually: sigmoid(logits), then apply weighted BCE formula:
              loss = -(pos_weight * targets * log(probs) + (1-targets) * log(1-probs))
            - Using BCEWithLogitsLoss is recommended for numerical stability.
        """
        # TODO: Create pos_weight tensor: torch.tensor([self.pos_weight])
        # TODO: Initialize nn.BCEWithLogitsLoss(pos_weight=...) and compute loss
        # TODO: Return the computed loss
        pass


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for Siamese Networks (LeCun et al., 2005).

    Trains an embedding space where:
    - Similar pairs (label=1) have small distance
    - Dissimilar pairs (label=0) have distance > margin

    Formula:
        L = label * d² + (1 - label) * max(margin - d, 0)²
    Where d = ||embedding1 - embedding2||₂

    Args:
        margin (float): Minimum required distance for dissimilar pairs (default 1.0)

    Usage:
        loss_fn = ContrastiveLoss(margin=1.0)
        loss = loss_fn(emb1, emb2, labels)
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embedding1, embedding2, labels):
        """
        Compute contrastive loss for a batch of pairs.

        Args:
            embedding1 (torch.Tensor): First embeddings, shape (N, D)
            embedding2 (torch.Tensor): Second embeddings, shape (N, D)
            labels (torch.Tensor): Pair labels — 1=similar, 0=dissimilar, shape (N,)

        Returns:
            torch.Tensor: Scalar mean contrastive loss

        Hints:
            - Compute Euclidean distance: d = torch.norm(embedding1 - embedding2, dim=1)
            - Positive term: labels * d ** 2
            - Negative term: (1 - labels) * torch.clamp(self.margin - d, min=0) ** 2
            - Loss per pair: positive_term + negative_term
            - Return torch.mean(loss_per_pair)
        """
        # TODO: Compute L2 distance between embedding pairs
        # TODO: Compute positive pair loss term (similar pairs attracted)
        # TODO: Compute negative pair loss term (dissimilar pairs repelled)
        # TODO: Sum terms and return mean loss
        pass


class TripletLoss(nn.Module):
    """
    Triplet Loss for metric learning (Schroff et al., FaceNet 2015).

    Trains embeddings so that:
        d(anchor, positive) + margin < d(anchor, negative)

    Formula:
        L = max(d(a, p) - d(a, n) + margin, 0)
    Where d is Euclidean distance.

    Args:
        margin (float): Required gap between positive and negative distances (default 0.5)

    Usage:
        loss_fn = TripletLoss(margin=0.5)
        loss = loss_fn(anchor, positive, negative)
    """

    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """
        Compute triplet loss for a batch of (anchor, positive, negative) triplets.

        Args:
            anchor (torch.Tensor): Anchor embeddings, shape (N, D)
            positive (torch.Tensor): Positive embeddings (same class as anchor), shape (N, D)
            negative (torch.Tensor): Negative embeddings (different class), shape (N, D)

        Returns:
            torch.Tensor: Scalar mean triplet loss

        Hints:
            - d_pos = torch.norm(anchor - positive, dim=1)   (anchor-positive distance)
            - d_neg = torch.norm(anchor - negative, dim=1)   (anchor-negative distance)
            - triplet_loss = torch.clamp(d_pos - d_neg + self.margin, min=0)
            - Return torch.mean(triplet_loss)
            - A loss of 0 means all triplets satisfy: d(a,p) + margin ≤ d(a,n)
        """
        # TODO: Compute anchor-positive distances
        # TODO: Compute anchor-negative distances
        # TODO: Compute triplet loss with margin, clamp at 0
        # TODO: Return mean loss
        pass


# ---------------------------------------------------------------------------
# Simple Classifier Network
# ---------------------------------------------------------------------------

class SimpleClassifier(nn.Module):
    """A simple 2-layer MLP for binary classification."""

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)  # Output logits, shape (N,)


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_with_custom_loss(X, y, loss_fn, n_epochs=50, lr=0.01, verbose=True):
    """
    Train a simple classifier using a custom loss function.

    Args:
        X (np.ndarray): Feature matrix, shape (N, D)
        y (np.ndarray): Binary labels {0, 1}, shape (N,)
        loss_fn (nn.Module): A custom loss function instance
        n_epochs (int): Number of training epochs
        lr (float): Learning rate for Adam optimizer
        verbose (bool): Print loss every 10 epochs

    Returns:
        tuple: (trained_model, loss_history)

    Hints:
        - Convert numpy arrays to torch tensors:
          X_tensor = torch.FloatTensor(X)
          y_tensor = torch.FloatTensor(y)
        - Initialize SimpleClassifier(input_dim=X.shape[1])
        - Use optim.Adam(model.parameters(), lr=lr)
        - Training loop:
            optimizer.zero_grad()
            logits = model(X_tensor)
            loss = loss_fn(logits, y_tensor)
            loss.backward()
            optimizer.step()
        - Append loss.item() to history each epoch
        - Return (model, loss_history)
    """
    # TODO: Convert data to PyTorch tensors
    # TODO: Initialize model and optimizer
    # TODO: Training loop over n_epochs
    # TODO: Return (model, loss_history)
    pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    Demonstrate each custom loss function.

    Part 1: WeightedCrossEntropyLoss on imbalanced classification
    Part 2: ContrastiveLoss on a synthetic pairs dataset
    Part 3: TripletLoss on a synthetic triplets dataset
    """
    print("=" * 60)
    print("Custom Loss Functions in PyTorch")
    print("=" * 60)
    torch.manual_seed(42)
    np.random.seed(42)

    # -----------------------------------------------------------------------
    # Part 1: Weighted Cross-Entropy for Class Imbalance
    # -----------------------------------------------------------------------
    print("\n--- Part 1: Weighted Cross-Entropy ---")

    # Create imbalanced dataset (5% positive, 95% negative)
    X_raw, y = make_classification(
        n_samples=1000, n_features=10, weights=[0.95, 0.05],
        n_informative=5, random_state=42
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    pos_weight = n_neg / n_pos
    print(f"Class distribution: {n_neg} negative, {n_pos} positive. pos_weight={pos_weight:.1f}")

    # TODO: Train with standard BCE
    # standard_bce = nn.BCEWithLogitsLoss()
    # model_standard, history_standard = train_with_custom_loss(X, y, standard_bce, n_epochs=100)

    # TODO: Train with weighted BCE
    # weighted_bce = WeightedCrossEntropyLoss(pos_weight=pos_weight)
    # model_weighted, history_weighted = train_with_custom_loss(X, y, weighted_bce, n_epochs=100)

    # TODO: Plot and compare loss curves

    # -----------------------------------------------------------------------
    # Part 2: Contrastive Loss for Metric Learning
    # -----------------------------------------------------------------------
    print("\n--- Part 2: Contrastive Loss ---")

    # TODO: Create synthetic pair data
    # Hint: Generate embeddings for "anchor" and "comparison" pairs
    # Randomly label pairs as similar (1) or dissimilar (0)
    # Batch shape: (N, D) for each of emb1, emb2; (N,) for labels
    # N = 100, D = 8

    # TODO: Instantiate ContrastiveLoss and compute loss on sample data
    # contrastive_loss = ContrastiveLoss(margin=1.0)
    # emb1 = torch.randn(100, 8)
    # emb2 = torch.randn(100, 8)
    # pair_labels = torch.randint(0, 2, (100,)).float()
    # loss = contrastive_loss(emb1, emb2, pair_labels)
    # print(f"Contrastive loss on random embeddings: {loss.item():.4f}")

    # -----------------------------------------------------------------------
    # Part 3: Triplet Loss
    # -----------------------------------------------------------------------
    print("\n--- Part 3: Triplet Loss ---")

    # TODO: Create synthetic triplet data
    # Hint: anchor = random embeddings
    # positive = anchor + small noise (same class)
    # negative = random embeddings (different class)

    # TODO: Instantiate TripletLoss and compute on sample data
    # triplet_loss = TripletLoss(margin=0.5)
    # anchor   = torch.randn(50, 8)
    # positive = anchor + 0.1 * torch.randn(50, 8)   # close to anchor
    # negative = torch.randn(50, 8)                   # random (far from anchor)
    # loss = triplet_loss(anchor, positive, negative)
    # print(f"Triplet loss: {loss.item():.4f}")
    # print("Expected: loss should be near 0 (positive is close, negative is far)")


if __name__ == "__main__":
    main()
