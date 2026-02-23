"""
Exercise 02: Positional Encoding
=================================
Implement and visualize sinusoidal and learned positional encodings.
Understand *why* Transformers need positional information and how
the sinusoidal encoding encodes relative positions.

Learning goals:
- Implement the original Transformer sinusoidal PE formula.
- Use nn.Embedding for learned PE.
- Visualize the encoding matrix and understand frequency structure.
- Test the relative position property of sinusoidal encodings.

References:
    - Vaswani et al. (2017): https://arxiv.org/abs/1706.03762
    - The Annotated Transformer: https://nlp.seas.harvard.edu/annotated-transformer/
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Sinusoidal Positional Encoding
# ---------------------------------------------------------------------------

def sinusoidal_encoding(seq_len, d_model):
    """
    Compute the sinusoidal positional encoding matrix from the original Transformer.

    Formulas:
        PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
        PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))

    Where:
        pos ∈ {0, 1, ..., seq_len - 1}
        i   ∈ {0, 1, ..., d_model/2 - 1}

    Args:
        seq_len (int): Number of positions (rows).
        d_model (int): Embedding dimension (columns). Should be even.

    Returns:
        np.ndarray: Positional encoding matrix, shape (seq_len, d_model).

    TODO:
        1. Initialize PE = np.zeros((seq_len, d_model)).
        2. Create position vector: pos = np.arange(seq_len).reshape(-1, 1)
           Shape: (seq_len, 1)
        3. Create dimension index vector: i = np.arange(0, d_model, 2)
           Shape: (d_model//2,)
        4. Compute the division term:
               div_term = 10000 ** (i / d_model)
           Or equivalently:
               div_term = np.exp(i * (-np.log(10000.0) / d_model))
        5. Fill even columns (2i):   PE[:, 0::2] = sin(pos / div_term)
        6. Fill odd  columns (2i+1): PE[:, 1::2] = cos(pos / div_term)
        7. Return PE.

    Note: Broadcasting handles the (seq_len, 1) × (d_model//2,) computation automatically.
    """
    # TODO: implement sinusoidal_encoding
    raise NotImplementedError("Implement sinusoidal_encoding(seq_len, d_model)")


# ---------------------------------------------------------------------------
# Learned Positional Encoding
# ---------------------------------------------------------------------------

def learned_positional_encoding(seq_len, d_model):
    """
    Create a learned positional encoding using nn.Embedding.

    Each position index maps to a trainable vector of size d_model.
    During training, these embeddings are updated by gradient descent
    like any other parameters.

    Args:
        seq_len (int): Maximum sequence length (number of positions).
        d_model (int): Embedding dimension.

    Returns:
        nn.Embedding: Positional embedding layer with shape (seq_len, d_model).

    TODO:
        1. Create and return nn.Embedding(seq_len, d_model).
        2. Initialize weights using normal distribution (optional):
               nn.init.normal_(embedding.weight, mean=0, std=0.02)

    Usage example:
        pe = learned_positional_encoding(100, 64)
        positions = torch.arange(seq_len).unsqueeze(0)  # (1, seq_len)
        encodings = pe(positions)                        # (1, seq_len, d_model)
    """
    # TODO: implement learned_positional_encoding
    raise NotImplementedError("Implement learned_positional_encoding(seq_len, d_model)")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_positional_encoding(PE, title="Positional Encoding"):
    """
    Visualize the positional encoding matrix as a heatmap.
    This function is complete — no changes needed.

    Args:
        PE (np.ndarray): Positional encoding matrix, shape (seq_len, d_model).
        title (str): Plot title.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Full heatmap
    im = axes[0].imshow(PE, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(im, ax=axes[0])
    axes[0].set_xlabel("Dimension (d_model)")
    axes[0].set_ylabel("Position")
    axes[0].set_title(f"{title}\nHeatmap (seq_len × d_model)")

    # First 4 dimensions over positions
    seq_len = PE.shape[0]
    pos_range = np.arange(seq_len)
    for dim in range(min(4, PE.shape[1])):
        axes[1].plot(pos_range, PE[:, dim], label=f"dim {dim}")
    axes[1].set_xlabel("Position")
    axes[1].set_ylabel("Encoding Value")
    axes[1].set_title(f"{title}\nFirst 4 Dimensions vs Position")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Relative Position Property
# ---------------------------------------------------------------------------

def test_relative_position_property(PE):
    """
    Test whether sinusoidal PE encodes relative position information
    through dot products.

    Intuition: The dot product PE[pos] · PE[pos + k] should depend mainly
    on the offset k, not on the absolute position pos.
    This is a key property that helps the model learn relative positional patterns.

    Args:
        PE (np.ndarray): Sinusoidal PE matrix, shape (seq_len, d_model).

    TODO:
        1. Normalize PE rows to unit vectors (L2 norm):
               PE_norm = PE / (np.linalg.norm(PE, axis=1, keepdims=True) + 1e-8)
        2. Compute dot product similarity matrix:
               similarity = PE_norm @ PE_norm.T     shape: (seq_len, seq_len)
        3. For each offset k from 0 to some max_k, compute the mean similarity
           for that offset across all valid position pairs:
               for k in range(max_k):
                   offsets = [similarity[pos, pos+k] for valid pos]
                   mean_sim[k] = mean(offsets)
        4. Plot: mean similarity vs offset k.
           Expected: similarity decreases as k increases (closer positions → more similar).
        5. Also plot the full similarity matrix as a heatmap.

    Note: Perfect encoding would show constant similarity for each k regardless of pos.
          The sinusoidal encoding has this property approximately.
    """
    # TODO: implement test_relative_position_property
    raise NotImplementedError("Implement test_relative_position_property(PE)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    SEQ_LEN = 100
    D_MODEL  = 64

    # ------------------------------------------------------------------
    # 1. Sinusoidal encoding
    # ------------------------------------------------------------------
    print("=" * 55)
    print("1. Sinusoidal Positional Encoding")
    print("=" * 55)

    PE_sin = sinusoidal_encoding(SEQ_LEN, D_MODEL)
    print(f"PE shape: {PE_sin.shape}")
    print(f"PE[0] (first position)[:8]:  {PE_sin[0, :8].round(3)}")
    print(f"PE[1] (second position)[:8]: {PE_sin[1, :8].round(3)}")

    # Verify: even dimensions are sin, odd are cos
    # At pos=0: sin(0) = 0, cos(0) = 1
    print(f"\nAt pos=0: even dims should be 0, odd dims should be 1")
    print(f"  PE[0, 0::2] (even, sin): {PE_sin[0, 0:8:2].round(3)}")
    print(f"  PE[0, 1::2] (odd, cos):  {PE_sin[0, 1:8:2].round(3)}")

    visualize_positional_encoding(PE_sin, title="Sinusoidal Positional Encoding")

    # ------------------------------------------------------------------
    # 2. Learned positional encoding (random initialized)
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("2. Learned Positional Encoding (random init)")
    print("=" * 55)

    pe_embedding = learned_positional_encoding(SEQ_LEN, D_MODEL)
    positions = torch.arange(SEQ_LEN).unsqueeze(0)    # shape: (1, SEQ_LEN)
    PE_learned = pe_embedding(positions).detach().numpy().squeeze(0)  # (SEQ_LEN, D_MODEL)

    print(f"Learned PE shape: {PE_learned.shape}")
    print("(Before training, learned PE is random — no structure.)")
    visualize_positional_encoding(PE_learned, title="Learned Positional Encoding (untrained)")

    # ------------------------------------------------------------------
    # 3. Relative position property
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("3. Relative Position Property of Sinusoidal PE")
    print("=" * 55)

    test_relative_position_property(PE_sin)

    # ------------------------------------------------------------------
    # 4. Compare frequency patterns
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("4. Frequency Analysis: Low vs High Dimensions")
    print("=" * 55)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    pos = np.arange(SEQ_LEN)

    PE_large = sinusoidal_encoding(SEQ_LEN, 512)  # more dimensions to show range

    # Low frequency dimensions (large d_model → slow oscillation)
    for i, dim in enumerate([0, 2]):
        axes[0, i].plot(pos, PE_large[:, dim], color='steelblue')
        axes[0, i].set_title(f"Dim {dim} (Low frequency — slow)")
        axes[0, i].set_xlabel("Position")
        axes[0, i].set_ylabel("Value")
        axes[0, i].grid(True, alpha=0.3)

    # High frequency dimensions (small d_model index → fast oscillation)
    for i, dim in enumerate([500, 510]):
        axes[1, i].plot(pos, PE_large[:, dim], color='darkorange')
        axes[1, i].set_title(f"Dim {dim} (High frequency — fast)")
        axes[1, i].set_xlabel("Position")
        axes[1, i].set_ylabel("Value")
        axes[1, i].grid(True, alpha=0.3)

    plt.suptitle("Sinusoidal PE: Low vs High Frequency Dimensions", fontsize=13)
    plt.tight_layout()
    plt.show()

    print("\nObservation: Lower-index dimensions oscillate slowly (long-range patterns).")
    print("Higher-index dimensions oscillate quickly (fine-grained position distinction).")
    print("Together, they create a unique fingerprint for each position.")


if __name__ == "__main__":
    main()
