"""
Exercise 01: Scaled Dot-Product Attention from Scratch
======================================================
Implement the core attention mechanism using NumPy.
This is the fundamental building block of every Transformer.

Formula:
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

Learning goals:
- Understand Q, K, V matrix shapes and operations.
- See how attention weights form a probability distribution over positions.
- Visualize which tokens attend to which (attention heatmap).
- Understand causal masking for autoregressive generation.

References:
    - Original paper: https://arxiv.org/abs/1706.03762
    - The Illustrated Transformer: https://jalammar.github.io/illustrated-transformer/
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ---------------------------------------------------------------------------
# Core Attention
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention.

    Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V

    Args:
        Q (np.ndarray): Query matrix, shape (..., n_q, d_k).
        K (np.ndarray): Key matrix,   shape (..., n_k, d_k).
        V (np.ndarray): Value matrix,  shape (..., n_k, d_v).
        mask (np.ndarray or None): Additive mask, shape (..., n_q, n_k).
            Use 0 for positions to keep, -inf (or very large negative) to block.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - output: Attention output, shape (..., n_q, d_v).
            - attention_weights: Softmax weights, shape (..., n_q, n_k).

    TODO:
        1. Get d_k from Q.shape[-1].
        2. Compute scores = Q @ K.T / sqrt(d_k).
           For batched inputs: scores = Q @ K.swapaxes(-2, -1) / sqrt(d_k)
        3. If mask is not None: scores = scores + mask  (adds -inf where masked).
        4. Apply softmax over the last axis (axis=-1):
               attention_weights = exp(scores - max(scores)) / sum(exp(...))
               (subtract max for numerical stability — per-row)
        5. output = attention_weights @ V
        6. Return (output, attention_weights).

    Hint for softmax:
        scores_stable = scores - scores.max(axis=-1, keepdims=True)
        exp_scores    = np.exp(scores_stable)
        attention_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    """
    # TODO: implement scaled_dot_product_attention
    raise NotImplementedError("Implement scaled_dot_product_attention(Q, K, V, mask)")


def create_causal_mask(seq_len):
    """
    Create a causal (autoregressive) mask for decoder self-attention.

    A causal mask prevents position i from attending to any position j > i.
    Implemented as an upper-triangular matrix filled with -inf (so softmax → 0).

    Shape: (seq_len, seq_len)
    - mask[i, j] = 0    if j <= i  (position i CAN attend to j)
    - mask[i, j] = -inf if j > i   (position i CANNOT attend to j)

    Args:
        seq_len (int): Length of the sequence.

    Returns:
        np.ndarray: Causal mask, shape (seq_len, seq_len).

    TODO:
        1. Create a matrix of zeros, shape (seq_len, seq_len).
        2. Fill the upper triangle (k=1) with -np.inf:
               np.triu(matrix, k=1) fills upper triangular (excluding diagonal).
        3. Return the mask.

    Hint:
        mask = np.zeros((seq_len, seq_len))
        mask[np.triu_indices(seq_len, k=1)] = -np.inf
        return mask
    """
    # TODO: implement create_causal_mask
    raise NotImplementedError("Implement create_causal_mask(seq_len)")


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------

def demonstrate_attention(tokens, d_k=8, d_v=8, seed=42):
    """
    Create random Q, K, V matrices for a list of tokens and compute attention.
    Visualize both the unmasked and causally masked attention weights.

    Args:
        tokens (list[str]): List of word/token strings (e.g., ["the", "cat", "sat"]).
        d_k (int): Key/Query dimension.
        d_v (int): Value dimension.
        seed (int): Random seed for reproducibility.

    TODO:
        1. Set np.random.seed(seed).
        2. n = len(tokens)
        3. Create Q = np.random.randn(n, d_k), K = np.random.randn(n, d_k),
                  V = np.random.randn(n, d_v).
        4. Compute unmasked attention:
               output, attn_weights = scaled_dot_product_attention(Q, K, V, mask=None)
        5. Print attention_weights (n × n matrix) nicely.
        6. Visualize with plot_attention_heatmap(attn_weights, tokens, title="Unmasked").
        7. Compute causally masked attention:
               mask = create_causal_mask(n)
               output_masked, attn_masked = scaled_dot_product_attention(Q, K, V, mask)
        8. Visualize with plot_attention_heatmap(attn_masked, tokens, title="Causal Mask").
        9. Return (attn_weights, attn_masked).
    """
    # TODO: implement demonstrate_attention
    raise NotImplementedError("Implement demonstrate_attention(tokens, d_k, d_v, seed)")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_attention_heatmap(attention_weights, tokens, title="Attention Weights"):
    """
    Plot attention weights as a heatmap.
    This function is complete — no changes needed.

    Args:
        attention_weights (np.ndarray): Shape (n, n). Row = query, column = key.
        tokens (list[str]): Token labels for axes.
        title (str): Plot title.
    """
    n = len(tokens)
    fig, ax = plt.subplots(figsize=(max(5, n * 0.8), max(4, n * 0.7)))

    im = ax.imshow(attention_weights, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(tokens, fontsize=10)

    # Annotate cells with weight values
    for i in range(n):
        for j in range(n):
            val = attention_weights[i, j]
            color = 'white' if val > 0.6 else 'black'
            ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                    fontsize=8, color=color)

    ax.set_xlabel("Key (attended to)", fontsize=11)
    ax.set_ylabel("Query (attends from)", fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    np.random.seed(42)

    # ------------------------------------------------------------------
    # 1. Basic attention mechanics
    # ------------------------------------------------------------------
    print("=" * 55)
    print("1. Basic Scaled Dot-Product Attention")
    print("=" * 55)

    n, d_k, d_v = 4, 8, 8
    Q = np.random.randn(n, d_k)
    K = np.random.randn(n, d_k)
    V = np.random.randn(n, d_v)

    output, attn = scaled_dot_product_attention(Q, K, V)
    print(f"Q shape:       {Q.shape}")
    print(f"K shape:       {K.shape}")
    print(f"V shape:       {V.shape}")
    print(f"Output shape:  {output.shape}")
    print(f"Attn shape:    {attn.shape}")
    print(f"Attn row sums: {attn.sum(axis=-1)}  (each should sum to 1.0)")

    # ------------------------------------------------------------------
    # 2. Causal mask
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("2. Causal Mask")
    print("=" * 55)

    seq_len = 5
    mask = create_causal_mask(seq_len)
    print(f"Causal mask (seq_len={seq_len}):")
    print(mask)
    print("Lower triangle = 0 (attend), upper = -inf (blocked)")

    output_masked, attn_masked = scaled_dot_product_attention(Q[:5, :], K[:5, :],
                                                               V[:5, :], mask=mask)
    print(f"\nMasked attention weights (upper triangle should be 0):")
    np.set_printoptions(precision=3, suppress=True)
    print(attn_masked)

    # ------------------------------------------------------------------
    # 3. Attention visualization on sentence
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("3. Attention Demonstration on Sentence Tokens")
    print("=" * 55)

    sentence = ["The", "cat", "sat", "on", "the", "mat"]
    print(f"Tokens: {sentence}")

    attn_weights, attn_causally_masked = demonstrate_attention(
        sentence, d_k=16, d_v=16, seed=42
    )

    # ------------------------------------------------------------------
    # 4. Effect of scaling
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("4. Effect of Scaling by 1/sqrt(d_k)")
    print("=" * 55)

    for d_k_test in [4, 16, 64, 256]:
        Q_t = np.random.randn(1, d_k_test)
        K_t = np.random.randn(6, d_k_test)

        raw_scores   = (Q_t @ K_t.T).flatten()
        scaled_scores = raw_scores / np.sqrt(d_k_test)

        exp_raw    = np.exp(raw_scores    - raw_scores.max())
        exp_scaled = np.exp(scaled_scores - scaled_scores.max())
        softmax_raw    = exp_raw    / exp_raw.sum()
        softmax_scaled = exp_scaled / exp_scaled.sum()

        entropy_raw    = -np.sum(softmax_raw    * np.log(softmax_raw    + 1e-9))
        entropy_scaled = -np.sum(softmax_scaled * np.log(softmax_scaled + 1e-9))

        print(f"  d_k={d_k_test:4d} | Entropy (unscaled): {entropy_raw:.3f}"
              f"  |  Entropy (scaled): {entropy_scaled:.3f}")

    print("\nObservation: Larger d_k → more extreme (low-entropy) softmax without scaling.")
    print("Scaling by 1/√d_k keeps the entropy stable across different dimensions.")


if __name__ == "__main__":
    main()
