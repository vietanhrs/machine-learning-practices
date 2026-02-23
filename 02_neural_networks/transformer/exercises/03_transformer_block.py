"""
Exercise 03: Transformer Block in PyTorch
==========================================
Implement the full Transformer encoder block from scratch using PyTorch:
    MultiHeadAttention → FeedForward → LayerNorm → residual connections.

This is the core repeated unit of BERT, GPT, and all modern Transformer models.

Learning goals:
- Implement multi-head attention with proper tensor reshaping.
- Combine attention + FFN + LayerNorm + residuals into a Transformer block.
- Stack multiple blocks into an encoder.
- Verify correct output shapes.

References:
    - Vaswani et al. (2017): https://arxiv.org/abs/1706.03762
    - Annotated Transformer: https://nlp.seas.harvard.edu/annotated-transformer/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Multi-Head Attention
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as described in "Attention Is All You Need".

    For each head h:
        head_h = Attention(Q W_Qh, K W_Kh, V W_Vh)

    MultiHead(Q, K, V) = Concat(head_1, ..., head_n_heads) W_O

    Scaled dot-product attention:
        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_head)) V

    Args:
        d_model (int): Model embedding dimension.
        n_heads (int): Number of attention heads. d_model must be divisible by n_heads.
    """

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads  # dimension per head

        # Linear projections for Q, K, V and output
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, x):
        """
        Reshape tensor to split the last dimension into (n_heads, d_head).

        Args:
            x (torch.Tensor): Shape (batch, seq_len, d_model).

        Returns:
            torch.Tensor: Shape (batch, n_heads, seq_len, d_head).

        TODO:
            1. Get batch_size, seq_len from x.shape[:2].
            2. Reshape x: x = x.view(batch_size, seq_len, self.n_heads, self.d_head)
            3. Transpose to (batch, n_heads, seq_len, d_head):
                   x = x.transpose(1, 2)
            4. Return x.
        """
        # TODO: implement split_heads
        raise NotImplementedError("Implement MultiHeadAttention.split_heads(x)")

    def forward(self, Q, K, V, mask=None):
        """
        Compute multi-head attention.

        Args:
            Q (torch.Tensor): Query, shape (batch, seq_q, d_model).
            K (torch.Tensor): Key,   shape (batch, seq_k, d_model).
            V (torch.Tensor): Value, shape (batch, seq_k, d_model).
            mask (torch.Tensor or None): Additive mask,
                shape (batch, 1, seq_q, seq_k) or broadcastable.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - output: Shape (batch, seq_q, d_model).
                - attention_weights: Shape (batch, n_heads, seq_q, seq_k).

        TODO:
            1. Project Q, K, V through linear layers:
                   Q = self.W_Q(Q)    # (batch, seq_q, d_model)
                   K = self.W_K(K)
                   V = self.W_V(V)
            2. Split into heads using split_heads:
                   Q = self.split_heads(Q)  # (batch, n_heads, seq_q, d_head)
                   K = self.split_heads(K)
                   V = self.split_heads(V)
            3. Compute scaled attention scores:
                   scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)
                   # shape: (batch, n_heads, seq_q, seq_k)
            4. Apply mask if provided:
                   if mask is not None: scores = scores + mask
            5. Compute attention weights:
                   attention_weights = F.softmax(scores, dim=-1)
            6. Compute attended values:
                   attended = attention_weights @ V
                   # shape: (batch, n_heads, seq_q, d_head)
            7. Concatenate heads (reverse split_heads):
                   attended = attended.transpose(1, 2).contiguous()
                   # (batch, seq_q, n_heads, d_head)
                   attended = attended.view(batch_size, seq_q, self.d_model)
                   # (batch, seq_q, d_model)
            8. Apply output projection:
                   output = self.W_O(attended)
            9. Return (output, attention_weights).
        """
        batch_size = Q.shape[0]
        seq_q      = Q.shape[1]
        # TODO: implement multi-head attention forward
        raise NotImplementedError("Implement MultiHeadAttention.forward(Q, K, V, mask)")


# ---------------------------------------------------------------------------
# Feed-Forward Network
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    FFN(x) = ReLU(x W_1 + b_1) W_2 + b_2

    Applied independently to each position (same parameters, different inputs).

    Args:
        d_model (int): Input and output dimension.
        d_ff (int): Inner (hidden) dimension. Typically 4 × d_model.
        dropout (float): Dropout probability between the two linear layers.

    TODO:
        In __init__:
            Define self.net = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
            )

        In forward(self, x):
            Return self.net(x).
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # TODO: define self.net
        raise NotImplementedError("Implement FeedForward.__init__")

    def forward(self, x):
        # TODO: return self.net(x)
        raise NotImplementedError("Implement FeedForward.forward(x)")


# ---------------------------------------------------------------------------
# Transformer Block (Encoder Layer)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    A single Transformer encoder block (layer).

    Architecture (Pre-LN variant, most stable for training):
        x → LayerNorm → MultiHeadAttention → + x → (residual)
          → LayerNorm → FeedForward        → + x → (residual)
          → output

    (Original paper uses Post-LN: Attention(x) + x → LayerNorm → ...)

    Args:
        d_model (int): Embedding dimension.
        n_heads (int): Number of attention heads.
        d_ff (int): FFN inner dimension.
        dropout (float): Dropout probability.

    TODO:
        In __init__:
            1. self.attn    = MultiHeadAttention(d_model, n_heads)
            2. self.ffn     = FeedForward(d_model, d_ff, dropout)
            3. self.norm1   = nn.LayerNorm(d_model)
            4. self.norm2   = nn.LayerNorm(d_model)
            5. self.dropout = nn.Dropout(dropout)

        In forward(self, x, mask=None):
            # Sub-layer 1: Self-attention with residual
            1. x_norm = self.norm1(x)
            2. attn_output, attn_weights = self.attn(x_norm, x_norm, x_norm, mask)
            3. x = x + self.dropout(attn_output)    ← residual connection

            # Sub-layer 2: FFN with residual
            4. x_norm = self.norm2(x)
            5. ffn_output = self.ffn(x_norm)
            6. x = x + self.dropout(ffn_output)     ← residual connection

            7. return x, attn_weights
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        # TODO: implement TransformerBlock.__init__
        raise NotImplementedError("Implement TransformerBlock.__init__")

    def forward(self, x, mask=None):
        # TODO: implement TransformerBlock.forward
        raise NotImplementedError("Implement TransformerBlock.forward(x, mask)")


# ---------------------------------------------------------------------------
# Transformer Encoder (Stack of N Blocks)
# ---------------------------------------------------------------------------

class TransformerEncoder(nn.Module):
    """
    A stack of N identical TransformerBlocks forming a Transformer encoder.

    Architecture:
        Input Embeddings + Positional Encoding
        → Block 1 → Block 2 → ... → Block N
        → Final LayerNorm
        → Output

    Args:
        n_blocks (int): Number of stacked TransformerBlocks.
        d_model (int): Embedding dimension.
        n_heads (int): Number of attention heads.
        d_ff (int): FFN inner dimension.
        dropout (float): Dropout probability.

    TODO:
        In __init__:
            1. self.blocks = nn.ModuleList([
                   TransformerBlock(d_model, n_heads, d_ff, dropout)
                   for _ in range(n_blocks)
               ])
            2. self.norm = nn.LayerNorm(d_model)  ← final normalization

        In forward(self, x, mask=None):
            1. all_attn_weights = []
            2. For each block in self.blocks:
                   x, attn_weights = block(x, mask)
                   all_attn_weights.append(attn_weights)
            3. x = self.norm(x)
            4. Return (x, all_attn_weights)
               # x shape: (batch, seq_len, d_model)
               # all_attn_weights: list of (batch, n_heads, seq, seq) — one per block
    """

    def __init__(self, n_blocks, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        # TODO: implement TransformerEncoder.__init__
        raise NotImplementedError("Implement TransformerEncoder.__init__")

    def forward(self, x, mask=None):
        # TODO: implement TransformerEncoder.forward
        raise NotImplementedError("Implement TransformerEncoder.forward(x, mask)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(42)

    # ------------------------------------------------------------------
    # Hyperparameters
    # ------------------------------------------------------------------
    batch_size = 4
    seq_len    = 16
    d_model    = 64
    n_heads    = 4      # d_head = d_model / n_heads = 16
    d_ff       = 256    # 4 × d_model
    n_blocks   = 3
    dropout    = 0.1

    assert d_model % n_heads == 0

    # ------------------------------------------------------------------
    # 1. Test MultiHeadAttention
    # ------------------------------------------------------------------
    print("=" * 55)
    print("1. MultiHeadAttention Shape Test")
    print("=" * 55)

    mha = MultiHeadAttention(d_model, n_heads)
    x   = torch.randn(batch_size, seq_len, d_model)
    out, attn_w = mha(x, x, x)
    print(f"Input shape:            {x.shape}")
    print(f"MHA output shape:       {out.shape}    (should be {batch_size, seq_len, d_model})")
    print(f"Attention weights:      {attn_w.shape}  (batch, heads, seq, seq)")

    # Verify attention weights sum to 1
    row_sums = attn_w.sum(dim=-1)
    print(f"Attn weights row sums (should be all 1.0): min={row_sums.min():.4f}, max={row_sums.max():.4f}")

    # ------------------------------------------------------------------
    # 2. Test FeedForward
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("2. FeedForward Shape Test")
    print("=" * 55)

    ffn = FeedForward(d_model, d_ff, dropout)
    out_ffn = ffn(x)
    print(f"FFN input shape:  {x.shape}")
    print(f"FFN output shape: {out_ffn.shape}  (should match input)")

    # ------------------------------------------------------------------
    # 3. Test TransformerBlock
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("3. TransformerBlock Shape Test")
    print("=" * 55)

    block = TransformerBlock(d_model, n_heads, d_ff, dropout)
    out_block, attn_block = block(x)
    print(f"Block input shape:  {x.shape}")
    print(f"Block output shape: {out_block.shape}")
    print(f"Block attn shape:   {attn_block.shape}")

    # Test with causal mask
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
    mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq) for broadcasting
    out_masked, _ = block(x, mask=mask)
    print(f"Block output (with causal mask): {out_masked.shape}")

    # ------------------------------------------------------------------
    # 4. Test TransformerEncoder
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("4. TransformerEncoder Shape Test")
    print("=" * 55)

    encoder = TransformerEncoder(n_blocks, d_model, n_heads, d_ff, dropout)
    encoder.eval()

    with torch.no_grad():
        enc_out, all_attn = encoder(x)

    print(f"Encoder input shape:       {x.shape}")
    print(f"Encoder output shape:      {enc_out.shape}")
    print(f"Number of attention maps:  {len(all_attn)} (one per block)")
    print(f"Attention map shape:       {all_attn[0].shape}")

    # Count parameters
    n_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"\nEncoder trainable parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # 5. Visualize attention patterns from one block
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("5. Visualizing Attention Patterns (Block 0, Head 0)")
    print("=" * 55)

    attn_map = all_attn[0][0, 0].numpy()  # block 0, batch 0, head 0
    plt.figure(figsize=(6, 5))
    plt.imshow(attn_map, cmap='Blues', vmin=0)
    plt.colorbar(label='Attention Weight')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.title('Attention Pattern (Block 0, Head 0)\nRandom input — no meaningful pattern expected')
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # 6. Parameter count breakdown
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("6. Parameter Count Breakdown")
    print("=" * 55)

    components = {
        "MultiHeadAttention": MultiHeadAttention(d_model, n_heads),
        "FeedForward":        FeedForward(d_model, d_ff),
    }
    for name, module in components.items():
        n = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {n:,} parameters")

    print(f"\n  Per TransformerBlock: {sum(p.numel() for p in TransformerBlock(d_model, n_heads, d_ff).parameters()):,}")
    print(f"  Full Encoder ({n_blocks} blocks): {n_params:,}")
    print(f"\n  Expected: ~{n_blocks * (4*d_model*d_model + 2*d_model*d_ff):,}")


if __name__ == "__main__":
    main()
