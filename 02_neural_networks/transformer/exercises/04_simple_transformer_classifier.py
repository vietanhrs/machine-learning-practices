"""
Exercise 04: Transformer Text Classifier
==========================================
Build an end-to-end Transformer-based text classifier:
    Token Embeddings + Positional Encoding
    → N Transformer Blocks
    → Mean Pooling
    → Linear Classifier Head

Apply to a synthetic text classification task with simple tokenization.

Learning goals:
- Combine all Transformer components into a complete model.
- Write a proper training loop for a Transformer classifier.
- Understand why mean pooling is preferred over CLS token pooling for simple tasks.
- Count and interpret model parameters.

References:
    - BERT paper: https://arxiv.org/abs/1810.04805
    - Sentence Transformers: https://www.sbert.net/
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import math


# ---------------------------------------------------------------------------
# Positional Encoding (Sinusoidal)
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as an nn.Module.
    Pre-computes the encoding matrix and adds it to input embeddings.

    This function is complete — no changes needed.
    """

    def __init__(self, d_model, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Build PE matrix: shape (1, max_seq_len, d_model)
        pe = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        i   = torch.arange(0, d_model, 2, dtype=torch.float)
        div_term = torch.exp(i * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Add positional encoding to input embeddings."""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Transformer Classifier
# ---------------------------------------------------------------------------

class TransformerClassifier(nn.Module):
    """
    Full Transformer encoder + mean pooling + linear classification head.

    Architecture:
        Token Embedding (vocab_size → d_model)
        + Sinusoidal Positional Encoding
        → N TransformerBlocks
        → Mean Pooling (average across seq_len dimension)
        → Dropout
        → Linear(d_model, n_classes)

    This is a simplified BERT-style encoder used for classification.

    Args:
        vocab_size (int): Vocabulary size for token embedding.
        d_model (int): Embedding and model dimension.
        n_heads (int): Number of attention heads.
        d_ff (int): FFN hidden dimension.
        n_blocks (int): Number of Transformer blocks.
        n_classes (int): Number of output classes.
        max_seq_len (int): Maximum sequence length.
        dropout (float): Dropout probability.

    TODO:
        In __init__:
            1. self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
            2. self.pos_enc   = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)
            3. self.blocks    = nn.ModuleList([
                   TransformerBlock(d_model, n_heads, d_ff, dropout)
                   for _ in range(n_blocks)
               ])
            4. self.norm      = nn.LayerNorm(d_model)
            5. self.dropout   = nn.Dropout(dropout)
            6. self.classifier = nn.Linear(d_model, n_classes)

            Note: You can import/reuse TransformerBlock from exercise 03,
                  or re-implement a minimal version here.
    """

    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_blocks, n_classes,
                 max_seq_len=128, dropout=0.1):
        super().__init__()
        # TODO: implement TransformerClassifier.__init__
        raise NotImplementedError("Implement TransformerClassifier.__init__")

    def forward(self, x, mask=None):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Token IDs, shape (batch, seq_len).
            mask (torch.Tensor or None): Padding mask, shape (batch, 1, 1, seq_len).
                Use -inf for padding positions so they get ~0 attention weight.

        Returns:
            torch.Tensor: Class logits, shape (batch, n_classes).

        TODO:
            1. Embed tokens:
                   x = self.embedding(x)       # (batch, seq_len, d_model)
            2. Add positional encoding:
                   x = self.pos_enc(x)         # (batch, seq_len, d_model)
            3. Pass through each Transformer block:
                   for block in self.blocks:
                       x, _ = block(x, mask)
            4. Apply final layer norm:
                   x = self.norm(x)
            5. Mean pooling across seq_len:
                   x = x.mean(dim=1)           # (batch, d_model)
                   (This averages representations across all positions)
            6. Apply dropout:
                   x = self.dropout(x)
            7. Linear classification:
                   logits = self.classifier(x) # (batch, n_classes)
            8. Return logits.

        Note on padding mask:
            Padding tokens should not contribute to the mean pooling.
            Advanced: use the mask to zero out padding positions before mean.
            Simple version: ignore padding mask (acceptable for short sequences).
        """
        # TODO: implement forward pass
        raise NotImplementedError("Implement TransformerClassifier.forward(x, mask)")


# ---------------------------------------------------------------------------
# Minimal TransformerBlock (self-contained, no dependency on exercise 03)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    Minimal Transformer encoder block for this exercise.
    (Self-contained to avoid cross-file imports.)

    TODO:
        Copy your implementation from exercise 03, or implement from scratch:
            - MultiHeadAttention (or use nn.MultiheadAttention for simplicity)
            - FeedForward (two Linear layers with ReLU)
            - LayerNorm + residual connections
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        # TODO: implement TransformerBlock for this exercise.
        # Simplest option: use PyTorch's built-in nn.MultiheadAttention:
        #   self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        #   self.ffn   = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        #   self.norm1 = nn.LayerNorm(d_model)
        #   self.norm2 = nn.LayerNorm(d_model)
        #   self.drop  = nn.Dropout(dropout)
        raise NotImplementedError("Implement TransformerBlock.__init__")

    def forward(self, x, mask=None):
        # TODO: implement forward
        # With nn.MultiheadAttention:
        #   x_norm = self.norm1(x)
        #   attn_out, attn_w = self.attn(x_norm, x_norm, x_norm, attn_mask=mask)
        #   x = x + self.drop(attn_out)
        #   x = x + self.drop(self.ffn(self.norm2(x)))
        #   return x, attn_w
        raise NotImplementedError("Implement TransformerBlock.forward(x, mask)")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_transformer(model, train_loader, val_loader, n_epochs, lr, device,
                      warmup_steps=100):
    """
    Train the TransformerClassifier with a linear warmup learning rate schedule.

    Warmup: LR increases linearly from 0 to lr over warmup_steps batches.
    After warmup: use a fixed LR (or add cosine decay as extension).

    Args:
        model (nn.Module): TransformerClassifier.
        train_loader (DataLoader): Training data.
        val_loader (DataLoader): Validation data.
        n_epochs (int): Number of training epochs.
        lr (float): Peak learning rate (after warmup).
        device: torch.device.
        warmup_steps (int): Number of warm-up gradient steps.

    Returns:
        dict: History with 'train_loss', 'val_loss', 'train_acc', 'val_acc'.

    TODO:
        1. Define optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
        2. Define criterion = nn.CrossEntropyLoss()
        3. Define a learning rate scheduler with linear warmup:
               scheduler = optim.lr_scheduler.LambdaLR(
                   optimizer,
                   lambda step: min(1.0, (step + 1) / warmup_steps)
               )
        4. For each epoch:
               a. Training loop: zero_grad, forward, loss, backward, clip_grad_norm_, step, scheduler.step.
               b. Validation loop.
               c. Print and record metrics.
        5. Return history.

    Hint for gradient clipping:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    """
    # TODO: implement train_transformer
    raise NotImplementedError("Implement train_transformer(...)")


# ---------------------------------------------------------------------------
# Parameter Counter
# ---------------------------------------------------------------------------

def count_parameters(model):
    """
    Count and display trainable parameters per layer group.
    This function is complete — no changes needed.

    Args:
        model (nn.Module): Any PyTorch model.

    Returns:
        int: Total number of trainable parameters.
    """
    total = 0
    print(f"{'Layer':<40} {'Parameters':>12}")
    print("-" * 54)
    for name, param in model.named_parameters():
        if param.requires_grad:
            n = param.numel()
            total += n
            print(f"  {name:<38} {n:>12,}")
    print("-" * 54)
    print(f"  {'TOTAL':<38} {total:>12,}")
    return total


# ---------------------------------------------------------------------------
# Dataset: Synthetic Topic Classification
# ---------------------------------------------------------------------------

class TopicDataset(Dataset):
    """
    Synthetic text classification dataset with 4 topics.

    Topics and keyword clusters:
        0 - Science:   physics, chemistry, biology, experiment, theory, quantum
        1 - Sports:    football, basketball, soccer, championship, player, score
        2 - Politics:  election, president, congress, policy, government, vote
        3 - Technology: software, computer, AI, algorithm, startup, digital

    Each sample is a padded sequence of token IDs of length max_seq_len.
    """

    TOPIC_WORDS = {
        0: ["physics", "chemistry", "biology", "experiment", "theory", "quantum",
            "atom", "molecule", "laboratory", "research", "science", "discovery"],
        1: ["football", "basketball", "soccer", "championship", "player", "score",
            "team", "match", "stadium", "coach", "league", "tournament"],
        2: ["election", "president", "congress", "policy", "government", "vote",
            "senator", "parliament", "democratic", "republic", "campaign", "ballot"],
        3: ["software", "computer", "artificial", "algorithm", "startup", "digital",
            "programming", "network", "database", "cloud", "machine", "technology"],
    }
    FILLER = ["the", "a", "is", "was", "and", "in", "of", "to", "for", "with"]

    def __init__(self, n_samples=3000, max_seq_len=32, seed=42):
        np.random.seed(seed)
        self.max_seq_len = max_seq_len
        self.n_classes   = len(self.TOPIC_WORDS)

        # Build vocabulary
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        for words in self.TOPIC_WORDS.values():
            for w in words:
                if w not in self.vocab:
                    self.vocab[w] = len(self.vocab)
        for w in self.FILLER:
            if w not in self.vocab:
                self.vocab[w] = len(self.vocab)
        self.vocab_size = len(self.vocab)

        # Generate samples
        sequences, labels = [], []
        for i in range(n_samples):
            topic = i % self.n_classes
            n_topic = np.random.randint(3, 8)
            n_filler = max_seq_len - n_topic

            words = (np.random.choice(self.TOPIC_WORDS[topic], n_topic).tolist()
                     + np.random.choice(self.FILLER, n_filler).tolist())
            np.random.shuffle(words)

            ids = [self.vocab.get(w, 1) for w in words[:max_seq_len]]
            ids += [0] * (max_seq_len - len(ids))   # pad

            sequences.append(ids)
            labels.append(topic)

        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels    = torch.tensor(labels,    dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset = TopicDataset(n_samples=4000, max_seq_len=32)
    print(f"Vocabulary size: {dataset.vocab_size}")
    print(f"Dataset size:    {len(dataset)} samples, {dataset.n_classes} classes")

    n_train = int(0.75 * len(dataset))
    n_val   = len(dataset) - n_train
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    model = TransformerClassifier(
        vocab_size   = dataset.vocab_size,
        d_model      = 64,
        n_heads      = 4,
        d_ff         = 256,
        n_blocks     = 2,
        n_classes    = dataset.n_classes,
        max_seq_len  = 32,
        dropout      = 0.1,
    ).to(device)

    print("\nModel parameter breakdown:")
    total = count_parameters(model)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print(f"\nTraining TransformerClassifier for 20 epochs...")
    history = train_transformer(
        model, train_loader, val_loader,
        n_epochs=20, lr=5e-4, device=device,
        warmup_steps=50
    )

    # ------------------------------------------------------------------
    # Plot training curves
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], label='Train Loss')
    ax1.plot(epochs, history['val_loss'],   label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Loss')
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history['train_acc'], label='Train Acc')
    ax2.plot(epochs, history['val_acc'],   label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.set_title('Accuracy')
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Transformer Classifier Training", fontsize=13)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # Compare model sizes
    # ------------------------------------------------------------------
    print("\nModel size comparison:")
    configs = [
        ("Tiny  (1 block,  d=32,  h=2)", dict(d_model=32,  n_heads=2, d_ff=128,  n_blocks=1)),
        ("Small (2 blocks, d=64,  h=4)", dict(d_model=64,  n_heads=4, d_ff=256,  n_blocks=2)),
        ("Base  (4 blocks, d=128, h=8)", dict(d_model=128, n_heads=8, d_ff=512,  n_blocks=4)),
        ("Large (6 blocks, d=256, h=8)", dict(d_model=256, n_heads=8, d_ff=1024, n_blocks=6)),
    ]

    for name, cfg in configs:
        m = TransformerClassifier(
            vocab_size=dataset.vocab_size,
            n_classes=dataset.n_classes,
            max_seq_len=32,
            **cfg
        )
        n = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"  {name}: {n:>10,} parameters")


if __name__ == "__main__":
    main()
