"""
Exercise 02: Seq2Seq with Bahdanau Attention
============================================
Implement a complete sequence-to-sequence model with Bahdanau attention
in PyTorch. Train it on a toy number-to-word translation task to observe
how attention learns to align source and target sequences.

Learning Goals:
    - Implement Encoder (GRU-based) that outputs all hidden states
    - Implement Bahdanau attention (additive attention)
    - Implement Decoder that uses attention context at each step
    - Visualize the attention matrix to see source-target alignment
    - Understand teacher forcing during training

Requirements:
    pip install torch numpy matplotlib
"""

import random
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Toy Translation Task: Numbers to Words
# ---------------------------------------------------------------------------
# Source: "1 2 3" → Target: "one two three"
# This tiny task lets us clearly visualize attention alignment.

DIGIT_TO_WORD = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
}

# Special tokens
PAD, SOS, EOS = "<PAD>", "<SOS>", "<EOS>"


def build_toy_dataset(
    n_samples: int = 2000, max_len: int = 5, seed: int = 42
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Generate number→word translation pairs.

    Returns:
        src_sentences: List of digit-string lists, e.g., [['3', '1', '4']]
        tgt_sentences: List of word-string lists, e.g., [['three', 'one', 'four']]
    """
    rng = np.random.default_rng(seed)
    src_sentences, tgt_sentences = [], []
    for _ in range(n_samples):
        length = rng.integers(1, max_len + 1)
        digits = rng.integers(0, 10, size=length).tolist()
        src_sentences.append([str(d) for d in digits])
        tgt_sentences.append([DIGIT_TO_WORD[d] for d in digits])
    return src_sentences, tgt_sentences


def build_vocab(sentences: List[List[str]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build word→index and index→word vocabularies."""
    vocab = {PAD: 0, SOS: 1, EOS: 2}
    for sentence in sentences:
        for token in sentence:
            if token not in vocab:
                vocab[token] = len(vocab)
    idx_to_word = {i: w for w, i in vocab.items()}
    return vocab, idx_to_word


def encode_sentence(
    tokens: List[str],
    vocab: Dict[str, int],
    add_sos: bool = False,
    add_eos: bool = True,
    max_len: Optional[int] = None,
) -> List[int]:
    """Convert tokens to indices, optionally adding SOS/EOS and padding."""
    ids = []
    if add_sos:
        ids.append(vocab[SOS])
    ids.extend(vocab.get(t, vocab[PAD]) for t in tokens)
    if add_eos:
        ids.append(vocab[EOS])
    if max_len and len(ids) < max_len:
        ids += [vocab[PAD]] * (max_len - len(ids))
    return ids[:max_len] if max_len else ids


# ---------------------------------------------------------------------------
# 1. Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """
    GRU-based encoder for Seq2Seq.

    Embeds source tokens and processes them with a GRU, returning
    ALL hidden states (not just the final one) for use by attention.

    Args:
        vocab_size:  Source vocabulary size.
        embed_dim:   Embedding dimension.
        hidden_dim:  GRU hidden state dimension.
        n_layers:    Number of GRU layers.
        dropout:     Dropout between layers.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # TODO:
        #   1. Define self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0).
        #   2. Define self.gru = nn.GRU(embed_dim, hidden_dim, n_layers,
        #                               batch_first=True, dropout=dropout if n_layers > 1 else 0).
        #   3. Store hidden_dim and n_layers.
        raise NotImplementedError("TODO: implement Encoder.__init__()")

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode source sequence.

        Args:
            src: LongTensor of shape (batch, src_len) — source token indices.

        Returns:
            encoder_outputs: FloatTensor of shape (batch, src_len, hidden_dim)
                             — hidden state at EVERY source position.
            hidden:          FloatTensor of shape (n_layers, batch, hidden_dim)
                             — final encoder hidden state (used to initialize decoder).

        TODO:
            1. embedded = self.embedding(src) → shape (batch, src_len, embed_dim).
            2. encoder_outputs, hidden = self.gru(embedded)
               → encoder_outputs: (batch, src_len, hidden_dim)
               → hidden: (n_layers, batch, hidden_dim)
            3. Return (encoder_outputs, hidden).
        """
        raise NotImplementedError("TODO: implement Encoder.forward()")


# ---------------------------------------------------------------------------
# 2. Bahdanau Attention
# ---------------------------------------------------------------------------

class BahdanauAttention(nn.Module):
    """
    Bahdanau (additive) attention mechanism.

    Computes attention weights over encoder outputs given the current
    decoder hidden state, then returns a weighted context vector.

    Score function:
        score(s, h_i) = v^T * tanh(W1 * s + W2 * h_i)

    Args:
        hidden_dim: Dimension of decoder hidden state and encoder outputs.
        attn_dim:   Internal dimension for the attention computation.
    """

    def __init__(self, hidden_dim: int = 128, attn_dim: int = 64) -> None:
        super().__init__()
        # TODO:
        #   1. Define self.W1 = nn.Linear(hidden_dim, attn_dim, bias=False).
        #      (Projects decoder hidden state.)
        #   2. Define self.W2 = nn.Linear(hidden_dim, attn_dim, bias=False).
        #      (Projects each encoder output.)
        #   3. Define self.v  = nn.Linear(attn_dim, 1, bias=False).
        #      (Produces a scalar score for each source position.)
        raise NotImplementedError("TODO: implement BahdanauAttention.__init__()")

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute context vector and attention weights.

        Args:
            decoder_hidden:  FloatTensor of shape (batch, hidden_dim)
                             — current decoder hidden state.
            encoder_outputs: FloatTensor of shape (batch, src_len, hidden_dim)
                             — all encoder hidden states.

        Returns:
            context:     FloatTensor of shape (batch, hidden_dim)
                         — weighted sum of encoder outputs.
            attn_weights: FloatTensor of shape (batch, src_len)
                          — attention distribution over source positions.

        TODO:
            1. Expand decoder_hidden for broadcasting over src_len:
                   decoder_hidden_exp = decoder_hidden.unsqueeze(1)
                   → shape (batch, 1, hidden_dim)
            2. Compute combined = tanh(W1(decoder_hidden_exp) + W2(encoder_outputs))
                   W1_part = self.W1(decoder_hidden_exp)  → (batch, 1, attn_dim)
                   W2_part = self.W2(encoder_outputs)      → (batch, src_len, attn_dim)
                   combined = torch.tanh(W1_part + W2_part) → (batch, src_len, attn_dim)
            3. Compute scores = self.v(combined).squeeze(-1) → (batch, src_len)
            4. Compute attn_weights = F.softmax(scores, dim=-1) → (batch, src_len)
            5. Compute context = weighted sum:
                   attn_weights_exp = attn_weights.unsqueeze(1)  → (batch, 1, src_len)
                   context = torch.bmm(attn_weights_exp, encoder_outputs) → (batch, 1, hidden_dim)
                   context = context.squeeze(1) → (batch, hidden_dim)
            6. Return (context, attn_weights).
        """
        raise NotImplementedError("TODO: implement BahdanauAttention.forward()")


# ---------------------------------------------------------------------------
# 3. Decoder
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    """
    GRU-based decoder with Bahdanau attention.

    At each step:
        1. Embed the current target token.
        2. Use attention to compute a context vector from encoder outputs.
        3. Concatenate embedding + context and pass through GRU.
        4. Project GRU output to target vocabulary logits.

    Args:
        vocab_size: Target vocabulary size.
        embed_dim:  Embedding dimension.
        hidden_dim: GRU hidden state dimension.
        attn_dim:   Internal attention dimension.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        attn_dim: int = 64,
    ) -> None:
        super().__init__()
        # TODO:
        #   1. self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0).
        #   2. self.attention = BahdanauAttention(hidden_dim, attn_dim).
        #   3. self.gru = nn.GRU(embed_dim + hidden_dim, hidden_dim, batch_first=True).
        #      Note: GRU input is [embedding, context] concatenated → embed_dim + hidden_dim.
        #   4. self.fc_out = nn.Linear(hidden_dim, vocab_size).
        raise NotImplementedError("TODO: implement Decoder.__init__()")

    def forward(
        self,
        tgt_token: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One decoding step.

        Args:
            tgt_token:       LongTensor of shape (batch,) — current target token index.
            hidden:          FloatTensor of shape (1, batch, hidden_dim) — decoder hidden state.
            encoder_outputs: FloatTensor of shape (batch, src_len, hidden_dim).

        Returns:
            logits:       FloatTensor of shape (batch, vocab_size) — unnormalized scores.
            new_hidden:   FloatTensor of shape (1, batch, hidden_dim) — updated hidden state.
            attn_weights: FloatTensor of shape (batch, src_len) — attention distribution.

        TODO:
            1. embedded = self.embedding(tgt_token.unsqueeze(1))
               → shape (batch, 1, embed_dim).
            2. Compute context and attn_weights using attention:
                   hidden_for_attn = hidden.squeeze(0)  → (batch, hidden_dim)
                   context, attn_weights = self.attention(hidden_for_attn, encoder_outputs)
            3. Concatenate embedding and context:
                   context_exp = context.unsqueeze(1)   → (batch, 1, hidden_dim)
                   gru_input = torch.cat([embedded, context_exp], dim=-1)
                   → shape (batch, 1, embed_dim + hidden_dim)
            4. Pass through GRU:
                   gru_out, new_hidden = self.gru(gru_input, hidden)
                   → gru_out: (batch, 1, hidden_dim)
            5. Compute logits:
                   logits = self.fc_out(gru_out.squeeze(1)) → (batch, vocab_size)
            6. Return (logits, new_hidden, attn_weights).
        """
        raise NotImplementedError("TODO: implement Decoder.forward()")


# ---------------------------------------------------------------------------
# 4. Full Seq2Seq Model
# ---------------------------------------------------------------------------

class Seq2SeqAttention(nn.Module):
    """
    Full Seq2Seq model with Bahdanau attention.

    Args:
        encoder: Encoder module.
        decoder: Decoder module.
    """

    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass through the Seq2Seq model.

        Args:
            src:                   LongTensor (batch, src_len) — source token indices.
            tgt:                   LongTensor (batch, tgt_len) — target token indices
                                   (including SOS at position 0, EOS at end).
            teacher_forcing_ratio: Probability of using ground-truth target token
                                   as next decoder input (vs. model's own prediction).

        Returns:
            all_logits:      FloatTensor (batch, tgt_len-1, tgt_vocab_size)
                             — logits for positions 1 through tgt_len-1.
            all_attn_weights: FloatTensor (batch, tgt_len-1, src_len)
                              — attention weights for each decoding step.

        TODO:
            1. Encode source: encoder_outputs, hidden = self.encoder(src).
            2. Initialize decoder input: dec_input = tgt[:, 0] (the SOS token).
            3. Create storage: all_logits = [], all_attn_weights = [].
            4. For each target position t from 1 to tgt_len-1:
                 a. logits, hidden, attn_weights = self.decoder(dec_input, hidden, encoder_outputs)
                 b. all_logits.append(logits)
                 c. all_attn_weights.append(attn_weights)
                 d. Teacher forcing:
                        if random.random() < teacher_forcing_ratio:
                            dec_input = tgt[:, t]          ← use ground truth
                        else:
                            dec_input = logits.argmax(dim=-1)  ← use prediction
            5. Stack: all_logits = torch.stack(all_logits, dim=1).
            6. Stack: all_attn_weights = torch.stack(all_attn_weights, dim=1).
            7. Return (all_logits, all_attn_weights).
        """
        raise NotImplementedError("TODO: implement Seq2SeqAttention.forward()")


# ---------------------------------------------------------------------------
# 5. Translation (Greedy Decode)
# ---------------------------------------------------------------------------

def translate(
    model: Seq2SeqAttention,
    src_sentence: List[str],
    src_vocab: Dict[str, int],
    tgt_vocab: Dict[str, int],
    tgt_idx_to_word: Dict[int, str],
    max_len: int = 20,
    device: Optional[torch.device] = None,
) -> Tuple[List[str], np.ndarray]:
    """
    Translate a single source sentence using greedy decoding.

    Args:
        model:           Trained Seq2SeqAttention model.
        src_sentence:    List of source tokens.
        src_vocab:       Source token → index mapping.
        tgt_vocab:       Target token → index mapping.
        tgt_idx_to_word: Target index → token mapping.
        max_len:         Maximum number of tokens to generate.
        device:          Torch device.

    Returns:
        translation:     List of predicted target tokens (excluding SOS/EOS).
        attn_matrix:     2D numpy array of shape (tgt_len, src_len) — attention weights.

    TODO:
        1. Set model to eval mode.
        2. Encode src_sentence: convert to indices, add EOS, pad to max_len,
           create a batch of size 1, move to device.
        3. Run encoder: encoder_outputs, hidden = model.encoder(src_tensor).
        4. Initialize dec_input = tensor([tgt_vocab[SOS]]).
        5. Loop until EOS or max_len:
             a. logits, hidden, attn_weights = model.decoder(dec_input, hidden, encoder_outputs)
             b. next_token_idx = logits.argmax(dim=-1).item()
             c. Store attn_weights and next_token_idx.
             d. If next_token_idx == tgt_vocab[EOS]: break.
             e. Set dec_input = tensor([next_token_idx]).
        6. Decode indices → words (exclude EOS).
        7. Stack attention weights → numpy array.
        8. Return (translation, attn_matrix).
    """
    raise NotImplementedError("TODO: implement translate()")


# ---------------------------------------------------------------------------
# 6. Attention Matrix Visualization (provided — complete implementation)
# ---------------------------------------------------------------------------

def plot_attention_matrix(
    attention_weights: np.ndarray,
    src_tokens: List[str],
    tgt_tokens: List[str],
    title: str = "Attention Matrix",
) -> None:
    """
    Plot the attention weight matrix as a heatmap.

    Args:
        attention_weights: 2D array of shape (tgt_len, src_len).
        src_tokens:        List of source tokens (x-axis).
        tgt_tokens:        List of target tokens (y-axis).
        title:             Plot title.
    """
    fig, ax = plt.subplots(figsize=(max(6, len(src_tokens) * 0.8),
                                    max(4, len(tgt_tokens) * 0.7)))
    im = ax.imshow(attention_weights, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(src_tokens)))
    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_xticklabels(src_tokens, rotation=45, ha="right")
    ax.set_yticklabels(tgt_tokens)
    ax.set_xlabel("Source Tokens")
    ax.set_ylabel("Target Tokens")
    ax.set_title(title)

    # Annotate cells
    for i in range(len(tgt_tokens)):
        for j in range(len(src_tokens)):
            if i < attention_weights.shape[0] and j < attention_weights.shape[1]:
                ax.text(j, i, f"{attention_weights[i, j]:.2f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if attention_weights[i, j] > 0.5 else "black")

    plt.tight_layout()
    filename = title.lower().replace(" ", "_") + ".png"
    plt.savefig(filename, dpi=100)
    plt.show()
    print(f"[Plot saved: {filename}]")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_seq2seq(
    model: Seq2SeqAttention,
    src_data: torch.Tensor,
    tgt_data: torch.Tensor,
    n_epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> List[float]:
    """
    Train the Seq2Seq model.

    This function is fully implemented — study it to understand the training loop.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore PAD index

    dataset = TensorDataset(src_data, tgt_data)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for src_batch, tgt_batch in loader:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            optimizer.zero_grad()
            # teacher_forcing_ratio decreases over training
            tf_ratio = max(0.3, 1.0 - epoch / n_epochs)
            all_logits, _ = model(src_batch, tgt_batch, teacher_forcing_ratio=tf_ratio)

            # all_logits: (batch, tgt_len-1, vocab_size)
            # tgt_batch[:, 1:]: (batch, tgt_len-1)  ← skip SOS, predict from pos 1
            logits_flat = all_logits.reshape(-1, all_logits.size(-1))
            targets_flat = tgt_batch[:, 1:].reshape(-1)
            loss = criterion(logits_flat, targets_flat)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        losses.append(avg_loss)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1:3d}/{n_epochs}]  Loss: {avg_loss:.4f}")

    return losses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Train Seq2Seq attention model on number-to-word translation.

    Steps:
        1. Generate toy dataset and build vocabularies.
        2. Encode sentences and prepare tensors.
        3. Build Encoder, BahdanauAttention, Decoder, Seq2SeqAttention.
        4. Train for several epochs.
        5. Translate sample sentences and visualize attention matrices.
    """
    torch.manual_seed(42)
    random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Data ---
    MAX_LEN = 7  # max sentence length (including EOS)
    src_sentences, tgt_sentences = build_toy_dataset(n_samples=2000, max_len=5)
    src_vocab, src_idx_to_word = build_vocab(src_sentences)
    tgt_vocab, tgt_idx_to_word = build_vocab(tgt_sentences)
    print(f"\nSrc vocab size: {len(src_vocab)}")
    print(f"Tgt vocab size: {len(tgt_vocab)}")

    # Encode all sentences
    src_encoded = [encode_sentence(s, src_vocab, add_eos=True, max_len=MAX_LEN) for s in src_sentences]
    tgt_encoded = [encode_sentence(t, tgt_vocab, add_sos=True, add_eos=True, max_len=MAX_LEN+1) for t in tgt_sentences]

    src_tensor = torch.LongTensor(src_encoded)
    tgt_tensor = torch.LongTensor(tgt_encoded)

    # --- Model ---
    EMBED_DIM  = 32
    HIDDEN_DIM = 64
    encoder = Encoder(len(src_vocab), EMBED_DIM, HIDDEN_DIM)
    decoder = Decoder(len(tgt_vocab), EMBED_DIM, HIDDEN_DIM)
    model   = Seq2SeqAttention(encoder, decoder)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # --- Train ---
    losses = train_seq2seq(model, src_tensor, tgt_tensor,
                            n_epochs=40, lr=1e-3, batch_size=64, device=device)

    # --- Translate and visualize attention ---
    test_cases = [
        ["1", "2", "3"],
        ["9", "0", "5", "3"],
        ["7", "7", "7"],
        ["4", "2"],
    ]

    print("\n--- Translations ---")
    for src in test_cases:
        translation, attn = translate(model, src, src_vocab, tgt_vocab, tgt_idx_to_word, device=device)
        print(f"  Source: {' '.join(src)}")
        print(f"  Predicted: {' '.join(translation)}")
        expected = [DIGIT_TO_WORD[int(d)] for d in src]
        print(f"  Expected:  {' '.join(expected)}")
        print()

        plot_attention_matrix(
            attn,
            src_tokens=src + [EOS],
            tgt_tokens=translation,
            title=f"Attention: {' '.join(src)} → {' '.join(translation)}",
        )

    # --- Training curve ---
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Seq2Seq Training Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("seq2seq_training_loss.png", dpi=100)
    plt.show()


if __name__ == "__main__":
    main()
