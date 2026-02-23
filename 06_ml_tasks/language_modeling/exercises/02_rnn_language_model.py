"""
Exercise 02: RNN Character-Level Language Model
================================================
Build a character-level language model using a PyTorch LSTM.
The model learns to predict the next character given the previous
characters, and can then generate new text character by character.

Learning Goals:
    - Implement a character tokenizer (vocabulary, encode, decode)
    - Build input/target sequence pairs with a sliding window
    - Define and train an LSTM language model
    - Sample text at different temperatures to explore diversity vs. quality
    - Compute perplexity on held-out text

Requirements:
    pip install torch numpy
"""

import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Sample text — replace with a larger corpus for better results
# ---------------------------------------------------------------------------
SAMPLE_TEXT = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take Arms against a Sea of troubles,
And by opposing end them: to die, to sleep
No more; and by a sleep, to say we end
The heart-ache, and the thousand Natural shocks
That Flesh is heir to? 'Tis a consummation
Devoutly to be wished. To die, to sleep,
To sleep, perchance to Dream; aye, there's the rub,
For in that sleep of death, what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause.
""" * 5  # Repeat to give the model more data


# ---------------------------------------------------------------------------
# 1. Character Tokenizer
# ---------------------------------------------------------------------------

class CharTokenizer:
    """
    Maps characters to integer indices and back.

    Attributes:
        char_to_idx: Dict mapping character → integer index.
        idx_to_char: Dict mapping integer index → character.
        vocab_size:  Total number of unique characters.
    """

    def __init__(self) -> None:
        self.char_to_idx: Dict[str, int] = {}
        self.idx_to_char: Dict[int, str] = {}
        self.vocab_size: int = 0

    def fit(self, text: str) -> "CharTokenizer":
        """
        Build the character vocabulary from the given text.

        Args:
            text: Raw string to extract vocabulary from.

        Returns:
            self (for method chaining).

        TODO:
            1. Get the sorted set of unique characters in text.
            2. Build char_to_idx: {char: i for i, char in enumerate(sorted_chars)}.
            3. Build idx_to_char: {i: char for char, i in char_to_idx.items()}.
            4. Set self.vocab_size = len(char_to_idx).
            5. Return self.
        """
        raise NotImplementedError("TODO: implement CharTokenizer.fit()")

    def encode(self, text: str) -> List[int]:
        """
        Convert a string into a list of integer indices.

        Args:
            text: Input string.

        Returns:
            List of integer indices, one per character.

        TODO:
            1. For each character in text, look up char_to_idx.
            2. Skip characters not in the vocabulary (or raise an error).
            3. Return the list of indices.
        """
        raise NotImplementedError("TODO: implement CharTokenizer.encode()")

    def decode(self, indices: List[int]) -> str:
        """
        Convert a list of integer indices back to a string.

        Args:
            indices: List of integer indices.

        Returns:
            Decoded string.

        TODO:
            1. For each index, look up idx_to_char.
            2. Join characters into a string.
            3. Return the result.
        """
        raise NotImplementedError("TODO: implement CharTokenizer.decode()")


# ---------------------------------------------------------------------------
# 2. Sequence Creation
# ---------------------------------------------------------------------------

def create_sequences(
    encoded_text: List[int], seq_len: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create input/target pairs using a sliding window.

    For each position i, the input is encoded_text[i : i+seq_len]
    and the target is encoded_text[i+1 : i+seq_len+1].
    The target is the input shifted by one position to the right.

    Args:
        encoded_text: List of integer-encoded characters.
        seq_len:      Length of each input sequence.

    Returns:
        X: LongTensor of shape (num_sequences, seq_len) — inputs.
        y: LongTensor of shape (num_sequences, seq_len) — targets (shifted by 1).

    TODO:
        1. Compute num_sequences = len(encoded_text) - seq_len.
        2. Build lists X_list and y_list:
             for i in range(num_sequences):
                 X_list.append(encoded_text[i : i + seq_len])
                 y_list.append(encoded_text[i+1 : i + seq_len + 1])
        3. Convert to torch.LongTensor.
        4. Return (X, y).
    """
    raise NotImplementedError("TODO: implement create_sequences()")


# ---------------------------------------------------------------------------
# 3. Model Definition
# ---------------------------------------------------------------------------

class CharRNNModel(nn.Module):
    """
    Character-level language model using LSTM.

    Architecture:
        Embedding → LSTM → Linear → (logits over vocabulary)

    Args:
        vocab_size:   Number of unique characters.
        embed_dim:    Dimension of character embeddings.
        hidden_dim:   Number of LSTM hidden units.
        num_layers:   Number of LSTM layers.
        dropout:      Dropout probability between LSTM layers.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        # TODO:
        #   1. Define self.embedding = nn.Embedding(vocab_size, embed_dim).
        #   2. Define self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
        #                                  batch_first=True, dropout=dropout).
        #   3. Define self.fc = nn.Linear(hidden_dim, vocab_size).
        #   4. Store hidden_dim, num_layers, vocab_size as attributes.
        raise NotImplementedError("TODO: implement CharRNNModel.__init__()")

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x:      LongTensor of shape (batch, seq_len) — input token indices.
            hidden: Optional initial hidden state tuple (h_0, c_0).

        Returns:
            logits: FloatTensor of shape (batch, seq_len, vocab_size).
            hidden: Updated hidden state tuple.

        TODO:
            1. Pass x through self.embedding → shape (batch, seq_len, embed_dim).
            2. Pass embeddings and hidden through self.lstm
               → output shape (batch, seq_len, hidden_dim), new_hidden.
            3. Pass lstm output through self.fc → logits of shape (batch, seq_len, vocab_size).
            4. Return (logits, new_hidden).
        """
        raise NotImplementedError("TODO: implement CharRNNModel.forward()")

    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden state and cell state to zeros.

        Args:
            batch_size: Number of sequences in the batch.
            device:     Device to create tensors on.

        Returns:
            Tuple (h_0, c_0) each of shape (num_layers, batch_size, hidden_dim).
        """
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h0, c0)


# ---------------------------------------------------------------------------
# 4. Training
# ---------------------------------------------------------------------------

def train_rnn_lm(
    model: CharRNNModel,
    X: torch.Tensor,
    y: torch.Tensor,
    n_epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> List[float]:
    """
    Train the character RNN language model.

    Args:
        model:      The CharRNNModel to train.
        X:          Input tensor of shape (num_sequences, seq_len).
        y:          Target tensor of shape (num_sequences, seq_len).
        n_epochs:   Number of training epochs.
        lr:         Learning rate for Adam optimizer.
        batch_size: Mini-batch size.
        device:     Torch device (cpu or cuda). Defaults to auto-detect.

    Returns:
        List of average training losses per epoch.

    TODO:
        1. Set device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
           if not provided. Move model to device.
        2. Create a DataLoader from TensorDataset(X, y) with batch_size and shuffle=True.
        3. Define optimizer = torch.optim.Adam(model.parameters(), lr=lr).
        4. Define criterion = nn.CrossEntropyLoss().
        5. For each epoch:
             a. Set model.train().
             b. For each batch (x_batch, y_batch):
                  - Move batch to device.
                  - Initialize hidden = model.init_hidden(x_batch.size(0), device).
                  - Forward pass: logits, hidden = model(x_batch, hidden).
                  - Detach hidden to prevent BPTT through the full sequence history.
                  - Reshape logits: (batch * seq_len, vocab_size).
                  - Reshape y_batch: (batch * seq_len,).
                  - Compute loss = criterion(reshaped_logits, reshaped_y_batch).
                  - Backward + optimizer step.
             c. Record and print average epoch loss.
        6. Return the list of epoch losses.

    Hint:
        - Detach hidden: hidden = tuple(h.detach() for h in hidden)
        - torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) helps stability.
    """
    raise NotImplementedError("TODO: implement train_rnn_lm()")


# ---------------------------------------------------------------------------
# 5. Text Generation
# ---------------------------------------------------------------------------

def generate_text_rnn(
    model: CharRNNModel,
    tokenizer: CharTokenizer,
    seed: str,
    max_len: int = 200,
    temperature: float = 1.0,
    device: Optional[torch.device] = None,
) -> str:
    """
    Generate text autoregressively using the trained model.

    Args:
        model:       Trained CharRNNModel.
        tokenizer:   The CharTokenizer used during training.
        seed:        Seed string to start generation from.
        max_len:     Number of characters to generate (not including seed).
        temperature: Sampling temperature.
                     - temperature < 1.0 → more deterministic
                     - temperature > 1.0 → more random
        device:      Torch device.

    Returns:
        Seed + generated characters as a single string.

    TODO:
        1. Set model to eval mode. Set device.
        2. Encode the seed string using tokenizer.encode().
        3. Initialize hidden = model.init_hidden(1, device).
        4. Feed the seed tokens one-by-one through the model to warm up the hidden state.
           (Keep only the last output's logits for sampling.)
        5. Sample up to max_len characters:
             a. Get logits for the current input character.
             b. Apply temperature: logits = logits / temperature.
             c. Convert to probabilities: probs = F.softmax(logits, dim=-1).
             d. Sample next character index: torch.multinomial(probs, num_samples=1).
             e. Decode the index to a character and append to generated list.
             f. Use the sampled index as the next input.
        6. Return seed + ''.join(generated_chars).

    Hint:
        - Keep track of the hidden state across steps so the model remembers context.
        - torch.multinomial(probs.squeeze(), num_samples=1).item() gives a scalar index.
    """
    raise NotImplementedError("TODO: implement generate_text_rnn()")


# ---------------------------------------------------------------------------
# 6. Perplexity
# ---------------------------------------------------------------------------

def compute_perplexity_rnn(
    model: CharRNNModel,
    tokenizer: CharTokenizer,
    test_text: str,
    seq_len: int = 100,
    device: Optional[torch.device] = None,
) -> float:
    """
    Compute perplexity of the model on a test string.

    Args:
        model:     Trained CharRNNModel.
        tokenizer: The CharTokenizer used during training.
        test_text: Text to evaluate perplexity on.
        seq_len:   Sequence length for batching.
        device:    Torch device.

    Returns:
        Perplexity score (lower is better).

    TODO:
        1. Encode test_text using tokenizer.encode().
        2. Create sequences using create_sequences(encoded, seq_len).
        3. For each sequence (x, y):
             a. Forward pass through model to get logits.
             b. Compute cross-entropy loss (token-level average).
             c. Accumulate total log-likelihood.
        4. Compute perplexity = exp(total_loss / total_tokens).
        5. Return perplexity.

    Note:
        nn.CrossEntropyLoss() computes the mean loss per token by default,
        which equals the average negative log-likelihood per token.
        Perplexity = exp(average NLL per token).
    """
    raise NotImplementedError("TODO: implement compute_perplexity_rnn()")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Train a character RNN on the sample text and generate samples.

    Steps:
        1. Fit tokenizer on the corpus.
        2. Create training sequences.
        3. Train the model for several epochs.
        4. Generate text at temperatures 0.5, 1.0, and 1.5.
        5. Report perplexity on a held-out portion of the text.
    """
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Tokenizer ---
    tokenizer = CharTokenizer()
    tokenizer.fit(SAMPLE_TEXT)
    print(f"Vocabulary size: {tokenizer.vocab_size} characters")

    # --- Sequences ---
    SEQ_LEN = 50
    encoded = tokenizer.encode(SAMPLE_TEXT)
    X, y = create_sequences(encoded, SEQ_LEN)
    print(f"Training sequences: {X.shape[0]}")

    # --- Model ---
    model = CharRNNModel(
        vocab_size=tokenizer.vocab_size,
        embed_dim=32,
        hidden_dim=128,
        num_layers=2,
    )

    # --- Train ---
    losses = train_rnn_lm(model, X, y, n_epochs=30, lr=2e-3, batch_size=64, device=device)

    # --- Generate at different temperatures ---
    seed = "To be"
    for temp in [0.5, 1.0, 1.5]:
        print(f"\n--- Temperature = {temp} ---")
        print(generate_text_rnn(model, tokenizer, seed, max_len=150, temperature=temp, device=device))

    # --- Perplexity ---
    test_text = SAMPLE_TEXT[-200:]
    pp = compute_perplexity_rnn(model, tokenizer, test_text, seq_len=SEQ_LEN, device=device)
    print(f"\nTest perplexity: {pp:.2f}")


if __name__ == "__main__":
    main()
