"""
Exercise 02: LSTM Sentiment Classification with PyTorch
========================================================
Build an LSTM-based text classifier using PyTorch's nn.LSTM module.
Apply it to a synthetic sentiment classification task.

Gate reminder:
    Forget gate (f_t): Decides what to ERASE from cell state.
                       "How much of my past should I keep?"
    Input gate  (i_t): Decides what NEW information to STORE.
                       "What new fact should I remember?"
    Output gate (o_t): Decides what to EXPOSE from cell state as hidden state.
                       "What part of my memory is relevant right now?"
    Cell state  (C_t): The long-term memory — flows with only linear ops.
    Hidden state(h_t): The working memory — filtered version of C_t.

Learning goals:
- Use nn.Embedding + nn.LSTM + nn.Linear in PyTorch.
- Handle variable-length sequences with padding.
- Understand the shapes of LSTM outputs.

References:
    - PyTorch LSTM: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
    - PyTorch Embedding: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# LSTM Classifier Model
# ---------------------------------------------------------------------------

class LSTMClassifier(nn.Module):
    """
    LSTM-based sequence classifier.

    Architecture:
        Embedding layer       → converts token IDs to dense vectors
        LSTM (possibly multi-layer, with dropout between layers)
        Final hidden state    → linear classification head

    Args:
        vocab_size (int): Number of unique tokens in vocabulary.
        embed_dim (int): Dimensionality of token embeddings.
        hidden_size (int): LSTM hidden state size.
        n_layers (int): Number of stacked LSTM layers.
        n_classes (int): Number of output classes.
        dropout (float): Dropout probability (applied between LSTM layers).
    """

    def __init__(self, vocab_size, embed_dim, hidden_size, n_layers, n_classes,
                 dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers    = n_layers

        # Embedding layer: maps integer token IDs to dense vectors
        # padding_idx=0: the padding token (index 0) has a zero embedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0
        )

        # LSTM layer
        # Input shape:  (seq_len, batch, embed_dim)  [batch_first=False]
        # Output shape: (seq_len, batch, hidden_size)
        # h_n shape:    (n_layers, batch, hidden_size) — final hidden state
        # c_n shape:    (n_layers, batch, hidden_size) — final cell state

        # TODO:
        #   Define self.lstm = nn.LSTM(
        #       input_size=embed_dim,
        #       hidden_size=hidden_size,
        #       num_layers=n_layers,
        #       dropout=dropout if n_layers > 1 else 0,
        #       batch_first=True,     ← input shape: (batch, seq_len, embed_dim)
        #       bidirectional=False
        #   )
        raise NotImplementedError("Define self.lstm in LSTMClassifier.__init__")

        # Dropout for regularization after LSTM
        self.dropout = nn.Dropout(dropout)

        # TODO:
        #   Define self.fc = nn.Linear(hidden_size, n_classes)
        #   (If bidirectional=True, use hidden_size * 2 as input to fc)
        raise NotImplementedError("Define self.fc in LSTMClassifier.__init__")

    def forward(self, x):
        """
        Forward pass for a batch of tokenized sequences.

        Args:
            x (torch.Tensor): Token IDs, shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Class logits, shape (batch_size, n_classes).

        TODO:
            1. Embed inputs:
                   embedded = self.embedding(x)   # (batch, seq_len, embed_dim)
            2. Run through LSTM:
                   output, (h_n, c_n) = self.lstm(embedded)
                   # output: (batch, seq_len, hidden_size)  — all time steps
                   # h_n:    (n_layers, batch, hidden_size) — final hidden state per layer
                   # c_n:    (n_layers, batch, hidden_size) — final cell state per layer
            3. Extract the last layer's hidden state:
                   h_last = h_n[-1]   # shape: (batch, hidden_size)
            4. Apply dropout:
                   h_last = self.dropout(h_last)
            5. Pass through linear layer:
                   logits = self.fc(h_last)
            6. Return logits.

        Note: We use h_n[-1] (final hidden state of last layer) for classification.
              Alternatively, you could use output[:, -1, :] (last time step of output).
              For bidirectional, concatenate forward and backward final hidden states.
        """
        # TODO: implement forward pass
        raise NotImplementedError("Implement LSTMClassifier.forward(x)")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_lstm(model, train_loader, val_loader, n_epochs, device, lr=1e-3):
    """
    Train the LSTM classifier.

    Args:
        model (nn.Module): LSTMClassifier.
        train_loader (DataLoader): Training data.
        val_loader (DataLoader): Validation data.
        n_epochs (int): Number of training epochs.
        device: torch.device.
        lr (float): Learning rate.

    Returns:
        dict: Training history with 'train_loss', 'val_loss', 'train_acc', 'val_acc'.

    TODO:
        1. Define optimizer = optim.Adam(model.parameters(), lr=lr)
        2. Define criterion = nn.CrossEntropyLoss()
        3. For each epoch:
               a. Training loop (model.train()):
                  - Move X, y to device.
                  - optimizer.zero_grad()
                  - logits = model(X)
                  - loss = criterion(logits, y)
                  - loss.backward()
                  - torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    (gradient clipping — important for RNNs!)
                  - optimizer.step()
                  - Track loss and accuracy.
               b. Validation loop (model.eval(), torch.no_grad()).
               c. Print and store metrics.
        4. Return history dict.

    Note: Gradient clipping prevents exploding gradients in RNNs.
          The standard clip norm for LSTMs is 1.0 or 5.0.
    """
    # TODO: implement train_lstm
    raise NotImplementedError("Implement train_lstm(...)")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_sentiment(model, text_tokens, vocab, device):
    """
    Predict the sentiment of a single text sequence.

    Args:
        model (nn.Module): Trained LSTMClassifier.
        text_tokens (list[str]): Tokenized text as a list of words.
        vocab (dict): Word to index mapping. OOV words map to index 1 (<UNK>).
        device: torch.device.

    Returns:
        tuple[int, float]: (predicted_class, confidence_probability)

    TODO:
        1. Convert text_tokens to token IDs using vocab:
               ids = [vocab.get(w, 1) for w in text_tokens]  # 1 = <UNK>
        2. Convert to tensor and add batch dimension:
               x = torch.tensor([ids], dtype=torch.long).to(device)
        3. Set model.eval() and use torch.no_grad().
        4. Get logits from model(x), shape (1, n_classes).
        5. Apply softmax to get probabilities.
        6. Return (argmax class, max probability).
    """
    # TODO: implement predict_sentiment
    raise NotImplementedError("Implement predict_sentiment(...)")


# ---------------------------------------------------------------------------
# Synthetic Dataset
# ---------------------------------------------------------------------------

class SentimentDataset(Dataset):
    """
    Synthetic sentiment dataset with simple positive/negative templates.

    Vocabulary:
        0: <PAD>, 1: <UNK>
        Positive words → higher indices
        Negative words → lower-middle indices

    Each sample is a padded sequence of token IDs.
    """

    POSITIVE_WORDS = ["great", "excellent", "amazing", "wonderful", "fantastic",
                      "love", "best", "brilliant", "happy", "good"]
    NEGATIVE_WORDS = ["terrible", "awful", "horrible", "bad", "worst",
                      "hate", "dreadful", "poor", "disappointing", "sad"]
    NEUTRAL_WORDS  = ["the", "a", "is", "was", "movie", "film", "book", "it"]

    def __init__(self, n_samples=2000, seq_len=20, seed=42):
        np.random.seed(seed)
        self.seq_len  = seq_len

        # Build vocabulary
        all_words   = self.POSITIVE_WORDS + self.NEGATIVE_WORDS + self.NEUTRAL_WORDS
        self.vocab  = {"<PAD>": 0, "<UNK>": 1}
        for w in all_words:
            self.vocab[w] = len(self.vocab)
        self.vocab_size = len(self.vocab)

        # Generate samples
        self.sequences = []
        self.labels    = []

        for i in range(n_samples):
            label = i % 2  # alternating positive/negative
            seq_words = []
            if label == 1:
                # Mostly positive words + some neutral
                pos_count = np.random.randint(3, 7)
                words     = (np.random.choice(self.POSITIVE_WORDS, pos_count).tolist()
                             + np.random.choice(self.NEUTRAL_WORDS, seq_len - pos_count).tolist())
            else:
                # Mostly negative words + some neutral
                neg_count = np.random.randint(3, 7)
                words     = (np.random.choice(self.NEGATIVE_WORDS, neg_count).tolist()
                             + np.random.choice(self.NEUTRAL_WORDS, seq_len - neg_count).tolist())

            np.random.shuffle(words)
            ids = [self.vocab.get(w, 1) for w in words[:seq_len]]
            # Pad to seq_len
            ids += [0] * (seq_len - len(ids))

            self.sequences.append(ids)
            self.labels.append(label)

        self.sequences = torch.tensor(self.sequences, dtype=torch.long)
        self.labels    = torch.tensor(self.labels,    dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset = SentimentDataset(n_samples=2000, seq_len=20)
    print(f"Vocabulary size: {dataset.vocab_size}")
    print(f"Dataset size:    {len(dataset)} samples")

    n_train = int(0.8 * len(dataset))
    n_val   = len(dataset) - n_train
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)

    # ------------------------------------------------------------------
    # Build LSTM model
    # ------------------------------------------------------------------
    model = LSTMClassifier(
        vocab_size   = dataset.vocab_size,
        embed_dim    = 32,
        hidden_size  = 64,
        n_layers     = 2,
        n_classes    = 2,
        dropout      = 0.3
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    history = train_lstm(
        model, train_loader, val_loader,
        n_epochs=20, device=device, lr=1e-3
    )

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'],   label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Loss')

    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'],   label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.set_title('Accuracy')

    plt.suptitle("LSTM Sentiment Classifier Training")
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # Inference demo
    # ------------------------------------------------------------------
    print("\nSentiment Prediction Examples:")
    test_sentences = [
        (["this", "movie", "was", "great", "amazing", "love", "it"], "Positive"),
        (["terrible", "film", "worst", "experience", "awful"],        "Negative"),
        (["the", "book", "was", "a", "movie"],                        "Neutral context"),
    ]

    for tokens, true_label in test_sentences:
        pred_class, confidence = predict_sentiment(model, tokens, dataset.vocab, device)
        label_str = "POSITIVE" if pred_class == 1 else "NEGATIVE"
        print(f"  Tokens: {tokens}")
        print(f"  → Predicted: {label_str} (conf={confidence:.2%})  | Expected: {true_label}\n")


if __name__ == "__main__":
    main()
