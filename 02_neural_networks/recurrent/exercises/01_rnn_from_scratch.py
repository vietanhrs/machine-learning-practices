"""
Exercise 01: RNN from Scratch
==============================
Implement a Vanilla RNN using only NumPy.
This exercise builds intuition for the recurrence formula and
manual backpropagation through time.

Learning goals:
- Understand the h_t = tanh(W_h h_{t-1} + W_x x_t + b) update.
- See how hidden state evolves across time steps.
- Observe the RNN's inability to handle long-range dependencies.

Reference:
    - Karpathy "The Unreasonable Effectiveness of RNNs": http://karpathy.github.io/2015/05/21/rnn-effectiveness/
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Activation Function
# ---------------------------------------------------------------------------

def tanh_activation(z):
    """
    Hyperbolic tangent activation.

    tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))

    Args:
        z (np.ndarray): Input array, any shape.

    Returns:
        np.ndarray: tanh(z), values in (-1, 1).

    TODO:
        Implement tanh.
        Hint: np.tanh(z) is the standard library call,
              or you can implement from scratch using np.exp.
    """
    # TODO: implement tanh_activation
    raise NotImplementedError("Implement tanh_activation(z)")


# ---------------------------------------------------------------------------
# RNN Cell (Single Time Step)
# ---------------------------------------------------------------------------

class RNNCell:
    """
    A single Vanilla RNN cell computing one time step.

    Equations:
        h_t = tanh(W_h @ h_{t-1} + W_x @ x_t + b)

    Parameters are randomly initialized and not trained in this exercise
    (forward-only demonstration).

    Attributes:
        input_size (int): Dimensionality of input x_t.
        hidden_size (int): Dimensionality of hidden state h_t.
        W_h (np.ndarray): Recurrent weight matrix, shape (hidden_size, hidden_size).
        W_x (np.ndarray): Input weight matrix, shape (hidden_size, input_size).
        b (np.ndarray): Bias vector, shape (hidden_size,).
    """

    def __init__(self, input_size, hidden_size):
        """
        Initialize RNN cell weights using Xavier initialization.

        Args:
            input_size (int): Input dimensionality.
            hidden_size (int): Hidden state dimensionality.
        """
        self.input_size  = input_size
        self.hidden_size = hidden_size

        # Xavier initialization
        scale_h = np.sqrt(1.0 / hidden_size)
        scale_x = np.sqrt(1.0 / input_size)
        self.W_h = np.random.randn(hidden_size, hidden_size) * scale_h
        self.W_x = np.random.randn(hidden_size, input_size)  * scale_x
        self.b   = np.zeros(hidden_size)

    def forward(self, x_t, h_prev):
        """
        Compute one RNN time step.

        h_t = tanh(W_h @ h_prev + W_x @ x_t + b)

        Args:
            x_t (np.ndarray): Input at current time step, shape (input_size,).
            h_prev (np.ndarray): Hidden state from previous step, shape (hidden_size,).

        Returns:
            np.ndarray: New hidden state h_t, shape (hidden_size,).

        TODO:
            1. Compute the pre-activation: z = W_h @ h_prev + W_x @ x_t + self.b
            2. Apply tanh_activation to z.
            3. Return the new hidden state.
        """
        # TODO: implement RNNCell.forward
        raise NotImplementedError("Implement RNNCell.forward(x_t, h_prev)")


# ---------------------------------------------------------------------------
# RNN (Over Full Sequence)
# ---------------------------------------------------------------------------

class RNN:
    """
    A Vanilla RNN that processes an entire sequence using an RNNCell.

    Attributes:
        cell (RNNCell): The shared RNN cell applied at each time step.
    """

    def __init__(self, input_size, hidden_size):
        """
        Initialize the RNN.

        Args:
            input_size (int): Dimensionality of input at each time step.
            hidden_size (int): Dimensionality of hidden state.
        """
        self.cell = RNNCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, X):
        """
        Process a sequence of inputs.

        Args:
            X (np.ndarray): Input sequence, shape (seq_len, input_size).

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - all_hidden: All hidden states, shape (seq_len, hidden_size).
                - h_final: Final hidden state, shape (hidden_size,).

        TODO:
            1. Initialize h_prev = np.zeros(self.hidden_size).
            2. Loop over each time step t from 0 to seq_len-1:
                   h_t = self.cell.forward(X[t], h_prev)
                   Append h_t to a list.
                   h_prev = h_t
            3. Stack the list of hidden states into a numpy array.
            4. Return (all_hidden_states_array, h_prev).
        """
        # TODO: implement RNN.forward
        raise NotImplementedError("Implement RNN.forward(X)")


# ---------------------------------------------------------------------------
# Simple RNN Classifier
# ---------------------------------------------------------------------------

class SimpleRNNClassifier:
    """
    RNN + linear output layer for sequence classification.

    Architecture:
        RNN over input sequence → final hidden state → linear layer → output logits

    Attributes:
        rnn (RNN): The recurrent layer.
        W_out (np.ndarray): Output weight matrix, shape (n_classes, hidden_size).
        b_out (np.ndarray): Output bias, shape (n_classes,).
    """

    def __init__(self, input_size, hidden_size, n_classes):
        """
        Initialize classifier.

        Args:
            input_size (int): Input feature size at each time step.
            hidden_size (int): RNN hidden state size.
            n_classes (int): Number of output classes.
        """
        self.rnn = RNN(input_size, hidden_size)
        self.W_out = np.random.randn(n_classes, hidden_size) * 0.01
        self.b_out = np.zeros(n_classes)

    def forward(self, X):
        """
        Classify a single input sequence.

        Args:
            X (np.ndarray): Input sequence, shape (seq_len, input_size).

        Returns:
            np.ndarray: Class probabilities via softmax, shape (n_classes,).

        TODO:
            1. Run self.rnn.forward(X) to get all hidden states and h_final.
            2. Compute logits: z = W_out @ h_final + b_out
            3. Apply softmax: exp(z) / sum(exp(z))  (subtract max for stability)
            4. Return the probability vector.
        """
        # TODO: implement SimpleRNNClassifier.forward
        raise NotImplementedError("Implement SimpleRNNClassifier.forward(X)")

    def predict(self, X):
        """
        Return predicted class index.

        Args:
            X (np.ndarray): Input sequence, shape (seq_len, input_size).

        Returns:
            int: Predicted class index (argmax of probabilities).
        """
        probs = self.forward(X)
        return int(np.argmax(probs))


# ---------------------------------------------------------------------------
# Toy Dataset: Sequence Classification
# ---------------------------------------------------------------------------

def generate_sequences(n_samples=200, seq_len=10, input_size=4, n_classes=2, seed=42):
    """
    Generate synthetic sequences for a classification task.

    Class 0: sequences with higher values in the first half.
    Class 1: sequences with higher values in the second half.

    This tests whether the RNN can remember information from early time steps
    when the sequence is long.

    Args:
        n_samples (int): Number of sequences to generate.
        seq_len (int): Length of each sequence.
        input_size (int): Dimensionality of input at each step.
        n_classes (int): Number of classes.
        seed (int): Random seed.

    Returns:
        tuple[list[np.ndarray], np.ndarray]:
            - sequences: List of (seq_len, input_size) arrays.
            - labels: Integer class labels, shape (n_samples,).
    """
    np.random.seed(seed)
    sequences = []
    labels    = []

    for i in range(n_samples):
        label = i % n_classes
        seq = np.random.randn(seq_len, input_size) * 0.5

        if label == 0:
            # Stronger signal in first half
            seq[:seq_len // 2] += 1.0
        else:
            # Stronger signal in second half
            seq[seq_len // 2:] += 1.0

        sequences.append(seq)
        labels.append(label)

    return sequences, np.array(labels)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    np.random.seed(42)

    INPUT_SIZE  = 4
    HIDDEN_SIZE = 16
    N_CLASSES   = 2
    SEQ_LEN     = 10

    # ------------------------------------------------------------------
    # 1. Demonstrate the RNNCell forward pass
    # ------------------------------------------------------------------
    print("=" * 50)
    print("1. RNNCell Forward Pass")
    print("=" * 50)

    cell = RNNCell(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE)
    x_t = np.random.randn(INPUT_SIZE)
    h_prev = np.zeros(HIDDEN_SIZE)

    h_t = cell.forward(x_t, h_prev)
    print(f"  Input x_t shape:   {x_t.shape}")
    print(f"  h_prev shape:      {h_prev.shape}")
    print(f"  Output h_t shape:  {h_t.shape}")
    print(f"  h_t range:         [{h_t.min():.3f}, {h_t.max():.3f}]  (should be in (-1, 1))")

    # ------------------------------------------------------------------
    # 2. Demonstrate RNN over a sequence
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("2. RNN Over Full Sequence")
    print("=" * 50)

    rnn = RNN(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE)
    X_seq = np.random.randn(SEQ_LEN, INPUT_SIZE)

    all_hidden, h_final = rnn.forward(X_seq)
    print(f"  Sequence shape:        {X_seq.shape}")
    print(f"  All hidden states:     {all_hidden.shape}  (seq_len × hidden_size)")
    print(f"  Final hidden state:    {h_final.shape}")

    # Visualize how the hidden state evolves
    plt.figure(figsize=(10, 4))
    plt.imshow(all_hidden.T, aspect='auto', cmap='RdBu_r',
               vmin=-1, vmax=1)
    plt.colorbar(label='Activation')
    plt.xlabel('Time Step')
    plt.ylabel('Hidden Unit')
    plt.title('RNN Hidden State Evolution Over Time')
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # 3. Classification task
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("3. SimpleRNNClassifier on Toy Sequences")
    print("=" * 50)

    classifier = SimpleRNNClassifier(INPUT_SIZE, HIDDEN_SIZE, N_CLASSES)

    sequences, labels = generate_sequences(
        n_samples=200, seq_len=SEQ_LEN, input_size=INPUT_SIZE, n_classes=N_CLASSES
    )

    # Forward-only evaluation (no training — just checking the pipeline)
    preds = [classifier.predict(seq) for seq in sequences]
    preds = np.array(preds)
    acc   = np.mean(preds == labels)
    print(f"  (Untrained) Accuracy: {acc:.2%}  (should be ~50% — random weights)")

    # Evaluate a long-range memory challenge
    print("\n  Long-range memory challenge:")
    for length in [5, 10, 20, 50]:
        seqs_long, labels_long = generate_sequences(
            n_samples=100, seq_len=length,
            input_size=INPUT_SIZE, n_classes=N_CLASSES
        )
        rnn_long = RNN(INPUT_SIZE, HIDDEN_SIZE)
        # Simple test: does h_final differ between classes on average?
        h_class0 = np.mean([rnn_long.forward(s)[1] for s, l in
                            zip(seqs_long, labels_long) if l == 0], axis=0)
        h_class1 = np.mean([rnn_long.forward(s)[1] for s, l in
                            zip(seqs_long, labels_long) if l == 1], axis=0)
        diff = np.linalg.norm(h_class0 - h_class1)
        print(f"    Seq length={length:3d} | ||h_class0 - h_class1|| = {diff:.4f}")

    print("\nObservation: As sequence length grows, the difference in final")
    print("hidden states shrinks — the RNN 'forgets' information from early steps.")
    print("This motivates LSTM (see 02_lstm_pytorch.py).")


if __name__ == "__main__":
    main()
