# Recurrent Neural Networks (Mạng truy hồi)

Recurrent Neural Networks (RNNs) and their variants (LSTM, GRU) are designed for **sequential data** — data where the order of elements matters. Unlike feedforward networks, they maintain an internal **memory state** that persists across time steps.

---

## 1. Why Sequential Models?

Many real-world problems involve sequences where context from earlier positions affects the interpretation of later ones:

| Domain | Sequential Data | Example Task |
|--------|----------------|--------------|
| NLP | Text, sentences | Sentiment analysis, machine translation |
| Time series | Stock prices, sensor readings | Forecasting, anomaly detection |
| Audio | Speech waveforms | Speech recognition, music generation |
| Video | Frame sequences | Action recognition, video captioning |
| Biology | DNA sequences | Protein structure prediction |

A feedforward network treats each input independently — it has no sense of "earlier" or "later". RNNs solve this by passing information forward through time via a **hidden state**.

---

## 2. Vanilla RNN

### The Recurrence Formula

At each time step `t`, the RNN computes:

```
h_t = tanh(W_h · h_{t-1} + W_x · x_t + b)
y_t = W_y · h_t + b_y
```

Where:
- `x_t`: Input vector at time step `t`
- `h_t`: Hidden state at time `t` (the "memory")
- `h_{t-1}`: Hidden state from the previous time step
- `W_h`: Weight matrix for the recurrent connection
- `W_x`: Weight matrix for the input
- `W_y`: Weight matrix for the output

### Unrolling Through Time

An RNN can be visualized by "unrolling" it:

```
x_1    x_2    x_3    ...    x_T
 │      │      │              │
 ▼      ▼      ▼              ▼
[RNN]→[RNN]→[RNN]→  ...  →[RNN]
 │      │      │              │
 ▼      ▼      ▼              ▼
 y_1    y_2    y_3           y_T

→ h_0 → h_1 → h_2 → h_3 →  ...  → h_T
```

The same weight matrices `W_h`, `W_x`, `W_y` are **shared across all time steps** — this is what makes it a "recurrent" structure.

### Backpropagation Through Time (BPTT)

Training RNNs requires computing gradients through the unrolled graph. The gradient of the loss with respect to `W_h` requires multiplying the recurrent weight matrix many times:

```
∂L/∂W_h = Σ_t (∂L/∂h_t) · (∂h_t/∂W_h)
```

Each term in the sum involves a product of `(t - k)` Jacobian matrices — leading to the vanishing gradient problem for long sequences.

---

## 3. Vanishing Gradient in RNNs

The gradient of the hidden state at time `T` with respect to `h_t` involves:

```
∂h_T/∂h_t = Π_{k=t+1}^{T} (W_h · diag(tanh'(h_k)))
```

If the largest eigenvalue of `W_h · diag(tanh'(·))` is less than 1, the gradient shrinks exponentially with `(T - t)`. For long sequences (e.g., T = 100), gradients from early time steps essentially vanish — the RNN cannot learn long-term dependencies.

**Practical implication**: Vanilla RNNs work for short sequences (< ~10 steps) but fail to capture dependencies spanning 50+ time steps.

---

## 4. Long Short-Term Memory (LSTM)

LSTMs (Hochreiter & Schmidhuber, 1997) solve the vanishing gradient problem with a **cell state** `C_t` — a protected memory pathway — controlled by three **gates**.

### The Cell State

The cell state `C_t` carries information through time with only linear interactions (addition), unlike the hidden state which passes through tanh. This allows gradients to flow without shrinking.

### LSTM Equations

**Forget Gate** (what to erase from cell state):
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
```

**Input Gate** (what new information to store):
```
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
```

**Cell State Update**:
```
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
```
(⊙ = element-wise product)

**Output Gate** (what to expose as hidden state):
```
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
h_t = o_t ⊙ tanh(C_t)
```

### Gate Intuition

| Gate | Role | Example (language model) |
|------|------|--------------------------|
| **Forget** `f_t` | Erase irrelevant past information | Forget previous subject when starting new sentence |
| **Input** `i_t` | Decide what new info to store | Remember current subject's gender for pronoun agreement |
| **Output** `o_t` | Decide what to expose as output | Only expose relevant memory for the current prediction |

The key insight: the cell state `C_t` flows through gates with only element-wise operations and addition — no matrix multiplication — so gradients flow back cleanly.

---

## 5. Gated Recurrent Unit (GRU)

The GRU (Cho et al., 2014) is a simplified version of LSTM with fewer parameters but comparable performance.

### GRU Equations

**Reset Gate** (how much past to forget):
```
r_t = σ(W_r · [h_{t-1}, x_t])
```

**Update Gate** (interpolate between past and new):
```
z_t = σ(W_z · [h_{t-1}, x_t])
```

**Candidate Hidden State**:
```
h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t])
```

**New Hidden State**:
```
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

### LSTM vs GRU

| Property | LSTM | GRU |
|----------|------|-----|
| Gates | 3 (forget, input, output) | 2 (reset, update) |
| Memory | Cell state + hidden state | Hidden state only |
| Parameters | More | Fewer (~25% less) |
| Performance | Slightly better on some tasks | Faster to train |
| Best for | Very long sequences | Shorter sequences, limited data |

**Rule of thumb**: Start with GRU for fast experimentation; use LSTM for tasks requiring very long-range memory.

---

## 6. Bidirectional RNNs

A standard RNN processes sequences left to right. A **Bidirectional RNN** processes the sequence in both directions:

```
Forward:  x_1 → x_2 → x_3 → ... → x_T   →  h_T_forward
Backward: x_T → x_{T-1} → ... → x_1     →  h_1_backward
```

At each time step, the outputs from both directions are concatenated:
```
h_t = [h_t_forward ; h_t_backward]
```

**Use cases**: Any task where future context helps understand the current token.
- Named entity recognition (knowing what comes after helps).
- Masked language modeling (BERT uses bidirectional attention).
- Machine translation encoders.

**Limitation**: Cannot be used for online/streaming prediction (requires the full sequence in advance).

---

## 7. Sequence-to-Sequence (Encoder-Decoder)

Seq2Seq architecture maps a variable-length input sequence to a variable-length output sequence.

```
Encoder:                              Decoder:
x_1 → x_2 → ... → x_n               <SOS> → y_1 → y_2 → ... → y_m
  │      │           │                  │       │      │
[RNN] → [RNN] → [RNN]  → context c → [RNN] → [RNN] → [RNN]
                                              y_1     y_2   y_m
```

The encoder compresses the input sequence into a **context vector** `c = h_n`.
The decoder generates the output sequence conditioned on `c`.

### Applications
- Machine translation (French → English)
- Text summarization
- Speech recognition
- Code generation

### Teacher Forcing

During training, the decoder receives the **ground-truth** previous token as input (not its own prediction). This speeds up convergence but creates a gap between training and inference behavior (exposure bias).

---

## 8. Reference Links

- [Understanding LSTMs — Christopher Olah's Blog](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [PyTorch LSTM Documentation](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
