# Transformer Networks (Bộ biến đổi)

The Transformer (Vaswani et al., 2017 — "Attention Is All You Need") is the architecture underlying virtually all state-of-the-art NLP models (BERT, GPT, T5, LLaMA) and increasingly computer vision (ViT) and other domains. It replaces recurrence entirely with **self-attention**, enabling massive parallelism and superior modeling of long-range dependencies.

---

## 1. Motivation: Limitations of RNNs

| Problem | RNN Issue | Transformer Solution |
|---------|-----------|---------------------|
| Long-range dependencies | Vanishing gradients across many steps | Direct attention between any two positions |
| Parallelism | Sequential computation — step t depends on t-1 | Fully parallel — all positions processed simultaneously |
| Training speed | O(T) sequential operations | O(1) sequential operations (per layer) |
| Scalability | Hard to scale to very long sequences | Scales to 100k+ tokens with modern techniques |

The key insight: **every output position can directly attend to every input position** without passing through intermediate states.

---

## 2. Self-Attention Mechanism

### Query, Key, Value Intuition

Attention can be thought of as a soft database lookup:
- **Query (Q)**: "What am I looking for?" — the question asked by the current position.
- **Key (K)**: "What do I offer?" — a description of what each position contains.
- **Value (V)**: "What do I actually contain?" — the content to be retrieved.

The attention score between position `i` (query) and position `j` (key) determines how much position `i` borrows from position `j`'s value.

### Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax( Q K^T / √d_k ) V
```

Where:
- `Q`: Query matrix, shape `(n, d_k)` — n = sequence length, d_k = key dimension
- `K`: Key matrix, shape `(n, d_k)`
- `V`: Value matrix, shape `(n, d_v)`
- `d_k`: Scaling factor — prevents dot products from growing too large in magnitude

### Why Scale by √d_k?

The dot product `Q K^T` has variance proportional to `d_k`. Without scaling, for large `d_k`, dot products become very large, pushing softmax into regions with near-zero gradients. Dividing by `√d_k` keeps the variance at approximately 1, giving softmax well-behaved gradients.

### Step-by-Step Example

For input `"The cat sat"` (3 tokens):

```
Step 1: Project input embeddings to Q, K, V
        Q = X W_Q     (X: input, W_Q: learned projection)
        K = X W_K
        V = X W_V

Step 2: Compute attention scores
        scores = Q K^T / √d_k    shape: (3, 3)
        scores[i, j] = how much token i attends to token j

Step 3: Softmax (per row)
        attention_weights = softmax(scores, dim=-1)   shape: (3, 3)

Step 4: Weighted sum of values
        output = attention_weights @ V    shape: (3, d_v)
```

Each output token is a weighted sum of all value vectors — the weights reflect semantic relevance.

---

## 3. Multi-Head Attention

A single attention head can only capture one type of relationship at a time (e.g., syntactic agreement). **Multi-head attention** runs `h` attention operations in parallel with different learned projections:

```
head_i = Attention(Q W_Qi, K W_Ki, V W_Vi)
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O
```

Where `W_Qi`, `W_Ki`, `W_Vi` are learned projections unique to each head, and `W_O` combines all heads.

**Why multiple heads?**
- Head 1 might learn syntactic dependencies (subject-verb agreement).
- Head 2 might capture coreference (pronoun → noun).
- Head 3 might learn positional proximity.
- Different heads specialize in different types of relationships simultaneously.

**Computational note**: Each head uses `d_model / h` dimensions, so total cost is the same as one full-dimensional head.

---

## 4. Positional Encoding

Self-attention is **permutation invariant** — it treats the input as a set, not a sequence. If you shuffle the words "dog bites man" to "man bites dog", the attention mechanism would give identical attention patterns (without positional encoding). This is clearly wrong for language.

**Positional encoding** adds position information to the input embeddings:

```
Input to Transformer = Token Embedding + Positional Encoding
```

### Sinusoidal Encoding (original Transformer)

```
PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
```

Where:
- `pos`: Position in the sequence (0, 1, 2, ...)
- `i`: Dimension index (0, 1, ..., d_model/2 - 1)
- `d_model`: Embedding dimension

Properties:
- Different positions get unique encodings.
- The dot product of PE(pos) and PE(pos + k) is a function of k only — the model can learn relative positions.
- Works for sequences longer than those seen during training (extrapolation).

### Learned Positional Encoding (BERT, GPT)

Simply an `nn.Embedding(max_seq_len, d_model)` trained alongside the model. More flexible but cannot extrapolate beyond `max_seq_len`.

---

## 5. Transformer Block

Each Transformer block (also called a "layer") has two sub-layers:

```
Block Input: x
       │
       ├──────────────────────┐
       ▼                      │
 Multi-Head Attention         │  (residual)
       │                      │
       ▼                      │
 Layer Norm(x + Attention(x))─┘
       │
       ├──────────────────────┐
       ▼                      │
 Feed-Forward Network         │  (residual)
       │                      │
       ▼                      │
 Layer Norm(x + FFN(x))───────┘
       │
Block Output
```

### Layer Normalization

```
LayerNorm(x) = γ · (x - μ) / √(σ² + ε) + β
```

Where `μ` and `σ²` are computed across the **feature dimension** (unlike BatchNorm which normalizes across the batch). This is more stable for variable-length sequences and small batches.

### Feed-Forward Network (FFN)

```
FFN(x) = ReLU(x W_1 + b_1) W_2 + b_2
```

Typically: `d_model → d_ff → d_model` where `d_ff = 4 × d_model`.
Applies the same transformation independently to each position — adds non-linearity and capacity.

### Residual Connections

Each sub-layer has a skip connection: `output = LayerNorm(x + Sublayer(x))`. This:
- Allows gradients to flow directly to earlier layers.
- Makes it easy for layers to learn "identity + small correction" rather than a full transformation.
- Enables training of very deep networks (dozens of layers).

---

## 6. Encoder-Decoder Architecture

The full Transformer has:

**Encoder** (processes input):
- Stack of N identical blocks, each with Self-Attention + FFN.
- All positions attend to all other positions — fully bidirectional.

**Decoder** (generates output):
- Stack of N identical blocks with three sub-layers:
  1. **Masked Self-Attention**: Attends only to previous output positions (causal mask).
  2. **Cross-Attention**: Queries from decoder, Keys/Values from encoder output.
  3. **FFN**: Position-wise feed-forward.

**Causal Masking**: In the decoder, future positions are masked with `-∞` before softmax, ensuring the model cannot "peek" at future tokens during generation.

---

## 7. BERT and GPT

### BERT (Encoder-Only)

BERT (Bidirectional Encoder Representations from Transformers) uses only the Transformer encoder.

- **Pretraining**: Two tasks:
  1. **Masked Language Modeling (MLM)**: 15% of tokens masked with `[MASK]`; model predicts original tokens.
  2. **Next Sentence Prediction (NSP)**: Predict whether sentence B follows sentence A.
- **Fine-tuning**: Add a task-specific head and fine-tune all weights on labeled data.
- **Bidirectional**: Each token sees the full context (past and future).
- **Best for**: Classification, NER, question answering — tasks requiring full context.

### GPT (Decoder-Only)

GPT (Generative Pretrained Transformer) uses only the causal Transformer decoder.

- **Pretraining**: Autoregressive language modeling — predict the next token given all previous tokens.
- **Causal masking**: Each token can only attend to itself and previous tokens.
- **Generation**: Sample or greedy-decode one token at a time.
- **Best for**: Text generation, code generation, instruction following.

### Summary

| Model | Architecture | Training | Strengths |
|-------|-------------|----------|-----------|
| BERT | Encoder-only | MLM, NSP | Understanding tasks (classification, QA) |
| GPT | Decoder-only | Autoregressive LM | Generation tasks (completion, dialogue) |
| T5 | Full Enc-Dec | Span prediction | Translation, summarization |

---

## 8. Reference Links

- [Attention Is All You Need (original paper)](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer — Jay Alammar](https://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer — Harvard NLP](https://nlp.seas.harvard.edu/annotated-transformer/)
