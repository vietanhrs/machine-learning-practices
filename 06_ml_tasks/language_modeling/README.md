# Language Modeling

Language modeling is one of the most fundamental tasks in Natural Language Processing (NLP).
A language model assigns a probability to sequences of words, enabling machines to understand
and generate human language.

---

## 1. What is Language Modeling?

A **language model** is a probability distribution over sequences of words (or tokens).
Given a sequence of words `w_1, w_2, ..., w_n`, a language model estimates:

```
P(w_1, w_2, ..., w_n)
```

Using the chain rule of probability, this expands to:

```
P(w_1, w_2, ..., w_n) = P(w_1) * P(w_2 | w_1) * P(w_3 | w_1, w_2) * ... * P(w_n | w_1, ..., w_{n-1})
```

Language models are the backbone of text generation, speech recognition, machine translation,
and many other NLP applications.

---

## 2. N-gram Language Models

### Core Idea

An **N-gram** is a contiguous sequence of N words. N-gram models approximate the full
history of a sequence using only the last `n-1` words (the **Markov assumption**).

| Model    | Assumption                                 | Example                         |
|----------|--------------------------------------------|---------------------------------|
| Unigram  | Each word is independent                   | P(w)                            |
| Bigram   | Each word depends on the previous one      | P(w_i | w_{i-1})               |
| Trigram  | Each word depends on the previous two      | P(w_i | w_{i-2}, w_{i-1})     |

### The Markov Assumption

The Markov assumption states that the probability of a word depends only on a fixed window
of previous words, not the entire history:

```
P(w_n | w_1, ..., w_{n-1}) ≈ P(w_n | w_{n-k}, ..., w_{n-1})
```

**Limitation:** This approximation ignores long-range dependencies. For example, in the
sentence "The cat that sat on the mat was ___", a bigram model cannot capture that
"cat" is the subject that determines the verb.

### Estimating N-gram Probabilities

Probabilities are estimated from counts in a training corpus:

```
P(w_i | w_{i-1}) = count(w_{i-1}, w_i) / count(w_{i-1})
```

### Smoothing: Add-k Smoothing

**Problem:** If an n-gram never appears in training data, its probability is 0, which
makes the probability of any sentence containing it equal to 0.

**Solution — Add-k (Laplace) smoothing:** Add a small constant `k` to all counts:

```
P(w_i | w_{i-1}) = (count(w_{i-1}, w_i) + k) / (count(w_{i-1}) + k * |V|)
```

Where `|V|` is the vocabulary size. When `k=1` this is called Laplace smoothing.
Smaller values of `k` (e.g., 0.01) are often more effective.

Other smoothing methods include **Kneser-Ney smoothing** and **Good-Turing smoothing**,
which are more sophisticated and generally outperform add-k.

---

## 3. Neural Language Models

### Feed-Forward Language Models

Proposed by Bengio et al. (2003), feed-forward neural language models represent words
as dense vectors (embeddings) and use a fixed-size context window:

```
P(w_t | w_{t-n+1}, ..., w_{t-1}) = softmax(W * tanh(C * [e(w_{t-n+1}), ..., e(w_{t-1})]) + b)
```

**Advantages over n-grams:**
- Generalize better through shared word embeddings
- Can capture semantic similarity between words

**Limitation:** Still limited to a fixed context window.

### RNN-Based Language Models

Recurrent Neural Networks (RNNs) maintain a **hidden state** that summarizes the entire
history of the sequence, enabling theoretically unlimited context:

```
h_t = f(W_h * h_{t-1} + W_x * x_t + b)
P(w_{t+1} | w_1, ..., w_t) = softmax(W_out * h_t)
```

**Variants:** LSTM and GRU address the vanishing gradient problem in vanilla RNNs,
allowing them to capture longer-range dependencies.

---

## 4. Autoregressive Models (GPT-Style)

**Autoregressive language models** generate text by predicting the next token given all
previous tokens — one token at a time from left to right.

```
P(w_1, ..., w_n) = ∏ P(w_t | w_1, ..., w_{t-1})
```

### GPT Architecture

GPT (Generative Pre-trained Transformer) uses a **decoder-only Transformer** with
masked self-attention so each token only attends to previous tokens.

```
Input:  [BOS] The quick brown fox
Target: The   quick brown fox [EOS]
```

### Teacher Forcing During Training

During training, the model receives the **ground-truth previous tokens** as input,
regardless of what the model would have predicted. This stabilizes training but creates
a gap between training and inference (exposure bias).

```
Training:   Model sees ground-truth tokens → predicts next token
Inference:  Model sees its own generated tokens → predicts next token
```

**Scheduled sampling** is one approach to bridge this gap by gradually replacing
ground-truth inputs with model predictions during training.

---

## 5. Masked Language Models (BERT-Style)

Rather than predicting the next token, **masked language models (MLMs)** predict
randomly masked tokens given the surrounding context (both left and right).

### BERT's Masking Strategy

BERT masks 15% of tokens. Of the masked positions:
- 80% are replaced with `[MASK]`
- 10% are replaced with a random token
- 10% are kept unchanged

```
Input:  The [MASK] sat on the [MASK]
Target:      cat                mat
```

### Key Difference from Autoregressive Models

| Property             | GPT (Autoregressive)        | BERT (Masked LM)              |
|---------------------|-----------------------------|-------------------------------|
| Context             | Left context only           | Bidirectional (left + right)  |
| Training objective  | Predict next token          | Predict masked tokens         |
| Good for            | Text generation             | Understanding tasks (NLU)     |
| Generation          | Natural (left-to-right)     | Not naturally generative      |

---

## 6. Evaluation: Perplexity

**Perplexity (PP)** measures how well a language model predicts a held-out test set.
It is the exponential of the average negative log-likelihood per token:

```
PP(W) = exp(-1/N * Σ log P(w_i | context_i))
```

Where `N` is the total number of tokens and the sum runs over all tokens in the test set.

### Interpretation

- **Lower perplexity = better model** (the model is less "surprised" by the data)
- A perplexity of `k` means the model is as uncertain as if choosing uniformly among `k` options at each step
- A fair coin flip has perplexity 2; a perfect model has perplexity 1
- GPT-2 achieves ~18 perplexity on WikiText-103; early n-gram models had perplexity > 100

### Limitations

Perplexity only measures a model's ability to predict held-out text from the **same
distribution** as training data. It does not capture whether generated text is coherent,
factually correct, or grammatically natural for humans.

---

## 7. Tokenization

Tokenization is the process of splitting raw text into discrete units (tokens) that
the model processes. The choice of tokenization affects vocabulary size, handling of
rare words, and model performance.

### Word-Level Tokenization

Split text on whitespace and punctuation. Each unique word is a token.

- **Pros:** Intuitive, tokens are human-readable
- **Cons:** Large vocabulary; out-of-vocabulary (OOV) words at inference time; fails
  on morphologically rich languages (Finnish, Turkish)

### Character-Level Tokenization

Each character is a token.

- **Pros:** No OOV problem; tiny vocabulary (~100 characters)
- **Cons:** Sequences are very long; model must learn words from scratch; slow to train

### Byte Pair Encoding (BPE) / WordPiece

**BPE** starts with characters and iteratively merges the most frequent adjacent pair
of symbols, building a vocabulary of subword units.

- **Pros:** Balances vocabulary size and OOV handling; common words remain whole;
  rare words are split into subwords
- **Cons:** Tokenization depends on training corpus; may split words unexpectedly

**WordPiece** (used in BERT) is similar but selects merges that maximize likelihood
of the training data rather than frequency.

**SentencePiece** (used in many multilingual models) works directly on raw text without
pre-tokenization, making it language-agnostic.

| Method     | Vocab size | OOV handling | Sequence length |
|------------|------------|--------------|-----------------|
| Word       | 50k–200k   | Poor         | Short           |
| Character  | ~100       | Perfect      | Very long       |
| BPE/WP     | 10k–50k    | Good         | Medium          |

---

## 8. Applications

Language models are at the core of nearly every modern NLP application:

| Application          | How LM is Used                                                    |
|----------------------|-------------------------------------------------------------------|
| Text Generation      | Autoregressive sampling from P(w_t | previous tokens)            |
| Autocomplete         | Predict most likely next word/phrase                              |
| Machine Translation  | LM scores used as decoder in MT systems                          |
| Summarization        | Conditional generation: P(summary | document)                    |
| Speech Recognition   | Re-rank ASR hypotheses using language model score                |
| Spell Checking       | Flag low-probability word sequences                              |
| Code Generation      | Trained on code corpora (GitHub Copilot, CodeLlama)             |

---

## 9. Reference Links

- [Wikipedia: Language Model](https://en.wikipedia.org/wiki/Language_model)
- [Hugging Face NLP Course](https://huggingface.co/course/chapter1)
- [BERT Paper (Devlin et al., 2018)](https://arxiv.org/abs/1810.04805)
- [The Annotated Transformer (Harvard NLP)](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [Karpathy: The Unreasonable Effectiveness of RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
