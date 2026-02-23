# Machine Translation

Machine translation (MT) is the task of automatically converting text from one natural
language (source) into another (target) while preserving meaning. It is one of the
oldest and most challenging problems in NLP.

---

## 1. What is Machine Translation?

Given a sentence in a source language, produce an equivalent sentence in a target language:

```
Source (English): "The cat sat on the mat."
Target (French):  "Le chat était assis sur le tapis."
```

MT is a **sequence-to-sequence (Seq2Seq)** mapping problem: both input and output are
variable-length sequences, and the lengths of source and target sequences can differ.

**Why it is hard:**
- Words don't map one-to-one across languages
- Word order can differ dramatically (e.g., SOV vs. SVO languages)
- Ambiguity: one word can have many translations depending on context
- Morphologically rich languages (German, Finnish) have many word forms
- Idiomatic expressions don't translate literally

---

## 2. Statistical MT (Brief History)

Before neural networks, MT used **statistical methods** based on large parallel corpora
(sentence pairs in two languages).

### IBM Models (1990s)

Proposed word alignment models to learn which source words "generate" which target words.
IBM Model 1–5 estimated alignment probabilities: P(target | source).

### Phrase-Based MT (2000s)

Translated **phrases** (not individual words) using a phrase table of source→target
translation pairs learned from parallel data.

```
P(target | source) ≈ P(source | target) * P(target)
```
(Translation model × Language model — a log-linear combination)

**State of the art pre-2014:** Moses, a phrase-based MT system.

**Limitations:** Struggled with long-range reordering; required extensive feature engineering.

---

## 3. Seq2Seq Architecture

**Sequence-to-Sequence (Seq2Seq)** models (Sutskever et al., 2014) use two RNNs:

### Encoder

Reads the source sentence token by token and produces a **fixed-size context vector** `c`
(the final hidden state) summarizing the entire source sentence:

```
h_1, h_2, ..., h_T = Encoder(x_1, x_2, ..., x_T)
c = h_T   (or a transformation of it)
```

### Decoder

Generates the target sentence token by token, conditioned on the context vector:

```
s_0 = c
s_t = Decoder(s_{t-1}, y_{t-1}, c)
P(y_t | y_{<t}, x) = softmax(W * s_t)
```

### The Bottleneck Problem

The entire source sentence is compressed into a single fixed-size vector `c`.
For long sentences, this bottleneck loses information from the beginning of the sentence
by the time the end is processed. This limits translation quality, especially for long texts.

---

## 4. Attention in MT

**Attention mechanism** (Bahdanau et al., 2015) solves the bottleneck by allowing the
decoder to **attend to all encoder hidden states** at each decoding step, rather than
relying solely on the final hidden state.

### Bahdanau (Additive) Attention

At each decoder step t, compute a **context vector** as a weighted sum of all encoder
hidden states:

```
score(s_{t-1}, h_i) = v^T * tanh(W_1 * s_{t-1} + W_2 * h_i)
α_{t,i} = softmax(score(s_{t-1}, h_i))        (attention weights)
c_t      = Σ_i α_{t,i} * h_i                  (context vector for step t)
```

The attention weights `α_{t,i}` indicate how much the decoder "attends" to source
position i when generating target position t.

### Interpretability

The attention matrix (α for all t and i) reveals the implicit alignment learned by the
model — which source words correspond to which target words.

### Luong (Multiplicative) Attention

A simpler alternative that uses dot product similarity:
```
score(s_t, h_i) = s_t^T * W * h_i    (general)
               or s_t^T * h_i          (dot)
```

---

## 5. Transformer for MT

The **Transformer** (Vaswani et al., 2017) replaced RNNs entirely with attention.

### Encoder

A stack of identical layers, each with:
1. **Multi-head self-attention:** Each token attends to all other tokens in the source sentence
2. **Feed-forward network:** Applied position-wise
3. **Layer normalization + residual connections**

### Decoder

A stack of identical layers, each with:
1. **Masked multi-head self-attention:** Each target token attends only to previous target tokens (causal masking to prevent peeking)
2. **Cross-attention:** Each target token attends to all encoder outputs (equivalent to Bahdanau attention but using queries from the decoder)
3. **Feed-forward network**

### Advantages over RNNs

| Aspect                | RNN                            | Transformer                        |
|-----------------------|--------------------------------|------------------------------------|
| Parallelism           | Sequential (slow to train)     | Fully parallel                     |
| Long-range dependency | Degrades with distance         | O(1) attention across all positions|
| Context              | Fixed hidden state             | Attends to all positions directly  |
| Training speed        | Slow                           | Much faster with GPUs              |

---

## 6. Evaluation: BLEU Score

**BLEU (Bilingual Evaluation Understudy)** is the most widely used MT metric.
It measures n-gram overlap between the hypothesis (model output) and reference translations.

### Formula

```
BLEU = BP * exp(Σ_{n=1}^{N} w_n * log p_n)
```

Where:
- `p_n` = **clipped n-gram precision**: count of n-grams in hypothesis that appear in any reference, divided by total n-grams in hypothesis. Clipping prevents the model from repeating n-grams to inflate the score.
- `w_n` = weight for each n-gram order (typically 1/4 for N=4)
- `BP` = **brevity penalty**: penalizes hypotheses shorter than the reference

### Brevity Penalty

```
BP = 1                     if len(hyp) > len(ref)
BP = exp(1 - len(ref)/len(hyp))  otherwise
```

Prevents the model from generating very short outputs to achieve high precision.

### Interpretation

- BLEU is reported as a score from 0 to 100 (after multiplying by 100).
- BLEU 25: Understandable translation with many errors
- BLEU 40: Good translation, approaching human quality
- BLEU 60+: High-quality, near-human translation
- Human reference translations vs. each other achieve BLEU ≈ 70–90

### Limitations of BLEU

1. **Does not capture meaning:** Two sentences can convey the same meaning with different words. BLEU only measures surface-level n-gram overlap.
2. **No synonyms:** "big" and "large" are completely different to BLEU.
3. **Word order:** BLEU with n=1 (BLEU-1) does not penalize reordering at all.
4. **Single reference:** BLEU is sensitive to the choice of reference translation.
5. **Alternatives:** chrF, METEOR, BERTScore, COMET are generally better correlates of human judgment.

---

## 7. Challenges

| Challenge                    | Description                                                              |
|------------------------------|--------------------------------------------------------------------------|
| Long sentences               | Attention scales as O(n²) with sequence length; harder for RNNs        |
| Rare and OOV words           | Words not in vocabulary cannot be translated directly                   |
| Morphologically rich languages| German, Turkish, Finnish: one concept = many word forms               |
| Low-resource languages       | Many languages have few parallel corpora                                |
| Domain adaptation            | A model trained on news fails on medical or legal texts                 |
| Hallucination                | Model generates fluent but wrong translations                           |

**OOV handling with BPE:** Subword tokenization (BPE) allows any word to be represented
as a sequence of known subword pieces, eliminating the OOV problem entirely.

---

## 8. Modern MT Systems

| System              | Organization   | Notes                                                      |
|---------------------|----------------|-----------------------------------------------------------|
| Helsinki-NLP OPUS-MT| University of Helsinki | Open-source Transformer models for 1000+ language pairs |
| mBART              | Meta AI        | Multilingual denoising autoencoder pre-trained on 25 languages |
| NLLB (No Language Left Behind) | Meta AI | 200 languages; focus on low-resource languages |
| Google Translate    | Google         | Combination of NMT and LLM-based approaches               |
| DeepL              | DeepL SE       | High-quality commercial MT, especially for European languages |

---

## 9. Reference Links

- [Bahdanau Attention Paper (2015)](https://arxiv.org/abs/1409.0473)
- [Wikipedia: BLEU Score](https://en.wikipedia.org/wiki/BLEU)
- [Helsinki-NLP on Hugging Face](https://huggingface.co/Helsinki-NLP)
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [NLLB (No Language Left Behind) Meta AI, 2022](https://arxiv.org/abs/2207.04672)
