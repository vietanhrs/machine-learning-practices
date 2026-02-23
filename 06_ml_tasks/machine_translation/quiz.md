# Machine Translation — Quiz

Test your understanding of machine translation architectures, mechanisms, and evaluation.

---

**Q1.** What is the "bottleneck problem" in vanilla Seq2Seq models, and why is it a problem for long sentences?

<details>
<summary>Answer</summary>

In a vanilla Seq2Seq model (Sutskever et al., 2014), the entire source sentence is
compressed into a **single fixed-size context vector** (the encoder's final hidden state).
This context vector must carry all the information needed to generate the entire target sentence.

**Why it is a problem for long sentences:**

As the source sentence grows longer, more information must be packed into the same fixed-size
vector. Empirically, RNNs struggle to retain information from the beginning of a long sentence
by the time they reach the end — the early words' information gets "washed out."

This was confirmed experimentally: Seq2Seq performance degrades sharply for sentences longer
than ~30 words, while the attention mechanism largely mitigates this degradation.

</details>

---

**Q2.** How does the **attention mechanism** solve the bottleneck problem?

<details>
<summary>Answer</summary>

Instead of relying on a single context vector (the last encoder hidden state), attention
allows the decoder to **directly access all encoder hidden states** at each decoding step.

At each step t, attention computes a **weighted sum** of all encoder states:

```
α_{t,i} = softmax(score(s_{t-1}, h_i))   ← how much to attend to source position i
c_t      = Σ_i α_{t,i} * h_i             ← step-specific context vector
```

This means:
- The decoder "looks back" at relevant parts of the source for each target token.
- The model learns which source positions are most relevant when generating each target token.
- Information from the beginning of a long source sentence is directly accessible.
- No information bottleneck — the context vector changes at every decoder step.

</details>

---

**Q3.** What is the difference between **Bahdanau attention** and **Luong attention**?

<details>
<summary>Answer</summary>

| Property               | Bahdanau (Additive)                                     | Luong (Multiplicative)                                     |
|------------------------|----------------------------------------------------------|------------------------------------------------------------|
| Score function         | `v^T * tanh(W1 * s_{t-1} + W2 * h_i)`                 | `s_t^T * W * h_i` (general) or `s_t^T * h_i` (dot)      |
| When context is used   | Context fed into GRU at each step **before** RNN step  | Context used **after** the RNN step                        |
| Query                  | Previous decoder hidden state `s_{t-1}`                | Current decoder hidden state `s_t`                         |
| Computational cost     | More parameters (two weight matrices + vector)          | Simpler (one matrix or just dot product)                   |
| Performance            | Generally comparable                                    | Slightly simpler to implement                              |

Both compute a context vector as a weighted sum of encoder states, but differ in
**when** the decoder state is computed relative to attention and in the **score function**.

</details>

---

**Q4.** BLEU score uses n-gram precision. What is "**clipped** n-gram precision" and why is clipping necessary?

<details>
<summary>Answer</summary>

**Standard n-gram precision** counts how many n-grams in the hypothesis appear in the reference,
divided by total n-grams in the hypothesis.

**Problem:** A degenerate system could repeat the same high-frequency word ("the the the the")
to get 100% unigram precision since "the" appears in most references.

**Clipped precision:** Each n-gram in the hypothesis is counted at most as many times as
it appears in the best-matching reference:

```
count_clip(ngram) = min(count_hyp(ngram), max_ref_count(ngram))
```

**Example:**
- Hypothesis: "the the the the the"
- Reference:  "the cat sat on the mat"
- "the" appears 5 times in hypothesis, but only 2 in reference
- Clipped count for "the" = min(5, 2) = 2
- Clipped precision = 2/5 = 0.4 (not 5/5 = 1.0)

Clipping prevents systems from gaming the metric by repeating common words.

</details>

---

**Q5.** What is the **brevity penalty** in BLEU score? What does it prevent?

<details>
<summary>Answer</summary>

The **brevity penalty (BP)** penalizes hypotheses that are shorter than the reference:

```
BP = 1                                       if len(hyp) > len(ref)
BP = exp(1 - len(ref_closest) / len(hyp))   otherwise
```

**What it prevents:**

A system could achieve very high n-gram precision by generating very short hypotheses
that contain only high-confidence n-grams. For example, generating just "the" would
have 100% unigram precision if "the" appears in the reference. But this is useless.

The brevity penalty reduces the BLEU score when the hypothesis is shorter than the
reference, discouraging overly short outputs.

**Note:** There is no length penalty for hypotheses that are too long — BLEU relies
on n-gram precision to penalize long, rambling translations that add extra unwanted words.

</details>

---

**Q6.** BLEU score is widely criticized. Name **three** limitations of BLEU as an MT evaluation metric.

<details>
<summary>Answer</summary>

1. **No semantic understanding:** Two sentences with the same meaning but different words
   receive very different BLEU scores. Synonyms ("large" vs. "big") are counted as wrong.
   Example: "A huge cat" vs. "A large cat" — if reference uses "big," both score similarly
   despite one being much closer in meaning.

2. **No word order sensitivity at low n:** BLEU-1 (unigram precision) is completely
   insensitive to word order. "Cat the sat mat the on" has the same BLEU-1 as
   "The cat sat on the mat."

3. **Reference-dependent:** BLEU depends heavily on which reference translations are
   available. The same translation can receive very different scores depending on the
   reference choice. Human translators asked to translate the same sentence often produce
   quite different valid translations.

4. **Poor at sentence level:** BLEU is most meaningful at the corpus level. At the
   sentence level, it is often 0 (no n-gram matches) and unreliable.

5. **Doesn't correlate well with human judgment in all domains:** In domains where
   fluency matters more than lexical precision (poetry, creative writing), BLEU fails
   badly. Even in standard MT, newer metrics like COMET and BERTScore correlate
   significantly better with human judgments.

</details>

---

**Q7.** In the Transformer decoder, what is **cross-attention** and how does it differ from **self-attention**?

<details>
<summary>Answer</summary>

**Self-attention** (in encoder or decoder):
- Query (Q), Key (K), and Value (V) all come from the **same sequence**.
- In the encoder: each source token attends to all other source tokens.
- In the decoder (masked): each target token attends to previous target tokens.

**Cross-attention** (in decoder only):
- Query (Q) comes from the **decoder** (the current target sequence state).
- Key (K) and Value (V) come from the **encoder output** (the encoded source sequence).
- Each target token position queries the encoder to find the most relevant source positions.

```
Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V

Cross-attention:
    Q = decoder state
    K = V = encoder output
```

Cross-attention is the Transformer's equivalent of Bahdanau attention — it allows
each decoder position to selectively attend to different parts of the source sentence.

</details>

---

**Q8.** What is **teacher forcing** in MT training, and what is the drawback known as "exposure bias"?

<details>
<summary>Answer</summary>

**Teacher forcing:** During training, at each decoder step, the model receives the
**ground-truth target token** as input (rather than its own previous prediction),
even if the model would have predicted something different.

```
Training step:  s_t = Decoder(y_{t-1}^*,  s_{t-1}, c)  ← y* = ground truth
Inference step: s_t = Decoder(ŷ_{t-1},    s_{t-1}, c)  ← ŷ  = model's own prediction
```

**Why teacher forcing:** Without it, early in training the model makes bad predictions,
and feeding those bad predictions back leads to error accumulation — the model is
"lost" and gradients become useless for learning. Teacher forcing provides a clean,
stable training signal.

**Exposure bias:** The model is trained with perfect history (ground-truth tokens)
but must generate with its own imperfect predictions at inference time. This training
vs. inference discrepancy is called exposure bias.

**Consequence:** Small errors at one step can cascade — the model makes a bad prediction,
which is then fed as input to the next step, leading to another bad prediction.

**Mitigation:** Scheduled sampling (gradually replacing teacher inputs with model
predictions during training) partially addresses exposure bias.

</details>

---

**Q9.** What is **beam search** and why is it preferred over greedy decoding for MT?

<details>
<summary>Answer</summary>

**Greedy decoding:** At each step, select the single highest-probability token.
Simple and fast, but locally optimal choices may not lead to globally optimal sequences.

**Beam search:** Maintain the top-k (beam size) highest-scoring partial hypotheses
at each step. At each step, expand each hypothesis by all vocabulary tokens and keep
only the top-k candidates by cumulative log-probability.

```
Beam size k=3, step t=1:
    Hypothesis 1: "The"   (log-prob = -0.2)
    Hypothesis 2: "A"     (log-prob = -0.5)
    Hypothesis 3: "These" (log-prob = -0.9)

Step t=2: expand each with all vocab → 3 * V candidates → keep top 3
```

**Why beam search is better than greedy:**

Greedy can get stuck in local optima. A word that has slightly lower probability
at one step may lead to a much better full sentence. Beam search explores k paths
simultaneously, increasing the chance of finding a high-probability full translation.

**Practical BLEU improvements:** Beam search with k=4–5 typically improves BLEU
by 1–2 points over greedy on standard benchmarks.

**Trade-off:** Beam search is k times slower than greedy. Very large beams can
actually hurt diversity and sometimes hurt BLEU (beam search curse).

</details>

---

**Q10.** How does **BPE (Byte Pair Encoding)** tokenization help handle **out-of-vocabulary (OOV) words** in MT?

<details>
<summary>Answer</summary>

**OOV problem in word-level MT:** If a source word was never seen in training, it has
no embedding or representation. The model cannot translate it.

**BPE solution:** BPE segments all words into subword units. The vocabulary is built
from frequently occurring character sequences, not full words.

**Example:**
- Word: "unbelievable" (rare, might be OOV)
- BPE segments: "un" + "believ" + "able" (all common subwords)

**Why this helps:**
1. **No OOV:** Any word can be represented as a sequence of known subword pieces.
   Even completely novel words (names, new technical terms) can be encoded character by character.
2. **Shared morphology:** Languages with rich morphology (German: "Unglaublichkeit" = "incredibility")
   share subword units across related word forms, helping the model generalize.
3. **Bilingual sharing:** Using shared BPE vocabulary across source and target languages
   helps the model align related words across languages (e.g., cognates).

**Joint BPE:** In practice, BPE is often learned on the concatenation of source and
target vocabularies, creating a shared subword vocabulary that is particularly
beneficial for related language pairs.

</details>

---

**Q11.** What does the **attention matrix** reveal in machine translation?

<details>
<summary>Answer</summary>

The attention matrix (dimensions: target positions × source positions) shows, for
each target word generated, which source words the model gave the most attention to.

**What it reveals:**

1. **Word alignment:** Often corresponds closely to the implicit word alignment
   between source and target. In English→French translation, attention from the
   French word "chat" is usually highest at the English word "cat."

2. **Reordering:** Reveals how word order changes between languages. In English→German,
   the verb often appears at the end of a German sentence — the attention matrix shows
   that when generating the German verb, the model attends back to the English verb.

3. **Phrase-level correspondence:** Multi-word expressions in the source (e.g., "on the
   other hand") may have a single target token attending to multiple source tokens.

4. **Model behavior:** Unusual attention patterns can indicate where the model is likely
   to make errors — if it attends to irrelevant source positions, the translation is
   likely wrong.

Attention matrices are a valuable interpretability tool, though they are only a proxy
for alignment and don't always perfectly reflect which information the model uses.

</details>

---

**Q12.** A Transformer with beam size=1 achieves BLEU 28. With beam size=4, it achieves BLEU 31. With beam size=20, BLEU drops to 30. Explain why BLEU can decrease with very large beams.

<details>
<summary>Answer</summary>

This is called the **beam search curse** or **beam search pathology**.

**Why BLEU can decrease with very large beam sizes:**

1. **Length bias:** Beam search with cumulative log-probabilities has a systematic bias
   toward shorter sequences. Each additional token incurs a negative log-probability cost,
   so shorter sequences tend to accumulate less negative score. With larger beams,
   shorter but lower-quality translations can "survive" longer.

2. **Length normalization:** Without length normalization, beam search with large beams
   finds shorter and shorter translations, eventually producing degenerate outputs.

3. **Mode collapse toward generic translations:** Larger beams explore more of the
   probability mass, which can concentrate on very "safe" but uninformative translations
   (generic sentences that are always at least somewhat probable).

4. **Mismatch with BLEU:** The sequence with highest model probability is not necessarily
   the one with highest BLEU. Model training (maximum likelihood) and BLEU are not
   perfectly aligned objectives.

**Common fixes:**
- Length penalty: Normalize cumulative log-probs by length^α (α ≈ 0.6)
- Minimum length constraints to prevent premature EOS
- Diverse beam search to encourage hypothesis diversity

</details>

---

**Q13.** What is **mBART** and how does it differ from training a separate translation model for each language pair?

<details>
<summary>Answer</summary>

**mBART (Multilingual BART)** (Liu et al., 2020) is a multilingual denoising autoencoder
pre-trained on large monolingual corpora in 25 languages using a reconstruction objective
(corrupted text → original text). It is then fine-tuned on parallel data for translation.

**Key differences from language-pair-specific models:**

1. **Shared parameters:** A single model handles all 25×24 = 600 translation directions.
   Individual models require training one model per pair.

2. **Zero-shot and low-resource translation:** Because mBART shares representations across
   languages, it can translate language pairs for which very little parallel data exists,
   using knowledge transferred from high-resource pairs.

3. **Transfer learning:** Pre-training on monolingual data (much more abundant than
   parallel data) gives the model rich representations of each language before seeing
   any parallel data.

4. **Multilingual representations:** The shared encoder/decoder learns to represent
   semantically similar content from different languages in similar vector spaces,
   enabling cross-lingual transfer.

**Limitation:** A single large model may underfit high-resource language pairs compared
to a dedicated model trained with abundant parallel data. There can also be "language
interference" where optimization for many languages simultaneously hurts any individual pair.

</details>
