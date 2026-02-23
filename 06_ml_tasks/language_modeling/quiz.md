# Language Modeling — Quiz

Test your understanding of language modeling concepts.

---

**Q1.** What does a language model fundamentally compute?

- A) The sentiment of a sentence
- B) The probability of a sequence of words
- C) The grammatical parse tree of a sentence
- D) The topic of a document

<details>
<summary>Answer</summary>

**B) The probability of a sequence of words.**

A language model defines a probability distribution P(w_1, w_2, ..., w_n) over sequences
of words. This is what makes it useful for generation, ranking, and many downstream tasks.

</details>

---

**Q2.** In an N-gram language model, what does "bigram" mean?

- A) The model uses two separate corpora
- B) Each word's probability depends on the previous two words
- C) Each word's probability depends on the single preceding word
- D) The model has two hidden layers

<details>
<summary>Answer</summary>

**C) Each word's probability depends on the single preceding word.**

A bigram (2-gram) model uses the Markov assumption: P(w_i | w_1,...,w_{i-1}) ≈ P(w_i | w_{i-1}).
It conditions each word on exactly one predecessor. A trigram (3-gram) would use the two
preceding words.

</details>

---

**Q3.** What is the Markov assumption in the context of N-gram language models, and what is its main limitation?

<details>
<summary>Answer</summary>

The **Markov assumption** states that the probability of a word depends only on a fixed
window of the immediately preceding words (the last `n-1` words for an n-gram model),
not on the entire history.

**Limitation:** It ignores long-range dependencies. For example, in "The software that
the engineers who joined last year wrote ___", the word that fills the blank depends
on "software", which may be many words back — beyond the bigram or trigram window.

</details>

---

**Q4.** Why is smoothing necessary in N-gram language models?

- A) To speed up training
- B) To handle zero probabilities for unseen n-grams
- C) To reduce the vocabulary size
- D) To improve GPU utilization

<details>
<summary>Answer</summary>

**B) To handle zero probabilities for unseen n-grams.**

If an n-gram was never observed in the training corpus, its count is 0, giving it a
probability of 0. Any test sentence containing that n-gram would then have probability 0,
regardless of how reasonable the sentence is. Smoothing redistributes some probability
mass to unseen events to avoid this.

</details>

---

**Q5.** A language model has a perplexity of 50 on a test set. Another model achieves a perplexity of 12. Which model is better, and what does this mean intuitively?

<details>
<summary>Answer</summary>

The model with **perplexity 12 is better.** Lower perplexity indicates a better model.

Intuitively, a perplexity of 12 means the model is as uncertain as if choosing uniformly
among 12 equally likely options at each step. A perplexity of 50 means the model is as
uncertain as choosing among 50 options. The lower-perplexity model assigns higher
probability to the actual words in the test set — it is "less surprised."

</details>

---

**Q6.** What is teacher forcing, and why is it used during language model training?

<details>
<summary>Answer</summary>

**Teacher forcing** is a training technique where the model receives the **ground-truth
tokens** as input at each step, rather than its own previously generated tokens.

For example, when training to predict "The cat sat", at each step the model is given
the true previous token ("The" → predict "cat"; "cat" → predict "sat"), even if the
model previously predicted something wrong.

**Why it is used:**
- Makes training faster and more stable (gradients flow through clean sequences)
- Avoids error accumulation during training

**Downside:** Creates a gap between training (sees ground truth) and inference (sees own
predictions), known as **exposure bias**. Scheduled sampling partially addresses this.

</details>

---

**Q7.** GPT and BERT are both Transformer-based language models. What is the key difference in their training objectives?

<details>
<summary>Answer</summary>

| Property          | GPT (Autoregressive)                        | BERT (Masked LM)                             |
|-------------------|---------------------------------------------|----------------------------------------------|
| Objective         | Predict the **next token** left-to-right   | Predict **randomly masked tokens**           |
| Context           | Only left (past) context                    | Bidirectional (left + right context)         |
| Architecture      | Decoder-only Transformer                    | Encoder-only Transformer                     |
| Best suited for   | Text generation                             | Classification, NLU tasks                    |

GPT uses a causal (autoregressive) objective; BERT uses a masked (non-causal) objective.

</details>

---

**Q8.** Why is Byte Pair Encoding (BPE) generally preferred over word-level tokenization for large language models?

- A) BPE produces shorter sequences
- B) BPE eliminates the out-of-vocabulary problem while keeping vocabulary size manageable
- C) BPE is faster to compute at inference time
- D) BPE was invented specifically for Transformers

<details>
<summary>Answer</summary>

**B) BPE eliminates the out-of-vocabulary problem while keeping vocabulary size manageable.**

Word-level tokenization suffers from two problems: (1) huge vocabularies for large corpora
and (2) OOV words at inference time that the model has never seen. BPE splits rare or
unknown words into known subword pieces, so any word can be represented. It also allows
common words to remain as single tokens while splitting rare words into components.

</details>

---

**Q9.** During autoregressive text generation, what effect does increasing the **temperature** parameter have on the output?

<details>
<summary>Answer</summary>

Temperature `T` scales the logits before the softmax:

```
P(w_i) = softmax(logits / T)
```

- **T < 1.0 (low temperature):** Distribution becomes sharper (more peaked). The model
  favors high-probability tokens more strongly. Output is more predictable and repetitive.

- **T = 1.0:** Standard softmax — no change to distribution.

- **T > 1.0 (high temperature):** Distribution becomes flatter. Lower-probability tokens
  get relatively more probability. Output is more diverse, creative, but also more random
  and potentially incoherent.

A temperature of 0 is equivalent to greedy decoding (always pick the argmax).

</details>

---

**Q10.** What is the perplexity formula? Write it out and explain each term.

<details>
<summary>Answer</summary>

```
PP(W) = exp( -1/N * Σ_{i=1}^{N} log P(w_i | w_1, ..., w_{i-1}) )
```

- `N` — total number of tokens in the test set
- `P(w_i | w_1, ..., w_{i-1})` — the probability the model assigns to the i-th token
  given all previous tokens
- `Σ log P(...)` — sum of log-probabilities (log-likelihood of the test set)
- `-1/N * Σ log P(...)` — average negative log-likelihood per token (cross-entropy)
- `exp(...)` — exponentiation converts from nats/bits back to a human-readable scale

Lower perplexity = higher log-likelihood = model assigns higher probability to test data.

</details>

---

**Q11.** Explain what "hallucination" means in the context of language models and why it occurs.

<details>
<summary>Answer</summary>

**Hallucination** occurs when a language model generates text that is fluent and
confident-sounding but factually incorrect, made-up, or not grounded in real information.

**Why it occurs:**
1. **Training objective mismatch:** LMs are trained to maximize the probability of the
   next token, not to be factually accurate. Fluent text gets rewarded even if wrong.
2. **No external memory:** The model's "knowledge" is encoded in its parameters. It cannot
   look up facts and may confabulate plausible-sounding but incorrect details.
3. **Distribution of training data:** If the training corpus contains false information,
   the model may reproduce it.
4. **Autoregressive compounding:** Once the model generates a false premise, subsequent
   tokens are conditioned on that false premise, compounding the error.

Mitigation strategies include retrieval-augmented generation (RAG), fine-tuning with RLHF,
and chain-of-thought prompting.

</details>

---

**Q12.** What is the difference between a **unigram** and a **trigram** language model in terms of the independence assumption?

<details>
<summary>Answer</summary>

- **Unigram model:** Assumes all words are **completely independent** of each other.
  P(w_1, ..., w_n) = P(w_1) * P(w_2) * ... * P(w_n). No context is used.

- **Trigram model:** Assumes each word depends on the **two immediately preceding words**.
  P(w_i | w_1, ..., w_{i-1}) ≈ P(w_i | w_{i-2}, w_{i-1}).

Trigrams capture more context and produce more coherent text, but require more data to
estimate reliably (data sparsity increases with n).

</details>

---

**Q13.** What is the difference between **character-level** and **word-level** tokenization? Name one advantage of each.

<details>
<summary>Answer</summary>

| Aspect           | Character-level                           | Word-level                               |
|------------------|-------------------------------------------|------------------------------------------|
| Vocabulary size  | Tiny (~100 for English)                   | Large (50k–200k+)                        |
| OOV words        | Never — any string can be encoded         | Frequent problem                         |
| Sequence length  | Very long (slow to process)               | Short                                    |
| Learning burden  | Model must learn words from characters    | Words are pre-defined units              |

**Advantage of character-level:** No out-of-vocabulary problem; works on any language.

**Advantage of word-level:** Much shorter sequences; each token carries more meaning,
making it easier for the model to learn semantic relationships.

</details>

---

**Q14.** A language model trained on news articles is deployed to generate medical reports. What problems might arise, and why?

<details>
<summary>Answer</summary>

This is a **distribution shift** problem. The model was trained on one domain (news) but
is being used in a different domain (medical text). Problems include:

1. **Vocabulary mismatch:** Medical terminology (drug names, anatomical terms, Latin
   abbreviations) may be rare or absent in news corpora. The model may handle these
   poorly or generate incorrect terms.

2. **Style mismatch:** News writing style differs from clinical report style (passive
   voice, structured sections, specific formatting conventions).

3. **Higher-stakes hallucination:** In medical contexts, generating a wrong drug name or
   incorrect diagnosis is potentially dangerous, not just embarrassing.

4. **Domain knowledge gap:** The model may lack the implicit medical knowledge needed
   to produce coherent reports (e.g., knowing which symptoms go with which conditions).

**Solution:** Domain-adaptive pre-training or fine-tuning on medical text corpora
(e.g., PubMed, clinical notes).

</details>
