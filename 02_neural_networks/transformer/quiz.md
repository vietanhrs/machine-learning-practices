# Quiz: Transformer Networks

Answer each question in your own words. For formula questions, write out the equations clearly.

---

**Q1.** In the attention mechanism, explain what **Query (Q)**, **Key (K)**, and **Value (V)** represent conceptually. Use the analogy of a database lookup to clarify how they interact. Why are three separate matrices needed rather than using the same representation for all three roles?

---

**Q2.** Write the **scaled dot-product attention formula** in full. Explain each component:
- What does `Q K^T` compute?
- Why do we divide by `√d_k`?
- What does the softmax do?
- What is the final result after multiplying by `V`?

---

**Q3.** Describe the advantage of **multi-head attention** over single-head attention. If `d_model = 512` and `n_heads = 8`, what is the dimension of each head (`d_k = d_v`)? How are the outputs of all heads combined?

---

**Q4.** Self-attention is **permutation invariant** — the output is the same regardless of the order of input tokens. Explain:
- (a) Why this is a problem for language modeling.
- (b) How **positional encoding** solves it.
- (c) Write the sinusoidal positional encoding formulas for even and odd dimensions.

---

**Q5.** Compare **Layer Normalization** and **Batch Normalization**:
- What statistics does each normalize over (batch dimension vs. feature dimension)?
- Why is LayerNorm preferred for Transformers?
- What are the two learnable parameters in LayerNorm?

---

**Q6.** Explain the purpose of **residual (skip) connections** in Transformer blocks. Write the general form `output = LayerNorm(x + Sublayer(x))`. What are two specific benefits residual connections provide for training deep Transformers?

---

**Q7.** Describe the difference between the Transformer **encoder** and **decoder**:
- How many sub-layers does each have?
- What type of attention does each use (self-attention, cross-attention)?
- Which direction of attention is allowed in each?

---

**Q8.** What is **causal masking** in the Transformer decoder? Write or describe the mask matrix for a sequence of length 4. Why is it implemented as `-∞` (negative infinity) rather than simply setting scores to 0?

---

**Q9.** BERT uses **Masked Language Modeling (MLM)** as its pretraining objective. Describe:
- (a) The masking procedure (what percentage, what substitutions).
- (b) What the model is trained to predict.
- (c) Why MLM creates a bidirectional representation (unlike standard LM).

---

**Q10.** GPT uses **autoregressive generation**. Describe the generation process step by step for generating a sequence of T tokens given a prompt. What is the **temperature** parameter and how does it affect diversity of outputs?

---

**Q11.** The standard Transformer attention has **O(n²) time and space complexity** with respect to sequence length n. Explain why, and what practical problem this creates for very long documents (e.g., n = 100,000). Name one technique that reduces this complexity.

---

**Q12.** Explain how Transformers handle **long-range dependencies** differently from RNNs. For a 512-token sequence, compare the maximum path length between the first and last tokens in an LSTM vs. a Transformer. What does this mean for gradient flow?

---

**Q13.** In the Feed-Forward Network (FFN) sub-layer, what are the input and output dimensions relative to `d_model`? Why is the intermediate dimension typically `4 × d_model`? What type of non-linearity is used, and how does FFN complement the attention mechanism?

---

**Q14.** Describe the concept of **transfer learning** in the context of BERT and GPT. What does "fine-tuning" mean specifically for these models (which parameters change, which are frozen)? Give two examples of downstream tasks that can be fine-tuned with minimal architecture changes.

---

**Q15.** You are building a system to classify customer support emails into 10 categories. You have 500 labeled examples. Compare these approaches:
- (a) LSTM trained from scratch
- (b) BERT fine-tuned on your 500 examples
- (c) GPT-4 few-shot prompted (no fine-tuning)

Analyze each in terms of expected performance, data requirements, cost, and deployment complexity. Which would you choose as a first experiment?

---

## Answers (Hidden — attempt questions before reading)

<details>
<summary>Click to reveal answer guidance</summary>

1. Q: "What am I searching for?" K: "What do I have to offer?" V: "What I actually provide." Database analogy: Q is a query, K is an index, V is the stored value. Matching Q against K gives attention weights; these weight the V vectors. Three separate matrices allow the model to learn independent subspaces for searching, indexing, and content.

2. Attention(Q,K,V) = softmax(QK^T / √d_k) V. QK^T: dot product similarity between every query and key pair. √d_k: prevents large magnitude dot products that push softmax to near-zero gradients. Softmax: converts scores to a probability distribution (per row). @V: each output is a weighted average of values, where weights reflect relevance.

3. Multiple heads capture different relationship types simultaneously (syntax, semantics, coreference). d_k = d_v = 512/8 = 64. Outputs concatenated: Concat(head_1,...,head_h) shape (n, d_model), then projected by W_O (d_model, d_model).

4. (a) "The cat sat" and "sat cat The" would get the same attention output — word order is meaningless. (b) Adds a unique vector to each position's embedding before the first layer. (c) PE(pos,2i) = sin(pos/10000^(2i/d_model)); PE(pos,2i+1) = cos(pos/10000^(2i/d_model)).

5. BatchNorm: normalizes across batch (statistics per feature across samples). LayerNorm: normalizes across features (statistics per sample across features). LayerNorm: sequence length-independent, works on batch_size=1, more stable for variable-length text. Parameters: γ (scale), β (shift).

6. output = LayerNorm(x + Sublayer(x)). Benefits: (1) Gradient can flow directly through the skip path, avoiding vanishing gradients. (2) Each layer only needs to learn a residual correction (Sublayer(x) ≈ 0 initially), making optimization easier.

7. Encoder: 2 sub-layers (self-attention + FFN). Full bidirectional self-attention. Decoder: 3 sub-layers (masked self-attention + cross-attention + FFN). Masked self-attention: causal (past only). Cross-attention: K,V from encoder, Q from decoder (bidirectional to encoder output).

8. Causal mask: lower triangular matrix of 0s and -∞ above diagonal. For seq=4: row i can attend to positions 0..i. Using -∞ so exp(-∞) = 0 exactly in softmax, making those positions contribute nothing to attention weights.

9. (a) 15% of tokens randomly selected: 80% replaced with [MASK], 10% random word, 10% unchanged. (b) Predicts the original token at each masked position. (c) Each [MASK] position can attend to all other positions (left and right context) — no causal constraint.

10. Start with prompt tokens. Predict probability distribution over vocabulary. Sample (or take argmax). Append sampled token. Repeat until EOS or max length. Temperature T > 1 flattens distribution (more random/diverse); T < 1 sharpens it (more deterministic).

11. QK^T has shape (n, n) — requires O(n²) memory and O(n²·d) compute. For n=100,000: 10^10 elements — infeasible. Solutions: Sparse attention, Longformer (local + global attention), FlashAttention (memory-efficient exact), linear attention approximations.

12. In LSTM, information travels T steps from position 1 to position T (max path length T). In Transformer, any pair of positions is connected in a single layer (max path length 1). Shorter paths → gradients flow more easily → better long-range dependency learning.

13. FFN: d_model → d_ff (=4·d_model) → d_model. Large intermediate dim provides high capacity for memorization and feature transformation. ReLU or GELU non-linearity. Attention handles "routing" (where to look); FFN handles "processing" (what to do with that information).

14. Pretraining learns general language representations from large unlabeled corpora. Fine-tuning: all model weights updated (not frozen) on labeled task data, with a task-specific head added. Typically low learning rate to avoid catastrophic forgetting. Examples: text classification (add linear layer on [CLS] token), NER (linear layer on each token output).

15. (a) LSTM from scratch: likely poor with 500 samples; no knowledge transfer; fast and cheap. (b) BERT fine-tuned: likely strong — pretrained representations; needs labeled fine-tuning only; moderate cost. (c) GPT-4 few-shot: competitive or better; zero labeled examples needed; high per-query cost; no deployment control. Recommended first experiment: BERT (sentence-transformers + logistic regression is even faster as a baseline).

</details>
