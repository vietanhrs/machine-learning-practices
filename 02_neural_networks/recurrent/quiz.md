# Quiz: Recurrent Neural Networks (RNN / LSTM / GRU)

Answer each question in your own words. For formula questions, write out the equations clearly.

---

**Q1.** What is the **hidden state** in a Vanilla RNN, and what purpose does it serve? Write the RNN recurrence formula and identify every variable. Why is the hidden state sometimes called the network's "memory"?

---

**Q2.** The gradient of the loss at time step `T` with respect to the hidden state at time `t` involves a product of `(T - t)` Jacobian matrices. Explain why this causes vanishing gradients in Vanilla RNNs for long sequences, and why this problem does not occur as severely with addition-based updates.

---

**Q3.** Describe the **LSTM cell state** `C_t` and explain how it differs from the hidden state `h_t`. Why can gradients flow more easily through the cell state than through the hidden state in a Vanilla RNN? Refer to the specific mathematical operations involved.

---

**Q4.** Write the LSTM **forget gate** equation and explain in your own words what it does. Give a concrete example from natural language processing where a forget gate would be useful (e.g., when reading a paragraph with multiple subjects).

---

**Q5.** Compare **GRU** and **LSTM** in terms of:
- Number of gates and their names
- Whether a separate cell state exists
- Parameter count (relative)
- When you would prefer one over the other

---

**Q6.** What is **Backpropagation Through Time (BPTT)** and how does it differ from standard backpropagation? What is the **truncated BPTT** technique and why is it used in practice?

---

**Q7.** A Bidirectional RNN runs two separate RNNs over the same sequence. Describe:
- (a) What each direction learns.
- (b) How the outputs are combined.
- (c) A task where bidirectionality is essential (and why).
- (d) A task where you cannot use a bidirectional RNN (and why).

---

**Q8.** What is **teacher forcing** in sequence-to-sequence training? Why does it speed up training? What is the downside, and what is the name of the resulting training-inference mismatch problem?

---

**Q9.** You need to classify a sequence of 200 sensor readings as either "normal" or "anomalous." Compare using a:
- Vanilla RNN
- LSTM
- Transformer

For each, describe the key advantage/disadvantage for this specific task. Which would you choose as a first experiment and why?

---

**Q10.** An LSTM has: input size = 50, hidden size = 128. How many **learnable parameters** does one LSTM cell have? Show your calculation.

```
Each gate has: W_x of shape (hidden, input) + W_h of shape (hidden, hidden) + b of shape (hidden,)
LSTM has 4 such gate computations.
Total = 4 × (hidden × input + hidden × hidden + hidden)
```

---

**Q11.** The **output gate** in an LSTM controls what portion of the cell state is exposed as the hidden state. Write the output gate equations and explain why not simply exposing `C_t` directly would be less effective.

---

**Q12.** Describe the encoder-decoder (Seq2Seq) architecture for machine translation:
- What does the encoder produce?
- How does the decoder use it?
- What is the "bottleneck problem" with using a single context vector?
- How does the **attention mechanism** solve this bottleneck?

---

**Q13.** You are training an LSTM for next-word prediction on a text corpus. Describe two signs that indicate your model is overfitting, and three regularization techniques specifically useful for RNNs.

---

**Q14.** The **update gate** in a GRU is described as an "interpolation" between the previous hidden state and the candidate new state. Write the equation and explain geometrically/intuitively what setting `z_t = 0` vs `z_t = 1` means for information retention.

---

## Answers (Hidden — attempt questions before reading)

<details>
<summary>Click to reveal answer guidance</summary>

1. h_t = tanh(W_h h_{t-1} + W_x x_t + b). h_t is the hidden state at step t. It summarizes all past inputs into a fixed-size vector — "memory" because it carries information from previous steps to influence future outputs.

2. ∂h_T/∂h_t = product of (T-t) Jacobian matrices. If max eigenvalue < 1, gradient → 0 exponentially. Addition-based updates (like in LSTM cell state: C_t = f*C_{t-1} + i*C̃) have gradient 1 flowing through the addition, not multiplied by small derivatives.

3. C_t flows through the network via element-wise multiplication and addition — no matrix multiplication or non-linearity in the skip path. Gradient can flow back with minimal distortion. h_t goes through tanh(C_t), which can saturate.

4. f_t = σ(W_f [h_{t-1}, x_t] + b_f). Values near 0 = forget, near 1 = keep. Example: When a new sentence starts with a new subject, the forget gate erases the old subject's gender/number stored in the cell state.

5. LSTM: 3 gates (forget, input, output), separate cell state, more parameters. GRU: 2 gates (reset, update), no separate cell state, ~25% fewer params. Use GRU for speed/small data; LSTM for very long sequences or when you need the extra capacity.

6. BPTT unrolls the RNN through time and applies backprop to the entire unrolled graph. More memory and time than standard backprop. Truncated BPTT cuts the gradient flow after k steps to save memory and avoid vanishing gradients.

7. (a) Forward: left-to-right context. Backward: right-to-left context. (b) Concatenate at each step: h_t = [h_t_fwd; h_t_bwd]. (c) Named Entity Recognition — "Bank" in "Bank of England" needs future context. (d) Language generation / online streaming — future tokens not available.

8. Teacher forcing: at each decoder step, feed the ground truth token as input (not the model's previous prediction). Faster convergence. Downside: at inference, errors compound (exposure bias). Also: scheduled sampling as a fix.

9. Vanilla RNN: fast but fails for 200-step dependencies. LSTM: handles long dependencies well, good first choice. Transformer: excellent but needs more data and compute. Start with LSTM — proven for time series.

10. 4 × (128×50 + 128×128 + 128) = 4 × (6400 + 16384 + 128) = 4 × 22912 = 91,648 parameters.

11. o_t = σ(...), h_t = o_t ⊙ tanh(C_t). Directly exposing C_t would output raw cell values (unbounded). The output gate provides a learned, selective filter — not all cell contents are relevant at every step.

12. Encoder produces a context vector c = h_n (final hidden state). Decoder uses c as its initial state. Bottleneck: all input information compressed into one fixed-size vector — fails for long sequences. Attention: decoder attends to all encoder hidden states, computing a weighted context vector per decoder step.

13. Overfitting signs: training loss much lower than validation loss; validation perplexity increases while training perplexity decreases. RNN regularization: (1) Dropout on non-recurrent connections, (2) Zoneout (stochastically preserve hidden states), (3) L2 regularization on weights, (4) Reduce model size.

14. h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t. z_t=0: keep all of h_{t-1}, ignore new input (pure memory). z_t=1: completely replace with new candidate (no memory). Values between 0 and 1 blend the two.

</details>
