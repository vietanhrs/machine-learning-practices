# Quiz: Feedforward Neural Networks

Answer each question in your own words. For calculation questions, show your work.

---

**Q1.** What makes a neural network "deep"? How is a deep network different from a shallow one (single hidden layer), and why does depth matter for learning complex functions?

---

**Q2.** A network without activation functions between layers is mathematically equivalent to a single linear transformation, no matter how many layers it has. Explain why this is true, and what property activation functions add to break this limitation.

---

**Q3.** The sigmoid function outputs values in `(0, 1)` and has a maximum derivative of `0.25`. Explain step by step why this causes the vanishing gradient problem in networks with many layers. What happens to the gradient signal as it passes through 10 sigmoid layers?

---

**Q4.** ReLU solves the vanishing gradient problem for positive inputs, but introduces a new problem known as "dying ReLU." Describe what dying ReLU is, what causes it, and name two activation function variants that address it.

---

**Q5.** Trace through the forward pass of the following 2-layer network:
- Input: `x = [1.0, 2.0]`
- Layer 1: `W1 = [[0.5, -0.5], [0.3, 0.2]]`, `b1 = [0, 0]`, activation: ReLU
- Layer 2: `W2 = [[1.0, -1.0]]`, `b2 = [0]`, activation: Sigmoid

Compute `z1`, `a1`, `z2`, and the final output `a2`. Show each step.

---

**Q6.** Backpropagation computes gradients efficiently using the chain rule. In your own words, what quantity does backprop actually compute, and why is it more efficient than computing gradients naively for each weight one at a time?

---

**Q7.** What is the role of the **bias term** `b` in a neuron? What would happen if all biases were forced to zero? Give a concrete example where a bias is necessary.

---

**Q8.** The Universal Approximation Theorem states that a single hidden layer MLP can approximate any continuous function. Does this mean we should always prefer shallow networks? Explain the practical tradeoffs between width (more neurons in one layer) and depth (more layers).

---

**Q9.** For a **regression** task (predicting a continuous value like house price) versus a **binary classification** task (spam vs. not spam) versus a **10-class classification** task (digit recognition), which activation function should you use at the output layer in each case? Justify your choices.

---

**Q10.** Explain the difference between **batch gradient descent**, **stochastic gradient descent (SGD)**, and **mini-batch gradient descent**. How does batch size affect:
- (a) the quality/variance of gradient estimates
- (b) training speed (wall-clock time)
- (c) memory requirements

---

**Q11.** He initialization sets weights from a normal distribution with variance `2 / n_in`, where `n_in` is the number of input connections. Why is careful weight initialization important, and why does He initialization use a factor of `2` specifically (rather than `1`)?

---

**Q12.** Describe the **forward pass** and **backward pass** in terms of what information flows and in which direction during each. What is stored during the forward pass and why is it needed for backpropagation?

---

**Q13.** Batch Normalization is inserted between the linear transformation and the activation function. Explain what it does mathematically and describe two benefits it provides for training deep networks beyond solving vanishing gradients.

---

**Q14.** A student trains an MLP on a training set and achieves 99% training accuracy but only 62% validation accuracy. Identify this problem and list four concrete techniques to address it, explaining the mechanism of each.

---

## Answers (Hidden — attempt questions before reading)

<details>
<summary>Click to reveal answer guidance</summary>

1. "Deep" means two or more hidden layers. Depth allows hierarchical feature learning (edges → shapes → objects) and is exponentially more parameter-efficient than width for many functions.

2. Composing linear functions gives a linear function: `W2(W1x + b1) + b2 = (W2W1)x + (W2b1 + b2)`. Activation functions are non-linear, breaking this collapse.

3. Gradient at layer `k` from output = product of all derivatives. With sigmoid max derivative 0.25: `0.25^10 ≈ 9.5 × 10^-7`. The gradient vanishes exponentially.

4. Dying ReLU: neurons where `z < 0` output 0 and receive zero gradient — they never recover. Causes: high learning rate, large negative biases. Fixes: Leaky ReLU (small negative slope), ELU.

5. z1 = [1·0.5 + 2·(-0.5), 1·0.3 + 2·0.2] = [-0.5, 0.7]; a1 = ReLU([-0.5, 0.7]) = [0, 0.7]; z2 = 1·0 + (-1)·0.7 = -0.7; a2 = σ(-0.7) ≈ 0.332.

6. Backprop computes `∂L/∂W` for every weight simultaneously using the chain rule, reusing intermediate computations. Naive approach would require a separate forward pass per weight (O(N) passes vs O(1)).

7. Bias shifts the activation threshold independently of inputs. Without bias, the decision boundary must pass through the origin. Example: learning `y = 1 if x > 3` requires a bias of `-3·w`.

8. Width can theoretically approximate any function but may require exponentially many neurons. Depth composes simpler functions hierarchically and is far more parameter-efficient in practice.

9. Regression: linear (no activation) or ReLU if output must be positive. Binary classification: Sigmoid. 10-class: Softmax.

10. (a) Larger batch → lower variance, smoother gradients. (b) Larger batch → more compute per step but fewer steps needed; smaller batch → more updates, faster early progress. (c) Larger batch → more GPU memory.

11. Poor initialization can cause activations to saturate (all near 0 or 1) at the start, producing zero gradients. Factor of 2 compensates for ReLU zeroing out half of inputs on average.

12. Forward: input → activations stored at each layer → loss. Backward: loss gradient → back through layers using stored activations → weight gradients. Stored activations are needed to compute `∂Z/∂W = A_prev`.

13. BN normalizes each feature to zero mean and unit variance per mini-batch, then applies learnable scale/shift. Benefits: faster convergence, acts as regularization, reduces sensitivity to learning rate.

14. Overfitting. Fixes: (1) Dropout — randomly zeros neurons, reducing co-adaptation. (2) L2 regularization — penalizes large weights. (3) More data / data augmentation. (4) Reduce model capacity (fewer layers/neurons). (5) Early stopping.

</details>
