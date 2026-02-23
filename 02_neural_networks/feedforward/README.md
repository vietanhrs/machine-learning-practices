# Feedforward Neural Networks (Lan truyền thuận)

A feedforward neural network — also called a Multi-Layer Perceptron (MLP) — is the simplest and most fundamental type of artificial neural network. Data flows in one direction only: from input to output, with no cycles or feedback loops.

---

## 1. What is a Feedforward Neural Network (MLP)?

A **Multi-Layer Perceptron (MLP)** consists of an ordered sequence of layers where every neuron in one layer is fully connected to every neuron in the next layer. The "feedforward" name indicates that information flows strictly forward — there is no recurrence or memory.

Key properties:
- **Universal function approximators** (see Section 7).
- Trained via **gradient descent** and **backpropagation**.
- Foundation for more advanced architectures (CNNs, RNNs, Transformers).

A network is called **deep** when it has two or more hidden layers.

---

## 2. Neurons and Layers

### The Artificial Neuron
Each neuron computes:

```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b  =  wᵀx + b
a = f(z)
```

where `w` are weights, `b` is a bias, `x` is the input vector, and `f` is an **activation function**.

### Layer Types

| Layer | Role |
|-------|------|
| **Input layer** | Receives raw features. No computation — just passes data forward. Size = number of features. |
| **Hidden layer(s)** | Learn intermediate representations. Each applies a linear transformation then an activation. |
| **Output layer** | Produces the final prediction. Activation depends on the task (see Section 3). |

### Architecture notation
A network `[784 → 256 → 128 → 10]` means:
- Input: 784 features
- Hidden 1: 256 neurons
- Hidden 2: 128 neurons
- Output: 10 classes

---

## 3. Activation Functions

Without activation functions, stacking linear layers collapses to a single linear transformation. Activation functions introduce **non-linearity**, enabling the network to learn complex patterns.

### Sigmoid

```
σ(z) = 1 / (1 + e^(-z))     range: (0, 1)
σ'(z) = σ(z) · (1 − σ(z))
```

- **Use case**: Binary classification output layer.
- **Problem**: Saturates at extremes → vanishing gradients in deep networks.

### Tanh

```
tanh(z) = (e^z − e^(-z)) / (e^z + e^(-z))     range: (−1, 1)
tanh'(z) = 1 − tanh²(z)
```

- **Use case**: Hidden layers in shallow networks; RNNs.
- Zero-centered (better than sigmoid for hidden layers), but still saturates.

### ReLU (Rectified Linear Unit)

```
ReLU(z) = max(0, z)
ReLU'(z) = 1 if z > 0, else 0
```

- **Use case**: Hidden layers in most deep networks.
- Fast to compute, does not saturate for positive values.
- **Problem**: "Dying ReLU" — neurons with z < 0 always output 0 and receive no gradient.

### Softmax

```
Softmax(zᵢ) = e^(zᵢ) / Σⱼ e^(zⱼ)     range: (0, 1), sums to 1
```

- **Use case**: Multi-class classification output layer.
- Converts raw logits into a probability distribution.

### Summary Table

| Function | Range | Use Case | Problem |
|----------|-------|----------|---------|
| Sigmoid | (0, 1) | Binary output | Vanishing gradients |
| Tanh | (−1, 1) | Hidden (shallow) | Saturates |
| ReLU | [0, ∞) | Hidden (deep) | Dying neurons |
| Softmax | (0,1), Σ=1 | Multi-class output | — |

---

## 4. Forward Pass

The forward pass computes predictions by applying each layer's linear transformation and activation in sequence.

### Matrix Form (one layer)

```
Z = X · Wᵀ + b      (linear step)
A = f(Z)             (activation step)
```

where `X` is shape `(batch_size, n_in)`, `W` is shape `(n_out, n_in)`, `b` is shape `(n_out,)`.

### Step-by-Step Example

Consider a tiny network: `[2 → 3 → 1]` with ReLU hidden and sigmoid output.

**Input:** `x = [0.5, -0.3]`

**Layer 1 weights:**
```
W1 = [[0.1, 0.4],    b1 = [0.0, 0.1, -0.1]
      [0.2, -0.3],
      [-0.1, 0.5]]
```

Step 1: Linear transform
```
z1 = W1 @ x + b1
   = [0.1·0.5 + 0.4·(−0.3) + 0.0,
      0.2·0.5 + (−0.3)·(−0.3) + 0.1,
      (−0.1)·0.5 + 0.5·(−0.3) − 0.1]
   = [−0.07, 0.29, −0.30]
```

Step 2: ReLU activation
```
a1 = ReLU(z1) = [0.0, 0.29, 0.0]
```

Step 3: Output layer (sigmoid)
```
z2 = W2 @ a1 + b2     →     a2 = σ(z2)  ∈ (0, 1)
```

This is the network's prediction (probability for binary classification).

---

## 5. Backpropagation

Backpropagation is an efficient algorithm that computes gradients of the loss function with respect to every weight in the network using the **chain rule** of calculus.

### Chain Rule Intuition

If `L` depends on `a`, `a` depends on `z`, and `z` depends on `w`:

```
∂L/∂w = (∂L/∂a) · (∂a/∂z) · (∂z/∂w)
```

Backprop applies this recursively, propagating the gradient signal from the output layer back through each layer.

### Gradient Flow

```
Forward:   X → Z1 → A1 → Z2 → A2 → ... → Loss
Backward:  ∂L/∂W1 ← ∂L/∂Z1 ← ∂L/∂A1 ← ∂L/∂Z2 ← ...
```

### Weight Update (Gradient Descent)

```
W ← W − η · ∂L/∂W
b ← b − η · ∂L/∂b
```

where `η` (eta) is the **learning rate**.

### Key Gradients

For a layer with sigmoid activation and binary cross-entropy loss:

```
δ_output = a_output − y              (loss gradient at output)
δ_hidden = (W_next.T @ δ_next) * f'(z_hidden)
∂L/∂W = δ.T @ a_prev
∂L/∂b = sum(δ, axis=0)
```

---

## 6. Vanishing Gradient Problem

### The Problem

During backprop, gradients are multiplied together as they flow backward through layers. Activation functions like sigmoid and tanh have derivatives in `(0, 0.25]`. Multiplied across many layers:

```
∂L/∂W1 ≈ 0.25^L  (L = number of layers)
```

For `L = 10`, the gradient is `≈ 10^(-6)` — effectively zero. Early layers learn extremely slowly or not at all.

### Solutions

| Solution | How it Helps |
|----------|-------------|
| **ReLU activation** | Gradient is 1 for positive values — no shrinkage. |
| **Batch Normalization** | Normalizes layer inputs, stabilizes gradient magnitudes. |
| **Skip/Residual Connections** | Gradient can flow directly through shortcut paths (ResNet). |
| **Careful initialization** | He/Xavier initialization prevents activations from saturating at start. |
| **Gradient clipping** | Prevents exploding gradients in RNNs. |

### Batch Normalization

For a mini-batch `{x₁, ..., xₘ}`:
```
μ_B = (1/m) Σ xᵢ
σ²_B = (1/m) Σ (xᵢ − μ_B)²
x̂ᵢ = (xᵢ − μ_B) / √(σ²_B + ε)
yᵢ = γ · x̂ᵢ + β
```

where `γ` and `β` are learnable scale and shift parameters.

### Residual (Skip) Connections

```
output = F(x) + x
```

The identity shortcut ensures gradients can flow unimpeded through the skip path.

---

## 7. Universal Approximation Theorem

**Statement:** A feedforward network with a single hidden layer containing a finite number of neurons can approximate any continuous function on a compact subset of ℝⁿ to arbitrary precision, given a non-polynomial activation function.

**Intuition:** Each neuron carves out a "bump" or "step" in the function space. With enough neurons, you can approximate any smooth function by combining these bumps — similar to how Fourier series approximate functions with sine waves.

**Practical implications:**
- MLPs are theoretically powerful enough to learn any mapping.
- However, shallow networks may need exponentially many neurons.
- **Depth** provides an exponential advantage in representational efficiency.
- Having expressive capacity does not guarantee learning — optimization and generalization matter too.

---

## 8. Reference Links

- [Feedforward Neural Network — Wikipedia](https://en.wikipedia.org/wiki/Feedforward_neural_network)
- [Build the Neural Network — PyTorch Official Tutorial](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)
