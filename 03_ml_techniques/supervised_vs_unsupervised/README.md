# Supervised vs Unsupervised Learning

A comprehensive guide to the major machine learning paradigms: how they differ, when to use each, and what problems they solve.

---

## 1. Supervised Learning

### Definition

Supervised learning is a paradigm where the model is trained on **labeled data** — every training example consists of an input `X` and a corresponding output label `y`. The model learns to map inputs to outputs by minimizing a loss function that measures prediction error.

### Key Characteristics

- **Labeled data required**: A human (or automated process) must annotate the training set.
- **Explicit feedback**: The model receives a clear signal (error) on every prediction.
- **Goal**: Learn a function `f: X → y` that generalizes to unseen inputs.

### Training Process

```
Training Data: {(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)}
              ↓
         Model f(x; θ)
              ↓
    Loss = L(f(xᵢ; θ), yᵢ)
              ↓
    Minimize Loss via Optimization
              ↓
    Evaluate on Held-Out Test Set
```

### Task Types

#### Classification
Predict a **discrete category** label.

- Binary classification: spam vs. not spam, disease vs. healthy
- Multi-class: handwritten digit recognition (0–9), image categorization
- Multi-label: a photo can have tags "beach", "sunset", "people" simultaneously

#### Regression
Predict a **continuous numeric** value.

- House price prediction
- Stock price forecasting
- Age estimation from a photo

### Common Algorithms

| Algorithm | Classification | Regression |
|-----------|:--------------:|:----------:|
| Linear/Logistic Regression | Yes | Yes |
| Decision Trees | Yes | Yes |
| Random Forest | Yes | Yes |
| Support Vector Machines | Yes | Yes |
| Neural Networks | Yes | Yes |
| k-Nearest Neighbors | Yes | Yes |

### Evaluation Metrics

- **Classification**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Regression**: MSE, MAE, RMSE, R²

---

## 2. Unsupervised Learning

### Definition

Unsupervised learning works with **unlabeled data** — there are no target outputs `y`. The model must discover hidden patterns, structures, or representations within the data on its own.

### Key Characteristics

- **No labels needed**: Cheaper and easier to collect large datasets.
- **No explicit feedback**: The model has no direct error signal.
- **Goal**: Find structure — clusters, manifolds, latent representations, or generative models.

### Why Unsupervised is Harder

Without ground truth labels, it is difficult to objectively measure whether the discovered structure is meaningful. Evaluation often relies on:
- **Intrinsic metrics** (e.g., silhouette score, Davies-Bouldin index)
- **Downstream task performance** (use learned representations for classification)
- **Human visual inspection** (for low-dimensional projections)

### Task Types

#### Clustering
Group similar data points together without knowing the group labels in advance.

- K-Means: partition data into K clusters by minimizing within-cluster variance
- DBSCAN: density-based clusters, handles noise and arbitrary shapes
- Hierarchical clustering: build a tree (dendrogram) of nested clusters

#### Dimensionality Reduction
Map high-dimensional data to a lower-dimensional space while preserving structure.

- **PCA** (Principal Component Analysis): linear projection preserving maximum variance
- **t-SNE**: nonlinear, preserves local neighborhoods — excellent for visualization
- **UMAP**: faster than t-SNE, preserves more global structure
- **Autoencoders**: neural network encoder-decoder that learns a compressed representation

#### Density Estimation
Model the probability distribution of the data.

- Gaussian Mixture Models (GMM)
- Kernel Density Estimation (KDE)

#### Anomaly Detection
Identify data points that deviate significantly from learned normal patterns.

- Isolation Forest
- One-class SVM
- Autoencoder reconstruction error

### Common Algorithms

| Algorithm | Task |
|-----------|------|
| K-Means | Clustering |
| DBSCAN | Clustering |
| PCA | Dimensionality Reduction |
| t-SNE | Visualization |
| UMAP | Dimensionality Reduction |
| Autoencoder | Representation Learning |
| GMM | Density Estimation |
| Isolation Forest | Anomaly Detection |

---

## 3. Semi-Supervised Learning

### Definition

Semi-supervised learning uses a **small amount of labeled data combined with a large amount of unlabeled data**. This reflects many real-world scenarios where labeling is expensive or time-consuming but raw data is abundant.

### Core Idea

Labeled data guides the model toward correct predictions, while unlabeled data provides additional structural information about the data distribution.

### Methods

- **Self-Training**: Train on labeled data, then use the model's confident predictions on unlabeled data as pseudo-labels, and retrain.
- **Label Propagation**: Propagate labels through a graph based on data point similarity.
- **Consistency Regularization**: Augmented versions of the same input should produce consistent predictions (e.g., MixMatch, FixMatch).
- **Co-Training**: Train two models on different views of the data; each labels data for the other.

### When to Use

- Medical imaging: Few annotated scans, millions of unannotated images
- Natural language: Small labeled corpus, vast unlabeled web text
- Any domain where labeling requires domain expertise

---

## 4. Self-Supervised Learning

### Definition

Self-supervised learning is a form of unsupervised learning where **the labels are automatically derived from the data itself** — no human annotation required. The model is trained on a pretext task whose labels come from the raw data.

### Key Idea

Design a task where the answer is hidden within the data:

- Mask some words in a sentence → predict the masked words (BERT)
- Predict the next word given previous words (GPT)
- Predict a missing image patch given surrounding patches
- Predict whether two image crops come from the same image (contrastive learning)

### Examples

| Model | Modality | Pretext Task |
|-------|----------|-------------|
| BERT | Text | Masked Language Modeling (MLM) |
| GPT | Text | Next token prediction (causal LM) |
| Word2Vec | Text | Skip-gram / CBOW |
| SimCLR | Images | Contrastive learning of augmented views |
| MAE | Images | Masked Autoencoder (reconstruct masked patches) |
| wav2vec 2.0 | Audio | Contrastive learning on speech segments |

### Why It Matters

Self-supervised pretraining allows models to learn rich general-purpose representations from massive unlabeled corpora, then **fine-tune** on small labeled datasets for downstream tasks. This is the foundation of modern large language models and vision transformers.

### Relationship to Unsupervised Learning

Self-supervised learning is technically unsupervised (no human labels), but it uses a supervised training objective derived from the data structure. It bridges the gap between unsupervised and supervised learning.

---

## 5. Reinforcement Learning

### Definition

Reinforcement Learning (RL) is a fundamentally different paradigm where an **agent** learns to take **actions** in an **environment** to maximize cumulative **reward** over time. There are no labeled examples — feedback comes from the consequences of actions.

### Core Components

```
Agent ──── Action ────→ Environment
  ↑                          |
  └──── State + Reward ───────┘
```

- **Agent**: The learner/decision-maker
- **Environment**: The world the agent interacts with
- **State (s)**: Current observation of the environment
- **Action (a)**: Choice made by the agent
- **Reward (r)**: Scalar feedback signal from the environment
- **Policy (π)**: Mapping from states to actions (what the agent learns)

### Key Algorithms

- **Q-Learning / DQN**: Learn action-value function Q(s, a)
- **Policy Gradient / REINFORCE**: Directly optimize the policy
- **PPO / A3C**: Modern actor-critic methods
- **AlphaGo / AlphaZero**: RL + Monte Carlo Tree Search

### Applications

- Game playing (Atari, Go, Chess)
- Robot locomotion and manipulation
- Autonomous driving
- Recommendation systems
- RLHF (Reinforcement Learning from Human Feedback) — fine-tuning LLMs

---

## 6. Comparison Table

| Paradigm | Data Required | Goal | Evaluation | Key Examples |
|----------|--------------|------|------------|-------------|
| **Supervised** | Labeled (X, y) pairs | Learn f: X → y | Accuracy, F1, MSE on test set | Classification, Regression |
| **Unsupervised** | Unlabeled X only | Discover structure | Silhouette, visual inspection, downstream task | Clustering, PCA, Autoencoders |
| **Semi-Supervised** | Small labeled + large unlabeled | Learn from both | Same as supervised on labeled test set | FixMatch, Label Propagation |
| **Self-Supervised** | Unlabeled X (labels from data) | Learn representations | Downstream fine-tune performance | BERT, GPT, SimCLR, MAE |
| **Reinforcement** | Environment interactions | Maximize reward | Episode reward, win rate | DQN, PPO, AlphaZero |

---

## 7. When to Use Each

### Use Supervised Learning when:
- You have sufficient labeled data (or labeling is cheap)
- You need high accuracy on a well-defined prediction task
- Ground truth labels are unambiguous and reliable

### Use Unsupervised Learning when:
- Labeling is expensive, time-consuming, or impossible
- You want to explore unknown structure in data
- You need anomaly detection without known anomaly examples
- You want to reduce dimensionality for visualization or downstream tasks

### Use Semi-Supervised Learning when:
- You have a small labeled set but large unlabeled pool
- Labeling requires expert knowledge (medical, legal domains)
- Adding unlabeled data improves model robustness

### Use Self-Supervised Learning when:
- You have large unlabeled corpora (text, images, audio)
- You want to pretrain a general-purpose representation
- Downstream labeled data is scarce but pretraining data is abundant
- You are working in NLP (BERT, GPT family) or computer vision (MAE, DINO)

### Use Reinforcement Learning when:
- The task is sequential decision-making
- Feedback comes from the environment, not labeled data
- The goal is to maximize long-term reward, not predict a single output
- Exploration of the solution space is necessary

---

## 8. Reference Links

- [Wikipedia: Supervised Learning](https://en.wikipedia.org/wiki/Supervised_learning)
- [Wikipedia: Unsupervised Learning](https://en.wikipedia.org/wiki/Unsupervised_learning)
