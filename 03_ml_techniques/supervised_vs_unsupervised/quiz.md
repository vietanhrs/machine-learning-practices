# Quiz: Supervised vs Unsupervised Learning

Test your understanding of machine learning paradigms.

---

**Q1.** What is the fundamental difference between supervised and unsupervised learning?

<details>
<summary>Answer</summary>

Supervised learning requires **labeled training data** — each example has an input `X` and a target output `y`. The model learns a mapping `f: X → y` guided by a loss function comparing predictions to true labels.

Unsupervised learning works with **unlabeled data** — there are no target outputs. The model must discover hidden structure, patterns, or representations on its own without any explicit feedback signal.

</details>

---

**Q2.** Classify each of the following as supervised or unsupervised tasks:

- (a) Predicting the price of a house given its features
- (b) Grouping news articles by topic without knowing the topics in advance
- (c) Detecting fraudulent credit card transactions using historical labeled fraud records
- (d) Compressing a dataset from 100 dimensions to 2 for visualization
- (e) Classifying email as spam or not spam

<details>
<summary>Answer</summary>

- (a) **Supervised** — Regression. Target label is the price.
- (b) **Unsupervised** — Clustering (e.g., topic modeling with LDA or K-Means on TF-IDF vectors).
- (c) **Supervised** — Binary classification. Labels indicate fraud vs. legitimate.
- (d) **Unsupervised** — Dimensionality reduction (e.g., PCA or t-SNE). No labels used.
- (e) **Supervised** — Binary classification. Labels indicate spam vs. not spam.

</details>

---

**Q3.** Why is it generally harder to evaluate unsupervised learning models compared to supervised models?

<details>
<summary>Answer</summary>

In supervised learning, evaluation is straightforward: compare predictions to known ground truth labels using metrics like accuracy, F1, or MSE.

In unsupervised learning, **there are no ground truth labels** for the learned structure. Clusters, embeddings, and density estimates cannot be directly validated against a reference. Evaluation must rely on:

- **Intrinsic metrics**: Silhouette score, Davies-Bouldin index, inertia — these measure internal cluster quality without labels but may not reflect real-world meaningfulness.
- **Downstream task evaluation**: Use learned representations as features for a supervised task and measure performance there.
- **Human visual inspection**: Project to 2D and visually assess whether clusters make sense.
- **Proxy tasks**: Evaluate whether clusters align with known categories (using labels only for evaluation, not training).

None of these is as objective and reliable as direct supervised evaluation.

</details>

---

**Q4.** Give two concrete examples of self-supervised learning in NLP and explain how the labels are derived from the data.

<details>
<summary>Answer</summary>

**Example 1: BERT (Masked Language Modeling)**
- Input: A sentence like "The cat sat on the [MASK]."
- Task: Predict the masked word ("mat").
- Labels come from the original unmasked text. 15% of tokens are randomly masked, and the model learns to predict them.
- No human annotation required — the labels are derived automatically from the raw text corpus.

**Example 2: GPT (Causal Language Modeling / Next Token Prediction)**
- Input: "The quick brown fox"
- Task: Predict the next token "jumps".
- Labels are simply the next word in the sequence — the model predicts token `t+1` given tokens `1...t`.
- Training on billions of text documents with zero human-annotated labels.

Both models develop rich language understanding through these pretext tasks, enabling effective fine-tuning on downstream tasks with small labeled datasets.

</details>

---

**Q5.** When would you prefer semi-supervised learning over purely supervised learning? Give a realistic example.

<details>
<summary>Answer</summary>

Semi-supervised learning is preferred when **labeling is expensive, time-consuming, or requires specialized expertise**, but **unlabeled data is abundant and cheap to collect**.

**Realistic example — Medical Image Diagnosis:**
- A hospital has 500,000 chest X-ray images.
- Only 2,000 are annotated by radiologists (who are expensive and scarce).
- The remaining 498,000 are unlabeled.

A semi-supervised approach (e.g., FixMatch, MixMatch, or pseudo-labeling) can:
1. Train an initial model on the 2,000 labeled images.
2. Use the model's confident predictions on unlabeled images as pseudo-labels.
3. Retrain on the combined set, effectively leveraging all 500,000 images.

This typically outperforms training on only the 2,000 labeled examples, sometimes approaching the performance of having far more human labels.

</details>

---

**Q6.** What is the "inductive bias" of K-Means clustering? How does this bias affect which structures it can and cannot discover?

<details>
<summary>Answer</summary>

K-Means has a strong **spherical / isotropic cluster inductive bias**:

- It assumes clusters are **convex** and roughly **spherical** (equal-radius, equal-density).
- It minimizes within-cluster sum of squared Euclidean distances, which implicitly assumes Voronoi partitioning.
- It assumes **equal cluster sizes** (assigns each point to the nearest centroid).

**Consequences:**
- Works well: Gaussian blobs with similar sizes and variances.
- Fails: Elongated clusters, crescent-shaped clusters (two moons), clusters of very different densities, clusters with non-convex shapes.
- Fails: DBSCAN or Spectral Clustering are more appropriate for non-convex shapes.

The inductive bias of a learning algorithm determines what generalizations it can make. A bias that matches the true data structure leads to good results; a mismatch leads to poor clustering regardless of the data size.

</details>

---

**Q7.** Explain the difference between GPT pretraining and BERT pretraining in terms of the self-supervised learning objective.

<details>
<summary>Answer</summary>

**GPT — Causal (Auto-Regressive) Language Modeling:**
- Predicts the next token given all previous tokens: `P(wₜ | w₁, w₂, ..., wₜ₋₁)`
- Uses a **left-to-right (causal) attention mask** — each token can only attend to tokens before it.
- The entire sequence is processed in one forward pass; every token position produces a prediction.
- Best suited for **generation tasks** (text completion, story generation).

**BERT — Masked Language Modeling (MLM) + Next Sentence Prediction (NSP):**
- Randomly masks 15% of tokens and predicts the masked tokens.
- Uses **bidirectional attention** — each token can attend to all other tokens (both left and right context).
- Cannot be used for autoregressive generation since it sees the full context.
- Best suited for **understanding tasks** (classification, NER, QA) where full context is available.

**Key difference**: GPT is unidirectional (left context only), BERT is bidirectional (full context). GPT excels at generation; BERT excels at representation and understanding.

</details>

---

**Q8.** A data scientist trains a clustering model on customer purchase data. The silhouette score is high (0.72), but when the clusters are reviewed by the business team, they find the segments are not actionable. What does this illustrate about evaluation in unsupervised learning?

<details>
<summary>Answer</summary>

This illustrates the **disconnect between intrinsic evaluation metrics and real-world utility** in unsupervised learning.

The silhouette score measures how similar a point is to its own cluster compared to other clusters — a high score means clusters are geometrically well-separated in feature space. However, this says nothing about whether the clusters are:

- **Meaningful** in a business or domain sense
- **Actionable** (can the business treat each segment differently?)
- **Stable** (do they change with slight variations in data?)
- **Interpretable** (can human experts understand what each cluster represents?)

**Lessons:**
1. Always pair intrinsic metrics with domain expert review.
2. Define what "good" means for your specific use case before training.
3. Evaluate on downstream utility (e.g., do cluster-specific marketing campaigns outperform generic ones?).
4. Unsupervised learning requires human judgment in a way that supervised learning's objective metrics do not.

</details>

---

**Q9.** What is the key practical advantage of self-supervised pretraining followed by supervised fine-tuning over training a supervised model from scratch?

<details>
<summary>Answer</summary>

**Leveraging vast unlabeled data for representation learning:**

When training a supervised model from scratch, the model can only learn from the available labeled examples. If the labeled set is small, the model will likely overfit and generalize poorly.

Self-supervised pretraining allows the model to:
1. Learn rich, general-purpose representations from billions of unlabeled examples (web text, images, audio).
2. Capture syntax, semantics, visual features, and world knowledge at scale.

Then, fine-tuning on a small labeled dataset:
1. Adapts these representations to the specific task with far fewer labeled examples.
2. Requires much less computation and data than training from scratch.
3. Often significantly outperforms training from scratch with limited labels.

**Concrete example**: GPT-3 pretrained on 570GB of text achieves strong performance on many tasks with zero or few labeled examples (zero-shot / few-shot learning), which is impossible for models trained from scratch on small labeled datasets.

</details>

---

**Q10.** Why might reinforcement learning be inappropriate for a medical diagnosis task?

<details>
<summary>Answer</summary>

Reinforcement learning is inappropriate for most medical diagnosis tasks for several reasons:

1. **Delayed and sparse rewards**: RL agents learn from reward signals. In medicine, the "reward" (patient outcome) may come weeks or months after a diagnosis — making credit assignment extremely difficult.

2. **Exploration is dangerous**: RL requires exploration — trying actions to learn their consequences. In medicine, a wrong diagnosis or treatment to "explore" could harm or kill a patient. This is ethically unacceptable.

3. **Supervised data is available**: Medical datasets typically have labeled examples (diagnosis + outcome). Supervised learning is more appropriate when ground truth labels exist.

4. **Sample efficiency**: RL requires millions of interactions to learn. Each "interaction" in medicine is a real patient — there are not enough patients to train an RL agent from scratch.

5. **Interpretability and accountability**: Medical decisions require explainability. RL policies are often black boxes.

**Appropriate use of RL in medicine**: Treatment sequencing for chronic diseases (where exploration can be simulated), drug dosing optimization in controlled settings, or training in medical simulators before deployment.

</details>

---

**Q11.** Describe a scenario where semi-supervised learning could actually perform worse than purely supervised learning. What went wrong?

<details>
<summary>Answer</summary>

**Scenario: Distribution mismatch between labeled and unlabeled data**

Suppose you are training a sentiment classifier:
- Labeled data: 1,000 labeled movie reviews (in-domain)
- Unlabeled data: 100,000 unlabeled tweets (different domain, different language style)

If a semi-supervised method generates pseudo-labels for the tweets and retrains, the model will:
1. Be exposed to a different data distribution during pseudo-label training.
2. Potentially degrade its representations for the original domain.
3. If pseudo-labels are noisy (tweets have different sentiment patterns), the model reinforces errors.

**Other failure modes:**
- **Confirmation bias in self-training**: The model assigns high-confidence (but wrong) pseudo-labels to hard examples, reinforcing its own errors.
- **Class imbalance in unlabeled data**: If unlabeled data has very different class proportions, pseudo-labels skew the training distribution.
- **Poor initial model**: If the labeled set is too small for a useful initial model, pseudo-labels will be mostly wrong and hurt performance.

**Lesson**: Semi-supervised learning assumes the unlabeled data comes from the same distribution as labeled data. Violating this assumption can make things worse.

</details>

---

**Q12.** What does it mean for a model to have high bias vs high variance? Relate these concepts to the supervised vs unsupervised distinction.

<details>
<summary>Answer</summary>

**High Bias (Underfitting)**:
- The model is too simple to capture the true relationship in the data.
- Training error and test error are both high.
- The model makes systematic errors regardless of which training data it sees.
- Example: Fitting a linear model to a nonlinear relationship.

**High Variance (Overfitting)**:
- The model is too complex and memorizes the training data, including noise.
- Training error is low but test error is high.
- Small changes in training data lead to large changes in the model.
- Example: A deep decision tree with no depth limit on a small dataset.

**Relation to supervised vs unsupervised:**

In supervised learning, bias-variance tradeoff is directly observable via train/test error comparison. Regularization, cross-validation, and learning curves help diagnose and correct it.

In unsupervised learning, the bias-variance tradeoff is harder to assess:
- An overcomplicated clustering model (e.g., K-Means with K=100 on a simple dataset) overfits to noise — high "variance" in cluster assignments.
- An undercomplicated model (K=2 on data with 10 natural groups) misses structure — high "bias".
- Without labels, we cannot directly measure generalization error, so diagnosing bias vs variance requires intrinsic metrics, domain knowledge, or downstream evaluation.

</details>

---

*End of Quiz — 12 Questions*
