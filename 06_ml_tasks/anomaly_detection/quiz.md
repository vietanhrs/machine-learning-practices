# Anomaly Detection — Quiz

Test your understanding of anomaly detection concepts and algorithms.

---

**Q1.** What are the three types of anomalies? Give a real-world example of each.

<details>
<summary>Answer</summary>

1. **Point anomaly:** A single data point that is unusual compared to the rest.
   - Example: A credit card transaction of $10,000 when the customer's typical
     transactions are $10–$100.

2. **Contextual anomaly:** A data point that is only unusual given its context.
   - Example: A body temperature of 39°C (102°F) is abnormal for a person at rest
     but might be expected after intense exercise.

3. **Collective anomaly:** A group of related observations that is anomalous together,
   even if individual observations are not.
   - Example: A series of 30 small transactions of $9.99 each within 5 minutes is
     a collective anomaly indicative of card-testing fraud, even though $9.99 is a
     completely normal transaction amount.

</details>

---

**Q2.** What z-score threshold is commonly used to flag anomalies, and what statistical assumption underlies this choice?

- A) |z| > 1 (captures 68% of normal data)
- B) |z| > 2 (captures 95% of normal data)
- C) |z| > 3 (captures 99.7% of normal data)
- D) |z| > 5 (captures 99.9999% of normal data)

<details>
<summary>Answer</summary>

**C) |z| > 3.**

The z-score assumes the data follows a **Gaussian (normal) distribution**. Under a
Gaussian, approximately 99.7% of data falls within 3 standard deviations of the mean
(the "3-sigma rule"). Points beyond this threshold are considered statistically unusual.

**Caveat:** Real-world data is often not Gaussian. The IQR method is more robust because
it makes no distributional assumption and uses median-based statistics that are resistant
to the influence of outliers.

</details>

---

**Q3.** Explain how Isolation Forest isolates anomalies. Why does a shorter path length in the tree indicate an anomaly?

<details>
<summary>Answer</summary>

**Isolation Forest** works by building an ensemble of random "isolation trees":

1. Randomly select a feature.
2. Randomly select a split threshold between the feature's min and max values.
3. Recurse into left and right subtrees until each sample is isolated.

**Why shorter path = more anomalous:**

Anomalies are rare and occupy low-density regions of the feature space. Because they
are isolated from the bulk of the data, it takes very few random splits to separate
them into a leaf node. Normal points cluster together in dense regions and require
many splits to isolate.

- Anomalous point: path length ≈ 2–4 splits (easy to isolate)
- Normal point: path length ≈ 10–15 splits (surrounded by similar points)

The anomaly score normalizes path lengths by c(n), the expected path length of a random
BST, so scores are comparable across datasets of different sizes.

</details>

---

**Q4.** What is Local Outlier Factor (LOF) and how does it differ from global methods like z-score?

<details>
<summary>Answer</summary>

**LOF** is a **density-based, local** method that compares the local density of a point
to the local density of its k nearest neighbors.

- LOF ≈ 1: point has similar density to its neighbors → **normal**
- LOF >> 1: point is in a much less dense region than its neighbors → **anomaly**

**Difference from global methods:**

Global methods (z-score, IQR) compare each point to the global distribution. This
fails when data has **clusters of varying densities** — a point in a sparse cluster
would be flagged even if it fits well within that cluster's local density.

LOF adapts to local structure. A point only needs to be locally dense (similar to
its own neighbors) to be considered normal, regardless of the global density.

**Example:** In a dataset with two clusters — one dense (many transactions per minute)
and one sparse (few high-value transactions) — global z-score might flag all sparse-cluster
points, but LOF would recognize them as locally normal.

</details>

---

**Q5.** How does an autoencoder detect anomalies? What assumption does this method make about normal vs. anomalous data?

<details>
<summary>Answer</summary>

**Autoencoder anomaly detection:**

1. An autoencoder (encoder-bottleneck-decoder) is trained **only on normal data**.
2. It learns to compress and reconstruct normal patterns efficiently.
3. At inference, reconstruction error is computed for every sample:
   ```
   error(x) = ||x - decode(encode(x))||²
   ```
4. Normal samples have **low reconstruction error** (the model learned their patterns).
5. Anomalous samples have **high reconstruction error** (the model has never seen such
   patterns and cannot reconstruct them faithfully).

**Core assumption:** The bottleneck forces the autoencoder to generalize only the
patterns it has seen repeatedly (normal data). It cannot generalize to novel patterns
(anomalies) it was never trained on.

**Caveat:** If the autoencoder is too powerful (too many parameters, too large a
bottleneck), it may memorize everything and reconstruct anomalies well too. The
bottleneck must be constrained enough to force meaningful compression.

</details>

---

**Q6.** Why is anomaly detection difficult to evaluate? What metric is more appropriate than accuracy?

<details>
<summary>Answer</summary>

**Evaluation challenges:**

1. **Severe class imbalance:** Anomalies are typically <1% of the data. A model that
   labels everything as "normal" achieves >99% accuracy, making accuracy useless.

2. **Few labeled anomalies:** Obtaining ground-truth labels for anomalies requires
   domain experts and is expensive and time-consuming.

3. **Unknown anomaly types:** A model is often evaluated on anomaly types it has
   never seen during training, making generalization hard.

**Better metrics:**

- **AUC-ROC:** Threshold-independent measure of ranking performance. Good for comparing
  models when the threshold is not fixed.
- **AUC-PR (Average Precision):** Area under the Precision-Recall curve. Better than
  AUC-ROC for highly imbalanced datasets because it focuses on the positive class.
- **Precision@k:** Precision among the top-k highest-scored points. Useful when a
  human will review the top alerts.
- **F1 score at a chosen threshold:** Balances precision and recall for a specific
  operating point.

</details>

---

**Q7.** What is the key difference between **One-Class SVM** and **Isolation Forest** in terms of how they model normality?

<details>
<summary>Answer</summary>

| Aspect                | One-Class SVM                                  | Isolation Forest                             |
|-----------------------|------------------------------------------------|----------------------------------------------|
| Core idea             | Learn a tight **boundary** around normal data  | Exploit **path length** in random trees      |
| Model type            | Kernel-based (discriminative boundary)         | Ensemble of random trees                     |
| Scaling               | O(n²) training — slow on large datasets       | O(n log n) — scales well                    |
| High dimensions       | Can struggle (curse of dimensionality)         | Works reasonably well                        |
| Hyperparameters       | ν (contamination), kernel, γ — sensitive      | n_estimators, contamination — less sensitive |
| Normal assumption     | Compact region in kernel space                 | Dense clusters (many splits to isolate)      |

**When to prefer One-Class SVM:** When data is low-dimensional and you need a nonlinear
decision boundary; when you have very clean normal data without any contamination.

**When to prefer Isolation Forest:** Large datasets; high dimensions; when you want a
fast, relatively parameter-free method.

</details>

---

**Q8.** What is the difference between **semi-supervised** and **unsupervised** anomaly detection?

<details>
<summary>Answer</summary>

**Unsupervised anomaly detection:**
- No labels are available (neither normal nor anomaly labels).
- The model discovers anomalies by finding points that deviate from the majority.
- Examples: Isolation Forest, LOF, k-means-based methods, autoencoders trained on
  unlabeled data.
- **Risk:** The model assumes the majority of data is normal. If anomalies are frequent,
  this assumption breaks down.

**Semi-supervised anomaly detection:**
- Labels for **normal data** are available, but anomaly labels are scarce or absent.
- The model learns what "normal" looks like explicitly and flags deviations.
- Examples: One-Class SVM, autoencoder trained only on labeled normal data, SVDD.
- **Advantage:** More principled definition of normal; can adapt to specific normal
  class characteristics rather than relying on majority-vote assumptions.

**Fully supervised anomaly detection:**
- Labels for both normal and anomaly classes exist → standard binary classification.
- Rare in practice because anomaly labels are expensive to obtain and novel anomaly
  types may not appear in the training set.

</details>

---

**Q9.** A credit card fraud detection system flags 1,000 transactions as fraudulent. Of these, 100 are actual fraud. Meanwhile, 50 fraud cases were missed. Calculate Precision, Recall, and F1 score. Is this a good detector?

<details>
<summary>Answer</summary>

**Calculations:**

- True Positives (TP): 100 (fraud correctly flagged)
- False Positives (FP): 1000 - 100 = 900 (legitimate transactions falsely flagged)
- False Negatives (FN): 50 (fraud missed)

```
Precision = TP / (TP + FP) = 100 / 1000 = 0.10 (10%)
Recall    = TP / (TP + FN) = 100 / 150 ≈ 0.667 (66.7%)
F1        = 2 * (0.10 * 0.667) / (0.10 + 0.667) ≈ 0.174 (17.4%)
```

**Assessment:** Recall is decent (67% of fraud is caught), but Precision is very poor
(only 1 in 10 flagged transactions is actually fraud). This would overwhelm fraud
analysts with false alarms, reducing operational efficiency.

**Trade-off:** Raising the detection threshold would improve Precision but reduce Recall
(miss more fraud). The right balance depends on business cost of:
- False Negative: Missed fraud → customer loss + financial loss
- False Positive: Incorrectly blocked legitimate transaction → customer friction

</details>

---

**Q10.** What does the **contamination** parameter in sklearn's `IsolationForest` represent, and how do you choose it?

<details>
<summary>Answer</summary>

The **contamination** parameter specifies the expected proportion of anomalies in the
training data. It is used to set the decision threshold — the model flags the top
`contamination * n` samples (by anomaly score) as anomalies.

**How to choose it:**

1. **Domain knowledge:** If you know that fraud is approximately 0.5% of transactions,
   set `contamination=0.005`.

2. **Validation set:** If you have some labeled examples, sweep contamination values
   and maximize a metric like F1 or AUC on validation data.

3. **Default:** sklearn uses `contamination='auto'`, which sets the threshold such
   that the decision function returns 0 at the median of scores — not tied to a
   specific anomaly proportion.

**Important:** If you set contamination too high, the model will falsely flag normal
points. Too low, and it misses genuine anomalies.

</details>

---

**Q11.** In a dataset of 1 million network packets, only 100 are actual intrusions. Why is this class imbalance a problem for anomaly detection?

<details>
<summary>Answer</summary>

**Class imbalance ratio:** 100 intrusions vs. 999,900 normal packets (0.01% anomaly rate).

**Problems this creates:**

1. **Biased learning (for supervised methods):** A classifier trained on this data
   learns to always predict "normal" because it's correct 99.99% of the time.

2. **Metric misleading:** Accuracy is 99.99% even with a trivially useless model.
   The true positives (detected intrusions) contribute negligibly to accuracy.

3. **Insufficient signal:** 100 positive examples may not be enough to learn the
   diverse patterns of attacks. Attackers also evolve their methods.

4. **Threshold calibration:** With so few positives, any small shift in threshold
   causes large swings in recall.

**Solutions:**
- Use appropriate metrics (AUC-PR, F1, Precision@k instead of accuracy)
- Oversampling (SMOTE) or undersampling
- One-class / semi-supervised approaches that don't require anomaly labels
- Cost-sensitive learning (penalize false negatives more heavily)

</details>

---

**Q12.** The IQR method defines anomalies as points outside `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`. Why is the IQR method more **robust** than the z-score method?

<details>
<summary>Answer</summary>

**Robustness** refers to resistance to the influence of extreme values (outliers themselves).

**Z-score problem:**
The z-score uses the **mean (μ)** and **standard deviation (σ)**, both of which are
sensitive to outliers. If there are extreme outliers, they inflate σ and pull μ toward
them. This "masking effect" makes outliers appear less extreme (closer to the inflated μ),
potentially hiding them from the z-score detector.

**IQR advantage:**
The IQR method uses **Q1, Q3, and IQR** — all based on order statistics (percentiles).
These are **resistant statistics** that are not distorted by extreme values.
The median (50th percentile) is also unaffected by how extreme the outliers are.

**Example:**
Normal data: [10, 11, 12, 13, 14] → μ=12, σ=1.6
With outlier: [10, 11, 12, 13, 100] → μ=29.2, σ=37.8

With z-score, the value 100 now has z = (100-29.2)/37.8 ≈ 1.87, which may not exceed
the threshold of 3. With IQR, the fences barely move because Q1 and Q3 are robust,
so 100 is clearly flagged as anomalous.

</details>
