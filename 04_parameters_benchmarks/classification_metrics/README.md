# Classification Metrics

A comprehensive guide to evaluating classification models. Choosing the right metric is as important as choosing the right model — the wrong metric can make a bad model look great.

---

## 1. Confusion Matrix

The confusion matrix is the foundation of all classification metrics. It shows the counts of correct and incorrect predictions broken down by class.

### Structure (Binary Classification)

```
                  Predicted Positive    Predicted Negative
Actual Positive        TP                    FN
Actual Negative        FP                    TN
```

| Term | Name              | Meaning                                      |
|------|-------------------|----------------------------------------------|
| TP   | True Positive     | Model said YES, reality is YES               |
| TN   | True Negative     | Model said NO,  reality is NO                |
| FP   | False Positive    | Model said YES, reality is NO  (Type I Error)|
| FN   | False Negative    | Model said NO,  reality is YES (Type II Error)|

### Worked Example

Suppose we have an email spam detector evaluated on 100 emails:

```
                  Predicted Spam    Predicted Not Spam
Actual Spam            40                  10
Actual Not Spam         5                  45
```

- TP = 40  (correctly caught spam)
- FN = 10  (missed spam — landed in inbox)
- FP = 5   (legit email flagged as spam)
- TN = 45  (correctly passed through)

Total = 100, Correct = TP + TN = 85

---

## 2. Accuracy

**Formula:**

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Using the example above: `(40 + 45) / 100 = 0.85` → 85%

### When Accuracy is Misleading

Consider a cancer screening dataset: 990 healthy, 10 sick patients.

A model that **always predicts "healthy"** achieves:

```
Accuracy = 990 / 1000 = 99%
```

This looks excellent, but the model catches **zero cancer cases**. This is the **class imbalance problem**.

Rule of thumb: When classes are imbalanced (e.g., 90/10 or worse), **never rely on accuracy alone**.

---

## 3. Precision

**Formula:**

```
Precision = TP / (TP + FP)
```

"Of all the times the model said YES, how often was it actually YES?"

Spam example: `40 / (40 + 5) = 0.889` → 88.9%

### When Precision Matters

Use precision when **false positives are costly**:

- Spam filter: A FP means a legitimate email is deleted — bad user experience.
- Legal document review: Flagging innocent documents wastes expensive lawyer time.
- Fraud alerts: Blocking a genuine customer transaction causes frustration.

High precision = the model is conservative; when it says "positive" it is usually right.

---

## 4. Recall (Sensitivity)

**Formula:**

```
Recall = TP / (TP + FN)
```

"Of all the actual positives, how many did the model find?"

Spam example: `40 / (40 + 10) = 0.80` → 80%

### When Recall Matters

Use recall when **false negatives are costly**:

- Cancer detection: Missing a cancer case (FN) is far worse than a false alarm.
- Security intrusion detection: Missing an attack is catastrophic.
- Drug safety testing: Missing a dangerous side effect endangers lives.

High recall = the model casts a wide net; it rarely misses a positive.

### Precision-Recall Tradeoff

Lowering the classification threshold increases recall but decreases precision (and vice versa). There is almost always a tradeoff.

---

## 5. F1 Score

The F1 score is the **harmonic mean** of precision and recall. It balances both concerns into a single number.

**Formula:**

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
   = 2*TP / (2*TP + FP + FN)
```

Spam example: `2 * (0.889 * 0.80) / (0.889 + 0.80) = 0.842` → 84.2%

The harmonic mean is used (not arithmetic mean) because it punishes extreme imbalances between precision and recall. A model with precision=1.0 and recall=0.0 gets F1=0.0, not 0.5.

### F-Beta Score (Generalization)

When one concern outweighs the other, use F-beta:

```
F_beta = (1 + beta²) * (Precision * Recall) / (beta² * Precision + Recall)
```

- `beta > 1` → weights **recall** more heavily (use when FN is more costly, e.g., disease detection)
- `beta < 1` → weights **precision** more heavily (use when FP is more costly, e.g., spam)
- `beta = 1` → standard F1 (equal weight)

Common choices: F2 (recall twice as important), F0.5 (precision twice as important).

---

## 6. Specificity and Balanced Accuracy

**Specificity (True Negative Rate):**

```
Specificity = TN / (TN + FP)
```

"Of all actual negatives, how many did the model correctly identify as negative?"

This is the recall for the negative class. Spam example: `45 / (45 + 5) = 0.90`.

**Balanced Accuracy:**

```
Balanced Accuracy = (Recall + Specificity) / 2
                  = (TPR + TNR) / 2
```

Balanced accuracy is the arithmetic mean of sensitivity (recall) and specificity. It works well for **imbalanced datasets** because it gives equal weight to both classes regardless of their size.

Spam example: `(0.80 + 0.90) / 2 = 0.85`

---

## 7. ROC Curve (Receiver Operating Characteristic)

The ROC curve plots **True Positive Rate (TPR = Recall)** vs **False Positive Rate (FPR)** at every possible classification threshold.

```
TPR = TP / (TP + FN)    ← y-axis
FPR = FP / (FP + TN)    ← x-axis
```

### Reading the Curve

```
TPR
1.0 |          *---*
    |       *
    |     *
    |   *
    | *
0.0 *-----------*----> FPR
    0.0               1.0

    Diagonal = random classifier (AUC = 0.5)
    Top-left corner = perfect classifier (AUC = 1.0)
```

### AUC Interpretation

AUC (Area Under the Curve) summarizes the ROC curve in a single number:

| AUC  | Interpretation              |
|------|-----------------------------|
| 1.0  | Perfect classifier          |
| 0.9+ | Excellent                   |
| 0.8+ | Good                        |
| 0.7+ | Fair                        |
| 0.5  | Random guessing (no skill)  |
| <0.5 | Worse than random           |

AUC has a probabilistic interpretation: **the probability that the model ranks a random positive higher than a random negative**.

ROC curves are threshold-independent — they show performance across all possible operating points.

---

## 8. Precision-Recall Curve

Instead of TPR vs FPR, plot **Precision vs Recall** at every threshold.

### Why Use PR Curves?

ROC curves can be **overly optimistic** when the negative class is very large (many TN inflate the FPR denominator, making the curve look better than it is).

PR curves focus entirely on the positive class behavior — **better for imbalanced datasets** where the positive class is rare (e.g., fraud, rare disease).

```
Precision
1.0 *---*
    |     *
    |       *
    |         *
    |           *
0.0 *-----------*----> Recall
    0.0               1.0

    Top-right corner = perfect classifier
    Baseline = class prevalence (flat horizontal line)
```

**Average Precision (AP):** The area under the PR curve. Equivalent to the weighted mean of precisions achieved at each threshold.

---

## 9. Matthews Correlation Coefficient (MCC)

MCC is considered one of the most **robust single metrics** for binary classification, especially with imbalanced classes.

**Formula:**

```
MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

### Interpretation

| MCC  | Meaning                |
|------|------------------------|
| +1   | Perfect prediction     |
|  0   | No better than random  |
| -1   | Perfect inverse prediction |

MCC takes **all four cells** of the confusion matrix into account. It is equivalent to the Pearson correlation coefficient between the actual and predicted binary labels.

Unlike F1, MCC is **symmetric** — it treats both classes equally. A high MCC requires good performance on both positives and negatives.

---

## 10. Macro vs Micro vs Weighted Averaging (Multiclass)

When you have more than two classes, you need a strategy to aggregate per-class metrics.

### Setup

Suppose you have 3 classes with these per-class F1 scores:

| Class | F1   | Support (count) |
|-------|------|-----------------|
| Cat   | 0.90 | 500             |
| Dog   | 0.80 | 300             |
| Bird  | 0.50 | 200             |

### Macro Average

```
Macro F1 = (0.90 + 0.80 + 0.50) / 3 = 0.733
```

- Simple unweighted mean across classes.
- **Treats all classes equally**, regardless of support.
- Best when **every class is equally important**, even rare ones.
- Sensitive to poor performance on small classes.

### Micro Average

```
Micro F1 = 2*TP_total / (2*TP_total + FP_total + FN_total)
```

- Aggregates TP, FP, FN across all classes first, then computes F1.
- **Dominated by frequent classes**.
- Equivalent to accuracy when there are no multi-label predictions.
- Best when you care most about overall correct predictions.

### Weighted Average

```
Weighted F1 = (0.90*500 + 0.80*300 + 0.50*200) / 1000 = 0.800
```

- Weighted mean by class support.
- **Accounts for class imbalance** while still crediting large classes.
- Good default choice for imbalanced multiclass problems.

### When to Use Which?

| Scenario | Recommended Averaging |
|----------|----------------------|
| Balanced classes | Macro or Micro (similar results) |
| Imbalanced, every class matters | Macro |
| Imbalanced, overall accuracy matters | Micro or Weighted |
| Production system reporting | Weighted |

---

## Reference Links

- [scikit-learn Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Wikipedia: Receiver Operating Characteristic](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
- [Wikipedia: Matthews Correlation Coefficient](https://en.wikipedia.org/wiki/Phi_coefficient)
- [Wikipedia: Confusion Matrix](https://en.wikipedia.org/wiki/Confusion_matrix)
