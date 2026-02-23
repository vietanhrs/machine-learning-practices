# Classification Metrics — Quiz

Test your understanding of evaluation metrics for classification models. Try to answer each question before revealing the answer.

---

**Q1.** A model predicts 80 emails as spam and 120 as not spam. Of the 80 spam predictions, 60 are actually spam. Of the 120 not-spam predictions, 10 are actually spam. Fill in TP, TN, FP, FN.

<details>
<summary>Answer</summary>

- TP = 60 (predicted spam, actually spam)
- FP = 20 (predicted spam, actually not spam)
- FN = 10 (predicted not spam, actually spam)
- TN = 110 (predicted not spam, actually not spam)

Total = 200 emails.
</details>

---

**Q2.** A hospital dataset has 990 healthy patients and 10 sick patients. A model that always predicts "healthy" achieves 99% accuracy. What is its recall for the sick class? Why does this matter?

<details>
<summary>Answer</summary>

Recall for sick class = TP / (TP + FN) = 0 / (0 + 10) = **0%**.

The model catches zero sick patients despite 99% accuracy. This illustrates why accuracy is a **misleading metric for imbalanced datasets**. Use recall, F1, or AUC instead.
</details>

---

**Q3.** Define Precision in plain English. Give one real-world scenario where maximizing precision is the right objective.

<details>
<summary>Answer</summary>

Precision = "Of all positive predictions, what fraction are actually positive?"

Formula: `TP / (TP + FP)`

Scenario: **Legal document review** — if the system flags documents for expensive manual review, false positives waste attorney time. High precision means fewer wasted reviews.

Other examples: spam filtering (FP = good email deleted), fraud alerts (FP = customer's card wrongly blocked).
</details>

---

**Q4.** Define Recall (Sensitivity) in plain English. Give one real-world scenario where maximizing recall is the right objective.

<details>
<summary>Answer</summary>

Recall = "Of all actual positives, what fraction did the model find?"

Formula: `TP / (TP + FN)`

Scenario: **Cancer detection** — missing a cancer case (FN) is far more dangerous than a false alarm that triggers a follow-up biopsy. High recall means few cases are missed.

Other examples: security intrusion detection, drug side-effect screening.
</details>

---

**Q5.** Explain the Precision-Recall tradeoff. How does lowering the classification threshold affect each metric?

<details>
<summary>Answer</summary>

Lowering the threshold (e.g., from 0.5 to 0.3) means the model predicts "positive" more often:

- **Recall increases**: Fewer actual positives are missed (fewer FN).
- **Precision decreases**: More negative cases are incorrectly flagged (more FP), diluting positive predictions.

Raising the threshold has the opposite effect: precision increases but recall decreases. This fundamental tradeoff means you must choose a threshold based on the cost of FP vs FN in your application.
</details>

---

**Q6.** Write the F1 score formula. Why is the harmonic mean used instead of the arithmetic mean?

<details>
<summary>Answer</summary>

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

The **harmonic mean** punishes extreme imbalances between precision and recall. Consider:
- Precision = 1.0, Recall = 0.01
- Arithmetic mean = 0.505 (looks decent)
- Harmonic mean (F1) = 0.0198 (correctly reflects failure)

A model must perform well on **both** precision and recall to achieve a high F1. The arithmetic mean would give misleadingly high scores when one value is near 0.
</details>

---

**Q7.** When should you use F-beta with `beta = 2` versus `beta = 0.5`? Write the F-beta formula.

<details>
<summary>Answer</summary>

```
F_beta = (1 + beta²) * (Precision * Recall) / (beta² * Precision + Recall)
```

- **beta = 2** (F2): Weights recall twice as much as precision. Use when **false negatives are more costly** than false positives (e.g., disease detection, security alerts). You prefer missing fewer positives even at the cost of more false alarms.

- **beta = 0.5** (F0.5): Weights precision twice as much as recall. Use when **false positives are more costly** (e.g., spam filter, recommendation systems where irrelevant suggestions hurt UX).
</details>

---

**Q8.** What does an AUC-ROC score of 0.5 mean? What does 1.0 mean? Can AUC be below 0.5?

<details>
<summary>Answer</summary>

- **AUC = 0.5**: The model performs **no better than random guessing**. Its ROC curve lies along the diagonal. The model has no discriminative ability.

- **AUC = 1.0**: **Perfect classifier**. Every positive is ranked above every negative. The ROC curve reaches the top-left corner (TPR=1, FPR=0).

- **AUC < 0.5**: Yes — this means the model is **systematically wrong** (worse than random). Flipping all predictions would give AUC = 1 - original_AUC. In practice, this often indicates a label encoding error.

AUC has a probabilistic interpretation: P(model ranks a random positive higher than a random negative).
</details>

---

**Q9.** What is the key difference between a ROC curve and a Precision-Recall curve? When should you prefer the PR curve?

<details>
<summary>Answer</summary>

| | ROC Curve | PR Curve |
|--|-----------|----------|
| X-axis | FPR = FP / (FP + TN) | Recall = TP / (TP + FN) |
| Y-axis | TPR = TP / (TP + FN) | Precision = TP / (TP + FP) |
| Baseline | Diagonal (AUC = 0.5) | Horizontal at class prevalence |
| TN involvement | Yes (FPR uses TN) | No |

**Prefer PR curves when:**
- The dataset is **highly imbalanced** (many more negatives than positives).
- ROC can be overly optimistic because a large TN count makes FPR look small even when many FP exist.
- You care specifically about the model's behavior on the **positive (minority) class**.

Examples: fraud detection, rare disease identification.
</details>

---

**Q10.** Explain the Matthews Correlation Coefficient (MCC). What advantage does it have over F1 for imbalanced datasets?

<details>
<summary>Answer</summary>

```
MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

MCC ranges from -1 (perfect inverse) to 0 (random) to +1 (perfect).

**Advantages over F1:**
1. MCC uses **all four cells** of the confusion matrix (F1 ignores TN entirely).
2. MCC is **symmetric** — it treats both classes equally. A model that always predicts the majority class gets MCC ≈ 0, but can get a high F1 if the majority class is the positive class.
3. A high MCC requires good performance on **both** classes simultaneously.

MCC is considered the most informative single metric for binary classification on imbalanced data.
</details>

---

**Q11.** Describe the difference between Macro, Micro, and Weighted averaging for multiclass metrics. Give a use case for each.

<details>
<summary>Answer</summary>

Given 3 classes with F1 scores [0.9, 0.8, 0.5] and supports [500, 300, 200]:

**Macro**: `(0.9 + 0.8 + 0.5) / 3 = 0.733`
- Unweighted mean. Every class counts equally.
- Use when: all classes matter equally, even rare ones (e.g., rare disease classification).

**Micro**: Aggregate TP/FP/FN first across all classes, then compute.
- Dominated by frequent classes. Equivalent to accuracy for single-label problems.
- Use when: overall prediction correctness matters most.

**Weighted**: `(0.9*500 + 0.8*300 + 0.5*200) / 1000 = 0.80`
- Weighted by class support. Accounts for imbalance.
- Use when: class imbalance exists and you want a balanced overall picture.
</details>

---

**Q12.** What is Specificity? How does it relate to the ROC curve?

<details>
<summary>Answer</summary>

```
Specificity = TN / (TN + FP)    (True Negative Rate)
```

Specificity measures how well the model identifies **actual negatives**. It is the recall for the negative class.

**Relation to ROC:**

```
FPR = 1 - Specificity = FP / (FP + TN)
```

The ROC curve's x-axis is FPR = 1 - Specificity. So the ROC curve directly plots sensitivity (recall/TPR) vs (1 - specificity). Moving left on the ROC curve means increasing specificity.

A model at the top-left corner of the ROC curve has both high sensitivity (recall) and high specificity.
</details>

---

**Q13.** What does a **perfect ROC curve** look like geometrically? What point does it pass through?

<details>
<summary>Answer</summary>

A perfect ROC curve goes:
1. Vertically from (0, 0) to (0, 1) — achieves TPR=1 with FPR=0.
2. Horizontally from (0, 1) to (1, 1) — maintains TPR=1 as FPR increases.

It passes through the point **(FPR=0, TPR=1)** — the top-left corner.

This means there exists a threshold where the model correctly identifies **all positives** (zero FN) while generating **zero false alarms** (zero FP). AUC = 1.0.

In practice, ROC curves lie somewhere between the diagonal (AUC=0.5) and this ideal top-left corner.
</details>

---

**Q14.** How do you choose the optimal classification threshold on a ROC curve? Describe Youden's J statistic.

<details>
<summary>Answer</summary>

**Youden's J statistic:**

```
J = Sensitivity + Specificity - 1
  = TPR - FPR
```

The optimal threshold is the point on the ROC curve that **maximizes J** — i.e., maximizes the vertical distance from the ROC curve to the diagonal.

This corresponds to the threshold with the best combined sensitivity and specificity.

**Other threshold selection strategies:**
- Minimize cost: use a cost-sensitive objective (e.g., cost(FP) and cost(FN) from business context).
- Maximize F1: search thresholds on a validation set.
- Fixed recall: set a minimum acceptable recall, then maximize precision.
- Distance to (0,1): minimize Euclidean distance to the perfect top-left corner.
</details>

---

**Q15.** A model achieves 95% accuracy, F1=0.40, AUC=0.72 on a dataset where 95% of samples belong to class 0. What does this tell you about the model's behavior?

<details>
<summary>Answer</summary>

- **95% accuracy** is almost certainly achieved by the model predicting class 0 most of the time (matches the baseline of always predicting majority class).
- **F1=0.40** reveals the model struggles significantly with class 1 (the minority). It is missing many positives and/or generating many false positives.
- **AUC=0.72** shows the model has **some** real discriminative ability (better than random 0.5), but it is not excellent.

**Conclusion:** The model has learned some signal but is biased toward the majority class. Strategies to improve: threshold tuning, resampling (SMOTE/oversampling), class weights in the loss function, or collecting more minority-class data. Accuracy is a useless metric here — focus on F1, AUC, or MCC.
</details>

---

## Summary Cheat Sheet

| Metric | Formula | Best When |
|--------|---------|-----------|
| Accuracy | (TP+TN)/Total | Balanced classes |
| Precision | TP/(TP+FP) | FP is costly |
| Recall | TP/(TP+FN) | FN is costly |
| F1 | 2*P*R/(P+R) | Need balance of P and R |
| Specificity | TN/(TN+FP) | Care about negative class |
| Balanced Acc | (Recall+Specificity)/2 | Imbalanced, need both classes |
| AUC-ROC | Area under ROC | Threshold-independent comparison |
| MCC | See formula | Imbalanced, single robust metric |
