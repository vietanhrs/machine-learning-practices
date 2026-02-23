# Decision Trees — Quiz

Test your understanding of decision trees and ensemble methods. Questions mix multiple-choice, true/false, and short-answer formats.

---

**Q1. (Multiple Choice)**
For a binary classification problem, what is the maximum possible Gini impurity at a single node?

- A) 1.0
- B) 0.5
- C) 0.25
- D) It depends on the dataset size.

---

**Q2. (Short Answer)**
Define **Information Gain** in the context of decision tree splitting. Why do we maximize it?

---

**Q3. (True/False)**
A fully grown decision tree (no depth limit) will almost always achieve 100% accuracy on the training set.

---

**Q4. (Multiple Choice)**
Which of the following BEST explains why decision trees tend to overfit?

- A) They use Gini impurity instead of entropy.
- B) They grow complex, deep structures that memorize noise in training data.
- C) They only look at one feature per split.
- D) They cannot learn non-linear boundaries.

---

**Q5. (Multiple Choice)**
How does a Random Forest reduce variance compared to a single decision tree?

- A) By using a simpler splitting criterion.
- B) By training trees on the full dataset and averaging.
- C) By aggregating predictions from many diverse trees trained on bootstrap samples.
- D) By limiting all trees to depth 1.

---

**Q6. (Short Answer)**
What is the difference between **bagging** and **boosting**? Describe the key conceptual difference in how trees are built.

---

**Q7. (True/False)**
In gradient boosting, each new tree is trained to predict the raw target values, just like a regular decision tree.

---

**Q8. (Multiple Choice)**
In a Random Forest with 100 features, how many features are typically considered at each split?

- A) All 100 features
- B) 50 features (half)
- C) Approximately √100 = 10 features
- D) 1 feature (completely random)

---

**Q9. (Multiple Choice)**
Which of the following statements about feature importance (Mean Decrease in Impurity) is TRUE?

- A) It is always more reliable than permutation importance.
- B) It can be biased toward high-cardinality (many-valued) features.
- C) It measures how much a feature's removal hurts model accuracy.
- D) It is equivalent to the correlation between the feature and the target.

---

**Q10. (Short Answer)**
How does `max_depth` affect the bias-variance trade-off in a decision tree? What happens at `max_depth=1` vs. `max_depth=100`?

---

**Q11. (Multiple Choice)**
Which scenario would generally favour a tree-based model over logistic regression?

- A) The relationship between features and target is strictly linear.
- B) The dataset is very small (fewer than 50 samples).
- C) The data contains complex non-linear interactions between features.
- D) The features are already perfectly normalized.

---

**Q12. (True/False)**
Decision trees require feature scaling (e.g., standardization) before training.

---

**Q13. (Multiple Choice)**
In gradient boosting, the learning rate (η) controls:

- A) How many trees are grown.
- B) The depth of each tree.
- C) How much each successive tree contributes to the ensemble prediction.
- D) The fraction of features sampled at each split.

---

**Q14. (Short Answer)**
A colleague says "I have a Random Forest and its training accuracy is 100% but test accuracy is only 70%. The single decision tree has 95% training accuracy and 68% test accuracy. Which model is better and what does this tell us?"

---

<details>
<summary><strong>Answer Key (click to expand)</strong></summary>

**A1.** B — For binary classification, Gini impurity is `1 - p² - (1-p)²`. This is maximised at `p = 0.5`, giving `1 - 0.25 - 0.25 = 0.5`. So the maximum Gini for binary classification is `0.5`.

**A2.** Information Gain (IG) is the reduction in entropy (or impurity) achieved by splitting a dataset S on feature f:
`IG(S, f, t) = H(S) - [|S_L|/|S| · H(S_L) + |S_R|/|S| · H(S_R)]`
We maximize IG at each node because a split with high IG separates the classes as cleanly as possible — the resulting child nodes are more pure (closer to containing only one class) than the parent. This creates the most informative splits and leads to shorter trees.

**A3.** True — A fully grown tree creates leaf nodes that contain as few as one training sample. Since each leaf node returns the majority class (or the single class in that leaf), it memorizes the training data perfectly, achieving 100% training accuracy. This is a classic sign of overfitting.

**A4.** B — Decision trees can grow arbitrarily deep, creating leaf nodes that perfectly fit (memorize) individual training samples, including noise and outliers. This results in a very complex, wiggly decision boundary that does not generalize to unseen data. Controlling depth via `max_depth` or `min_samples_leaf` is the primary remedy.

**A5.** C — Random Forest uses **bagging**: each tree is trained on a random bootstrap sample (sample with replacement) of the training data, and only a random subset of features is considered at each split. This creates decorrelated trees. When their predictions are aggregated (majority vote or mean), errors that are random and uncorrelated across trees cancel out, reducing overall variance without substantially increasing bias.

**A6.** Bagging trains trees **in parallel** on independent bootstrap samples. Each tree is fully grown and has low bias but high variance. The final prediction is a simple average or majority vote. Boosting trains trees **sequentially** — each new tree focuses on correcting the errors made by the current ensemble (by fitting residuals). Boosting reduces bias primarily, while bagging reduces variance primarily.

**A7.** False — In gradient boosting, each new tree is trained to predict the **residuals** (or more precisely, the negative gradient of the loss function with respect to the current prediction). The trees fit the "error signal" from the previous iteration, not the raw target values.

**A8.** C — The standard rule of thumb for Random Forest feature subsampling is to consider `√n_features` at each split for classification. For 100 features, this is approximately 10. This decorrelates the trees: even if one feature is very predictive, it won't appear in every split of every tree.

**A9.** B — Mean Decrease in Impurity (MDI) is computed from the tree structure and can be biased toward **high-cardinality features** (features with many unique values), because they have more possible split points and thus more opportunities to reduce impurity just by chance. Permutation importance avoids this bias because it evaluates features by measuring their actual effect on out-of-sample predictions.

**A10.** `max_depth=1` (a "stump") has **high bias** (underfits — the model is too simple, one threshold on one feature) and **low variance** (the model changes little with new data). `max_depth=100` (very deep) has **low bias** (fits training data nearly perfectly) and **high variance** (tiny changes in training data cause very different trees). The optimal depth balances bias and variance, typically found via cross-validation.

**A11.** C — Tree-based models (decision trees, Random Forests, gradient boosting) excel at capturing **non-linear interactions** because they partition the feature space with axis-aligned splits. Logistic regression assumes a linear relationship between features and log-odds. For complex, non-linear datasets, tree-based models are typically far superior.

**A12.** False — Decision trees make splits based on thresholds (e.g., "feature > 5.3"), not distances. Scaling features does not change the rank ordering of values within each feature, so splits are identical regardless of scale. Unlike K-Means or KNN, decision trees are **scale-invariant**.

**A13.** C — The learning rate η (also called "shrinkage") scales the contribution of each new tree to the ensemble: `ŷ ← ŷ + η · tree_prediction`. A small η means each tree contributes only a little, requiring more trees but generally producing a better-regularized model. A large η means fewer trees are needed but the model may overfit.

**A14.** The Random Forest is better (70% > 68% test accuracy). The 100% training accuracy of the Random Forest indicates it has overfit — it has memorized the training data. However, its test accuracy of 70% shows it generalizes better than the decision tree (68%). The gap between training and test accuracy (100% vs 70% = 30% gap) is large, suggesting the forest could be further regularized (by reducing `max_depth`, increasing `min_samples_leaf`, or reducing `n_estimators`). The decision tree's smaller gap (95% - 68% = 27%) shows it also overfits. Both models are overfitting; the Random Forest just generalizes marginally better due to ensemble averaging.

</details>
