# Logistic Regression — Quiz

Test your understanding of logistic regression concepts. Questions mix multiple-choice, true/false, and short-answer formats.

---

**Q1. (Multiple Choice)**
What is the output range of the sigmoid function σ(z)?

- A) (-∞, +∞)
- B) [0, 1]
- C) (0, 1)
- D) [-1, 1]

---

**Q2. (Short Answer)**
Why is binary cross-entropy (log-loss) preferred over Mean Squared Error (MSE) as the loss function for logistic regression?

---

**Q3. (Multiple Choice)**
Applying L1 regularization (Lasso) to logistic regression tends to:

- A) Shrink all coefficients proportionally but never zero them out.
- B) Increase all coefficient magnitudes uniformly.
- C) Set some coefficients exactly to zero, performing implicit feature selection.
- D) Have no effect on the coefficients.

---

**Q4. (Multiple Choice)**
What shape is the decision boundary learned by logistic regression in 2D feature space?

- A) A quadratic curve
- B) A circle
- C) A straight line
- D) A step function

---

**Q5. (True/False)**
Logistic regression can perfectly classify data that is not linearly separable, as long as you use a large enough dataset.

---

**Q6. (Multiple Choice)**
You are training a logistic regression classifier for a 5-class problem. Which approach does Softmax use?

- A) Train 5 separate binary classifiers, one for each class.
- B) Map the output of a single linear model through sigmoid.
- C) Compute a probability distribution over all 5 classes simultaneously using the Softmax function.
- D) Use a decision tree to split into 5 regions.

---

**Q7. (Short Answer)**
A logistic regression model assigns a coefficient of `+2.3` to the feature "age". Interpret this coefficient in terms of **log-odds**.

---

**Q8. (Multiple Choice)**
Compared to L2 regularization, L2 regularization (Ridge) tends to:

- A) Produce sparser solutions.
- B) Zero out irrelevant features entirely.
- C) Shrink coefficients toward zero but keep all features in the model.
- D) Increase model complexity.

---

**Q9. (True/False)**
The sigmoid function is a special case of the Softmax function when there are exactly two classes.

---

**Q10. (Multiple Choice)**
In sklearn's LogisticRegression, the hyperparameter `C` controls:

- A) The number of classes.
- B) The learning rate of gradient descent.
- C) The inverse of regularization strength (larger C = less regularization).
- D) The number of iterations.

---

**Q11. (Short Answer)**
Describe two scenarios in which logistic regression is likely to fail as a classifier.

---

**Q12. (Multiple Choice)**
Which of the following is a key reason logistic regression may fail when features are highly correlated (multicollinearity)?

- A) The sigmoid function becomes undefined.
- B) The model cannot compute a loss.
- C) Coefficient estimates become unstable and have high variance.
- D) The model can only output 0 or 1, not probabilities.

---

<details>
<summary><strong>Answer Key (click to expand)</strong></summary>

**A1.** C — The sigmoid output is in the **open interval** `(0, 1)`. It approaches 0 as z → -∞ and approaches 1 as z → +∞ but never reaches the endpoints.

**A2.** MSE applied to logistic regression produces a **non-convex** loss surface because the sigmoid is non-linear. Gradient descent can get stuck in local minima. Binary cross-entropy, combined with the sigmoid function, yields a **convex** loss surface for logistic regression, guaranteeing that gradient descent converges to the global minimum.

**A3.** C — L1 (Lasso) regularization applies a penalty proportional to the absolute value of each coefficient. This creates a "corner" in the optimization landscape that encourages solutions where some coefficients are **exactly zero**, effectively removing those features from the model.

**A4.** C — Logistic regression learns weights `w` and a bias `b` such that the decision boundary is defined by `wᵀx + b = 0`. In 2D this is a straight line. In n-D it is a hyperplane of dimension n-1.

**A5.** False — Logistic regression's decision boundary is always a hyperplane. No matter how large the dataset, it cannot learn a non-linear boundary. If the classes are not linearly separable, the model will never perfectly classify the data. Non-linear classifiers (SVM with RBF kernel, neural networks, tree-based models) are needed.

**A6.** C — Softmax extends logistic regression to multiple classes by computing a score for each class and normalising them into a valid probability distribution where all class probabilities sum to 1. It handles all classes simultaneously in a single model.

**A7.** The coefficient `+2.3` means that for every one-unit increase in "age", the **log-odds** of the positive class increase by 2.3. Equivalently, the **odds** (P(y=1)/P(y=0)) are multiplied by `e^{2.3} ≈ 9.97`. Age is a strong positive predictor of the outcome in this model.

**A8.** C — L2 (Ridge) regularization applies a penalty proportional to the square of each coefficient. This shrinks all coefficients toward zero but never exactly to zero — all features remain in the model. Option A describes L1 (Lasso) behavior.

**A9.** True — With two classes, Softmax reduces to: `P(y=1|x) = exp(w₁ᵀx) / (exp(w₁ᵀx) + exp(w₀ᵀx))`. After simplification (subtracting w₀ᵀx from both), this equals the sigmoid: `σ((w₁ - w₀)ᵀx)`.

**A10.** C — In sklearn, `C` is the **inverse** of regularization strength: `C = 1/λ`. A large `C` applies weak regularization (model may overfit). A small `C` applies strong regularization (model is penalized for large weights, may underfit).

**A11.** (Any two of the following are valid):
1. **Non-linear decision boundary:** If the true boundary is curved (e.g., XOR problem, concentric circles), logistic regression cannot capture it without manual feature engineering.
2. **Highly correlated features (multicollinearity):** Unstable, high-variance coefficient estimates.
3. **Class imbalance:** The model may be biased toward the majority class without additional techniques.
4. **Complete separation:** If one feature perfectly separates the classes, maximum likelihood estimation does not converge (coefficients → ∞).
5. **High-dimensional sparse data:** Without regularization, the model overfits easily.

**A12.** C — With multicollinearity, the design matrix `X` is nearly singular, making `(XᵀX)` ill-conditioned. Small perturbations in the data produce large swings in the estimated coefficients. The model is technically still trained, but the coefficients are **unreliable** and have very high variance, making interpretation and generalization poor. L2 regularization mitigates this by adding a term to the diagonal of `XᵀX`.

</details>
