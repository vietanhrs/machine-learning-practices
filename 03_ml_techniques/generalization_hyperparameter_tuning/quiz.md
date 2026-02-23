# Quiz: Generalization and Hyperparameter Tuning

Test your understanding of model generalization, cross-validation, and hyperparameter search.

---

**Q1.** Explain the bias-variance tradeoff using a concrete example. How does model complexity affect each component?

<details>
<summary>Answer</summary>

**Concrete Example: Polynomial Regression on Noisy Sine Data**

We have 20 training points sampled from `y = sin(x) + noise`. We fit polynomials of degree 1, 5, and 15.

**Degree 1 (linear model) — High Bias, Low Variance:**
- The linear model cannot capture the curved sine relationship.
- No matter which 20 points we sample, the linear model always makes a straight line fit.
- Low variance: the model looks almost the same across different training sets.
- High bias: the model systematically underpredicts the peaks and troughs of the sine curve.
- Both training and test error are high.

**Degree 5 — Low Bias, Low Variance:**
- The degree-5 polynomial can roughly approximate the sine curve.
- Moderate flexibility without extreme sensitivity to individual training points.
- Good generalization — the "sweet spot" for this dataset.

**Degree 15 — Low Bias, High Variance:**
- The polynomial passes through all 20 training points perfectly (zero training error).
- Dramatic oscillations between points — "Runge's phenomenon."
- High variance: different training sets of 20 points produce wildly different 15th-degree fits.
- High test error from memorized noise.

**Effect of Complexity:**

```
As degree increases:
    Bias² ↓ (better fit to true sine function on training data)
    Variance ↑ (more sensitive to which specific points were sampled)
    Noise: constant (irreducible)
    
Total Error = Bias² + Variance + Noise
    → U-shaped: high at both extremes, minimum at the right complexity
```

</details>

---

**Q2.** Why is k-fold cross-validation better than a single train/validation/test split for estimating model performance?

<details>
<summary>Answer</summary>

**Problem with Single Split:**

A single train/val split suffers from **high variance in the performance estimate**:

1. **Lucky/unlucky splits**: The particular 20% of data held out for validation may be unusually easy or hard. Performance estimates can vary by 5-15% depending on which specific examples end up in the validation set.

2. **Wasted data**: With limited data, dedicating 20% to a fixed validation set means training on only 80%. With 1000 examples, 200 are "locked away" and never used for training.

3. **Single estimate**: You get one performance number with no confidence interval — no way to assess the uncertainty of the estimate.

**Advantages of k-Fold CV:**

1. **Lower variance estimate**: Performance is averaged over k folds — the lucky/unlucky split problem is averaged out. The standard deviation across folds gives a confidence interval.

2. **All data used for training**: Each example is in the training set k-1 times and the validation set once. No data is permanently locked away.

3. **More reliable for model selection**: Comparing two models' k-fold CV scores is more reliable than comparing single validation scores — less likely to select a model that got lucky on the split.

4. **Especially important with limited data**: With 100 examples, k-fold CV allows meaningful performance estimation where a single split would be unreliable.

**When single split is still appropriate:**
- Very large datasets (millions of examples) where each fold takes hours to train.
- When you explicitly need a held-out test set that was never seen during any model development.

</details>

---

**Q3.** Why does random search often outperform grid search with the same number of hyperparameter evaluations?

<details>
<summary>Answer</summary>

**Bergstra & Bengio (2012) insight:**

In real-world hyperparameter optimization, some hyperparameters matter much more than others. For example, in a neural network:
- Learning rate: critical (10× change dramatically affects performance)
- Random seed: irrelevant (negligible effect on final performance)

**Grid Search's Inefficiency:**

```
Grid example: 5 LR values × 5 momentum values = 25 evaluations

If only LR matters (momentum is irrelevant):
    Each of the 5 LR values is evaluated only 5 times (with different momentum).
    You effectively have only 5 distinct LR values tested.
```

Grid search "wastes" evaluations by repeating the same unimportant hyperparameter values in each row/column.

**Random Search's Efficiency:**

```
Random search: 25 random (LR, momentum) pairs

Each of the 25 evaluations uses a different LR value.
You effectively test 25 distinct LR values.
```

When only one hyperparameter matters, random search explores it 5× more thoroughly than grid search with the same compute budget.

**Formally:** If there are `n` hyperparameters but only `k` are important:
- Grid search: tests m values for each important hyperparameter (total mⁿ evaluations, m^k for important ones).
- Random search: with m^k evaluations, tests ~m^k values along each important dimension.

**Practical implication:**
- Random search is usually preferred for > 2 hyperparameters.
- Grid search is reasonable for 1-2 hyperparameters where exhaustive search is feasible.
- Both are outperformed by Bayesian optimization when evaluations are expensive.

</details>

---

**Q4.** What is nested cross-validation, and when is it necessary? What happens if you don't use it?

<details>
<summary>Answer</summary>

**Nested Cross-Validation:**

A two-level cross-validation scheme for simultaneous model selection (hyperparameter tuning) and performance estimation:

```
Outer CV (performance estimation, k=5):
    For each of 5 folds as test fold:
        Remaining 4 folds = available data for inner CV
        
        Inner CV (hyperparameter selection, k=3):
            Grid/random search over hyperparameters
            Select best hyperparameters based on inner CV score
        
        Retrain model with best hyperparameters on all 4 outer-train folds
        Evaluate on outer test fold → record performance
        
Final estimate = mean of 5 outer test scores
```

**Why it's necessary:**

**Without nested CV (the pitfall):**

1. You tune hyperparameters using k-fold CV on the full dataset.
2. You report the best cross-validated score as your model's performance.

**Problem:** The best hyperparameter set was selected because it happened to score highest on those particular folds. The reported score is **optimistically biased** — you've "fit" the hyperparameters to the validation folds.

This is exactly the same problem as evaluating on training data — you've used the validation folds for both selection and evaluation.

**With nested CV:** The outer test fold is **never used for hyperparameter selection**. It provides a truly held-out estimate of generalization performance for the best model found by inner CV.

**When is it necessary:**
- Comparing different model families where hyperparameter selection is part of the process.
- Publishing research results that claim a generalization performance.
- High-stakes decisions (medical, financial) where optimistic bias is unacceptable.

**When it's not needed:**
- When you have a separate test set reserved solely for final evaluation.
- Exploratory analysis where approximate performance estimates are acceptable.

</details>

---

**Q5.** How do you interpret a learning curve to diagnose overfitting vs underfitting? What would the curves look like in each case?

<details>
<summary>Answer</summary>

A learning curve plots training set size on the x-axis and model performance (accuracy or loss) on the y-axis, with separate curves for training and validation scores.

**Pattern 1: Underfitting (High Bias)**
```
Accuracy
1.0 |
    |
0.8 |  ____________________  ← Training accuracy (plateaus low)
    | /
0.6 |/____________________  ← Validation accuracy (plateaus low, near training)
    |
    +------------------------→ Training set size
```
- Both training and validation accuracy plateau at the same **low** value.
- Small gap between the two curves — the model is not sensitive to training data.
- Adding more data does **not help** — the model is the bottleneck.
- Diagnosis: Model too simple for the task. Add features, use more complex model.

**Pattern 2: Overfitting (High Variance)**
```
Accuracy
1.0 |______________________ ← Training accuracy (stays high)
    |
    |
0.8 |
    |              /¯¯¯¯¯¯¯ ← Validation accuracy (still rising)
0.6 |_____________/
    |
    +------------------------→ Training set size
```
- Large gap between training (high) and validation (lower) accuracy.
- Validation accuracy is still increasing as training set grows.
- Adding more data **would help** — the model can generalize if given enough examples.
- Diagnosis: Need more data OR regularization.

**Pattern 3: Good Generalization**
```
Accuracy
1.0 |
    |__________ ← Training accuracy
0.9 |          \
    |           \___________← Validation accuracy
0.8 |
    |
    +------------------------→ Training set size
```
- Small gap between training and validation.
- Both curves converge at acceptable performance.
- The model generalizes well.

**Practical use of learning curves:**
- If the gap is large and val is improving: collect more data.
- If both curves are low: improve the model or features.
- If the curves have converged with acceptable performance: you're done.

</details>

---

**Q6.** What is Bayesian optimization and why is it more sample-efficient than random or grid search?

<details>
<summary>Answer</summary>

**Bayesian Optimization:**

A sequential model-based optimization strategy that builds a probabilistic surrogate model of the objective function (hyperparameter → performance) and uses it to intelligently select the next evaluation point.

**Algorithm:**
1. Evaluate a small number of random configurations → initial observations.
2. Fit a **surrogate model** (typically Gaussian Process or Tree-Structured Parzen Estimator) to map hyperparameters → expected performance.
3. Use an **acquisition function** to select the next point to evaluate:
   - **Expected Improvement (EI)**: Sample where the expected improvement over the current best is highest.
   - **Upper Confidence Bound (UCB)**: Balance exploration (high uncertainty) and exploitation (high mean).
4. Evaluate the chosen configuration, update the surrogate model.
5. Repeat until the evaluation budget is exhausted.

**Why Sample-Efficient:**

- **Uses all past observations**: Each evaluation informs the surrogate model, making future queries smarter. Random search ignores past results.
- **Balances exploration and exploitation**: The acquisition function tries promising regions while also probing uncertain regions where the surrogate model might be wrong.
- **Focuses on promising regions**: The surrogate model directs search toward the most promising parts of the hyperparameter space, rather than spending evaluations in already-explored poor regions.

**Quantitative advantage:**
- For a neural network training job taking 12 hours, you might have 50 evaluation budget. Bayesian optimization with 50 evaluations often matches grid/random search with 200+ evaluations.

**Limitations:**
- Overhead of fitting the surrogate model (minor for cheap surrogates like TPE, significant for full GPs).
- Less parallelizable — next point depends on all previous observations.
- Less effective for very high-dimensional hyperparameter spaces (>20 hyperparameters).

</details>

---

**Q7.** A model achieves 96% training accuracy and 94% validation accuracy. Is this overfitting? What if the baseline accuracy (always predict majority class) is 93%?

<details>
<summary>Answer</summary>

**Initial Analysis:**

The gap is 2% (96% − 94% = 2%). A 2% gap seems small. Is this overfitting?

**Context matters enormously:**

The 2% train-val gap does suggest some overfitting, but it may or may not be practically significant. The more important question is: what is the model actually learning?

**With 93% baseline accuracy:**

If always predicting the majority class gives 93% accuracy, then:
- Training accuracy: 96% → only 3% above baseline.
- Validation accuracy: 94% → only 1% above baseline.

The model is barely doing better than a trivial classifier that learns nothing. The small 2% gap now represents a meaningful fraction of the actual learned signal.

**Interpretation with 93% baseline:**

- The model has learned very little on top of the baseline distribution.
- It is severely underfitting the minority class — likely failing to detect the class of interest.
- The 2% gap: if the model achieves 3% above baseline on training and only 1% above baseline on validation, that's a 66% drop in learned signal from train to val — significant overfitting relative to what was learned.
- Need to look at **per-class metrics** (precision, recall, F1) rather than accuracy.

**Lesson:** Always contextualize performance metrics against baselines and look at class-specific metrics for imbalanced data. A 2% train-val gap means little without understanding what the model has actually learned.

</details>

---

**Q8.** What is "test set contamination" and how does it compromise model evaluation? Give examples of how it happens accidentally.

<details>
<summary>Answer</summary>

**Test Set Contamination:**

Test set contamination occurs when information from the test set (or future data) "leaks" into the model training or selection process, causing optimistically biased performance estimates.

**How it happens accidentally:**

1. **Preprocessing on full dataset before split:**
```python
# WRONG - contamination!
scaler = StandardScaler().fit(X_all)           # sees test data!
X_scaled = scaler.transform(X_all)
X_train, X_test = train_test_split(X_scaled)

# CORRECT
X_train, X_test = train_test_split(X_all)
scaler = StandardScaler().fit(X_train)         # fit on train only
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)              # transform test with train stats
```

2. **Feature selection using full dataset:**
```python
# WRONG - contamination!
selected_features = select_k_best(X_all, y_all, k=50)  # uses test labels!
X_train_sel = X_train[:, selected_features]
```

3. **Hyperparameter tuning on test set:**
```python
# WRONG - contamination!
for lr in [0.001, 0.01, 0.1]:
    model.train(X_train, y_train)
    test_acc = evaluate(model, X_test, y_test)  # evaluating on test!
    # Selecting the lr based on test performance
```

4. **Multiple evaluation and cherry-picking:**
Evaluating the same test set after each modification and reporting the best result. The test set acts as an implicit validation set.

5. **Data leakage from future information:**
In time-series, using future data to predict the past: including features computed from t+1 when predicting at time t.

**Consequences:**
- The reported test accuracy is higher than actual deployment performance.
- The model appears to generalize well but fails in production.
- Leads to wasted resources on a model that doesn't work as advertised.

**Rule:** The test set must be used **exactly once** for final evaluation, after all training and model selection decisions are complete.

</details>

---

**Q9.** Explain what the "no free lunch theorem" means for machine learning. Does it mean all algorithms are equally good?

<details>
<summary>Answer</summary>

**The No Free Lunch Theorem (Wolpert & Macready, 1997):**

The theorem states that, when averaged across all possible data-generating distributions, no machine learning algorithm performs better than any other. Every algorithm that performs better on some problems must perform worse on others — the "gains" on one set of problems are exactly offset by "losses" on another set.

**Formal statement:** When averaged across all possible target functions uniformly, all algorithms have the same expected performance on out-of-sample predictions.

**What this means in practice:**

1. **No universal best algorithm**: There is no single algorithm that is best for every problem. Neural networks dominate on image classification, but gradient-boosted trees often outperform on tabular data.

2. **Inductive biases matter**: Every algorithm encodes assumptions about the problem (inductive biases). The algorithm that works best is the one whose inductive bias best matches the true data-generating process.

3. **Domain knowledge is essential**: Since there's no free lunch, you must use problem-specific knowledge to choose or design appropriate algorithms. Random search over all algorithms is not optimal.

**Does it mean all algorithms are equally good?**

No — the theorem averages over ALL possible problems, including many that are practically irrelevant. In reality:

- Real-world data comes from a restricted subset of all possible distributions.
- The distributions that arise in practice (images, text, tabular data) are not uniformly distributed.
- On the specific distributions encountered in practice, some algorithms (deep learning, gradient boosting) consistently outperform others.

**Practical takeaway:** For any specific domain (vision, NLP, tabular data), empirical evidence from the research community tells you which algorithms work best. The NFL theorem is a theoretical statement about worst-case behavior over all problems, not a practical guide.

</details>

---

**Q10.** In k-fold cross-validation with k=5, training takes 10 minutes per fold. A colleague suggests using k=10 for a better estimate. What is the time cost, and is it always worth it?

<details>
<summary>Answer</summary>

**Time Cost:**

- k=5: 5 folds × 10 minutes = 50 minutes total.
- k=10: 10 folds × 10 minutes = 100 minutes total (2× more expensive).

**Is k=10 always worth it?**

The benefit of higher k is **lower variance in the performance estimate** and **more training data per fold** (90% vs 80% of data).

**When k=10 is worth it:**
- Small datasets where leaving out 20% (k=5) is costly.
- When the performance estimate variance with k=5 is unacceptably high (large std across folds).
- When each fold is cheap (< 1 minute) — the 2× overhead is negligible.

**When k=5 is sufficient:**
- Large datasets where each fold uses enough data for a reliable estimate.
- When computational resources are limited.
- Most practical applications — k=5 and k=10 rarely differ by more than 0.5% in practice.

**The trade-off:**
Increasing k:
- Reduces bias in performance estimate (more training data per fold → closer to full-dataset performance).
- Reduces variance slightly (more folds averaged).
- Increases computational cost linearly.

**Standard practice:**
- k=5 is the most common choice — a good balance of reliability and speed.
- k=10 is used when you need a more precise estimate and can afford 2× the compute.
- LOO-CV (k=N) is only used for very small datasets (N < 50).

**Practical advice:** With 10 minutes per fold, k=5 (50 minutes) is reasonable. For k=10 (100 minutes), check if the std across k=5 folds is high — if all folds give similar scores, increasing k is unlikely to change your conclusion.

</details>

---

**Q11.** What is the difference between a hyperparameter and a model parameter? Give examples of each.

<details>
<summary>Answer</summary>

**Model Parameters:**
- Learned from training data via optimization (gradient descent).
- Their values are determined by the training algorithm to minimize the loss.
- Not set by the user — the algorithm finds the optimal values.
- Examples:
  - Neural network weights and biases
  - Linear regression coefficients `β₀, β₁, ..., βₙ`
  - SVM support vectors and their coefficients `αᵢ`
  - Decision tree split thresholds (learned from data)

**Hyperparameters:**
- Set by the user BEFORE training begins.
- Not optimized by gradient descent — must be tuned separately.
- Control the training process or model architecture, not the model's predictions directly.
- Examples:
  - Learning rate, batch size, number of epochs
  - Neural network: number of layers, number of neurons per layer, dropout rate, activation function
  - Regularization strength: λ (L1/L2), dropout probability
  - SVM: kernel type (RBF, linear), regularization parameter C, kernel bandwidth γ
  - Random Forest: n_estimators, max_depth, min_samples_split
  - k in k-nearest neighbors
  - k in k-fold cross-validation (though this is a CV hyperparameter, not model hyperparameter)

**Key distinction:**
- Parameters are **learned** from data.
- Hyperparameters are **chosen** by the practitioner.

**Why hyperparameters can't be learned by gradient descent:**
Most hyperparameters (depth, architecture, learning rate) either:
1. Are not differentiable with respect to the loss (discrete choices like number of layers).
2. Require training the full model to evaluate (creating an outer optimization loop).

Gradient-based hyperparameter optimization methods (DARTS, Hypergrad) exist for some cases but are specialized and expensive.

</details>

---

**Q12.** What is overfitting "to the validation set" in hyperparameter tuning? How does it occur?

<details>
<summary>Answer</summary>

**Overfitting to the Validation Set:**

Just as a model can overfit to training data by memorizing it, a hyperparameter search can "overfit" to the validation set — finding hyperparameters that happen to work well on that specific validation set but not on truly unseen test data.

**How it occurs:**

1. **Many hyperparameter evaluations**: With 100 random search trials, some combinations will score high on the validation set purely by chance. The more evaluations, the more opportunities to be lucky.

2. **Implicit search over validation sets**: When you repeatedly tweak preprocessing, architecture, and training based on validation performance, you're accumulating "degrees of freedom" on the validation set.

3. **Small validation sets**: With only 100 validation examples, a 2-3% difference in validation accuracy (2-3 examples) could be noise. Selecting hyperparameters based on this noisy signal leads to overfit.

**Evidence of validation set overfitting:**
- The model performs well on the validation set used for tuning.
- Performance drops significantly on the actual test set.
- The hyperparameter tuning "curve" keeps improving with more trials, but test performance plateaus.

**Solutions:**

1. **Nested cross-validation**: Use inner CV for hyperparameter selection, outer CV for performance estimation. The outer test folds are never used for selection.

2. **Separate test set**: Reserve a truly held-out test set that is never examined until final evaluation.

3. **Fewer hyperparameter evaluations**: Limit the search — fewer tries = less chance to overfit to the validation set.

4. **Larger validation sets**: More data = less noise = less overfitting to the specific validation set.

5. **Statistical tests**: Use paired statistical tests to determine if performance differences are significant vs. due to chance.

</details>

---

**Q13.** You have 1000 labeled examples. Compare these strategies for model evaluation: (a) 70/30 train/test split, (b) 5-fold CV, (c) 10-fold CV, (d) Leave-One-Out CV. When is each appropriate?

<details>
<summary>Answer</summary>

| Strategy | Training Size | Evaluations | Time Cost | Variance | Recommended When |
|----------|--------------|-------------|-----------|----------|-----------------|
| **70/30 split** | 700 | 1 | 1× | High | Large datasets, fast baseline |
| **5-fold CV** | 800 avg | 5 | 5× | Moderate | General purpose, standard practice |
| **10-fold CV** | 900 avg | 10 | 10× | Lower | Need precise estimate, fast model |
| **LOO-CV** | 999 avg | 1000 | 1000× | High* | Tiny datasets (N<50), specific cases |

*LOO-CV has low bias but surprisingly high variance for regression (each fold has only 1 test example).

**When each is appropriate:**

**(a) 70/30 single split:**
- Fast experimentation / rapid prototyping.
- 1000 examples: 700 train / 300 test — reasonable if classes are balanced.
- **Risk**: High variance in performance estimate. Different random seeds give different results.
- **Not recommended for final model evaluation**.

**(b) 5-fold CV (recommended default):**
- Best balance of reliability (5 estimates averaged) and speed (5 model fits).
- Training size: 800/fold — enough for meaningful learning.
- Standard in most ML papers and Kaggle competitions.
- **Use for: most hyperparameter tuning and model comparison**.

**(c) 10-fold CV:**
- Slightly better estimate than 5-fold (more folds averaged, more training data per fold: 900/fold).
- 2× more expensive than 5-fold.
- **Use when**: variance across 5-fold folds is high, or when each fold is cheap.

**(d) Leave-One-Out CV:**
- Trains 1000 models — 1000× more expensive than a single fit.
- Only 1 test example per fold — estimate variance is actually quite high for regression.
- **Use only when**: N < 30-50, and you cannot afford to hold out even one fold.
- Not recommended for classification with 1000 examples.

**For 1000 examples: 5-fold CV is the right choice.**

</details>

---

**Q14.** Your grid search with 3 hyperparameters (each with 5 values) takes 24 hours. How would you reduce this while maintaining good hyperparameter coverage?

<details>
<summary>Answer</summary>

**Current situation:**
3 hyperparameters × 5 values each = 5³ = 125 combinations.
With 5-fold CV: 125 × 5 = 625 model fits.
625 fits in 24 hours → ~2.3 minutes per fit.

**Strategies to reduce time:**

**1. Switch to Random Search (most impactful):**
- Replace grid search with random search using n_iter=50 (instead of 125).
- Time: 50 × 5 folds × 2.3 min = ~9.6 hours (60% reduction).
- Bergstra & Bengio (2012): random search finds equally good or better hyperparameters than grid search in 60% of the evaluations for most practical problems.

**2. Reduce Number of CV Folds:**
- Use 3-fold instead of 5-fold CV.
- Time: 125 × 3 × 2.3 min = ~9.6 hours for grid, ~5.8 hours for 50-iter random.
- Trade-off: slightly noisier estimates, but still useful for selection.

**3. Use Bayesian Optimization:**
- n_iter=30-50 intelligently selected configurations.
- Often finds better solutions than random search with same or fewer evaluations.
- Libraries: Optuna, Hyperopt.

**4. Coarsen the Grid + Zoom:**
- First pass: 3 values per hyperparameter (3³=27 combinations × 5 folds = ~2.1 hours).
- Identify promising region around best configuration.
- Second pass: fine-grained search in the promising region.

**5. Reduce Training Epochs for Search:**
- During hyperparameter search, train for fewer epochs (e.g., 50% of final epochs).
- Use the promising candidates from the search and retrain fully.

**6. Parallelize:**
- If you have multiple GPUs or machines, run folds/configurations in parallel.
- 625 fits across 8 machines → ~3 hours total.

**Recommended approach:** Bayesian optimization with Optuna, n_trials=50, 3-fold CV → ~5.75 hours total, likely better results than full grid search.

</details>

---

*End of Quiz — 14 Questions*
