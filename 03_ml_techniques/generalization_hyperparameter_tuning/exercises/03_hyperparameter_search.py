"""
Exercise 03: Hyperparameter Search
====================================
Implement Grid Search and Random Search from scratch for hyperparameter tuning.
Compare their efficiency in finding good hyperparameter configurations for a
Random Forest classifier. Also implement Nested CV for unbiased model selection.

Learning objectives:
- Implement grid search and random search from scratch
- Understand why random search is often more efficient than grid search
- Implement nested cross-validation to avoid validation set overfitting

Dependencies: numpy, scikit-learn, matplotlib, itertools
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# ---------------------------------------------------------------------------
# Grid Search from Scratch
# ---------------------------------------------------------------------------

def grid_search(model_fn, param_grid, X, y, cv=5):
    """
    Exhaustively evaluate all combinations of hyperparameters.

    Args:
        model_fn (callable): A function that takes **kwargs and returns an unfitted model.
                             Example: lambda **p: RandomForestClassifier(**p, random_state=42)
        param_grid (dict): {param_name: [list of values to try]}
                           Example: {"n_estimators": [100, 200], "max_depth": [3, 5]}
        X (np.ndarray): Feature matrix
        y (np.ndarray): Labels
        cv (int): Number of cross-validation folds

    Returns:
        list of dicts: Each dict is {"params": {...}, "mean_score": float, "std_score": float}
                       Sorted by mean_score (descending — best first).

    Algorithm:
        1. Generate all combinations using itertools.product
        2. For each combination:
           - Create the model with those hyperparameters
           - Run k-fold CV and compute mean/std accuracy
           - Record results
        3. Sort by mean_score descending
        4. Print progress every N combinations

    Hints:
        - keys = list(param_grid.keys())
        - values = list(param_grid.values())
        - for combo in itertools.product(*values):
              params = dict(zip(keys, combo))
              model = model_fn(**params)
              scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        - Total combinations = product of all list lengths
        - Print: "Evaluating combination {i}/{total}: {params} → mean={score:.4f}"
    """
    # TODO: Generate all hyperparameter combinations using itertools.product
    # TODO: Initialize results list
    # TODO: For each combination:
    #   - Build params dict
    #   - Create model with model_fn(**params)
    #   - Cross-validate: scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    #   - Append {"params": params, "mean_score": scores.mean(), "std_score": scores.std()} to results
    # TODO: Sort results by mean_score descending
    # TODO: Return sorted results list
    pass


# ---------------------------------------------------------------------------
# Random Search from Scratch
# ---------------------------------------------------------------------------

def random_search(model_fn, param_distributions, X, y, n_iter=50, cv=5, random_state=42):
    """
    Randomly sample hyperparameter combinations and evaluate them.

    Args:
        model_fn (callable): Function returning a model given **kwargs
        param_distributions (dict): {param_name: list_of_values or callable}
                                   If callable: sampler() → value (e.g., lambda: np.random.randint(10,100))
                                   If list: uniform random choice from the list
        X (np.ndarray): Feature matrix
        y (np.ndarray): Labels
        n_iter (int): Number of random combinations to evaluate
        cv (int): Number of CV folds
        random_state (int): Random seed

    Returns:
        list of dicts: Same format as grid_search, sorted by mean_score descending.

    Algorithm:
        1. For n_iter iterations:
           - For each param, sample a value:
             - If list: np.random.choice(list_of_values)
             - If callable: call it — callable()
           - Build params dict
           - Evaluate with cross_val_score
        2. Sort and return results

    Hints:
        - np.random.seed(random_state) at the start
        - for _ in range(n_iter):
              params = {}
              for key, dist in param_distributions.items():
                  if callable(dist): params[key] = dist()
                  else: params[key] = np.random.choice(dist)
        - Much faster than grid search when param space is large
        - Print progress every 10 iterations
    """
    np.random.seed(random_state)

    # TODO: Initialize results list
    # TODO: For n_iter iterations:
    #   - Sample parameters from distributions
    #   - Create and evaluate model
    #   - Append results
    # TODO: Sort results by mean_score descending
    # TODO: Return sorted results
    pass


# ---------------------------------------------------------------------------
# Visualization (Complete — Do Not Modify)
# ---------------------------------------------------------------------------

def plot_search_results(results, param_name, score_name='mean_score', top_n=20):
    """
    Plot relationship between a specific hyperparameter and CV score.

    Args:
        results (list of dicts): Output of grid_search or random_search
        param_name (str): Name of hyperparameter to plot on x-axis
        score_name (str): Key in result dict for the score
        top_n (int): Number of top results to highlight
    """
    param_values = [r["params"][param_name] for r in results if param_name in r["params"]]
    scores = [r[score_name] for r in results if param_name in r["params"]]

    if not param_values:
        print(f"Parameter '{param_name}' not found in results.")
        return

    plt.figure(figsize=(10, 5))
    plt.scatter(param_values, scores, alpha=0.6, color='steelblue', label='All trials')

    # Highlight top N results
    sorted_indices = np.argsort(scores)[::-1][:top_n]
    top_values = [param_values[i] for i in sorted_indices]
    top_scores = [scores[i] for i in sorted_indices]
    plt.scatter(top_values, top_scores, color='red', s=80, zorder=5, label=f'Top {top_n}')

    plt.xlabel(param_name)
    plt.ylabel("CV Accuracy")
    plt.title(f"Hyperparameter vs Performance: {param_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Nested Cross-Validation
# ---------------------------------------------------------------------------

def nested_cv(model_fn, param_grid, X, y, outer_k=5, inner_k=3):
    """
    Nested cross-validation for unbiased model performance estimation.

    Unlike standard CV + hyperparameter tuning (which overfits to the val set),
    nested CV provides an unbiased estimate of the best model's performance.

    Structure:
        Outer CV (outer_k folds): estimates generalization performance
            └── Inner CV (inner_k folds): selects hyperparameters

    Args:
        model_fn (callable): Function returning a model given **kwargs
        param_grid (dict): Hyperparameter grid for inner search
        X (np.ndarray): Feature matrix
        y (np.ndarray): Labels
        outer_k (int): Number of outer folds (for unbiased performance estimate)
        inner_k (int): Number of inner folds (for hyperparameter selection)

    Returns:
        dict: {
            "outer_scores": list of outer fold accuracy scores,
            "best_params_per_fold": list of best params selected in each outer fold,
            "mean": float mean of outer scores,
            "std": float std of outer scores
        }

    Algorithm:
        For each outer fold (train_outer, test_outer):
            1. Use grid_search on (X[train_outer], y[train_outer]) with inner_k CV
               to find best_params
            2. Retrain model with best_params on ALL of X[train_outer]
            3. Evaluate on X[test_outer] → record outer score
        Final estimate = mean of outer scores

    Hints:
        - Use StratifiedKFold for outer splits (preserves class proportions)
        - For inner search, call grid_search(model_fn, param_grid,
                                             X[train_outer], y[train_outer], cv=inner_k)
        - best_params = inner_results[0]["params"]  (first = highest mean CV score)
        - best_model = model_fn(**best_params); best_model.fit(X[train_outer], y[train_outer])
        - outer_score = accuracy_score(y[test_outer], best_model.predict(X[test_outer]))
        - Print best hyperparameters and score for each outer fold
    """
    outer_cv = StratifiedKFold(n_splits=outer_k, shuffle=True, random_state=42)
    outer_scores = []
    best_params_per_fold = []

    # TODO: For each outer fold (train_outer, test_outer):
    #   - Run grid_search on outer training data
    #   - Get best hyperparameters from inner search
    #   - Retrain model on all outer training data
    #   - Evaluate on outer test fold
    #   - Record score and best params
    # TODO: Print summary: mean ± std across outer folds
    # TODO: Return results dict
    pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    Compare grid search vs random search on a Random Forest.

    Hyperparameters to search:
        - n_estimators: number of trees [50, 100, 200, 300, 500]
        - max_depth: max tree depth [3, 5, 10, 15, None]
        - min_samples_split: min samples to split a node [2, 5, 10, 20]

    Steps:
    1. Generate classification data
    2. Run grid search — all combinations (5×5×4=100 configurations)
    3. Run random search — 30 random configurations
    4. Compare: best score found, time taken, search landscape
    5. Run nested CV on a smaller subset for demonstration
    """
    print("=" * 60)
    print("Hyperparameter Search: Grid vs Random vs Nested CV")
    print("=" * 60)

    np.random.seed(42)

    # Generate classification data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10,
        n_redundant=5, n_classes=3, n_clusters_per_class=1,
        random_state=42
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print(f"Dataset: {X.shape}, Classes: {np.unique(y)}")

    # Define hyperparameter spaces
    param_grid = {
        "n_estimators": [50, 100, 200, 300, 500],
        "max_depth": [3, 5, 10, 15, None],
        "min_samples_split": [2, 5, 10, 20]
    }

    param_distributions = {
        "n_estimators": [50, 75, 100, 150, 200, 300, 400, 500],
        "max_depth": [3, 5, 7, 10, 12, 15, 20, None],
        "min_samples_split": [2, 3, 5, 7, 10, 15, 20]
    }

    model_fn = lambda **params: RandomForestClassifier(random_state=42, **params)

    # TODO: Run grid search
    print("\n--- Grid Search ---")
    total_combinations = 1
    for v in param_grid.values():
        total_combinations *= len(v)
    print(f"Total combinations: {total_combinations}")
    # grid_results = grid_search(model_fn, param_grid, X, y, cv=3)
    # print(f"Best params: {grid_results[0]['params']}")
    # print(f"Best CV score: {grid_results[0]['mean_score']:.4f} ± {grid_results[0]['std_score']:.4f}")

    # TODO: Run random search
    print("\n--- Random Search (n_iter=30) ---")
    # random_results = random_search(model_fn, param_distributions, X, y, n_iter=30, cv=3)
    # print(f"Best params: {random_results[0]['params']}")
    # print(f"Best CV score: {random_results[0]['mean_score']:.4f} ± {random_results[0]['std_score']:.4f}")

    # TODO: Plot search results for n_estimators
    # plot_search_results(grid_results, "n_estimators")
    # plot_search_results(random_results, "n_estimators")

    # TODO: Compare top-1 vs top-5 vs top-10 scores for both methods

    # TODO: Nested CV (use smaller subset for speed: X[:300], y[:300])
    print("\n--- Nested CV (outer=5, inner=3) ---")
    print("Running nested CV on subset of 300 samples...")
    # Small grid for nested CV demo
    small_param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None]
    }
    # nested_results = nested_cv(model_fn, small_param_grid, X[:300], y[:300],
    #                             outer_k=5, inner_k=3)
    # print(f"Nested CV estimate: {nested_results['mean']:.4f} ± {nested_results['std']:.4f}")

    print("\n--- Expected Findings ---")
    print("1. Grid search: exhaustive, finds global best within grid")
    print("2. Random search: with 30/100 trials, often finds equally good params")
    print("3. Nested CV: unbiased estimate — slightly lower than inner CV score")
    print("   (Inner CV optimistically selects best from noise; outer CV corrects for this)")


if __name__ == "__main__":
    main()
