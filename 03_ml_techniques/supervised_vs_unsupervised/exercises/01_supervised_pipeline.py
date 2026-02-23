"""
Exercise 01: Full Supervised Learning Pipeline
================================================
Implement a complete supervised learning pipeline from data loading to
model evaluation. Compare multiple classifiers on the same dataset.

Learning objectives:
- Understand train/validation/test split strategy
- Apply preprocessing without data leakage
- Train and evaluate multiple classifiers
- Visualize learning curves to diagnose over/underfitting

Dependencies: scikit-learn, matplotlib, numpy
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import learning_curve


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_and_split(dataset_name="iris", test_size=0.2, val_size=0.1):
    """
    Load a sklearn dataset and split it into train, validation, and test sets.

    Args:
        dataset_name (str): One of "iris", "digits", "breast_cancer", "wine"
        test_size (float): Fraction of data for the test set
        val_size (float): Fraction of data for the validation set (from remaining after test split)

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)

    Hints:
        - Use getattr(datasets, f"load_{dataset_name}")() to load the dataset
        - Use sklearn.model_selection.train_test_split twice:
            1. First split off the test set using test_size
            2. Then split the remaining data into train/val using val_size
        - Set random_state=42 for reproducibility
        - Stratify splits by y to preserve class proportions
        - Print the shapes of all splits so the user can verify
    """
    # TODO: Load the dataset using sklearn.datasets
    # TODO: First split: separate test set
    # TODO: Second split: separate validation set from training data
    # TODO: Print split sizes
    # TODO: Return (X_train, X_val, X_test, y_train, y_val, y_test)
    pass


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(X_train, X_val, X_test):
    """
    Standardize features to zero mean and unit variance.

    IMPORTANT — No data leakage:
        The scaler must be FIT only on X_train, then used to TRANSFORM
        X_val and X_test. Never fit on val or test data.

    Args:
        X_train (np.ndarray): Training features
        X_val (np.ndarray): Validation features
        X_test (np.ndarray): Test features

    Returns:
        tuple: (X_train_scaled, X_val_scaled, X_test_scaled)

    Hints:
        - Use sklearn.preprocessing.StandardScaler
        - scaler.fit_transform(X_train) — fit AND transform training data
        - scaler.transform(X_val) — only transform (scaler already fitted)
        - scaler.transform(X_test) — only transform
    """
    # TODO: Initialize StandardScaler
    # TODO: Fit scaler on X_train only, then transform X_train
    # TODO: Transform X_val using the already-fitted scaler
    # TODO: Transform X_test using the already-fitted scaler
    # TODO: Return (X_train_scaled, X_val_scaled, X_test_scaled)
    pass


# ---------------------------------------------------------------------------
# Model Training
# ---------------------------------------------------------------------------

def train_classifier(X_train, y_train, model_type="logistic_regression"):
    """
    Train a classifier on the training data.

    Args:
        X_train (np.ndarray): Training features (already scaled)
        y_train (np.ndarray): Training labels
        model_type (str): One of "logistic_regression", "decision_tree", "random_forest"

    Returns:
        Fitted sklearn model

    Hints:
        - LogisticRegression: use max_iter=1000, random_state=42
        - DecisionTreeClassifier: use max_depth=5, random_state=42
        - RandomForestClassifier: use n_estimators=100, random_state=42
        - Call model.fit(X_train, y_train) and return the fitted model
        - Raise ValueError for unknown model_type
    """
    # TODO: Create the appropriate model based on model_type
    # TODO: Fit the model on training data
    # TODO: Return the fitted model
    pass


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_classifier(model, X, y, split_name=""):
    """
    Evaluate a trained classifier and print metrics.

    Args:
        model: Fitted sklearn classifier
        X (np.ndarray): Feature matrix
        y (np.ndarray): True labels
        split_name (str): Label for this split (e.g., "Validation", "Test")

    Returns:
        dict: {"accuracy": float, "report": str}

    Hints:
        - Use model.predict(X) to get predictions
        - Use sklearn.metrics.accuracy_score for overall accuracy
        - Use sklearn.metrics.classification_report for per-class metrics
        - Print results with clear formatting
    """
    # TODO: Generate predictions
    # TODO: Compute accuracy
    # TODO: Generate classification report
    # TODO: Print results
    # TODO: Return dict with accuracy and report
    pass


# ---------------------------------------------------------------------------
# Learning Curve
# ---------------------------------------------------------------------------

def learning_curve_plot(model, X_train, y_train, title="Learning Curve"):
    """
    Plot training and cross-validation accuracy vs. number of training samples.

    This plot helps diagnose:
    - High bias (underfitting): Both curves plateau at low accuracy
    - High variance (overfitting): Large gap between train and CV curves

    Args:
        model: sklearn estimator (not yet fitted)
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        title (str): Plot title

    Hints:
        - Use sklearn.model_selection.learning_curve with cv=5
        - train_sizes should range from 10% to 100% of training data
        - Compute mean and std of train_scores and val_scores across folds
        - Plot mean ± 1 std dev as a shaded region (plt.fill_between)
        - X-axis: number of training samples, Y-axis: accuracy
    """
    # TODO: Call sklearn learning_curve with appropriate parameters
    # TODO: Compute mean and std of train and validation scores
    # TODO: Plot training curve (mean ± std)
    # TODO: Plot validation curve (mean ± std)
    # TODO: Add labels, title, legend, and show the plot
    pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    Full pipeline: load data → preprocess → train 3 models → compare results.

    Steps:
    1. Load the "digits" dataset (or "breast_cancer" for binary classification)
    2. Split into train/val/test
    3. Preprocess (standardize)
    4. Train LogisticRegression, DecisionTree, RandomForest
    5. Evaluate each on validation and test sets
    6. Plot learning curves for each model
    7. Print a summary table of results
    """
    print("=" * 60)
    print("Supervised Learning Pipeline")
    print("=" * 60)

    # TODO: Step 1 — Load and split dataset
    # Hint: Try dataset_name="digits" for a multi-class problem

    # TODO: Step 2 — Preprocess (standardize features)

    # TODO: Step 3 — Train all three classifiers

    # TODO: Step 4 — Evaluate each model on validation set
    # Hint: Use evaluate_classifier for each model

    # TODO: Step 5 — Evaluate best model on test set
    # Hint: Select best model based on validation accuracy, then evaluate on test

    # TODO: Step 6 — Plot learning curves for each model
    # Hint: Call learning_curve_plot for each trained model type

    # TODO: Step 7 — Print summary comparison table
    # Hint: Print model name, val accuracy, test accuracy in a formatted table


if __name__ == "__main__":
    main()
