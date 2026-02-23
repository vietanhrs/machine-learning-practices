"""
Exercise 01: Feature Extraction + Traditional Classification
============================================================
Extract hand-crafted or flattened features from images and train
classical ML classifiers (SVM, logistic regression, random forest).
This demonstrates the traditional ML pipeline before deep learning.

Learning Goals:
    - Implement HOG feature extraction (or use skimage)
    - Train and compare SVM, logistic regression, random forest on features
    - Evaluate with accuracy, confusion matrix, and per-class metrics
    - Visualize misclassified samples

Requirements:
    pip install scikit-learn scikit-image matplotlib numpy
    (Optional: pip install tensorflow  # for FashionMNIST loader)
"""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

try:
    from skimage.feature import hog
    from skimage import color
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: skimage not available. HOG features will use simplified implementation.")


# ---------------------------------------------------------------------------
# Class names for FashionMNIST (used in main)
# ---------------------------------------------------------------------------
FASHION_MNIST_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


# ---------------------------------------------------------------------------
# 1. HOG Feature Extraction
# ---------------------------------------------------------------------------

def hog_features(image: np.ndarray) -> np.ndarray:
    """
    Compute Histogram of Oriented Gradients (HOG) features for a single image.

    Args:
        image: 2D numpy array (grayscale) or 3D (H, W, C).
               For MNIST/FashionMNIST, shape is (28, 28).

    Returns:
        1D numpy array of HOG feature values.

    TODO (Option A — use skimage.feature.hog):
        1. If image is 3D (color), convert to grayscale using color.rgb2gray(image).
        2. Call:
               features = hog(
                   image,
                   orientations=8,
                   pixels_per_cell=(4, 4),
                   cells_per_block=(2, 2),
                   visualize=False,
               )
        3. Return features.

    TODO (Option B — simplified implementation without skimage):
        1. Compute x-gradient: Gx = np.gradient(image, axis=1)
        2. Compute y-gradient: Gy = np.gradient(image, axis=0)
        3. Compute magnitude: mag = sqrt(Gx² + Gy²)
        4. Compute orientation: ang = arctan2(Gy, Gx) * (180/pi) % 180  (unsigned)
        5. Divide image into cells of size (cell_size × cell_size).
        6. For each cell, compute a histogram of orientations (n_bins bins from 0–180°).
        7. Concatenate all cell histograms into a single feature vector.
        8. L2-normalize the feature vector.
        9. Return the normalized feature vector.

    Hint (Option A):
        hog() returns a feature vector of length determined by image size and parameters.
        For a 28×28 image with pixels_per_cell=(4,4), cells_per_block=(2,2), orientations=8:
        → approximately 288 features.
    """
    raise NotImplementedError("TODO: implement hog_features()")


# ---------------------------------------------------------------------------
# 2. Flatten Features
# ---------------------------------------------------------------------------

def flatten_features(images: np.ndarray) -> np.ndarray:
    """
    Reshape a batch of images into a 2D feature matrix for sklearn.

    Args:
        images: Array of shape (n_samples, H, W) or (n_samples, H, W, C).

    Returns:
        2D array of shape (n_samples, H*W*C) with each image flattened.

    TODO:
        1. Compute n_samples = images.shape[0].
        2. Return images.reshape(n_samples, -1).

    Note:
        This is the simplest "feature extraction" — raw pixel values.
        It works surprisingly well with SVM for small, aligned images.
    """
    raise NotImplementedError("TODO: implement flatten_features()")


# ---------------------------------------------------------------------------
# 3. Extract HOG Features for a Batch
# ---------------------------------------------------------------------------

def extract_hog_features(images: np.ndarray) -> np.ndarray:
    """
    Extract HOG features for a batch of images.

    Args:
        images: Array of shape (n_samples, H, W) or (n_samples, H, W, C).

    Returns:
        2D array of shape (n_samples, n_hog_features).

    TODO:
        1. Apply hog_features(image) to each image in the batch.
        2. Stack results into a 2D array.
        3. Return the array.

    Hint:
        np.array([hog_features(img) for img in images])
    """
    raise NotImplementedError("TODO: implement extract_hog_features()")


# ---------------------------------------------------------------------------
# 4. Train Classifier
# ---------------------------------------------------------------------------

def train_classifier_on_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    classifier_type: str = "svm",
) -> Pipeline:
    """
    Train a classifier on pre-extracted features.

    Args:
        X_train:         2D array of shape (n_train, n_features).
        y_train:         1D array of integer labels.
        classifier_type: One of "svm", "logistic", "random_forest".

    Returns:
        Trained sklearn Pipeline (StandardScaler + classifier).

    TODO:
        1. Normalize features with StandardScaler (important for SVM and logistic regression).
        2. Choose the classifier based on classifier_type:
               "svm":          SVC(kernel='rbf', C=5.0, gamma='scale')
               "logistic":     LogisticRegression(max_iter=1000, C=1.0)
               "random_forest": RandomForestClassifier(n_estimators=100, random_state=42)
        3. Build a Pipeline([('scaler', scaler), ('clf', classifier)]).
        4. Fit the pipeline on (X_train, y_train).
        5. Return the fitted pipeline.

    Note:
        Pipeline ensures the same scaling is applied to test data automatically
        when calling pipeline.predict(X_test).
    """
    raise NotImplementedError("TODO: implement train_classifier_on_features()")


# ---------------------------------------------------------------------------
# 5. Evaluate Classifier
# ---------------------------------------------------------------------------

def evaluate_classifier(
    model: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Tuple[float, np.ndarray]:
    """
    Evaluate a trained classifier and display metrics.

    Args:
        model:       Trained sklearn Pipeline.
        X_test:      2D array of shape (n_test, n_features).
        y_test:      1D array of integer labels.
        class_names: Optional list of class name strings for display.

    Returns:
        accuracy:  Overall accuracy as a float.
        cm:        Confusion matrix as a 2D numpy array.

    TODO:
        1. Predict: y_pred = model.predict(X_test).
        2. Compute accuracy = accuracy_score(y_test, y_pred).
        3. Print accuracy.
        4. Print classification_report(y_test, y_pred, target_names=class_names).
        5. Compute cm = confusion_matrix(y_test, y_pred).
        6. Display the confusion matrix with ConfusionMatrixDisplay(cm, display_labels=class_names).
           Call disp.plot(xticks_rotation=45).
        7. Return (accuracy, cm).
    """
    raise NotImplementedError("TODO: implement evaluate_classifier()")


# ---------------------------------------------------------------------------
# 6. Plot Misclassified Samples (provided — complete implementation)
# ---------------------------------------------------------------------------

def plot_misclassified(
    X_test_images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]],
    n: int = 9,
) -> None:
    """
    Display a grid of misclassified images with true and predicted labels.

    Args:
        X_test_images: Original images (not features), shape (n_test, H, W).
        y_true:        Ground truth labels (integers).
        y_pred:        Predicted labels (integers).
        class_names:   List of class name strings.
        n:             Number of misclassified samples to display.
    """
    wrong_idx = np.where(y_true != y_pred)[0]
    if len(wrong_idx) == 0:
        print("No misclassified samples found!")
        return

    n = min(n, len(wrong_idx))
    sample_idx = wrong_idx[:n]

    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    for i, idx in enumerate(sample_idx):
        axes[i].imshow(X_test_images[idx], cmap="gray")
        true_label = class_names[y_true[idx]] if class_names else str(y_true[idx])
        pred_label = class_names[y_pred[idx]] if class_names else str(y_pred[idx])
        axes[i].set_title(f"True: {true_label}\nPred: {pred_label}", fontsize=8)
        axes[i].axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Misclassified Samples ({n} shown)", fontsize=12)
    plt.tight_layout()
    plt.savefig("misclassified_samples.png", dpi=100)
    plt.show()
    print("[Plot saved: misclassified_samples.png]")


# ---------------------------------------------------------------------------
# Helper: Load FashionMNIST (uses keras/tensorflow or torchvision)
# ---------------------------------------------------------------------------

def load_fashion_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load FashionMNIST dataset. Returns (X_train, y_train, X_test, y_test).
    Images are (n, 28, 28) uint8. Labels are integers 0-9.
    Falls back to MNIST if FashionMNIST is unavailable.
    """
    try:
        from torchvision import datasets
        train_data = datasets.FashionMNIST(root="/tmp/fmnist", train=True, download=True)
        test_data  = datasets.FashionMNIST(root="/tmp/fmnist", train=False, download=True)
        X_train = train_data.data.numpy()
        y_train = train_data.targets.numpy()
        X_test  = test_data.data.numpy()
        y_test  = test_data.targets.numpy()
        return X_train, y_train, X_test, y_test
    except Exception:
        pass

    try:
        from tensorflow.keras.datasets import fashion_mnist
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        return X_train, y_train, X_test, y_test
    except Exception:
        pass

    # Fallback: synthetic data
    print("Warning: Could not load FashionMNIST. Using synthetic data.")
    rng = np.random.default_rng(42)
    X_train = rng.integers(0, 255, (1000, 28, 28), dtype=np.uint8)
    y_train = rng.integers(0, 10, 1000, dtype=np.int64)
    X_test  = rng.integers(0, 255, (200, 28, 28), dtype=np.uint8)
    y_test  = rng.integers(0, 10, 200, dtype=np.int64)
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Compare feature extraction strategies on FashionMNIST.

    Steps:
        1. Load FashionMNIST (subsample for speed).
        2. Extract raw pixel features (flatten) and HOG features.
        3. Train SVM, Logistic Regression, and Random Forest on HOG features.
        4. Report accuracy and confusion matrix for each.
        5. Visualize misclassified samples.
    """
    print("=" * 65)
    print("Feature Extraction + Classification on FashionMNIST")
    print("=" * 65)

    # Load data (subsample for speed — use full dataset for best results)
    X_train, y_train, X_test, y_test = load_fashion_mnist()
    n_train, n_test = 5000, 1000
    X_train, y_train = X_train[:n_train], y_train[:n_train]
    X_test, y_test   = X_test[:n_test],   y_test[:n_test]
    print(f"\nTrain: {X_train.shape} | Test: {X_test.shape}")

    # Normalize to [0, 1]
    X_train_norm = X_train.astype(np.float32) / 255.0
    X_test_norm  = X_test.astype(np.float32) / 255.0

    # --- Feature extraction ---
    print("\nExtracting HOG features...")
    X_train_hog = extract_hog_features(X_train_norm)
    X_test_hog  = extract_hog_features(X_test_norm)
    print(f"HOG feature shape: {X_train_hog.shape}")

    print("Extracting flattened pixel features...")
    X_train_flat = flatten_features(X_train_norm)
    X_test_flat  = flatten_features(X_test_norm)
    print(f"Flat feature shape: {X_train_flat.shape}")

    # --- Train and evaluate classifiers ---
    experiments = [
        ("SVM on HOG",            X_train_hog,  X_test_hog,  "svm"),
        ("Logistic on HOG",        X_train_hog,  X_test_hog,  "logistic"),
        ("Random Forest on HOG",   X_train_hog,  X_test_hog,  "random_forest"),
        ("SVM on Flat Pixels",     X_train_flat, X_test_flat, "svm"),
    ]

    results = {}
    for name, X_tr, X_te, clf_type in experiments:
        print(f"\n{'='*50}")
        print(f"Experiment: {name}")
        print(f"{'='*50}")
        model = train_classifier_on_features(X_tr, y_train, classifier_type=clf_type)
        acc, cm = evaluate_classifier(model, X_te, y_test, class_names=FASHION_MNIST_CLASSES)
        results[name] = acc

        # Misclassified visualization (first experiment only)
        if name == "SVM on HOG":
            y_pred = model.predict(X_te)
            plot_misclassified(X_test, y_test, y_pred, class_names=FASHION_MNIST_CLASSES, n=9)

    # --- Summary ---
    print("\n--- Accuracy Summary ---")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name:<35} {acc:.4f}")


if __name__ == "__main__":
    main()
