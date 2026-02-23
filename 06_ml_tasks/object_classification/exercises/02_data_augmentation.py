"""
Exercise 02: Data Augmentation for Image Classification
========================================================
Implement common image augmentation techniques from scratch and study
their effect on model generalization. Then compare training with and
without augmentation on a small dataset.

Learning Goals:
    - Implement standard augmentations (flip, rotation, crop, color jitter, mixup)
    - Build a composable augmentation pipeline
    - Quantify the benefit of augmentation through train/test accuracy comparison
    - Visualize the effect of each augmentation type

Requirements:
    pip install numpy scikit-learn matplotlib pillow scikit-image
"""

import random
from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 1. Random Horizontal Flip
# ---------------------------------------------------------------------------

def random_horizontal_flip(image: np.ndarray, p: float = 0.5) -> np.ndarray:
    """
    Randomly flip the image horizontally (left-right mirror).

    Args:
        image: 2D (H, W) grayscale or 3D (H, W, C) color array.
        p:     Probability of applying the flip (default 0.5).

    Returns:
        Flipped image or original image (both with same shape).

    TODO:
        1. Generate a random float: r = random.random() or np.random.rand().
        2. If r < p:
               return np.fliplr(image)  # flips along axis=1 (horizontal)
        3. Else:
               return image.copy()

    Note:
        np.fliplr works on 2D and 3D arrays alike.
    """
    raise NotImplementedError("TODO: implement random_horizontal_flip()")


# ---------------------------------------------------------------------------
# 2. Random Rotation
# ---------------------------------------------------------------------------

def random_rotation(image: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """
    Rotate the image by a uniformly sampled angle in [-max_angle, +max_angle].

    Args:
        image:     2D (H, W) or 3D (H, W, C) image array. Values in [0, 1] or [0, 255].
        max_angle: Maximum rotation angle in degrees.

    Returns:
        Rotated image of the same shape.

    TODO:
        1. Sample angle: angle = np.random.uniform(-max_angle, max_angle).
        2. Use skimage.transform.rotate(image, angle, mode='reflect', preserve_range=True)
           OR implement a manual rotation using scipy.ndimage.rotate.
        3. Clip output values to [0, 1] or [0, 255] as appropriate.
        4. Return rotated image (same dtype as input).

    Hint (without skimage):
        from scipy.ndimage import rotate
        return rotate(image, angle, reshape=False, mode='reflect')

    Hint (with skimage):
        from skimage.transform import rotate as sk_rotate
        return sk_rotate(image, angle, mode='reflect', preserve_range=True)
    """
    raise NotImplementedError("TODO: implement random_rotation()")


# ---------------------------------------------------------------------------
# 3. Random Crop
# ---------------------------------------------------------------------------

def random_crop(image: np.ndarray, crop_size: Tuple[int, int]) -> np.ndarray:
    """
    Crop a random subregion of size crop_size from the image, then
    resize it back to the original image dimensions.

    Args:
        image:     2D (H, W) or 3D (H, W, C) image array.
        crop_size: (crop_H, crop_W) — size of the crop to extract.

    Returns:
        Cropped and resized image of the same shape as input.

    TODO:
        1. Get original height H and width W from image.shape[:2].
        2. Sample top-left corner of the crop:
               top  = np.random.randint(0, H - crop_size[0])
               left = np.random.randint(0, W - crop_size[1])
        3. Extract the crop:
               cropped = image[top:top+crop_size[0], left:left+crop_size[1]]
        4. Resize the crop back to (H, W) using:
               from skimage.transform import resize
               resized = resize(cropped, (H, W), preserve_range=True, anti_aliasing=True)
           OR use PIL.Image.fromarray(...).resize((W, H)).
        5. Return resized crop (same shape as input).

    Note:
        crop_size must be smaller than (H, W). Add assertion if desired.
    """
    raise NotImplementedError("TODO: implement random_crop()")


# ---------------------------------------------------------------------------
# 4. Color Jitter
# ---------------------------------------------------------------------------

def color_jitter(
    image: np.ndarray,
    brightness: float = 0.2,
    contrast: float = 0.2,
) -> np.ndarray:
    """
    Randomly adjust brightness and contrast of the image.

    Args:
        image:      Image array. Values should be in [0, 1].
                    If (H, W, C) color, applies per-channel; if (H, W) grayscale, applies directly.
        brightness: Maximum brightness perturbation factor.
                    Adds a random value in [-brightness, +brightness].
        contrast:   Maximum contrast perturbation factor.
                    Multiplies by a random factor in [1-contrast, 1+contrast].

    Returns:
        Jittered image, clipped to [0, 1].

    TODO:
        1. Apply brightness jitter:
               bright_delta = np.random.uniform(-brightness, brightness)
               image = image + bright_delta
        2. Apply contrast jitter:
               contrast_factor = np.random.uniform(1 - contrast, 1 + contrast)
               mean = image.mean()
               image = (image - mean) * contrast_factor + mean
        3. Clip to [0.0, 1.0]: image = np.clip(image, 0.0, 1.0)
        4. Return the jittered image.

    Note:
        Input should be float in [0, 1]. If the input is uint8 [0, 255], normalize
        before calling and denormalize after.
    """
    raise NotImplementedError("TODO: implement color_jitter()")


# ---------------------------------------------------------------------------
# 5. Mixup Augmentation
# ---------------------------------------------------------------------------

def mixup(
    image1: np.ndarray,
    label1: int,
    image2: np.ndarray,
    label2: int,
    n_classes: int,
    alpha: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Mixup augmentation: blend two images and their labels.

    Mixup creates a new training sample by linearly interpolating between
    two existing samples and their labels. This encourages the model to
    predict smoothly between classes.

    Args:
        image1:    First image array (H, W) or (H, W, C).
        label1:    Integer class label for image1.
        image2:    Second image array (same shape as image1).
        label2:    Integer class label for image2.
        n_classes: Total number of classes (for one-hot encoding).
        alpha:     Parameter of the Beta distribution used to sample λ.
                   Higher α → λ closer to 0.5 (more mixing).
                   Lower α → λ closer to 0 or 1 (less mixing).

    Returns:
        mixed_image: λ * image1 + (1-λ) * image2.
        mixed_label: λ * one_hot(label1) + (1-λ) * one_hot(label2).
                     (Soft label vector of shape (n_classes,).)

    TODO:
        1. Sample λ from Beta(alpha, alpha): lam = np.random.beta(alpha, alpha).
        2. Compute mixed_image = lam * image1 + (1 - lam) * image2.
        3. One-hot encode labels:
               oh1 = np.zeros(n_classes); oh1[label1] = 1.0
               oh2 = np.zeros(n_classes); oh2[label2] = 1.0
        4. Compute mixed_label = lam * oh1 + (1 - lam) * oh2.
        5. Return (mixed_image, mixed_label).

    Reference:
        Zhang et al. (2018) "mixup: Beyond Empirical Risk Minimization"
        https://arxiv.org/abs/1710.09412
    """
    raise NotImplementedError("TODO: implement mixup()")


# ---------------------------------------------------------------------------
# 6. Augmentation Pipeline
# ---------------------------------------------------------------------------

def augmentation_pipeline(
    image: np.ndarray,
    augmentations: List[Callable[[np.ndarray], np.ndarray]],
) -> np.ndarray:
    """
    Apply a sequence of augmentation functions to an image.

    Args:
        image:        Input image array.
        augmentations: List of augmentation callables, each accepting an image
                       and returning an augmented image.

    Returns:
        Augmented image after applying all functions in order.

    TODO:
        1. For each augmentation function aug in augmentations:
               image = aug(image)
        2. Return the final image.

    Example usage:
        pipeline = [
            lambda img: random_horizontal_flip(img, p=0.5),
            lambda img: random_rotation(img, max_angle=10),
            lambda img: color_jitter(img, brightness=0.2, contrast=0.2),
        ]
        augmented = augmentation_pipeline(image, pipeline)
    """
    raise NotImplementedError("TODO: implement augmentation_pipeline()")


# ---------------------------------------------------------------------------
# 7. Compare Training With/Without Augmentation
# ---------------------------------------------------------------------------

def compare_training_with_without_augmentation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    augment_factor: int = 3,
) -> Tuple[float, float]:
    """
    Compare test accuracy of a logistic regression classifier trained on
    original data vs. augmented data.

    Args:
        X_train:        Original training images (n_train, H, W), float [0, 1].
        y_train:        Training labels.
        X_test:         Test images (n_test, H, W), float [0, 1].
        y_test:         Test labels.
        augment_factor: How many augmented copies to add per original image.

    Returns:
        acc_no_aug:   Test accuracy without augmentation.
        acc_with_aug: Test accuracy with augmentation.

    TODO:
        1. Build baseline (no augmentation):
               a. Flatten X_train: X_flat = X_train.reshape(n_train, -1).
               b. Train LogisticRegression on X_flat.
               c. Evaluate on flattened X_test.
               d. Record acc_no_aug.

        2. Build augmented dataset:
               Start with the original training images.
               For each image, generate augment_factor augmented copies by applying:
                   - random_horizontal_flip(img, p=0.5)
                   - random_rotation(img, max_angle=12)
                   - color_jitter(img, brightness=0.2, contrast=0.2)
               Append augmented images and their labels to the training set.

        3. Flatten augmented training set and train a new LogisticRegression.
        4. Evaluate on (same) X_test.
        5. Record acc_with_aug.
        6. Print comparison table.
        7. Return (acc_no_aug, acc_with_aug).

    Hint:
        augmented_images = []
        augmented_labels = []
        pipeline = [
            lambda img: random_horizontal_flip(img),
            lambda img: random_rotation(img, max_angle=12),
            lambda img: color_jitter(img),
        ]
        for img, lbl in zip(X_train, y_train):
            for _ in range(augment_factor):
                aug_img = augmentation_pipeline(img, pipeline)
                augmented_images.append(aug_img)
                augmented_labels.append(lbl)
    """
    raise NotImplementedError("TODO: implement compare_training_with_without_augmentation()")


# ---------------------------------------------------------------------------
# 8. Visualize Augmentations (provided — complete implementation)
# ---------------------------------------------------------------------------

def visualize_augmentations(
    image: np.ndarray,
    augmentations: List[Tuple[str, Callable[[np.ndarray], np.ndarray]]],
) -> None:
    """
    Display the original image alongside augmented versions.

    Args:
        image:        Original image (H, W) or (H, W, C), values in [0, 1].
        augmentations: List of (name, augmentation_function) tuples.
    """
    n = len(augmentations) + 1  # +1 for original
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    # Original
    axes[0].imshow(image, cmap="gray" if image.ndim == 2 else None, vmin=0, vmax=1)
    axes[0].set_title("Original", fontsize=9)
    axes[0].axis("off")

    # Augmented versions
    for i, (name, aug_fn) in enumerate(augmentations):
        aug_img = aug_fn(image.copy())
        axes[i + 1].imshow(
            np.clip(aug_img, 0, 1),
            cmap="gray" if aug_img.ndim == 2 else None,
            vmin=0, vmax=1,
        )
        axes[i + 1].set_title(name, fontsize=9)
        axes[i + 1].axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Image Augmentation Effects", fontsize=12)
    plt.tight_layout()
    plt.savefig("augmentation_visualization.png", dpi=100)
    plt.show()
    print("[Plot saved: augmentation_visualization.png]")


# ---------------------------------------------------------------------------
# Helper: Load or generate a small image dataset
# ---------------------------------------------------------------------------

def load_small_dataset(
    n_train: int = 500, n_test: int = 200, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Try to load FashionMNIST; fall back to synthetic data.
    Returns (X_train, y_train, X_test, y_test) with images in [0, 1].
    """
    try:
        from torchvision import datasets
        train_data = datasets.FashionMNIST(root="/tmp/fmnist", train=True, download=True)
        test_data  = datasets.FashionMNIST(root="/tmp/fmnist", train=False, download=True)
        X_tr = train_data.data.numpy()[:n_train].astype(np.float32) / 255.0
        y_tr = train_data.targets.numpy()[:n_train]
        X_te = test_data.data.numpy()[:n_test].astype(np.float32) / 255.0
        y_te = test_data.targets.numpy()[:n_test]
        return X_tr, y_tr, X_te, y_te
    except Exception:
        pass

    print("Warning: Using synthetic random images.")
    rng = np.random.default_rng(seed)
    X_tr = rng.random((n_train, 28, 28), dtype=np.float32)
    y_tr = rng.integers(0, 10, n_train)
    X_te = rng.random((n_test, 28, 28), dtype=np.float32)
    y_te = rng.integers(0, 10, n_test)
    return X_tr, y_tr, X_te, y_te


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Demonstrate augmentation techniques and their effect on classifier performance.

    Steps:
        1. Load a small image dataset (FashionMNIST subset).
        2. Visualize each augmentation type on a sample image.
        3. Compare train/test accuracy with and without augmentation.
    """
    print("=" * 60)
    print("Data Augmentation for Image Classification")
    print("=" * 60)

    X_train, y_train, X_test, y_test = load_small_dataset(n_train=500, n_test=200)
    sample_image = X_train[0]
    print(f"\nDataset: train={X_train.shape}, test={X_test.shape}")
    print(f"Image shape: {sample_image.shape}, range: [{sample_image.min():.2f}, {sample_image.max():.2f}]")

    # --- Visualize augmentations ---
    augmentation_list = [
        ("H-Flip (p=1)",         lambda img: np.fliplr(img)),
        ("Rotation +15°",         lambda img: random_rotation(img, max_angle=15)),
        ("Rotation -15°",         lambda img: random_rotation(img, max_angle=15)),
        ("Brightness +0.3",       lambda img: color_jitter(img, brightness=0.3, contrast=0.0)),
        ("Contrast ×1.5",         lambda img: color_jitter(img, brightness=0.0, contrast=0.4)),
        ("Random Crop (22×22)",   lambda img: random_crop(img, (22, 22))),
        ("Full Pipeline",         lambda img: augmentation_pipeline(img, [
                                      lambda i: random_horizontal_flip(i),
                                      lambda i: random_rotation(i, max_angle=10),
                                      lambda i: color_jitter(i),
                                  ])),
    ]
    visualize_augmentations(sample_image, augmentation_list)

    # --- Mixup example ---
    print("\n--- Mixup Example ---")
    img1, lbl1 = X_train[0], int(y_train[0])
    img2, lbl2 = X_train[1], int(y_train[1])
    mixed_img, mixed_lbl = mixup(img1, lbl1, img2, lbl2, n_classes=10, alpha=0.4)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(img1, cmap="gray"); axes[0].set_title(f"Image 1 (class {lbl1})")
    axes[1].imshow(img2, cmap="gray"); axes[1].set_title(f"Image 2 (class {lbl2})")
    axes[2].imshow(np.clip(mixed_img, 0, 1), cmap="gray")
    axes[2].set_title(f"Mixup\n{np.round(mixed_lbl, 2)}")
    for ax in axes: ax.axis("off")
    plt.tight_layout()
    plt.savefig("mixup_example.png", dpi=100)
    plt.show()
    print("[Plot saved: mixup_example.png]")

    # --- Compare with/without augmentation ---
    print("\n--- Effect of Augmentation on Classification Accuracy ---")
    acc_no_aug, acc_with_aug = compare_training_with_without_augmentation(
        X_train, y_train, X_test, y_test, augment_factor=3
    )
    print(f"\nNo augmentation:   {acc_no_aug:.4f}")
    print(f"With augmentation: {acc_with_aug:.4f}")
    improvement = acc_with_aug - acc_no_aug
    print(f"Improvement:       {improvement:+.4f} ({'gain' if improvement >= 0 else 'loss'})")


if __name__ == "__main__":
    main()
