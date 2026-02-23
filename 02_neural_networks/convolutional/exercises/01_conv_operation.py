"""
Exercise 01: Convolution from Scratch
=======================================
Implement 2D convolution and max pooling using only NumPy.
Understanding these operations at the loop level builds intuition
before using PyTorch's optimized C++/CUDA implementations.

Learning goals:
- Understand the sliding window computation in 2D convolution.
- Apply the output size formula.
- Visualize how different kernels produce different feature maps.

References:
    - CS231n Conv Layer: https://cs231n.github.io/convolutional-networks/
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import urllib.request
import os


# ---------------------------------------------------------------------------
# Core Operations
# ---------------------------------------------------------------------------

def output_size(input_size, kernel_size, stride, padding):
    """
    Compute the output spatial dimension of a convolution or pooling layer.

    Formula: floor((input_size + 2*padding - kernel_size) / stride) + 1

    Args:
        input_size (int): Spatial size of the input (height or width).
        kernel_size (int): Size of the kernel (height or width, assumed square).
        stride (int): Step size of the sliding window.
        padding (int): Number of zero-padding pixels added to each side.

    Returns:
        int: Output spatial dimension.

    TODO:
        Implement the formula above.
        Hint: Use Python's // operator for integer floor division.
    """
    # TODO: implement output_size formula
    raise NotImplementedError("Implement output_size(...)")


def conv2d(input_array, kernel, stride=1, padding=0):
    """
    2D convolution of a single-channel input with a single kernel.

    Args:
        input_array (np.ndarray): 2D input, shape (H, W).
        kernel (np.ndarray): 2D kernel, shape (kH, kW). Assumed square: kH == kW.
        stride (int): Sliding window step size.
        padding (int): Zero-padding added to all sides of the input.

    Returns:
        np.ndarray: Output feature map, shape (H_out, W_out).

    TODO:
        1. Pad the input using np.pad with mode='constant' (zero padding).
           Hint: np.pad(input_array, pad_width=padding, mode='constant')
        2. Compute output height and width using output_size().
        3. Initialize output array of zeros with shape (H_out, W_out).
        4. Slide the kernel over the padded input:
               for i in range(H_out):
                   for j in range(W_out):
                       row_start = i * stride
                       col_start = j * stride
                       patch = padded_input[row_start:row_start+kH, col_start:col_start+kW]
                       output[i, j] = np.sum(patch * kernel)
        5. Return the output array.
    """
    # TODO: implement conv2d
    raise NotImplementedError("Implement conv2d(input_array, kernel, stride, padding)")


def max_pool2d(input_array, pool_size=2, stride=2):
    """
    2D max pooling with a square pooling window.

    Args:
        input_array (np.ndarray): 2D input feature map, shape (H, W).
        pool_size (int): Size of the pooling window (height and width).
        stride (int): Step size of the pooling window.

    Returns:
        np.ndarray: Pooled feature map, shape (H_out, W_out).

    TODO:
        1. Compute H_out and W_out using output_size() with padding=0.
        2. Initialize output zeros of shape (H_out, W_out).
        3. Slide the pooling window:
               for i in range(H_out):
                   for j in range(W_out):
                       patch = input_array[i*stride:i*stride+pool_size,
                                           j*stride:j*stride+pool_size]
                       output[i, j] = np.max(patch)
        4. Return output.
    """
    # TODO: implement max_pool2d
    raise NotImplementedError("Implement max_pool2d(input_array, pool_size, stride)")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_kernels(kernels, kernel_names=None):
    """
    Visualize a set of 2D kernels as a grid of heatmaps.
    This function is complete — no changes needed.

    Args:
        kernels (list[np.ndarray]): List of 2D kernel arrays.
        kernel_names (list[str], optional): Names for each kernel.
    """
    n = len(kernels)
    if kernel_names is None:
        kernel_names = [f"Kernel {i+1}" for i in range(n)]

    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]

    for ax, kernel, name in zip(axes, kernels, kernel_names):
        im = ax.imshow(kernel, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title(name, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle("Convolutional Kernels", fontsize=13)
    plt.tight_layout()
    plt.show()


def get_sample_image(size=(64, 64)):
    """
    Create a synthetic sample image for demonstration.
    Returns a grayscale image as a numpy array.
    """
    # Create a simple synthetic image with geometric shapes
    img = np.zeros(size, dtype=np.float64)
    H, W = size

    # Add a rectangle
    img[H//4:3*H//4, W//4:3*W//4] = 0.8

    # Add a diagonal gradient
    for i in range(H):
        for j in range(W):
            img[i, j] += 0.2 * (i + j) / (H + W)

    # Normalize
    img = (img - img.min()) / (img.max() - img.min())
    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    np.random.seed(42)

    # ------------------------------------------------------------------
    # 1. Demonstrate output size formula
    # ------------------------------------------------------------------
    print("Output Size Formula Examples:")
    print(f"  Input=28, K=3, S=1, P=0 → {output_size(28, 3, 1, 0)}")  # expects 26
    print(f"  Input=28, K=3, S=1, P=1 → {output_size(28, 3, 1, 1)}")  # expects 28 (same)
    print(f"  Input=28, K=3, S=2, P=0 → {output_size(28, 3, 2, 0)}")  # expects 13
    print(f"  Input=32, K=5, S=2, P=0 → {output_size(32, 5, 2, 0)}")  # expects 14

    # ------------------------------------------------------------------
    # 2. Define classic image processing kernels
    # ------------------------------------------------------------------
    kernels = {
        "Horizontal Edge": np.array([
            [ 1,  1,  1],
            [ 0,  0,  0],
            [-1, -1, -1]
        ], dtype=np.float64),

        "Vertical Edge": np.array([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ], dtype=np.float64),

        "Sobel X": np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=np.float64),

        "Blur (Box)": np.ones((3, 3), dtype=np.float64) / 9,

        "Sharpen": np.array([
            [ 0, -1,  0],
            [-1,  5, -1],
            [ 0, -1,  0]
        ], dtype=np.float64),
    }

    visualize_kernels(
        list(kernels.values()),
        kernel_names=list(kernels.keys())
    )

    # ------------------------------------------------------------------
    # 3. Apply kernels to sample image, visualize feature maps
    # ------------------------------------------------------------------
    img = get_sample_image(size=(64, 64))

    n_kernels = len(kernels)
    fig, axes = plt.subplots(2, n_kernels + 1, figsize=(3 * (n_kernels + 1), 6))

    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')

    for col, (name, kernel) in enumerate(kernels.items(), start=1):
        # Apply convolution
        feature_map = conv2d(img, kernel, stride=1, padding=0)

        # Apply max pooling to the feature map
        pooled = max_pool2d(feature_map, pool_size=2, stride=2)

        axes[0, col].imshow(feature_map, cmap='gray')
        axes[0, col].set_title(f"{name}\n({feature_map.shape[0]}×{feature_map.shape[1]})")
        axes[0, col].axis('off')

        axes[1, col].imshow(pooled, cmap='gray')
        axes[1, col].set_title(f"After MaxPool\n({pooled.shape[0]}×{pooled.shape[1]})")
        axes[1, col].axis('off')

    axes[0, 0].set_ylabel("After Conv")
    axes[1, 0].set_ylabel("After MaxPool")

    plt.suptitle("Convolution Feature Maps and Max Pooling", fontsize=13)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # 4. Apply random learned kernels (as a CNN would use)
    # ------------------------------------------------------------------
    print("\nApplying 6 random learned kernels (simulating CNN layer 1)...")
    random_kernels = [np.random.randn(3, 3) for _ in range(6)]

    fig, axes = plt.subplots(2, 6, figsize=(14, 5))
    for col, k in enumerate(random_kernels):
        fmap = conv2d(img, k, stride=1, padding=1)  # same padding
        pooled = max_pool2d(fmap, pool_size=2, stride=2)

        axes[0, col].imshow(fmap, cmap='RdBu_r')
        axes[0, col].set_title(f"Feature {col+1}")
        axes[0, col].axis('off')

        axes[1, col].imshow(pooled, cmap='RdBu_r')
        axes[1, col].axis('off')

    axes[0, 0].set_ylabel("Conv output")
    axes[1, 0].set_ylabel("Pooled")
    plt.suptitle("6 Random Kernels → Feature Maps → Max Pooled", fontsize=13)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
