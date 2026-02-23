"""
Exercise 02: CNN Classifier with PyTorch
=========================================
Build a convolutional neural network for image classification using PyTorch.
You will implement a ConvBlock module, a SimpleCNN model, training and evaluation
loops, and feature map visualization.

Learning goals:
- Use nn.Conv2d, nn.BatchNorm2d, nn.MaxPool2d in PyTorch.
- Implement training and evaluation with GPU support.
- Visualize what CNN layers "see" with feature map hooks.

References:
    - PyTorch CIFAR-10 tutorial: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    - PyTorch Conv2d: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """
    A standard convolutional block: Conv2d → BatchNorm2d → ReLU → MaxPool2d.

    This is the fundamental building block of modern CNNs.

    Args:
        in_channels (int): Number of input feature map channels.
        out_channels (int): Number of output feature map channels (number of filters).
        kernel_size (int): Convolutional kernel size (default 3).
        pool_size (int): Max pooling window size (default 2).

    TODO:
        In __init__, define the following sequence as self.block using nn.Sequential:
            1. nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
               (padding=kernel_size//2 for "same" padding with odd kernel sizes)
            2. nn.BatchNorm2d(out_channels)
            3. nn.ReLU(inplace=True)
            4. nn.MaxPool2d(pool_size)

        In forward(self, x):
            Return self.block(x).
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, pool_size=2):
        super().__init__()
        # TODO: define self.block as nn.Sequential with the components above
        raise NotImplementedError("Implement ConvBlock.__init__")

    def forward(self, x):
        # TODO: return self.block(x)
        raise NotImplementedError("Implement ConvBlock.forward")


class SimpleCNN(nn.Module):
    """
    A simple CNN with 3 ConvBlocks followed by a classifier head.

    Architecture:
        ConvBlock(in_channels=3, out_channels=32)    → halves spatial dims
        ConvBlock(in_channels=32, out_channels=64)   → halves spatial dims
        ConvBlock(in_channels=64, out_channels=128)  → halves spatial dims
        AdaptiveAvgPool2d(1)                         → global average pooling
        Flatten
        Linear(128, 256)
        ReLU
        Dropout(0.5)
        Linear(256, n_classes)

    For FashionMNIST (1 channel, 28×28):
        First ConvBlock: in_channels=1 (grayscale)
    For CIFAR-10 (3 channels, 32×32):
        First ConvBlock: in_channels=3 (RGB)

    Args:
        n_classes (int): Number of output classes.
        in_channels (int): Number of input image channels (1 or 3).

    TODO:
        In __init__:
            1. Create self.features as nn.Sequential with the 3 ConvBlocks.
            2. Create self.pool = nn.AdaptiveAvgPool2d(1).
            3. Create self.classifier as nn.Sequential with:
               Linear(128, 256) → ReLU → Dropout(0.5) → Linear(256, n_classes)

        In forward(self, x):
            1. x = self.features(x)
            2. x = self.pool(x)
            3. x = x.view(x.size(0), -1)   ← flatten
            4. x = self.classifier(x)
            5. return x
    """

    def __init__(self, n_classes=10, in_channels=1):
        super().__init__()
        # TODO: implement architecture
        raise NotImplementedError("Implement SimpleCNN.__init__")

    def forward(self, x):
        # TODO: implement forward pass
        raise NotImplementedError("Implement SimpleCNN.forward")


# ---------------------------------------------------------------------------
# Training and Evaluation
# ---------------------------------------------------------------------------

def train_model(model, train_loader, val_loader, n_epochs, lr, device):
    """
    Full training loop for a CNN classifier.

    For each epoch:
        - Train on train_loader (model.train() mode).
        - Evaluate on val_loader (model.eval() mode).
        - Print epoch summary.

    Args:
        model (nn.Module): The CNN model.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        n_epochs (int): Number of training epochs.
        lr (float): Learning rate.
        device: torch.device.

    Returns:
        dict: History with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'.

    TODO:
        1. Define optimizer: optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        2. Define criterion: nn.CrossEntropyLoss()
        3. Define a learning rate scheduler: optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        4. For each epoch:
               a. Training loop: zero_grad, forward, loss, backward, step.
               b. Validation loop: torch.no_grad(), compute loss and accuracy.
               c. scheduler.step()
               d. Print and store metrics.
        5. Return history dict.

    Hint:
        model.to(device) should be called before this function.
        Move batches to device: X, y = X.to(device), y.to(device)
    """
    # TODO: implement train_model
    raise NotImplementedError("Implement train_model(...)")


def evaluate_model(model, test_loader, device, class_names=None):
    """
    Evaluate model on the test set.
    Returns accuracy and plots a confusion matrix.

    Args:
        model (nn.Module): Trained CNN model.
        test_loader (DataLoader): Test data loader.
        device: torch.device.
        class_names (list[str], optional): Class labels for confusion matrix axes.

    Returns:
        float: Overall test accuracy.

    TODO:
        1. Set model.eval() and use torch.no_grad().
        2. Collect all true labels (y_true) and predicted labels (y_pred).
        3. Compute accuracy = correct / total.
        4. Compute confusion matrix using sklearn.metrics.confusion_matrix.
        5. Plot the confusion matrix using seaborn.heatmap (provided scaffold below).
        6. Print and return accuracy.

    Hint:
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X, y in test_loader:
                ...
                preds = output.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
    """
    # TODO: implement evaluate_model
    raise NotImplementedError("Implement evaluate_model(...)")


# ---------------------------------------------------------------------------
# Feature Map Visualization
# ---------------------------------------------------------------------------

def visualize_feature_maps(model, sample_image, layer_name, device):
    """
    Visualize the feature maps produced by a named layer of the CNN
    for a single input image.

    Uses PyTorch forward hooks to capture intermediate activations.

    Args:
        model (nn.Module): Trained CNN.
        sample_image (torch.Tensor): Single image, shape (1, C, H, W).
        layer_name (str): Name of the layer to hook (e.g., 'features.0').
        device: torch.device.

    TODO:
        1. Register a forward hook on the specified layer:
               activation = {}
               def hook_fn(module, input, output):
                   activation['output'] = output.detach()
               layer = dict(model.named_modules())[layer_name]
               handle = layer.register_forward_hook(hook_fn)
        2. Run a forward pass on sample_image (no_grad).
        3. Remove the hook: handle.remove()
        4. Plot the first 16 feature maps as a 4×4 grid.

    Hint:
        feature_maps = activation['output'][0]  # remove batch dim
        for i in range(min(16, feature_maps.shape[0])):
            axes[i//4, i%4].imshow(feature_maps[i].cpu().numpy(), cmap='viridis')
    """
    # TODO: implement feature map visualization
    raise NotImplementedError("Implement visualize_feature_maps(...)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Dataset: FashionMNIST (10 classes, 28×28 grayscale)
    # Change to CIFAR-10 (torchvision.datasets.CIFAR10) for a harder challenge.
    # ------------------------------------------------------------------
    DATASET = "FashionMNIST"  # or "CIFAR10"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # single channel for FashionMNIST
    ])

    if DATASET == "FashionMNIST":
        train_ds = torchvision.datasets.FashionMNIST(
            root='./data', train=True, download=True, transform=transform
        )
        test_ds = torchvision.datasets.FashionMNIST(
            root='./data', train=False, download=True, transform=transform
        )
        IN_CHANNELS = 1
        class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    else:
        # CIFAR-10: change normalize to ((0.5,0.5,0.5),(0.5,0.5,0.5))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_ds = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
        test_ds = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
        IN_CHANNELS = 3
        class_names = ['plane', 'car', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']

    # Use a subset of training data for speed
    n_train = 10000
    train_ds = torch.utils.data.Subset(train_ds, range(n_train))

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False, num_workers=2)

    print(f"Dataset: {DATASET}")
    print(f"Training samples: {len(train_ds)}, Test samples: {len(test_ds)}")

    # ------------------------------------------------------------------
    # Build and train model
    # ------------------------------------------------------------------
    model = SimpleCNN(n_classes=10, in_channels=IN_CHANNELS).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    history = train_model(
        model, train_loader, test_loader,
        n_epochs=10, lr=1e-3, device=device
    )

    # ------------------------------------------------------------------
    # Plot training curves
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Loss')

    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.set_title('Accuracy')

    plt.suptitle(f"CNN Training on {DATASET}")
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # Evaluate and visualize
    # ------------------------------------------------------------------
    evaluate_model(model, test_loader, device, class_names=class_names)

    # Visualize feature maps from the first ConvBlock
    sample_img, _ = test_ds[0]
    sample_img = sample_img.unsqueeze(0).to(device)
    visualize_feature_maps(model, sample_img, layer_name='features', device=device)


if __name__ == "__main__":
    main()
