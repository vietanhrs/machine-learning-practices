"""
Exercise 03: Transfer Learning with ResNet18
=============================================
Fine-tune a ResNet18 model pretrained on ImageNet for a new classification task.
Compare performance against training the same architecture from scratch.

Learning goals:
- Load and modify a pretrained model.
- Understand layer freezing and selective fine-tuning.
- Quantify the benefit of transfer learning on small datasets.

References:
    - PyTorch Transfer Learning Tutorial: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    - torchvision models: https://pytorch.org/vision/stable/models.html
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import time


# ---------------------------------------------------------------------------
# Model Setup
# ---------------------------------------------------------------------------

def load_pretrained_resnet(num_classes, freeze_backbone=True):
    """
    Load ResNet18 pretrained on ImageNet and adapt it for a new task.

    Args:
        num_classes (int): Number of output classes for the new task.
        freeze_backbone (bool): If True, freeze all layers except the classifier head.

    Returns:
        nn.Module: Modified ResNet18 model.

    TODO:
        1. Load pretrained ResNet18:
               model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        2. If freeze_backbone is True, freeze all parameters:
               for param in model.parameters():
                   param.requires_grad = False
        3. Replace the final fully-connected layer with a new one for num_classes:
               in_features = model.fc.in_features
               model.fc = nn.Linear(in_features, num_classes)
           (The new layer has requires_grad=True by default.)
        4. Return the model.

    Note: Only model.fc parameters are trainable when freeze_backbone=True.
          This is the classic "feature extraction" transfer learning mode.
    """
    # TODO: implement load_pretrained_resnet
    raise NotImplementedError("Implement load_pretrained_resnet(...)")


def replace_classifier_head(model, num_classes):
    """
    Replace the classifier head of a ResNet model with a deeper head.

    This more expressive head can help when the target task differs
    significantly from ImageNet.

    New head: Linear(in_features, 256) → ReLU → Dropout(0.3) → Linear(256, num_classes)

    Args:
        model (nn.Module): ResNet model (assumes model.fc is the current head).
        num_classes (int): Number of output classes.

    Returns:
        nn.Module: Model with replaced head. All other weights remain unchanged.

    TODO:
        1. Get in_features = model.fc.in_features.
        2. Replace model.fc with nn.Sequential containing:
               nn.Linear(in_features, 256),
               nn.ReLU(),
               nn.Dropout(0.3),
               nn.Linear(256, num_classes)
        3. Return model.
    """
    # TODO: implement replace_classifier_head
    raise NotImplementedError("Implement replace_classifier_head(...)")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_transfer(model, train_loader, val_loader, n_epochs, device, lr=1e-3):
    """
    Train a model using transfer learning strategy.

    Training phase 1 (head only, if backbone frozen):
        - Use Adam with lr for the head parameters only.

    Args:
        model (nn.Module): ResNet18 with frozen/unfrozen backbone.
        train_loader (DataLoader): Training data.
        val_loader (DataLoader): Validation data.
        n_epochs (int): Number of epochs.
        device: torch.device.
        lr (float): Learning rate for trainable parameters.

    Returns:
        dict: History with 'train_loss', 'val_loss', 'train_acc', 'val_acc'.

    TODO:
        1. Move model to device.
        2. Define criterion = nn.CrossEntropyLoss().
        3. Define optimizer to train only parameters with requires_grad=True:
               params = filter(lambda p: p.requires_grad, model.parameters())
               optimizer = optim.Adam(params, lr=lr)
        4. For each epoch:
               a. Training loop: model.train(), zero_grad, forward, loss, backward, step.
               b. Validation loop: model.eval(), torch.no_grad(), compute loss + accuracy.
               c. Print epoch summary.
        5. Return history dict.
    """
    # TODO: implement train_transfer
    raise NotImplementedError("Implement train_transfer(...)")


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_scratch_vs_transfer(train_loader, val_loader, num_classes, n_epochs, device):
    """
    Train two models and compare:
        1. ResNet18 trained FROM SCRATCH (random weights, no pretraining).
        2. ResNet18 with TRANSFER LEARNING (pretrained backbone, head only).

    Args:
        train_loader (DataLoader): Training data.
        val_loader (DataLoader): Validation data.
        num_classes (int): Number of target classes.
        n_epochs (int): Training epochs for each model.
        device: torch.device.

    Returns:
        tuple[dict, dict]: (history_scratch, history_transfer)

    TODO:
        1. Build scratch model:
               scratch_model = models.resnet18(weights=None)  # random init
               scratch_model.fc = nn.Linear(scratch_model.fc.in_features, num_classes)
        2. Train scratch model using train_transfer(..., freeze_backbone=False or similar).
        3. Build transfer model using load_pretrained_resnet(num_classes, freeze_backbone=True).
        4. Train transfer model using train_transfer(...).
        5. Plot both val_acc curves together for comparison.
        6. Return both history dicts.

    Note:
        Transfer learning should reach higher validation accuracy much faster,
        especially with limited training data.
    """
    # TODO: implement compare_scratch_vs_transfer
    raise NotImplementedError("Implement compare_scratch_vs_transfer(...)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Dataset: CIFAR-10 subset (simulate a "small custom dataset")
    # We use only 1000 training samples to show the power of transfer learning.
    # ------------------------------------------------------------------
    NUM_CLASSES = 10
    N_TRAIN = 1000    # intentionally small to show transfer learning advantage
    N_EPOCHS = 10

    # ResNet expects 224×224 input; CIFAR-10 is 32×32.
    # Resize + ImageNet normalization:
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])
    transform_val = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    full_train = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    val_ds = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_val
    )

    # Use a small random subset for training
    indices = torch.randperm(len(full_train))[:N_TRAIN]
    train_ds = Subset(full_train, indices)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=2)

    print(f"Small training set: {N_TRAIN} samples, Validation: {len(val_ds)} samples")

    # ------------------------------------------------------------------
    # Compare scratch vs transfer
    # ------------------------------------------------------------------
    hist_scratch, hist_transfer = compare_scratch_vs_transfer(
        train_loader, val_loader, NUM_CLASSES, N_EPOCHS, device
    )

    # ------------------------------------------------------------------
    # Plot comparison
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(hist_scratch['val_acc'],  label='Scratch (random init)', color='tomato')
    plt.plot(hist_transfer['val_acc'], label='Transfer Learning (pretrained)', color='steelblue')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title(f'Scratch vs Transfer Learning ({N_TRAIN} training samples)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    final_scratch  = hist_scratch['val_acc'][-1]
    final_transfer = hist_transfer['val_acc'][-1]
    print(f"\nFinal Validation Accuracy:")
    print(f"  From scratch:      {final_scratch:.3f}")
    print(f"  Transfer learning: {final_transfer:.3f}")
    print(f"  Improvement:       +{final_transfer - final_scratch:.3f}")

    # ------------------------------------------------------------------
    # Fine-tune the full network (unfreeze backbone)
    # ------------------------------------------------------------------
    print("\nFine-tuning: unfreezing backbone with lower LR...")
    pretrained_model = load_pretrained_resnet(NUM_CLASSES, freeze_backbone=False)

    # Use a lower LR when training the full network to protect pretrained features
    hist_finetune = train_transfer(
        pretrained_model, train_loader, val_loader,
        n_epochs=N_EPOCHS, device=device, lr=1e-4
    )

    plt.figure(figsize=(8, 5))
    plt.plot(hist_transfer['val_acc'], label='Transfer (head only)', linestyle='--')
    plt.plot(hist_finetune['val_acc'], label='Full fine-tuning (low LR)')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Transfer Learning: Head Only vs Full Fine-Tuning')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
