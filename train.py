"""
NeuroFusionNet — Training Loop

Paper: "NeuroFusionNet: a hybrid EEG feature fusion framework for accurate
        and explainable Alzheimer's Disease detection"
Scientific Reports (2025) 15:43742
DOI: https://doi.org/10.1038/s41598-025-28070-x

Implements:
  §Classification network — training configuration (Table 14)
  §Classification network — ReduceLROnPlateau schedule
  §Classification network — stratified 5-fold cross-validation (Table 16, Table 20)
  §Classification network, Eq. 20 — loss with L2 via weight_decay

Section references:
  Table 14 — Full training configuration
  Table 16 — Subject-level data split
  Table 20 — 5-fold CV results
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.data import EEGDataset
from src.loss import NeuroFusionNetLoss, compute_class_weights
from src.model import ClassifierConfig, NeuroFusionNetClassifier


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> Adam:
    """§Classification, Table 14 — Adam optimizer with L2 regularization.

    Table 14:
      Optimizer: Adam
      Initial Learning Rate: 3e-4
      Regularization: L2 (λ = 0.001)

    Note: weight_decay in Adam implements L2 regularization (Eq. 20, λ term).

    Args:
        model: NeuroFusionNetClassifier
        lr: Initial learning rate (Table 14: 3e-4)
        weight_decay: L2 regularization coefficient λ (Table 14: 0.001)

    Returns:
        Adam optimizer
    """
    # [UNSPECIFIED] Adam beta1, beta2, epsilon not stated
    # Using: PyTorch defaults (beta1=0.9, beta2=0.999, eps=1e-8)
    # Alternatives: (0.9, 0.98, 1e-9) as in original Transformer paper
    return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer: Adam, factor: float = 0.5, patience: int = 10) -> ReduceLROnPlateau:
    """§Classification, Table 14 — ReduceLROnPlateau learning rate scheduler.

    Table 14: "ReduceLROnPlateau (factor = 0.5, patience = 10)"

    "A dynamic learning rate adjustment was employed through the ReduceLROnPlateau
    scheduler... which automatically reduced the learning rate by half after 10
    consecutive epochs without validation improvement."

    "This mechanism lowered the learning rate to approximately 7.5e-5 between
    epochs 20 and 40."

    Args:
        optimizer: Adam optimizer
        factor: LR reduction factor (Table 14: 0.5)
        patience: Epochs without improvement before reducing LR (Table 14: 10)

    Returns:
        ReduceLROnPlateau scheduler
    """
    # [UNSPECIFIED] ReduceLROnPlateau mode ('min' or 'max') not stated
    # Using: 'min' on validation loss (standard practice)
    # Alternatives: 'max' on validation F1
    return ReduceLROnPlateau(optimizer, mode="min", factor=factor, patience=patience)


def train_one_epoch(
    model: NeuroFusionNetClassifier,
    loader: DataLoader,
    optimizer: Adam,
    criterion: NeuroFusionNetLoss,
    device: torch.device,
) -> Tuple[float, float]:
    """One training epoch.

    Args:
        model: NeuroFusionNetClassifier
        loader: Training DataLoader
        optimizer: Adam optimizer
        criterion: Loss function
        device: Compute device

    Returns:
        (mean_loss, accuracy) for this epoch
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for features, labels in loader:
        features = features.to(device)  # (batch, dfinal)
        labels = labels.to(device)       # (batch,)

        optimizer.zero_grad()

        # §Classification — use logits (not softmax) for cross-entropy
        logits = model.forward_logits(features)  # (batch, n_classes)

        # §Classification, Eq. 20 — cross-entropy (L2 handled by weight_decay)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += len(labels)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: NeuroFusionNetClassifier,
    loader: DataLoader,
    criterion: NeuroFusionNetLoss,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on a data split.

    Reports metrics matching Table 18: accuracy, macro precision/recall/F1, AUC.

    Args:
        model: NeuroFusionNetClassifier
        loader: DataLoader for eval split
        criterion: Loss function
        device: Compute device

    Returns:
        Dict with keys: loss, accuracy, macro_f1
    """
    from sklearn.metrics import accuracy_score, f1_score

    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        logits = model.forward_logits(features)
        loss = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    n = len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    # Table 18 — "macro-averaged" F1
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return {"loss": total_loss / n, "accuracy": acc, "macro_f1": macro_f1}


def train_fold(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    config: ClassifierConfig,
    device: torch.device,
    # Table 14 hyperparameters
    lr: float = 3e-4,
    weight_decay: float = 0.001,
    batch_size: int = 32,
    max_epochs: int = 500,
    early_stopping_patience: int = 10,
    lr_factor: float = 0.5,
    lr_patience: int = 10,
) -> Tuple[NeuroFusionNetClassifier, Dict]:
    """Train one cross-validation fold.

    §Classification, Table 14 — Full training procedure.

    "Training typically continued for 90–120 epochs until validation loss
    plateaued for 10 consecutive epochs, triggering early stopping."

    Args:
        train_features: Training features — shape: (n_train, dfinal)
        train_labels: Training labels — shape: (n_train,)
        val_features: Validation features — shape: (n_val, dfinal)
        val_labels: Validation labels — shape: (n_val,)
        config: ClassifierConfig
        device: Compute device
        lr: Initial learning rate (Table 14: 3e-4)
        weight_decay: L2 coefficient λ (Table 14: 0.001)
        batch_size: Mini-batch size (Table 14/Table 19 ablation: 32)
        max_epochs: Maximum epochs (Table 14: 500)
        early_stopping_patience: Epochs without val improvement (Table 14: 10)
        lr_factor: ReduceLROnPlateau factor (Table 14: 0.5)
        lr_patience: ReduceLROnPlateau patience (Table 14: 10)

    Returns:
        best_model: Model with highest validation F1 across epochs
        history: Training metrics history
    """
    # Compute inverse-frequency class weights
    # Table 14 — "Class Weights: Inverse frequency-based"
    unique, counts = np.unique(train_labels, return_counts=True)
    class_weights = compute_class_weights(counts.tolist(), device)

    # Build datasets and loaders
    train_dataset = EEGDataset(train_features, train_labels)
    val_dataset = EEGDataset(val_features, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Build model
    model = NeuroFusionNetClassifier(config).to(device)
    optimizer = build_optimizer(model, lr=lr, weight_decay=weight_decay)
    scheduler = build_scheduler(optimizer, factor=lr_factor, patience=lr_patience)
    criterion = NeuroFusionNetLoss(class_weights=class_weights)

    # Training loop with early stopping
    best_val_f1 = -1.0
    best_state = None
    no_improve_count = 0
    history: Dict = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_f1": []}

    for epoch in range(max_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        # §Classification, Table 14 — ReduceLROnPlateau on validation loss
        scheduler.step(val_metrics["loss"])

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["macro_f1"])

        # §Classification — "model achieving the highest validation F1-score within
        #                     each fold has been used for testing"
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve_count = 0
        else:
            no_improve_count += 1

        # Table 14 — "Early Stopping = 10"
        if no_improve_count >= early_stopping_patience:
            break

    # Restore best model state
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history
