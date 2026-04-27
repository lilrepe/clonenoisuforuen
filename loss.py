"""
NeuroFusionNet — Loss Function

Paper: "NeuroFusionNet: a hybrid EEG feature fusion framework for accurate
        and explainable Alzheimer's Disease detection"
Scientific Reports (2025) 15:43742
DOI: https://doi.org/10.1038/s41598-025-28070-x

Implements:
  §Classification network, Eq. 20 — categorical cross-entropy + L2 weight regularization

  L = -Σ_k y_k log ŷ_k + λ Σ_l ||W^(l)||_2^2
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuroFusionNetLoss(nn.Module):
    """§Classification network, Eq. 20 — Combined cross-entropy + L2 regularization loss.

    L = -Σ_k y_k log ŷ_k  +  λ Σ_l ||W^(l)||_2^2

    Note: L2 regularization is handled by PyTorch optimizer weight_decay parameter,
    which is equivalent to adding ||W||_2^2 to the loss. This class implements only
    the cross-entropy term; L2 is set via optimizer weight_decay=λ (Table 14: λ=0.001).

    For class imbalance, inverse-frequency class weights are used per Table 14.
    """

    def __init__(self, class_weights: Optional[torch.Tensor] = None):
        """
        Args:
            class_weights: Per-class weights for imbalanced data.
                Table 14 — "Class Weights: Inverse frequency-based"
                If None, uniform weights are used.
        """
        super().__init__()
        # §Classification, Eq. 20 — categorical cross-entropy
        # Note: nn.CrossEntropyLoss expects LOGITS, not probabilities.
        # Use model.forward_logits() during training.
        self.cross_entropy = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """§Classification, Eq. 20 — Cross-entropy term of the loss.

        L2 term (λ Σ_l ||W^(l)||_2^2) is handled by the optimizer's weight_decay.
        Set optimizer weight_decay = λ = 0.001 (Table 14).

        Args:
            logits: Raw model outputs (pre-softmax) — shape: (batch, n_classes)
                    Use model.forward_logits(), NOT model.forward()
            targets: Ground-truth class indices — shape: (batch,)
                    dtype: torch.long

        Returns:
            Scalar cross-entropy loss
        """
        # §Classification, Eq. 20 — -Σ_k y_k log ŷ_k
        return self.cross_entropy(logits, targets)


def compute_class_weights(class_counts: list, device: torch.device) -> torch.Tensor:
    """Table 14 — Compute inverse-frequency class weights.

    "Class Weights: Inverse frequency-based"

    weight_k = total_samples / (n_classes * count_k)

    Args:
        class_counts: List of sample counts per class [count_0, count_1, ..., count_K]
        device: Target device

    Returns:
        Class weight tensor — shape: (n_classes,)
    """
    counts = torch.tensor(class_counts, dtype=torch.float32)
    total = counts.sum()
    n_classes = len(counts)
    # Inverse frequency weighting (sklearn convention)
    weights = total / (n_classes * counts)
    return weights.to(device)
