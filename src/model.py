"""
NeuroFusionNet — Classification Network

Paper: "NeuroFusionNet: a hybrid EEG feature fusion framework for accurate
        and explainable Alzheimer's Disease detection"
Scientific Reports (2025) 15:43742
DOI: https://doi.org/10.1038/s41598-025-28070-x

Implements:
  §Classification network — 5-layer DNN with skip connection (Table 15, Fig. 7)
  §Classification network, Eq. 17-19 — forward pass
  §Classification network, Eq. 20 — loss (cross-entropy + L2 regularization)

Section references:
  §Classification network, Table 15 — layer-wise architecture
  §Classification network, Fig. 7 — skip connection from Dense-1 to Dense-3
  §Classification network, Eq. 18 — h^(l) = f(W^(l) h^(l-1) + b^(l)), f = LeakyReLU
  §Classification network, Eq. 19 — ŷ = softmax(W^(L) h^(L-1) + b^(L))
  §Classification network, Eq. 20 — L = -Σ yk log ŷk + λ ||W^(l)||_2^2
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ClassifierConfig:
    """Configuration for NeuroFusionNet classification module.

    Defaults from Table 14 and Table 15 unless marked [UNSPECIFIED].
    """
    # Table 15 — layer widths
    input_dim: int = 70             # Table 12 — dfinal ≈ 60-80; using 70 as midpoint
                                    # [PARTIALLY_SPECIFIED] exact value depends on PCA result
    dense1_units: int = 512         # Table 15 — "Dense-1: 512 units, p=0.4"
    dense2_units: int = 256         # Table 15 — "Dense-2: 256 units (ReLU)" (skip connection source)
    dense3_units: int = 256         # Table 15 — "Dense-3: 256 units + skip, p=0.4"
    dense4_units: int = 128         # Table 15 — "Dense-4: 128 units, p=0.3"
    dense5_units: int = 64          # Table 15 — "Dense-5: 64 units, p=0.3"
    n_classes: int = 3              # §Data acquisition — AD / FTD / CN
    # Table 14 — dropout rates
    dropout_p1: float = 0.4         # Table 14 — "p1 = 0.4"
    dropout_p2: float = 0.4         # Table 14 — "p2 = 0.4"
    dropout_p3: float = 0.3         # Table 14 — "p3 = 0.3"
    dropout_p4: float = 0.3         # Table 14 — "p4 = 0.3"
    # [UNSPECIFIED] LeakyReLU negative slope not stated
    # Using: 0.01 (PyTorch default for LeakyReLU)
    # Alternatives: 0.1, 0.2 — steeper slope for stronger gradient flow
    leaky_relu_slope: float = 0.01
    # [UNSPECIFIED] BatchNorm epsilon not stated
    # Using: 1e-5 (PyTorch BatchNorm1d default)
    bn_eps: float = 1e-5


class NeuroFusionNetClassifier(nn.Module):
    """§Classification network — 5-layer DNN with residual skip connection.

    "It consists of five fully connected (dense) layers, each followed by batch
    normalization (BN), dropout, and a non-linear activation. A skip connection
    between the first and third dense layers facilitates residual learning."

    Architecture (Table 15, Fig. 7):
        Input (dfinal) →
        Dense-1 (512) + LeakyReLU + Dropout(p1=0.4) →
        Dense-2 (256) + ReLU         ← skip connection stored here
        Dense-3 (256) + Add(skip) + Dropout(p2=0.4) →
        Dense-4 (128) + BN + Dropout(p3=0.3) →
        Dense-5 (64)  + BN + Dropout(p4=0.3) →
        Output (n_classes) + Softmax
    """

    def __init__(self, config: ClassifierConfig):
        super().__init__()
        self.config = config

        # §Classification, Eq. 17 — input: x^(0) = f_final^(i) ∈ R^dfinal

        # Dense-1: §Table 15 — "Dense-1: Dense + LeakyReLU + Dropout, 512 units, p=0.4"
        self.dense1 = nn.Linear(config.input_dim, config.dense1_units)
        self.act1 = nn.LeakyReLU(negative_slope=config.leaky_relu_slope)  # §Eq. 18 — LeakyReLU
        self.drop1 = nn.Dropout(p=config.dropout_p1)

        # Dense-2: §Table 15 — "Dense-2: Skip Connection, 256 units (ReLU)"
        # This branch provides the skip connection tensor
        self.dense2 = nn.Linear(config.dense1_units, config.dense2_units)
        self.act2 = nn.ReLU()  # Table 15 explicitly says "ReLU" for Dense-2 (not LeakyReLU)

        # Dense-3: §Table 15 — "Dense-3: Dense + Add + Dropout, 256 units + skip, p=0.4"
        # Receives Dense-2 output + skip from Dense-1 projection
        # Skip from Dense-1 (512) must be projected to match Dense-3 size (256)
        # [UNSPECIFIED] how Dense-1 (512) skip connects to Dense-3 (256) not fully described
        # Fig. 7 shows the skip going from Dense-1 output into Dense-3; a linear projection is needed
        # Using: linear projection 512 → 256 for skip alignment
        self.skip_proj = nn.Linear(config.dense1_units, config.dense2_units)
        self.dense3 = nn.Linear(config.dense2_units, config.dense3_units)
        self.act3 = nn.LeakyReLU(negative_slope=config.leaky_relu_slope)  # §Eq. 18 — LeakyReLU
        self.drop3 = nn.Dropout(p=config.dropout_p2)  # Table 14: p2=0.4 applied at Dense-3

        # Dense-4: §Table 15 — "Dense-4: Dense + BN + Dropout, 128 units, p=0.3"
        self.dense4 = nn.Linear(config.dense3_units, config.dense4_units)
        self.bn4 = nn.BatchNorm1d(config.dense4_units, eps=config.bn_eps)
        self.act4 = nn.LeakyReLU(negative_slope=config.leaky_relu_slope)  # §Eq. 18
        self.drop4 = nn.Dropout(p=config.dropout_p3)

        # Dense-5: §Table 15 — "Dense-5: Dense + BN + Dropout, 64 units, p=0.3"
        self.dense5 = nn.Linear(config.dense4_units, config.dense5_units)
        self.bn5 = nn.BatchNorm1d(config.dense5_units, eps=config.bn_eps)
        self.act5 = nn.LeakyReLU(negative_slope=config.leaky_relu_slope)  # §Eq. 18
        self.drop5 = nn.Dropout(p=config.dropout_p4)

        # Output: §Table 15 — "Output: Dense + Softmax, 3 classes"
        # §Classification, Eq. 19 — ŷ = softmax(W^(L) h^(L-1) + b^(L))
        self.output = nn.Linear(config.dense5_units, config.n_classes)

        # [UNSPECIFIED] Weight initialization not stated
        # Using: PyTorch defaults (Kaiming uniform for Linear layers)
        # Alternatives: Xavier uniform, normal std=0.02 (GPT-style)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """§Classification, Eq. 17-19 — Forward pass through DNN.

        Args:
            x: PCA-reduced feature vector — shape: (batch, dfinal)

        Returns:
            Class probability distribution — shape: (batch, n_classes)
        """
        # §Classification, Eq. 17 — x^(0) = f_final^(i)
        # shape: (batch, dfinal)

        # Dense-1: §Table 15 — LeakyReLU + Dropout, 512 units
        h1 = self.dense1(x)       # (batch, 512)
        h1 = self.act1(h1)        # (batch, 512) — §Eq. 18 LeakyReLU
        h1 = self.drop1(h1)       # (batch, 512)

        # Dense-2: §Table 15 — ReLU, 256 units (skip connection source)
        h2 = self.dense2(h1)      # (batch, 256)
        h2 = self.act2(h2)        # (batch, 256) — Table 15: ReLU

        # Skip connection: project Dense-1 output (512) to Dense-3 size (256)
        # §Classification, Fig. 7 — "skip connection between the first and third dense layers"
        skip = self.skip_proj(h1) # (batch, 256) — [UNSPECIFIED] projection details

        # Dense-3: §Table 15 — Dense + Add(skip) + Dropout, 256 units
        h3 = self.dense3(h2)      # (batch, 256)
        h3 = h3 + skip            # (batch, 256) — §Fig. 7 residual addition
        h3 = self.act3(h3)        # (batch, 256) — §Eq. 18 LeakyReLU
        h3 = self.drop3(h3)       # (batch, 256)

        # Dense-4: §Table 15 — Dense + BN + Dropout, 128 units
        h4 = self.dense4(h3)      # (batch, 128)
        h4 = self.bn4(h4)         # (batch, 128) — §Table 15 BN
        h4 = self.act4(h4)        # (batch, 128) — §Eq. 18 LeakyReLU
        h4 = self.drop4(h4)       # (batch, 128)

        # Dense-5: §Table 15 — Dense + BN + Dropout, 64 units
        h5 = self.dense5(h4)      # (batch, 64)
        h5 = self.bn5(h5)         # (batch, 64) — §Table 15 BN
        h5 = self.act5(h5)        # (batch, 64) — §Eq. 18 LeakyReLU
        h5 = self.drop5(h5)       # (batch, 64)

        # Output: §Eq. 19 — ŷ = softmax(W^(L) h^(L-1) + b^(L))
        logits = self.output(h5)         # (batch, n_classes)
        return F.softmax(logits, dim=-1)  # (batch, n_classes) — probabilities

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits (pre-softmax) for use with nn.CrossEntropyLoss.

        Note: nn.CrossEntropyLoss expects logits, not softmax outputs.
        Use this method during training; use forward() for inference.

        Args:
            x: PCA-reduced feature vector — shape: (batch, dfinal)

        Returns:
            Logits — shape: (batch, n_classes)
        """
        h1 = self.dense1(x)
        h1 = self.act1(h1)
        h1 = self.drop1(h1)

        h2 = self.dense2(h1)
        h2 = self.act2(h2)

        skip = self.skip_proj(h1)
        h3 = self.dense3(h2) + skip
        h3 = self.act3(h3)
        h3 = self.drop3(h3)

        h4 = self.dense4(h3)
        h4 = self.bn4(h4)
        h4 = self.act4(h4)
        h4 = self.drop4(h4)

        h5 = self.dense5(h4)
        h5 = self.bn5(h5)
        h5 = self.act5(h5)
        h5 = self.drop5(h5)

        return self.output(h5)  # (batch, n_classes) — raw logits
