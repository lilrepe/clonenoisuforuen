"""
NeuroFusionNet — Feature Extraction

Paper: "NeuroFusionNet: a hybrid EEG feature fusion framework for accurate
        and explainable Alzheimer's Disease detection"
Scientific Reports (2025) 15:43742
DOI: https://doi.org/10.1038/s41598-025-28070-x

Implements:
  - Handcrafted features: §Feature extraction, Table 10
      Spectral (Welch PSD), Wavelet (db4, level-5), Statistical, Permutation Entropy
  - Automated features: §Automated feature extraction
      1D-CNN (3 conv blocks) → Global Average Pooling → d2=256 embedding
"""

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pywt
import torch
import torch.nn as nn
from scipy.signal import welch
from scipy.stats import kurtosis, skew


# ---------------------------------------------------------------------------
# Handcrafted Feature Extraction
# §Feature extraction, Table 10
# ---------------------------------------------------------------------------

def bandpower(psd: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float) -> float:
    """§Feature extraction, Eq. 8 — Log bandpower.

    Pband = log( ∫[f1,f2] P(f) df + ε )

    Args:
        psd: Power spectral density — shape: (n_freqs,)
        freqs: Frequency axis — shape: (n_freqs,)
        fmin: Band lower bound (Hz)
        fmax: Band upper bound (Hz)

    Returns:
        Scalar log bandpower.
    """
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    # §Feature extraction, Eq. 8 — ε is a "stabilization constant"
    # [UNSPECIFIED] exact value of ε not stated
    # Using: 1e-10 (small constant for numerical stability)
    eps = 1e-10
    bp = np.trapz(psd[idx], freqs[idx]) + eps
    return float(np.log(bp))


def extract_spectral_features(
    segment: np.ndarray,
    fs: float,
    bands: Dict[str, Tuple[float, float]],
    nperseg: int = 256,
    noverlap: int = 128,
) -> np.ndarray:
    """§Feature extraction, Table 10 — Spectral features via Welch PSD.

    Computes log bandpower (Eq. 8) in delta/theta/alpha/beta/gamma for each channel.

    Args:
        segment: EEG segment — shape: (L, C)
        fs: Sampling frequency (Hz)
        bands: Dict mapping band name -> (fmin, fmax)
        nperseg: Welch window length
            [UNSPECIFIED] not stated; using 256-sample Hann window
            Alternatives: full segment length (Bartlett), 512
        noverlap: Welch overlap
            [UNSPECIFIED] not stated; using 128 (50% overlap)

    Returns:
        Spectral feature vector — shape: (C * n_bands,)
    """
    L, C = segment.shape
    n_bands = len(bands)
    features = np.zeros(C * n_bands)

    for c in range(C):
        freqs, psd = welch(segment[:, c], fs=fs, nperseg=min(nperseg, L), noverlap=noverlap)
        for b_idx, (band_name, (fmin, fmax)) in enumerate(bands.items()):
            features[c * n_bands + b_idx] = bandpower(psd, freqs, fmin, fmax)

    return features


def extract_wavelet_features(
    segment: np.ndarray,
    wavelet: str = "db4",
    level: int = 5,
) -> np.ndarray:
    """§Feature extraction, Eq. 9 — Wavelet features.

    Daubechies-4 (db4) wavelet decomposition to level 5.
    Extracts mean and std of each coefficient level per channel.

        μ_Wk = (1/n) Σ_j Wk[j]
        σ_Wk = sqrt( (1/(n-1)) Σ_j (Wk[j] - μ_Wk)^2 )

    Args:
        segment: EEG segment — shape: (L, C)
        wavelet: Wavelet type — "db4" per §Feature extraction
        level: Decomposition level — 5 per §Feature extraction

    Returns:
        Wavelet feature vector — shape: (C * (level+1) * 2,)
            2 = [mean, std]; (level+1) levels including approximation
    """
    L, C = segment.shape
    features = []

    for c in range(C):
        coeffs = pywt.wavedec(segment[:, c], wavelet=wavelet, level=level)
        # coeffs[0] = approximation, coeffs[1..level] = detail coefficients
        for wk in coeffs:  # §Feature extraction — "from each coefficient set Wk"
            mu = float(np.mean(wk))
            sigma = float(np.std(wk, ddof=1)) if len(wk) > 1 else 0.0
            features.extend([mu, sigma])

    return np.array(features)


def extract_statistical_features(segment: np.ndarray) -> np.ndarray:
    """§Feature extraction, Table 10 and Eq. 10 — Statistical features.

    Per channel: mean, std, skewness, kurtosis.

        γ = (1/n) Σ ((xj - μ)/σ)^3    (skewness)
        κ = (1/n) Σ ((xj - μ)/σ)^4    (kurtosis)

    Args:
        segment: EEG segment — shape: (L, C)

    Returns:
        Statistical feature vector — shape: (C * 4,)
    """
    L, C = segment.shape
    features = []

    for c in range(C):
        x = segment[:, c]
        mu = float(np.mean(x))
        sigma = float(np.std(x, ddof=1)) if L > 1 else 0.0
        # §Feature extraction, Eq. 10
        sk = float(skew(x))
        kurt = float(kurtosis(x))  # scipy: excess (Fisher) kurtosis; [UNSPECIFIED] convention not stated
        features.extend([mu, sigma, sk, kurt])

    return np.array(features)


def permutation_entropy(x: np.ndarray, order: int = 3) -> float:
    """§Feature extraction, Eq. 11 — Permutation entropy.

    H_perm = -Σ_k p_k * log(p_k)

    where p_k is the relative frequency of each ordinal pattern of length m.

    Args:
        x: 1-D signal
        order: Ordinal pattern length m
            [UNSPECIFIED] m not stated in paper
            Using: 3 — minimal pattern length, suitable for short EEG windows
            Alternatives: m=4 (24 patterns), m=5 (120 patterns) — require more data

    Returns:
        Permutation entropy (in nats).
    """
    n = len(x)
    if n < order:
        return 0.0

    patterns = [tuple(np.argsort(x[i : i + order])) for i in range(n - order + 1)]
    counts = Counter(patterns)
    total = sum(counts.values())
    entropy = -sum((c / total) * math.log(c / total) for c in counts.values())
    return entropy


def extract_entropy_features(segment: np.ndarray, order: int = 3) -> np.ndarray:
    """§Feature extraction, Table 10 — Entropy: permutation entropy per channel.

    Args:
        segment: EEG segment — shape: (L, C)
        order: Permutation entropy order m [UNSPECIFIED in paper]

    Returns:
        Entropy feature vector — shape: (C,)
    """
    L, C = segment.shape
    return np.array([permutation_entropy(segment[:, c], order=order) for c in range(C)])


def extract_handcrafted_features(
    segment: np.ndarray,
    fs: float,
    bands: Dict[str, Tuple[float, float]],
    wavelet: str = "db4",
    wavelet_level: int = 5,
    perm_entropy_order: int = 3,
    welch_nperseg: int = 256,
    welch_noverlap: int = 128,
) -> np.ndarray:
    """§Feature extraction — Full handcrafted feature vector f_hand^(i,j) ∈ R^d1.

    Concatenates: spectral + wavelet + statistical + entropy features.
    Expected d1 ≈ 250–300 per Table 12.

    Args:
        segment: Preprocessed, normalized EEG segment S^(i,j) — shape: (L, C)
        fs: Sampling frequency (Hz)
        bands: EEG frequency band definitions (from config)
        wavelet: Wavelet type
        wavelet_level: Wavelet decomposition depth
        perm_entropy_order: Ordinal pattern length m [UNSPECIFIED]
        welch_nperseg: Welch PSD window length [UNSPECIFIED]
        welch_noverlap: Welch PSD overlap [UNSPECIFIED]

    Returns:
        Handcrafted feature vector — shape: (d1,)
    """
    spectral = extract_spectral_features(segment, fs, bands, nperseg=welch_nperseg, noverlap=welch_noverlap)
    wavelet_feats = extract_wavelet_features(segment, wavelet=wavelet, level=wavelet_level)
    statistical = extract_statistical_features(segment)
    entropy = extract_entropy_features(segment, order=perm_entropy_order)

    # §Feature extraction — "concatenated into a unified vector f_hand^(i,j)"
    return np.concatenate([spectral, wavelet_feats, statistical, entropy])


# ---------------------------------------------------------------------------
# Automated Feature Extraction: 1D-CNN
# §Automated feature extraction, Eq. 12
# ---------------------------------------------------------------------------

@dataclass
class CNNConfig:
    """Configuration for the 1D-CNN automated feature extractor.

    Defaults from §Automated feature extraction unless marked [UNSPECIFIED].
    """
    n_channels: int = 18                    # §Data acquisition — "18 electrodes"
    segment_length: int = 1000              # §Preprocessing, Eq. 7 — L = 1000
    # [UNSPECIFIED] Per-layer filter counts not stated; paper gives range 32-128
    # Using: [32, 64, 128] — standard doubling within stated range
    filter_progression: Tuple[int, ...] = field(default_factory=lambda: (32, 64, 128))
    kernel_size: int = 5                    # §Automated feature extraction — "kernel size = 5"
    dropout: float = 0.3                    # §Automated feature extraction — "Dropout rates (p=0.3–0.4)"
    output_dim: int = 256                   # Table 12 — "CNN Features (Global Pooling) d2 = 256"


class ConvBlock1D(nn.Module):
    """§Automated feature extraction, Eq. 12 — 1D convolutional block.

    h^(l) = ReLU( BN( W^(l) * h^(l-1) + b^(l) ) )

    Conv1D → BatchNorm1D → ReLU → MaxPool1D → Dropout
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dropout: float):
        super().__init__()
        # §Automated feature extraction, Eq. 12
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size=kernel_size,
            # [UNSPECIFIED] padding not stated; using 'same' padding to preserve T
            padding=kernel_size // 2,
        )
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        # §Automated feature extraction — "three convolutional layers followed by max pooling"
        # [UNSPECIFIED] MaxPool kernel size not stated; using 2 (standard halving)
        # Alternatives: kernel=3, adaptive pooling
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_ch, T)
        Returns:
            (batch, out_ch, T//2)
        """
        x = self.conv(x)     # (batch, out_ch, T)  — Eq. 12 linear transform
        x = self.bn(x)       # (batch, out_ch, T)  — Eq. 12 BN
        x = self.relu(x)     # (batch, out_ch, T)  — Eq. 12 ReLU
        x = self.pool(x)     # (batch, out_ch, T//2)
        x = self.dropout(x)  # (batch, out_ch, T//2)
        return x


class EEG1DCNN(nn.Module):
    """§Automated feature extraction — Lightweight 1D-CNN for EEG temporal features.

    "It consists of three convolutional layers followed by max pooling and a global
    average pooling layer for dimensionality reduction."

    Architecture: ConvBlock1 → ConvBlock2 → ConvBlock3 → GlobalAvgPool → Linear → d2=256
    """

    def __init__(self, config: CNNConfig):
        super().__init__()
        self.config = config
        filters = config.filter_progression

        # §Automated feature extraction — three convolutional blocks
        self.block1 = ConvBlock1D(config.n_channels, filters[0], config.kernel_size, config.dropout)
        self.block2 = ConvBlock1D(filters[0], filters[1], config.kernel_size, config.dropout)
        self.block3 = ConvBlock1D(filters[1], filters[2], config.kernel_size, config.dropout)

        # §Automated feature extraction — "global average pooling layer"
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Project to d2 = 256 (Table 12 — "CNN Features (Global Pooling) d2 = 256")
        self.proj = nn.Linear(filters[2], config.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: EEG segment — shape: (batch, C, L)

        Returns:
            CNN temporal embedding f_cnn — shape: (batch, d2=256)
        """
        x = self.block1(x)               # (batch, 32, L//2)
        x = self.block2(x)               # (batch, 64, L//4)
        x = self.block3(x)               # (batch, 128, L//8)
        x = self.global_avg_pool(x)      # (batch, 128, 1)
        x = x.squeeze(-1)                # (batch, 128)
        x = self.proj(x)                 # (batch, 256) — Table 12: d2=256
        return x
