"""
NeuroFusionNet — Data Loading Skeleton

Paper: "NeuroFusionNet: a hybrid EEG feature fusion framework for accurate
        and explainable Alzheimer's Disease detection"
Scientific Reports (2025) 15:43742
DOI: https://doi.org/10.1038/s41598-025-28070-x

Provides dataset skeletons for the three EEG datasets used in the paper.
Users must download the data separately (see README.md for links).

Data sources:
  ds004504: https://openneuro.org/datasets/ds004504/versions/1.0.8
  ds006036: https://openneuro.org/datasets/ds006036/versions/1.0.2
  OSF:      https://osf.io/2v5md/

Section references:
  §Data acquisition — dataset descriptions and recording parameters
  §Preprocessing, Eq. 4-7 — z-score normalization and segmentation
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# DO NOT auto-download datasets. Provide paths and download instructions only.


class EEGDataset(Dataset):
    """Base EEG dataset class.

    Provides pre-extracted feature vectors and labels.
    Feature extraction (handcrafted + CNN) must be run separately via
    the feature extraction pipeline (src/features.py + src/fusion.py).

    This class operates on the final PCA-reduced feature matrix F_final.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ):
        """
        Args:
            features: PCA-reduced feature matrix — shape: (n_segments, dfinal)
            labels: Class indices — shape: (n_segments,)
                    Class mapping: 0=AD, 1=FTD, 2=CN (or 0=AD, 1=HC for OSF)
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# OpenNeuro ds004504 / ds006036 — raw EEG loading skeleton
# §Data acquisition — "88 subjects (AD:36, FTD:23, CN:29), 500 Hz, 18 channels"
# ---------------------------------------------------------------------------

def load_openneuro_raw(
    dataset_path: str,
    condition: str = "eyes_closed",  # "eyes_closed" (ds004504) or "eyes_open" (ds006036)
) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """Load raw EEG from OpenNeuro BIDS-format datasets.

    §Data acquisition:
      - ds004504: eyes-closed, 88 subjects (AD:36, FTD:23, CN:29)
      - ds006036: eyes-open, same 88 subjects
      - 500 Hz sampling rate
      - 18 electrodes (10-20 system), A1-A2 references
      - Format: BIDS .tsv / .csv / .h5 (check dataset readme)

    Download:
      ds004504: https://openneuro.org/datasets/ds004504/versions/1.0.8
      ds006036: https://openneuro.org/datasets/ds006036/versions/1.0.2

    Args:
        dataset_path: Local path to downloaded BIDS dataset root
        condition: Dataset variant identifier (for logging)

    Returns:
        recordings: List of EEG arrays, one per subject — each shape: (T, C)
        labels: Integer class label per subject (0=AD, 1=FTD, 2=CN)
        subject_ids: Subject identifiers (for stratified split)

    Raises:
        NotImplementedError: Implement based on your local BIDS file structure.
    """
    # TODO: Implement based on local file layout.
    # Typical BIDS structure:
    #   {dataset_path}/sub-{id}/eeg/sub-{id}_task-eyesclosed_eeg.set  (or .fif, .tsv)
    #
    # Recommended library: MNE-Python
    #   pip install mne
    #   raw = mne.io.read_raw_eeglab(path, preload=True)
    #   data = raw.get_data().T  # (T, C)
    #
    # Label assignment:
    #   Check participants.tsv for diagnosis column:
    #   "A" or "Alzheimer" → 0 (AD)
    #   "F" or "FTD"       → 1 (FTD)
    #   "C" or "Control"   → 2 (CN)
    raise NotImplementedError(
        "Implement load_openneuro_raw() based on your local file layout. "
        "See docstring for guidance. Recommended: MNE-Python."
    )


def load_osf_raw(dataset_path: str) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """Load raw EEG from the OSF external validation dataset.

    §Data acquisition (Table 4):
      - 109 subjects: AD (49), MCI (37), HC (23)
      - 128 Hz sampling rate
      - 21 channels
      - Eyes-closed resting state
      - 8-second segments

    Download: https://osf.io/2v5md/

    Args:
        dataset_path: Local path to downloaded OSF dataset

    Returns:
        recordings: List of EEG arrays — each shape: (T, C)
        labels: Integer class label (0=AD, 1=MCI, 2=HC)
        subject_ids: Subject identifiers

    Raises:
        NotImplementedError: Implement based on local file structure.
    """
    # TODO: Implement. OSF datasets typically use .mat or .csv format.
    # Check the OSF repository page for file structure documentation.
    raise NotImplementedError(
        "Implement load_osf_raw() based on the OSF dataset file layout. "
        "Typical format: .mat (scipy.io.loadmat) or .csv (numpy.loadtxt). "
        "Download from: https://osf.io/2v5md/"
    )


# ---------------------------------------------------------------------------
# Subject-level stratified split
# §Classification, Table 16 — 70/15/15 split at subject level
# ---------------------------------------------------------------------------

def stratified_subject_split(
    subject_ids: List[str],
    labels: List[int],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """§Classification, Table 16 — Subject-level stratified train/val/test split.

    "The OpenNeuro (ds004504) dataset was partitioned at the subject level.
    A stratified assignment maintained class balance across five folds, with
    each fold comprising 70% training, 15% validation, and 15% testing subjects.
    This ensures that no EEG segment from a single participant appears in more
    than one subset and prevents temporal or spatial leakage."

    Args:
        subject_ids: Subject identifier strings
        labels: Per-subject class labels
        train_ratio: Training fraction (paper: 0.70)
        val_ratio: Validation fraction (paper: 0.15)
        random_state: Random seed [UNSPECIFIED in paper]

    Returns:
        train_subjects, val_subjects, test_subjects: Subject ID lists
    """
    from sklearn.model_selection import train_test_split

    # First split: train vs (val + test)
    test_ratio = 1.0 - train_ratio
    train_ids, remaining_ids, train_labels, remaining_labels = train_test_split(
        subject_ids, labels,
        test_size=test_ratio,
        stratify=labels,
        random_state=random_state,
    )

    # Second split: val vs test (from remaining)
    # val_ratio relative to total → relative to remaining
    relative_val = val_ratio / test_ratio
    val_ids, test_ids = train_test_split(
        remaining_ids,
        test_size=1.0 - relative_val,
        stratify=remaining_labels,
        random_state=random_state,
    )

    return train_ids, val_ids, test_ids
