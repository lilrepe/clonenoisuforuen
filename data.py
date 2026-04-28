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

def download_openneuro(dataset_id: str, target_dir: str, version: str = "1.0.8") -> None:
    """Download an OpenNeuro dataset using openneuro-py.

    Install: pip install openneuro-py

    Args:
        dataset_id: e.g. "ds004504" or "ds006036"
        target_dir: Local directory to download into
        version: Dataset version string

    Example:
        download_openneuro("ds004504", "data/ds004504", version="1.0.8")
        download_openneuro("ds006036", "data/ds006036", version="1.0.2")
    """
    try:
        import openneuro
    except ImportError:
        raise ImportError("Run: pip install openneuro-py")

    openneuro.download(
        dataset=dataset_id,
        target_dir=target_dir,
        verify_hash=False,  # skip slow hash verification for large EEG files
    )


def load_openneuro_raw(
    dataset_path: str,
    condition: str = "eyes_closed",
) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """Load raw EEG from an OpenNeuro BIDS dataset using MNE-BIDS.

    §Data acquisition:
      ds004504 — eyes-closed, 88 subjects (AD:36, FTD:23, CN:29), 500 Hz, 18 ch
      ds006036 — eyes-open,   same 88 subjects, 500 Hz, 18 ch

    Requires: pip install mne mne-bids

    BIDS layout for these datasets:
      {root}/participants.tsv          ← diagnosis column: "A", "F", "C"
      {root}/sub-{id}/eeg/
            sub-{id}_task-eyesclosed_eeg.set   (EEGLAB .set format)
            sub-{id}_task-eyesclosed_eeg.fdt

    Args:
        dataset_path: Local path to the downloaded BIDS root directory
        condition: For logging only ("eyes_closed" or "eyes_open")

    Returns:
        recordings:  List of EEG arrays, shape (T, C), one per subject
        labels:      Integer class label per subject — 0=AD, 1=FTD, 2=CN
        subject_ids: Subject ID strings (e.g. "sub-001")
    """
    try:
        import mne
        import mne_bids
        import pandas as pd
    except ImportError:
        raise ImportError("Run: pip install mne mne-bids pandas")

    root = Path(dataset_path)

    # ── 1. Read participants.tsv to get diagnosis labels ──────────────────
    participants_file = root / "participants.tsv"
    if not participants_file.exists():
        raise FileNotFoundError(f"participants.tsv not found in {root}")

    participants = pd.read_csv(participants_file, sep="\t")
    # ds004504/ds006036 use column "Group": "A" (AD), "F" (FTD), "C" (Control)
    GROUP_MAP = {"A": 0, "F": 1, "C": 2}
    # Fallback column names seen in OpenNeuro EEG datasets
    group_col = next(
        (c for c in ["Group", "group", "diagnosis", "Diagnosis"] if c in participants.columns),
        None,
    )
    if group_col is None:
        raise ValueError(
            f"Could not find a diagnosis column in participants.tsv. "
            f"Columns found: {list(participants.columns)}"
        )

    # ── 2. Discover EEG files via MNE-BIDS ───────────────────────────────
    layout = mne_bids.BIDSPath(root=str(root))
    bids_paths = mne_bids.find_matching_paths(
        root=str(root),
        datatypes="eeg",
        suffixes="eeg",
        extensions=[".set", ".fif", ".edf", ".bdf"],
    )

    recordings: List[np.ndarray] = []
    labels: List[int] = []
    subject_ids: List[str] = []

    for bids_path in bids_paths:
        subject = bids_path.subject  # e.g. "001"
        sub_id = f"sub-{subject}"

        # Look up diagnosis for this subject
        row = participants[participants["participant_id"] == sub_id]
        if row.empty:
            continue
        group_str = str(row.iloc[0][group_col]).strip().upper()
        if group_str not in GROUP_MAP:
            continue
        label = GROUP_MAP[group_str]

        # Load raw EEG with MNE
        try:
            raw = mne_bids.read_raw_bids(bids_path, verbose=False)
            raw.load_data()
        except Exception as e:
            print(f"  ⚠ Could not load {bids_path}: {e}")
            continue

        # §Data acquisition — "18 electrodes... 500 Hz"
        # Return as (T, C) array matching preprocessing pipeline expectation
        data = raw.get_data()  # (C, T) in MNE convention
        data = data.T           # → (T, C)

        recordings.append(data)
        labels.append(label)
        subject_ids.append(sub_id)

    if not recordings:
        raise RuntimeError(
            f"No EEG files loaded from {dataset_path}. "
            "Check that the dataset was fully downloaded and the BIDS structure is intact."
        )

    print(
        f"Loaded {len(recordings)} subjects from {dataset_path} ({condition}). "
        f"Class distribution: "
        + ", ".join(f"{k}={labels.count(v)}" for k, v in {"AD":0,"FTD":1,"CN":2}.items())
    )
    return recordings, labels, subject_ids


def load_osf_raw(dataset_path: str) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """Load raw EEG from the OSF external validation dataset.

    §Data acquisition (Table 4):
      - 109 subjects: AD (49), MCI (37), HC (23)
      - 128 Hz sampling rate
      - 21 channels, eyes-closed resting state, 8-second segments

    Download: https://osf.io/2v5md/

    The OSF dataset (Rezaee & Zhu, 2025) stores data as .mat files:
      {dataset_path}/
        AD/     ← sub-*.mat  (or AD_*.mat)
        MCI/    ← sub-*.mat
        HC/     ← sub-*.mat
      OR a flat structure with a labels CSV.

    Requires: pip install scipy

    Args:
        dataset_path: Local path to the downloaded OSF dataset root

    Returns:
        recordings:  List of EEG arrays, shape (T, C)
        labels:      0=AD, 1=MCI, 2=HC
        subject_ids: Filename stems used as subject IDs
    """
    import scipy.io as sio

    root = Path(dataset_path)
    CLASS_DIRS = {"AD": 0, "MCI": 1, "HC": 2}

    recordings: List[np.ndarray] = []
    labels: List[int] = []
    subject_ids: List[str] = []

    # Strategy 1: class-named subdirectories (common OSF layout)
    for class_name, label in CLASS_DIRS.items():
        class_dir = root / class_name
        if not class_dir.exists():
            continue
        for mat_file in sorted(class_dir.glob("*.mat")):
            mat = sio.loadmat(str(mat_file))
            # OSF dataset stores EEG in variable named 'data' or 'EEG'
            eeg_key = next(
                (k for k in mat.keys() if k in ("data", "EEG", "eeg", "x", "X")),
                None,
            )
            if eeg_key is None:
                # Fallback: use the first non-metadata key
                eeg_key = next(k for k in mat.keys() if not k.startswith("_"))
            arr = np.array(mat[eeg_key], dtype=np.float64)
            # Ensure shape is (T, C)
            if arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
                arr = arr.T  # (C, T) → (T, C)
            recordings.append(arr)
            labels.append(label)
            subject_ids.append(mat_file.stem)

    if not recordings:
        # Strategy 2: flat directory with a CSV index file
        index_file = next(root.glob("*.csv"), None)
        if index_file is None:
            raise FileNotFoundError(
                f"Could not find class subdirectories (AD/MCI/HC) or a CSV index "
                f"in {root}. Check the OSF download structure at https://osf.io/2v5md/"
            )
        import pandas as pd
        index = pd.read_csv(index_file)
        for _, row in index.iterrows():
            mat_path = root / row["filename"]
            label = CLASS_DIRS.get(str(row["label"]).upper(), -1)
            if label == -1 or not mat_path.exists():
                continue
            mat = sio.loadmat(str(mat_path))
            eeg_key = next(k for k in mat.keys() if not k.startswith("_"))
            arr = np.array(mat[eeg_key], dtype=np.float64)
            if arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
                arr = arr.T
            recordings.append(arr)
            labels.append(label)
            subject_ids.append(mat_path.stem)

    print(
        f"Loaded {len(recordings)} subjects from OSF dataset. "
        f"Class distribution: "
        + ", ".join(f"{k}={labels.count(v)}" for k, v in CLASS_DIRS.items())
    )
    return recordings, labels, subject_ids


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