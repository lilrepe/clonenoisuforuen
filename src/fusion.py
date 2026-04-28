"""
NeuroFusionNet — Feature Fusion Pipeline

Paper: "NeuroFusionNet: a hybrid EEG feature fusion framework for accurate
        and explainable Alzheimer's Disease detection"
Scientific Reports (2025) 15:43742
DOI: https://doi.org/10.1038/s41598-025-28070-x

Implements:
  §Key contributions / §Feature extraction — Feature fusion pipeline:
    1. Concatenate handcrafted + CNN features → f^(i,j) = [f_hand || f_cnn], Eq. 13
    2. PCC-based redundancy removal
    3. PSO-based feature selection (wrapper skeleton — PSO details UNSPECIFIED)
    4. SMOTE class balancing, Eq. 14
    5. PCA dimensionality reduction (99% variance), Eq. 15-16
    → Final representation F_final ∈ R^(M × dfinal), Eq. 16

Section references:
  §Feature extraction, Eq. 13 — concatenation
  §Feature extraction, Eq. 14 — SMOTE interpolation
  §Feature extraction, Eq. 15-16 — PCA
  §Key contributions — "Pearson Correlation Coefficient (PCC) and Particle Swarm Optimization (PSO)"
  Table 12 — feature dimensionality at each stage
"""

from typing import List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    raise ImportError("Install imbalanced-learn: pip install imbalanced-learn")


# ---------------------------------------------------------------------------
# Step 1: Concatenation — Eq. 13
# ---------------------------------------------------------------------------

def fuse_features(f_hand: np.ndarray, f_cnn: np.ndarray) -> np.ndarray:
    """§Feature extraction, Eq. 13 — Concatenate handcrafted and CNN features.

    f^(i,j) = [f_hand^(i,j) || f_cnn^(i,j)] ∈ R^(d1 + d2)

    Table 12: d1 ≈ 250-300, d2 = 256 → concatenated ≈ 550

    Args:
        f_hand: Handcrafted feature vector — shape: (d1,)
        f_cnn: CNN feature embedding — shape: (d2=256,)

    Returns:
        Fused feature vector — shape: (d1 + d2,)
    """
    return np.concatenate([f_hand, f_cnn], axis=-1)


# ---------------------------------------------------------------------------
# Step 2: PCC-based redundancy removal
# §Key contributions — "Pearson Correlation Coefficient (PCC)... to remove features
#                        with high correlation"
# ---------------------------------------------------------------------------

def pcc_feature_selection(X: np.ndarray, threshold: float = 0.95) -> Tuple[np.ndarray, List[int]]:
    """§Key contributions — PCC-based removal of highly correlated features.

    "Handcrafted features are first selected through Pearson-correlation analysis
    and PSO-based feature selection to remove features with high correlation."

    Removes one feature from each pair with |PCC| > threshold.

    Args:
        X: Feature matrix — shape: (n_samples, n_features)
        threshold: Correlation threshold above which one feature is dropped
            [UNSPECIFIED] exact threshold not stated in paper
            Using: 0.95 (common in EEG literature)
            Alternatives: 0.90, 0.85

    Returns:
        X_reduced: Reduced feature matrix — shape: (n_samples, n_selected)
        selected_indices: List of retained feature indices
    """
    n_features = X.shape[1]
    corr_matrix = np.corrcoef(X.T)  # (n_features, n_features)

    # Greedily remove one feature from each highly-correlated pair
    keep = list(range(n_features))
    to_remove = set()

    for i in range(n_features):
        if i in to_remove:
            continue
        for j in range(i + 1, n_features):
            if j in to_remove:
                continue
            if abs(corr_matrix[i, j]) > threshold:
                to_remove.add(j)  # Remove the later feature in each correlated pair

    selected_indices = [i for i in keep if i not in to_remove]
    return X[:, selected_indices], selected_indices


# ---------------------------------------------------------------------------
# Step 3: PSO-based feature selection
# §Key contributions — "Particle Swarm Optimization (PSO)"
# ---------------------------------------------------------------------------

def pso_feature_selection_placeholder(
    X: np.ndarray,
    y: np.ndarray,
    n_particles: int = 30,
    n_iterations: int = 100,
) -> Tuple[np.ndarray, List[int]]:
    """§Key contributions — PSO-based feature selection (STUB).

    "Employed Pearson correlation analysis, PSO-based selection, and bottleneck
    fusion to maximize complementarity and minimize redundancy."

    [UNSPECIFIED] PSO hyperparameters (swarm size, inertia, c1, c2, velocity
    bounds, fitness function definition) are NOT stated in the paper.

    This function is a stub. To reproduce the paper's results, you must implement
    or use a PSO library (e.g., pyswarms) with an appropriate fitness function
    (e.g., cross-validation accuracy of a fast classifier on the selected subset).

    Recommended PSO fitness function: 5-fold CV accuracy of SVM on selected features.
    Recommended library: pyswarms (pip install pyswarms)

    Args:
        X: Feature matrix post-PCC — shape: (n_samples, n_features)
        y: Class labels — shape: (n_samples,)
        n_particles: PSO swarm size [UNSPECIFIED; using 30]
        n_iterations: PSO iterations [UNSPECIFIED; using 100]

    Returns:
        X_selected: Feature matrix with PSO-selected features
        selected_indices: Indices of selected features

    Raises:
        NotImplementedError: Always — this is a stub.
    """
    raise NotImplementedError(
        "PSO feature selection is a stub. "
        "PSO hyperparameters (swarm size, inertia weight, c1, c2, velocity bounds, "
        "fitness function) are not specified in the paper. "
        "See REPRODUCTION_NOTES.md §Ambiguities for guidance. "
        "To proceed without PSO, call pcc_feature_selection() only and set "
        "pso_enabled=False in your pipeline config."
    )


# ---------------------------------------------------------------------------
# Step 4: SMOTE class balancing — Eq. 14
# ---------------------------------------------------------------------------

def apply_smote(
    X: np.ndarray,
    y: np.ndarray,
    k_neighbors: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """§Data preparation, Eq. 14 — SMOTE oversampling for class balance.

    x_syn = x_a + λ(x_b - x_a),   λ ~ U(0, 1)

    where x_a, x_b are minority class samples and λ is a random interpolation
    coefficient.

    "SMOTE was applied strictly within training folds during cross-validation;
    validation and test folds remained unchanged."
    ↑ CRITICAL: Call this function only on training data within each CV fold.

    Table 7: Before SMOTE — AD:720, FTD:480, CN:1450
             After SMOTE  — AD:1450, FTD:1450, CN:1450

    Args:
        X: Training feature matrix — shape: (n_train, n_features)
        y: Training labels — shape: (n_train,)
        k_neighbors: Number of nearest neighbors for interpolation
            [UNSPECIFIED] not stated; using imbalanced-learn default (5)
        random_state: Random seed for reproducibility [UNSPECIFIED]

    Returns:
        X_resampled: Oversampled feature matrix
        y_resampled: Oversampled labels
    """
    smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled


# ---------------------------------------------------------------------------
# Step 5: PCA dimensionality reduction — Eq. 15-16
# ---------------------------------------------------------------------------

class PCAReducer:
    """§Feature extraction, Eq. 15-16 — PCA compression to dfinal.

    Z = XW,   W = argmax_W Var(XW)   [Eq. 15]

    F_final = {f_pca^(i,j)}_{i,j} ∈ R^(M × dfinal)   [Eq. 16]

    "PCA retaining 99% of variance" → dfinal ≈ 60-80 (Table 12)
    """

    def __init__(self, variance_retained: float = 0.99):
        """
        Args:
            variance_retained: Fraction of variance to retain.
                §Feature extraction — "PCA retaining 99% of variance"
        """
        self.variance_retained = variance_retained
        self.pca: Optional[PCA] = None
        self.scaler: Optional[StandardScaler] = None

    def fit(self, X: np.ndarray) -> "PCAReducer":
        """Fit StandardScaler + PCA on training data.

        §Feature extraction — "All features were standardized via z-score normalization"
        followed by PCA.

        Args:
            X: Training feature matrix — shape: (n_train, n_features)

        Returns:
            self (fitted)
        """
        # §Feature extraction — z-score normalization before PCA
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # §Feature extraction, Eq. 15-16 — PCA
        self.pca = PCA(n_components=self.variance_retained)
        self.pca.fit(X_scaled)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted scaler + PCA to new data.

        Args:
            X: Feature matrix — shape: (n_samples, n_features)

        Returns:
            PCA-reduced feature matrix — shape: (n_samples, dfinal)
        """
        assert self.pca is not None and self.scaler is not None, "Call fit() first."
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one call (use only on training data)."""
        self.fit(X)
        return self.transform(X)

    @property
    def n_components_(self) -> int:
        """Number of PCA components retained (dfinal)."""
        assert self.pca is not None, "Call fit() first."
        return self.pca.n_components_
