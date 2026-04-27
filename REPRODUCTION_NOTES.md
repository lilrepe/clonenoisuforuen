# Reproduction Notes — NeuroFusionNet

**Paper:** "NeuroFusionNet: a hybrid EEG feature fusion framework for accurate and explainable Alzheimer's Disease detection"  
**Published:** Scientific Reports (2025) 15:43742  
**DOI:** https://doi.org/10.1038/s41598-025-28070-x  
**Authors:** Frnaz Akbar, Yazeed Alkhrijah, Syed Muhammad Usman, Shehzad Khalid, Imran Ihsan, Mohamad A. Alawad

---

## Scope decisions

### Implemented
- **Handcrafted feature extraction** (`src/features.py`) — core contribution; fully specified via Table 10, Eq. 8–11
- **1D-CNN automated feature extractor** (`src/features.py`) — core contribution; architecture described in §Automated feature extraction, Eq. 12
- **Feature fusion pipeline** (`src/fusion.py`) — concatenation (Eq. 13), PCC selection, PSO stub, SMOTE (Eq. 14), PCA (Eq. 15–16)
- **NeuroFusionNet classifier** (`src/model.py`) — 5-layer DNN with skip connection; fully specified in Table 15, Fig. 7, Eq. 17–19
- **Loss function** (`src/loss.py`) — cross-entropy + L2 (Eq. 20); L2 handled by optimizer `weight_decay`
- **Training loop** (`src/train.py`) — Adam + ReduceLROnPlateau; hyperparameters from Table 14
- **Evaluation metrics** (`src/evaluate.py`) — accuracy, macro precision/recall/F1, AUC matching Table 18 and Table 20

### Intentionally excluded
- **SHAP explainability** — paper uses SHAP Kernel Explainer post-hoc; standard library call (`shap.KernelExplainer`), not a novel contribution
- **Grad-CAM visualization** — standard technique applied to the CNN sub-module; not a novel contribution
- **Baseline models** (SVM, RF, KNN, XGBoost, 1D-CNN baseline) — comparison methods from Table 31/32, not the paper's contribution
- **t-SNE visualization** — post-hoc analysis, not part of the model
- **Docker containerization** — deployment infrastructure, not the model
- **EEG preprocessing pipeline** (bandpass filter, ASR, ICA) — uses standard tools (MNE, EEGLAB); fully documented in `src/data.py` but not re-implemented from scratch

### Would need for full reproduction
- **Raw EEG data** — must be downloaded from OpenNeuro and OSF (see §Data sources below)
- **EEGLAB/MNE** — for ASR and ICA preprocessing (standard tools, not re-implemented)
- **PSO implementation** — requires a PSO library with manually chosen hyperparameters (see §Critical ambiguities below)

---

## Critical ambiguities

These are details where the paper's specification is insufficient for exact reproduction.
Each is flagged in code with `[UNSPECIFIED]` or `[PARTIALLY_SPECIFIED]` comments.

### 1. PSO Feature Selection — UNSPECIFIED
**Paper says:** "Particle Swarm Optimization (PSO)-based feature selection"  
**Missing:** Swarm size, number of iterations, inertia weight, cognitive coefficient (c1), social coefficient (c2), velocity bounds, fitness function definition, initialization strategy.  
**Impact:** HIGH — PSO is a stochastic optimizer; different hyperparameters produce different feature subsets, affecting accuracy.  
**Recommendation:** Use `pyswarms` library. A reasonable starting point:
- `n_particles=30`, `n_iterations=100`
- Fitness: 5-fold CV accuracy of LinearSVC on selected feature subset
- Standard PSO parameters: inertia=0.729, c1=1.494, c2=1.494

The PSO stub in `src/fusion.py` raises `NotImplementedError` to force a conscious decision.  
To skip PSO and use PCC-only selection, set `pso_enabled=False` in your pipeline.

### 2. PCC Threshold — UNSPECIFIED
**Paper says:** "Pearson-correlation analysis... to remove features with high correlation"  
**Missing:** The correlation threshold value.  
**Impact:** MEDIUM — affects how many features are removed before PSO.  
**Our choice:** 0.95 (common in EEG literature). Alternatives: 0.90, 0.85.

### 3. Welch PSD Parameters — UNSPECIFIED
**Paper says:** "Power spectral density (PSD), P(f), was estimated using Welch's method"  
**Missing:** Window type, window length, overlap fraction.  
**Impact:** MEDIUM — affects spectral feature values directly.  
**Our choice:** 256-sample Hann window, 50% overlap (scipy defaults).

### 4. Permutation Entropy Pattern Length — UNSPECIFIED
**Paper says:** "permutation entropy quantified temporal complexity: H_perm = -Σ p_k log p_k, where p_k is the probability of each ordinal pattern of length m"  
**Missing:** The value of m.  
**Impact:** LOW-MEDIUM — m=3 vs m=5 produces different entropy values.  
**Our choice:** m=3 (smallest meaningful value; suitable for 1000-sample windows).

### 5. 1D-CNN Per-Layer Filter Counts — UNSPECIFIED
**Paper says:** filters in range 32–128; optimal is 64 filters total  
**Missing:** Which layer gets which number of filters (e.g., is 64 the count for all 3 layers, or just the optimal overall?)  
**Impact:** LOW — affects parameter count but not architecture structure.  
**Our choice:** 32→64→128 (standard doubling progression within stated range 32–128).

### 6. Skip Connection Projection — PARTIALLY SPECIFIED
**Paper says:** "skip connection between the first and third dense layers"  
**Detail:** Dense-1 has 512 units; Dense-3 has 256 units. A direct skip requires a projection.  
**Impact:** LOW — choice of projection (linear vs zero-padding vs truncation) has minor effect.  
**Our choice:** Linear projection 512→256 (`self.skip_proj` in `src/model.py`).

### 7. Adam Hyperparameters — UNSPECIFIED
**Paper says:** "Adam optimizer with an initial learning rate of 3×10⁻⁴"  
**Missing:** β₁, β₂, ε.  
**Our choice:** PyTorch defaults (β₁=0.9, β₂=0.999, ε=1e-8).

### 8. Weight Initialization — UNSPECIFIED
**Paper says:** Nothing about weight initialization.  
**Our choice:** PyTorch defaults (Kaiming uniform for Linear layers via `nn.Linear`).

### 9. MaxPool Kernel Size in CNN — UNSPECIFIED
**Paper says:** "three convolutional layers followed by max pooling"  
**Missing:** MaxPool kernel size and stride.  
**Our choice:** kernel=2, stride=2 (standard halving). Alternatives: kernel=3.

### 10. CNN Padding — UNSPECIFIED
**Paper says:** Conv layers with kernel sizes 3, 5, 7  
**Missing:** Whether same-padding or valid-padding is used.  
**Our choice:** Same-padding (kernel//2) to preserve temporal length before pooling.

### 11. BatchNorm Epsilon — UNSPECIFIED
**Paper says:** "batch normalization (BN)"  
**Missing:** epsilon parameter.  
**Our choice:** PyTorch default (1e-5).

### 12. SMOTE k_neighbors — UNSPECIFIED
**Paper says:** "SMOTE... generates synthetic feature vectors via linear interpolation between nearest neighbors"  
**Missing:** k (number of neighbors).  
**Our choice:** k=5 (imbalanced-learn default).

### 13. dfinal Exact Value — PARTIALLY SPECIFIED
**Paper says:** "dfinal ≈ 60–80" (Table 12)  
**Detail:** The exact value depends on the PCA result on your data. The `ClassifierConfig.input_dim` must be set to match after running PCA.  
**Impact:** MEDIUM — wrong input_dim causes a shape error at runtime (easily caught and fixed).

---

## What the paper clearly specifies (SPECIFIED items)

These are faithfully implemented:

| Item | Value | Source |
|------|-------|--------|
| Butterworth filter | 4th-order, 0.5–45 Hz | §Preprocessing, Eq. 2 |
| ICA algorithm | Infomax | §Preprocessing |
| ICA kurtosis threshold | >5 | §Preprocessing |
| ICA skewness threshold | >2 | §Preprocessing |
| ICA variance threshold | ±3 SD | §Preprocessing |
| Segment length | 1000 samples | §Preprocessing, Eq. 7 |
| Segment overlap | Non-overlapping | §Preprocessing, Eq. 7 |
| Wavelet type | db4 | §Feature extraction |
| Wavelet levels | 5 | §Feature extraction |
| Wavelet features | Mean and std per level | §Feature extraction, Eq. 9 |
| Statistical features | Mean, std, skewness, kurtosis | §Feature extraction, Eq. 10, Table 10 |
| CNN filters (optimal) | 64 | §Automated feature extraction |
| CNN kernel size (optimal) | 5 | §Automated feature extraction |
| CNN conv blocks | 3 | §Automated feature extraction |
| CNN output dim | d2 = 256 | Table 12 |
| PCA variance retained | 99% | §Feature extraction, Eq. 15–16 |
| Dense-1 | 512 units, LeakyReLU, dropout p=0.4 | Table 15 |
| Dense-2 | 256 units, ReLU (skip source) | Table 15 |
| Dense-3 | 256 units + skip, dropout p=0.4 | Table 15 |
| Dense-4 | 128 units, BN, dropout p=0.3 | Table 15 |
| Dense-5 | 64 units, BN, dropout p=0.3 | Table 15 |
| Output | 3 classes, Softmax | Table 15 |
| Optimizer | Adam, lr=3e-4 | Table 14 |
| L2 regularization | λ=0.001 | Table 14 |
| Dropout rates | p1=p2=0.4, p3=p4=0.3 | Table 14 |
| Batch size | 32 | Table 14 / Table 19 ablation |
| Max epochs | 500 | Table 14 |
| Early stopping patience | 10 | Table 14 |
| LR scheduler | ReduceLROnPlateau factor=0.5, patience=10 | Table 14 |
| Class weights | Inverse frequency | Table 14 |
| Cross-validation | Stratified 5-fold | §Results |
| Data split | 70/15/15 subject-level | Table 16 |
| SMOTE | Applied within training folds only | §Data preparation |
| Activation (DNN) | LeakyReLU (layers 1,3,4,5), ReLU (layer 2) | Eq. 18, Table 15 |

---

## Data sources

| Dataset | URL | Description |
|---------|-----|-------------|
| OpenNeuro ds004504 | https://openneuro.org/datasets/ds004504/versions/1.0.8 | Primary: eyes-closed, 88 subjects (AD/FTD/CN), 500 Hz |
| OpenNeuro ds006036 | https://openneuro.org/datasets/ds006036/versions/1.0.2 | Eyes-open, same 88 subjects, 500 Hz |
| OSF EEG | https://osf.io/2v5md/ | External validation: AD/MCI/HC, 128 Hz, 109 subjects |

---

## Expected results (from paper)

| Dataset | Accuracy | Macro F1 | AUC |
|---------|----------|----------|-----|
| ds004504 (eyes-closed) | 94.27% ± 0.28% | 0.94 ± 0.01 | 0.94 ± 0.01 |
| ds006036 (eyes-open) | 92.15% | 0.92 | 0.93 |
| OSF (external) | 89.5% | 0.893 | — |
| OSF (zero-shot from ds004504) | 88.4% | 0.88 | — |

**Note:** PSO feature selection is stochastic. Even with a fixed random seed, results may differ from the paper if PSO hyperparameters are set differently. Expect ±1–2% accuracy variation from this source alone.
