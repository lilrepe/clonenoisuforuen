# NeuroFusionNet

Minimal PyTorch implementation of **NeuroFusionNet**, a hybrid EEG feature fusion framework for Alzheimer's Disease detection.

> Akbar et al., "NeuroFusionNet: a hybrid EEG feature fusion framework for accurate and explainable Alzheimer's Disease detection," *Scientific Reports* (2025) 15:43742. DOI: https://doi.org/10.1038/s41598-025-28070-x

---

## What this implementation covers

- **Handcrafted feature extraction** (`src/features.py`): spectral (Welch PSD), wavelet (db4 level-5), statistical (mean/std/skew/kurtosis), permutation entropy — per §Feature extraction, Table 10
- **1D-CNN automated features** (`src/features.py`): 3-block 1D-CNN → Global Average Pooling → 256-dim embedding — per §Automated feature extraction, Eq. 12
- **Feature fusion pipeline** (`src/fusion.py`): concatenation → PCC selection → PSO stub → SMOTE → PCA — per §Key contributions, Eq. 13–16
- **NeuroFusionNet classifier** (`src/model.py`): 5-layer DNN with skip connection — per Table 15, Fig. 7, Eq. 17–19
- **Training loop** (`src/train.py`): Adam + ReduceLROnPlateau + early stopping + class weights — per Table 14
- **Evaluation** (`src/evaluate.py`): accuracy, macro F1, AUC — per Table 18/20

**Not included:** SHAP/Grad-CAM (standard library calls), baseline models, distributed training, preprocessing ASR/ICA (use MNE), Docker deployment.

---

## Quick start

```bash
pip install -r requirements.txt

# (Optional) For PSO feature selection:
# pip install pyswarms

# (Optional) For EEG data loading:
# pip install mne
```

Download datasets (manual — see REPRODUCTION_NOTES.md for links):
```
data/
  ds004504/    ← OpenNeuro eyes-closed
  ds006036/    ← OpenNeuro eyes-open
  osf_eeg/     ← OSF external validation
```

Run the walkthrough notebook:
```bash
jupyter notebook notebooks/walkthrough.ipynb
```

---

## File structure

```
neurofusionnet/
├── configs/
│   └── base.yaml           All hyperparameters, cited to paper sections
├── src/
│   ├── features.py         Handcrafted + 1D-CNN feature extraction
│   ├── fusion.py           PCC → PSO → SMOTE → PCA fusion pipeline
│   ├── model.py            NeuroFusionNet 5-layer DNN classifier
│   ├── loss.py             Cross-entropy + L2 (Eq. 20)
│   ├── data.py             Dataset skeleton + subject-level splits
│   ├── train.py            Training loop (Table 14)
│   └── evaluate.py         Metrics (Table 18/20)
├── notebooks/
│   └── walkthrough.ipynb   Paper → code walkthrough with sanity checks
├── REPRODUCTION_NOTES.md   Full ambiguity audit + unspecified choices
├── requirements.txt
└── README.md
```

---

## Known ambiguities

See `REPRODUCTION_NOTES.md` for the complete list. The most impactful:

1. **PSO hyperparameters** — swarm size, inertia, c1/c2 not stated. The PSO call raises `NotImplementedError` until you configure it.
2. **PCC threshold** — not stated; using 0.95
3. **Welch PSD window** — not stated; using 256-sample, 50% overlap
4. **Permutation entropy order** m — not stated; using m=3

All unspecified choices are marked `[UNSPECIFIED]` in code.

---

## Citation

```bibtex
@article{akbar2025neurofusionnet,
  title={NeuroFusionNet: a hybrid EEG feature fusion framework for accurate
         and explainable Alzheimer's Disease detection},
  author={Akbar, Frnaz and Alkhrijah, Yazeed and Usman, Syed Muhammad and
          Khalid, Shehzad and Ihsan, Imran and Alawad, Mohamad A.},
  journal={Scientific Reports},
  volume={15},
  pages={43742},
  year={2025},
  doi={10.1038/s41598-025-28070-x}
}
```
