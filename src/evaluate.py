"""
NeuroFusionNet — Evaluation

Paper: "NeuroFusionNet: a hybrid EEG feature fusion framework for accurate
        and explainable Alzheimer's Disease detection"
Scientific Reports (2025) 15:43742
DOI: https://doi.org/10.1038/s41598-025-28070-x

Implements:
  §Results — Metrics matching Table 18 and Table 20:
    Accuracy, Macro Precision/Recall/F1, AUC
"""

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """§Results — Compute classification metrics matching Table 18 and Table 20.

    Table 18 reports: Precision, Recall, F1-Measure per class + Macro and Weighted averages.
    Table 20 reports: Accuracy (%), Macro F1, AUC per fold.

    Args:
        y_true: Ground-truth class indices — shape: (n_samples,)
        y_pred: Predicted class indices — shape: (n_samples,)
        y_prob: Predicted class probabilities — shape: (n_samples, n_classes)
                Required for AUC computation.
        class_names: Class name list (e.g., ["AD", "FTD", "CN"])
                     Default: ["AD", "FTD", "CN"] for ds004504

    Returns:
        Dict with keys: accuracy, macro_precision, macro_recall, macro_f1,
                        weighted_precision, weighted_recall, weighted_f1,
                        auc (if y_prob provided)
    """
    if class_names is None:
        class_names = ["AD", "FTD", "CN"]  # §Data acquisition ds004504

    results = {
        # Table 18 / Table 20 — "Accuracy"
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # Table 18 — "Macro-Averaged"
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        # Table 18 — "Weighted-Averaged"
        "weighted_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    # Table 20 — AUC (requires probability scores)
    if y_prob is not None:
        # Table 20 reports macro-averaged AUC = 0.94 for ds004504
        # Fig. 9 shows per-class AUC values (HC:0.98, FTD:0.98, AD:0.99)
        try:
            results["auc"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            )
        except ValueError:
            results["auc"] = float("nan")

    # Per-class metrics (Table 18)
    per_class_report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    for cls_name in class_names:
        if cls_name in per_class_report:
            results[f"{cls_name}_precision"] = float(per_class_report[cls_name]["precision"])
            results[f"{cls_name}_recall"] = float(per_class_report[cls_name]["recall"])
            results[f"{cls_name}_f1"] = float(per_class_report[cls_name]["f1-score"])

    return results


def aggregate_cv_results(fold_results: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """§Results, Table 20 — Aggregate 5-fold CV metrics (mean ± SD).

    Table 20: Mean ± SD across folds:
      Accuracy: 94.27 ± 0.28
      Macro F1: 0.94 ± 0.01
      AUC:      0.94 ± 0.01

    Args:
        fold_results: List of per-fold metric dicts

    Returns:
        Dict mapping metric_name → {"mean": float, "std": float}
    """
    all_keys = fold_results[0].keys()
    aggregated = {}
    for key in all_keys:
        values = [r[key] for r in fold_results if key in r]
        aggregated[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
    return aggregated
