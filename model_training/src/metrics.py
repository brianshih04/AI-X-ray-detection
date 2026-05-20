"""
Metrics Module: AUC-ROC, F1 (Macro/Micro), PR-AUC per class, threshold tuning.

All metrics follow CheXpert evaluation conventions (Irvin et al., 2019).
"""

import logging
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    precision_recall_curve,
)

logger = logging.getLogger(__name__)


def compute_auroc_per_class(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
) -> Dict[str, float]:
    """Compute AUC-ROC for each class independently (OvR)."""
    results = {}
    for i, name in enumerate(class_names):
        y_t = y_true[:, i]
        y_p = y_proba[:, i]
        if y_t.sum() == 0 or y_t.sum() == len(y_t):
            results[name] = float("nan")
        else:
            try:
                results[name] = roc_auc_score(y_t, y_p)
            except ValueError:
                results[name] = float("nan")
    return results


def compute_pr_auc_per_class(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
) -> Dict[str, float]:
    """Compute PR-AUC (Average Precision) for each class."""
    results = {}
    for i, name in enumerate(class_names):
        try:
            results[name] = average_precision_score(y_true[:, i], y_proba[:, i])
        except ValueError:
            results[name] = float("nan")
    return results


def compute_f1_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute Macro, Micro, Weighted, and Sample-based F1."""
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "sample_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
    }


def compute_precision_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute Macro/Micro precision and recall."""
    return {
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "micro_precision": float(
            precision_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "micro_recall": float(
            recall_score(y_true, y_pred, average="micro", zero_division=0)
        ),
    }


def find_optimal_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> np.ndarray:
    """Find F1-optimal threshold for each class.

    For each class, sweep thresholds and pick the one that maximizes F1.
    """
    n_classes = y_true.shape[1]
    optimal_thresholds = np.full(n_classes, 0.5)

    for i in range(n_classes):
        prec, rec, thresholds = precision_recall_curve(y_true[:, i], y_proba[:, i])
        f1_scores = 2 * prec * rec / (prec + rec + 1e-8)
        # Last entry is for threshold=0 with recall=1, precision=0
        best_idx = np.argmax(f1_scores[:-1])
        if best_idx < len(thresholds):
            optimal_thresholds[i] = thresholds[best_idx]

    return optimal_thresholds


def compute_all_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
    threshold: float = 0.5,
) -> Dict:
    """Compute the full evaluation suite.

    Returns dict with:
    - per-class AUROC, PR-AUC
    - mean AUROC, mean PR-AUC
    - Macro/Micro/Weighted F1
    - optimal thresholds
    """
    y_pred = (y_proba >= threshold).astype(np.float32)

    auroc_per_class = compute_auroc_per_class(y_true, y_proba, class_names)
    pr_auc_per_class = compute_pr_auc_per_class(y_true, y_proba, class_names)
    f1_scores = compute_f1_scores(y_true, y_pred)
    prec_rec = compute_precision_recall(y_true, y_pred)

    # Mean AUROC (ignoring NaN)
    valid_aucs = [v for v in auroc_per_class.values() if not np.isnan(v)]
    mean_auroc = float(np.mean(valid_aucs)) if valid_aucs else 0.0

    valid_pr = [v for v in pr_auc_per_class.values() if not np.isnan(v)]
    mean_pr_auc = float(np.mean(valid_pr)) if valid_pr else 0.0

    return {
        "mean_auroc": mean_auroc,
        "mean_pr_auc": mean_pr_auc,
        "auroc_per_class": auroc_per_class,
        "pr_auc_per_class": pr_auc_per_class,
        **f1_scores,
        **prec_rec,
        "threshold": threshold,
        "num_samples": len(y_true),
    }


def format_metrics_report(
    metrics: Dict,
    class_names: List[str],
) -> str:
    """Format metrics into a human-readable report string."""
    lines = [
        "=" * 60,
        "EVALUATION REPORT",
        "=" * 60,
        "",
        f"Samples: {metrics['num_samples']} | Threshold: {metrics['threshold']}",
        f"Mean AUROC: {metrics['mean_auroc']:.4f} | Mean PR-AUC: {metrics['mean_pr_auc']:.4f}",
        f"Macro F1: {metrics['macro_f1']:.4f} | Micro F1: {metrics['micro_f1']:.4f}",
        "",
        "-" * 60,
        f"{'Class':<32s} {'AUROC':>8s} {'PR-AUC':>8s}",
        "-" * 60,
    ]
    for name in class_names:
        auroc = metrics["auroc_per_class"].get(name, float("nan"))
        pr_auc = metrics["pr_auc_per_class"].get(name, float("nan"))
        auroc_str = f"{auroc:.4f}" if not np.isnan(auroc) else "  N/A"
        pr_str = f"{pr_auc:.4f}" if not np.isnan(pr_auc) else "  N/A"
        lines.append(f"{name:<32s} {auroc_str:>8s} {pr_str:>8s}")

    lines.extend([
        "-" * 60,
        "",
        f"Weighted F1: {metrics.get('weighted_f1', 0):.4f}",
        f"Sample F1:   {metrics.get('sample_f1', 0):.4f}",
        "=" * 60,
    ])
    return "\n".join(lines)
