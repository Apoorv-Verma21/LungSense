"""Evaluation metrics with group-level bootstrap confidence intervals."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def sensitivity_specificity(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> tuple[float, float]:
    y_pred = y_score >= threshold
    positives = y_true == 1
    negatives = ~positives
    sensitivity = float((y_pred & positives).sum() / positives.sum()) if positives.any() else float("nan")
    specificity = float((~y_pred & negatives).sum() / negatives.sum()) if negatives.any() else float("nan")
    return sensitivity, specificity


def sensitivity_at_specificity(y_true: np.ndarray, y_score: np.ndarray, target_specificity: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    allowed = (1 - fpr) >= target_specificity
    return float(tpr[allowed].max()) if allowed.any() else 0.0


def youden_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    best = int(np.argmax(tpr - fpr))
    return float(min(thresholds[best], 1.0))


def threshold_for_sensitivity(y_true: np.ndarray, y_score: np.ndarray, target: float) -> float:
    """Highest threshold whose sensitivity is at least `target`."""
    _, tpr, thresholds = roc_curve(y_true, y_score)
    allowed = np.where(tpr >= target)[0]
    return float(min(thresholds[allowed[0]], 1.0))


def threshold_for_specificity(y_true: np.ndarray, y_score: np.ndarray, target: float) -> float:
    """Lowest threshold whose specificity is at least `target`."""
    fpr, _, thresholds = roc_curve(y_true, y_score)
    allowed = np.where((1 - fpr) >= target)[0]
    return float(min(thresholds[allowed[-1]], 1.0))


def per_class_recall(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> list[float]:
    return [float((y_pred[y_true == k] == k).mean()) if (y_true == k).any() else float("nan") for k in range(n_classes)]


def macro_ovr_auroc(y_true: np.ndarray, probs: np.ndarray) -> float:
    present = [k for k in range(probs.shape[1]) if (y_true == k).any() and (y_true != k).any()]
    return float(np.mean([roc_auc_score(y_true == k, probs[:, k]) for k in present]))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (y_prob > low) & (y_prob <= high) if low > 0 else (y_prob >= low) & (y_prob <= high)
        if in_bin.any():
            ece += in_bin.mean() * abs(y_prob[in_bin].mean() - y_true[in_bin].mean())
    return float(ece)


def group_bootstrap(
    metric: Callable[[np.ndarray], float],
    groups: np.ndarray,
    n_resamples: int,
    seed: int,
) -> dict:
    """Percentile 95% CI resampling whole groups (pseudo-patients), not single slices.

    `metric` receives an index array into the evaluation rows and returns a float;
    resamples where the metric is undefined (for example, one class only) are skipped
    and counted in `n_valid`.
    """
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    rows_by_group = {group: np.where(groups == group)[0] for group in unique_groups}
    estimates = []
    for _ in range(n_resamples):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        index = np.concatenate([rows_by_group[group] for group in sampled])
        try:
            value = metric(index)
        except ValueError:
            continue
        if np.isfinite(value):
            estimates.append(value)
    point = metric(np.arange(len(groups)))
    if not estimates:
        return {"value": point, "ci_low": None, "ci_high": None, "n_valid": 0}
    low, high = np.percentile(estimates, [2.5, 97.5])
    return {"value": float(point), "ci_low": float(low), "ci_high": float(high), "n_valid": len(estimates)}


def full_report(
    labels: np.ndarray,
    probs: np.ndarray,
    groups: np.ndarray,
    positive_class: int,
    threshold: float,
    n_resamples: int,
    seed: int,
    fixed_specificity: float = 0.90,
) -> dict:
    """Binary (positive class vs rest) and multi-class metrics, each with a group bootstrap CI."""
    y_binary = (labels == positive_class).astype(int)
    p_positive = probs[:, positive_class]
    y_pred = probs.argmax(axis=1)

    def boot(fn):
        return group_bootstrap(fn, groups, n_resamples, seed)

    report = {
        "auroc": boot(lambda i: roc_auc_score(y_binary[i], p_positive[i])),
        "auprc": boot(lambda i: average_precision_score(y_binary[i], p_positive[i])),
        "sensitivity": boot(lambda i: sensitivity_specificity(y_binary[i], p_positive[i], threshold)[0]),
        "specificity": boot(lambda i: sensitivity_specificity(y_binary[i], p_positive[i], threshold)[1]),
        "sens_at_90_spec": boot(lambda i: sensitivity_at_specificity(y_binary[i], p_positive[i], fixed_specificity)),
        "macro_ovr_auroc_3class": boot(lambda i: macro_ovr_auroc(labels[i], probs[i])),
        "per_class_recall": {},
        "ece_positive_class": expected_calibration_error(y_binary, p_positive),
    }
    for k in range(probs.shape[1]):
        report["per_class_recall"][str(k)] = boot(lambda i, k=k: per_class_recall(labels[i], y_pred[i], probs.shape[1])[k])
    return report
