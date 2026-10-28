"""Evaluation figures written to results/figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve


def _save(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def roc_plot(y_true, y_score, out_path: Path, title: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot(fpr, tpr, lw=2, label=f"AUROC {roc_auc_score(y_true, y_score):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set(xlabel="1 - specificity", ylabel="Sensitivity", title=title)
    ax.legend(loc="lower right")
    _save(fig, out_path)


def pr_plot(y_true, y_score, out_path: Path, title: str) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot(recall, precision, lw=2, label=f"AUPRC {average_precision_score(y_true, y_score):.3f}")
    ax.axhline(np.mean(y_true), ls="--", color="grey", lw=1, label=f"Prevalence {np.mean(y_true):.3f}")
    ax.set(xlabel="Recall (sensitivity)", ylabel="Precision", title=title)
    ax.legend(loc="lower left")
    _save(fig, out_path)


def calibration_plot(y_true, raw_prob, calibrated_prob, out_path: Path, title: str, n_bins: int = 10) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Perfect")
    for prob, name in ((raw_prob, "Before scaling"), (calibrated_prob, "After temperature scaling")):
        frac_pos, mean_pred = calibration_curve(y_true, prob, n_bins=n_bins, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", lw=1.5, label=name)
    ax.set(xlabel="Predicted probability", ylabel="Observed frequency", title=title, xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper left")
    _save(fig, out_path)


def confusion_plot(y_true, y_pred, class_names, out_path: Path, title: str) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(matrix, cmap="Blues")
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, str(value), ha="center", va="center", color="white" if value > matrix.max() / 2 else "black")
    ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)), xticklabels=class_names, yticklabels=class_names,
           xlabel="Predicted", ylabel="True", title=title)
    _save(fig, out_path)


def training_curve_plot(history: list[dict], out_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, ax1 = plt.subplots(figsize=(5.6, 3.6))
    ax1.plot(epochs, [row["train_loss"] for row in history], label="Train loss")
    ax1.set(xlabel="Epoch", ylabel="Loss")
    ax2 = ax1.twinx()
    ax2.plot(epochs, [row["val_macro_auroc"] for row in history], color="tab:orange", label="Val macro AUROC")
    ax2.set_ylabel("Validation macro AUROC")
    fig.legend(loc="lower center", ncol=2, fontsize=8)
    _save(fig, out_path)
