"""Shortcut baseline: how well do global intensity statistics alone predict the label?

A logistic regression on each slice's normalised 32-bin grayscale histogram has no access to
anatomy or lesion shape. If it scores close to the CNN, the CNN's performance may rest on
acquisition differences (windowing, exposure, reconstruction) between classes, not pathology.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.dataset import load_image, preprocessor
from src.eval.metrics import group_bootstrap

N_BINS = 32


def histogram_features(paths: list[str], step) -> np.ndarray:
    rows = []
    for path in paths:
        image = load_image(path)
        pixels = np.asarray(step(image) if step is not None else image, dtype=np.uint8).ravel()
        counts = np.bincount(pixels // (256 // N_BINS), minlength=N_BINS).astype(float)
        rows.append(counts / counts.sum())
    return np.stack(rows)


def shortcut_baseline(frame: pd.DataFrame, positive_label: str, n_resamples: int, seed: int, preprocess: str) -> dict:
    step = preprocessor({"imaging": {"preprocess": preprocess}}, preprocess)
    train, test = frame[frame["split"] == "train"], frame[frame["split"] == "test"]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    model.fit(histogram_features(train["path"].tolist(), step), (train["label"] == positive_label).astype(int))
    y_test = (test["label"] == positive_label).astype(int).to_numpy()
    p_test = model.predict_proba(histogram_features(test["path"].tolist(), step))[:, 1]
    auroc = group_bootstrap(lambda i: roc_auc_score(y_test[i], p_test[i]), test["group"].to_numpy(), n_resamples, seed)
    return {
        "description": "logistic regression on 32-bin intensity histograms (no spatial information), malignant vs rest",
        "input_preprocess": preprocess,
        "test_auroc": auroc,
    }
