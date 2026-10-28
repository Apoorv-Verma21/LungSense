"""Smoke tests run after the pipeline. They fail (not skip) when an artifact is missing."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from src.data.dataset import SPLITS, build_transforms, load_image, load_split_frame
from src.utils.common import cfg_path, load_config, read_json

CFG = load_config()
NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")
# Identifiers, not values: image file names, URLs and DOIs are removed before numbers are checked.
FILE_NAME = re.compile(r"\S*\.(?:jpe?g|png|JPG|PNG)\b|https?://\S+|(?i:\bdoi\b):?\s*\S+")


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return load_split_frame(CFG)


def test_splits_have_zero_overlap(frame):
    payload = read_json(cfg_path(CFG, "splits"))
    assert payload["seed"] == CFG["seed"]
    assert set(frame["split"]) == set(SPLITS)
    assert frame["path"].is_unique
    groups = {name: set(frame.loc[frame["split"] == name, "group"]) for name in SPLITS}
    for a in SPLITS:
        for b in SPLITS:
            if a < b:
                assert not groups[a] & groups[b], f"group overlap between {a} and {b}"
    for name in SPLITS:
        assert set(frame.loc[frame["split"] == name, "label"]) == set(CFG["classes"]), f"{name} is missing a class"


def test_no_identical_image_files_across_splits(frame):
    index = pd.read_csv(cfg_path(CFG, "image_index"))[["path", "sha256"]]
    merged = frame.merge(index, on="path", how="left")
    assert merged["sha256"].notna().all()
    assert (merged.groupby("sha256")["split"].nunique() == 1).all(), "identical file appears in more than one split"


def test_model_loads_and_predicts_on_test_sample(frame):
    import torch

    from src.models.imaging import load_trained, softmax

    model, temperature = load_trained(CFG, "cpu")
    predictions = pd.read_csv(cfg_path(CFG, "results_dir") / "test_predictions.csv")
    row = predictions.iloc[0]
    assert frame.loc[frame["path"] == row["path"], "split"].item() == "test"
    batch = build_transforms(CFG, False)(load_image(row["path"])).unsqueeze(0)
    with torch.no_grad():
        probs = softmax(model(batch).numpy(), temperature)[0]
    assert np.isclose(probs.sum(), 1.0)
    saved = np.array([row[f"p_{name}"] for name in CFG["classes"]])
    assert np.allclose(probs, saved, atol=1e-3), "live prediction differs from results/test_predictions.csv"


def test_gradcam_produces_overlay(frame):
    from src.explain.gradcam import gradcam_overlay
    from src.models.imaging import load_trained

    model, temperature = load_trained(CFG, "cpu")
    path = frame.loc[frame["split"] == "test", "path"].iloc[0]
    image = load_image(path)
    result = gradcam_overlay(image, model, build_transforms(CFG, False), "cpu", temperature)
    size = CFG["imaging"]["image_size"]
    assert result["overlay"].size == (size, size)
    assert result["cam"].shape == (size, size)
    assert 0.0 <= result["cam"].min() and result["cam"].max() <= 1.0 and result["cam"].max() > 0


PREDICTIONS_FILE = "test_predictions.csv"


def _allowed_numbers(include_predictions: bool = False) -> list[float]:
    """Numbers in results/, the cohort summary and the config. Per-image test predictions are only
    allowed on the Predict page (its built-in samples are test images); elsewhere their hundreds of
    probabilities would let almost any made-up three-decimal value pass."""
    results = cfg_path(CFG, "results_dir")
    files = [*results.rglob("*.json"), *results.rglob("*.csv"), *results.rglob("*.md"), cfg_path(CFG, "cohort_summary"), ROOT / "configs" / "config.yaml"]
    files = [f for f in files if include_predictions or f.name != PREDICTIONS_FILE]
    values = set()
    for file in files:
        values.update(float(match) for match in NUMBER.findall(FILE_NAME.sub(" ", file.read_text(encoding="utf-8"))))
    return sorted(values)


def _is_allowed(token: str, allowed: list[float]) -> bool:
    value = float(token)
    decimals = len(token.split(".")[1]) if "." in token else 0
    for candidate in allowed:
        for scaled in (candidate, candidate * 100):
            if round(scaled, decimals) == value:
                return True
    return False


def _rendered_text(at) -> list[str]:
    texts = []
    for kind in ("markdown", "caption", "header", "subheader", "title", "info", "warning", "error", "success", "code", "json"):
        for element in getattr(at, kind, []):
            texts.append(str(element.value))
    for element in at.metric:
        texts.extend([str(element.label), str(element.value)])
    for element in at.dataframe:
        texts.append(element.value.to_string())
    return texts


def test_app_renders_only_numbers_from_results():
    from streamlit.testing.v1 import AppTest

    allowed, allowed_predict = _allowed_numbers(), _allowed_numbers(include_predictions=True)
    assert not _is_allowed("12.34", allowed), "allowed set is too permissive to catch made-up numbers"
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=180)
    at.run()
    assert not at.exception, at.exception
    pages = at.sidebar.radio(key="page").options
    offenders = {}
    for page in pages:
        at.sidebar.radio(key="page").set_value(page).run()
        assert not at.exception, f"{page}: {at.exception}"
        page_allowed = allowed_predict if page == "Predict" else allowed
        for text in _rendered_text(at):
            for token in NUMBER.findall(FILE_NAME.sub(" ", text)):
                if not _is_allowed(token, page_allowed):
                    offenders.setdefault(page, set()).add(token)
    assert not offenders, f"numbers rendered that are not in results/ or config: {offenders}"
