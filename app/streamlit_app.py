"""LungSense dashboard. Every number shown is read from results/, data/cohort_summary.json,
configs/config.yaml, or computed live by the trained model on an image; nothing is typed in."""
from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.utils.common import cfg_path, load_config, read_json

DISCLAIMER = "Research/education demo. Not a medical device. Not for clinical use."
MAX_UPLOAD_MB = 10
PAGES = [
    "Overview",
    "Predict",
    "Model performance",
    "Explainability gallery",
    "Clinical model",
    "Fusion and ablation",
    "Similar cases",
    "Model card and limitations",
]

st.set_page_config(page_title="LungSense", page_icon=":material/pulmonology:", layout="wide")
CFG = load_config()
RESULTS = cfg_path(CFG, "results_dir")
CLASSES = CFG["classes"]
CI = CFG["eval"]["ci_level"]


def load_json(path: Path):
    return read_json(path) if path.exists() else None


def not_available(what: str, script: str) -> None:
    st.info(f"{what}: not available - run `{script}`.")


def fmt_ci(item: dict | None, pct: bool = False) -> str:
    if not item or item.get("value") is None:
        return "not available"
    scale, suffix = (100, "%") if pct else (1, "")
    text = f"{item['value'] * scale:.1f}{suffix}" if pct else f"{item['value']:.3f}"
    if item.get("ci_low") is None:
        return text
    low, high = item["ci_low"] * scale, item["ci_high"] * scale
    return f"{text} ({low:.1f}-{high:.1f}{suffix})" if pct else f"{text} ({low:.3f}-{high:.3f})"


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    import torch

    from src.models.imaging import load_trained

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model, temperature = load_trained(CFG, device)
    except FileNotFoundError:
        return None
    return model, temperature, device


@st.cache_data
def test_samples() -> pd.DataFrame | None:
    """One built-in sample per class, from the TEST split only (first path in sorted order)."""
    splits_file = cfg_path(CFG, "splits")
    if not splits_file.exists():
        return None
    frame = pd.DataFrame(read_json(splits_file)["images"])
    test = frame[frame["split"] == "test"].sort_values("path")
    return test.groupby("label", sort=False).head(1).set_index("label").reindex(CLASSES).dropna().reset_index()


def risk_band(p_malignant: float, thresholds: dict) -> str:
    if p_malignant >= thresholds["rule_in_threshold"]:
        return "High"
    if p_malignant < thresholds["rule_out_threshold"]:
        return "Low"
    return "Indeterminate"


def read_upload(upload) -> Image.Image | None:
    if upload.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"File is larger than {MAX_UPLOAD_MB} MB.")
        return None
    try:
        image = Image.open(io.BytesIO(upload.getvalue()))
        image.load()
    except (UnidentifiedImageError, OSError):
        st.error("Could not read this file as an image. Upload a PNG or JPEG CT slice.")
        return None
    if min(image.size) < 64:
        st.error("Image is too small (minimum 64 px per side).")
        return None
    return image.convert("L")


def selected_input() -> tuple[Image.Image | None, str | None, str | None]:
    """Return (image, label shown to the user, ground-truth label for test samples)."""
    samples = test_samples()
    options = ["Upload an image"]
    if samples is not None and len(samples):
        options = [f"Test-split sample: {row.label}" for row in samples.itertuples()] + options
    choice = st.radio("Input", options, horizontal=True, key="input_choice")
    if choice == "Upload an image":
        st.caption("The model was trained only on axial chest CT slices from this source; other images give meaningless output.")
        upload = st.file_uploader("CT slice (PNG/JPEG)", type=["png", "jpg", "jpeg"])
        if upload is None:
            return None, None, None
        return read_upload(upload), upload.name, None
    row = samples[samples["label"] == choice.split(": ", 1)[1]].iloc[0]
    from src.data.dataset import load_image

    return load_image(row["path"]), row["path"], row["label"]


def shortcut_warning() -> None:
    shortcut = load_json(RESULTS / "shortcut_baseline.json")
    if shortcut is None:
        not_available("Shortcut baseline", "python scripts/03_train_imaging.py")
        return
    parts = [f"{fmt_ci(item['test_auroc'])} on `{item['input_preprocess']}` input" for item in shortcut.values()]
    st.error(
        "**Shortcut warning.** A logistic regression that sees only each slice's intensity histogram (no shapes, "
        f"no anatomy) reaches malignant AUROC {'; '.join(parts)}. Near-perfect malignant-vs-rest scores on this "
        "dataset mostly reflect how the scans were acquired and exported. They are not evidence that the model "
        "detects cancer. See the Model card page."
    )


def page_overview() -> None:
    st.header("LungSense")
    st.write(
        "LungSense classifies single chest CT slices as normal, benign or malignant and reports a calibrated "
        "probability of malignancy, with Grad-CAM heatmaps showing where the model looked. It uses one real, "
        "public data source and is imaging-only: the source has no clinical variables, so there is no clinical "
        "model and no fusion."
    )
    source_file = cfg_path(CFG, "source_file")
    if source_file.exists():
        st.subheader("Data source")
        st.code(source_file.read_text(encoding="utf-8"), language=None)
    else:
        not_available("Data source", "python scripts/00_setup_data.py")

    summary = load_json(RESULTS / "cohort_summary.json")
    flow = load_json(RESULTS / "cohort_flow.json")
    if summary and flow:
        cols = st.columns(3)
        cols[0].metric("Labelled images downloaded", summary["n_images"])
        cols[1].metric("Images included", flow["included_images"])
        cols[2].metric("Pseudo-patient groups", flow["pseudo_patient_groups"])
        rows = [{"split": name, "images": item["images"], "groups": item["groups"],
                 **{f"{c} images": item["images_per_class"].get(c, 0) for c in CLASSES}}
                for name, item in flow["splits"].items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption(
            f"Excluded identical files with conflicting labels: {flow['excluded_identical_files_with_conflicting_labels']}. "
            "Splits are made by pseudo-patient group because the source has no patient IDs (see Model card)."
        )
    else:
        not_available("Cohort summary", "python scripts/01_build_cohort.py")

    st.subheader("Architecture")
    st.graphviz_chart(
        """
        digraph {
          rankdir=LR; node [shape=box, style=rounded, fontsize=11];
          src [label="Kaggle mirror of\\nIQ-OTH/NCCD CT slices"];
          cohort [label="Cohort: standard-size slices,\\npseudo-patient group split"];
          prep [label="Body mask +\\nequalisation inside body"];
          model [label="EfficientNet-B0\\nnormal / benign / malignant"];
          cal [label="Temperature scaling\\n(validation split)"];
          out [label="P(malignant), risk band,\\nGrad-CAM"];
          src -> cohort -> prep -> model -> cal -> out;
        }
        """
    )
    st.subheader("Provenance")
    data_sources = ROOT / "docs" / "DATA_SOURCES.md"
    if data_sources.exists():
        # Drop the document's own title and demote its sections under this page's subheader.
        body = data_sources.read_text(encoding="utf-8").split("\n", 1)[1].replace("\n## ", "\n#### ")
        st.markdown(body)
    else:
        not_available("Provenance record", "python scripts/05_render_docs.py")
    st.subheader("Limitations")
    st.markdown(
        "- Small dataset from a single centre; results may not transfer to other scanners or hospitals.\n"
        "- No patient IDs: slices were grouped into pseudo-patients by image similarity, so some leakage between splits may remain.\n"
        "- Imaging only, single 2D slices; no clinical data, no 3D context.\n"
        "- Labels are the source's case classes, not linked to biopsy outcomes in this release.\n"
        "- See the Model card page for the full list."
    )


def page_predict() -> None:
    st.header("Predict")
    loaded = load_model()
    thresholds = load_json(RESULTS / "thresholds.json")
    if loaded is None or thresholds is None:
        not_available("Imaging model", "python scripts/03_train_imaging.py")
        return
    model, temperature, device = loaded
    st.caption("Malignant-vs-rest scores on this dataset are inflated by acquisition shortcuts; see Model performance and Model card.")
    image, name, truth = selected_input()
    if image is None:
        return

    from src.data.dataset import build_transforms
    from src.explain.gradcam import gradcam_overlay

    result = gradcam_overlay(image, model, build_transforms(CFG, False), device, temperature)
    probs = result["probs"]
    p_malignant = float(probs[CLASSES.index("malignant")])
    band = risk_band(p_malignant, thresholds)

    left, right = st.columns(2)
    left.image(image, caption=f"Input: {name}", width="stretch")
    right.image(result["overlay"], caption=f"Grad-CAM for predicted class: {CLASSES[result['class_idx']]}", width="stretch")

    cols = st.columns(3)
    cols[0].metric("P(malignant), calibrated", f"{p_malignant * 100:.1f}%")
    cols[1].metric("Risk band", band)
    cols[2].metric("Predicted class", CLASSES[result["class_idx"]])
    if truth is not None:
        st.caption(f"Ground-truth label of this test-split sample: **{truth}**")
    st.dataframe(pd.DataFrame({"class": CLASSES, "calibrated probability": [f"{p * 100:.1f}%" for p in probs]}), hide_index=True)
    st.markdown(
        f"**Risk bands** (thresholds fit on the validation split): High if P(malignant) >= "
        f"{thresholds['rule_in_threshold']:.3f} (validation specificity >= {thresholds['rule_in_target_specificity']:.2f}); "
        f"Low if below {thresholds['rule_out_threshold']:.3f} (validation sensitivity >= {thresholds['rule_out_target_sensitivity']:.2f}); "
        "otherwise Indeterminate."
    )
    st.subheader("Clinical explanation (SHAP)")
    st.info("Not available in this data source: it has no clinical variables, so there is no clinical model to explain.")


def page_performance() -> None:
    st.header("Model performance (test split)")
    metrics = load_json(RESULTS / "imaging_metrics.json")
    if metrics is None:
        not_available("Test metrics", "python scripts/03_train_imaging.py")
        return
    test = metrics["test"]
    st.caption(
        f"Task: {metrics['task']}. {CI}% CIs from {metrics['bootstrap']['resamples']} bootstrap resamples of "
        f"{metrics['bootstrap']['unit']}s. Test split: {metrics['n_images']['test']} images in "
        f"{metrics['n_groups']['test']} groups."
    )
    rows = [
        ("AUROC (malignant vs rest)", fmt_ci(test["auroc"])),
        ("AUPRC (malignant vs rest)", fmt_ci(test["auprc"])),
        ("Sensitivity at validation Youden threshold", fmt_ci(test["sensitivity"])),
        ("Specificity at validation Youden threshold", fmt_ci(test["specificity"])),
        (f"Sensitivity at {CFG['eval']['fixed_specificity']:.2f} specificity", fmt_ci(test["sens_at_90_spec"])),
        ("Macro one-vs-rest AUROC (three classes)", fmt_ci(test["macro_ovr_auroc_3class"])),
    ] + [(f"Recall: {name}", fmt_ci(item)) for name, item in test["per_class_recall"].items()]
    shortcut_warning()
    st.dataframe(pd.DataFrame(rows, columns=["metric", f"value ({CI}% CI)"]), hide_index=True)
    st.caption(
        f"Temperature {metrics['temperature']:.3f}; malignant-class ECE {metrics['test_ece_malignant']['before_scaling']:.3f} before and "
        f"{metrics['test_ece_malignant']['after_scaling']:.3f} after scaling."
    )
    figures = cfg_path(CFG, "figures_dir")
    pairs = [("roc.png", "pr.png"), ("calibration.png", "confusion_matrix.png"), ("training_curve.png", "grouping_correlation.png")]
    for pair in pairs:
        cols = st.columns(2)
        for col, file in zip(cols, pair):
            if (figures / file).exists():
                col.image(str(figures / file), width="stretch")
            else:
                with col:
                    not_available(file, "python scripts/03_train_imaging.py")
    st.subheader("Ablation table")
    ablation = RESULTS / "ablation_table.md"
    if ablation.exists():
        st.markdown(ablation.read_text(encoding="utf-8"))
    else:
        not_available("Ablation table", "python scripts/04_fusion_eval.py")
    st.subheader("Leakage audit")
    audit = load_json(RESULTS / "leakage_audit.json")
    if audit:
        st.json(audit, expanded=False)
        st.caption("Nearest-training-image similarity for validation and test images (thumbnail correlation).")
    else:
        not_available("Leakage audit", "python scripts/01_build_cohort.py")


def page_gallery() -> None:
    st.header("Explainability gallery (Grad-CAM, test split)")
    checks = load_json(RESULTS / "gradcam_checks.json")
    if checks is None:
        not_available("Grad-CAM examples", "python scripts/03_train_imaging.py")
        return
    st.caption(
        f"Shortcut checks over {checks['n_test_images_checked']} test images: mean share of heatmap in the outer border band "
        f"{checks['mean_border_mass']:.3f} (band covers {checks['border_band_fraction_of_image_area']:.3f} of the image); "
        f"share of heatmap peaks inside the band {checks['share_peak_in_border']:.3f}; mean share outside the body "
        f"{checks['mean_outside_body_mass']:.3f}."
    )
    gallery_dir = cfg_path(CFG, "gradcam_dir")
    for correct in (True, False):
        items = [item for item in checks["examples"] if item["correct"] is correct]
        st.subheader("Correct predictions" if correct else "Wrong predictions")
        if not items:
            st.write("None in the test split.")
            continue
        cols = st.columns(2)
        for n, item in enumerate(items):
            cols[n % 2].image(str(gallery_dir / item["file"]),
                              caption=f"true {item['true']}, predicted {item['pred']}, P(malignant) {item['p_malignant']:.3f}",
                              width="stretch")


def page_clinical() -> None:
    st.header("Clinical model")
    status = load_json(RESULTS / "clinical_status.json")
    if status is None:
        not_available("Clinical model status", "python scripts/02_train_clinical.py")
        return
    st.warning(f"Not available in this data source. {status['reason']}")


def page_fusion() -> None:
    st.header("Fusion and ablation")
    ablation = RESULTS / "ablation_table.md"
    if not ablation.exists():
        not_available("Ablation table", "python scripts/04_fusion_eval.py")
        return
    st.warning("Fusion is not available in this data source: single modality only.")
    st.markdown(ablation.read_text(encoding="utf-8"))


def page_similar() -> None:
    st.header("Similar cases (retrieval, not diagnosis)")
    st.caption("Nearest training-split images in the model's embedding space. This shows what the model considers similar; it is not a diagnosis.")
    loaded = load_model()
    embeddings_file = cfg_path(CFG, "embeddings_file")
    if loaded is None or not embeddings_file.exists():
        not_available("Embedding index", "python scripts/03_train_imaging.py")
        return
    model, _, device = loaded
    image, name, _ = selected_input()
    if image is None:
        return
    import torch

    from src.data.dataset import build_transforms, load_image
    from src.models.imaging import embed

    index = np.load(embeddings_file, allow_pickle=True)
    with torch.no_grad():
        query = embed(model, build_transforms(CFG, False)(image).unsqueeze(0).to(device)).float().cpu().numpy()[0]
    vectors = index["vectors"]
    sims = vectors @ query / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query) + 1e-8)
    st.image(image, caption=f"Query: {name}", width=260)
    nearest = np.argsort(-sims)[:5]
    cols = st.columns(len(nearest))
    for col, idx in zip(cols, nearest):
        col.image(load_image(index["paths"][idx]), caption=f"train image, label {index['labels'][idx]}", width="stretch")


def page_model_card() -> None:
    st.header("Model card and limitations")
    card = ROOT / "docs" / "MODEL_CARD.md"
    if card.exists():
        st.markdown(card.read_text(encoding="utf-8"))
    else:
        not_available("Model card", "python scripts/05_render_docs.py")


st.warning(DISCLAIMER, icon=":material/warning:")
requested = st.query_params.get("page")
page = st.sidebar.radio("Page", PAGES, index=PAGES.index(requested) if requested in PAGES else 0, key="page")
{
    "Overview": page_overview,
    "Predict": page_predict,
    "Model performance": page_performance,
    "Explainability gallery": page_gallery,
    "Clinical model": page_clinical,
    "Fusion and ablation": page_fusion,
    "Similar cases": page_similar,
    "Model card and limitations": page_model_card,
}[page]()
st.divider()
st.caption(DISCLAIMER)
