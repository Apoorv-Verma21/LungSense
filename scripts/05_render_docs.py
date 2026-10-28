"""Write docs/DATA_SOURCES.md and the generated sections of README.md and docs/MODEL_CARD.md.

Generated sections sit between `<!-- GENERATED:<name>:START -->` and `<!-- GENERATED:<name>:END -->`
markers and are rebuilt from results/ and configs/config.yaml on every run, so no result in the
documentation is typed by hand.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.common import cfg_path, load_config, read_json


def ci(item: dict | None) -> str:
    if not item or item.get("value") is None:
        return "not available"
    if item.get("ci_low") is None:
        return f"{item['value']:.3f}"
    return f"{item['value']:.3f} ({item['ci_low']:.3f}-{item['ci_high']:.3f})"


def replace_block(file: Path, name: str, body: str) -> None:
    text = file.read_text(encoding="utf-8")
    pattern = re.compile(rf"(<!-- GENERATED:{name}:START -->\n).*?(<!-- GENERATED:{name}:END -->)", re.S)
    if not pattern.search(text):
        raise SystemExit(f"{file} has no GENERATED:{name} markers")
    file.write_text(pattern.sub(lambda m: m.group(1) + body.rstrip() + "\n" + m.group(2), text), encoding="utf-8")


def data_sources(cfg: dict) -> str:
    official = cfg["source"]["official"]
    summary_file = cfg_path(cfg, "results_dir") / "cohort_summary.json"
    summary = read_json(summary_file) if summary_file.exists() else None
    lines = [
        "# Data sources",
        "",
        "LungSense uses exactly one source: **Source B, the IQ-OTH/NCCD lung cancer CT dataset**, downloaded "
        "from its Kaggle mirror. Source A (MIMIC-IV + MIMIC-CXR-JPG) was not available (no PhysioNet credentials).",
        "",
        "## Official record (provenance check)",
        "",
        f"- Title: {official['title']}",
        f"- Official page: {cfg['source']['official_page']}",
        f"- Version: {official['version']}, published {official['published']}, DOI {official['doi']}",
        f"- Licence: {official['licence']} ({official['licence_url']})",
        f"- Origin: {official['institution']}; collected over {official['collected']}; scanner {official['scanner']}",
        "- The official page states the scans are of real patients (lung cancer at different stages and healthy subjects), "
        "that the study was approved by the institutional review boards of the participating centres with written consent "
        "waived, and that all images were de-identified. It does not describe the data as synthetic or simulated.",
        f"- Stated size: {official['documented_images']} images from {official['documented_cases']} cases "
        f"({', '.join(f'{k} {v}' for k, v in official['documented_cases_per_class'].items())} cases).",
        f"- Citation: {official['citation']}",
        "",
        "## Copy used in this project",
        "",
        f"- Kaggle dataset: https://www.kaggle.com/datasets/{cfg['source']['kaggle_slug']}",
    ]
    if summary:
        download = summary["download"]
        lines += [
            f"- Accessed: {summary['access_date']}",
            f"- Kaggle reference: {download['slug']}, dataset version {download.get('kaggle_version', 'not recorded')}, "
            f"downloaded with {download['method']}",
            f"- Image-set SHA256 fingerprint (sorted path and file hashes): {summary['image_set_sha256']}",
            f"- Ignored unlabelled images: {summary['ignored_unlabelled_images']} in folder(s) {', '.join(summary['ignored_folders'])}; "
            "the Kaggle mirror bundles them, they carry no class label and their file names follow a different scheme "
            "from the IQ-OTH/NCCD images, so they were not used.",
            f"- Labelled images found: {summary['n_images']} "
            f"({', '.join(f'{k} {v}' for k, v in summary['class_counts'].items())}); "
            f"exact duplicate files: {summary['exact_duplicate_images']}",
            "- The Kaggle copy's image count differs from the official page's stated count when the two numbers above differ; "
            "the project uses the Kaggle copy as downloaded and does not add, remove or alter images beyond the documented "
            "exclusions in results/cohort_flow.json.",
        ]
    else:
        lines.append("- Download details: not available - run scripts/00_setup_data.py")
    return "\n".join(lines) + "\n"


def results_block(cfg: dict) -> str:
    results = cfg_path(cfg, "results_dir")
    metrics_file = results / "imaging_metrics.json"
    if not metrics_file.exists():
        return "Results not available - run `bash scripts/run_all.sh`.\n"
    metrics = read_json(metrics_file)
    test = metrics["test"]
    flow = read_json(results / "cohort_flow.json")
    spec = cfg["eval"]["fixed_specificity"]
    rows = [
        ("AUROC, malignant vs rest", ci(test["auroc"])),
        ("AUPRC, malignant vs rest", ci(test["auprc"])),
        ("Sensitivity (validation Youden threshold)", ci(test["sensitivity"])),
        ("Specificity (validation Youden threshold)", ci(test["specificity"])),
        (f"Sensitivity at {spec:.2f} specificity", ci(test["sens_at_90_spec"])),
        ("Macro one-vs-rest AUROC, three classes", ci(test["macro_ovr_auroc_3class"])),
        *[(f"Recall, {name}", ci(item)) for name, item in test["per_class_recall"].items()],
    ]
    lines = [
        f"Test split: {metrics['n_images']['test']} images in {metrics['n_groups']['test']} pseudo-patient groups "
        f"(train {metrics['n_images']['train']}, validation {metrics['n_images']['val']}; "
        f"{flow['included_images']} images included in total). "
        f"{cfg['eval']['ci_level']}% CIs from {metrics['bootstrap']['resamples']} group-level bootstrap resamples.",
        "",
        f"| Metric | Value ({cfg['eval']['ci_level']}% CI) |",
        "|---|---|",
        *[f"| {name} | {value} |" for name, value in rows],
        "",
        f"Calibration: temperature {metrics['temperature']:.3f}; malignant-class ECE "
        f"{metrics['test_ece_malignant']['before_scaling']:.3f} before and {metrics['test_ece_malignant']['after_scaling']:.3f} after scaling. "
        f"Best epoch {metrics['best_epoch']} of {metrics['epochs_run']} run on {metrics['device']}; input preprocessing "
        f"`{metrics['preprocess']}`.",
    ]
    shortcut_file = results / "shortcut_baseline.json"
    if shortcut_file.exists():
        shortcut = read_json(shortcut_file)
        lines += [
            "",
            "**Shortcut warning:** a logistic regression on intensity histograms alone (no spatial information) reaches "
            + "; ".join(f"malignant AUROC {ci(item['test_auroc'])} on `{item['input_preprocess']}` input" for item in shortcut.values())
            + ". The near-perfect malignant-vs-rest numbers therefore mostly reflect how the dataset was acquired and "
            "exported, not demonstrated cancer detection. See the model card.",
        ]
    return "\n".join(lines) + "\n"


def model_card_block(cfg: dict) -> str:
    results = cfg_path(cfg, "results_dir")
    if not (results / "imaging_metrics.json").exists():
        return "Metrics not available - run `bash scripts/run_all.sh`.\n"
    grouping = read_json(results / "grouping_diagnostics.json")
    audit = read_json(results / "leakage_audit.json")
    gradcam = read_json(results / "gradcam_checks.json")
    metrics = read_json(results / "imaging_metrics.json")
    thresholds = metrics["thresholds"]
    lines = [
        "### Test metrics",
        "",
        results_block(cfg).rstrip(),
        "",
        "### Operating thresholds (fit on validation, calibrated P(malignant))",
        "",
        f"- Youden threshold (used for sensitivity/specificity): {thresholds['youden_threshold']:.3f}",
        f"- Rule-out threshold (validation sensitivity >= {thresholds['rule_out_target_sensitivity']:.2f}): {thresholds['rule_out_threshold']:.3f}",
        f"- Rule-in threshold (validation specificity >= {thresholds['rule_in_target_specificity']:.2f}): {thresholds['rule_in_threshold']:.3f}",
        "",
        "### Pseudo-patient grouping and residual leakage",
        "",
        f"- Groups per class: {grouping['groups_per_class']} versus cases stated by the source: {grouping['documented_cases_per_class']}",
        f"- Link threshold {grouping['link_threshold']}; near-duplicate threshold {grouping['near_duplicate_threshold']}",
        f"- Group overlap between splits: {audit['group_overlap_between_splits']}",
        *[
            f"- {name}: {audit[name]['n_images']} images; median nearest-train similarity "
            f"{audit[name]['max_train_similarity_quantiles']['0.5']:.3f}; at or above link threshold "
            f"{audit[name]['n_at_or_above_link_threshold']}; near-duplicates of a training image {audit[name]['n_near_duplicate_of_train']}"
            for name in ("val", "test")
        ],
        "",
        "### Grad-CAM shortcut checks (all test images)",
        "",
        f"- Mean heatmap share in the outer border band: {gradcam['mean_border_mass']:.3f} (band covers "
        f"{gradcam['border_band_fraction_of_image_area']:.3f} of the image area)",
        f"- Share of heatmap peaks inside the border band: {gradcam['share_peak_in_border']:.3f}",
        f"- Mean heatmap share outside the body mask: {gradcam['mean_outside_body_mass']:.3f}",
        f"- Gallery: {gradcam['n_correct_examples']} correct and {gradcam['n_wrong_examples']} wrong examples "
        f"({gradcam['n_wrong_available']} wrong test predictions available)",
    ]
    ablation = results / "ablation_table.md"
    if ablation.exists():
        text = ablation.read_text(encoding="utf-8")
        if "## Preprocessing comparison" in text:
            lines += ["", "### " + text.split("## ", 2)[-1].strip()]
    return "\n".join(lines) + "\n"


def main() -> None:
    cfg = load_config()
    (ROOT / "docs" / "DATA_SOURCES.md").write_text(data_sources(cfg), encoding="utf-8")
    replace_block(ROOT / "README.md", "results", results_block(cfg))
    replace_block(ROOT / "docs" / "MODEL_CARD.md", "metrics", model_card_block(cfg))
    print("Rendered docs/DATA_SOURCES.md and generated sections of README.md and docs/MODEL_CARD.md")


if __name__ == "__main__":
    main()
