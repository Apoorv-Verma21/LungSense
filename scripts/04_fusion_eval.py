"""Fusion and ablation (Source A only).

Under Source B there is a single modality, so no fusion is possible. This script writes
results/ablation_table.md / .csv stating "single modality only", with the imaging-only row read
from results/imaging_metrics.json (never typed by hand).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.utils.common import cfg_path, load_config, read_json, read_source


def fmt(item: dict) -> str:
    if item["ci_low"] is None:
        return f"{item['value']:.3f}"
    return f"{item['value']:.3f} ({item['ci_low']:.3f}-{item['ci_high']:.3f})"


def main() -> None:
    cfg = load_config()
    if read_source(cfg) != "B":
        raise SystemExit("Source A fusion pipeline is not implemented in this build (see docs/DECISIONS.md).")
    results_dir = cfg_path(cfg, "results_dir")
    metrics_file = results_dir / "imaging_metrics.json"
    if not metrics_file.exists():
        raise SystemExit(f"{metrics_file} is missing - run scripts/03_train_imaging.py first.")
    test = read_json(metrics_file)["test"]
    n_test = read_json(metrics_file)["n_images"]["test"]
    ci, spec = cfg["eval"]["ci_level"], round(cfg["eval"]["fixed_specificity"] * 100)

    row = {
        "model": "CT imaging only (EfficientNet-B0)",
        "auroc": test["auroc"]["value"], "auroc_ci_low": test["auroc"]["ci_low"], "auroc_ci_high": test["auroc"]["ci_high"],
        "auprc": test["auprc"]["value"], "auprc_ci_low": test["auprc"]["ci_low"], "auprc_ci_high": test["auprc"]["ci_high"],
        "sens_at_90_spec": test["sens_at_90_spec"]["value"],
        "sens_at_90_spec_ci_low": test["sens_at_90_spec"]["ci_low"], "sens_at_90_spec_ci_high": test["sens_at_90_spec"]["ci_high"],
        "n_test": n_test,
    }
    pd.DataFrame([row]).to_csv(results_dir / "ablation_table.csv", index=False)
    lines = [
        "# Ablation table",
        "",
        "**Single modality only.** The data source (IQ-OTH/NCCD CT slices) has no clinical variables "
        "and no second modality for the same patients, so clinical-only, late-fusion and "
        "intermediate-fusion rows cannot be computed. No other dataset was paired in.",
        "",
        f"| Model | AUROC ({ci}% CI) | AUPRC ({ci}% CI) | Sens @ {spec}% spec ({ci}% CI) | n_test images |",
        "|---|---|---|---|---|",
        f"| {row['model']} | {fmt(test['auroc'])} | {fmt(test['auprc'])} | {fmt(test['sens_at_90_spec'])} | {n_test} |",
        "",
        f"Task: malignant vs (normal + benign). CIs: {cfg['eval']['bootstrap']} pseudo-patient-group bootstrap resamples.",
        "",
        "## Preprocessing comparison (same single modality)",
        "",
        "Not a fusion ablation: the same CT model trained on differently preprocessed inputs, to show how much the "
        "headline numbers depend on acquisition cues. The shortcut baseline uses only each slice's intensity histogram.",
        "",
        f"| Input preprocessing | Malignant AUROC ({ci}% CI) | Three-class macro AUROC ({ci}% CI) | Recall normal | Recall benign | Recall malignant |",
        "|---|---|---|---|---|---|",
    ]
    runs = [("main", results_dir / "imaging_metrics.json")] + sorted(
        (path.parent.name, path) for path in (results_dir / "variants").glob("*/imaging_metrics.json"))
    for name, path in runs:
        run = read_json(path)
        recall = run["test"]["per_class_recall"]
        lines.append(f"| {run['preprocess']}{' (main model)' if name == 'main' else ''} | {fmt(run['test']['auroc'])} | "
                     f"{fmt(run['test']['macro_ovr_auroc_3class'])} | {recall['normal']['value']:.3f} | "
                     f"{recall['benign']['value']:.3f} | {recall['malignant']['value']:.3f} |")
    shortcut_file = results_dir / "shortcut_baseline.json"
    if shortcut_file.exists():
        lines += ["", f"| Shortcut baseline input | Malignant AUROC ({ci}% CI) |", "|---|---|"]
        lines += [f"| {item['input_preprocess']} | {fmt(item['test_auroc'])} |" for item in read_json(shortcut_file).values()]
    (results_dir / "ablation_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
