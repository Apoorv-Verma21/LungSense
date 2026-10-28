"""Train, calibrate and evaluate the CT imaging model, then write Grad-CAM examples.

EfficientNet-B0 (ImageNet weights) on 224px slices, 3 classes (normal/benign/malignant);
the binary task malignant vs (normal + benign) uses the calibrated malignant probability.
Weighted cross-entropy, AdamW + cosine schedule, mixed precision on CUDA, early stopping on
validation macro one-vs-rest AUROC, temperature scaling fit on validation. Test metrics use a
pseudo-patient-group bootstrap. Use --fast for a short CPU run.
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, log_loss
from torch.utils.data import DataLoader

from src.data.dataset import PREPROCESS_CHOICES, CTSliceDataset, build_transforms, load_image, load_split_frame, preprocessor
from src.eval import plots
from src.eval.metrics import (
    expected_calibration_error,
    full_report,
    macro_ovr_auroc,
    threshold_for_sensitivity,
    threshold_for_specificity,
    youden_threshold,
)
from src.eval.shortcut import shortcut_baseline
from src.explain.gradcam import attention_checks, gradcam_overlay
from src.models.imaging import build_model, embed, fit_temperature, predict_logits, softmax
from src.utils.common import cfg_path, load_config, read_source, seed_everything, write_json


def train(cfg: dict, frame: pd.DataFrame, device: torch.device, epochs: int, batch_size: int, preprocess: str) -> tuple[torch.nn.Module, list[dict], int]:
    icfg = cfg["imaging"]
    train_rows, val_rows = frame[frame["split"] == "train"], frame[frame["split"] == "val"]
    generator = torch.Generator().manual_seed(cfg["seed"])
    step = preprocessor(cfg, preprocess)
    train_loader = DataLoader(CTSliceDataset(train_rows, build_transforms(cfg, True, "none"), step), batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(CTSliceDataset(val_rows, build_transforms(cfg, False, "none"), step), batch_size=batch_size)

    counts = np.bincount(train_rows["label_idx"], minlength=len(cfg["classes"]))
    class_weights = torch.tensor(counts.sum() / (len(counts) * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    model = build_model(icfg["backbone"], len(cfg["classes"]), pretrained=True).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=icfg["lr"], weight_decay=icfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_auroc, best_state, best_epoch, stale, history = -1.0, None, 0, 0, []
    for epoch in range(1, epochs + 1):
        started = time.time()
        model.train()
        running, seen = 0.0, 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(images), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * len(targets)
            seen += len(targets)
        scheduler.step()
        val_logits, val_labels = predict_logits(model, val_loader, device)
        val_auroc = macro_ovr_auroc(val_labels, softmax(val_logits))
        history.append({"epoch": epoch, "train_loss": running / seen, "val_macro_auroc": val_auroc, "seconds": time.time() - started})
        print(f"epoch {epoch:2d}  loss {running / seen:.4f}  val macro AUROC {val_auroc:.4f}  ({time.time() - started:.1f}s)")
        if val_auroc > best_auroc + 1e-4:
            best_auroc, best_state, best_epoch, stale = val_auroc, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
            if stale >= cfg["imaging"]["patience"]:
                print(f"early stopping at epoch {epoch}; best epoch {best_epoch}")
                break
    model.load_state_dict(best_state)
    return model, history, best_epoch


@torch.no_grad()
def train_embeddings(model, frame: pd.DataFrame, cfg: dict, device) -> None:
    rows = frame[frame["split"] == "train"]
    loader = DataLoader(CTSliceDataset(rows, build_transforms(cfg, False, "none"), preprocessor(cfg)), batch_size=64)
    model.eval()
    vectors = torch.cat([embed(model, images.to(device)).float().cpu() for images, _ in loader]).numpy()
    np.savez_compressed(cfg_path(cfg, "embeddings_file"), vectors=vectors, paths=rows["path"].to_numpy(), labels=rows["label"].to_numpy())


def gradcam_examples(model, predictions: pd.DataFrame, cfg: dict, temperature: float, device) -> dict:
    out_dir = cfg_path(cfg, "gradcam_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()
    transform = build_transforms(cfg, False)
    classes = cfg["classes"]
    checks = []
    for row in predictions.itertuples():
        image = load_image(row.path)
        result = gradcam_overlay(image, model, transform, device, temperature, class_idx=int(row.pred_idx))
        checks.append({"path": row.path, "correct": bool(row.correct), **attention_checks(image, result["cam"], cfg["gradcam"]["border_fraction"])})
    check_frame = pd.DataFrame(checks)

    rng = np.random.default_rng(cfg["seed"])
    chosen = []
    correct = predictions[predictions["correct"]]
    per_class = max(1, cfg["gradcam"]["n_correct"] // len(classes))
    for k in range(len(classes)):
        pool = correct[correct["label_idx"] == k]
        if len(pool):
            chosen.append(pool.iloc[rng.choice(len(pool), size=min(per_class, len(pool)), replace=False)])
    wrong = predictions[~predictions["correct"]]
    if len(wrong):
        chosen.append(wrong.iloc[rng.choice(len(wrong), size=min(cfg["gradcam"]["n_wrong"], len(wrong)), replace=False)])
    gallery = []
    for n, row in enumerate(pd.concat(chosen).itertuples()):
        image = load_image(row.path)
        result = gradcam_overlay(image, model, transform, device, temperature, class_idx=int(row.pred_idx))
        size = result["overlay"].size
        panel = Image.new("RGB", (size[0] * 2, size[1]))
        panel.paste(image.convert("RGB").resize(size), (0, 0))
        panel.paste(result["overlay"], (size[0], 0))
        name = f"{n:02d}_{'correct' if row.correct else 'wrong'}_true-{classes[row.label_idx]}_pred-{classes[row.pred_idx]}.png"
        panel.save(out_dir / name)
        check = check_frame[check_frame["path"] == row.path].iloc[0].to_dict()
        gallery.append({
            "file": name,
            "source_image": row.path,
            "true": classes[row.label_idx],
            "pred": classes[row.pred_idx],
            "p_malignant": float(row.p_malignant),
            "correct": bool(row.correct),
            "border_mass": check["border_mass"],
            "outside_body_mass": check["outside_body_mass"],
            "peak_in_border": check["peak_in_border"],
        })
    summary = {
        "n_test_images_checked": int(len(check_frame)),
        "border_band_fraction_of_image_area": float(check_frame["border_area"].iloc[0]),
        "mean_border_mass": float(check_frame["border_mass"].mean()),
        "share_peak_in_border": float(check_frame["peak_in_border"].mean()),
        "mean_outside_body_mass": float(check_frame["outside_body_mass"].mean()),
        "n_correct_examples": int(sum(item["correct"] for item in gallery)),
        "n_wrong_examples": int(sum(not item["correct"] for item in gallery)),
        "n_wrong_available": int(len(wrong)),
        "examples": gallery,
    }
    write_json(cfg_path(cfg, "results_dir") / "gradcam_checks.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="short run for CPU-only machines")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--variant", choices=["main", "raw", "equalize"], default="main",
                        help="comparison runs with other preprocessing; metrics only, written to results/variants/<variant>")
    args = parser.parse_args()
    cfg = load_config()
    if read_source(cfg) != "B":
        raise SystemExit("This build implements the Source B (CT) imaging model only (see docs/DECISIONS.md).")
    seed_everything(cfg["seed"])
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    epochs = args.epochs or (cfg["imaging"]["fast_epochs"] if args.fast else cfg["imaging"]["epochs"])
    batch_size = 16 if args.fast else cfg["imaging"]["batch_size"]
    print(f"device {device}, epochs {epochs}, batch {batch_size}")

    frame = load_split_frame(cfg)
    classes = cfg["classes"]
    malignant = classes.index("malignant")
    results_dir, figures_dir = cfg_path(cfg, "results_dir"), cfg_path(cfg, "figures_dir")
    preprocess = cfg["imaging"]["preprocess"] if args.variant == "main" else {"raw": "none", "equalize": "equalize"}[args.variant]
    if args.variant != "main":
        results_dir = results_dir / "variants" / args.variant
        figures_dir = results_dir / "figures"

    model, history, best_epoch = train(cfg, frame, device, epochs, batch_size, preprocess)
    eval_transform, step = build_transforms(cfg, False, "none"), preprocessor(cfg, preprocess)
    loaders = {name: DataLoader(CTSliceDataset(frame[frame["split"] == name], eval_transform, step), batch_size=64) for name in ("val", "test")}
    val_logits, val_labels = predict_logits(model, loaders["val"], device)
    test_logits, test_labels = predict_logits(model, loaders["test"], device)

    temperature = fit_temperature(val_logits, val_labels)
    val_probs = softmax(val_logits, temperature)
    test_raw, test_probs = softmax(test_logits), softmax(test_logits, temperature)
    val_binary = (val_labels == malignant).astype(int)
    thresholds = {
        "youden_threshold": youden_threshold(val_binary, val_probs[:, malignant]),
        "rule_out_threshold": threshold_for_sensitivity(val_binary, val_probs[:, malignant], cfg["eval"]["rule_out_sensitivity"]),
        "rule_in_threshold": threshold_for_specificity(val_binary, val_probs[:, malignant], cfg["eval"]["rule_in_specificity"]),
        "rule_out_target_sensitivity": cfg["eval"]["rule_out_sensitivity"],
        "rule_in_target_specificity": cfg["eval"]["rule_in_specificity"],
        "fit_on": "validation split, calibrated malignant probability",
    }

    test_rows = frame[frame["split"] == "test"].reset_index(drop=True)
    groups = test_rows["group"].to_numpy()
    report = full_report(test_labels, test_probs, groups, malignant, thresholds["youden_threshold"], cfg["eval"]["bootstrap"], cfg["seed"],
                         cfg["eval"]["fixed_specificity"])
    report["per_class_recall"] = {classes[int(k)]: v for k, v in report["per_class_recall"].items()}
    test_binary = (test_labels == malignant).astype(int)
    test_pred = test_probs.argmax(axis=1)
    metrics = {
        "model": cfg["imaging"]["backbone"],
        "task": "malignant vs (normal + benign); 3-class head",
        "variant": args.variant,
        "preprocess": preprocess,
        "device": device.type,
        "fast_mode": args.fast,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_val_macro_auroc": max(row["val_macro_auroc"] for row in history),
        "temperature": temperature,
        "n_images": {name: int((frame["split"] == name).sum()) for name in ("train", "val", "test")},
        "n_groups": {name: int(frame.loc[frame["split"] == name, "group"].nunique()) for name in ("train", "val", "test")},
        "test_positive_images": int(test_binary.sum()),
        "bootstrap": {"resamples": cfg["eval"]["bootstrap"], "unit": "pseudo-patient group", "interval": "percentile 95%"},
        "thresholds": thresholds,
        "test": report,
        "test_nll": {"before_scaling": float(log_loss(test_labels, test_raw, labels=range(len(classes)))),
                     "after_scaling": float(log_loss(test_labels, test_probs, labels=range(len(classes))))},
        "test_ece_malignant": {"before_scaling": expected_calibration_error(test_binary, test_raw[:, malignant]),
                               "after_scaling": report["ece_positive_class"]},
        "confusion_matrix": {"labels": classes, "counts": confusion_matrix(test_labels, test_pred, labels=range(len(classes))).tolist()},
    }
    write_json(results_dir / "imaging_metrics.json", metrics)
    write_json(results_dir / "training_history.json", history)
    write_json(results_dir / "thresholds.json", thresholds)

    predictions = test_rows[["path", "group", "label", "label_idx"]].copy()
    for k, name in enumerate(classes):
        predictions[f"p_{name}"] = test_probs[:, k]
    predictions["pred_idx"] = test_pred
    predictions["pred"] = [classes[k] for k in test_pred]
    predictions["correct"] = predictions["pred_idx"] == predictions["label_idx"]
    predictions.to_csv(results_dir / "test_predictions.csv", index=False)

    plots.roc_plot(test_binary, test_probs[:, malignant], figures_dir / "roc.png", "Test ROC: malignant vs rest")
    plots.pr_plot(test_binary, test_probs[:, malignant], figures_dir / "pr.png", "Test PR: malignant vs rest")
    plots.calibration_plot(test_binary, test_raw[:, malignant], test_probs[:, malignant], figures_dir / "calibration.png", "Test calibration (malignant)")
    plots.confusion_plot(test_labels, test_pred, classes, figures_dir / "confusion_matrix.png", "Test confusion matrix")
    plots.training_curve_plot(history, figures_dir / "training_curve.png")
    print(f"[{args.variant}] temperature {temperature:.3f}")
    for key in ("auroc", "auprc", "sensitivity", "specificity", "sens_at_90_spec", "macro_ovr_auroc_3class"):
        item = report[key]
        print(f"[{args.variant}] test {key}: {item['value']:.3f} (95% CI {item['ci_low']}-{item['ci_high']}, n_valid {item['n_valid']})")
    print(f"[{args.variant}] per-class recall: { {k: round(v['value'], 3) for k, v in report['per_class_recall'].items()} }")
    if args.variant != "main":
        return

    model_file = cfg_path(cfg, "model_file")
    model_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "backbone": cfg["imaging"]["backbone"],
        "classes": classes,
        "temperature": temperature,
        "image_size": cfg["imaging"]["image_size"],
        "best_epoch": best_epoch,
    }, model_file)
    train_embeddings(model, frame, cfg, device)
    gradcam = gradcam_examples(model, predictions, cfg, temperature, device)
    shortcut = {name: shortcut_baseline(frame, "malignant", cfg["eval"]["bootstrap"], cfg["seed"], name) for name in PREPROCESS_CHOICES}
    write_json(results_dir / "shortcut_baseline.json", shortcut)
    for name, item in shortcut.items():
        print(f"shortcut baseline on {name} (intensity histogram only) test AUROC: {item['test_auroc']['value']:.3f} "
              f"({item['test_auroc']['ci_low']}-{item['test_auroc']['ci_high']})")
    print(f"thresholds {thresholds}")
    print(f"Grad-CAM: mean border mass {gradcam['mean_border_mass']:.3f} vs border area {gradcam['border_band_fraction_of_image_area']:.3f}; "
          f"peak in border {gradcam['share_peak_in_border']:.3f}; wrong examples {gradcam['n_wrong_examples']} of {gradcam['n_wrong_available']} available")


if __name__ == "__main__":
    main()
