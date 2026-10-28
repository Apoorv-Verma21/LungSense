"""Light learning-rate tuning on the validation split only (the test split is never read).

Trains the main configuration at each learning rate in `imaging.lr_grid` and records the best
validation macro one-vs-rest AUROC and its epoch in results/lr_sweep.json. Copy the winner to
`imaging.lr` in configs/config.yaml before running scripts/03_train_imaging.py.
"""
from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.data.dataset import load_split_frame
from src.utils.common import cfg_path, load_config, seed_everything, write_json

spec = importlib.util.spec_from_file_location("train_imaging", ROOT / "scripts" / "03_train_imaging.py")
train_imaging = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_imaging)


def main() -> None:
    cfg = load_config()
    frame = load_split_frame(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for lr in cfg["imaging"]["lr_grid"]:
        seed_everything(cfg["seed"])
        trial = copy.deepcopy(cfg)
        trial["imaging"]["lr"] = lr
        _, history, best_epoch = train_imaging.train(trial, frame, device, cfg["imaging"]["epochs"], cfg["imaging"]["batch_size"], cfg["imaging"]["preprocess"])
        best = max(row["val_macro_auroc"] for row in history)
        rows.append({"lr": lr, "best_val_macro_auroc": best, "best_epoch": best_epoch, "epochs_run": len(history)})
        print(f"lr {lr}: best val macro AUROC {best:.4f} at epoch {best_epoch}")
    winner = max(rows, key=lambda row: row["best_val_macro_auroc"])
    write_json(cfg_path(cfg, "results_dir") / "lr_sweep.json", {"selection": "best validation macro AUROC", "trials": rows, "selected_lr": winner["lr"]})
    print(f"selected lr {winner['lr']}")


if __name__ == "__main__":
    main()
