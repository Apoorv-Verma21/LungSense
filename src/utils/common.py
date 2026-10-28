"""Shared helpers: repository paths, config loading, seeding and JSON IO."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path | None = None) -> dict:
    with open(config_path or ROOT / "configs" / "config.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def cfg_path(cfg: dict, key: str) -> Path:
    return ROOT / cfg["paths"][key]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _to_builtin(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Not JSON serialisable: {type(value)!r}")


def write_json(file_path: str | Path, obj: Any) -> None:
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(obj, indent=2, default=_to_builtin), encoding="utf-8")


def read_json(file_path: str | Path) -> Any:
    return json.loads(Path(file_path).read_text(encoding="utf-8"))


def read_source(cfg: dict) -> str:
    """Return the chosen data source letter ('A' or 'B') written by scripts/00_setup_data.py."""
    source_file = cfg_path(cfg, "source_file")
    if not source_file.exists():
        raise SystemExit(f"{source_file} is missing - run scripts/00_setup_data.py first.")
    return source_file.read_text(encoding="utf-8").splitlines()[0].strip()
