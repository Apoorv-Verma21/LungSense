"""Image index, split loading and PyTorch dataset for the CT slice data."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

from src.utils.common import ROOT, cfg_path, read_json

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


def label_from_folder(folder_name: str) -> str | None:
    """Map a source folder name to a class. The source spells benign as 'Bengin'."""
    name = folder_name.lower()
    if "malignant" in name:
        return "malignant"
    if "bengin" in name or "benign" in name:
        return "benign"
    if "normal" in name:
        return "normal"
    return None


def load_image(relative_path: str | Path) -> Image.Image:
    return Image.open(ROOT / relative_path).convert("L")


def load_split_frame(cfg: dict) -> pd.DataFrame:
    """Return one row per image with columns path, label, label_idx, group, split."""
    splits_file = cfg_path(cfg, "splits")
    if not splits_file.exists():
        raise SystemExit(f"{splits_file} is missing - run scripts/01_build_cohort.py first.")
    payload = read_json(splits_file)
    frame = pd.DataFrame(payload["images"])
    frame["label_idx"] = frame["label"].map({name: idx for idx, name in enumerate(cfg["classes"])})
    return frame


class Equalize:
    """Per-image histogram equalisation: removes global intensity/windowing cues from the input."""

    def __call__(self, image: Image.Image) -> Image.Image:
        return ImageOps.equalize(image.convert("L"))


PREPROCESS_CHOICES = ("none", "equalize", "body_mask_equalize")


def preprocessor(cfg: dict, preprocess: str | None = None):
    """Deterministic per-image step (cacheable); `preprocess` defaults to `imaging.preprocess`."""
    from src.data.preprocess import BodyMaskEqualize

    preprocess = cfg["imaging"]["preprocess"] if preprocess is None else preprocess
    return {"none": None, "equalize": Equalize(), "body_mask_equalize": BodyMaskEqualize()}[preprocess]


def build_transforms(cfg: dict, train: bool, preprocess: str | None = None):
    """Full model input pipeline for one PIL slice, including the deterministic preprocessing."""
    from torchvision import transforms

    image_size = cfg["imaging"]["image_size"]
    step = preprocessor(cfg, preprocess)
    prefix = [step] if step is not None else []
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if train:
        return transforms.Compose([
            *prefix,
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        *prefix,
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        normalize,
    ])


class CTSliceDataset:
    """Holds decoded (and, if given, deterministically preprocessed) slices in memory.

    Pass a `transform` built with preprocess="none" together with `preprocess_fn`, so the
    expensive deterministic step runs once per image instead of once per epoch.
    """

    def __init__(self, frame: pd.DataFrame, transform, preprocess_fn=None):
        self.paths = frame["path"].tolist()
        self.labels = frame["label_idx"].astype(int).tolist()
        images = [load_image(p) for p in self.paths]
        self.images = [preprocess_fn(image) for image in images] if preprocess_fn is not None else images
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        return self.transform(self.images[index]), self.labels[index]
