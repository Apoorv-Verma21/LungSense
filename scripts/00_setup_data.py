"""Choose the data source, download it, and summarise what was downloaded.

Source A (MIMIC-IV + MIMIC-CXR-JPG) is used only when PHYSIONET_USER / PHYSIONET_PASS are
set and an authenticated request succeeds. Otherwise Source B (IQ-OTH/NCCD CT slices on
Kaggle) is used. Credentials are read from the environment / ~/.kaggle/kaggle.json and never
printed. Outputs: data/SOURCE.txt, data/raw/, data/image_index.csv, data/cohort_summary.json.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import zipfile
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import requests
from PIL import Image

from src.data.dataset import IMAGE_SUFFIXES, label_from_folder
from src.utils.common import cfg_path, load_config, write_json


def source_a_available(cfg: dict) -> tuple[bool, str]:
    user, password = os.environ.get("PHYSIONET_USER"), os.environ.get("PHYSIONET_PASS")
    if not user or not password:
        return False, "PHYSIONET_USER / PHYSIONET_PASS not set"
    try:
        response = requests.get(cfg["source"]["physionet_check_url"], auth=(user, password), timeout=30)
    except requests.RequestException as exc:
        return False, f"PhysioNet request failed: {type(exc).__name__}"
    if response.status_code != 200:
        return False, f"PhysioNet returned HTTP {response.status_code}"
    return True, "authenticated request to PhysioNet returned HTTP 200"


def has_kaggle_credentials() -> bool:
    has_file = (Path.home() / ".kaggle" / "kaggle.json").exists()
    return has_file or bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))


def download_with_kaggle_api(cfg: dict, raw_dir: Path) -> dict:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    slug = cfg["source"]["kaggle_slug"]
    try:
        api.dataset_download_files(slug, path=str(raw_dir), unzip=False, quiet=False)
    except Exception as exc:  # noqa: BLE001 - Kaggle raises several client error types
        print(f"Download of {slug} failed ({type(exc).__name__}); searching Kaggle for '{cfg['source']['kaggle_search']}'")
        candidates = api.dataset_list(search=cfg["source"]["kaggle_search"])
        if not candidates:
            raise SystemExit("No IQ-OTH/NCCD dataset found on Kaggle; stopping (see docs/DECISIONS.md).")
        slug = str(candidates[0].ref)
        api.dataset_download_files(slug, path=str(raw_dir), unzip=False, quiet=False)
    archive = sorted(raw_dir.glob("*.zip"))[0]
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(raw_dir / "extracted")
    return {"method": "kaggle API (authenticated)", "slug": slug, "data_root": (raw_dir / "extracted").relative_to(ROOT).as_posix()}


def download_with_kagglehub(cfg: dict, raw_dir: Path) -> dict:
    """Kaggle's official kagglehub client downloads public datasets without credentials."""
    os.environ["KAGGLEHUB_CACHE"] = str(raw_dir / "kagglehub")
    import kagglehub

    slug = cfg["source"]["kaggle_slug"]
    root = Path(kagglehub.dataset_download(slug))
    return {"method": "kagglehub (anonymous, public dataset)", "slug": slug, "kaggle_version": root.name,
            "data_root": root.resolve().relative_to(ROOT).as_posix()}


def existing_data_root(raw_dir: Path) -> Path | None:
    for candidate in (raw_dir / "extracted", *sorted((raw_dir / "kagglehub").glob("datasets/*/*/versions/*"))):
        if candidate.is_dir():
            return candidate
    return None


def download_source_b(cfg: dict, raw_dir: Path) -> dict:
    raw_dir.mkdir(parents=True, exist_ok=True)
    if has_kaggle_credentials():
        return download_with_kaggle_api(cfg, raw_dir)
    print("No Kaggle credentials found; using kagglehub's anonymous download of the public dataset.")
    return download_with_kagglehub(cfg, raw_dir)


def fingerprint(index: pd.DataFrame) -> str:
    """SHA256 over sorted (relative path, file hash) pairs: identifies the exact image set used."""
    digest = hashlib.sha256()
    for path, file_hash in sorted(zip(index["path"], index["sha256"])):
        digest.update(f"{path}\t{file_hash}\n".encode())
    return digest.hexdigest()


def index_images(data_root: Path) -> tuple[pd.DataFrame, list[str]]:
    rows, ignored = [], []
    for file in sorted(data_root.rglob("*")):
        if not file.is_file() or file.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = label_from_folder(file.parent.name)
        relative = file.relative_to(ROOT).as_posix()
        if label is None:
            ignored.append(relative)
            continue
        with Image.open(file) as image:
            width, height = image.size
            mode = image.mode
        rows.append({
            "path": relative,
            "label": label,
            "folder": file.parent.name,
            "width": width,
            "height": height,
            "mode": mode,
            "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
        })
    return pd.DataFrame(rows), ignored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-download", action="store_true", help="reuse data/raw if it is already present")
    args = parser.parse_args()
    cfg = load_config()
    raw_dir = cfg_path(cfg, "raw_dir")

    available, reason = source_a_available(cfg)
    print(f"Source A (MIMIC-IV + MIMIC-CXR-JPG): {'available' if available else 'unavailable'} - {reason}")
    if available:
        raise SystemExit(
            "Source A is available, but this build implements only Source B because Source A was "
            "unavailable when it was built (see docs/DECISIONS.md). Unset PHYSIONET_USER/PHYSIONET_PASS to run Source B."
        )

    data_root = existing_data_root(raw_dir) if args.skip_download else None
    if data_root is not None:
        method = "kagglehub (anonymous, public dataset); local cache reused" if "kagglehub" in data_root.parts \
            else "kaggle API (authenticated); local copy reused"
        download = {"method": method, "slug": cfg["source"]["kaggle_slug"], "data_root": data_root.relative_to(ROOT).as_posix()}
        if "versions" in data_root.parts:
            download["kaggle_version"] = data_root.name
    else:
        download = download_source_b(cfg, raw_dir)
        data_root = ROOT / download["data_root"]

    index, ignored = index_images(data_root)
    if index.empty:
        raise SystemExit("No labelled images found after extraction; stopping.")
    duplicates = index[index.duplicated("sha256", keep=False)]
    cross_class = duplicates.groupby("sha256")["label"].nunique()
    index_file = cfg_path(cfg, "image_index")
    index.to_csv(index_file, index=False)

    summary = {
        "source": "B",
        "dataset": "IQ-OTH/NCCD lung cancer dataset (Kaggle mirror)",
        "access_date": date.today().isoformat(),
        "download": download,
        "image_set_sha256": fingerprint(index),
        "n_images": int(len(index)),
        "class_counts": {k: int(v) for k, v in Counter(index["label"]).items()},
        "folders": {k: int(v) for k, v in Counter(index["folder"]).items()},
        "ignored_unlabelled_images": len(ignored),
        "ignored_folders": sorted({Path(p).parent.name for p in ignored}),
        "image_sizes": {f"{w}x{h}": int(n) for (w, h), n in index.groupby(["width", "height"]).size().items()},
        "image_modes": {k: int(v) for k, v in Counter(index["mode"]).items()},
        "exact_duplicate_images": int(len(duplicates)),
        "exact_duplicate_sets": int(duplicates["sha256"].nunique()),
        "duplicate_sets_spanning_classes": int((cross_class > 1).sum()),
    }
    write_json(cfg_path(cfg, "cohort_summary"), summary)
    write_json(cfg_path(cfg, "results_dir") / "cohort_summary.json", summary)
    cfg_path(cfg, "source_file").write_text(
        f"B\nIQ-OTH/NCCD lung cancer CT dataset via Kaggle ({download['slug']}, version {download.get('kaggle_version', 'n/a')}, {download['method']})\n"
        f"Source A unavailable: {reason}\n",
        encoding="utf-8",
    )
    print(f"Images: {summary['n_images']}  classes: {summary['class_counts']}")
    print(f"Ignored unlabelled images: {len(ignored)} in {summary['ignored_folders']}")
    print(f"Exact duplicate images: {summary['exact_duplicate_images']} in {summary['exact_duplicate_sets']} sets "
          f"({summary['duplicate_sets_spanning_classes']} sets span classes)")
    print(f"Image sizes: {summary['image_sizes']}")


if __name__ == "__main__":
    main()
