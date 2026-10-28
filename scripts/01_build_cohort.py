"""Build the cohort and the pseudo-patient-level train/val/test split (Source B).

The IQ-OTH/NCCD release has no patient identifiers, so slices are grouped into
pseudo-patients before splitting: within each class, images are ordered by the case number in
the file name and neighbouring images are linked when their 64x64 thumbnails correlate at or
above a per-class threshold (the highest grid value whose group count does not exceed the
source's stated cases for that class); near-identical images (correlation >= 0.97 or same file
hash) are always linked. Images that are not the standard size are excluded first. Connected components are the groups. Whole groups are assigned to splits,
stratified by class, seed 42. Outputs: data/splits.json, results/cohort_flow.json,
results/grouping_diagnostics.json, results/leakage_audit.json, results/figures/grouping_correlation.png.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.dataset import SPLITS, load_image
from src.utils.common import cfg_path, load_config, read_source, write_json

NEAR_DUPLICATE = 0.97


def case_number(path: str) -> int:
    match = re.search(r"\((\d+)\)", Path(path).name)
    return int(match.group(1)) if match else -1


def thumbnails(paths: list[str], size: int) -> np.ndarray:
    vectors = np.stack([np.asarray(load_image(p).resize((size, size)), dtype=np.float32).ravel() for p in paths])
    vectors -= vectors.mean(axis=1, keepdims=True)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
    return vectors


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)


def build_groups(index: pd.DataFrame, vectors: np.ndarray, threshold: float | dict) -> tuple[np.ndarray, list[float]]:
    """`threshold` is one value for all classes or a {class: threshold} mapping."""
    uf = UnionFind(len(index))
    consecutive = []
    for label, rows in index.groupby("label"):
        class_threshold = threshold[label] if isinstance(threshold, dict) else threshold
        order = rows.sort_values(["case_number", "path"]).index.to_numpy()
        for a, b in zip(order[:-1], order[1:]):
            similarity = float(vectors[a] @ vectors[b])
            consecutive.append(similarity)
            if similarity >= class_threshold:
                uf.union(a, b)
        sims = vectors[order] @ vectors[order].T
        for i, j in zip(*np.where(np.triu(sims, k=1) >= NEAR_DUPLICATE)):
            uf.union(order[i], order[j])
        for _, same in rows.groupby("sha256"):
            members = same.index.to_numpy()
            for other in members[1:]:
                uf.union(members[0], other)
    roots = np.array([uf.find(i) for i in range(len(index))])
    _, group_ids = np.unique(roots, return_inverse=True)
    return group_ids, consecutive


def choose_thresholds(index: pd.DataFrame, vectors: np.ndarray, cfg: dict) -> dict[str, float]:
    """Per class, the highest link threshold whose group count does not exceed the source's stated
    number of cases for that class. More groups than patients would guarantee split patients."""
    grid = np.round(np.arange(cfg["grouping"]["threshold_min"], cfg["grouping"]["threshold_max"] + 1e-9, cfg["grouping"]["threshold_step"]), 3)
    documented = cfg["source"]["official"]["documented_cases_per_class"]
    chosen = {}
    for label, rows in index.groupby("label"):
        subset = rows.reset_index(drop=True)
        sub_vectors = vectors[rows.index.to_numpy()]
        best = float(grid[0])
        for threshold in grid:
            groups, _ = build_groups(subset, sub_vectors, float(threshold))
            if len(np.unique(groups)) <= documented[label]:
                best = float(threshold)
        chosen[label] = best
    return chosen


def assign_splits(index: pd.DataFrame, ratios: dict, seed: int) -> pd.Series:
    """Greedy group assignment per class towards the target image fractions."""
    rng = np.random.default_rng(seed)
    split_of_group: dict[int, str] = {}
    for _, rows in index.groupby("label"):
        sizes = rows.groupby("group").size()
        groups = sizes.index.to_numpy().copy()
        rng.shuffle(groups)
        total = sizes.sum()
        filled = {name: 0 for name in SPLITS}
        # Seed val and test with one group each so every class appears in every split.
        for name, group in zip(("test", "val"), groups[:2]):
            split_of_group[group] = name
            filled[name] += sizes[group]
        for group in groups[2:]:
            deficits = {name: ratios[name] * total - filled[name] for name in SPLITS}
            name = max(deficits, key=deficits.get)
            split_of_group[group] = name
            filled[name] += sizes[group]
    return index["group"].map(split_of_group)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=None, help="override grouping.link_threshold")
    args = parser.parse_args()
    cfg = load_config()
    if read_source(cfg) != "B":
        raise SystemExit("This build implements the Source B cohort only (see docs/DECISIONS.md).")
    threshold = args.threshold  # None: choose per class (see choose_thresholds)
    results_dir = cfg_path(cfg, "results_dir")

    index = pd.read_csv(cfg_path(cfg, "image_index"))
    n_found = len(index)
    # Exclude identical files that carry different labels: the true label cannot be known.
    conflicting = index.groupby("sha256")["label"].transform("nunique") > 1
    excluded_conflicts = index[conflicting]
    index = index[~conflicting]
    width, height = cfg["cohort"]["required_size"]
    off_size = (index["width"] != width) | (index["height"] != height)
    excluded_size = index[off_size]
    index = index[~off_size].reset_index(drop=True)
    index["case_number"] = index["path"].map(case_number)

    vectors = thumbnails(index["path"].tolist(), cfg["grouping"]["thumb_size"])
    if threshold is None:
        threshold = choose_thresholds(index, vectors, cfg)
    group_ids, consecutive = build_groups(index, vectors, threshold)
    index["group"] = [f"{label}-{gid:04d}" for label, gid in zip(index["label"], group_ids)]
    index["split"] = assign_splits(index, cfg["split"], cfg["seed"])

    # Residual leakage audit: nearest same-class training image for each validation/test image.
    per_class = threshold if isinstance(threshold, dict) else {label: threshold for label in index["label"].unique()}
    audit = {}
    for name in ("val", "test"):
        nearest, limits = [], []
        for label, rows in index[index["split"] == name].groupby("label"):
            train_rows = index[(index["split"] == "train") & (index["label"] == label)].index.to_numpy()
            nearest.extend((vectors[rows.index.to_numpy()] @ vectors[train_rows].T).max(axis=1))
            limits.extend([per_class[label]] * len(rows))
        nearest, limits = np.array(nearest), np.array(limits)
        audit[name] = {
            "n_images": int(len(nearest)),
            "max_train_similarity_quantiles": {q: float(np.quantile(nearest, float(q))) for q in ("0.5", "0.9", "0.99", "1.0")},
            "n_at_or_above_link_threshold": int((nearest >= limits).sum()),
            "n_near_duplicate_of_train": int((nearest >= NEAR_DUPLICATE).sum()),
        }
    groups_by_split = {name: set(index.loc[index["split"] == name, "group"]) for name in SPLITS}
    overlap = {f"{a}&{b}": len(groups_by_split[a] & groups_by_split[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    assert not any(overlap.values()), f"group overlap between splits: {overlap}"

    consecutive_arr = np.array(consecutive)
    diagnostics = {
        "method": "within-class consecutive case-number linking on 64x64 thumbnail correlation, plus near-duplicate and identical-file linking; "
                  "per-class threshold = highest grid value whose group count does not exceed the source's stated cases",
        "link_threshold": per_class,
        "near_duplicate_threshold": NEAR_DUPLICATE,
        "n_groups": int(index["group"].nunique()),
        "groups_per_class": {k: int(v) for k, v in index.groupby("label")["group"].nunique().items()},
        "documented_cases_per_class": cfg["source"]["official"]["documented_cases_per_class"],
        "group_size_quantiles": {q: float(np.quantile(index.groupby("group").size(), float(q))) for q in ("0.0", "0.5", "0.9", "1.0")},
        "consecutive_similarity_quantiles": {q: float(np.quantile(consecutive_arr, float(q))) for q in ("0.05", "0.25", "0.5", "0.75", "0.95")},
    }
    write_json(results_dir / "grouping_diagnostics.json", diagnostics)
    write_json(results_dir / "leakage_audit.json", {"group_overlap_between_splits": overlap, **audit})

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.hist(consecutive_arr, bins=50, color="tab:blue")
    for (label, value), color in zip(sorted(per_class.items()), ("tab:red", "tab:orange", "tab:green")):
        ax.axvline(value, color=color, ls="--", label=f"{label} threshold {value}")
    ax.set(xlabel="Thumbnail correlation of consecutive images", ylabel="Image pairs", title="Pseudo-patient grouping")
    ax.legend()
    fig.tight_layout()
    (results_dir / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(results_dir / "figures" / "grouping_correlation.png", dpi=130)
    plt.close(fig)

    split_counts = {
        name: {
            "images": int((index["split"] == name).sum()),
            "groups": int(index.loc[index["split"] == name, "group"].nunique()),
            "images_per_class": {k: int(v) for k, v in index[index["split"] == name]["label"].value_counts().items()},
            "groups_per_class": {k: int(v) for k, v in index[index["split"] == name].groupby("label")["group"].nunique().items()},
        }
        for name in SPLITS
    }
    flow = {
        "labelled_images_found": int(n_found),
        "excluded_identical_files_with_conflicting_labels": int(len(excluded_conflicts)),
        "excluded_non_standard_size": int(len(excluded_size)),
        "excluded_non_standard_size_per_class": {k: int(v) for k, v in excluded_size["label"].value_counts().items()},
        "excluded_non_standard_size_dimensions": {f"{w}x{h}": int(n) for (w, h), n in excluded_size.groupby(["width", "height"]).size().items()},
        "included_images": int(len(index)),
        "included_images_per_class": {k: int(v) for k, v in index["label"].value_counts().items()},
        "pseudo_patient_groups": int(index["group"].nunique()),
        "splits": split_counts,
    }
    write_json(results_dir / "cohort_flow.json", flow)
    write_json(cfg_path(cfg, "splits"), {
        "seed": cfg["seed"],
        "level": "pseudo-patient group (no patient IDs in source)",
        "ratios": cfg["split"],
        "images": index[["path", "label", "group", "split"]].to_dict(orient="records"),
    })

    print(f"Included {len(index)} of {n_found} images; excluded {len(excluded_conflicts)} conflicting duplicates, "
          f"{len(excluded_size)} non-standard-size images {flow['excluded_non_standard_size_per_class']}")
    print(f"Groups per class: {diagnostics['groups_per_class']} (documented cases: {diagnostics['documented_cases_per_class']})")
    print(f"Consecutive similarity quantiles: {diagnostics['consecutive_similarity_quantiles']}")
    for name in SPLITS:
        print(f"{name:5s}: {split_counts[name]['images']} images, {split_counts[name]['groups']} groups, {split_counts[name]['images_per_class']}")
    print(f"Leakage audit: {audit}")


if __name__ == "__main__":
    main()
