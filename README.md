# LungSense

> **Research/education demo. Not a medical device. Not for clinical use.**

LungSense classifies single chest CT slices as **normal, benign or malignant**, reports a calibrated
probability of malignancy with a Low / Indeterminate / High risk band, and shows a Grad-CAM heatmap
of where the model looked. A Streamlit dashboard presents predictions, test-set performance,
explanations and the model card.

It uses **real data from a single public source** (the IQ-OTH/NCCD lung cancer CT dataset) and is
**imaging-only**: the source has no clinical variables, so there is no clinical model and no fusion.
The dashboard says so on those pages rather than filling the gap with other data.

## Architecture

```
Kaggle mirror of IQ-OTH/NCCD  ->  00_setup_data   (provenance, download, image index)
                              ->  01_build_cohort (pseudo-patient groups, 70/15/15 split, leakage audit)
                              ->  02_train_clinical (records "not available": no clinical data)
                              ->  03_train_imaging (EfficientNet-B0, temperature scaling, metrics, Grad-CAM)
                              ->  04_fusion_eval   (records "single modality only")
                              ->  05_render_docs   (README / model card results from results/)
                              ->  app/streamlit_app.py
```

```
configs/config.yaml   all settings (seed, split, training, thresholds, quoted source facts)
scripts/              pipeline steps 00-05 and run_all.sh
src/data              image index, splits, transforms, dataset
src/models            model construction, loading, temperature scaling
src/explain           Grad-CAM and shortcut checks
src/eval              metrics with group bootstrap CIs, figures
app/                  Streamlit dashboard
results/              metrics, figures and Grad-CAM examples written by the scripts
docs/                 DECISIONS.md, DATA_SOURCES.md, MODEL_CARD.md, screenshots
tests/                pytest smoke tests
```

## Run

Requirements: Python 3.12 and about 6 GB free disk. A Kaggle token is optional: without one,
`00_setup_data.py` downloads the public dataset anonymously with Kaggle's `kagglehub` client; with
`~/.kaggle/kaggle.json` it uses the authenticated Kaggle API. An NVIDIA GPU is used if present;
otherwise pass `--fast` for a short CPU run.

```bash
python -m venv .venv
source .venv/Scripts/activate        # Windows Git Bash; use .venv/bin/activate on Linux/macOS
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
python scripts/00_setup_data.py      # downloads the dataset (first run only)
bash scripts/run_all.sh              # cohort, training (main + comparison runs), evaluation, docs, tests
python scripts/03a_lr_sweep.py       # optional: the validation-only learning-rate sweep
streamlit run app/streamlit_app.py
```

## Data source

IQ-OTH/NCCD lung cancer dataset, Iraq-Oncology Teaching Hospital / National Center for Cancer
Diseases, collected in fall 2019. Official record: https://data.mendeley.com/datasets/bhmdr45bh2/4
(Mendeley Data V4, DOI 10.17632/bhmdr45bh2.4), licence CC BY 4.0. Downloaded from the Kaggle mirror
https://www.kaggle.com/datasets/adityamahimkar/iqothnccd-lung-cancer-dataset. Full provenance, access
date and checksum: [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).

## Results

Generated from `results/` by `scripts/05_render_docs.py`; do not edit by hand.

<!-- GENERATED:results:START -->
Test split: 166 images in 16 pseudo-patient groups (train 722, validation 148; 1036 images included in total). 95% CIs from 1000 group-level bootstrap resamples.

| Metric | Value (95% CI) |
|---|---|
| AUROC, malignant vs rest | 1.000 (0.999-1.000) |
| AUPRC, malignant vs rest | 1.000 (0.999-1.000) |
| Sensitivity (validation Youden threshold) | 1.000 (1.000-1.000) |
| Specificity (validation Youden threshold) | 0.950 (0.815-1.000) |
| Sensitivity at 0.90 specificity | 1.000 (1.000-1.000) |
| Macro one-vs-rest AUROC, three classes | 0.921 (0.805-1.000) |
| Recall, normal | 0.590 (0.282-0.783) |
| Recall, benign | 0.263 (0.000-0.333) |
| Recall, malignant | 1.000 (1.000-1.000) |

Calibration: temperature 1.470; malignant-class ECE 0.074 before and 0.106 after scaling. Best epoch 5 of 13 run on cuda; input preprocessing `body_mask_equalize`.

**Shortcut warning:** a logistic regression on intensity histograms alone (no spatial information) reaches malignant AUROC 0.995 (0.978-1.000) on `none` input; malignant AUROC 0.778 (0.546-0.950) on `equalize` input; malignant AUROC 0.822 (0.629-0.949) on `body_mask_equalize` input. The near-perfect malignant-vs-rest numbers therefore mostly reflect how the dataset was acquired and exported, not demonstrated cancer detection. See the model card.
<!-- GENERATED:results:END -->

Ablation: [results/ablation_table.md](results/ablation_table.md) (single modality only).
Figures: [results/figures](results/figures). Grad-CAM gallery: [results/gradcam](results/gradcam).

## Screenshots

Taken from the running app: [docs/screenshots](docs/screenshots).

## Limitations

- **Acquisition shortcuts in the data (most important).** On this dataset, malignant slices can be
  told apart from the rest using intensity statistics alone, and an early model looked at the
  scanner table rather than the lungs. Non-standard image sizes (all malignant) are excluded and
  the main model only sees the masked, equalised body, but the near-perfect malignant-vs-rest
  scores still mostly reflect how the data was collected. They are not evidence of cancer
  detection. Details and the shortcut baseline are in the model card.
- **Small, single-centre dataset.** One hospital, one scanner family, a few months of collection;
  performance will not transfer to other sites without validation.
- **Possible slice-level leakage.** The source has no patient IDs. Slices are grouped into
  pseudo-patients by image similarity before splitting, but slices of one patient may still land in
  different splits, which would inflate test metrics. The residual risk is measured in
  `results/leakage_audit.json` and discussed in the model card.
- **Imaging only, single 2D slices.** No clinical variables, no 3D context, no prior scans.
- **Labels.** Labels are the source's case classes assigned by its oncologists and radiologists;
  in this release they are not linked to biopsy or follow-up outcomes.
- **No demographic subgroup analysis.** The source provides no sex or age per image.
- **Not a screening or diagnostic tool.** See the disclaimer above.

## Licence

Code: for research and education. Data: CC BY 4.0, cite alyasriy, hamdalla; AL-Huseiny, Muayed
(2023), "The IQ-OTH/NCCD lung cancer dataset", Mendeley Data, V4, doi: 10.17632/bhmdr45bh2.4.

> **Research/education demo. Not a medical device. Not for clinical use.**
