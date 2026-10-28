# Decisions log

Every non-obvious choice made while building LungSense, in order. Results quoted elsewhere are
generated from `results/`; this log records choices, not results.

## Setup

1. **Source B, not Source A.** `PHYSIONET_USER` / `PHYSIONET_PASS` were not set on the build
   machine, so the authenticated PhysioNet check cannot succeed. Even with access, MIMIC-IV
   `hosp` plus the cohort's MIMIC-CXR-JPG files would not fit in the roughly 10 GB of free disk,
   and PhysioNet's rules forbid passing credentialed data to online services, which includes the
   AI assistant that built this project reading rows. `scripts/00_setup_data.py` still performs
   the Source A check programmatically and records the reason in `data/SOURCE.txt`.
2. **Source A code paths are not implemented.** Writing a MIMIC pipeline that cannot be run or
   verified here would ship untested code. Scripts 01-04 stop with a clear message if the source
   is not B, and 02 / 04 write explicit "not available" outputs under Source B.
3. **Provenance verified on the official page before use.** The Kaggle page needs a login, so the
   provenance check uses the dataset's official Mendeley Data record (V4, DOI
   10.17632/bhmdr45bh2.4, CC BY 4.0), which states real, IRB-approved, de-identified CT scans.
   Quoted facts live in `configs/config.yaml` (`source.official`) so they are never retyped.
   Recorded in `docs/DATA_SOURCES.md`.
4. **Project location** `C:\Users\HP\lungsense`, outside OneDrive, so the data, model weights and
   virtual environment are not synced to the cloud. Its own git repository; commits are local only.
5. **Python 3.12 virtual environment, PyTorch 2.6.0 with CUDA 12.4 wheels** for the RTX 4060
   laptop GPU; packages installed without a pip cache to save disk.

## Data and splits

6. **Labels from folder names.** The source spells benign as "Bengin"; `label_from_folder` maps
   both spellings. Images in folders without a class name are ignored and counted in
   `data/cohort_summary.json`.
7. **Binary target** malignant vs (normal + benign), derived from a single three-class model's
   calibrated malignant probability, so one model serves both the binary and three-class reports.
8. **Pseudo-patient groups.** The source has no patient IDs (files are named like
   `Malignant case (12).jpg`). Within each class, images are ordered by that number; neighbouring
   images are linked when their 64x64 grayscale thumbnails correlate at or above a link
   threshold, and near-identical images (correlation >= 0.97 or identical file hash) are always
   linked. Connected components are the groups; whole groups go to one split. Consecutive images
   correlate very highly (median in `results/grouping_diagnostics.json`), which supports that the
   numbering follows slices within a scan.
8a. **Per-class threshold rule.** A single threshold did not fit all classes: at 0.80 the benign
   class split into far more groups than its stated case count, while at 0.70 the normal class
   merged into far fewer. There is no clean gap in the benign similarities. So each class uses the
   highest threshold on a 0.50-0.95 grid (step 0.01) whose group count does not exceed the source's
   stated number of cases for that class. Having no more groups than patients is necessary for
   keeping every patient whole; merging distinct patients is the safe direction (it only widens
   confidence intervals). The chosen thresholds and resulting group counts are in
   `results/grouping_diagnostics.json`. This is still a heuristic: a group can contain parts of two
   patients and, in principle, one patient can still be split. Residual leakage is measured in
   `results/leakage_audit.json` and flagged in the model card.
9. **Identical files with different labels would be excluded**, because their true label cannot be
   known. None were found; the source does contain identical files within a class, which the
   grouping always keeps together. Counts are in `data/cohort_summary.json` and
   `results/cohort_flow.json`.
9a. **Non-standard image sizes are excluded.** Almost all images are 512x512. Every wider image
   (623x512 and 801x512) is from the malignant class, and visual inspection showed they are zoomed
   reconstructions without the scanner table or circular field of view. Keeping them would let the
   model learn the image format instead of anatomy, and cropping cannot remove the zoom cue. All
   images that are not 512x512 are excluded (`cohort.required_size`); counts per class are in
   `results/cohort_flow.json`.
9b. **The mirror's unlabelled `Test cases` folder is not used.** The Kaggle mirror bundles a folder
   of unlabelled PNG files whose names follow a different scheme from the IQ-OTH/NCCD images, so
   their origin cannot be confirmed from the official record. They carry no label and are ignored
   (count in `data/cohort_summary.json`).
9c. **Kaggle copy differs in size from the official record.** The mirror has fewer labelled images
   than the official page states; the project uses the mirror as downloaded and records both counts.
9d. **Download without a Kaggle token.** No Kaggle credentials were configured, so
   `00_setup_data.py` uses Kaggle's official `kagglehub` client, which downloads public datasets
   anonymously; with credentials it uses the authenticated Kaggle API. The exact image set is
   identified by a SHA256 fingerprint over sorted path and file hashes.
10. **Split** 70/15/15 by image count, stratified by class, groups assigned greedily towards the
    target fractions after seeding validation and test with one group per class, seed 42. With
    whole groups of very different sizes the realised fractions differ somewhat from the targets;
    the actual counts are in `results/cohort_flow.json`.

## Model and evaluation

10a. **Shortcut baseline.** A logistic regression on 32-bin intensity histograms (no spatial
    information) was fit on train and scored on test. On raw slices it separates malignant from
    the rest almost perfectly, which a tumour's pixel share cannot explain; the source readme says
    window settings varied. Kept as a permanent check (`results/shortcut_baseline.json`).
10b. **Input preprocessing: body mask + equalisation inside the body.** Per-image histogram
    equalisation alone lowered the histogram baseline but Grad-CAM of the resulting model put
    malignant-prediction attention on the scanner table and below the patient. Masking (Otsu
    threshold, opening, largest component, hole filling) removes table and background; equalisation
    is then computed inside the mask. Masks were checked visually on samples of every class. The
    zoom/framing cue (body filling more of the frame) cannot be removed this way and is documented.
10c. **Comparison runs.** The same model is also trained on equalise-only and raw inputs
    (`--variant`), metrics only, and reported next to the main model so readers can see how much
    the numbers depend on preprocessing. These are not used for model selection.
10d. **Learning rate chosen on validation only.** The first main run's validation AUROC peaked at
    epoch one and fell while training loss kept dropping (memorising few patients). A three-value
    sweep (`scripts/03a_lr_sweep.py`, `results/lr_sweep.json`) picked the rate with the best
    validation macro AUROC; the test split was not read.
11. **EfficientNet-B0, ImageNet weights, 224 px**, grayscale slices replicated to three channels.
    Augmentation: small random resized crop, horizontal flip, rotation up to 10 degrees, mild
    brightness/contrast. No synthetic oversampling; class-weighted cross-entropy for imbalance.
12. **AdamW + cosine schedule, mixed precision on CUDA, early stopping** on validation macro
    one-vs-rest AUROC (patience in config); the best-epoch weights are kept.
13. **Temperature scaling** fit on the validation split only (single scalar, LBFGS on NLL).
14. **Operating thresholds** fit on validation calibrated P(malignant): Youden threshold for the
    reported sensitivity/specificity; rule-out (sensitivity >= 0.95) and rule-in
    (specificity >= 0.95) thresholds define the dashboard's Low / Indeterminate / High bands.
    High is checked first if the two thresholds cross.
15. **Bootstrap CIs resample pseudo-patient groups, not slices**, because slices from one patient
    are correlated; resamples where a metric is undefined (one class only) are skipped and the
    number of valid resamples is stored with each metric.
16. **Subgroup metrics by sex and age are not possible**: the source provides no per-image or
    per-patient demographics. The model card says so instead of estimating anything.
17. **Grad-CAM** on the last convolutional block, computed for the predicted class and overlaid on
    the original slice. Shortcut checks over every test image: heatmap share in the outer 10%
    border band, share of peaks in that band, and share outside the same body mask used in
    preprocessing. The gallery was also reviewed by eye; observations are in the model card.
17a. **Risk bands when thresholds coincide.** With near-perfect validation separation the rule-out
    and rule-in thresholds can be equal, leaving no Indeterminate band; this is reported as is.

## Dashboard and docs

18. **No hand-typed numbers.** The app reads `results/`, `data/cohort_summary.json` (mirrored to
    `results/cohort_summary.json`) and `configs/config.yaml`; sample predictions are computed live.
    `tests/test_pipeline.py` renders every page and fails if any displayed number is not found in
    those files. README and model card results are generated by `scripts/05_render_docs.py`.
19. **Built-in samples** are the first test-split image of each class in sorted path order.
20. **Similar-cases page** shows nearest training-split images by cosine similarity of pooled
    EfficientNet features; it shows labels but no similarity scores, and is labelled as retrieval,
    not diagnosis.
