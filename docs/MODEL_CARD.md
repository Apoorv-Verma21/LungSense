# Model card: LungSense CT slice classifier

**Research/education demo. Not a medical device. Not for clinical use.**

## Model details

- EfficientNet-B0 convolutional network, ImageNet-pretrained, fine-tuned on single axial chest CT
  slices with a normal / benign / malignant output head.
- Input preprocessing: everything outside the patient's body is masked out (removes the scanner
  table and background), then the histogram is equalised inside the body only.
- The malignancy probability is temperature-scaled on the validation split. Risk bands use
  thresholds fit on the validation split (see the metrics section).
- Training and evaluation code: `scripts/03_train_imaging.py`; settings: `configs/config.yaml`.

## Intended use

- Teaching and research on medical image classification, calibration and explanation methods.
- Demonstrating an honest evaluation workflow on a small public dataset.

## Out-of-scope use

- Any clinical decision, screening, triage or diagnosis.
- Scans from other hospitals, scanners or protocols; non-chest images; full 3D studies.
- Any claim about a real person's health.

## Data

- One source: the IQ-OTH/NCCD lung cancer CT dataset (see `docs/DATA_SOURCES.md`), real
  de-identified scans from a single Iraqi oncology centre, released under CC BY 4.0.
- No patient identifiers are provided, so slices were grouped into pseudo-patients by image
  similarity and whole groups were assigned to the train, validation and test splits.
- No clinical variables or demographics are provided.

## Metrics

Generated from `results/` by `scripts/05_render_docs.py`.

<!-- GENERATED:metrics:START -->
### Test metrics

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

### Operating thresholds (fit on validation, calibrated P(malignant))

- Youden threshold (used for sensitivity/specificity): 0.489
- Rule-out threshold (validation sensitivity >= 0.95): 0.489
- Rule-in threshold (validation specificity >= 0.95): 0.489

### Pseudo-patient grouping and residual leakage

- Groups per class: {'benign': 15, 'malignant': 40, 'normal': 54} versus cases stated by the source: {'malignant': 40, 'benign': 15, 'normal': 55}
- Link threshold {'benign': 0.67, 'malignant': 0.79, 'normal': 0.85}; near-duplicate threshold 0.97
- Group overlap between splits: {'train&val': 0, 'train&test': 0, 'val&test': 0}
- val: 148 images; median nearest-train similarity 0.826; at or above link threshold 117; near-duplicates of a training image 0
- test: 166 images; median nearest-train similarity 0.789; at or above link threshold 81; near-duplicates of a training image 0

### Grad-CAM shortcut checks (all test images)

- Mean heatmap share in the outer border band: 0.128 (band covers 0.354 of the image area)
- Share of heatmap peaks inside the border band: 0.060
- Mean heatmap share outside the body mask: 0.197
- Gallery: 6 correct and 6 wrong examples (39 wrong test predictions available)

### Preprocessing comparison (same single modality)

Not a fusion ablation: the same CT model trained on differently preprocessed inputs, to show how much the headline numbers depend on acquisition cues. The shortcut baseline uses only each slice's intensity histogram.

| Input preprocessing | Malignant AUROC (95% CI) | Three-class macro AUROC (95% CI) | Recall normal | Recall benign | Recall malignant |
|---|---|---|---|---|---|
| body_mask_equalize (main model) | 1.000 (0.999-1.000) | 0.921 (0.805-1.000) | 0.590 | 0.263 | 1.000 |
| equalize | 1.000 (1.000-1.000) | 0.943 (0.856-1.000) | 0.902 | 0.316 | 1.000 |
| none | 1.000 (1.000-1.000) | 0.949 (0.868-1.000) | 0.344 | 0.737 | 1.000 |

| Shortcut baseline input | Malignant AUROC (95% CI) |
|---|---|
| none | 0.995 (0.978-1.000) |
| equalize | 0.778 (0.546-0.950) |
| body_mask_equalize | 0.822 (0.629-0.949) |
<!-- GENERATED:metrics:END -->

## Subgroup metrics

Not possible with this source: it provides no sex or age for any image or patient, so performance
by sex or age band cannot be measured. Nothing was estimated in their place.

## Most important finding: this dataset contains strong acquisition shortcuts

Read every metric above with this in mind.

- **Image format.** Every non-standard-size image in the source (wider, zoomed reconstructions
  without the scanner table) is malignant. These were excluded before splitting.
- **Intensity.** A logistic regression that sees only each slice's intensity histogram, with no
  spatial information at all, separates malignant from other slices almost perfectly on the raw
  images (see the shortcut baseline table above). A tumour occupies too few pixels to explain that,
  so the classes most likely differ in how the scans were windowed and exported.
- **Scanner table.** A first model trained on histogram-equalised slices placed its Grad-CAM
  attention for malignant predictions on the scanner table and the area below the patient, not on
  the lungs. The main model therefore masks out everything outside the body and equalises the
  histogram inside the body only. After masking, attention moved onto the body (see the border
  checks above).
- **Remaining cues.** Some malignant slices are framed more tightly, so the body fills more of the
  image; masking cannot remove that. Intensity differences inside the body may also remain; the
  shortcut baseline on masked, equalised slices is still well above chance.

**Conclusion:** the near-perfect malignant-vs-rest discrimination reflects how this particular
dataset was assembled at least as much as it reflects pathology. It must not be read as evidence
that the model detects lung cancer. The benign-vs-normal distinction, which the shortcuts do not
solve, is where the model is weak, and with very few benign patients in the test split those
estimates are unstable (compare the preprocessing variants above).

## Grad-CAM observations

Reviewed by eye on the saved gallery (`results/gradcam`) for the main model:

- Most heatmaps fall inside the body, over the lungs, hila and mediastinum, including near the mass
  in the correctly classified malignant examples. Attention is broad, not focal on lesions.
- A minority of heatmaps still extend outside the body or sit on its lateral edge, mostly for
  normal slices; no image contains text markers or burned-in annotations.
- Grad-CAM here is computed on a coarse final feature map and upsampled, so it cannot localise
  small nodules; it shows the region that drove the score, not a lesion outline.
- With the equalise-only model (before body masking) malignant predictions focused on the scanner
  table and the area below the patient; this was the reason body masking was added.

## Failure modes and risks

- **Leakage inflating results.** Without patient IDs, slices of one patient can end up in both
  training and test data. Pseudo-patient grouping reduces this but cannot rule it out; see the
  residual-leakage numbers above. Treat test metrics as an upper bound.
- **Shortcut learning.** The model may use scanner, windowing or annotation artefacts that differ
  between classes instead of anatomy. The Grad-CAM checks look for attention on image borders and
  outside the body, but they cannot prove the model uses the right features.
- **Small classes.** The benign class is small, so its recall estimate is unstable and its
  confidence interval wide.
- **Distribution shift.** Any other hospital, scanner, slice thickness or reconstruction kernel is
  outside the training distribution.
- **Single slices.** Clinical CT reading uses the whole volume and prior scans; one slice can miss
  or mimic findings.
- **Calibration of one class.** Temperature scaling is fit on the three-class output; it can leave
  a single class (here malignant) less well calibrated than before. Compare the before and after
  calibration errors in the metrics section.
- **Unstable minority-class results.** Normal-vs-benign recall swings widely between preprocessing
  variants and runs because the test split holds very few benign patients.

## Ethics

- The data is de-identified and was released publicly under CC BY 4.0 with institutional review
  board approval, according to the official record.
- Errors in either direction matter: a false negative could falsely reassure, a false positive
  could cause anxiety. This is why the tool is restricted to education and research.
- The dataset comes from one population in one country; fairness across populations could not be
  assessed.
