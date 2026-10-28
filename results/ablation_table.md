# Ablation table

**Single modality only.** The data source (IQ-OTH/NCCD CT slices) has no clinical variables and no second modality for the same patients, so clinical-only, late-fusion and intermediate-fusion rows cannot be computed. No other dataset was paired in.

| Model | AUROC (95% CI) | AUPRC (95% CI) | Sens @ 90% spec (95% CI) | n_test images |
|---|---|---|---|---|
| CT imaging only (EfficientNet-B0) | 1.000 (0.999-1.000) | 1.000 (0.999-1.000) | 1.000 (1.000-1.000) | 166 |

Task: malignant vs (normal + benign). CIs: 1000 pseudo-patient-group bootstrap resamples.

## Preprocessing comparison (same single modality)

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
