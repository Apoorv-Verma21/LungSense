#!/usr/bin/env bash
# Run the whole pipeline; stops on the first error. Pass --fast to train briefly on CPU.
# The learning-rate sweep (scripts/03a_lr_sweep.py) is run separately; its winner is in config.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
FAST=""
if [[ "${1:-}" == "--fast" ]]; then FAST="--fast"; fi

"$PYTHON" scripts/00_setup_data.py --skip-download
"$PYTHON" scripts/01_build_cohort.py
"$PYTHON" scripts/02_train_clinical.py
"$PYTHON" scripts/03_train_imaging.py $FAST
"$PYTHON" scripts/03_train_imaging.py --variant equalize $FAST
"$PYTHON" scripts/03_train_imaging.py --variant raw $FAST
"$PYTHON" scripts/04_fusion_eval.py
"$PYTHON" scripts/05_render_docs.py
"$PYTHON" -m pytest -q
