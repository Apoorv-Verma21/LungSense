"""Clinical model (Source A only).

Source B (IQ-OTH/NCCD) contains CT slices and class labels only - no clinical variables - so
there is nothing to train. This script records that fact in results/clinical_status.json,
which the dashboard reads to show "not available in this data source".
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.common import cfg_path, load_config, read_source, write_json


def main() -> None:
    cfg = load_config()
    source = read_source(cfg)
    if source != "B":
        raise SystemExit("Source A clinical pipeline is not implemented in this build (see docs/DECISIONS.md).")
    status = {
        "available": False,
        "source": source,
        "reason": "Source B contains CT slice images and class labels only; it has no clinical variables. "
                  "No clinical model was trained and no other dataset was brought in.",
    }
    write_json(cfg_path(cfg, "results_dir") / "clinical_status.json", status)
    print(status["reason"])


if __name__ == "__main__":
    main()
