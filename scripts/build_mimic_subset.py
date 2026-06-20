from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rgca_baseline.mimic_cxr import build_mimic_subset, write_study_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a MIMIC-CXR subset JSONL for RGCA experiments.")
    parser.add_argument("--metadata", required=True, help="Path to mimic-cxr-jpg metadata CSV(.gz).")
    parser.add_argument("--split", required=True, help="Path to mimic-cxr-jpg split CSV(.gz).")
    parser.add_argument("--reports-root", required=True, help="Path to MIMIC-CXR reports root.")
    parser.add_argument("--images-root", required=True, help="Path to MIMIC-CXR-JPG files root.")
    parser.add_argument("--labels", help="Optional path to CheXpert or NegBio labels CSV(.gz).")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument(
        "--dataset-splits",
        nargs="*",
        default=["train", "validate"],
        help="MIMIC split names to include, e.g. train validate test.",
    )
    parser.add_argument(
        "--views",
        nargs="*",
        default=["PA", "AP"],
        help="View positions to include, e.g. PA AP LATERAL.",
    )
    parser.add_argument("--limit", type=int, help="Optional limit on exported studies.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    studies = build_mimic_subset(
        metadata_path=args.metadata,
        split_path=args.split,
        report_root=args.reports_root,
        image_root=args.images_root,
        label_path=args.labels,
        allowed_splits={split_name for split_name in args.dataset_splits},
        allowed_views={view.upper() for view in args.views},
        limit=args.limit,
    )
    write_study_records(args.output, studies)
    print(f"Wrote {len(studies)} MIMIC-CXR study records to {args.output}")


if __name__ == "__main__":
    main()
