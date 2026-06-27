from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rgca_baseline.mimic_cxr import plan_mimic_pilot_subset, write_pilot_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan a small MIMIC-CXR pilot subset manifest.")
    parser.add_argument("--metadata", required=True, help="Path to mimic-cxr-jpg metadata CSV(.gz).")
    parser.add_argument("--split", required=True, help="Path to mimic-cxr-jpg split CSV(.gz).")
    parser.add_argument("--images-root", required=True, help="Path to MIMIC-CXR-JPG files root.")
    parser.add_argument("--labels", help="Optional path to CheXpert labels CSV(.gz).")
    parser.add_argument("--output", required=True, help="Output pilot-plan JSONL path.")
    parser.add_argument("--train-limit", type=int, default=150, help="Number of train studies to plan.")
    parser.add_argument("--eval-limit", type=int, default=50, help="Number of validate studies to plan.")
    parser.add_argument(
        "--views",
        nargs="*",
        default=["PA", "AP"],
        help="View positions to include, e.g. PA AP.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plans = plan_mimic_pilot_subset(
        metadata_path=args.metadata,
        split_path=args.split,
        image_root=args.images_root,
        label_path=args.labels,
        train_limit=args.train_limit,
        eval_limit=args.eval_limit,
        allowed_views={view.upper() for view in args.views},
    )
    write_pilot_plan(args.output, plans)
    print(
        f"Wrote {len(plans)} pilot study plans to {args.output} "
        f"(train_limit={args.train_limit}, eval_limit={args.eval_limit})"
    )


if __name__ == "__main__":
    main()
