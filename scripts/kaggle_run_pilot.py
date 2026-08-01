from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rgca_baseline.evaluation import evaluate_generation_rows, write_evaluation_outputs
from rgca_baseline.io_utils import read_jsonl
from rgca_baseline.mimic_cxr import build_mimic_subset, write_study_records
from rgca_baseline.pipeline import load_studies, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the RGCA baseline pilot in a Kaggle-friendly layout.",
    )
    parser.add_argument(
        "--subset-jsonl",
        help="Existing study JSONL. If provided, MIMIC CSV/report/image paths are not required.",
    )
    parser.add_argument("--metadata", help="Path to MIMIC-CXR-JPG metadata CSV(.gz).")
    parser.add_argument("--split", help="Path to MIMIC-CXR-JPG split CSV(.gz).")
    parser.add_argument("--reports-root", help="Path to MIMIC-CXR report files root.")
    parser.add_argument("--images-root", help="Path to MIMIC-CXR-JPG image files root.")
    parser.add_argument("--labels", help="Optional path to CheXpert/NegBio labels CSV(.gz).")
    parser.add_argument("--output-dir", default="/kaggle/working/rgca_pilot")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--retriever",
        choices=["lexical", "mock_image", "hashing_text"],
        default="lexical",
    )
    parser.add_argument(
        "--dataset-splits",
        nargs="*",
        default=["train", "validate"],
        help="MIMIC split names to include when building the subset.",
    )
    parser.add_argument(
        "--views",
        nargs="*",
        default=["PA", "AP"],
        help="View positions to include when building the subset.",
    )
    return parser.parse_args()


def require_path(value: str | None, name: str) -> str:
    if not value:
        raise SystemExit(f"Missing required argument when building subset: {name}")
    return value


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    data_dir = output_dir / "data"
    baseline_dir = output_dir / "baseline"
    eval_dir = output_dir / "evaluation_mismatch"
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.subset_jsonl:
        subset_path = Path(args.subset_jsonl)
    else:
        subset_path = data_dir / "mimic_subset.jsonl"
        studies = build_mimic_subset(
            metadata_path=require_path(args.metadata, "--metadata"),
            split_path=require_path(args.split, "--split"),
            report_root=require_path(args.reports_root, "--reports-root"),
            image_root=require_path(args.images_root, "--images-root"),
            label_path=args.labels,
            allowed_splits={split_name for split_name in args.dataset_splits},
            allowed_views={view.upper() for view in args.views},
            limit=args.limit,
        )
        write_study_records(subset_path, studies)

    summary = run_pipeline(
        input_path=subset_path,
        output_dir=baseline_dir,
        mode="all",
        top_k=args.top_k,
        retriever_backend=args.retriever,
    )

    studies = {study.study_id: study for study in load_studies(subset_path)}
    generation_rows = read_jsonl(baseline_dir / "generations_mismatch.jsonl")
    retrieval_rows = {
        row["target_study"]: row
        for row in read_jsonl(baseline_dir / "mismatch_results.jsonl")
    }
    eval_details, eval_summary = evaluate_generation_rows(
        studies_by_id=studies,
        generation_rows=generation_rows,
        retrieval_rows_by_target=retrieval_rows,
    )
    write_evaluation_outputs(eval_dir, eval_details, eval_summary)

    manifest = {
        "subset_path": str(subset_path),
        "baseline_dir": str(baseline_dir),
        "evaluation_dir": str(eval_dir),
        "pipeline_summary": summary,
        "evaluation_summary": eval_summary,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "kaggle_pilot_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("Kaggle pilot complete.")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
