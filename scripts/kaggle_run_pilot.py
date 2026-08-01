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
    parser.add_argument(
        "--retrieval-limit",
        type=int,
        help="Number of train studies to reserve for the retrieval pool. Defaults to 80 percent of --limit.",
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        help="Number of validation/test studies to reserve for evaluation. Defaults to the remainder of --limit.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--retriever",
        choices=["lexical", "mock_image", "hashing_text"],
        default="lexical",
    )
    parser.add_argument(
        "--generator",
        choices=["mock", "retrieval_copy_stress"],
        default="mock",
        help="Generator backend. Use retrieval_copy_stress for a controlled retrieval-copy stress test.",
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


def balanced_limits(total_limit: int, retrieval_limit: int | None, eval_limit: int | None) -> tuple[int, int]:
    if retrieval_limit is not None and eval_limit is not None:
        return retrieval_limit, eval_limit
    if retrieval_limit is not None:
        return retrieval_limit, max(total_limit - retrieval_limit, 1)
    if eval_limit is not None:
        return max(total_limit - eval_limit, 1), eval_limit

    eval_count = max(round(total_limit * 0.2), 1)
    retrieval_count = max(total_limit - eval_count, 1)
    return retrieval_count, eval_count


def build_balanced_mimic_subset(args: argparse.Namespace) -> list:
    metadata = require_path(args.metadata, "--metadata")
    split = require_path(args.split, "--split")
    reports_root = require_path(args.reports_root, "--reports-root")
    images_root = require_path(args.images_root, "--images-root")
    views = {view.upper() for view in args.views}
    requested_splits = {split_name.lower() for split_name in args.dataset_splits}

    if {"train", "validate"}.issubset(requested_splits):
        retrieval_limit, eval_limit = balanced_limits(
            args.limit,
            args.retrieval_limit,
            args.eval_limit,
        )
        retrieval_records = build_mimic_subset(
            metadata_path=metadata,
            split_path=split,
            report_root=reports_root,
            image_root=images_root,
            label_path=args.labels,
            allowed_splits={"train"},
            allowed_views=views,
            limit=retrieval_limit,
        )
        eval_records = build_mimic_subset(
            metadata_path=metadata,
            split_path=split,
            report_root=reports_root,
            image_root=images_root,
            label_path=args.labels,
            allowed_splits={"validate"},
            allowed_views=views,
            limit=eval_limit,
        )
        return retrieval_records + eval_records

    return build_mimic_subset(
        metadata_path=metadata,
        split_path=split,
        report_root=reports_root,
        image_root=images_root,
        label_path=args.labels,
        allowed_splits=requested_splits,
        allowed_views=views,
        limit=args.limit,
    )


def require_pipeline_outputs(baseline_dir: Path, summary: dict) -> None:
    required_files = [
        baseline_dir / "retrieval_results.jsonl",
        baseline_dir / "mismatch_results.jsonl",
        baseline_dir / "generations_no_retrieval.jsonl",
        baseline_dir / "generations_retrieval.jsonl",
        baseline_dir / "generations_mismatch.jsonl",
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if not missing:
        return

    raise SystemExit(
        "Baseline pilot did not produce all expected outputs.\n"
        f"Pipeline summary: {json.dumps(summary, indent=2)}\n"
        f"Missing files: {json.dumps(missing, indent=2)}\n\n"
        "Most common cause: the subset contains no eval studies. "
        "Use both train and validate splits, or pass explicit "
        "--retrieval-limit and --eval-limit values."
    )


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
        studies = build_balanced_mimic_subset(args)
        write_study_records(subset_path, studies)

    summary = run_pipeline(
        input_path=subset_path,
        output_dir=baseline_dir,
        mode="all",
        top_k=args.top_k,
        retriever_backend=args.retriever,
        generator_backend=args.generator,
    )
    require_pipeline_outputs(baseline_dir, summary)

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
