from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rgca_baseline.integrity import jsonl_fingerprint, validate_image_paths, validate_study_records
from rgca_baseline.io_utils import write_json, write_jsonl
from rgca_baseline.pipeline import load_studies
from rgca_baseline.real_retrieval import create_retriever_backend, get_retrieval_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run retrieval-only validation before expensive VLM generation.",
    )
    parser.add_argument("--subset", required=True, help="Study subset JSONL path.")
    parser.add_argument(
        "--backend",
        default="hashing_text",
        choices=["lexical", "hashing_text", "mock_image", "biomedclip"],
        help="Retriever backend to validate.",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for retrieval artifacts.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=20,
        help="Limit evaluation queries for a cheap first real-retrieval run.",
    )
    parser.add_argument(
        "--require-images",
        action="store_true",
        help="Fail if any study image_path is missing. This is implied by --backend biomedclip.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subset_path = Path(args.subset)
    if not subset_path.exists():
        raise SystemExit(f"Subset JSONL does not exist: {subset_path}")

    studies = load_studies(subset_path)
    validation = validate_study_records(studies)
    if not validation["valid"]:
        raise SystemExit(f"Invalid study dataset: {json.dumps(validation, indent=2)}")

    if args.backend == "biomedclip" or args.require_images:
        image_validation = validate_image_paths(studies)
        if not image_validation["valid"]:
            raise SystemExit(
                "Image files are required for this retrieval validation but some paths are missing:\n"
                f"{json.dumps(image_validation, indent=2)}"
            )
    else:
        image_validation = None

    retrieval_pool = [study for study in studies if study.split == "retrieval_pool"]
    eval_studies = [study for study in studies if study.split == "eval"][: args.eval_limit]
    retriever = create_retriever_backend(args.backend, retrieval_pool)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    retrieval_rows = []
    mismatch_rows = []
    for study in eval_studies:
        retrieval_rows.append(retriever.retrieve(study, args.top_k).to_dict())
        mismatch_rows.append(retriever.mismatch(study, args.top_k).to_dict())

    write_jsonl(output_dir / "retrieval_results.jsonl", retrieval_rows)
    write_jsonl(output_dir / "mismatch_results.jsonl", mismatch_rows)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "subset": str(subset_path),
        "dataset_fingerprint": jsonl_fingerprint(subset_path),
        "dataset_validation": validation,
        "image_validation": image_validation,
        "backend": args.backend,
        "retrieval_plan": get_retrieval_plan(args.backend).__dict__,
        "top_k": args.top_k,
        "retrieval_pool_size": len(retrieval_pool),
        "eval_size": len(eval_studies),
        "outputs": {
            "retrieval_results": str(output_dir / "retrieval_results.jsonl"),
            "mismatch_results": str(output_dir / "mismatch_results.jsonl"),
        },
    }
    write_json(output_dir / "retrieval_validation_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
