from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rgca_baseline.evaluation import evaluate_generation_rows, write_evaluation_outputs
from rgca_baseline.io_utils import read_jsonl
from rgca_baseline.pipeline import load_studies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated RGCA baseline outputs.")
    parser.add_argument("--studies", required=True, help="Study JSONL path.")
    parser.add_argument("--generations", required=True, help="Generation JSONL path.")
    parser.add_argument("--retrieval-results", required=True, help="Retrieval result JSONL path.")
    parser.add_argument("--output-dir", required=True, help="Directory for evaluation outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    studies = {study.study_id: study for study in load_studies(args.studies)}
    generation_rows = read_jsonl(args.generations)
    retrieval_rows = {row["target_study"]: row for row in read_jsonl(args.retrieval_results)}
    detailed_rows, summary = evaluate_generation_rows(studies, generation_rows, retrieval_rows)
    write_evaluation_outputs(args.output_dir, detailed_rows, summary)
    print(summary)


if __name__ == "__main__":
    main()
