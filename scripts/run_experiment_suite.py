from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rgca_baseline.evaluation import evaluate_generation_rows, write_evaluation_outputs
from rgca_baseline.integrity import (
    assert_dataset_allowed,
    assert_suite_allowed,
    jsonl_fingerprint,
    validate_study_records,
)
from rgca_baseline.io_utils import read_jsonl, write_json
from rgca_baseline.pipeline import load_studies, run_pipeline


EVALUATION_TARGETS = {
    "retrieval": ("generations_retrieval.jsonl", "retrieval_results.jsonl"),
    "mismatch": ("generations_mismatch.jsonl", "mismatch_results.jsonl"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a structured RGCA experiment suite.")
    parser.add_argument("--config", required=True, help="Experiment suite JSON config.")
    parser.add_argument("--input", help="Override config input_path.")
    parser.add_argument("--output-dir", help="Override config output_dir.")
    parser.add_argument(
        "--only",
        nargs="*",
        help="Optional experiment names to run. Defaults to all experiments in config order.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output directories for selected experiments.",
    )
    parser.add_argument(
        "--execution-mode",
        choices=["debug", "stress", "real"],
        default="debug",
        help="Integrity mode. real blocks mock/stress/debug backends.",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_config(config: dict) -> None:
    required = {"suite_name", "input_path", "output_dir", "experiments"}
    missing = sorted(required - set(config))
    if missing:
        raise SystemExit(f"Suite config is missing required keys: {missing}")

    names = [experiment["name"] for experiment in config["experiments"]]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise SystemExit(f"Experiment names must be unique. Duplicates: {duplicates}")


def evaluate_mode(
    studies_by_id: dict,
    baseline_dir: Path,
    evaluation_root: Path,
    mode: str,
) -> dict | None:
    generation_file, retrieval_file = EVALUATION_TARGETS[mode]
    generation_path = baseline_dir / generation_file
    retrieval_path = baseline_dir / retrieval_file
    if not generation_path.exists() or not retrieval_path.exists():
        return None

    generation_rows = read_jsonl(generation_path)
    retrieval_rows = {row["target_study"]: row for row in read_jsonl(retrieval_path)}
    details, summary = evaluate_generation_rows(studies_by_id, generation_rows, retrieval_rows)
    write_evaluation_outputs(evaluation_root / mode, details, summary)
    return summary


def run_experiment(
    experiment: dict,
    input_path: Path,
    suite_output_dir: Path,
    studies_by_id: dict,
    overwrite: bool,
) -> dict:
    experiment_dir = suite_output_dir / experiment["name"]
    baseline_dir = experiment_dir / "baseline"
    evaluation_dir = experiment_dir / "evaluation"

    if experiment_dir.exists():
        if not overwrite:
            raise SystemExit(
                f"Output already exists for {experiment['name']}: {experiment_dir}. "
                "Use --overwrite or choose a new output_dir."
            )
        shutil.rmtree(experiment_dir)

    pipeline_summary = run_pipeline(
        input_path=input_path,
        output_dir=baseline_dir,
        mode=experiment.get("mode", "all"),
        top_k=int(experiment.get("top_k", 3)),
        retriever_backend=experiment.get("retriever", "lexical"),
        generator_backend=experiment.get("generator", "mock"),
    )

    evaluation_summaries = {}
    for mode in EVALUATION_TARGETS:
        summary = evaluate_mode(studies_by_id, baseline_dir, evaluation_dir, mode)
        if summary is not None:
            evaluation_summaries[mode] = summary

    result = {
        "name": experiment["name"],
        "purpose": experiment.get("purpose", ""),
        "experiment": experiment,
        "experiment_dir": str(experiment_dir),
        "baseline_dir": str(baseline_dir),
        "evaluation_dir": str(evaluation_dir),
        "pipeline_summary": pipeline_summary,
        "evaluation_summaries": evaluation_summaries,
    }
    write_json(experiment_dir / "experiment_manifest.json", result)
    return result


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    validate_config(config)

    input_path = Path(args.input or config["input_path"])
    output_dir = Path(args.output_dir or config["output_dir"])
    if not input_path.exists():
        raise SystemExit(f"Input study JSONL does not exist: {input_path}")
    try:
        assert_dataset_allowed(input_path, args.execution_mode)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    selected = set(args.only or [])
    experiments = [
        experiment
        for experiment in config["experiments"]
        if not selected or experiment["name"] in selected
    ]
    if selected and len(experiments) != len(selected):
        found = {experiment["name"] for experiment in experiments}
        raise SystemExit(f"Unknown experiment names requested: {sorted(selected - found)}")

    experiment_tiers = assert_suite_allowed(experiments, args.execution_mode)
    studies = load_studies(input_path)
    dataset_validation = validate_study_records(studies)
    if not dataset_validation["valid"]:
        raise SystemExit(f"Invalid study dataset: {json.dumps(dataset_validation, indent=2)}")

    studies_by_id = {study.study_id: study for study in studies}
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for experiment in experiments:
        print(f"Running {experiment['name']}...")
        results.append(
            run_experiment(
                experiment=experiment,
                input_path=input_path,
                suite_output_dir=output_dir,
                studies_by_id=studies_by_id,
                overwrite=args.overwrite,
            )
        )

    suite_manifest = {
        "suite_name": config["suite_name"],
        "config_path": str(config_path),
        "input_path": str(input_path),
        "execution_mode": args.execution_mode,
        "dataset_fingerprint": jsonl_fingerprint(input_path),
        "dataset_validation": dataset_validation,
        "experiment_tiers": experiment_tiers,
        "output_dir": str(output_dir),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "notes": config.get("notes", ""),
        "experiments": results,
    }
    write_json(output_dir / "suite_manifest.json", suite_manifest)
    print("Experiment suite complete.")
    print(json.dumps(suite_manifest, indent=2))


if __name__ == "__main__":
    main()
