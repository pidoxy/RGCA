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

from rgca_baseline.evaluation import (
    evaluate_generation_rows,
    infer_labels_from_text,
    normalize_labels,
)
from rgca_baseline.integrity import jsonl_fingerprint, validate_image_paths, validate_study_records
from rgca_baseline.io_utils import read_jsonl, write_json, write_jsonl
from rgca_baseline.pipeline import load_studies
from rgca_baseline.prompts import build_prompt
from rgca_baseline.schemas import GenerationResult, StudyRecord
from rgca_baseline.vlm_client import create_vlm_client


MODES = ["no_retrieval", "retrieval", "mismatch"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate reports from saved retrieval artifacts. This keeps retrieval fixed "
            "while comparing no-retrieval, clean-retrieval, and mismatch prompts."
        )
    )
    parser.add_argument(
        "--subset",
        "--studies",
        dest="subset",
        required=True,
        help="Study subset JSONL path. --studies is kept as a compatibility alias.",
    )
    parser.add_argument("--retrieval-results", required=True, help="Clean retrieval JSONL artifact.")
    parser.add_argument("--mismatch-results", required=True, help="Mismatch retrieval JSONL artifact.")
    parser.add_argument("--output-dir", required=True, help="Output directory for generated reports/evaluation.")
    parser.add_argument(
        "--mode",
        default="all",
        choices=[*MODES, "all"],
        help="Generation condition to run.",
    )
    parser.add_argument(
        "--generator",
        default="mock",
        choices=["mock", "retrieval_copy_stress", "hf_vlm"],
        help="Generator backend. Use hf_vlm for real HuggingFace vision-language inference.",
    )
    parser.add_argument("--model-id", help="HuggingFace model id for --generator hf_vlm.")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--limit", type=int, default=20, help="Maximum eval studies to generate.")
    parser.add_argument(
        "--top-k",
        type=int,
        help=(
            "Compatibility no-op. Top-k is already fixed inside retrieval_results.jsonl "
            "and mismatch_results.jsonl for this generation-only runner."
        ),
    )
    parser.add_argument(
        "--require-real-generator",
        action="store_true",
        help="Fail unless --generator hf_vlm. Use for publication-quality generation runs.",
    )
    parser.add_argument(
        "--require-images",
        action="store_true",
        help="Fail if any generated study image path is missing. This is implied by --generator hf_vlm.",
    )
    parser.add_argument(
        "--align-to-retrieval-artifacts",
        action="store_true",
        help=(
            "Select eval studies from the subset that have the required retrieval/mismatch "
            "artifact rows before applying --limit. This prevents late GPU failures when "
            "the subset contains more eval studies than the retrieval artifacts cover."
        ),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate subset/retrieval coverage and write generation_preflight.json without loading a generator.",
    )
    return parser.parse_args()


def rows_by_target(path: str | Path) -> dict[str, dict]:
    rows = read_jsonl(path)
    return {row["target_study"]: row for row in rows}


def select_eval_studies(studies: list[StudyRecord], limit: int) -> list[StudyRecord]:
    eval_studies = [study for study in studies if study.split == "eval"]
    return eval_studies[:limit] if limit else eval_studies


def required_target_ids_for_modes(
    modes: list[str],
    retrieval_by_target: dict[str, dict],
    mismatch_by_target: dict[str, dict],
) -> set[str] | None:
    required_sets: list[set[str]] = []
    if "retrieval" in modes:
        required_sets.append(set(retrieval_by_target))
    if "mismatch" in modes:
        required_sets.append(set(mismatch_by_target))
    if not required_sets:
        return None
    required_ids = required_sets[0]
    for target_ids in required_sets[1:]:
        required_ids = required_ids & target_ids
    return required_ids


def select_eval_studies_with_artifact_coverage(
    studies: list[StudyRecord],
    limit: int,
    modes: list[str],
    retrieval_by_target: dict[str, dict],
    mismatch_by_target: dict[str, dict],
    align_to_retrieval_artifacts: bool,
) -> tuple[list[StudyRecord], dict]:
    eval_studies_all = [study for study in studies if study.split == "eval"]
    required_ids = required_target_ids_for_modes(modes, retrieval_by_target, mismatch_by_target)

    if align_to_retrieval_artifacts and required_ids is not None:
        eval_studies = [study for study in eval_studies_all if study.study_id in required_ids]
        selected = eval_studies[:limit] if limit else eval_studies
    else:
        selected = eval_studies_all[:limit] if limit else eval_studies_all

    selected_ids = {study.study_id for study in selected}
    missing_retrieval = (
        sorted(selected_ids - set(retrieval_by_target)) if "retrieval" in modes else []
    )
    missing_mismatch = sorted(selected_ids - set(mismatch_by_target)) if "mismatch" in modes else []

    coverage = {
        "requested_limit": limit,
        "selected_eval_size": len(selected),
        "total_eval_studies_in_subset": len(eval_studies_all),
        "retrieval_rows": len(retrieval_by_target),
        "mismatch_rows": len(mismatch_by_target),
        "align_to_retrieval_artifacts": align_to_retrieval_artifacts,
        "missing_retrieval_count": len(missing_retrieval),
        "missing_mismatch_count": len(missing_mismatch),
        "missing_retrieval_examples": missing_retrieval[:20],
        "missing_mismatch_examples": missing_mismatch[:20],
    }
    if required_ids is not None:
        coverage["eval_studies_with_required_artifact_rows"] = sum(
            1 for study in eval_studies_all if study.study_id in required_ids
        )
    return selected, coverage


def mode_retrieval_context(
    mode: str,
    study: StudyRecord,
    retrieval_by_target: dict[str, dict],
    mismatch_by_target: dict[str, dict],
) -> tuple[list[str], list[str]]:
    if mode == "no_retrieval":
        return [], []

    source = retrieval_by_target if mode == "retrieval" else mismatch_by_target
    row = source.get(study.study_id)
    if not row:
        raise KeyError(f"Missing {mode} retrieval row for study_id={study.study_id}")

    return row.get("retrieved_reports", []), row.get("retrieved_studies", [])


def generate_mode_rows(
    mode: str,
    eval_studies: list[StudyRecord],
    generator,
    retrieval_by_target: dict[str, dict],
    mismatch_by_target: dict[str, dict],
    generator_backend: str,
) -> list[dict]:
    rows: list[dict] = []
    for index, study in enumerate(eval_studies, start=1):
        retrieved_reports, retrieved_studies = mode_retrieval_context(
            mode,
            study,
            retrieval_by_target,
            mismatch_by_target,
        )
        prompt = build_prompt(mode, study, retrieved_reports)
        generated_report = generator.generate_report(
            study=study,
            prompt=prompt,
            retrieved_reports=retrieved_reports,
            mode=mode,
        )
        generated_labels = infer_labels_from_text(generated_report)
        reference_labels = normalize_labels(study.labels)
        hallucination_flags = sorted(set(generated_labels) - set(reference_labels))
        rows.append(
            GenerationResult(
                study_id=study.study_id,
                mode=mode,
                prompt=prompt,
                target_report=study.report_text,
                retrieved_reports=retrieved_reports,
                generated_report=generated_report,
                retrieved_studies=retrieved_studies,
                labels_reference=reference_labels,
                labels_generated=generated_labels,
                hallucination_flags=hallucination_flags,
                retriever_backend="artifact",
                generator_backend=generator_backend,
            ).to_dict()
        )
        print(f"[{mode}] generated {index}/{len(eval_studies)} study_id={study.study_id}")
    return rows


def write_evaluation_bundle(
    output_dir: Path,
    mode: str,
    studies_by_id: dict[str, StudyRecord],
    generation_rows: list[dict],
    retrieval_rows: dict[str, dict],
) -> dict:
    details, summary = evaluate_generation_rows(studies_by_id, generation_rows, retrieval_rows)
    eval_dir = output_dir / f"evaluation_{mode}"
    write_jsonl(eval_dir / "evaluation_details.jsonl", details)
    write_json(eval_dir / "evaluation_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    if args.top_k is not None:
        print(
            "[info] Ignoring --top-k because retrieval artifacts already contain the "
            "retrieved reports for generation."
        )
    if args.require_real_generator and args.generator != "hf_vlm":
        raise SystemExit("--require-real-generator blocks mock/stress generation. Use --generator hf_vlm.")

    subset_path = Path(args.subset)
    retrieval_path = Path(args.retrieval_results)
    mismatch_path = Path(args.mismatch_results)
    for path in [subset_path, retrieval_path, mismatch_path]:
        if not path.exists():
            raise SystemExit(f"Required artifact is missing: {path}")

    studies = load_studies(subset_path)
    dataset_validation = validate_study_records(studies)
    if not dataset_validation["valid"]:
        raise SystemExit(f"Invalid subset: {json.dumps(dataset_validation, indent=2)}")

    retrieval_by_target = rows_by_target(retrieval_path)
    mismatch_by_target = rows_by_target(mismatch_path)
    modes = MODES if args.mode == "all" else [args.mode]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_studies, artifact_coverage = select_eval_studies_with_artifact_coverage(
        studies=studies,
        limit=args.limit,
        modes=modes,
        retrieval_by_target=retrieval_by_target,
        mismatch_by_target=mismatch_by_target,
        align_to_retrieval_artifacts=args.align_to_retrieval_artifacts,
    )
    write_json(output_dir / "generation_preflight.json", artifact_coverage)

    if artifact_coverage["missing_retrieval_count"] or artifact_coverage["missing_mismatch_count"]:
        raise SystemExit(
            "Selected eval studies are not fully covered by the retrieval artifacts. "
            "No generation was run, so GPU time is protected.\n"
            f"{json.dumps(artifact_coverage, indent=2)}\n"
            "Fix: regenerate/attach retrieval artifacts for the requested eval set, "
            "lower --limit, or pass --align-to-retrieval-artifacts to select only "
            "covered studies."
        )
    if args.limit and len(eval_studies) < args.limit:
        raise SystemExit(
            "The attached artifacts do not contain enough covered eval studies for the "
            f"requested --limit={args.limit}. No generation was run.\n"
            f"{json.dumps(artifact_coverage, indent=2)}"
        )
    if args.preflight_only:
        print(json.dumps(artifact_coverage, indent=2))
        print(f"Preflight passed. Coverage report: {output_dir / 'generation_preflight.json'}")
        return

    if args.generator == "hf_vlm" or args.require_images:
        image_validation = validate_image_paths(eval_studies)
        if not image_validation["valid"]:
            raise SystemExit(
                "Image files are required for this generation run but some eval images are missing:\n"
                f"{json.dumps(image_validation, indent=2)}"
            )
    else:
        image_validation = None

    studies_by_id = {study.study_id: study for study in studies}

    generator = create_vlm_client(
        backend=args.generator,
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
    )

    evaluation_summary: dict[str, dict] = {}
    generated_counts: dict[str, int] = {}

    for mode in modes:
        rows = generate_mode_rows(
            mode=mode,
            eval_studies=eval_studies,
            generator=generator,
            retrieval_by_target=retrieval_by_target,
            mismatch_by_target=mismatch_by_target,
            generator_backend=args.generator,
        )
        write_jsonl(output_dir / f"generations_{mode}.jsonl", rows)
        if mode == "no_retrieval":
            retrieval_context = {}
        elif mode == "mismatch":
            retrieval_context = mismatch_by_target
        else:
            retrieval_context = retrieval_by_target
        evaluation_summary.update(
            write_evaluation_bundle(
                output_dir=output_dir,
                mode=mode,
                studies_by_id=studies_by_id,
                generation_rows=rows,
                retrieval_rows=retrieval_context,
            )
        )
        generated_counts[mode] = len(rows)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "subset": str(subset_path),
        "dataset_fingerprint": jsonl_fingerprint(subset_path),
        "dataset_validation": dataset_validation,
        "image_validation": image_validation,
        "retrieval_results": str(retrieval_path),
        "mismatch_results": str(mismatch_path),
        "output_dir": str(output_dir),
        "mode": args.mode,
        "generator_backend": args.generator,
        "model_id": args.model_id,
        "max_new_tokens": args.max_new_tokens,
        "eval_size": len(eval_studies),
        "artifact_coverage": artifact_coverage,
        "generated_counts": generated_counts,
        "evaluation_summary": evaluation_summary,
    }
    write_json(output_dir / "generation_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
