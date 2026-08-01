from __future__ import annotations

from pathlib import Path

from rgca_baseline.evaluation import infer_labels_from_text, normalize_labels
from rgca_baseline.io_utils import read_jsonl, write_json, write_jsonl
from rgca_baseline.prompts import build_prompt
from rgca_baseline.real_retrieval import create_retriever_backend, get_retrieval_plan
from rgca_baseline.schemas import GenerationResult, StudyRecord
from rgca_baseline.vlm_client import create_vlm_client


def load_studies(path: str | Path) -> list[StudyRecord]:
    return [StudyRecord(**row) for row in read_jsonl(path)]


def run_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    mode: str,
    top_k: int,
    retriever_backend: str = "lexical",
    generator_backend: str = "mock",
) -> dict:
    studies = load_studies(input_path)
    retrieval_pool = [study for study in studies if study.split == "retrieval_pool"]
    eval_studies = [study for study in studies if study.split == "eval"]

    retriever = create_retriever_backend(retriever_backend, retrieval_pool)
    generator = create_vlm_client(generator_backend)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    retrieval_results: list[dict] = []
    mismatch_results: list[dict] = []
    generations_by_mode: dict[str, list[dict]] = {
        "no_retrieval": [],
        "retrieval": [],
        "mismatch": [],
    }

    modes = ["no_retrieval", "retrieval", "mismatch"] if mode == "all" else [mode]
    for study in eval_studies:
        retrieval_result = retriever.retrieve(study, top_k)
        mismatch_result = retriever.mismatch(study, top_k)
        retrieval_results.append(retrieval_result.to_dict())
        mismatch_results.append(mismatch_result.to_dict())

        for current_mode in modes:
            if current_mode == "no_retrieval":
                retrieved_reports: list[str] = []
                retrieved_ids: list[str] = []
            elif current_mode == "retrieval":
                retrieved_reports = retrieval_result.retrieved_reports
                retrieved_ids = retrieval_result.retrieved_studies
            else:
                retrieved_reports = mismatch_result.retrieved_reports
                retrieved_ids = mismatch_result.retrieved_studies

            prompt = build_prompt(current_mode, study, retrieved_reports)
            generated_report = generator.generate_report(
                study=study,
                prompt=prompt,
                mode=current_mode,
                retrieved_reports=retrieved_reports,
            )
            generated_labels = infer_labels_from_text(generated_report)
            reference_labels = normalize_labels(study.labels)
            hallucination_flags = sorted(set(generated_labels) - set(reference_labels))
            generations_by_mode[current_mode].append(
                GenerationResult(
                    study_id=study.study_id,
                    mode=current_mode,
                    prompt=prompt,
                    target_report=study.report_text,
                    retrieved_reports=retrieved_reports,
                    generated_report=generated_report,
                    retrieved_studies=retrieved_ids,
                    labels_reference=reference_labels,
                    labels_generated=generated_labels,
                    hallucination_flags=hallucination_flags,
                    retriever_backend=retriever_backend,
                    generator_backend=generator_backend,
                ).to_dict()
            )

    write_jsonl(output_root / "retrieval_results.jsonl", retrieval_results)
    write_jsonl(output_root / "mismatch_results.jsonl", mismatch_results)
    for current_mode, rows in generations_by_mode.items():
        if rows:
            write_jsonl(output_root / f"generations_{current_mode}.jsonl", rows)

    summary = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "mode": mode,
        "top_k": top_k,
        "retriever_backend": retriever_backend,
        "generator_backend": generator_backend,
        "retrieval_plan": get_retrieval_plan(retriever_backend).__dict__,
        "retrieval_pool_size": len(retrieval_pool),
        "eval_size": len(eval_studies),
        "generated_counts": {key: len(value) for key, value in generations_by_mode.items() if value},
    }
    write_json(output_root / "run_summary.json", summary)
    return summary
