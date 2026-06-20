from __future__ import annotations

from pathlib import Path

from rgca_baseline.generator import MockVLMGenerator
from rgca_baseline.io_utils import read_jsonl, write_json, write_jsonl
from rgca_baseline.prompts import build_prompt
from rgca_baseline.retrieval import LexicalRetriever
from rgca_baseline.schemas import GenerationResult, StudyRecord


def load_studies(path: str | Path) -> list[StudyRecord]:
    return [StudyRecord(**row) for row in read_jsonl(path)]


def run_pipeline(input_path: str | Path, output_dir: str | Path, mode: str, top_k: int) -> dict:
    studies = load_studies(input_path)
    retrieval_pool = [study for study in studies if study.split == "retrieval_pool"]
    eval_studies = [study for study in studies if study.split == "eval"]

    retriever = LexicalRetriever(retrieval_pool)
    generator = MockVLMGenerator()
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
            generated_report = generator.generate(
                study=study,
                mode=current_mode,
                retrieved_reports=retrieved_reports,
            )
            generations_by_mode[current_mode].append(
                GenerationResult(
                    study_id=study.study_id,
                    mode=current_mode,
                    prompt=prompt,
                    generated_report=generated_report,
                    retrieved_studies=retrieved_ids,
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
        "retrieval_pool_size": len(retrieval_pool),
        "eval_size": len(eval_studies),
        "generated_counts": {key: len(value) for key, value in generations_by_mode.items() if value},
    }
    write_json(output_root / "run_summary.json", summary)
    return summary
