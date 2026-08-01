from __future__ import annotations

import json
from pathlib import Path

from rgca_baseline.schemas import StudyRecord


LABEL_PATTERNS = {
    "pleural effusion": ["pleural effusion", "pleural effusions"],
    "cardiomegaly": ["cardiomegaly", "enlarged cardiac silhouette", "mild cardiomegaly"],
    "pneumonia": ["pneumonia", "airspace opacity", "consolidation"],
    "pulmonary edema": ["pulmonary edema", "vascular congestion", "interstitial opacities"],
    "atelectasis": ["atelectatic", "atelectasis"],
}


def normalize_label(label: str) -> str:
    return label.strip().lower()


def normalize_labels(labels: list[str]) -> list[str]:
    return sorted({normalize_label(label) for label in labels if label.strip()})


def infer_labels_from_text(text: str) -> list[str]:
    lowered = text.lower()
    labels = []
    for label, patterns in LABEL_PATTERNS.items():
        if any(pattern in lowered for pattern in patterns):
            labels.append(normalize_label(label))
    return sorted(set(labels))


def evaluate_generation_rows(
    studies_by_id: dict[str, StudyRecord],
    generation_rows: list[dict],
    retrieval_rows_by_target: dict[str, dict],
) -> tuple[list[dict], dict]:
    detailed_rows: list[dict] = []
    n = len(generation_rows)
    hallucination_cases = 0
    retrieval_induced_cases = 0
    retrieval_copy_cases = 0
    total_hallucinated_labels = 0
    total_retrieval_induced_hallucinations = 0

    for row in generation_rows:
        study = studies_by_id[row["study_id"]]
        generated_labels = infer_labels_from_text(row["generated_report"])
        retrieval_row = retrieval_rows_by_target.get(row["study_id"], {})
        retrieved_labels = normalize_labels(
            [
                label
                for labels in retrieval_row.get("retrieved_labels", [])
                for label in labels
            ]
        )
        reference_labels = normalize_labels(study.labels)
        hallucinated = sorted(set(generated_labels) - set(reference_labels))
        retrieval_induced = sorted(set(hallucinated) & set(retrieved_labels))
        copied = sorted(set(generated_labels) & set(retrieved_labels))

        if hallucinated:
            hallucination_cases += 1
            total_hallucinated_labels += len(hallucinated)
        if retrieval_induced:
            retrieval_induced_cases += 1
            total_retrieval_induced_hallucinations += len(retrieval_induced)
        if copied:
            retrieval_copy_cases += 1

        detailed_rows.append(
            {
                "study_id": study.study_id,
                "mode": row["mode"],
                "reference_labels": reference_labels,
                "retrieved_labels": retrieved_labels,
                "generated_labels": generated_labels,
                "hallucination_flags": hallucinated,
                "retrieval_induced_flags": retrieval_induced,
                "retrieval_copy_flags": copied,
            }
        )

    summary = {
        generation_rows[0]["mode"] if generation_rows else "unknown": {
            "n": n,
            "hallucination_rate": (hallucination_cases / n) if n else 0.0,
            "retrieval_induced_hallucination_rate": (retrieval_induced_cases / n) if n else 0.0,
            "retrieval_copy_rate": (retrieval_copy_cases / n) if n else 0.0,
            "total_hallucinated_labels": total_hallucinated_labels,
            "total_retrieval_induced_hallucinations": total_retrieval_induced_hallucinations,
        }
    }
    return detailed_rows, summary


def write_evaluation_outputs(output_dir: str | Path, detailed_rows: list[dict], summary: dict) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "evaluation_details.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in detailed_rows) + ("\n" if detailed_rows else ""),
        encoding="utf-8",
    )
    (target / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
