from __future__ import annotations

from rgca_baseline.schemas import StudyRecord


def render_retrieved_reports(reports: list[str]) -> str:
    if not reports:
        return "None"
    return "\n".join(f"{index}. {report}" for index, report in enumerate(reports, start=1))


def build_prompt(mode: str, study: StudyRecord, retrieved_reports: list[str]) -> str:
    if mode == "no_retrieval":
        return (
            "You are a clinical radiology assistant.\n\n"
            "Given the target chest X-ray image, generate a concise radiology report.\n\n"
            "Return the report in this format:\n"
            "Findings:\n"
            "Impression:\n"
        )

    intro = (
        "You are a clinical radiology assistant.\n\n"
        "Retrieved similar reports:\n"
        f"{render_retrieved_reports(retrieved_reports)}\n\n"
        "Now examine the target chest X-ray image and generate a concise radiology report.\n"
        "Only include findings supported by the target image.\n\n"
        "Return the report in this format:\n"
        "Findings:\n"
        "Impression:\n"
    )
    return intro
