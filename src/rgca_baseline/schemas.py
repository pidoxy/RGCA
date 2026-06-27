from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal


Split = Literal["retrieval_pool", "eval"]
Mode = Literal["no_retrieval", "retrieval", "mismatch"]


@dataclass
class StudyRecord:
    study_id: str
    image_path: str
    report_text: str
    findings: str
    impression: str
    labels: list[str] = field(default_factory=list)
    split: Split = "eval"
    subject_id: str | None = None
    dicom_id: str | None = None
    view_position: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetrievalResult:
    target_study: str
    mode: Literal["retrieval", "mismatch"]
    retrieved_studies: list[str]
    retrieved_reports: list[str]
    similarity_scores: list[float]
    retrieved_labels: list[list[str]] = field(default_factory=list)
    backend: str = "unknown"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GenerationResult:
    study_id: str
    mode: Mode
    prompt: str
    target_report: str
    retrieved_reports: list[str]
    generated_report: str
    retrieved_studies: list[str] = field(default_factory=list)
    labels_reference: list[str] = field(default_factory=list)
    labels_generated: list[str] = field(default_factory=list)
    hallucination_flags: list[str] = field(default_factory=list)
    retriever_backend: str = "unknown"
    generator_backend: str = "mock"

    def to_dict(self) -> dict:
        return asdict(self)
