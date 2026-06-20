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

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GenerationResult:
    study_id: str
    mode: Mode
    prompt: str
    generated_report: str
    retrieved_studies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)
