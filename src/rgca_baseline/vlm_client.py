from __future__ import annotations

from rgca_baseline.generator import MockVLMGenerator
from rgca_baseline.schemas import StudyRecord


class BaseVLMClient:
    backend_name = "unknown"

    def generate_report(
        self,
        study: StudyRecord,
        prompt: str,
        retrieved_reports: list[str] | None = None,
        mode: str = "no_retrieval",
    ) -> str:
        raise NotImplementedError


class MockVLMClient(BaseVLMClient):
    backend_name = "mock"

    def __init__(self) -> None:
        self.generator = MockVLMGenerator()

    def generate_report(
        self,
        study: StudyRecord,
        prompt: str,
        retrieved_reports: list[str] | None = None,
        mode: str = "no_retrieval",
    ) -> str:
        return self.generator.generate(study=study, mode=mode, retrieved_reports=retrieved_reports or [])
