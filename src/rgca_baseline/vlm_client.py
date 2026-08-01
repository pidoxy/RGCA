from __future__ import annotations

from rgca_baseline.generator import MockVLMGenerator, RetrievalCopyStressGenerator
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


class RetrievalCopyStressVLMClient(BaseVLMClient):
    backend_name = "retrieval_copy_stress"

    def __init__(self) -> None:
        self.generator = RetrievalCopyStressGenerator()

    def generate_report(
        self,
        study: StudyRecord,
        prompt: str,
        retrieved_reports: list[str] | None = None,
        mode: str = "no_retrieval",
    ) -> str:
        return self.generator.generate(study=study, mode=mode, retrieved_reports=retrieved_reports or [])


def create_vlm_client(backend: str) -> BaseVLMClient:
    if backend == "mock":
        return MockVLMClient()
    if backend == "retrieval_copy_stress":
        return RetrievalCopyStressVLMClient()
    raise ValueError(f"Unsupported generator backend: {backend}")
