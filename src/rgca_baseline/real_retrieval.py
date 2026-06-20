from __future__ import annotations

from dataclasses import dataclass

from rgca_baseline.retrieval import LexicalRetriever
from rgca_baseline.schemas import RetrievalResult, StudyRecord


@dataclass
class RetrievalPlan:
    backend: str
    similarity_space: str
    notes: str


class BaseRetrieverBackend:
    def retrieve(self, query: StudyRecord, top_k: int) -> RetrievalResult:
        raise NotImplementedError

    def mismatch(self, query: StudyRecord, top_k: int) -> RetrievalResult:
        raise NotImplementedError


class LexicalRetrieverBackend(BaseRetrieverBackend):
    def __init__(self, pool: list[StudyRecord]) -> None:
        self.retriever = LexicalRetriever(pool)

    def retrieve(self, query: StudyRecord, top_k: int) -> RetrievalResult:
        return self.retriever.retrieve(query, top_k)

    def mismatch(self, query: StudyRecord, top_k: int) -> RetrievalResult:
        return self.retriever.mismatch(query, top_k)


class BiomedCLIPRetrieverBackend(BaseRetrieverBackend):
    """
    First real retrieval integration point.

    This backend is intentionally a scaffold. It defines the contract and the setup path
    for image-text embedding retrieval, but it does not download or initialize large
    model weights automatically in this environment.
    """

    def __init__(self, pool: list[StudyRecord], model_name: str) -> None:
        self.pool = pool
        self.model_name = model_name
        self._dependency_error = self._check_dependencies()

    def _check_dependencies(self) -> Exception | None:
        try:
            import torch  # noqa: F401
            from PIL import Image  # noqa: F401
            from transformers import AutoModel, AutoProcessor  # noqa: F401
        except Exception as exc:  # pragma: no cover - setup dependent
            return exc
        return RuntimeError(
            "BiomedCLIP retrieval scaffold is present, but model-specific embedding "
            "implementation still needs to be completed for your chosen checkpoint."
        )

    def retrieve(self, query: StudyRecord, top_k: int) -> RetrievalResult:
        raise RuntimeError(
            "BiomedCLIPRetrieverBackend is a scaffold. Finish the embedding path after "
            f"installing dependencies and choosing the exact checkpoint. Setup issue: {self._dependency_error}"
        )

    def mismatch(self, query: StudyRecord, top_k: int) -> RetrievalResult:
        raise RuntimeError(
            "Mismatch retrieval for the BiomedCLIP backend should be derived after "
            "the real embedding retrieval path is implemented."
        )


def get_retrieval_plan(backend: str) -> RetrievalPlan:
    if backend == "lexical":
        return RetrievalPlan(
            backend="lexical",
            similarity_space="report text",
            notes="Debug backend only. Useful for pipeline validation, not final inference.",
        )
    if backend == "biomedclip":
        return RetrievalPlan(
            backend="biomedclip",
            similarity_space="shared image-text embedding space",
            notes=(
                "Recommended first real retrieval path. Query with target image embedding "
                "and retrieve nearest studies, then pass their reports to the generator."
            ),
        )
    raise ValueError(f"Unsupported backend: {backend}")
