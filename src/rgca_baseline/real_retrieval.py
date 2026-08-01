from __future__ import annotations

from dataclasses import dataclass

from rgca_baseline.embeddings import BiomedCLIPStudyEmbedder, HashingTextEmbedder, MockImageEmbedder
from rgca_baseline.indexing import NumpyStudyIndex
from rgca_baseline.mismatch import (
    select_random_mismatch_candidates,
    select_wrong_pathology_candidates,
)
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

    def mismatch(self, query: StudyRecord, top_k: int) -> RetrievalResult:
        result = self.retriever.mismatch(query, top_k)
        result.backend = "lexical"
        result.retrieved_labels = [
            next((candidate.labels for candidate in self.retriever.pool if candidate.study_id == study_id), [])
            for study_id in result.retrieved_studies
        ]
        return result

    def retrieve(self, query: StudyRecord, top_k: int) -> RetrievalResult:
        result = self.retriever.retrieve(query, top_k)
        result.backend = "lexical"
        result.retrieved_labels = [
            next((candidate.labels for candidate in self.retriever.pool if candidate.study_id == study_id), [])
            for study_id in result.retrieved_studies
        ]
        return result


class IndexedRetrieverBackend(BaseRetrieverBackend):
    def __init__(self, pool: list[StudyRecord], embedder, backend_name: str, index: NumpyStudyIndex | None = None) -> None:
        self.pool = pool
        self.embedder = embedder
        self.backend_name = backend_name
        self.index = index or NumpyStudyIndex.build(pool, embedder)

    def retrieve(self, query: StudyRecord, top_k: int) -> RetrievalResult:
        results = self.index.search(self.embedder.embed_query(query), top_k)
        return RetrievalResult(
            target_study=query.study_id,
            mode="retrieval",
            retrieved_studies=[study.study_id for study, _ in results],
            retrieved_reports=[study.report_text for study, _ in results],
            similarity_scores=[round(score, 4) for _, score in results],
            retrieved_labels=[study.labels for study, _ in results],
            backend=self.backend_name,
        )

    def mismatch(self, query: StudyRecord, top_k: int) -> RetrievalResult:
        candidates = select_wrong_pathology_candidates(query, self.pool)
        if len(candidates) < top_k:
            candidates = select_random_mismatch_candidates(query, self.pool)
        chosen = candidates[:top_k]
        return RetrievalResult(
            target_study=query.study_id,
            mode="mismatch",
            retrieved_studies=[study.study_id for study in chosen],
            retrieved_reports=[study.report_text for study in chosen],
            similarity_scores=[0.0 for _ in chosen],
            retrieved_labels=[study.labels for study in chosen],
            backend=f"{self.backend_name}_mismatch",
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
    if backend == "mock_image":
        return RetrievalPlan(
            backend="mock_image",
            similarity_space="mock study-level image proxy embedding",
            notes="Runnable image-like retrieval path for index and artifact validation.",
        )
    if backend == "hashing_text":
        return RetrievalPlan(
            backend="hashing_text",
            similarity_space="hashed text embedding space",
            notes="Dependency-light embedding retrieval fallback using report text.",
        )
    raise ValueError(f"Unsupported backend: {backend}")


def create_retriever_backend(
    backend: str,
    pool: list[StudyRecord],
    index: NumpyStudyIndex | None = None,
) -> BaseRetrieverBackend:
    if backend == "lexical":
        return LexicalRetrieverBackend(pool)
    if backend == "mock_image":
        return IndexedRetrieverBackend(pool=pool, embedder=MockImageEmbedder(), backend_name=backend, index=index)
    if backend == "hashing_text":
        return IndexedRetrieverBackend(pool=pool, embedder=HashingTextEmbedder(), backend_name=backend, index=index)
    if backend == "biomedclip":
        embedder = BiomedCLIPStudyEmbedder()
        return IndexedRetrieverBackend(pool=pool, embedder=embedder, backend_name=backend, index=index)
    raise ValueError(f"Unsupported backend: {backend}")
