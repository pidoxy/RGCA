from __future__ import annotations

from collections import Counter

from rgca_baseline.schemas import RetrievalResult, StudyRecord
from rgca_baseline.text_utils import cosine_similarity, text_to_counter


def build_retrieval_text(study: StudyRecord) -> str:
    return " ".join([study.findings, study.impression, study.report_text]).strip()


class LexicalRetriever:
    """
    Lightweight retriever for the first runnable baseline.

    This is deliberately simple so the baseline runs with no external dependencies.
    Replace this class with an image-embedding retriever for the real experiment.
    """

    def __init__(self, pool: list[StudyRecord]) -> None:
        self.pool = pool
        self.pool_vectors: dict[str, Counter[str]] = {
            study.study_id: text_to_counter(build_retrieval_text(study)) for study in pool
        }

    def retrieve(self, query: StudyRecord, top_k: int) -> RetrievalResult:
        query_vector = text_to_counter(build_retrieval_text(query))
        scored = []
        for candidate in self.pool:
            score = cosine_similarity(query_vector, self.pool_vectors[candidate.study_id])
            scored.append((candidate, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        chosen = scored[:top_k]
        return RetrievalResult(
            target_study=query.study_id,
            mode="retrieval",
            retrieved_studies=[study.study_id for study, _ in chosen],
            retrieved_reports=[study.report_text for study, _ in chosen],
            similarity_scores=[round(score, 4) for _, score in chosen],
        )

    def mismatch(self, query: StudyRecord, top_k: int) -> RetrievalResult:
        query_labels = set(query.labels)
        candidates = []
        for candidate in self.pool:
            candidate_labels = set(candidate.labels)
            overlap = len(query_labels & candidate_labels)
            penalty = 0 if query_labels or candidate_labels else 1
            candidates.append((candidate, overlap, penalty))

        candidates.sort(key=lambda item: (item[1], item[2], item[0].study_id))
        chosen = [study for study, _, _ in candidates[:top_k]]
        return RetrievalResult(
            target_study=query.study_id,
            mode="mismatch",
            retrieved_studies=[study.study_id for study in chosen],
            retrieved_reports=[study.report_text for study in chosen],
            similarity_scores=[0.0 for _ in chosen],
        )
