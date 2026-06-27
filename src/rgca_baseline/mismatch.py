from __future__ import annotations

import random

from rgca_baseline.schemas import StudyRecord


def select_wrong_pathology_candidates(query: StudyRecord, pool: list[StudyRecord]) -> list[StudyRecord]:
    query_labels = set(query.labels)
    if not query_labels:
        return [candidate for candidate in pool if candidate.labels]
    return [
        candidate
        for candidate in pool
        if candidate.study_id != query.study_id and not (query_labels & set(candidate.labels))
    ]


def select_random_mismatch_candidates(query: StudyRecord, pool: list[StudyRecord], seed: int = 42) -> list[StudyRecord]:
    generator = random.Random(f"{seed}:{query.study_id}")
    candidates = [candidate for candidate in pool if candidate.study_id != query.study_id]
    generator.shuffle(candidates)
    return candidates
