from __future__ import annotations

import hashlib

from rgca_baseline.retrieval import build_retrieval_text
from rgca_baseline.schemas import StudyRecord


def _normalize(vector: list[float]) -> list[float]:
    norm = sum(value * value for value in vector) ** 0.5
    if norm == 0:
        return [float(value) for value in vector]
    return [float(value / norm) for value in vector]


class StudyEmbedder:
    def embed_study(self, study: StudyRecord) -> list[float]:
        raise NotImplementedError

    def embed_query(self, study: StudyRecord) -> list[float]:
        return self.embed_study(study)


class HashingTextEmbedder(StudyEmbedder):
    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def embed_study(self, study: StudyRecord) -> list[float]:
        text = build_retrieval_text(study)
        values = [0.0] * self.dim
        for token in text.lower().split():
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            values[index] += sign
        return _normalize(values)


class MockImageEmbedder(StudyEmbedder):
    """
    Deterministic study-level image proxy embedder.

    This uses study metadata and labels to create a stable mock embedding so index,
    retrieval, and evaluation infrastructure can be exercised before real image
    encoders are available.
    """

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def embed_study(self, study: StudyRecord) -> list[float]:
        values = [0.0] * self.dim
        features = [study.image_path, study.study_id, " ".join(sorted(study.labels))]
        for feature in features:
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            for offset in range(0, min(len(digest), 32), 4):
                index = int.from_bytes(digest[offset : offset + 4], "big") % self.dim
                values[index] += 1.0
        for label in study.labels:
            digest = hashlib.sha1(label.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            values[index] += 3.0
        return _normalize(values)


class BiomedCLIPStudyEmbedder(StudyEmbedder):
    """
    Placeholder for the first real image-text embedding path.
    """

    def __init__(self, model_name: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224") -> None:
        self.model_name = model_name

    def embed_study(self, study: StudyRecord) -> list[float]:
        raise RuntimeError(
            "BiomedCLIPStudyEmbedder is a placeholder. Implement the actual image "
            f"embedding path for checkpoint: {self.model_name}"
        )
