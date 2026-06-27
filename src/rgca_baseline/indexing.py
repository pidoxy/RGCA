from __future__ import annotations

import json
from pathlib import Path

from rgca_baseline.schemas import StudyRecord


class NumpyStudyIndex:
    def __init__(self, embeddings: list[list[float]], metadata: list[StudyRecord]) -> None:
        self.embeddings = embeddings
        self.metadata = metadata

    @classmethod
    def build(cls, records: list[StudyRecord], embedder) -> "NumpyStudyIndex":
        matrix = [embedder.embed_study(record) for record in records]
        return cls(matrix, records)

    def save(self, output_dir: str | Path) -> None:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "embeddings.json").write_text(
            json.dumps(self.embeddings, ensure_ascii=True),
            encoding="utf-8",
        )
        with (target / "metadata.jsonl").open("w", encoding="utf-8") as handle:
            for study in self.metadata:
                handle.write(json.dumps(study.to_dict(), ensure_ascii=True) + "\n")

    @classmethod
    def load(cls, index_dir: str | Path) -> "NumpyStudyIndex":
        target = Path(index_dir)
        embeddings = json.loads((target / "embeddings.json").read_text(encoding="utf-8"))
        metadata: list[StudyRecord] = []
        with (target / "metadata.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                metadata.append(StudyRecord(**json.loads(line)))
        return cls(embeddings, metadata)

    def search(self, query_embedding: list[float], top_k: int) -> list[tuple[StudyRecord, float]]:
        query_norm = sum(value * value for value in query_embedding) ** 0.5
        if query_norm == 0:
            query = [float(value) for value in query_embedding]
        else:
            query = [float(value / query_norm) for value in query_embedding]

        scores: list[tuple[int, float]] = []
        for idx, embedding in enumerate(self.embeddings):
            score = sum(left * right for left, right in zip(embedding, query))
            scores.append((idx, float(score)))
        scores.sort(key=lambda item: item[1], reverse=True)
        return [(self.metadata[idx], score) for idx, score in scores[:top_k]]
