from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

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
    Image embedding backend for the first real retrieval path.

    Dependencies are intentionally optional so the rest of the repository remains
    runnable on laptops and in lightweight CI. Install them only in the GPU
    environment used for real retrieval:

    pip install open_clip_torch torch pillow
    """

    def __init__(
        self,
        model_name: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model: Any | None = None
        self._preprocess: Any | None = None
        self._torch: Any | None = None
        self._image_cls: Any | None = None

    def _load(self) -> None:
        if self._model is not None:
            return

        try:
            import open_clip
            import torch
            from PIL import Image
        except Exception as exc:  # pragma: no cover - depends on GPU notebook setup
            raise RuntimeError(
                "BiomedCLIP retrieval requires optional dependencies. Install them in "
                "the execution environment with: pip install open_clip_torch torch pillow"
            ) from exc

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, preprocess = open_clip.create_model_from_pretrained(self.model_name)
        model.eval()
        model.to(device)

        self.device = device
        self._model = model
        self._preprocess = preprocess
        self._torch = torch
        self._image_cls = Image

    def _embed_image_path(self, image_path: str) -> list[float]:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(
                f"BiomedCLIP image embedding requires an existing image file: {path}"
            )

        self._load()
        assert self._model is not None
        assert self._preprocess is not None
        assert self._torch is not None
        assert self._image_cls is not None

        image = self._image_cls.open(path).convert("RGB")
        tensor = self._preprocess(image).unsqueeze(0).to(self.device)
        with self._torch.inference_mode():
            embedding = self._model.encode_image(tensor)
        vector = embedding.squeeze(0).detach().cpu().float().tolist()
        return _normalize(vector)

    def embed_study(self, study: StudyRecord) -> list[float]:
        return self._embed_image_path(study.image_path)
