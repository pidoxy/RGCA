from __future__ import annotations

import os
from pathlib import Path

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


class HuggingFaceVisionLanguageClient(BaseVLMClient):
    backend_name = "hf_vlm"

    def __init__(
        self,
        model_id: str,
        max_new_tokens: int = 192,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = True,
    ) -> None:
        if not model_id:
            raise ValueError(
                "A HuggingFace VLM model id is required. Set RGCA_VLM_MODEL_ID or pass --model-id."
            )

        try:
            import torch
            from PIL import Image
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "The hf_vlm backend requires optional dependencies. Install them with:\n"
                "  pip install transformers accelerate pillow torch\n"
                "or use a Kaggle notebook image that already includes PyTorch."
            ) from exc

        try:
            from transformers import AutoModelForVision2Seq
        except ImportError:
            AutoModelForVision2Seq = None

        try:
            from transformers import LlavaForConditionalGeneration
        except ImportError:
            LlavaForConditionalGeneration = None

        self.torch = torch
        self.image_cls = Image
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        model_kwargs = {
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
        }
        if torch_dtype != "auto":
            model_kwargs["torch_dtype"] = getattr(torch, torch_dtype)
        else:
            model_kwargs["torch_dtype"] = "auto"

        model_id_lower = model_id.lower()
        if "llava" in model_id_lower and LlavaForConditionalGeneration is not None:
            self.model = LlavaForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
        elif AutoModelForVision2Seq is not None:
            try:
                self.model = AutoModelForVision2Seq.from_pretrained(model_id, **model_kwargs)
            except (ValueError, OSError):
                self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

        self.max_new_tokens = max_new_tokens

    def _format_prompt(self, prompt: str) -> str:
        if "llava" not in self.model_id.lower():
            return prompt

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        if hasattr(self.processor, "apply_chat_template"):
            try:
                return self.processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            except Exception:
                pass

        return f"USER: <image>\n{prompt}\nASSISTANT:"

    def generate_report(
        self,
        study: StudyRecord,
        prompt: str,
        retrieved_reports: list[str] | None = None,
        mode: str = "no_retrieval",
    ) -> str:
        image_path = Path(study.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image required by hf_vlm backend is missing: {image_path}")

        image = self.image_cls.open(image_path).convert("RGB")
        formatted_prompt = self._format_prompt(prompt)
        inputs = self.processor(text=formatted_prompt, images=image, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        with self.torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        input_length = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
        generated_ids = output_ids[:, input_length:] if input_length else output_ids
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        if text:
            return text
        return self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


def create_vlm_client(
    backend: str,
    model_id: str | None = None,
    max_new_tokens: int = 192,
) -> BaseVLMClient:
    if backend == "mock":
        return MockVLMClient()
    if backend == "retrieval_copy_stress":
        return RetrievalCopyStressVLMClient()
    if backend == "hf_vlm":
        return HuggingFaceVisionLanguageClient(
            model_id=model_id or os.environ.get("RGCA_VLM_MODEL_ID", ""),
            max_new_tokens=max_new_tokens,
        )
    raise ValueError(f"Unsupported generator backend: {backend}")
