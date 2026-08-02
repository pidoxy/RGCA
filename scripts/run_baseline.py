from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rgca_baseline.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the first RGCA baseline pipeline.")
    parser.add_argument("--input", required=True, help="Input study JSONL file.")
    parser.add_argument("--output-dir", required=True, help="Directory for outputs.")
    parser.add_argument(
        "--mode",
        default="all",
        choices=["no_retrieval", "retrieval", "mismatch", "all"],
        help="Baseline mode to run.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of retrieved reports.")
    parser.add_argument(
        "--retriever",
        default="lexical",
        choices=["lexical", "mock_image", "hashing_text"],
        help="Retriever backend to use.",
    )
    parser.add_argument(
        "--generator",
        default="mock",
        choices=["mock", "retrieval_copy_stress", "hf_vlm"],
        help="Generator backend to use.",
    )
    parser.add_argument("--model-id", help="HuggingFace model id for --generator hf_vlm.")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        mode=args.mode,
        top_k=args.top_k,
        retriever_backend=args.retriever,
        generator_backend=args.generator,
        generator_model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
    )
    print("Baseline run complete.")
    print(summary)


if __name__ == "__main__":
    main()
