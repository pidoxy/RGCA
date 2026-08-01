from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rgca_baseline.embeddings import BiomedCLIPStudyEmbedder, HashingTextEmbedder, MockImageEmbedder
from rgca_baseline.indexing import NumpyStudyIndex
from rgca_baseline.pipeline import load_studies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a retrieval index for RGCA baseline studies.")
    parser.add_argument("--subset", required=True, help="Study subset JSONL path.")
    parser.add_argument(
        "--backend",
        default="mock_image",
        choices=["mock_image", "hashing_text", "biomedclip"],
        help="Embedding backend used to create the index.",
    )
    parser.add_argument("--output-dir", required=True, help="Output index directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    studies = load_studies(args.subset)
    pool = [study for study in studies if study.split == "retrieval_pool"]
    if args.backend == "mock_image":
        embedder = MockImageEmbedder()
    elif args.backend == "hashing_text":
        embedder = HashingTextEmbedder()
    else:
        embedder = BiomedCLIPStudyEmbedder()
    index = NumpyStudyIndex.build(pool, embedder)
    index.save(args.output_dir)
    print(
        f"Built retrieval index with backend={args.backend}, pool_size={len(pool)}, "
        f"output_dir={args.output_dir}"
    )


if __name__ == "__main__":
    main()
