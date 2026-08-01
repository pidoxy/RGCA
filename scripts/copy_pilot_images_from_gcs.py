from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rgca_baseline.pipeline import load_studies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy only pilot subset JPG images from a GCS MIMIC-CXR-JPG bucket.")
    parser.add_argument("--subset-jsonl", required=True, help="Study JSONL containing image_path fields.")
    parser.add_argument("--images-root", required=True, help="Local image root used in image_path values.")
    parser.add_argument("--gcs-bucket", required=True, help="GCS bucket, e.g. gs://mimic-cxr-jpg-2.1.0.physionet.org")
    parser.add_argument("--output-root", required=True, help="Local output image root to copy files into.")
    parser.add_argument("--dry-run", action="store_true", help="Print copy commands without executing them.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    studies = load_studies(args.subset_jsonl)
    images_root = Path(args.images_root)
    output_root = Path(args.output_root)
    bucket = args.gcs_bucket.rstrip("/")
    copied = 0
    skipped = 0

    for study in studies:
        source_path = Path(study.image_path)
        try:
            relative_path = source_path.relative_to(images_root)
        except ValueError:
            relative_path = Path(*source_path.parts[-4:])

        destination = output_root / relative_path
        if destination.exists():
            skipped += 1
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        source_uri = f"{bucket}/files/{relative_path.as_posix()}"
        command = ["gcloud", "storage", "cp", source_uri, str(destination)]
        print("+", " ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)
        copied += 1

    print(f"Copied {copied} images. Skipped existing images: {skipped}.")


if __name__ == "__main__":
    main()
