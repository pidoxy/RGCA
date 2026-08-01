from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package private RGCA pilot artifacts for a Kaggle dataset upload.")
    parser.add_argument("--subset-jsonl", required=True, help="Study JSONL to include as mimic_subset.jsonl.")
    parser.add_argument("--output-dir", required=True, help="Output folder for private dataset contents.")
    parser.add_argument("--suite-output-dir", help="Optional experiment suite output folder to include.")
    parser.add_argument("--images-root", help="Optional local pilot image root to include.")
    parser.add_argument("--zip", action="store_true", help="Also create a .zip next to output-dir.")
    return parser.parse_args()


def copytree_if_exists(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subset_path = Path(args.subset_jsonl)
    if not subset_path.exists():
        raise SystemExit(f"Subset JSONL does not exist: {subset_path}")

    shutil.copy2(subset_path, output_dir / "mimic_subset.jsonl")

    if args.suite_output_dir:
        copytree_if_exists(Path(args.suite_output_dir), output_dir / "rgca_experiment_outputs")

    if args.images_root:
        copytree_if_exists(Path(args.images_root), output_dir / "mimic-cxr-jpg" / "files")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "subset_jsonl": "mimic_subset.jsonl",
        "contains_clinical_text": True,
        "visibility_required": "private",
        "notes": (
            "This package may contain MIMIC-CXR-derived report text and must remain private. "
            "Do not publish it publicly or commit it to GitHub."
        ),
    }
    (output_dir / "PRIVATE_DATASET_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.zip:
        archive_base = output_dir.with_suffix("")
        zip_path = shutil.make_archive(str(archive_base), "zip", root_dir=str(output_dir))
        print(f"Created zip: {zip_path}")

    print(f"Private dataset folder ready: {output_dir}")


if __name__ == "__main__":
    main()
