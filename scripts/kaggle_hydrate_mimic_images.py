from __future__ import annotations

import argparse
import getpass
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rgca_baseline.integrity import validate_image_paths
from rgca_baseline.io_utils import write_json
from rgca_baseline.mimic_cxr import write_study_records
from rgca_baseline.pipeline import load_studies
from rgca_baseline.schemas import StudyRecord


JPG_BASE_URL = "https://physionet.org/files/mimic-cxr-jpg/2.1.0/files"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download only the MIMIC-CXR-JPG images referenced by a pilot subset. "
            "This avoids downloading the full multi-terabyte image archive."
        )
    )
    parser.add_argument("--subset", required=True, help="Input MIMIC pilot subset JSONL.")
    parser.add_argument(
        "--physionet-user",
        help="PhysioNet username. Defaults to PHYSIONET_USERNAME or PHYSIONET_USER.",
    )
    parser.add_argument(
        "--output-root",
        default="/kaggle/working/physionet/mimic-cxr-jpg/files",
        help="Local root where the MIMIC-CXR-JPG files tree should be hydrated.",
    )
    parser.add_argument(
        "--updated-subset",
        default="/kaggle/working/rgca_hydrated_subset/mimic_subset.jsonl",
        help="Output subset JSONL with image_path values rewritten to hydrated JPGs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap for smoke testing. Omit for the full pilot subset.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Re-download files even if they already exist.")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def make_netrc(username: str, password: str) -> Path:
    netrc_path = Path.home() / ".netrc"
    netrc_path.write_text(f"machine physionet.org login {username} password {password}\n", encoding="utf-8")
    netrc_path.chmod(0o600)
    return netrc_path


def image_relative_path(study: StudyRecord) -> Path:
    if not study.subject_id or not study.study_id or not study.dicom_id:
        raise ValueError(
            f"Study {study.study_id!r} is missing subject_id/study_id/dicom_id; "
            "cannot derive its MIMIC-CXR-JPG download path."
        )
    subject_id = str(study.subject_id)
    study_id = str(study.study_id)
    dicom_id = str(study.dicom_id)
    return Path(f"p{subject_id[:2]}") / f"p{subject_id}" / f"s{study_id}" / f"{dicom_id}.jpg"


def download_image(relative_path: Path, destination: Path, overwrite: bool) -> str:
    if destination.exists() and not overwrite:
        return "already_exists"
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"{JPG_BASE_URL}/{relative_path.as_posix()}"
    run(["wget", "-q", "-c", "--netrc", url, "-O", str(destination)])
    return "downloaded"


def main() -> None:
    args = parse_args()
    subset_path = Path(args.subset)
    if not subset_path.exists():
        raise SystemExit(f"Subset JSONL does not exist: {subset_path}")

    username = args.physionet_user or os.environ.get("PHYSIONET_USERNAME") or os.environ.get("PHYSIONET_USER")
    if not username:
        raise SystemExit("Missing PhysioNet username. Set PHYSIONET_USERNAME or pass --physionet-user.")
    password = os.environ.get("PHYSIONET_PASS") or os.environ.get("PHYSIONET_PASSWORD")
    if not password:
        password = getpass.getpass("PhysioNet password: ")

    studies = load_studies(subset_path)
    if args.limit is not None:
        studies = studies[: args.limit]
    if not studies:
        raise SystemExit("No studies found in subset.")

    output_root = Path(args.output_root)
    updated_subset = Path(args.updated_subset)
    manifest_rows = []
    netrc_path = make_netrc(username, password)
    try:
        hydrated: list[StudyRecord] = []
        for index, study in enumerate(studies, start=1):
            relative = image_relative_path(study)
            destination = output_root / relative
            status = download_image(relative, destination, args.overwrite)
            row = study.to_dict()
            row["image_path"] = str(destination)
            hydrated.append(StudyRecord(**row))
            manifest_rows.append(
                {
                    "study_id": study.study_id,
                    "subject_id": study.subject_id,
                    "dicom_id": study.dicom_id,
                    "relative_path": relative.as_posix(),
                    "image_path": str(destination),
                    "status": status,
                }
            )
            if index % 25 == 0 or index == len(studies):
                print(f"Hydrated {index}/{len(studies)} images")
    finally:
        netrc_path.unlink(missing_ok=True)

    write_study_records(updated_subset, hydrated)
    image_validation = validate_image_paths(hydrated)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_subset": str(subset_path),
        "updated_subset": str(updated_subset),
        "output_root": str(output_root),
        "requested_records": len(studies),
        "downloaded": sum(1 for row in manifest_rows if row["status"] == "downloaded"),
        "already_exists": sum(1 for row in manifest_rows if row["status"] == "already_exists"),
        "image_validation": image_validation,
        "records": manifest_rows,
    }
    write_json(updated_subset.parent / "image_hydration_manifest.json", manifest)

    print(json.dumps({key: value for key, value in manifest.items() if key != "records"}, indent=2))
    if not image_validation["valid"]:
        raise SystemExit("Image hydration finished, but some expected images are still missing.")


if __name__ == "__main__":
    main()
