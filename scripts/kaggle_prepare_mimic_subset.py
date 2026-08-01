from __future__ import annotations

import argparse
import getpass
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from kaggle_run_pilot import build_balanced_mimic_subset
from rgca_baseline.mimic_cxr import write_study_records


JPG_BASE_URL = "https://physionet.org/files/mimic-cxr-jpg/2.1.0"
REPORTS_URL = "https://physionet.org/files/mimic-cxr/2.1.0/mimic-cxr-reports.zip"
SMALL_JPG_FILES = [
    "mimic-cxr-2.0.0-metadata.csv.gz",
    "mimic-cxr-2.0.0-split.csv.gz",
    "mimic-cxr-2.0.0-chexpert.csv.gz",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a MIMIC-CXR pilot subset on Kaggle without GCloud.",
    )
    parser.add_argument("--physionet-user", required=True, help="PhysioNet username.")
    parser.add_argument("--work-dir", default="/kaggle/working/physionet")
    parser.add_argument("--output-dir", default="/kaggle/working/rgca_pilot_500")
    parser.add_argument("--retrieval-limit", type=int, default=400)
    parser.add_argument("--eval-limit", type=int, default=100)
    parser.add_argument("--skip-download", action="store_true", help="Use already-downloaded files only.")
    parser.add_argument(
        "--require-images",
        action="store_true",
        help="Require a local mimic-cxr-jpg/files directory. Needed for real VLM experiments.",
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def download_with_netrc(url: str, destination: Path, username: str, password: str) -> None:
    if destination.exists():
        print("Already exists:", destination)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    netrc_path = Path.home() / ".netrc"
    netrc_path.write_text(f"machine physionet.org login {username} password {password}\n", encoding="utf-8")
    netrc_path.chmod(0o600)
    try:
        run(["wget", "-c", "--netrc", url, "-O", str(destination)])
    finally:
        netrc_path.unlink(missing_ok=True)


def find_reports_root(work_dir: Path) -> Path:
    candidates = [
        work_dir / "mimic-cxr" / "files",
        work_dir / "mimic-cxr" / "mimic-cxr-reports" / "files",
        work_dir / "mimic-cxr-reports" / "files",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(path / "files" for path in work_dir.rglob("mimic-cxr-reports") if (path / "files").exists())
    if matches:
        return matches[0]
    raise FileNotFoundError("Could not find extracted MIMIC-CXR reports files directory.")


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir)
    output_dir = Path(args.output_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = work_dir / "mimic-cxr-2.0.0-metadata.csv.gz"
    split_path = work_dir / "mimic-cxr-2.0.0-split.csv.gz"
    labels_path = work_dir / "mimic-cxr-2.0.0-chexpert.csv.gz"
    reports_zip = work_dir / "mimic-cxr-reports.zip"
    reports_extract_dir = work_dir / "mimic-cxr"
    images_root = work_dir / "mimic-cxr-jpg" / "files"

    if not args.skip_download:
        password = getpass.getpass("PhysioNet password: ")
        for filename in SMALL_JPG_FILES:
            download_with_netrc(f"{JPG_BASE_URL}/{filename}", work_dir / filename, args.physionet_user, password)
        download_with_netrc(REPORTS_URL, reports_zip, args.physionet_user, password)

    if reports_zip.exists():
        reports_extract_dir.mkdir(parents=True, exist_ok=True)
        if not any(reports_extract_dir.rglob("s*.txt")):
            run(["unzip", "-q", str(reports_zip), "-d", str(reports_extract_dir)])

    reports_root = None
    if reports_extract_dir.exists():
        try:
            reports_root = find_reports_root(work_dir)
        except FileNotFoundError:
            reports_root = None

    missing = [
        str(path)
        for path in [metadata_path, split_path, labels_path]
        if not path.exists()
    ]
    if reports_root is None:
        missing.append(str(reports_extract_dir / "files"))
    if args.require_images and not images_root.exists():
        missing.append(str(images_root))
    if missing:
        raise SystemExit("Missing required files/folders:\n" + "\n".join(missing))

    shim = argparse.Namespace(
        metadata=str(metadata_path),
        split=str(split_path),
        labels=str(labels_path),
        reports_root=str(reports_root),
        images_root=str(images_root),
        views=["PA", "AP"],
        dataset_splits=["train", "validate"],
        limit=args.retrieval_limit + args.eval_limit,
        retrieval_limit=args.retrieval_limit,
        eval_limit=args.eval_limit,
    )
    studies = build_balanced_mimic_subset(shim)
    subset_path = output_dir / "data" / "mimic_subset.jsonl"
    write_study_records(subset_path, studies)
    print(f"Prepared subset: {subset_path}")
    print(f"Records: {len(studies)}")
    print("Next command:")
    print(
        "python scripts/kaggle_bootstrap_baseline.py "
        f"--subset-jsonl {subset_path} "
        "--execution-mode stress "
        "--overwrite"
    )


if __name__ == "__main__":
    main()
