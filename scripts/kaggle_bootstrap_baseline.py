from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from kaggle_run_pilot import build_balanced_mimic_subset
from package_private_kaggle_dataset import main as package_main
from rgca_baseline.io_utils import write_json
from rgca_baseline.mimic_cxr import write_study_records
from rgca_baseline.pipeline import load_studies
from run_experiment_suite import load_config, run_experiment, validate_config
from summarize_experiment_suite import collect_rows, write_csv, write_markdown


DEFAULT_SUBSET_CANDIDATES = [
    "/kaggle/working/rgca_pilot_500/data/mimic_subset.jsonl",
    "/kaggle/working/rgca_private_dataset/mimic_subset.jsonl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-command Kaggle bootstrap for the RGCA baseline suite. No gcloud required.",
    )
    parser.add_argument("--subset-jsonl", help="Optional explicit mimic_subset.jsonl path.")
    parser.add_argument("--physionet-root", default="/kaggle/working/physionet")
    parser.add_argument("--output-dir", default="/kaggle/working/rgca_experiments/mimic_pilot_baseline_v0")
    parser.add_argument("--pilot-output-dir", default="/kaggle/working/rgca_pilot_500")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "mimic_pilot_suite.json"))
    parser.add_argument("--retrieval-limit", type=int, default=400)
    parser.add_argument("--eval-limit", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--allow-demo", action="store_true", help="Fall back to repo demo data if MIMIC data is absent.")
    parser.add_argument("--no-package", action="store_true", help="Skip private dataset package creation.")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def section(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def find_first_file(filename: str, roots: list[Path]) -> Path | None:
    matches: list[Path] = []
    for root in roots:
        if root.exists():
            matches.extend(root.rglob(filename))
    return sorted(matches)[0] if matches else None


def find_dataset_files_dir(dataset_name: str, roots: list[Path]) -> Path | None:
    direct_candidates: list[Path] = []
    for root in roots:
        direct_candidates.extend(
            [
                root / dataset_name / "files",
                root / dataset_name / "2.1.0" / "files",
                root / dataset_name / "2.0.0" / "files",
            ]
        )
    for candidate in sorted(direct_candidates):
        if candidate.exists():
            return candidate

    discovered: list[Path] = []
    for root in roots:
        if root.exists():
            discovered.extend(path / "files" for path in root.rglob(dataset_name) if (path / "files").exists())
    return sorted(discovered)[0] if discovered else None


def discover_subset(explicit_subset: str | None, allow_demo: bool) -> Path | None:
    candidates: list[Path] = []
    if explicit_subset:
        candidates.append(Path(explicit_subset))
    candidates.extend(Path(path) for path in DEFAULT_SUBSET_CANDIDATES)
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        candidates.extend(sorted(kaggle_input.rglob("mimic_subset.jsonl")))
    if allow_demo:
        candidates.append(PROJECT_ROOT / "data" / "demo" / "demo_studies.jsonl")

    print("Subset candidates:")
    for candidate in candidates:
        print(f"- {candidate} | exists={candidate.exists()}")
    return next((candidate for candidate in candidates if candidate.exists()), None)


def build_subset_from_local_files(args: argparse.Namespace) -> Path | None:
    roots = [Path(args.physionet_root), Path("/kaggle/input")]
    metadata_path = find_first_file("mimic-cxr-2.0.0-metadata.csv.gz", roots)
    split_path = find_first_file("mimic-cxr-2.0.0-split.csv.gz", roots)
    labels_path = find_first_file("mimic-cxr-2.0.0-chexpert.csv.gz", roots)
    reports_root = find_dataset_files_dir("mimic-cxr", roots)
    images_root = find_dataset_files_dir("mimic-cxr-jpg", roots)

    required = {
        "metadata": metadata_path,
        "split": split_path,
        "labels": labels_path,
        "reports_root": reports_root,
        "images_root": images_root,
    }
    print("Local raw-data discovery:")
    for name, path in required.items():
        print(f"- {name}: {path} | exists={bool(path and path.exists())}")

    missing = {name: str(path) for name, path in required.items() if not path or not path.exists()}
    if missing:
        print("Cannot build subset from raw files because required inputs are missing:")
        print(json.dumps(missing, indent=2))
        return None

    output_path = Path(args.pilot_output_dir) / "data" / "mimic_subset.jsonl"
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
    write_study_records(output_path, studies)
    print(f"Built subset: {output_path} ({len(studies)} records)")
    return output_path


def validate_subset(subset_path: Path) -> dict:
    studies = load_studies(subset_path)
    retrieval_pool = [study for study in studies if study.split == "retrieval_pool"]
    eval_studies = [study for study in studies if study.split == "eval"]
    summary = {
        "subset_path": str(subset_path),
        "total": len(studies),
        "retrieval_pool": len(retrieval_pool),
        "eval": len(eval_studies),
    }
    print(json.dumps(summary, indent=2))
    if not retrieval_pool or not eval_studies:
        raise SystemExit("Invalid subset: it must contain both retrieval_pool and eval records.")
    return summary


def run_suite(config_path: Path, subset_path: Path, output_dir: Path, overwrite: bool) -> dict:
    config = load_config(config_path)
    validate_config(config)
    studies_by_id = {study.study_id: study for study in load_studies(subset_path)}
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for experiment in config["experiments"]:
        print(f"Running {experiment['name']}...")
        results.append(
            run_experiment(
                experiment=experiment,
                input_path=subset_path,
                suite_output_dir=output_dir,
                studies_by_id=studies_by_id,
                overwrite=overwrite,
            )
        )

    manifest = {
        "suite_name": config["suite_name"],
        "config_path": str(config_path),
        "input_path": str(subset_path),
        "output_dir": str(output_dir),
        "experiments": results,
    }
    write_json(output_dir / "suite_manifest.json", manifest)
    return manifest


def summarize_suite(manifest: dict, output_dir: Path) -> Path:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(manifest)
    write_csv(tables_dir / "suite_summary.csv", rows)
    write_markdown(tables_dir / "suite_summary.md", rows)
    print(f"Wrote {len(rows)} table rows to {tables_dir}")
    return tables_dir


def package_outputs(subset_path: Path, output_dir: Path) -> Path:
    package_dir = Path("/kaggle/working/rgca_private_dataset")
    if not Path("/kaggle/working").exists():
        package_dir = PROJECT_ROOT / "outputs" / "rgca_private_dataset"
    if package_dir.exists():
        shutil.rmtree(package_dir)

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "package_private_kaggle_dataset.py",
            "--subset-jsonl",
            str(subset_path),
            "--output-dir",
            str(package_dir),
            "--suite-output-dir",
            str(output_dir),
            "--zip",
        ]
        package_main()
    finally:
        sys.argv = old_argv
    return package_dir.with_suffix(".zip")


def main() -> None:
    args = parse_args()

    section("1. Discover Subset")
    subset_path = discover_subset(args.subset_jsonl, args.allow_demo)

    if subset_path is None:
        section("2. Build Subset From Existing Local Files")
        subset_path = build_subset_from_local_files(args)

    if subset_path is None:
        raise SystemExit(
            "\nNo usable subset was found or built.\n"
            "Fastest fix: attach/upload a private Kaggle dataset containing mimic_subset.jsonl, "
            "or restore /kaggle/working/rgca_pilot_500/data/mimic_subset.jsonl.\n"
            "For a code-only smoke test, rerun with --allow-demo."
        )

    section("3. Validate Subset")
    subset_summary = validate_subset(subset_path)

    section("4. Run Experiment Suite")
    output_dir = Path(args.output_dir)
    manifest = run_suite(
        config_path=Path(args.config),
        subset_path=subset_path,
        output_dir=output_dir,
        overwrite=args.overwrite,
    )

    section("5. Summarize Results")
    tables_dir = summarize_suite(manifest, output_dir)
    summary_md = tables_dir / "suite_summary.md"
    print(summary_md.read_text(encoding="utf-8")[:4000])

    package_zip = None
    if not args.no_package:
        section("6. Package Private Dataset")
        package_zip = package_outputs(subset_path, output_dir)
        print(f"Private dataset zip: {package_zip}")

    section("Complete")
    bootstrap_summary = {
        "subset": subset_summary,
        "suite_manifest": str(output_dir / "suite_manifest.json"),
        "tables_dir": str(tables_dir),
        "private_dataset_zip": str(package_zip) if package_zip else None,
    }
    write_json(output_dir / "bootstrap_summary.json", bootstrap_summary)
    print(json.dumps(bootstrap_summary, indent=2))


if __name__ == "__main__":
    main()
