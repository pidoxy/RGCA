from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from rgca_baseline.io_utils import read_jsonl
from rgca_baseline.schemas import StudyRecord


DEBUG_RETRIEVERS = {"lexical", "hashing_text", "mock_image"}
DEBUG_GENERATORS = {"mock", "retrieval_copy_stress"}
STRESS_RETRIEVERS = {"lexical", "hashing_text"}
REAL_RETRIEVERS = {"biomedclip"}
REAL_GENERATORS: set[str] = set()
REPO_ROOT = Path(__file__).resolve().parents[2]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_fingerprint(path: str | Path) -> dict:
    rows = read_jsonl(path)
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, ensure_ascii=True).encode("utf-8"))
        digest.update(b"\n")
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "rows": len(rows),
    }


def validate_study_records(studies: Iterable[StudyRecord]) -> dict:
    studies = list(studies)
    retrieval_pool = [study for study in studies if study.split == "retrieval_pool"]
    eval_studies = [study for study in studies if study.split == "eval"]
    missing_required = [
        study.study_id
        for study in studies
        if not study.study_id or not study.report_text or not study.findings or not study.image_path
    ]
    id_counts = Counter(study.study_id for study in studies)
    duplicate_ids = sorted(study_id for study_id, count in id_counts.items() if count > 1)
    return {
        "total": len(studies),
        "retrieval_pool": len(retrieval_pool),
        "eval": len(eval_studies),
        "missing_required_count": len(missing_required),
        "missing_required_study_ids": missing_required[:20],
        "duplicate_study_ids": duplicate_ids[:20],
        "valid": bool(studies and retrieval_pool and eval_studies and not missing_required and not duplicate_ids),
    }


def validate_image_paths(studies: Iterable[StudyRecord]) -> dict:
    studies = list(studies)
    missing = [study.image_path for study in studies if not Path(study.image_path).exists()]
    return {
        "total": len(studies),
        "missing_image_count": len(missing),
        "missing_image_examples": missing[:20],
        "valid": not missing,
    }


def is_demo_dataset(path: str | Path) -> bool:
    target = Path(path).resolve()
    demo_root = (REPO_ROOT / "data" / "demo").resolve()
    return target == demo_root / "demo_studies.jsonl" or demo_root in target.parents


def assert_dataset_allowed(path: str | Path, execution_mode: str) -> None:
    if execution_mode != "debug" and is_demo_dataset(path):
        raise ValueError(
            f"Demo dataset is not allowed in execution_mode={execution_mode!r}. "
            "Use execution_mode='debug' for smoke tests, or prepare a real MIMIC pilot subset."
        )


def classify_experiment(experiment: dict) -> str:
    retriever = experiment.get("retriever")
    generator = experiment.get("generator")
    if retriever in REAL_RETRIEVERS and generator in REAL_GENERATORS:
        return "real"
    if retriever in STRESS_RETRIEVERS and generator == "retrieval_copy_stress":
        return "stress"
    return "debug"


def assert_experiment_allowed(experiment: dict, execution_mode: str) -> None:
    tier = classify_experiment(experiment)
    if execution_mode == "debug":
        return
    if execution_mode == "stress" and tier == "stress":
        return
    if execution_mode == "real" and tier == "real":
        return
    raise ValueError(
        f"Experiment {experiment.get('name')} is tier={tier!r}, "
        f"which is not allowed in execution_mode={execution_mode!r}. "
        "Use execution_mode=debug/stress for scaffolding, or configure real retriever/generator backends."
    )


def assert_suite_allowed(experiments: list[dict], execution_mode: str) -> dict:
    tiers = {}
    for experiment in experiments:
        assert_experiment_allowed(experiment, execution_mode)
        tiers[experiment["name"]] = classify_experiment(experiment)
    if execution_mode == "real" and not REAL_GENERATORS:
        raise ValueError(
            "Real execution mode is not available yet because no real VLM generator backend "
            "is registered. This prevents mock/stress outputs from being reported as real results."
        )
    return tiers
