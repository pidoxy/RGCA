from __future__ import annotations

import csv
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

from rgca_baseline.schemas import StudyRecord


SECTION_PATTERN = re.compile(r"^\s*([A-Za-z /_-]+)\s*:\s*(.*)$")


def _read_csv_maybe_gz(path: str | Path) -> list[dict[str, str]]:
    target = Path(path)
    opener = gzip.open if target.suffix == ".gz" else open
    with opener(target, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _parse_report_sections(report_text: str) -> tuple[str, str]:
    findings_lines: list[str] = []
    impression_lines: list[str] = []
    current_section: str | None = None

    for raw_line in report_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = SECTION_PATTERN.match(line)
        if match:
            header = match.group(1).strip().lower()
            content = match.group(2).strip()
            if "finding" in header:
                current_section = "findings"
                if content:
                    findings_lines.append(content)
                continue
            if "impression" in header:
                current_section = "impression"
                if content:
                    impression_lines.append(content)
                continue
            current_section = None
            continue

        if current_section == "findings":
            findings_lines.append(line)
        elif current_section == "impression":
            impression_lines.append(line)

    findings = " ".join(findings_lines).strip()
    impression = " ".join(impression_lines).strip()
    return findings, impression


def _load_reports_from_root(report_root: str | Path) -> dict[str, dict[str, str]]:
    root = Path(report_root)
    reports: dict[str, dict[str, str]] = {}
    for path in root.rglob("s*.txt"):
        study_id = path.stem.removeprefix("s")
        report_text = path.read_text(encoding="utf-8")
        findings, impression = _parse_report_sections(report_text)
        reports[study_id] = {
            "report_text": report_text,
            "findings": findings,
            "impression": impression,
        }
    return reports


def _load_labels(label_path: str | Path | None) -> dict[str, list[str]]:
    if label_path is None:
        return {}
    rows = _read_csv_maybe_gz(label_path)
    labels_by_study: dict[str, list[str]] = {}
    for row in rows:
        study_id = str(row["study_id"])
        labels: list[str] = []
        for key, value in row.items():
            if key in {"subject_id", "study_id"}:
                continue
            normalized = (value or "").strip()
            if normalized == "1.0" or normalized == "1":
                labels.append(key)
        labels_by_study[study_id] = labels
    return labels_by_study


def build_mimic_subset(
    metadata_path: str | Path,
    split_path: str | Path,
    report_root: str | Path,
    image_root: str | Path,
    label_path: str | Path | None = None,
    allowed_splits: set[str] | None = None,
    allowed_views: set[str] | None = None,
    limit: int | None = None,
) -> list[StudyRecord]:
    metadata_rows = _read_csv_maybe_gz(metadata_path)
    split_rows = _read_csv_maybe_gz(split_path)
    reports_by_study = _load_reports_from_root(report_root)
    labels_by_study = _load_labels(label_path)

    split_lookup = {
        (str(row["subject_id"]), str(row["study_id"]), str(row["dicom_id"])): row
        for row in split_rows
    }

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in metadata_rows:
        key = (str(row["subject_id"]), str(row["study_id"]), str(row["dicom_id"]))
        split_row = split_lookup.get(key)
        if not split_row:
            continue
        merged = dict(row)
        merged["split"] = split_row["split"]
        grouped[str(row["study_id"])].append(merged)

    studies: list[StudyRecord] = []
    for study_id, rows in grouped.items():
        report = reports_by_study.get(study_id)
        if not report:
            continue

        chosen_row = None
        if allowed_views:
            for row in rows:
                view = (row.get("ViewPosition") or "").upper()
                if view in allowed_views:
                    chosen_row = row
                    break
        if chosen_row is None:
            chosen_row = rows[0]

        split_name = chosen_row.get("split", "")
        if allowed_splits and split_name not in allowed_splits:
            continue

        view_position = chosen_row.get("ViewPosition")
        if allowed_views and (view_position or "").upper() not in allowed_views:
            continue

        subject_id = str(chosen_row["subject_id"])
        dicom_id = str(chosen_row["dicom_id"])
        jpg_path = (
            Path(image_root)
            / f"p{subject_id[:2]}"
            / f"p{subject_id}"
            / f"s{study_id}"
            / f"{dicom_id}.jpg"
        )

        record = StudyRecord(
            study_id=study_id,
            subject_id=subject_id,
            dicom_id=dicom_id,
            image_path=str(jpg_path),
            report_text=report["report_text"],
            findings=report["findings"] or report["report_text"],
            impression=report["impression"],
            labels=labels_by_study.get(study_id, []),
            split="retrieval_pool" if split_name == "train" else "eval",
            view_position=view_position,
        )
        studies.append(record)
        if limit is not None and len(studies) >= limit:
            break

    return studies


def plan_mimic_pilot_subset(
    metadata_path: str | Path,
    split_path: str | Path,
    image_root: str | Path,
    label_path: str | Path | None = None,
    train_limit: int = 150,
    eval_limit: int = 50,
    allowed_views: set[str] | None = None,
) -> list[dict]:
    metadata_rows = _read_csv_maybe_gz(metadata_path)
    split_rows = _read_csv_maybe_gz(split_path)
    labels_by_study = _load_labels(label_path)

    split_lookup = {
        (str(row["subject_id"]), str(row["study_id"]), str(row["dicom_id"])): row
        for row in split_rows
    }

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in metadata_rows:
        key = (str(row["subject_id"]), str(row["study_id"]), str(row["dicom_id"]))
        split_row = split_lookup.get(key)
        if not split_row:
            continue
        merged = dict(row)
        merged["split"] = split_row["split"]
        grouped[str(row["study_id"])].append(merged)

    counts = {"train": 0, "validate": 0}
    plans: list[dict] = []

    for study_id in sorted(grouped.keys()):
        rows = grouped[study_id]
        chosen_row = None
        if allowed_views:
            for row in rows:
                view = (row.get("ViewPosition") or "").upper()
                if view in allowed_views:
                    chosen_row = row
                    break
        if chosen_row is None:
            chosen_row = rows[0]

        split_name = chosen_row.get("split", "")
        if split_name not in {"train", "validate"}:
            continue
        if split_name == "train" and counts["train"] >= train_limit:
            continue
        if split_name == "validate" and counts["validate"] >= eval_limit:
            continue

        view_position = (chosen_row.get("ViewPosition") or "").upper()
        if allowed_views and view_position not in allowed_views:
            continue

        subject_id = str(chosen_row["subject_id"])
        dicom_id = str(chosen_row["dicom_id"])
        planned_split = "retrieval_pool" if split_name == "train" else "eval"
        image_path = (
            Path(image_root)
            / f"p{subject_id[:2]}"
            / f"p{subject_id}"
            / f"s{study_id}"
            / f"{dicom_id}.jpg"
        )
        plans.append(
            {
                "study_id": study_id,
                "subject_id": subject_id,
                "dicom_id": dicom_id,
                "source_split": split_name,
                "planned_split": planned_split,
                "view_position": view_position,
                "image_path": str(image_path),
                "labels": labels_by_study.get(study_id, []),
            }
        )
        counts[split_name] += 1

        if counts["train"] >= train_limit and counts["validate"] >= eval_limit:
            break

    return plans


def write_study_records(path: str | Path, studies: list[StudyRecord]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for study in studies:
            handle.write(json.dumps(study.to_dict(), ensure_ascii=True) + "\n")


def write_pilot_plan(path: str | Path, plans: list[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in plans:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
