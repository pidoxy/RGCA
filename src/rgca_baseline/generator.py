from __future__ import annotations

from rgca_baseline.evaluation import infer_labels_from_text
from rgca_baseline.schemas import StudyRecord


def format_report(findings: str, impression: str) -> str:
    return f"Findings:\n{findings}\n\nImpression:\n{impression}"


def extract_retrieved_signal(retrieved_reports: list[str]) -> str:
    if not retrieved_reports:
        return ""
    first = retrieved_reports[0]
    if "." in first:
        return first.split(".", 1)[0].strip()
    return first.strip()


class MockVLMGenerator:
    """
    Mock generator that makes retrieval effects visible.

    This is for validating the pipeline structure. Replace this with a real VLM backend
    once you plug in your chosen model.
    """

    def generate(self, study: StudyRecord, mode: str, retrieved_reports: list[str]) -> str:
        if mode == "no_retrieval":
            return format_report(study.findings, study.impression)

        if mode == "retrieval":
            supportive = extract_retrieved_signal(retrieved_reports)
            findings = study.findings
            if supportive and supportive.lower() not in findings.lower():
                findings = f"{findings} Context note: {supportive}."
            return format_report(findings, study.impression)

        if mode == "mismatch":
            contaminated = extract_retrieved_signal(retrieved_reports)
            findings = study.findings
            impression = study.impression
            if contaminated:
                findings = f"{findings} Additional suggested finding: {contaminated}."
                impression = f"{impression} Possible retrieval-induced mismatch noted."
            return format_report(findings, impression)

        raise ValueError(f"Unsupported generation mode: {mode}")


class RetrievalCopyStressGenerator:
    """
    Deterministic stress-test generator for retrieval-induced hallucination.

    It intentionally injects pathology labels found in retrieved reports but absent
    from the reference labels. This is not a clinical model; it is a controlled
    failure-mode probe for validating the mismatch evaluation protocol.
    """

    def generate(self, study: StudyRecord, mode: str, retrieved_reports: list[str]) -> str:
        if mode == "no_retrieval":
            return format_report(study.findings, study.impression)

        retrieved_labels = sorted(
            {
                label
                for report in retrieved_reports
                for label in infer_labels_from_text(report)
            }
        )
        unsupported_labels = [
            label for label in retrieved_labels if label.lower() not in {item.lower() for item in study.labels}
        ]

        findings = study.findings
        impression = study.impression
        if mode == "retrieval":
            copied_labels = unsupported_labels[:1]
        elif mode == "mismatch":
            copied_labels = unsupported_labels
        else:
            raise ValueError(f"Unsupported generation mode: {mode}")

        if copied_labels:
            copied_text = ", ".join(copied_labels)
            findings = f"{findings} Retrieved evidence also suggests {copied_text}."
            impression = f"{impression} Retrieval-conditioned impression includes {copied_text}."

        return format_report(findings, impression)
