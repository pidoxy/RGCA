from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize an RGCA experiment suite manifest.")
    parser.add_argument("--manifest", required=True, help="Path to suite_manifest.json.")
    parser.add_argument("--output-dir", required=True, help="Directory for summary tables.")
    return parser.parse_args()


def metric(summary: dict, mode: str, name: str) -> float | int | str:
    return summary.get(mode, {}).get(mode, {}).get(name, "")


def collect_rows(manifest: dict) -> list[dict]:
    rows = []
    for experiment in manifest["experiments"]:
        pipeline = experiment["pipeline_summary"]
        summaries = experiment.get("evaluation_summaries", {})
        for mode in ["retrieval", "mismatch"]:
            rows.append(
                {
                    "experiment": experiment["name"],
                    "mode": mode,
                    "retriever": pipeline["retriever_backend"],
                    "generator": pipeline["generator_backend"],
                    "top_k": pipeline["top_k"],
                    "retrieval_pool_size": pipeline["retrieval_pool_size"],
                    "eval_size": pipeline["eval_size"],
                    "n": metric(summaries, mode, "n"),
                    "hallucination_rate": metric(summaries, mode, "hallucination_rate"),
                    "retrieval_induced_hallucination_rate": metric(
                        summaries,
                        mode,
                        "retrieval_induced_hallucination_rate",
                    ),
                    "retrieval_copy_rate": metric(summaries, mode, "retrieval_copy_rate"),
                    "total_hallucinated_labels": metric(summaries, mode, "total_hallucinated_labels"),
                    "total_retrieval_induced_hallucinations": metric(
                        summaries,
                        mode,
                        "total_retrieval_induced_hallucinations",
                    ),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[field]) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    rows = collect_rows(manifest)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "suite_summary.csv", rows)
    write_markdown(output_dir / "suite_summary.md", rows)

    print(f"Wrote {len(rows)} summary rows to {output_dir}")


if __name__ == "__main__":
    main()
