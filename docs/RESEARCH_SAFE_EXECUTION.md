# Research-Safe Execution

This project separates code smoke tests from research evidence. The goal is to prevent demo, mock, or partial runs from being mistaken for experiment results.

## Execution Modes

`debug` is for code-only smoke testing. It may use the demo dataset and mock generator. Do not cite debug outputs as experimental evidence.

`stress` is for the current baseline stage. It requires a real study subset and allows controlled retrieval-copy stress tests that probe whether retrieved reports can contaminate generated findings.

The default stress suite intentionally excludes mock generators and mock-image retrieval. Those are useful for development, but they are not safe to report as pilot evidence.

`real` is reserved for real model inference. It requires a real MIMIC subset, existing image files, a real retrieval backend, and a real VLM generator. This mode currently fails by design until those backends are implemented.

## Required Dataset Contract

Every experiment-ready subset must be a JSONL file where each row contains:

```json
{
  "study_id": "50000001",
  "image_path": "/path/to/image.jpg",
  "report_text": "Full report text",
  "findings": "Findings section",
  "impression": "Impression section",
  "labels": ["cardiomegaly"],
  "split": "retrieval_pool"
}
```

The subset must include both `retrieval_pool` and `eval` records. Duplicate study IDs and missing required fields are treated as failures.

The repository demo dataset is blocked outside `debug` mode. This prevents toy outputs from being accidentally reused as pilot evidence.

## Kaggle Setup From Zero

Start from a fresh Kaggle notebook with no attached input. Clone the repo and install it:

```bash
cd /kaggle/working
git clone https://github.com/pidoxy/RGCA.git
cd /kaggle/working/RGCA
python -m pip install -e .
```

Prepare the pilot subset from PhysioNet. This downloads only metadata, split labels, CheXpert labels, and report text. It does not download the full image archive.

```bash
python scripts/kaggle_prepare_mimic_subset.py \
  --physionet-user YOUR_PHYSIONET_USERNAME \
  --work-dir /kaggle/working/physionet \
  --output-dir /kaggle/working/rgca_pilot_500 \
  --retrieval-limit 400 \
  --eval-limit 100
```

Run the guarded stress suite:

```bash
python scripts/kaggle_bootstrap_baseline.py \
  --subset-jsonl /kaggle/working/rgca_pilot_500/data/mimic_subset.jsonl \
  --execution-mode stress \
  --output-dir /kaggle/working/rgca_experiments/mimic_pilot_baseline_v0 \
  --pilot-output-dir /kaggle/working/rgca_pilot_500 \
  --overwrite
```

For a code-only smoke test, explicitly choose debug mode:

```bash
python scripts/kaggle_bootstrap_baseline.py \
  --execution-mode debug \
  --allow-demo \
  --output-dir /kaggle/working/rgca_experiments/demo_bootstrap \
  --overwrite
```

## Integrity Evidence

Every guarded suite writes:

- `suite_manifest.json`
- `bootstrap_summary.json`
- per-experiment manifests
- evaluation summaries
- CSV and Markdown summary tables

The manifest records the execution mode, experiment tiers, dataset validation summary, and dataset fingerprint. This makes the run auditable and prevents silent fallback.

## Current Limitation

The current stress suite is not real VLM inference. It is a controlled baseline/stress harness for demonstrating retrieval-induced failure modes. Paper claims should describe it as controlled stress testing until BioMedCLIP and a real LLaVA-Med-compatible generator are integrated.
