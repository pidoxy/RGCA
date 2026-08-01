# RGCA

This repository contains the first baseline implementation for retrieval-augmented radiology report generation before RGCA.

## Baseline Goal

Build a simple pipeline that lets us observe retrieval effects on report generation:

```text
Image -> Retriever -> Retrieved Reports -> VLM -> Generated Report
```

We start small. The first goal is not to beat SOTA. The first goal is to:

- produce reports
- compare no-retrieval vs retrieval vs mismatch behavior
- inspect whether retrieved context changes generated findings

## What Is In This Repo

This repository currently includes:

- a runnable baseline script
- a runnable baseline notebook
- a small demo dataset so the workflow runs end-to-end immediately
- a real-data MIMIC-CXR subset builder
- a study-level embedding and retrieval index skeleton
- a first-pass hallucination evaluation module
- retrieval, mismatch selection, prompt construction, and generation interfaces
- a mock VLM backend for local workflow validation

This means you can run the whole pipeline now, then replace the demo dataset and mock backend with real MIMIC-CXR and a real VLM.

## Repository Layout

```text
RGCA/
  docs/
    PROBLEM_FORMULATION.md
    LITERATURE_SYNTHESIS.md
    HALLUCINATION_FRAMEWORK.md
    RETRIEVAL_PLAN.md
    RESEARCH_SPEC.md
    PAPER_OUTLINE.md
    PAPER_CLAIMS.md
    EXPERIMENT_PLAN.md
    ROADMAP.md
  data/
    raw/
    interim/
    processed/
    demo/
      demo_studies.jsonl
  configs/
    baseline_demo.json
  src/
    rgca_baseline/
      __init__.py
      embeddings.py
      evaluation.py
      generator.py
      indexing.py
      io_utils.py
      mimic_cxr.py
      mismatch.py
      pipeline.py
      prompts.py
      real_retrieval.py
      retrieval.py
      schemas.py
      text_utils.py
      vlm_client.py
  scripts/
    build_mimic_subset.py
    build_retrieval_index.py
    evaluate_generations.py
    plan_mimic_pilot_subset.py
    run_baseline.py
  notebooks/
    baseline_experiment.ipynb
  results/
    runs/
    analysis/
    figures/
  paper/
    figures/
    tables/
    references.bib
    rgca.tex
  pyproject.toml
```

### Folder Roles

- `docs/` is the research-design layer. It holds the problem formulation, literature synthesis, hallucination framework, retrieval plan, experiment plan, and manuscript logic.
- `data/` is the dataset layer. Use `raw/` for source downloads, `interim/` for manifests and index artifacts, `processed/` for experiment-ready JSONL subsets, and `demo/` for lightweight runnable examples.
- `src/` is the reusable implementation layer. Retrieval, ingestion, indexing, generation, and evaluation code live here.
- `scripts/` is the execution layer. These are task-oriented entry points for subset building, index construction, baseline runs, and evaluation.
- `notebooks/` is the interactive analysis layer for retrieval inspection, side-by-side generation comparison, and result exploration.
- `results/` is the experiment-output layer. Store run artifacts, processed summaries, and analysis figures here.
- `paper/` is the manuscript layer. It holds figures, tables, bibliography entries, and the paper draft itself.

## Baselines Implemented

The script supports:

1. `no_retrieval`
2. `retrieval`
3. `mismatch`
4. `all`

Definitions:

- `no_retrieval`: generate a report from the target study only
- `retrieval`: retrieve top-k similar studies and include their reports in the prompt
- `mismatch`: choose intentionally poor retrieval context and observe report contamination

## Quick Start

Run the full demo:

```bash
cd "/Users/mac/Documents/New project/RGCA"
python3 scripts/run_baseline.py \
  --input data/demo/demo_studies.jsonl \
  --output-dir results/runs/demo_run \
  --mode all \
  --retriever lexical \
  --top-k 3
```

This writes:

- `results/runs/demo_run/retrieval_results.jsonl`
- `results/runs/demo_run/mismatch_results.jsonl`
- `results/runs/demo_run/generations_no_retrieval.jsonl`
- `results/runs/demo_run/generations_retrieval.jsonl`
- `results/runs/demo_run/generations_mismatch.jsonl`
- `results/runs/demo_run/run_summary.json`

Build a retrieval index artifact:

```bash
python3 scripts/build_retrieval_index.py \
  --subset data/demo/demo_studies.jsonl \
  --backend mock_image \
  --output-dir results/runs/indexes/demo_mock_image
```

Evaluate generated outputs:

```bash
python3 scripts/evaluate_generations.py \
  --studies data/demo/demo_studies.jsonl \
  --generations results/runs/demo_run/generations_mismatch.jsonl \
  --retrieval-results results/runs/demo_run/mismatch_results.jsonl \
  --output-dir results/runs/demo_eval_mismatch
```

## Notebook Workflow

For interactive experiment work, open:

- [notebooks/baseline_experiment.ipynb](notebooks/baseline_experiment.ipynb)

The notebook uses the same reusable code under `src/rgca_baseline/`, so you do not end up with notebook-only logic that is hard to maintain.

## Kaggle Pilot Workflow

Kaggle is the recommended first compute environment for the workshop pilot. Use it to run a small MIMIC-CXR subset and save a reproducible evidence bundle.

See:

- [docs/KAGGLE_PILOT_RUNBOOK.md](docs/KAGGLE_PILOT_RUNBOOK.md)

If you already have a private Kaggle dataset with `mimic_subset.jsonl`:

```bash
cd /kaggle/working/RGCA
python scripts/kaggle_run_pilot.py \
  --subset-jsonl /kaggle/input/rgca-mimic-pilot/mimic_subset.jsonl \
  --output-dir /kaggle/working/rgca_pilot \
  --retriever lexical \
  --top-k 3
```

If you attached MIMIC metadata, reports, and JPG files:

```bash
cd /kaggle/working/RGCA
python scripts/kaggle_run_pilot.py \
  --metadata /kaggle/input/rgca-mimic-pilot/mimic-cxr-jpg/files/mimic-cxr-2.0.0-metadata.csv.gz \
  --split /kaggle/input/rgca-mimic-pilot/mimic-cxr-jpg/files/mimic-cxr-2.0.0-split.csv.gz \
  --labels /kaggle/input/rgca-mimic-pilot/mimic-cxr-jpg/files/mimic-cxr-2.0.0-chexpert.csv.gz \
  --reports-root /kaggle/input/rgca-mimic-pilot/mimic-cxr/files \
  --images-root /kaggle/input/rgca-mimic-pilot/mimic-cxr-jpg/files \
  --output-dir /kaggle/working/rgca_pilot \
  --limit 100 \
  --retriever lexical \
  --top-k 3
```

After the pilot subset is created, run the structured experiment suite instead of running one-off cells:

```bash
cd /kaggle/working/RGCA
python scripts/run_experiment_suite.py \
  --config configs/mimic_pilot_suite.json \
  --overwrite
```

This writes one folder per named experiment plus a top-level `suite_manifest.json`.

## Real-Data Path

The repository now includes a MIMIC-CXR ingestion path:

```bash
python3 scripts/build_mimic_subset.py \
  --metadata /path/to/mimic-cxr-jpg/files/mimic-cxr-2.0.0-metadata.csv.gz \
  --split /path/to/mimic-cxr-jpg/files/mimic-cxr-2.0.0-split.csv.gz \
  --reports-root /path/to/mimic-cxr/files \
  --images-root /path/to/mimic-cxr-jpg/files \
  --labels /path/to/mimic-cxr-jpg/files/mimic-cxr-2.0.0-chexpert.csv.gz \
  --output data/processed/mimic_subset.jsonl \
  --dataset-splits train validate \
  --views PA AP \
  --limit 200
```

This creates a study-level JSONL that the baseline pipeline can use.

The dataset and retrieval investigation notes live in:

- [docs/MIMIC_SETUP.md](docs/MIMIC_SETUP.md)

## Current Backend

The default generator is a `mock` VLM backend that makes retrieval effects visible for workflow validation.

It is useful for:

- testing repository structure
- validating output artifacts
- confirming retrieval and mismatch logic
- checking prompt wiring

It is not your final experimental model.

## How To Replace The Demo With Your Real Baseline

### 1. Replace the dataset

Create a JSONL file where each record looks like:

```json
{
  "study_id": "12345678",
  "image_path": "/path/to/image.png",
  "report_text": "Full report text",
  "findings": "Cardiomediastinal silhouette is mildly enlarged...",
  "impression": "Mild cardiomegaly without focal consolidation.",
  "labels": ["cardiomegaly"],
  "split": "retrieval_pool"
}
```

Supported `split` values for the current script:

- `retrieval_pool`
- `eval`

### 2. Replace retrieval

The current retriever is a lightweight lexical retriever over study text fields so the workflow runs without external dependencies.

You can now also run:

- `lexical`
- `mock_image`
- `hashing_text`

For the real baseline, replace it with:

- image encoder: `BioMedCLIP` or similar
- vector search: `FAISS`
- retrieved payload: study reports

The clean replacement point is [src/rgca_baseline/retrieval.py](/Users/mac/Documents/New%20project/RGCA/src/rgca_baseline/retrieval.py:1).

### 3. Replace generation

The current generator is a mock backend that simulates retrieval influence.

For the real baseline, replace it with a VLM call in [src/rgca_baseline/generator.py](/Users/mac/Documents/New%20project/RGCA/src/rgca_baseline/generator.py:1), for example:

- `LLaVA-Med`
- another local HF-compatible medical VLM

### 4. Keep the same output contract

Even after replacing the internals, keep the artifacts:

- retrieval results
- mismatch results
- generated reports
- summary metrics

Each generation record is now shaped around a stable experiment contract with:

- `study_id`
- `mode`
- `target_report`
- `retrieved_reports`
- `generated_report`
- `labels_reference`
- `labels_generated`
- `hallucination_flags`

Those outputs are the evidence trail for your baseline.

## Immediate Next Steps

1. Wire your real MIMIC-CXR subset into a JSONL file.
2. Replace the retriever backend with image-based retrieval.
3. Replace the mock generator with your chosen VLM.
4. Add a simple hallucination review table from mismatch generations.

## Notes

- This repository is intentionally starting with a script rather than a notebook because it is easier to version, test, and extend.
- If you want, the next step after this can be adding:
  - a MIMIC subset builder
  - a FAISS index builder
  - a real Hugging Face VLM integration
