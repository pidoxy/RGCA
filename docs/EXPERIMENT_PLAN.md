# Experiment Plan

This document defines the structured experiment setup for the baseline phase before RGCA. The goal is to stop running one-off notebook commands and instead produce reproducible experiment artifacts that can support the workshop paper.

## Research Question

Does retrieval-augmented prompting improve chest X-ray report generation, and can mismatched retrieval evidence induce unsupported clinical findings?

## Current Scope

This phase is not RGCA training. It is a baseline and failure-mode study.

The pipeline under test is:

```text
Image/study record -> Retriever -> Retrieved reports -> Generator -> Generated report -> Hallucination evaluation
```

## Experiment Contract

Every experiment run must produce:

```text
run_summary.json
retrieval_results.jsonl
mismatch_results.jsonl
generations_no_retrieval.jsonl
generations_retrieval.jsonl
generations_mismatch.jsonl
evaluation/retrieval/evaluation_summary.json
evaluation/mismatch/evaluation_summary.json
experiment_manifest.json
```

Every suite must produce:

```text
suite_manifest.json
```

Do not treat a run as valid unless:

- `retrieval_pool_size > 0`
- `eval_size > 0`
- all requested generation files exist
- evaluation summaries exist for retrieval and mismatch modes
- the subset path and all experiment settings are recorded in the manifest

## Dataset Setup

Use one fixed pilot subset per experiment suite.

Recommended pilot:

- `500` total studies
- `400` retrieval-pool studies from MIMIC train
- `100` evaluation studies from MIMIC validate
- `PA/AP` frontal views
- one image per study

The current Kaggle-created subset is:

```text
/kaggle/working/rgca_pilot_500/data/mimic_subset.jsonl
```

This subset should be reused across all baseline comparisons so differences come from retriever/generator settings, not changing data.

## Experiment Phases

### Phase 0: Pipeline Validation

Purpose:

Confirm that ingestion, retrieval, generation, evaluation, and artifact writing all work end-to-end.

Run:

```text
E00_pipeline_mock_lexical_k3
```

Interpretation:

This run is infrastructure evidence only. It is not scientific evidence because the generator is mocked.

### Phase 1: Controlled Failure-Mode Probe

Purpose:

Confirm that the mismatch protocol and evaluator can detect retrieval-induced hallucination when unsupported retrieved findings are copied into generated text.

Runs:

```text
E01_stress_lexical_k1
E02_stress_lexical_k3
E03_stress_lexical_k5
```

Interpretation:

These runs are controlled stress tests. They are valid evidence that the evaluation protocol can detect the failure mode, but not evidence about real VLM behavior.

### Phase 2: Retrieval Backend Comparison

Purpose:

Compare dependency-light retrieval choices before implementing real image embeddings.

Runs:

```text
E02_stress_lexical_k3
E04_stress_hashing_text_k3
E05_stress_mock_image_k3
```

Interpretation:

This tells us whether retrieval artifacts and mismatch selection remain stable across retriever backends. It does not replace real image retrieval.

### Phase 3: Real Retrieval

Purpose:

Replace debug retrieval with a medical image/text embedding model.

Planned retriever:

```text
BioMedCLIP-style image/text embedding retrieval
```

Required output:

```text
target image embedding -> top-k retrieved study reports
```

This is the first scientifically meaningful retrieval baseline.

### Phase 4: Real Generator

Purpose:

Replace mock/stress generator with a real VLM backend.

Candidate options:

- LLaVA-Med-compatible checkpoint
- lightweight medical VLM available on Kaggle
- API-backed VLM if permitted by data policy

Required comparisons:

```text
No Retrieval vs Prompt Retrieval
Clean Retrieval vs Mismatched Retrieval
k = 1, 3, 5
```

This is the first experiment set that can support the paper's empirical claim about real model behavior.

## Structured Suite Command

After creating the pilot subset, run the full baseline suite:

```bash
cd /kaggle/working/RGCA
python scripts/run_experiment_suite.py \
  --config configs/mimic_pilot_suite.json \
  --overwrite
```

If your subset is stored somewhere else:

```bash
python scripts/run_experiment_suite.py \
  --config configs/mimic_pilot_suite.json \
  --input /path/to/mimic_subset.jsonl \
  --output-dir /kaggle/working/rgca_experiments/mimic_pilot_baseline_v0 \
  --overwrite
```

To run only one experiment:

```bash
python scripts/run_experiment_suite.py \
  --config configs/mimic_pilot_suite.json \
  --only E02_stress_lexical_k3 \
  --overwrite
```

Summarize suite results into CSV and Markdown:

```bash
python scripts/summarize_experiment_suite.py \
  --manifest /kaggle/working/rgca_experiments/mimic_pilot_baseline_v0/suite_manifest.json \
  --output-dir /kaggle/working/rgca_experiments/mimic_pilot_baseline_v0/tables
```

## Experiment Matrix

| ID | Retriever | Generator | k | Purpose |
| --- | --- | --- | --- | --- |
| E00 | lexical | mock | 3 | Pipeline validation |
| E01 | lexical | retrieval_copy_stress | 1 | Controlled mismatch stress test |
| E02 | lexical | retrieval_copy_stress | 3 | Default controlled stress test |
| E03 | lexical | retrieval_copy_stress | 5 | Top-k ablation |
| E04 | hashing_text | retrieval_copy_stress | 3 | Dependency-light embedding-style comparison |
| E05 | mock_image | retrieval_copy_stress | 3 | Image-like retrieval path smoke test |

## Metrics

Primary metrics:

- hallucination rate
- retrieval-induced hallucination rate
- retrieval copy rate
- total hallucinated labels
- total retrieval-induced hallucinations

Secondary review outputs:

- top mismatch examples
- target labels vs retrieved labels vs generated labels
- manual review table for 10 to 20 selected cases

## Evidence Tiers

### Tier 1: Infrastructure Evidence

Shows the code works.

Includes:

- successful subset build
- successful suite run
- all artifacts written

### Tier 2: Controlled Failure-Mode Evidence

Shows the evaluation protocol can detect retrieval-induced hallucination under a deliberately contaminated generator.

Includes:

- retrieval-copy stress run
- non-zero retrieval-induced hallucination rate
- examples showing unsupported retrieved findings copied into generated reports

### Tier 3: Real Baseline Evidence

Shows real model behavior.

Requires:

- real image/text retrieval
- real VLM generation
- no-retrieval/retrieval/mismatch comparison
- manual review of representative cases

The paper should clearly separate these tiers.

## Current Status

Completed:

- real MIMIC-style subset pipeline
- 500-study pilot run
- artifact generation for all three modes
- controlled retrieval-copy stress backend
- structured suite config
- structured suite runner

Not yet complete:

- real BioMedCLIP embedding retrieval
- real VLM generation
- final manual review table
- final paper-ready result tables

## Next Implementation Steps

1. Run the structured suite on Kaggle using `configs/mimic_pilot_suite.json`.
2. Summarize the suite with `scripts/summarize_experiment_suite.py`.
3. Download the private output zip and keep it out of GitHub.
4. Implement real image/text retrieval.
5. Add a real VLM backend and rerun only the minimum suite first.
