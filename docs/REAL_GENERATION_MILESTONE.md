# Real Generation Milestone

This milestone moves the project from retrieval validation to report generation
over fixed retrieval artifacts.

## Goal

Generate reports for the same target studies under three conditions:

1. `no_retrieval`
2. `retrieval`
3. `mismatch`

The important design choice is that generation consumes saved retrieval files:

```text
mimic_subset.jsonl
retrieval_results.jsonl
mismatch_results.jsonl
        |
        v
VLM generation
        |
        v
generations_*.jsonl + hallucination evaluation
```

This avoids rerunning retrieval every time a generator changes.

## Debug Run

Use this to validate the plumbing without GPU inference:

```bash
python scripts/run_generation_from_retrieval.py \
  --subset data/demo/demo_studies.jsonl \
  --retrieval-results outputs/retrieval_validation_demo/retrieval_results.jsonl \
  --mismatch-results outputs/retrieval_validation_demo/mismatch_results.jsonl \
  --output-dir outputs/generation_demo \
  --generator retrieval_copy_stress \
  --mode all \
  --limit 3
```

## Kaggle Real VLM Run

After running the full Kaggle retrieval notebook, use the hydrated subset and
BioMedCLIP retrieval artifacts:

```bash
python scripts/run_generation_from_retrieval.py \
  --subset /kaggle/working/rgca_hydrated_subset/mimic_subset.jsonl \
  --retrieval-results /kaggle/working/rgca_experiments/biomedclip_retrieval_validation_v0/retrieval_results.jsonl \
  --mismatch-results /kaggle/working/rgca_experiments/biomedclip_retrieval_validation_v0/mismatch_results.jsonl \
  --output-dir /kaggle/working/rgca_experiments/real_generation_v0 \
  --generator hf_vlm \
  --model-id "$RGCA_VLM_MODEL_ID" \
  --mode all \
  --limit 20 \
  --require-real-generator \
  --require-images
```

Set the model with either a Kaggle secret or an environment variable:

```bash
export RGCA_VLM_MODEL_ID="your-llava-or-llava-med-compatible-checkpoint"
```

## Outputs

The generation runner writes:

```text
generation_manifest.json
generations_no_retrieval.jsonl
generations_retrieval.jsonl
generations_mismatch.jsonl
evaluation_no_retrieval/evaluation_summary.json
evaluation_retrieval/evaluation_summary.json
evaluation_mismatch/evaluation_summary.json
```

## Interpretation

For the next paper milestone, compare:

- `retrieval_induced_hallucination_rate` in clean retrieval vs mismatch.
- `retrieval_copy_rate` in clean retrieval vs mismatch.
- Concrete generated examples where the mismatched retrieved report contains a
  pathology absent from the reference, and the generated report copies it.

The strongest evidence is not high report quality yet. The first paper-relevant
evidence is that retrieval-conditioned generation becomes more vulnerable under
controlled retrieval mismatch.
