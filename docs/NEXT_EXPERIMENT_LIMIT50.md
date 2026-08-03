# Next Experiment: Real Generation LIMIT=50

This run scales the real-generation baseline from 20 evaluation studies to 50
while keeping the same dataset, retrieval artifacts, and VLM.

## Why This Run

The `LIMIT=20` run already shows retrieval-induced hallucination, but the sample
is still small. The `LIMIT=50` run tests whether the failure mode remains visible
with a larger evaluation set.

## Kaggle Notebook

Use:

```text
notebooks/kaggle_real_generation_limit50_llava_v0.ipynb
```

Attach the same private Kaggle dataset:

```text
rgca-real-generation-input
```

Enable a GPU accelerator before running.

## Expected Configuration

```text
GENERATOR_BACKEND = hf_vlm
MODEL_ID = llava-hf/llava-1.5-7b-hf
LIMIT = 50
OUTPUT_DIR = /kaggle/working/rgca_experiments/real_generation_limit50_llava_v0
EVIDENCE_ZIP = /kaggle/working/rgca_real_generation_evidence_limit50_llava_v0.zip
```

## Run Order

Run the notebook from top to bottom.

The preflight cell must pass before model inference starts. It checks:

- GitHub repo clone/pull
- required input files
- JSONL schema
- GPU availability
- generation script CLI compatibility

If preflight fails, stop and fix the setup before continuing.

## Success Criteria

The run is valid only if:

- `generation_manifest.json` exists
- `generator_backend` is `hf_vlm`
- `model_id` is `llava-hf/llava-1.5-7b-hf`
- `eval_size` is `50`
- `missing_image_count` is `0`
- all three generation files have 50 rows
- all three evaluation summaries exist

## After the Run

Download:

```text
/kaggle/working/rgca_real_generation_evidence_limit50_llava_v0.zip
```

Save it locally as:

```text
/Users/mac/Desktop/career/Research/rgca/rgca_real_generation_evidence_limit50_llava_v0.zip
```

Then extract and compare it against `limit20_llava_v0`.
