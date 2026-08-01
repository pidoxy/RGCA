# Real Retrieval Milestone

The next major milestone is to move from controlled stress evidence to real retrieval evidence.

## Goal

Validate this path before running a real VLM:

```text
Target CXR image -> BioMedCLIP image embedding -> nearest studies -> retrieved reports
```

This should be tested independently from generation so retrieval failures do not get hidden inside VLM behavior.

## Current Status

Completed:

- MIMIC-derived 500-study pilot subset.
- Retrieval/mismatch/generation/evaluation artifact contract.
- Controlled stress generator evidence.
- Kaggle evidence bundle:

```text
/kaggle/working/mimic_pilot_baseline_v0.zip
```

Added in this milestone:

- Optional `BiomedCLIPStudyEmbedder` implementation.
- Retrieval-only validation script.
- Guardrails that fail clearly if image files or optional dependencies are missing.

## Kaggle Setup

Install optional retrieval dependencies in the Kaggle notebook before running BioMedCLIP:

```bash
pip install open_clip_torch pillow
```

Kaggle already includes PyTorch in GPU notebooks. If it does not, install the PyTorch build appropriate for the active Kaggle image.

## Required Data

The private Kaggle dataset must include:

- `mimic_subset.jsonl`
- Actual image files referenced by each record's `image_path`

The current metadata/report-only private dataset is enough for stress evidence, but not enough for BioMedCLIP image retrieval unless the referenced JPGs are also present.

If the private Kaggle dataset has `mimic_subset.jsonl` but not JPGs, hydrate only the pilot images from PhysioNet:

```bash
cd /kaggle/working/RGCA
python scripts/kaggle_hydrate_mimic_images.py \
  --subset /kaggle/input/datasets/emmapi/rgca-private-dataset/mimic_subset.jsonl \
  --physionet-user "$PHYSIONET_USERNAME" \
  --output-root /kaggle/working/physionet/mimic-cxr-jpg/files \
  --updated-subset /kaggle/working/rgca_hydrated_subset/mimic_subset.jsonl
```

This downloads only the referenced pilot JPGs, not the full 4.7 TB archive. It writes:

```text
/kaggle/working/rgca_hydrated_subset/mimic_subset.jsonl
/kaggle/working/rgca_hydrated_subset/image_hydration_manifest.json
```

## Retrieval-Only Validation

Run a cheap first validation on 20 evaluation studies:

```bash
cd /kaggle/working/RGCA
python scripts/run_retrieval_validation.py \
  --subset /kaggle/working/rgca_hydrated_subset/mimic_subset.jsonl \
  --backend biomedclip \
  --output-dir /kaggle/working/rgca_experiments/biomedclip_retrieval_validation_v0 \
  --top-k 3 \
  --eval-limit 20
```

Expected outputs:

```text
retrieval_validation_summary.json
retrieval_results.jsonl
mismatch_results.jsonl
```

## Definition Of Done

This milestone is complete when:

- BioMedCLIP dependencies install successfully on Kaggle GPU.
- All image paths in the pilot subset exist.
- Retrieval validation runs for at least 20 eval studies.
- Retrieved study IDs, reports, labels, and similarity scores are saved.
- The retrieval output can be manually inspected before VLM generation.

## Next Step After This

Once real retrieval is validated, add the real VLM backend and run:

```text
no_retrieval vs retrieval vs mismatch
```

on the same fixed pilot subset.
