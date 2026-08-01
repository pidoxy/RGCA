# Kaggle Pilot Runbook

This runbook describes how to use Kaggle as the first compute environment for the RGCA baseline pilot.

The goal is not to run the full MIMIC-CXR dataset. The goal is to produce a small, reproducible evidence package for the Africa in AI workshop submission:

```text
MIMIC subset -> retrieval -> retrieved reports -> VLM/mock generation -> hallucination table
```

## Why Kaggle

Kaggle is suitable for the pilot because it provides free notebook compute and optional GPU sessions. Use GPU time only for model inference or embeddings. Keep dataset preparation, retrieval wiring, and evaluation CPU-friendly.

## Data Safety

MIMIC-CXR is credentialed-access data. Do not publish raw MIMIC files, reports, images, or derived private datasets publicly.

Use one of these safe options:

- Attach a private Kaggle dataset that only you can access.
- Upload only a small private pilot subset.
- Keep outputs that contain report text private unless your data agreement allows sharing.

Do not push raw clinical data into this GitHub repo.

## Recommended Kaggle Dataset Layout

Create a private Kaggle dataset with this shape:

```text
/kaggle/input/rgca-mimic-pilot/
  mimic-cxr/
    files/
      p10/
      p11/
      ...
      mimic-cxr-reports.zip        # optional if not already extracted
  mimic-cxr-jpg/
    files/
      p10/
      p11/
      ...
      mimic-cxr-2.0.0-metadata.csv.gz
      mimic-cxr-2.0.0-split.csv.gz
      mimic-cxr-2.0.0-chexpert.csv.gz
```

If you only have an already-built subset JSONL, use:

```text
/kaggle/input/rgca-mimic-pilot/
  mimic_subset.jsonl
```

## Kaggle Notebook Setup

In a Kaggle notebook:

1. Turn on internet only if you need to clone the repository or install optional model packages.
2. Attach your private dataset.
3. Turn on GPU only for cells that actually run VLM inference or GPU embeddings.
4. Run:

```bash
git clone https://github.com/pidoxy/RGCA.git /kaggle/working/RGCA
cd /kaggle/working/RGCA
python -m pip install -e .
```

If internet is disabled, upload the repo as a Kaggle dataset or notebook file instead of cloning.

## Option A: Run From Existing Subset JSONL

Use this when you already have:

```text
/kaggle/input/rgca-mimic-pilot/mimic_subset.jsonl
```

Run:

```bash
cd /kaggle/working/RGCA
python scripts/kaggle_run_pilot.py \
  --subset-jsonl /kaggle/input/rgca-mimic-pilot/mimic_subset.jsonl \
  --output-dir /kaggle/working/rgca_pilot \
  --retriever lexical \
  --top-k 3
```

## Option B: Build A Pilot Subset On Kaggle

Use this when you attached MIMIC metadata, reports, and JPG files.

Run:

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

Start with `--limit 50` or `--limit 100`. Do not start with the full dataset.

By default, `--limit 100` builds a balanced pilot:

```text
80 train studies -> retrieval_pool
20 validate studies -> eval
```

If you want explicit control, use:

```bash
python scripts/kaggle_run_pilot.py \
  ... \
  --retrieval-limit 80 \
  --eval-limit 20
```

The baseline needs both groups. If the subset contains only `retrieval_pool` records, retrieval files can be written but generation files will be missing because there are no target studies to evaluate.

## Controlled Retrieval-Copy Stress Test

After the default mock run succeeds, run a controlled stress test:

```bash
cd /kaggle/working/RGCA
python scripts/kaggle_run_pilot.py \
  --metadata /kaggle/input/rgca-mimic-pilot/mimic-cxr-jpg/files/mimic-cxr-2.0.0-metadata.csv.gz \
  --split /kaggle/input/rgca-mimic-pilot/mimic-cxr-jpg/files/mimic-cxr-2.0.0-split.csv.gz \
  --labels /kaggle/input/rgca-mimic-pilot/mimic-cxr-jpg/files/mimic-cxr-2.0.0-chexpert.csv.gz \
  --reports-root /kaggle/input/rgca-mimic-pilot/mimic-cxr/files \
  --images-root /kaggle/input/rgca-mimic-pilot/mimic-cxr-jpg/files \
  --output-dir /kaggle/working/rgca_pilot_500_stress \
  --limit 500 \
  --retrieval-limit 400 \
  --eval-limit 100 \
  --retriever lexical \
  --generator retrieval_copy_stress \
  --top-k 3
```

This backend is intentionally not a clinical model. It copies detected pathology labels from retrieved reports into the generated report when those labels are absent from the target labels. Use it to validate that the mismatch protocol and retrieval-induced hallucination metric can detect the failure mode before spending GPU time on a real VLM.

## Expected Outputs

The script writes:

```text
/kaggle/working/rgca_pilot/
  data/mimic_subset.jsonl
  baseline/
    retrieval_results.jsonl
    mismatch_results.jsonl
    generations_no_retrieval.jsonl
    generations_retrieval.jsonl
    generations_mismatch.jsonl
    run_summary.json
  evaluation_mismatch/
    evaluation_summary.json
    evaluation_details.jsonl
```

Download the whole `/kaggle/working/rgca_pilot` folder after the run.

## What Counts As Submission Evidence

For the workshop paper, collect:

- number of studies in the pilot subset
- top-k setting
- retrieval mode
- hallucination summary JSON
- 5 to 10 mismatch examples where retrieved findings contaminate generated findings
- limitations stating that this is a pilot and not final clinical validation

## Minimum Pilot For The Deadline

If compute remains tight, run:

```text
50 studies
k = 3
lexical retriever
all modes: no_retrieval, retrieval, mismatch
manual review of 20 mismatch cases
```

If GPU becomes available, replace the mock generator with a real VLM backend for only the same 50 to 100 studies.
