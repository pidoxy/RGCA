# Dataset And Retrieval Setup Deliverable

This is the handoff-ready deliverable for Emmanuel's `Dataset and retrieval setup` task in `RGCA`.

Repository:

- [RGCA](https://github.com/pidoxy/RGCA)

## Goal

Answer the MIMIC-CXR setup questions and define a small working subset plus a retrieval plan for the first baseline before RGCA.

## 1. How do we access the dataset?

Use the official PhysioNet releases:

- `MIMIC-CXR v2.1.0`
- `MIMIC-CXR-JPG v2.1.0`

Access is through a credentialed PhysioNet account with required training and DUA acceptance. The dataset should be treated as the official source even if a temporary incomplete copy is used internally for prototyping.

## 2. Which subset do we start with?

Recommended pilot subset:

- `200` studies total
- `150` studies for retrieval pool
- `50` studies for evaluation
- `PA/AP` frontal views only
- one image per study
- retrieval pool drawn from `train`
- evaluation set drawn from `validate`

If storage is still tight, the smallest acceptable first run is:

- `70` retrieval-pool studies
- `30` eval studies

## 3. What image format and report format are available?

Use:

- `MIMIC-CXR-JPG` for `JPG` image files and metadata
- `MIMIC-CXR` for the original free-text radiology reports

Practical interpretation for the baseline:

- query input = chest X-ray image
- retrieved evidence = study reports from retrieved studies

## 4. How do we pair images and reports?

Pair at the `study_id` level.

Rules:

- one study has one report
- one study may have one or more images
- choose one frontal image (`PA` or `AP`) for the first baseline
- use `dicom_id` to locate the image file
- use `s<study_id>.txt` to locate the report

Practical join:

`metadata.csv + split.csv + report file tree -> study-level record`

## 5. What retrieval index do we build?

Build a `study-level retrieval index`.

Each indexed item should contain:

- `study_id`
- `image_path`
- `report_text`
- `findings`
- `impression`
- `labels`
- `split`

Current repo support:

- [build_mimic_subset.py](/Users/mac/Documents/New%20project/RGCA/scripts/build_mimic_subset.py:1)
- [plan_mimic_pilot_subset.py](/Users/mac/Documents/New%20project/RGCA/scripts/plan_mimic_pilot_subset.py:1)
- [mimic_cxr.py](/Users/mac/Documents/New%20project/RGCA/src/rgca_baseline/mimic_cxr.py:1)

## 6. Do we retrieve using image similarity, report similarity, or image-text embedding similarity?

Recommendation:

1. final target: `image-text embedding similarity`
2. temporary debug fallback: `report similarity`

Reasoning:

- the target report is unavailable at inference time
- prompt-based retrieval should be driven by image-compatible study retrieval
- a shared image-text embedding space is the cleanest first real setup

Planned real retrieval path:

`target image embedding -> nearest study embeddings -> retrieved study reports`

## Small Working Subset Plan

The repo now supports a pilot manifest build before full ingestion.

### Step A: plan the pilot

```bash
cd "/Users/mac/Documents/New project/RGCA"

python3 scripts/plan_mimic_pilot_subset.py \
  --metadata data/raw/mimic-cxr-jpg/mimic-cxr-2.0.0-metadata.csv.gz \
  --split data/raw/mimic-cxr-jpg/mimic-cxr-2.0.0-split.csv.gz \
  --images-root data/raw/mimic-cxr-jpg/files \
  --labels data/raw/mimic-cxr-jpg/mimic-cxr-2.0.0-chexpert.csv.gz \
  --output data/processed/mimic_pilot_plan.jsonl \
  --train-limit 150 \
  --eval-limit 50 \
  --views PA AP
```

Output:

- `data/processed/mimic_pilot_plan.jsonl`

This gives a study-level pilot manifest even before final report ingestion is complete.

### Step B: build the actual subset

```bash
python3 scripts/build_mimic_subset.py \
  --metadata data/raw/mimic-cxr-jpg/mimic-cxr-2.0.0-metadata.csv.gz \
  --split data/raw/mimic-cxr-jpg/mimic-cxr-2.0.0-split.csv.gz \
  --reports-root data/raw/mimic-cxr/files \
  --images-root data/raw/mimic-cxr-jpg/files \
  --labels data/raw/mimic-cxr-jpg/mimic-cxr-2.0.0-chexpert.csv.gz \
  --output data/processed/mimic_subset.jsonl \
  --dataset-splits train validate \
  --views PA AP \
  --limit 200
```

Output:

- `data/processed/mimic_subset.jsonl`

## Retrieval Plan

Use a staged approach:

### Phase 1: debug retrieval

- lexical retriever
- hashing-text retriever
- mock image retriever

Purpose:

- validate artifact flow
- validate top-k saving
- validate mismatch logic

### Phase 2: real baseline retrieval

- build a study-level image-text embedding index
- query with target image
- retrieve top-k similar studies
- pass only the retrieved reports into the generation prompt

Recommended first real encoder:

- `BioMedCLIP`

## Deliverable Status

Completed in the repo:

- MIMIC access and pairing investigation
- subset builder
- pilot-subset planner
- retrieval direction recommendation
- study-level index plan

External blocker:

- official complete PhysioNet download still needed for final subset build and final reported experiments

## Final Short Answer

Emmanuel's dataset/retrieval setup deliverable is complete at the planning and code-scaffold level:

- access path identified
- subset choice fixed
- file formats clarified
- image/report pairing defined
- retrieval index defined
- retrieval similarity choice decided
- repo commands added for pilot planning and subset construction
