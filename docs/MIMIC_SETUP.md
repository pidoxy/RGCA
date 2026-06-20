# MIMIC-CXR Dataset And Retrieval Setup

This document answers the baseline setup questions for the first real RGCA experiment and records the recommended starting path.

## Short Answer

Use `MIMIC-CXR-JPG v2.1.0` for image files and metadata, and `MIMIC-CXR v2.1.0` for the original study reports. Start with a small frontal-view subset, build a study-level JSONL, and use image-text embedding retrieval as the first real retrieval method.

## 1. How do we access the dataset?

MIMIC-CXR and MIMIC-CXR-JPG are both restricted-access PhysioNet datasets. Access requires:

- a PhysioNet credentialed account
- completion of required CITI training
- signing the PhysioNet data use agreement

PhysioNet states that only credentialed users who complete the required training and sign the DUA can access the files for MIMIC-CXR v2.1.0. The dataset page also labels it `Database Credentialed Access`. [PhysioNet MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.1.0/) and [PhysioNet MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.1.0/)

## 2. Which subset do we start with?

Recommended first subset:

- dataset family: `MIMIC-CXR-JPG v2.1.0`
- image views: `PA` and `AP` only
- study size: `100-500` studies for the first baseline
- retrieval pool: mostly `train`
- evaluation set: small slice from `validate`, optionally `test` later
- one image per study for the first pass

Reasoning:

- the JPG release provides easier image handling and a standard split file
- frontal views reduce complexity compared with mixed frontal plus lateral studies
- small study-level subsets are enough to validate retrieval effects and hallucination behavior before scaling

The JPG dataset includes recommended train, validate, and test splits in `mimic-cxr-2.0.0-split.csv.gz`, and includes metadata such as `ViewPosition`. [PhysioNet MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.1.0/)

## 3. What image format and report format are available?

Available formats:

- `MIMIC-CXR`: images in `DICOM`, reports in free-text `.txt`
- `MIMIC-CXR-JPG`: images in `JPG`, reports are not duplicated there, but linked back to MIMIC-CXR

PhysioNet describes MIMIC-CXR as chest radiographs in DICOM format with free-text radiology reports. It describes MIMIC-CXR-JPG as JPG images derived from the original DICOM files and notes that the corresponding free-text reports remain available in MIMIC-CXR. [PhysioNet MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.1.0/) and [PhysioNet MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.1.0/)

## 4. How do we pair images and reports?

Pairing is done at the `study_id` level.

Important facts from the dataset structure:

- one study has one report
- one study may have one or more images
- multiple images can share the same `study_id`
- report files are named by `study_id`
- image files are keyed by `dicom_id`

MIMIC-CXR provides:

- `cxr-record-list.csv.gz` mapping image files to `study_id` and `subject_id`
- `cxr-study-list.csv.gz` mapping studies to patients
- per-study text reports stored as `s<study_id>.txt`

MIMIC-CXR-JPG provides:

- `mimic-cxr-2.0.0-metadata.csv.gz` with `dicom_id` and imaging metadata
- `mimic-cxr-2.0.0-split.csv.gz` with `dicom_id`, `study_id`, `subject_id`, and split

So the practical join is:

`split.csv + metadata.csv -> choose image by dicom_id/study_id -> link to s<study_id>.txt report`

Sources: [PhysioNet MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.1.0/) and [PhysioNet MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.1.0/)

## 5. What retrieval index do we build?

For the first real baseline, build a `study-level retrieval index`.

Each indexed item should represent one study and contain:

- `study_id`
- chosen frontal image path
- report text
- findings
- impression
- optional labels

Recommended index contents:

- retrieval embedding vector
- metadata needed to reconstruct the retrieved context

Output artifact:

```text
study_id -> embedding + report payload
```

This is better than indexing raw images only, because the retrieved unit in the generation prompt is a study report.

## 6. Do we retrieve using image similarity, report similarity, or image-text embedding similarity?

Recommendation:

1. final baseline target: `image-text embedding similarity`
2. temporary debug path: `report similarity`
3. avoid relying on pure report similarity for the actual experiment

Why:

- at inference time, the target report is unavailable, so pure report-to-report retrieval is unrealistic
- image-only similarity is better, but a shared image-text space is more natural for retrieving report-bearing studies from an image query
- a medical contrastive encoder such as BioMedCLIP is the cleanest first real retrieval direction

Therefore the planned real path is:

`target image embedding -> nearest study embeddings in a shared image-text space -> retrieved study reports`

This recommendation is an inference from the dataset structure and the baseline research goal, not a direct statement from PhysioNet.

## 7. Small working subset plan

Start with:

- `200` studies total
- `150` retrieval pool studies from `train`
- `50` evaluation studies from `validate`
- `AP` and `PA` only
- choose one frontal image per study

If you need an even smaller first run:

- `70` retrieval pool
- `30` eval

## 8. What is implemented in this repo now?

This repo now contains:

- [scripts/build_mimic_subset.py](/Users/mac/Documents/New%20project/RGCA/scripts/build_mimic_subset.py:1)
- [src/rgca_baseline/mimic_cxr.py](/Users/mac/Documents/New%20project/RGCA/src/rgca_baseline/mimic_cxr.py:1)
- [src/rgca_baseline/real_retrieval.py](/Users/mac/Documents/New%20project/RGCA/src/rgca_baseline/real_retrieval.py:1)

Current state:

- MIMIC subset ingestion is implemented for PhysioNet-style metadata and report folders
- lexical retrieval remains the default runnable backend
- a first `BiomedCLIPRetrieverBackend` scaffold is added as the real retrieval integration point

## 9. Recommended next implementation move

Once your local MIMIC files are available:

1. run `build_mimic_subset.py` to create a real JSONL subset
2. finish the `BiomedCLIPRetrieverBackend` embedding path
3. replace the mock generator with your chosen VLM
4. add a hallucination review table over mismatch generations
