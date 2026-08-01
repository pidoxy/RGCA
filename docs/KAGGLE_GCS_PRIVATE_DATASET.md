# Kaggle + GCS Private Dataset Setup

This guide explains how to use a Google Cloud Storage bucket from Kaggle and save the RGCA pilot data as a private Kaggle dataset.

## What You Have

You linked this GCS bucket:

```text
gs://mimic-cxr-jpg-2.1.0.physionet.org
```

The browser URL is:

```text
https://console.cloud.google.com/storage/browser/mimic-cxr-jpg-2.1.0.physionet.org
```

Use the `gs://` path in notebooks and terminal commands.

## Important Dataset Split

The JPG bucket contains:

- JPG images
- MIMIC-CXR-JPG metadata
- split file
- CheXpert labels

It does not replace the MIMIC-CXR reports. For report generation, we still need:

```text
mimic-cxr-reports.zip
```

So the minimum real-data setup is:

```text
mimic-cxr-2.0.0-metadata.csv.gz
mimic-cxr-2.0.0-split.csv.gz
mimic-cxr-2.0.0-chexpert.csv.gz
mimic-cxr-reports.zip extracted to mimic-cxr/files/
```

For the current mock/stress baseline suite, the subset JSONL is enough once built. For a real VLM, copy the pilot JPG images too.

## Kaggle Notebook Path

Use:

```text
notebooks/baseline_experiment.ipynb
```

The notebook now has a `Restore Data From GCS Or PhysioNet` section.

## Step 1: Pull The Latest Repo

```bash
%cd /kaggle/working
!rm -rf RGCA
!git clone https://github.com/pidoxy/RGCA.git
%cd /kaggle/working/RGCA
!python -m pip install -e .
```

## Step 2: Check GCS Access

In the notebook, run:

```python
GCS_JPG_BUCKET = "gs://mimic-cxr-jpg-2.1.0.physionet.org"
```

Then:

```bash
!gcloud storage ls gs://mimic-cxr-jpg-2.1.0.physionet.org
```

If this fails, Kaggle does not currently have access to the GCS bucket.

## Step 3: Restore Small Metadata Files

Set:

```python
RESTORE_SMALL_FILES_FROM_GCS = True
```

Then run the metadata restore cell. It copies only:

```text
mimic-cxr-2.0.0-metadata.csv.gz
mimic-cxr-2.0.0-split.csv.gz
mimic-cxr-2.0.0-chexpert.csv.gz
```

It does not download the full image dataset.

## Step 4: Restore Reports

If you have a reports GCS bucket, set:

```python
GCS_REPORTS_BUCKET = "gs://..."
RESTORE_REPORTS_FROM_GCS = True
```

If not, download reports from PhysioNet:

```python
DOWNLOAD_REPORTS_FROM_PHYSIONET = True
PHYSIONET_USERNAME = "your_username"
```

The notebook will ask for your PhysioNet password and download:

```text
https://physionet.org/files/mimic-cxr/2.1.0/mimic-cxr-reports.zip
```

## Step 5: Build The Pilot Subset

Run the subset build section. Recommended first pilot:

```text
400 retrieval-pool studies
100 eval studies
```

Output:

```text
/kaggle/working/rgca_pilot_500/data/mimic_subset.jsonl
```

## Step 6: Optional Pilot Images

For the current mock/stress baseline, images are not opened.

For real VLM work, set:

```python
COPY_PILOT_IMAGES_FROM_GCS = True
```

This copies only the images referenced in `mimic_subset.jsonl`, not the full JPG dataset.

Output:

```text
/kaggle/working/rgca_private_dataset/mimic-cxr-jpg/files/
```

## Step 7: Run Experiments

```bash
!python scripts/run_experiment_suite.py \
  --config configs/mimic_pilot_suite.json \
  --input /kaggle/working/rgca_pilot_500/data/mimic_subset.jsonl \
  --output-dir /kaggle/working/rgca_experiments/mimic_pilot_baseline_v0 \
  --overwrite
```

## Step 8: Summarize Results

```bash
!python scripts/summarize_experiment_suite.py \
  --manifest /kaggle/working/rgca_experiments/mimic_pilot_baseline_v0/suite_manifest.json \
  --output-dir /kaggle/working/rgca_experiments/mimic_pilot_baseline_v0/tables
```

## Step 9: Package A Private Kaggle Dataset

The notebook packages:

```text
/kaggle/working/rgca_private_dataset.zip
```

Contents:

```text
mimic_subset.jsonl
PRIVATE_DATASET_MANIFEST.json
rgca_experiment_outputs/
mimic-cxr-jpg/files/        # only if pilot images were copied
```

## Step 10: Save It Privately

In Kaggle:

1. Open the right-side `Output` panel.
2. Download `rgca_private_dataset.zip`, or use the output folder.
3. Go to `Kaggle Datasets`.
4. Click `New Dataset`.
5. Upload the zip contents.
6. Set visibility to `Private`.
7. In future notebooks, click `Add Input` and attach this private dataset.

The notebook auto-detects `mimic_subset.jsonl` under `/kaggle/input`, so future runs can skip raw GCS/PhysioNet restore.

## Safety Rule

Keep the dataset private. It may contain MIMIC-CXR-derived clinical report text.
