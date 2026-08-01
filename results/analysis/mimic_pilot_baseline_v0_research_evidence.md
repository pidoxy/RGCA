# Baseline Evidence Note: Retrieval-Induced Hallucination Stress Test

Run date: 2026-08-01

Dataset: private Kaggle pilot subset derived from MIMIC-CXR

Notebook: `notebooks/kaggle_research_safe_baseline.ipynb`

Primary artifact:

```text
/kaggle/working/mimic_pilot_baseline_v0.zip
```

## What Was Tested

This baseline stress run tested whether retrieved reports can contaminate generated radiology reports when the retrieved evidence is mismatched with the target study.

The pipeline used the existing baseline contract:

```text
Image/study record -> Retriever -> Retrieved reports -> Stress generator -> Generated report -> Hallucination evaluator
```

This is not final VLM performance. It is a controlled stress experiment designed to verify the failure mode before implementing RGCA.

## Dataset Contract

The pilot subset passed validation:

| Field | Value |
| --- | --- |
| Total studies | 500 |
| Retrieval pool | 400 |
| Evaluation studies | 100 |
| Missing required fields | 0 |
| Duplicate study IDs | 0 |
| Dataset SHA256 | `89187a1df84a81506e12762505722a703f4ae916cc05cdb7c86c91d2ef69f3b1` |

## Key Result

The mismatch setting produced substantially higher retrieval-induced hallucination than the clean retrieval setting.

| Condition | Top-k | Retrieval-induced hallucination rate | Retrieval copy rate |
| --- | ---: | ---: | ---: |
| Clean lexical retrieval | 1 | 0.29 | 0.49 |
| Mismatched lexical retrieval | 1 | 0.60 | 0.60 |
| Clean lexical retrieval | 3 | 0.57 | 0.72 |
| Mismatched lexical retrieval | 3 | 0.91 | 0.91 |
| Clean lexical retrieval | 5 | 0.67 | 0.78 |
| Mismatched lexical retrieval | 5 | 0.91 | 0.91 |
| Clean hashing-text retrieval | 3 | 0.61 | 0.75 |
| Mismatched hashing-text retrieval | 3 | 0.23 | 0.23 |

## Interpretation

The baseline supports the central motivation for RGCA: retrieval is not automatically safe. When retrieved reports are mismatched, unsupported findings can be copied into the generated report.

The strongest evidence appears in the lexical mismatch condition. At `top_k=3` and `top_k=5`, the retrieval-induced hallucination rate reached `0.91`, meaning most mismatch cases copied unsupported retrieved findings into the generated report under the stress generator.

The top-k pattern is also informative. Increasing retrieved context from `k=1` to `k=3` or `k=5` increased the opportunity for unsupported evidence to enter the output. This motivates a gated retrieval mechanism rather than naive prompt concatenation.

## What This Proves

This run proves the experimental pipeline can:

- Prepare and validate a real MIMIC-derived pilot subset.
- Run retrieval and mismatch conditions over 100 evaluation studies.
- Produce structured generation and retrieval artifacts.
- Compute retrieval-induced hallucination metrics.
- Package the full evidence bundle for review and reuse.

## What This Does Not Prove Yet

This run does not prove that a deployed VLM will hallucinate at the same rate. The generator was `retrieval_copy_stress`, a controlled stress backend used to expose the mechanism clearly.

Before making final model-performance claims, the same experiment contract must be rerun with:

- Real image-aware retrieval, ideally BioMedCLIP or another medical CLIP-style encoder.
- A real VLM backend, such as LLaVA-Med or a compatible checkpoint.
- A stronger label extractor, such as CheXbert or RadGraph-style extraction.

## Paper-Ready Claim

Conservative wording:

> In a controlled baseline stress test on a 500-study MIMIC-CXR pilot subset, mismatched retrieved reports substantially increased unsupported finding propagation into generated reports. This supports the need for retrieval-aware grounding mechanisms rather than naive report concatenation.

Avoid claiming:

> RGCA improves VLM report generation.

That claim requires the real VLM and RGCA experiments.
