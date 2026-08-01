# MIMIC Pilot Baseline v0 Evidence Summary

Run date: 2026-08-01

Environment: Kaggle notebook

Notebook: `notebooks/kaggle_research_safe_baseline.ipynb`

Execution mode: `stress`

## Dataset

The run used an attached private Kaggle dataset:

```text
/kaggle/input/datasets/emmapi/rgca-private-dataset/mimic_subset.jsonl
```

Validation passed:

- Total studies: 500
- Retrieval pool: 400
- Evaluation studies: 100
- Missing required fields: 0
- Duplicate study IDs: 0
- Dataset SHA256: `89187a1df84a81506e12762505722a703f4ae916cc05cdb7c86c91d2ef69f3b1`

## Output Artifacts

Kaggle wrote the experiment outputs to:

```text
/kaggle/working/rgca_experiments/mimic_pilot_baseline_v0
/kaggle/working/rgca_private_dataset.zip
```

The zip artifact was confirmed to exist:

```text
/kaggle/working/rgca_private_dataset.zip
size: 0.858 MB
```

## Results

| experiment | mode | retriever | generator | top_k | retrieval_pool_size | eval_size | n | hallucination_rate | retrieval_induced_hallucination_rate | retrieval_copy_rate | total_hallucinated_labels | total_retrieval_induced_hallucinations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E01_stress_lexical_k1 | retrieval | lexical | retrieval_copy_stress | 1 | 400 | 100 | 100 | 0.93 | 0.29 | 0.49 | 166 | 32 |
| E01_stress_lexical_k1 | mismatch | lexical | retrieval_copy_stress | 1 | 400 | 100 | 100 | 1.0 | 0.6 | 0.6 | 274 | 120 |
| E02_stress_lexical_k3 | retrieval | lexical | retrieval_copy_stress | 3 | 400 | 100 | 100 | 1.0 | 0.57 | 0.72 | 184 | 64 |
| E02_stress_lexical_k3 | mismatch | lexical | retrieval_copy_stress | 3 | 400 | 100 | 100 | 1.0 | 0.91 | 0.91 | 327 | 156 |
| E03_stress_lexical_k5 | retrieval | lexical | retrieval_copy_stress | 5 | 400 | 100 | 100 | 1.0 | 0.67 | 0.78 | 193 | 80 |
| E03_stress_lexical_k5 | mismatch | lexical | retrieval_copy_stress | 5 | 400 | 100 | 100 | 1.0 | 0.91 | 0.91 | 347 | 194 |
| E04_stress_hashing_text_k3 | retrieval | hashing_text | retrieval_copy_stress | 3 | 400 | 100 | 100 | 1.0 | 0.61 | 0.75 | 190 | 71 |
| E04_stress_hashing_text_k3 | mismatch | hashing_text | retrieval_copy_stress | 3 | 400 | 100 | 100 | 0.99 | 0.23 | 0.23 | 225 | 46 |

## Preliminary Interpretation

This run supports the baseline failure-mode claim under controlled stress conditions: mismatched retrieved reports can strongly contaminate generated reports with unsupported findings.

The strongest mismatch condition was lexical retrieval with `top_k=3` and `top_k=5`, both producing a retrieval-induced hallucination rate of `0.91` on 100 evaluation studies.

The `top_k` trend is consistent with the intended stress hypothesis: increasing the number of retrieved reports increased opportunities for copied unsupported findings.

## Important Caveat

This is not yet real VLM evidence. The generator backend was `retrieval_copy_stress`, which is a controlled stress-test generator designed to expose retrieval-copy failure modes. These results should be described as controlled baseline stress evidence, not final model performance.

The next required step is to replace the stress generator with a real VLM backend while preserving the same dataset, retrieval, mismatch, and evaluation contract.
