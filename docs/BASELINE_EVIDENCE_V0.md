# Baseline Evidence v0

This document freezes the first real-generation evidence milestone for the
baseline phase before RGCA. The goal of this milestone is not to claim
state-of-the-art report generation. The goal is to show that prompt-based
retrieval can influence a VLM and can introduce unsupported clinical findings.

## Frozen Run

| Field | Value |
| --- | --- |
| Dataset | MIMIC-CXR pilot subset |
| Subset size | 500 studies |
| Retrieval pool | 400 studies |
| Evaluation pool | 100 studies |
| Real-generation eval size | 20 studies |
| Missing eval images | 0 |
| Retriever artifacts | BioMedCLIP retrieval validation artifacts |
| Generator backend | `hf_vlm` |
| Generator model | `llava-hf/llava-1.5-7b-hf` |
| Modes | `no_retrieval`, `retrieval`, `mismatch` |
| Dataset SHA256 | `89187a1df84a81506e12762505722a703f4ae916cc05cdb7c86c91d2ef69f3b1` |

The corresponding local evidence artifact is:

```text
/Users/mac/Desktop/career/Research/rgca/rgca_real_generation_evidence_limit20_llava_v0.zip
```

The extracted local analysis folder is:

```text
outputs/real_generation_limit20_llava_v0
```

The generated reports and evaluation summaries are not committed because
`outputs/` is ignored and the data are derived from restricted clinical data.

## Quantitative Results

| Mode | n | Hallucination Rate | Retrieval-Induced Hallucination Rate | Retrieval Copy Rate | Hallucinated Labels | Retrieval-Induced Labels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No retrieval | 20 | 0.00 | 0.00 | 0.00 | 0 | 0 |
| Retrieval | 20 | 0.85 | 0.35 | 0.45 | 36 | 9 |
| Mismatch | 20 | 0.95 | 0.25 | 0.25 | 38 | 10 |

## Manual Review

The automatic evaluator was followed by conservative manual review of all
flagged retrieval and mismatch rows.

| Review category | Count |
| --- | ---: |
| Reviewed flagged rows | 28 |
| Strong evidence | 7 |
| Moderate evidence | 5 |
| Weak or artifact | 14 |
| Supported copy, not hallucination | 2 |
| Strong/moderate retrieval-induced evidence rows | 12 |

Manual review confirmed that several retrieved findings were copied into the
generated report despite being unsupported by the target reference report.
The strongest examples are mismatch cases where the target report explicitly
states clear lungs, no effusion, no pneumonia, or no acute cardiopulmonary
disease, while the retrieved report contains consolidation or pleural effusion
and the generated report repeats those findings.

## Conservative Interpretation

This milestone supports the baseline claim:

> Prompt-based retrieval can introduce unsupported clinical findings into
> VLM-generated chest X-ray reports.

The evidence should be stated conservatively:

- The result is a pilot on 20 evaluation studies, not a definitive benchmark.
- The rule-based evaluator over-counts some negated findings, especially when
  generated text says "no pneumonia" or "no pleural effusion."
- Manual review therefore matters more than the raw automatic hallucination
  numbers.
- The strongest claim is not that every automatic flag is clinically correct,
  but that manual review confirms multiple clinically meaningful
  retrieval-induced hallucinations.

## Why This Justifies RGCA

The baseline demonstrates that retrieved reports can enter the generation
process too strongly. A standard prompt-retrieval pipeline has no explicit
mechanism to decide whether retrieved evidence is visually supported by the
target image. RGCA is motivated as a control mechanism that should align,
gate, and verify retrieved evidence before it influences visual decoding.

## Next Experiment

Run a larger real-generation pilot:

```text
Experiment: real_generation_limit50_llava_v0
Eval size: 50
Generator: llava-hf/llava-1.5-7b-hf
Retriever artifacts: same BioMedCLIP retrieval/mismatch files
```

This will test whether the retrieval-induced hallucination signal remains
stable when moving from 20 to 50 evaluation studies.
