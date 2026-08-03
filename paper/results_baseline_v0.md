# Baseline Results v0

## Experimental Setup

We evaluated a retrieval-augmented chest X-ray report generation baseline on a
500-study MIMIC-CXR pilot subset. The subset contains 400 studies used as the
retrieval pool and 100 studies reserved for evaluation. For this initial
real-generation milestone, we generated reports for 20 evaluation studies with
`llava-hf/llava-1.5-7b-hf`.

For each target study, we compared three generation conditions:

1. **No Retrieval:** the VLM receives the target image and a report-generation
   prompt only.
2. **Retrieval:** the VLM receives the target image plus top retrieved reports
   from the BioMedCLIP retrieval artifact.
3. **Mismatch:** the VLM receives the target image plus intentionally mismatched
   reports designed to test whether unsupported retrieved findings are copied.

All conditions used the same target studies and the same fixed retrieval
artifacts. This isolates the effect of retrieval context during generation.

## Quantitative Results

| Condition | n | Hallucination Rate | Retrieval-Induced Rate | Retrieval Copy Rate |
| --- | ---: | ---: | ---: | ---: |
| No Retrieval | 20 | 0.00 | 0.00 | 0.00 |
| Retrieval | 20 | 0.85 | 0.35 | 0.45 |
| Mismatch | 20 | 0.95 | 0.25 | 0.25 |

Under the rule-based evaluator, no-retrieval generation produced no detected
hallucinated pathology labels in this small pilot. In contrast, retrieval and
mismatch prompting produced substantially higher hallucination rates. The
retrieval condition showed a retrieval-induced hallucination rate of 0.35, and
the mismatch condition showed a retrieval-induced hallucination rate of 0.25.

## Manual Review

Because rule-based label extraction can over-count negated findings, we manually
reviewed all flagged retrieval and mismatch rows. Manual review identified 12
strong or moderate cases of retrieval-induced hallucination. The strongest
failure cases occurred when a target report described clear lungs or no acute
cardiopulmonary disease, but the retrieved report described focal consolidation
or pleural effusion, and the generated report repeated the retrieved abnormality.

This review also found several evaluator artifacts, especially cases where the
generated text contained phrases such as "no pneumonia" or "no pleural effusion"
but the rule-based evaluator still counted the label as present. These cases
were excluded from the strongest evidence category.

## Representative Failure Mode

In one mismatch case, the target reference report stated that the lungs were
clear with no pleural effusion or pneumothorax. The retrieved report described
focal left-base consolidation, possible aspiration or pneumonia, and central
vascular engorgement. The generated report then included focal left-base
consolidation and a small pleural effusion, despite those findings being absent
from the target reference. This is a clinically meaningful retrieval-induced
hallucination because the model appears to transfer abnormal retrieved evidence
into the target report.

## Interpretation

These results support the motivating hypothesis for RGCA: prompt-based retrieval
can introduce unsupported clinical findings when retrieved reports are noisy or
mismatched. The baseline does not yet establish that RGCA solves the problem.
Rather, it establishes the failure mode that RGCA is designed to address.

## Limitations

This is an early pilot with 20 real-generation cases. The VLM is a general
LLaVA checkpoint rather than a specialized radiology report generation model.
The automatic evaluator is rule-based and sensitive to negation errors. For
publication, the result should be strengthened with a larger evaluation set,
improved label extraction, and additional model backends.
