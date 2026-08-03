# Retrieval-Guided Cross-Attention for Reducing Retrieval-Induced Hallucination in Chest X-ray Report Generation

## Abstract

Retrieval-augmented vision-language models (VLMs) can improve chest X-ray report generation by providing similar prior reports as clinical context. However, retrieved reports are not always faithful to the target image. When retrieval is noisy, partially relevant, or clinically mismatched, a VLM may copy unsupported findings from retrieved evidence into the generated report. We study this failure mode as retrieval-induced hallucination: a generated clinical finding that is present in retrieved evidence but absent from the target reference report. Before introducing our proposed Retrieval-Guided Cross-Attention (RGCA) mechanism, we construct a baseline pipeline over a MIMIC-CXR pilot subset using BioMedCLIP retrieval artifacts and LLaVA-style generation. The baseline compares image-only generation, prompt-based retrieval, and deliberately mismatched retrieval. In a 20-study real-generation pilot, prompt retrieval increased automatic hallucination flags compared with no retrieval, and manual review confirmed multiple strong or moderate cases where unsupported retrieved findings appeared in generated reports. These results motivate RGCA, a planned fusion module that aligns retrieved evidence with image tokens, gates retrieval at the report level, and uses gated cross-attention to condition visual representations before decoding. The current paper draft presents the problem, baseline evidence, evaluation protocol, and method design; larger-scale experiments and RGCA implementation remain future milestones.

## 1. Introduction

Chest X-ray report generation is a high-stakes medical language generation task. A useful system must not only produce fluent radiology text, but also ensure that generated findings are grounded in the target image. This is difficult for modern medical VLMs because they combine visual representations, language priors, and external context. When these signals conflict, the generated report may contain clinically meaningful findings that are not supported by the image.

Retrieval-augmented generation is attractive in this setting because retrieved reports can provide examples of similar studies, common phrasing, and clinically relevant context. A target image can be embedded, used to retrieve top-k similar studies, and paired with their reports in the VLM prompt. This simple approach is easy to implement and often improves report fluency. The risk is that retrieved evidence may become an uncontrolled source of clinical content. If the retrieved reports mention pleural effusion, consolidation, cardiomegaly, or another finding that is absent from the target image, the model may still include that finding in its output.

We call this failure mode retrieval-induced hallucination. In this project, a retrieval-induced hallucination occurs when a generated finding is present in retrieved evidence but absent from the target reference report. This is narrower than hallucination in general. The goal is not to solve every form of medical hallucination, but to isolate the specific failure caused by retrieval and then design a mechanism that reduces it.

The core hypothesis is that retrieval should not merely be appended to the prompt. Instead, retrieved evidence should be aligned with image tokens, gated according to relevance, and allowed to influence generation only through a controlled visual grounding pathway. This motivates Retrieval-Guided Cross-Attention (RGCA), a proposed module that uses retrieved clinical evidence to modulate visual attention rather than providing raw text for the decoder to copy.

This draft reports the first baseline milestone before RGCA. The purpose of the baseline is not to beat state-of-the-art report generation systems. The purpose is to prove that the failure mode exists in a runnable experimental pipeline. We compare image-only generation, prompt retrieval, and mismatched retrieval on a MIMIC-CXR pilot subset. The resulting evidence shows that retrieval can change generated reports and can introduce unsupported findings, justifying the need for a more controlled retrieval-fusion mechanism.

## 2. Related Work

### Medical Vision-Language Models

Recent medical VLMs adapt general multimodal architectures to biomedical and clinical settings through image-text pretraining, instruction tuning, and domain-specific alignment. LLaVA-Med extends visual instruction tuning to biomedical images, Med-Flamingo explores few-shot medical multimodal reasoning, and CheXagent focuses on chest X-ray interpretation. BioMedCLIP provides biomedical image-text embeddings that are useful for retrieval and representation alignment.

These models show that clinical multimodal reasoning is feasible, but they do not fully solve image-grounded faithfulness. Many systems can generate plausible reports while still relying on language priors or external context in ways that are not visually justified.

### Retrieval-Augmented Generation

Retrieval-augmented systems improve generation by adding external memory, similar cases, or relevant documents. In clinical report generation, retrieved reports can help with structure, terminology, and disease-context priors. However, the most common integration strategy is prompt concatenation: retrieved text is simply added to the input context. This gives the model access to useful information, but it does not explicitly control whether retrieved findings are supported by the target image.

Prior retrieval and memory architectures, including RETRO-style systems, suggest that retrieval can be fused structurally through attention rather than only through text concatenation. The central question for this project is whether such structured fusion can make medical retrieval safer by reducing the tendency to copy unsupported retrieved findings.

### Hallucination and Grounding

Medical VLM hallucinations occur when a generated report includes findings that are not supported by the target image, reference report, or clinical evidence. Errors may involve false pathologies, incorrect severity, incorrect laterality, or negation mistakes. Retrieval introduces an additional source of hallucination: a model may copy a finding because it appears in retrieved evidence, even when the target image does not support it.

Existing hallucination evaluations often measure unsupported findings broadly. This project narrows the target to retrieval-induced hallucination, which can be isolated by comparing the generated report, reference report, and retrieved reports for each target image.

### Research Gap

Existing clinical VLMs and retrieval-augmented systems generally treat retrieved evidence as context rather than as a controlled grounding signal. As a result, retrieved findings may dominate generation when retrieval is noisy or mismatched. The gap is not simply the absence of retrieval, but the absence of an explicit mechanism that regulates how much retrieved evidence should influence visual reasoning.

## 3. Method

### Problem Definition

Let `x` be a target chest X-ray image, `y_ref` its reference radiology report, and `R(x) = {r_1, ..., r_k}` the top-k retrieved reports. A report generator produces `y_gen` under one of three conditions: no retrieval, prompt retrieval, or mismatched retrieval.

A retrieval-induced hallucination occurs when a clinically meaningful finding `p` satisfies:

```text
p in generated report
p in at least one retrieved report
p not in the target reference report
```

The baseline studies whether this event occurs under prompt-based retrieval. RGCA is then motivated as an architectural response to this failure mode.

### Baseline Pipeline

The baseline pipeline is:

```text
Image -> Retriever -> Retrieved Reports -> VLM -> Generated Report
```

We evaluate three modes:

```text
1. No Retrieval
   Image -> VLM -> Report

2. Prompt Retrieval
   Image -> Top-k retrieved reports -> Prompt -> VLM -> Report

3. Retrieval Mismatch
   Image -> Deliberately mismatched reports -> Prompt -> VLM -> Report
```

The mismatch mode is central to the experiment because it stress-tests whether the generator copies unsupported retrieved findings.

### RGCA Design

RGCA is a proposed fusion mechanism for future experiments. Instead of giving the decoder raw retrieved reports as prompt text, RGCA encodes retrieved reports separately and uses them to modulate visual representations.

The planned RGCA pipeline is:

```text
Image Tokens
Retrieved Reports
      |
      v
Alignment Module
      |
      v
Retrieval Gate
      |
      v
RGCA Block
      |
      v
Grounded Features
      |
      v
Decoder
```

For a target image embedding `v_bar` and retrieved report embedding `r_bar_k`, report-level alignment is computed as:

```text
alpha_k = cosine(v_bar, r_bar_k)
```

A learned gate decides how much each retrieved report can influence visual attention:

```text
g_k = sigmoid(w_g^T [v_bar; r_bar_k; alpha_k] + b_g)
```

The retrieved report tokens are then used as a gated memory for cross-attention. Image tokens act as queries, while retrieved-report tokens provide gated keys and values. The gate is applied as a log-bias in the attention logits so that low-relevance reports can be suppressed rather than simply rescaled.

The output is fused back into the visual stream using a residual connection:

```text
H = LayerNorm(V + lambda * Attn)
```

The design goal is for retrieval to guide visual grounding rather than to become text that the decoder copies directly.

## 4. Experimental Setup

### Dataset Access

The primary dataset is MIMIC-CXR with MIMIC-CXR-JPG. Access is through PhysioNet and requires a credentialed PhysioNet account, completion of the required CITI training, and acceptance of the dataset data-use agreement. MIMIC-CXR provides free-text reports, while MIMIC-CXR-JPG provides image files and metadata in a more convenient image format for model experiments.

### Pilot Subset

The current pilot uses 500 studies:

```text
400 studies: retrieval pool
100 studies: evaluation pool
20 studies: real-generation evaluation in the current frozen run
```

The intended larger next run uses 50 real-generation evaluation studies with the same retrieval setup.

For the first baseline, the subset is restricted to frontal chest X-rays where possible, using one image per study. This keeps the experiment small enough to run on Kaggle while preserving the study-level image-report structure required for retrieval.

### Image and Report Formats

MIMIC-CXR-JPG provides chest X-ray images as JPG files. MIMIC-CXR provides radiology reports as text files organized by patient and study. Each study may contain multiple images, but each study has one report. The baseline therefore pairs images and reports at the `study_id` level.

Each study record uses the following schema:

```json
{
  "study_id": "...",
  "image_path": "...",
  "reference_report": "...",
  "findings": "...",
  "impression": "...",
  "labels": ["pleural effusion", "cardiomegaly"],
  "split": "train | validate | test"
}
```

### Retrieval Index

The retrieval index is study-level. Each indexed item corresponds to one study, not one report section or one image crop. The real retrieval stage uses BioMedCLIP-style image-text embeddings and nearest-neighbor search. The query is the target chest X-ray image; the retrieved units are clinically similar studies; the retrieved reports are passed into the generation prompt for the baseline.

The final similarity strategy is image-text embedding similarity because the target report is unavailable at inference time. Report-to-report retrieval is only a debug fallback and is not treated as the final experimental design.

### Generation

The current real-generation milestone uses `llava-hf/llava-1.5-7b-hf` through the repository's `hf_vlm` backend. This is an intentionally pragmatic first model choice: it validates the real image-to-text generation path and shows retrieval effects, but it is not a final radiology-specialized generator.

### Evaluation

The primary baseline metric is retrieval-induced hallucination rate:

```text
retrieval-induced hallucination rate =
cases where generated findings copy unsupported retrieved findings / total evaluated cases
```

The current evaluator uses pathology-level label matching and retrieval overlap. Because rule-based matching can over-count negated findings, all flagged rows from the pilot were manually reviewed. Manual review is treated as the stronger evidence source for qualitative claims.

## 5. Results

### Real-Generation Pilot

The current frozen milestone evaluates 20 studies using the real LLaVA generation backend.

| Condition | n | Hallucination Rate | Retrieval-Induced Rate | Retrieval Copy Rate |
| --- | ---: | ---: | ---: | ---: |
| No Retrieval | 20 | 0.00 | 0.00 | 0.00 |
| Prompt Retrieval | 20 | 0.85 | 0.35 | 0.45 |
| Mismatch Retrieval | 20 | 0.95 | 0.25 | 0.25 |

The automatic evaluator found substantially more hallucination flags in the retrieval and mismatch settings than in the no-retrieval setting. The retrieval condition produced 9 retrieval-induced label events, while the mismatch condition produced 10.

### Manual Review

Manual review was performed because the automatic evaluator is sensitive to negation artifacts. Across 28 flagged retrieval and mismatch rows, manual review found:

| Category | Count |
| --- | ---: |
| Strong evidence | 7 |
| Moderate evidence | 5 |
| Weak or evaluator artifact | 14 |
| Supported copy, not hallucination | 2 |

The conservative manual result is that 12 flagged rows provide strong or moderate evidence of retrieval-induced hallucination. The strongest cases involve target reports describing clear lungs, no pleural effusion, no pneumonia, or no acute cardiopulmonary disease, while retrieved reports contain consolidation or pleural effusion and the generated report repeats those unsupported abnormalities.

## 6. Analysis

The baseline supports the first research claim: prompt-based retrieval can introduce unsupported clinical findings into generated chest X-ray reports. This matters because retrieval is often treated as an unqualified improvement. The pilot suggests that retrieval can improve access to useful context while also increasing the risk of copying wrong context.

The most important observation is not the exact automatic hallucination rate, because the evaluator is still noisy. The important observation is that manual review confirms clinically meaningful cases where retrieved evidence appears to enter the generated report despite being unsupported by the target reference.

This is precisely the failure mode RGCA is designed to address. Prompt retrieval gives the model retrieved text without an explicit verification step. RGCA instead creates an inspectable alignment score and relevance gate before retrieval influences the visual representation.

The pilot also exposes an evaluation lesson: automatic pathology matching must handle negation carefully. Phrases such as "no pneumonia" should not be counted as positive pneumonia findings. The next evaluation milestone should strengthen label extraction using CheXbert or RadGraph-style tools and keep manual review for high-impact examples.

## 7. Limitations

This draft reports an early pilot rather than a final benchmark. The current real-generation run uses only 20 evaluation studies. The generator is a general LLaVA checkpoint rather than a radiology-specialized report generation model. The automatic evaluator is rule-based and can over-count negated findings. Manual review is conservative but not yet radiologist-reviewed.

RGCA has been specified but not yet implemented or evaluated. Therefore, the current evidence justifies the RGCA research direction; it does not yet demonstrate that RGCA solves the problem. Larger runs, stronger label extraction, top-k ablations, additional generators, and eventual RGCA experiments are still needed before final submission.

## 8. Conclusion

This work studies retrieval-induced hallucination in chest X-ray report generation. The first baseline milestone shows that prompt-based retrieval can change VLM outputs and can introduce unsupported clinical findings when retrieved reports are noisy or mismatched. This motivates a controlled fusion mechanism, Retrieval-Guided Cross-Attention, where retrieved evidence is aligned, gated, and fused with image tokens before decoding.

The next milestone is to scale the real-generation baseline from 20 to 50 evaluation studies, strengthen the evaluator, and then implement RGCA-Core as the first controlled retrieval-fusion model.

## Appendix A. Dataset and Retrieval Setup

### How do we access the dataset?

Use the official PhysioNet releases:

```text
MIMIC-CXR v2.1.0
MIMIC-CXR-JPG v2.1.0
```

Access requires a credentialed PhysioNet account, CITI training, and DUA acceptance.

### Which subset do we start with?

The planned pilot starts with a small study-level subset:

```text
150-400 retrieval-pool studies from train
50-100 evaluation studies from validate/test
PA/AP frontal views only
one frontal image per study
```

The current frozen evidence uses 500 studies total, with 400 retrieval-pool studies and 100 evaluation-pool studies.

### What image format and report format are available?

Images are available as JPG files from MIMIC-CXR-JPG. Reports are available as free-text `.txt` files from MIMIC-CXR.

### How do we pair images and reports?

Pair at study level using `study_id`. Metadata identifies `dicom_id`, `study_id`, view position, and split. Reports are stored as `s<study_id>.txt`. For the first baseline, select one frontal image per study and pair it with the corresponding report.

### What retrieval index do we build?

Build a study-level nearest-neighbor index. Each record stores `study_id`, `image_path`, report text, findings, impression, labels, and split. The retrieval pool excludes evaluation studies to avoid leakage.

### Do we retrieve using image similarity, report similarity, or image-text embedding similarity?

The final retrieval strategy is image-text embedding similarity. The target image is embedded and used to retrieve nearby study embeddings. Retrieved study reports are then passed to the generator. Lexical or report-similarity retrieval may be used only as a debugging fallback.

## Appendix B. Planned Next Experiments

1. Scale real generation from 20 to 50 evaluation studies using the prepared Kaggle notebook.
2. Repeat with top-k values `k = 1, 3, 5`.
3. Add a stronger label extractor, preferably CheXbert and eventually RadGraph-style entity extraction.
4. Add a radiology-specialized generator if compute allows.
5. Implement RGCA-Core and compare it against prompt retrieval under the same retrieval artifacts.

