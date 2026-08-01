# Submission Strategy 2026

This project currently has controlled baseline stress evidence and a clear RGCA method proposal. It is not yet ready to claim final VLM performance gains.

## Primary Target: Africa in AI Workshop at NeurIPS 2026

Recommended role: primary target.

Why it fits:

- The project is about trustworthy clinical AI in a resource-sensitive deployment context.
- The failure mode is globally relevant but especially important for settings where radiology capacity and model oversight may be limited.
- The Africa in AI workshop theme can support the broader motivation: safe, grounded AI systems for biomedical and clinical science.

Recommended framing:

```text
Retrieval can improve clinical report generation, but naive retrieved-report conditioning can also introduce unsupported findings. We propose Retrieval-Guided Cross-Attention (RGCA), a grounding-aware architecture that gates retrieved evidence before it influences visual report generation.
```

Minimum evidence needed before submission:

- Current stress baseline evidence.
- Real retrieval validation with BioMedCLIP or a medical CLIP-style model.
- At least one real VLM pilot run on a small MIMIC-CXR subset.
- A small manual review table with concrete hallucination examples.

## Secondary Target: ESMRMB 2026 Late-Breaking Abstract

Recommended role: secondary or work-in-progress target.

Why it can fit:

- The call accepts late-breaking work, Project Abstracts, and Registered Reports.
- The project is still evolving, which matches a work-in-progress venue.
- It could be useful for feedback from imaging researchers.

Main caution:

- ESMRMB is MRI-focused, while the current project is chest X-ray report generation. This is not a perfect topical fit unless the abstract emphasizes imaging-AI methodology and generalizable retrieval-grounding safety.

Recommended framing if submitted:

```text
Project Abstract: Retrieval-induced hallucination in radiology report generation and a proposed grounding-aware cross-attention mechanism.
```

Avoid framing it as an MRI-specific contribution unless the experiment is extended to MRI data.

## Conservative Claims We Can Make Now

- A real MIMIC-derived 500-study pilot subset has been prepared and validated.
- The baseline pipeline produces structured retrieval, mismatch, generation, and hallucination-evaluation artifacts.
- Controlled stress tests show that mismatched retrieved reports can propagate unsupported findings into generated reports.
- These findings motivate a gated retrieval-grounding architecture.

## Claims To Avoid Until Real VLM Experiments

- RGCA improves report generation.
- RGCA reduces hallucination in real VLM inference.
- BioMedCLIP retrieval improves clinical quality.
- The current numeric hallucination rates represent deployed VLM behavior.

## Decision

Submit to Africa in AI if the real retrieval and first real VLM pilot are completed in time.

Submit to ESMRMB as a Project Abstract only if we want early community feedback and the abstract is clearly labeled as ongoing methodological work rather than final clinical validation.
