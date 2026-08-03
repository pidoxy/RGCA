# Baseline Evidence Table v0

| Mode | n | Hallucination Rate | Retrieval-Induced Rate | Retrieval Copy Rate | Manual Review Finding |
| --- | ---: | ---: | ---: | ---: | --- |
| No Retrieval | 20 | 0.00 | 0.00 | 0.00 | No detected hallucinations under current evaluator |
| Retrieval | 20 | 0.85 | 0.35 | 0.45 | Multiple copied unsupported findings; strongest clean-retrieval cases involve cardiomegaly, device details, atelectasis, and pleural effusion |
| Mismatch | 20 | 0.95 | 0.25 | 0.25 | Strongest evidence: normal/clear target reports contaminated by retrieved consolidation or pleural effusion |

Manual review of 28 flagged rows found:

| Manual Review Category | Count |
| --- | ---: |
| Strong evidence | 7 |
| Moderate evidence | 5 |
| Weak/artifact | 14 |
| Supported copy, not hallucination | 2 |
| Strong or moderate retrieval-induced evidence | 12 |
