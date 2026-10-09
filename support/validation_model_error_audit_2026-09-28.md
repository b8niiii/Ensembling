# Qualitative model-error audit on reviewed validation annotations

This report compares the **already saved** zero-shot Q8 outputs in `data/pilot_runs/slm_zero_shot_q8_schema_v2/` with the 12 text-only answers in [validation_blind_annotations_2026-09-28.md](validation_blind_annotations_2026-09-28.md). The answers were fixed before opening these bags' model outputs or distant-supervision labels. Annotation JSON SHA-256: `cc7f17b6c2487266f1f3e3c4ba99d5f4bc584917ce35109490d69856608bda90`. The 12-bag sample is deliberately balanced and far too small for an overall model-performance claim.

Before comparing predictions, a hand-constructed five-case scorer check passed: TP=2, FP=2, FN=3, precision=0.5, recall=0.4, micro-F1=4/9. Invalid saved answers retain the pilot's original rule: their scored positive prediction is empty.

| Model | Exact answers on 6 positive bags | Negative bags with ≥1 predicted relation, out of 6 | Invalid answers | TP / FP / FN | Audit micro-F1* |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | 2 | 5 | 4 | 2 / 17 / 4 | 0.160 |
| Falcon-H1-0.5B-Instruct | 0 | 5 | 1 | 0 / 17 / 6 | 0 |
| LFM2.5-350M | 0 | 6 | 1 | 1 / 109 / 5 | 0.017 |

\*Only a diagnostic on these 12 reviewed bags, under the saved strict parser; it is **not** the 150-bag distant-label F1, an official manual-test score, or a reliable estimate of population performance.

The concrete errors are more informative than this tiny F1. Qwen exactly matches **Atlanta → Turner Field** and **Jeffrey Katzenberg → DreamWorks Animation**, but repeats labels on four bags, causing whole answers to be discarded. It predicts unrelated positives on five of six text-negative bags. Falcon repeatedly selects `/location/neighborhood/neighborhood_of` for pairs where the reviewed answer is broad containment in the **opposite head-to-tail direction** (for example **Queens → Astoria**); its JSON validity does not imply correct relations. LFM outputs many unrelated labels on almost every bag: for **Iran → Tehran** it returns **all 24 positive labels**, and for **New York → Queens** it includes the reviewed relation among **22 predictions**, creating 21 false positives.

After the reviewed answers were fixed, five bags were found to disagree with the distant-supervision labels: **Honolulu → Hawaii Kai**, **Betty Smith → Brooklyn**, **Finland → Helsinki**, **Iran → Tehran**, and **Scotland → Edinburgh**. The extra distant labels include geographic containment, capital, residence, and birth-place facts that the **supplied sentences** do not clearly express under the strict text-only rule. This does not prove the underlying world facts are false. For Iran → Tehran, the pilot supplies **20 of 101** records; omitted records could contain evidence. The other four disagreement bags were supplied in full, so the 20-record cap cannot explain those differences.

[Gao et al. (2021)](https://aclanthology.org/2021.findings-acl.112.pdf) report that their official test labels were made by asking annotators whether each sentence expresses one or more of the 25 relations, without showing distant labels or model predictions; bag labels are unions of sentence-level annotations. Our strict rule is intended as a development audit in that spirit, but these reviewed validation annotations are **not** the official test annotations and may differ from how its annotators interpret implicit facts.

The evidence criterion was subsequently fixed: require a relation explicitly expressed in the supplied sentences for the ordered pair. The shared abstention prompt and duplicate-handling parser were later revised and tested in a separately versioned [v0.2 validation smoke run](protocol.md). This dated audit and its saved model outputs remain historical evidence; current model selection focuses on the encoder comparisons in [notes.md](notes.md). Do not use the manual test to select the method.
