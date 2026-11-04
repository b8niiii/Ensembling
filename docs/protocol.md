# Experimental protocol

The four official studies share the task, selected evidence and scoring rules below. The [result inventory](../results/README.md) links exact settings, identities, reports and audits for each study.

## Task and data splits

A bag groups sentences for one ordered `(head.id, tail.id)` pair. Each sentence provides the original text and zero-based, half-open character spans `[start, end)` for both mentions. Predictions are sets of 24 positive NYT10m relation IDs; `NA` is the empty set. Different sentences may support different relations for the same pair.

| Reference | Role | Interpretation |
|---|---|---|
| NYT10m train/validation distant labels | Development; full validation has 36,266 bags | Noisy database-derived supervision |
| Reviewed 30 validation bags | Separate development diagnostic: C01–C12 and N01–N18 | Selected and repeatedly used text annotations; not a representative benchmark |
| Expanded human-confirmed development | Random100 and previously selected42 | Design evidence for the late extension. Report the two groups separately. |
| Official manually annotated test | Four separately frozen studies on 5,174 bags | Official labels with the same selected evidence across studies |

Each study fixed its conditions before evaluating them on the official test. The later studies reused the benchmark and retained the preceding test exposure. Dataset and toolkit references are [Gao et al. (2021)](https://aclanthology.org/2021.findings-acl.112/) and [OpenNRE](https://github.com/thunlp/OpenNRE).

## Evidence selection and manual-test reconstruction

All systems receive the same selected evidence. The cap is 20 units per bag; longer bags use deterministic, evenly spaced indices including the first and last. Selection does not use labels or predictions.

Historical validation uses original record order and retains duplicate texts. The manual test instead serializes multi-label sentences as repeated `relation` rows. Before cap selection, rows are merged by **exact text, ordered entity IDs and both mention spans**; different mentions in the same text remain distinct. Units retain first-occurrence order. The reference is the union of positive manual labels attached to the selected units only.

The verified test contains 11,086 rows, 9,744 distinct sentence/mention units and 5,174 bags. Selection retains 9,001 units and shortens 37 bags. Selected and full-bag references both contain 3,899 positive facts, with no changed bag reference in this release. That equality is an observed audit result, not a general property of capped evaluation. See the [data reconstruction audit](../results/test_data_protocol_audit.json).

## Supplied-entity encoders and voting

GLiNER-relex base, GLiNER-relex large and GLiDRE use the supplied directed mentions. Entity recognition and predicted-name matching are bypassed. Word boundaries are refined at the supplied character offsets when necessary, preserving the original text. The head remains index 0 and the tail index 1 even when the tail occurs first in the sentence.

All 24 directed scores are retained per selected sentence. A bag's score for a relation is the maximum of its sentence scores. A relation is accepted only when this maximum is **strictly greater than** the member's cutoff. Scores are not assumed to be calibrated probabilities.

| Primary test system | Frozen rule |
|---|---|
| Individual large | Large score > 0.7 |
| Three-member majority | At least two votes per label: base > 0.5, large > 0.7, GLiDRE > 0.7 |

Base and GLiDRE scores and timings are required member artifacts, rather than separate primary test leaderboard rows. Validation also compares individual cutoffs and pairwise union/intersection rules. Ensemble cost includes every required member.

Workers run sequentially on CPU with offline checkpoint loading. Preflights check supplied-span alignment and direction, complete relation scores, native-layer parity and the longest selected input. GLiDRE uses pinned author code and an isolated GLiNER 0.2.13 helper; the recorded compatibility bridge reconnects existing pretrained components without changing their weights. Cache reuse requires the matching fingerprint and complete valid results. The hash-bound inference sources retain their original paths and bytes.

## API contract and failures

The test uses the recorded `deepseek-flash` alias, advertised as DeepSeek-V4.1-Flash at execution, with thinking enabled at `low`, JSON-object output and a 4,096-token total completion cap. The shared v2 development prompt supplies the ordered mentions, original sentences/spans and all 24 relation descriptions. Its positive JSON example illustrates syntax only. No reference answers, predictions, tools or in-context demonstrations enter the request.

The required output is `{"relations": ["allowed relation ID", ...]}`; `{"relations": []}` is valid abstention. Malformed JSON, repeated object keys, unknown labels, wrong types, empty content and non-stop/truncated responses are invalid. Repeated known label strings are deduplicated and counted. Invalid model responses remain empty predictions for positive-label scoring, with invalid counts/rates reported separately; they are neither dropped nor repaired through new generations. Transport failures and uncertain attempts are separate request states, retained in durable journals.

The two development pilots are historical conditions: v1 compared non-thinking/high with its original prompt; v2 compared non-thinking/low/high with the shared format instruction. Their different prompts prevent a reasoning-only interpretation of cross-version differences. The API alias and backend metadata identify the observed service, rather than cryptographically pinning remote weights. Tariffs in saved reports describe the recorded experiment.

## Reviewed annotation policy

The development convention `text_supported_inference_v2_20261006` accepts direct textual evidence and documented motivated inference for head → tail. An inferred positive must record its sentence evidence, connecting assumption and uncertainty. Unsupported external facts, mere co-occurrence and distant labels alone do not establish a positive relation. The official test labels remain unchanged by this convention.

The original Reviewed30 diagnostic consists of 12 C bags and 18 N bags. N18's Andrea Barzagli → Italy nationality annotation was confirmed from the national-team reading, with uncertainty documented. Historical B08's Iran → Tehran containment revision is outside this active cohort. This is a project annotation convention, not a universal inference rule. [Canonical annotation files and revision evidence](../support/README.md) preserve the labels, samples, rationales and prior versions.

## Metrics, uncertainty and costs

Precision, recall and micro-F1 aggregate positive `(ordered entity pair, relation)` TP/FP/FN. Empty-reference bags remain included because unsupported positive predictions contribute false positives. Invalid responses receive no success credit, even when their empty prediction matches a negative reference.

Each test analysis uses 20,000 paired whole-bag bootstrap replicates with the seed and contrasts recorded in its specification. Every system receives the same sampled bag indices; each bag's label set stays together and micro-F1 is recomputed from summed counts. Nominal 95% percentile intervals describe sampled-bag uncertainty conditional on the saved predictions and references. They do not measure repeated generation, annotation or model-selection uncertainty; approximate bag exchangeability is assumed.

Local CPU time includes required inference attempts and session label encoding, excluding loading/preflight. API time includes service/network and request attempts, excluding idle gaps/backoff. Reconstructed token charges and unknown-charge reserves remain separate. The released-model comparison records no peak runtime memory. Later studies report sampled individual process RSS, model loading and deployment storage separately. The ensemble benchmark times are sums of individual means, with joint memory and fusion overhead unmeasured.

## Numerical reduction

The ONNX study compares dynamic INT8, symmetric block-128 INT4 weight-only, and mixed FP16 Linear conversion against matched ONNX FP32. The conversion targets 144 encoder Linear matrices. The extraction heads and supplied-span adapter remain in PyTorch. Dynamic INT8 derives activation scales per input, INT4 reduces constant weights, and mixed FP16 retains selected operations in FP32.

The declared test comparisons center on INT8 and the equal mean of the three reduced bag scores. Both use strict >0.7. INT4 and FP16 are retained as individual controls. The within-ONNX comparison shares relation-description order. The historical PyTorch and ONNX controls use different description order, so their observed difference combines backend and prompt-serialization changes. See the [frozen reduction specification](../results/onnx_reduction_test_spec.json).

## LoRA adaptation

Last6 adapts Q/V projections in encoder blocks 18–23. Random6 uses one fixed draw of six blocks. Both use rank 8, alpha 8, dropout 0.05 and AdamW at 5e-5. The parent parameters and task head are frozen. Each placement completed three epochs on the same 5,000 train-only bags. Checkpoints were selected by distant-validation micro-F1 at strict >0.7, then by precision and earlier epoch.

The [LoRA test](../results/lora_test_spec_20261019.json) evaluates Last6 epoch 2 against a fresh matched original-large control. Random6 epoch 3, union and intersection are declared supporting systems. The API comparator reuses the original outputs. One random placement draw limits comparisons between placement strategies.

## Heuristic fusion

The expanded development packet contains the earlier selected42 and a human-confirmed Random100. Its pooled result is descriptive because these examples informed the design. The completed extension pairs Last6 with GLiClass v4 Edge or ModernBERT-NLI. The [frozen specification](../results/heuristic_ensembles_test_spec_20261022.json) carries all six primary variants and eight controls to the previously used official benchmark.

Each pair has global score weighting, specialist-label addition and specialist-label replacement. Global weights use normalized positive Jaccard on the inspected manual142 packet. The label-specific variants use the selected capital, birthplace and deathplace labels. Union and equal-score means are controls. Every pair runs both members on all bags and the base NLI cutoff of 0.7 is retained alongside the adopted 0.9 sensitivity condition. No rule, coefficient or cutoff was tuned on these official test outcomes.

## DeepSeek-relative F1

For matched test evidence, report 100 × F1(system) / F1(saved DeepSeek low) alongside absolute F1. The index expresses a measured ratio and can exceed 100 if a system exceeds that reference. Development cohorts use their own absolute metrics.
