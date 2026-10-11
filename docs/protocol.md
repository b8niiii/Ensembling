# Experimental protocol

The completed comparison is `nyt10m_selected_evidence_test_v1_20261007`. Its exact settings, checkpoint revisions, weight identities, prompt identity, input/reference hashes and nine inference-source hashes are in the [frozen specification](../results/test_comparison_spec.json). Published outcomes are in the [results report](results.md).

## Task and data splits

A bag groups sentences for one ordered `(head.id, tail.id)` pair. Each sentence provides the original text and zero-based, half-open character spans `[start, end)` for both mentions. Predictions are sets of 24 positive NYT10m relation IDs; `NA` is the empty set. Different sentences may support different relations for the same pair.

| Reference | Role | Interpretation |
|---|---|---|
| NYT10m train/validation distant labels | Development; full validation has 36,266 bags | Noisy database-derived supervision |
| Reviewed 30 validation bags | Separate development diagnostic: C01–C12 and N01–N18 | Selected and repeatedly used text annotations; not a representative benchmark |
| Official manually annotated test | Frozen final comparison: 5,174 bags | Official labels, without reassignment or added relation closure |

Checkpoint, cutoff, fusion, prompt, completion cap and scoring choices were frozen before test inference. Test errors and rankings do not drive revisions of this comparison or the later reduction design. Dataset and toolkit references are [Gao et al. (2021)](https://aclanthology.org/2021.findings-acl.112/) and [OpenNRE](https://github.com/thunlp/OpenNRE).

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

The active diagnostic consists of 12 C bags and 18 N bags. N18's Andrea Barzagli → Italy nationality annotation was confirmed from the national-team reading, with uncertainty documented. Historical B08's Iran → Tehran containment revision is outside this active cohort. This is a project annotation convention, not a universal inference rule. [Canonical annotation files and revision evidence](../support/README.md) preserve the labels, samples, rationales and prior versions.

## Metrics, uncertainty and costs

Precision, recall and micro-F1 aggregate positive `(ordered entity pair, relation)` TP/FP/FN. Empty-reference bags remain included because unsupported positive predictions contribute false positives. Invalid responses receive no success credit, even when their empty prediction matches a negative reference.

The statistical analysis uses 20,000 paired whole-bag bootstrap replicates, seed `20261008`. Every system receives the same sampled bag indices; each bag's label set stays together and micro-F1 is recomputed from summed counts. Nominal 95% percentile intervals describe sampled-bag uncertainty conditional on the saved predictions and references. They do not measure repeated generation, annotation or model-selection uncertainty; approximate bag exchangeability is assumed.

Local CPU time includes required inference attempts and session label encoding, excluding loading/preflight. API time includes service/network and request attempts, excluding idle gaps/backoff. Reconstructed token charges and unknown-charge reserves remain separate. Peak runtime memory and local monetary/energy costs were not measured for the primary comparison. Artifact size is not a substitute for runtime memory.

## Remaining experimental work

The primary next experiment is **ONNX Runtime quantization of GLiNER-relex large**, investigating the requested **Dynamic Weight and Activation Quantization at 4, 8 and 16 bits**. The fixed 100-bag/118-record validation pilot has prepared inputs, not reduced-model results. Preserve the original checkpoint/environment and version any new export wrapper separately from hash-bound baseline sources.

Validate FP32 export of the full directed supplied-entity relation path, all 24 scores, variable lengths and longest inputs. Initially target eligible encoder linear operations; record converted/excluded nodes and exact weight, activation and accumulator formats. **ONNX introduces a new backend:** remeasure PyTorch FP32 and ONNX FP32 under identical inputs, preprocessing, batching, threads and timing boundaries, then compare ONNX reduced variants with ONNX FP32.

The [official ORT documentation](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) distinguishes dynamic INT8 and blockwise INT4 weight-only routes; [FP16 conversion](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html) is a separate floating-point method with provider limitations. They do not establish a uniform dynamic W4A4/W8A8/W16A16 sweep or GLiNER compatibility. Execute supported levels and record unsupported activation formats; never relabel a weight-only or FP16 artifact as dynamic weight-and-activation quantization.

Report P/R/micro-F1, changes from the matched parent, controlled repeated latency, peak process memory, executable artifact bytes and one-off preparation cost. No predefined acceptable F1 loss or composite quality–cost score is imposed. Evaluate each variant alone before ensembles, counting every member's inference. Select and freeze on validation before later test use, and disclose that the benchmark was already evaluated once.

**PEFT on limited modules and differentiated NYT10m training subsets is an exploratory extension.** Study compatible methods, explicit target modules, a directed relation loss and deterministic train-subset manifests before a small training run. Compare adapted and untuned controls; record trainable parameters, subset size/coverage, noisy-label assumptions, hyperparameters, stopping/selection rules and training cost. Train/calibrate only on declared training inputs; validation selects/stops, while reviewed/test annotations are not training data. Isolate adaptation before combining it with quantization, verify any adapter merge/export and evaluate each member individually.

Pruning remains optional later work; dense masks do not establish smaller executable dimensions or faster inference. ICL, routing, extra decoders and new fine-tuning were not evaluated in the frozen comparison. ONNX export, reduced-model inference and PEFT training remain pending.

## DeepSeek-relative F1

The agreed empirical upper quality reference is DeepSeek thinking-low, normalized to **100**. On identical bags, selected evidence, reference and scoring, report `relative F1 = 100 × micro-F1(system) / micro-F1(DeepSeek)` alongside absolute precision, recall and micro-F1.

The [completed test report](../results/test_comparison_summary.json) supplies denominator **0.6735735271377764**, yielding large **77.62**, majority **74.38** and DeepSeek **100.00**. The denominator is the saved unrounded value. A relative score of 100 means matching this measured comparator, not perfect extraction; scores above 100 are allowed. Validation normalization requires matched validation DeepSeek outputs, never the test denominator. Use the test reference for reporting after the later system is frozen, not as selection feedback.
