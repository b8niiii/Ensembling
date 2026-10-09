# NYT10m development protocol

Updated 8 October 2026. The experiment uses supplied entities and fixed released encoder checkpoints. The first frozen official-test comparison and all required runtime preflights are complete. Historical validation contracts remain below for interpreting earlier results; publication defaults disable model/API execution.

## Input and label semantics

A bag contains records for the split-local ordered `(h.id, t.id)` pair. Each record supplies text and head/tail names, IDs, and half-open character spans `[start, end)`: `text[start:end]` selects the mention. Relation prediction is directed head → tail. There are 24 positive NYT10m IDs; `NA` is represented by the empty predicted set.

Historical validation uses all records in original order when bag size is at most 20, otherwise the frozen first/last-inclusive evenly spaced 20-index selection. It keeps duplicate texts and saves full size, selected indices, texts and supplied spans. Those runs remain unchanged. The verified manual-test serialization is reconstructed before selecting 20 distinct sentence/mention units, as specified below. Neither selection uses reference labels or predictions. Every compared system receives the same selected evidence; model-native encodings may differ.

| Cap | Validation bags shortened | Selected records | Omitted records |
|---|---:|---:|---:|
| 10 | 179 | 43,584 | 2,838 |
| 20, current | 75 | 44,751 | 1,671 |
| 40 | 25 | 45,593 | 829 |
| Full | 0 | 46,422 | 0 |

The cap is an input rule, not dataset deletion or an established optimum. Any cap/selection change is a new controlled development condition. Evaluate affected long bags and runtime fit before freezing the final policy.

## Supplied-entity encoder execution

The active runner supplies both exact mention spans to GLiNER-relex's pretrained relation layers. NER, entity classification scores, entity thresholds, and predicted-name matching are bypassed. Word boundaries are split at supplied character offsets if necessary; the original sentence is preserved. Direction remains head index 0 → tail index 1, including when the head occurs later in the text.

Base runs before large, sequentially on CPU with offline checkpoint loading. Checkpoint repositories/revisions and weight hashes are fixed in the runner. Preflight verifies native relation-layer parity on a synthetic example, absence of entity logits/selection calls, boundary alignment, and word limits on every selected record. The successful preflight must match the frozen fingerprint before full inference.

Save all 24 positive scores per selected sentence, bag-wise maxima, selected source-record indices, status, timing, and fingerprint. Accept a relation iff its bag maximum is **strictly greater than** the relation cutoff; 0.5 is the recorded primary condition, not a universal calibration standard. Fixed 0.7/0.9 tables reuse the score cache. A model score has not been established as a calibrated probability.

The current run is `relex_full_validation_given_entities_v1_20261004`, fingerprint `c9a68bf3bb8d0af2830fabbaa7948bb165077e4bea69e90b7233a744f4743793`. Its runner, adapter, and legacy-named relation-description module are immutable dependencies. Modifying them requires a new version rather than bypassing hash checks. Atomic writes and complete inventory validation control resume. Three consecutive failed bags or less than 2 GiB free disk space stop inference. Incomplete experiments are not scored as completed runs.

## Normalization, scoring, and ensembles

Every system yields a unique set of positive IDs for the supplied ordered pair. Precision, recall, and micro-F1 aggregate positive label TP/FP/FN across bags. Empty predictions are valid abstentions. Preserve native scores; do not force encoder outputs through a chat-JSON parser.

Intersection accepts labels selected by both base and large; union accepts labels selected by either. Both require the complete cost of both model calls. The diagnostic grid compares six individual settings and eighteen ensemble settings, including all nine per-member combinations from {0.5, 0.7, 0.9}, reusing unchanged cached scores. Compare ensembles with competitive individually selected cutoffs, not just the original 0.5 example. Annotation revisions produce a new grid version; original aggregates remain available for comparison.

Full-validation labels are distant supervision. Report distant-NA, positive, and size cohorts separately; do not interpret missing distant labels as definitive textual negatives. The reviewed 30 use separately confirmed text labels and remain an error-analysis diagnostic. Do not pool their score with distant labels or claim a representative accuracy estimate. NYT10m train-membership cohorts do not identify these checkpoints' actual training exposure and are excluded from the active report. Training inventories remain incomplete.

## Annotation conventions

The working text-annotation rule accepts direct evidence and motivated inference from the supplied sentences for head → tail. Each inferred positive must document the sentence evidence, the connecting assumption, and uncertainty. A fact recalled from external knowledge remains distinct when the supplied text provides no supporting argument. Co-occurrence or a distant label alone is insufficient. Locative wording such as “Helsinki, Finland” supports containment in Finland → Helsinki direction but does not alone express capital status. Explicit capital wording uses the confirmed capital/containment convention recorded in the reviewed samples. `/people/person/children` runs parent → child; it is not symmetric. A location → company query is not automatically location containment under this ontology.

The [literature-based reassessment](validation_annotation_reassessment_2026-10-06.md) records the policy and all 42 previously reviewed bags. On 6 October, the user confirmed two pragmatic inferences: historical B08 Iran → Tehran contains, without capital; and N18 Andrea Barzagli → Italy nationality, reading the sentence as national-team representation. These are documented annotation conventions with uncertainty, rather than universal deduction rules. Only N18 belongs to the active reviewed30. The current annotation version is `text_supported_inference_v2_20261006`; prior annotations and aggregates remain archived. The active reporter loads the public C/N annotation files through `source/reviewed_validation_reference.py`, checking their version, sample identity, and hashes; experiment-local references retain their historical labels.

Review all selected sentences, use `[]` when no positive relation is supported under this policy, and record uncertainty. The complete allowed inventory is:

- `/business/company/advisors`
- `/business/company/founders`
- `/business/company/majorshareholders`
- `/business/company/place_founded`
- `/business/location`
- `/business/person/company`
- `/film/film/featured_film_locations`
- `/location/administrative_division/country`
- `/location/country/administrative_divisions`
- `/location/country/capital`
- `/location/location/contains`
- `/location/neighborhood/neighborhood_of`
- `/location/region/capital`
- `/location/us_county/county_seat`
- `/people/deceasedperson/place_of_burial`
- `/people/deceasedperson/place_of_death`
- `/people/ethnicity/geographic_distribution`
- `/people/person/children`
- `/people/person/ethnicity`
- `/people/person/nationality`
- `/people/person/place_lived`
- `/people/person/place_of_birth`
- `/people/person/religion`
- `/time/event/locations`

The replacement12 and confirmed additional18 supply the active reviewed30; original versions are preserved in the [annotation archive](archive/annotations_before_inference_revision_2026-10-06/). Current error analysis reads selected texts and current confirmed labels, compares supplied-entity predictions across cutoffs/fusion, and classifies evidence, inference, direction, and distant-label mismatches. No further manual annotation is scheduled. These selected, repeatedly used development bags are not an independent representative benchmark.

The previously prepared audit20 is unannotated and **inactive** under the 6 October revision. Its inputs and private sampling provenance remain as preparation history, not a planned reference or scoring cohort. No labels are fabricated or inferred from predictions to activate it.

Official test labels must come from the verified manual release, without reassignment or relation closure. The primary capped-evidence reference uses only the manual labels of selected sentence/mention units. Full-bag unions remain separate provenance; the two references coincide for this actual release. Earlier audits remain in the [historical snapshot](../results/preliminary_experiments.json). The 7 October format/aggregation verification is documented in notes.md; no test-model inference or test-result-driven method selection was performed.

## Reduction study and cost

Reduction is a later phase, after the first frozen released-SLM/static-ensemble versus API-LLM comparison. It remains part of the thesis scope. The prepared reduction pilot is 100 uniformly sampled validation bags, seed 20261006, with the same cap/selected sentences and a large parent checkpoint. First verify one working CPU-compatible reduction against the original on identical inputs. Record parent hash, preparation/runtime versions, reduction configuration, artifact bytes, supported operations, failures, quality, measured latency, and peak memory. Keep the original checkpoint and frozen environment intact.

Quantization changes precision rather than parameter count. Distinguish unstructured zero masks, executable sparse methods, and structural removal. Any recovery training uses permitted training inputs, with compute recorded separately. Calibration/data partitions must be named explicitly; do not confuse them with parameter partitions. Reduced members must be evaluated individually before static uniform/mixed-precision ensembles are compared. Preparation cost and full deployment inference cost remain distinct.

## API validation: completed v1 and three-mode v2

[DeepSeek_flash_validation.ipynb](../source/DeepSeek_flash_validation.ipynb) uses the official DeepSeek Chat Completions endpoint and `deepseek-flash`, currently advertised as V4.1 Flash. The completed v1 pilot compared disabled/none with enabled/high. Prepared protocol `deepseek_three_mode_validation_v2_20261007` compares disabled/none, enabled/low and enabled/high with independent flags. All use the same ordered pair, original sentence text/spans, all 24 positive descriptions, JSON-object output and 4,096-token total completion cap. V2 adds the same positive JSON syntax example and explicit flat-string-array requirement to every mode. This is an output-format illustration without an example sentence or reference answer. Sampling parameters are omitted because thinking ignores temperature control; deterministic equivalence is not asserted. No reference answers, model predictions, tools or ICL demonstrations are included.

The default 130-bag pilot consists of the existing reviewed30 and fixed random100. References stay separate, current confirmed annotation hashes are reread, and missing results make a cohort incomplete rather than removing difficult examples. Known repeated IDs are deduplicated/count-reported; malformed JSON, repeated object keys, unknown labels, wrong types, empty content and non-stop/truncated responses are invalid and scored as empty positive sets. Output repair/regeneration is disabled. Completed invalid responses remain cached. Local supplied-entity encoder scores are comparison inputs, with strict > cutoffs and complete member time.

Paid calls require explicit activation and a local API key. The notebook defaults to a $2 model-usage budget, sequential rotating mode order, optional off-peak submission/waiting (disabled by default), atomic result writes and an attempt journal. Inference fingerprints include prompt/settings/input/implementation, excluding reference labels and sample scope. The enabled-mode inventory is part of the fingerprint. V2 uses a separate cache directory and does not reuse v1 responses under the changed prompt: the default 130-bag comparison needs up to 390 new requests before transport retries. Subsequent identical v2 runs resume. All attempts count toward reconstructed charges or an unknown-charge reserve. Reasoning is included in completion tokens, not added twice. Explicit transient HTTP errors have bounded retries; unknown transport/interruption states block automatic resubmission. Explicit account-reviewed recovery retains the unknown reserve. API request time includes network/service latency and is not the same measurement as local CPU inference.

The optional full-validation scope reuses identical same-version pilot responses and requires an explicitly larger budget; it is not scheduled. No official-test execution is implemented. Model aliases/backend fingerprints and token/caching/price-period metadata are retained; the advertised API release is not a cryptographic weight pin. Both pilots are complete: v1 has 260 calls and v2 has 390. Read-only v2 analysis verified every response/journal, request identities, current references, API/cohort metric rows and token charges. Low under the shared v2 prompt and 4,096-token completion cap is the selected first-test API configuration. Reports compare all mode pairs, count truncation/empty finals, and score large and the existing majority candidates from caches on matched bags. A historical v1/v2 table requires matching inputs/annotation hashes and labels prompt changes explicitly; it is not a controlled reasoning-only comparison. Detailed findings and limitations are in notes.md.

## Historical generative contract and deferred ICL

Earlier GGUF/llama.cpp experiments requested exactly `{"relations": ["positive ID", ...]}` and no surrounding text; `{"relations": []}` denotes abstention. Unknown IDs, malformed/truncated JSON, repeated object keys, wrong types, and explicit NA outputs were invalid. The original 150-bag strict condition invalidated duplicate relation strings; the later development condition deduplicated otherwise valid known labels and recorded duplicate counts. Invalid answers were scored as empty positive sets with invalid rate reported, not silently repaired. Transport failures were distinct from model-format failures. Historical scores are not recomputed under a different parser.

ICL is deferred. If resumed, demonstrations must come only from training and be fixed across compatible conditions. The three-bag rich/simple/NA suggestion and earlier four-example alternative are unselected proposals. Encoder-informed decoder context and new fine-tuning are separate future conditions, not retroactive changes to existing experiments.

## Publication and final freezing

The completed [Full_validation_relex.ipynb](../source/Full_validation_relex.ipynb) defaults to `RUN_RELEX_BASE=False`, `RUN_RELEX_LARGE=False`, `RUN_GLIDRE=False`, asset prefetch disabled and `ANALYZE_LOCAL_SCORES=False`. Run All displays the compact published report matching current annotations without local caches or model loading. Explicit local analysis can rescore complete caches; separately enabled model reruns remain resumable. The test notebooks likewise default to published summaries; `ANALYZE_LOCAL_RESULTS=True` requests local verification, while the separate execution flag enables inference or paid requests. The public repository includes code, notebook outputs, documentation, annotations, and compact aggregate provenance; datasets, weights, detailed results, environment, secrets, and unrelated output files stay local.

The GLiDRE development branch retains exact h→t spans, all selected source sentences, all 24 positive relation scores and bag maxima. Pinned author code uses isolated GLiNER 0.2.13 helpers without changing the original runtime. Its 512-word preprocessing guard is raised to 2048 with tokenizer truncation disabled; pretrained positional layers and weights are unchanged. Runtime checks verify strict loading, sub-1B parameter count, native-score/decoder and cached-label parity, all selected inputs and the longest actual input before full inference. This compatibility preflight and full validation completed; the test repeats the checks on its own selected inputs. Relation descriptions use deterministic uppercase/underscore formatting; output relation IDs are unchanged. The fixed three-model diagnostic grid has 90 configurations: nine singles, 54 pairwise union/intersection settings and 27 label-wise two-of-three votes. Each member uses its own strict cutoff; raw confidence scores are not averaged. Cost includes every required member, with GLiDRE session label encoding/failed attempts counted and loading/preflight reported separately. Annotation edits affect versioned reports, not cache fingerprints.

The inspected author implementation has inconsistent tuple unpacking between the bi-encoder, word-attention extractor and relation head. A fingerprinted local bridge reconnects those existing functions and returns the word-attention tensor required by the released ATLOP head. It does not replace parameterized layers or introduce a new RNN invocation. Native mention preparation, cached/uncached label paths, multi-label decoding and long-text execution passed on a tiny synthetic architecture with random weights, followed by successful released-checkpoint validation checks. Parity remains a mandatory runtime gate for the test; this does not assert that the uncorrected public path executes successfully.

Before the first official test-model comparison, complete the execution manifest for the selected large, static ensemble and API LLM, including every ensemble member's checkpoint, label descriptions, input selection, thresholds, aggregation, output/failure rules and API configuration. Count all member calls, attempts and actual API usage; report loading/preparation/training costs separately from inference where relevant. Standalone base/GLiDRE test leaderboard rows are not planned; their scores, predictions and timings remain required fusion/audit artifacts.

## Staged test evaluation

The implemented comparison is designated `nyt10m_selected_evidence_test_v1_20261007` (superseding the earlier specification draft): large @ 0.7, label-wise two-of-three majority with base @ 0.5 / large @ 0.7 / GLiDRE @ 0.7, and DeepSeek v2 thinking-low. No full-validation API run is scheduled. Supplied directed mentions, strict > encoder cutoffs, sentence-score maxima, exact v2 prompt/parser and 4,096-token completion cap are retained. Large0.7 and the chosen majority are compromises across distant validation and reviewed30, not configurations that independently maximize both references.

The hash-verified official manual test stores labels in `relation`, not `anno_relation_list`. Multiple rows with the same ordered IDs, exact text and both mention spans encode a multilabel sentence. Merge these labels before label-blind, first-occurrence, evenly spaced selection of at most 20 units. The primary reference is the union of positive manual labels of those selected units only; `NA` is empty, repeated labels count once and head→tail is preserved. Different mentions in the same text remain distinct. No facts are looked up in a database or added from omitted sentences. Reconstructing the 11,086 rows yields 9,744 units and 5,174 bags; selection supplies 9,001 units and shortens37 bags. Full and selected unions both contain3,899 facts, with no changed bag reference. The generic OpenNRE `anno_relation_list` branch is not assumed to describe the downloaded file.

Invalid model responses remain failures with separate invalid count/rate. Positive-label P/R/micro-F1 maps them to empty predictions without dropping bags or inventing FP labels; on negative references this alone gives no TP/FP/FN penalty. Any prospective exact-match/success measure must additionally require a valid response, so invalid negatives are not credited as successful abstentions. Historical pilot scores remain unchanged. Transport failures are unresolved requests rather than model-format errors. No retry-strategy experiment is planned; retain existing v2 handling (at most two retries for explicit HTTP 429/502/503/504, review before resubmitting uncertain-charge attempts), resume completed caches and preserve attempt records/actual usage. Test completion requires every bag to be accounted for.

Report three primary systems with precision, recall, micro-F1, total/per-bag wall time and invalid counts. Ensemble time includes all members even when standalone large reuses its inference. API timing includes network/service latency and is labelled separately from local CPU time; token usage and monetary spend remain recorded. [test_comparison_spec.json](../results/test_comparison_spec.json) and [test_data_protocol_audit.json](../results/test_data_protocol_audit.json) record the verified specification and reconstruction. Final scoring uses fixed declared cutoffs; OpenNRE's additional test-best-threshold F1 search is not used. [SLM_test.ipynb](../source/SLM_test.ipynb) and [DeepSeek_test.ipynb](../source/DeepSeek_test.ipynb) call shared external modules and default to inference disabled. Runtime checkpoint/native-parity/all-input preflights remain mandatory before local test inference; actual API/backend usage is recorded during the user-run experiment.

The first test comparison is a released-model baseline study. It completed on all 5,174 bags under the frozen specification, with micro-F1 0.5228 for large, 0.5010 for majority and 0.6736 for DeepSeek low. All input, scorer, cache and accounting checks matched; see [notes.md](notes.md) and the [combined report](../results/test_comparison_summary.json). Later quantization/pruning decisions must use training/validation evidence, not that first test's errors or rankings. Version and predeclare later systems before their test evaluation and disclose the sequential use of the same benchmark. Keeping later choices on validation reduces feedback leakage; it does not make a previously evaluated test newly untouched.

The 8 October cleanup retains all validation/test scores and paid-response journals, both relex checkpoints, original datasets and the frozen environment. GLiDRE's weight blob was removed from the local cache; pinned metadata, tokenizers, runtime receipt and extracted helpers remain so cached reports still verify. A new GLiDRE inference requires explicitly fetching that pinned weight again before offline execution. No experiment settings or frozen inference source were changed by cleanup.
