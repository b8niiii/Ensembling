# NYT10m development protocol

Updated 6 October 2026. The active experiment uses supplied entities and frozen released encoder checkpoints. Historical generative contracts are retained below for interpreting earlier results. Final test settings remain to be frozen on development evidence.

## Input and label semantics

A bag contains records for the split-local ordered `(h.id, t.id)` pair. Each record supplies text and head/tail names, IDs, and half-open character spans `[start, end)`: `text[start:end]` selects the mention. Relation prediction is directed head → tail. There are 24 positive NYT10m IDs; `NA` is represented by the empty predicted set.

Use all records in original order when bag size is at most 20. Otherwise use the frozen first/last-inclusive evenly spaced 20-index selection. Keep duplicate texts and save full size, selected indices, texts, and supplied spans. Do not use reference labels or predictions to select sentences. Every compared system receives the same selected evidence; model-native encodings may differ.

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

Intersection accepts labels selected by both base and large; union accepts labels selected by either. Both require the complete cost of both model calls. The same cutoff is used by both members in the existing diagnostic tables. The updated work order adds a separately versioned validation-only analysis of the nine per-member combinations from {0.5, 0.7, 0.9}, reusing unchanged cached scores. No mixed-cutoff result is currently claimed. Compare ensembles with competitive individually selected cutoffs, not just the original 0.5 example.

Full-validation labels are distant supervision. Report distant-NA, positive, and size cohorts separately; do not interpret missing distant labels as definitive textual negatives. The reviewed 30 use separately confirmed text labels and remain an error-analysis diagnostic. Do not pool their score with distant labels or claim a representative accuracy estimate. NYT10m train-membership cohorts do not identify these checkpoints' actual training exposure and are excluded from the active report. Training inventories remain incomplete.

## Annotation conventions

The working text-annotation rule requires an explicit relation in the supplied sentences for head → tail. Co-occurrence, distant labels, or unstated world knowledge are insufficient. Locative wording such as “Helsinki, Finland” supports containment in Finland → Helsinki direction but does not alone express capital status. Explicit capital wording uses the confirmed capital/containment convention recorded in the reviewed samples. `/people/person/children` runs parent → child; it is not symmetric. A location → company query is not automatically location containment under this ontology.

Review all selected sentences, use `[]` when no positive relation is expressed, and record uncertainty. The complete allowed inventory is:

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

The original replacement12 and confirmed additional18 are preserved as annotation/provenance files. Current error analysis reuses these reviewed30: read selected texts and confirmed labels, compare supplied-entity predictions across cutoffs/fusion, and classify explicit-evidence, direction, and distant-label mismatches. No further manual annotation is scheduled. These selected, repeatedly used development bags are not an independent representative benchmark.

The previously prepared audit20 is unannotated and **inactive** under the 6 October revision. Its inputs and private sampling provenance remain as preparation history, not a planned reference or scoring cohort. No labels are fabricated or inferred from predictions to activate it.

Official test scoring must use the released manual annotations and defined aggregation unchanged. Earlier input-only overlap and aggregate direction/label-convention audits are preserved in the [historical snapshot](../../../results/preliminary_experiments.json); no model inference on official test was performed. No new test-label inspection or test-model evaluation is part of cleanup or audit preparation.

## Reduction study and cost

Reduction is a later phase, after the first frozen released-SLM/static-ensemble versus API-LLM comparison. It remains part of the thesis scope. The prepared reduction pilot is 100 uniformly sampled validation bags, seed 20261006, with the same cap/selected sentences and a large parent checkpoint. First verify one working CPU-compatible reduction against the original on identical inputs. Record parent hash, preparation/runtime versions, reduction configuration, artifact bytes, supported operations, failures, quality, measured latency, and peak memory. Keep the original checkpoint and frozen environment intact.

Quantization changes precision rather than parameter count. Distinguish unstructured zero masks, executable sparse methods, and structural removal. Any recovery training uses permitted training inputs, with compute recorded separately. Calibration/data partitions must be named explicitly; do not confuse them with parameter partitions. Reduced members must be evaluated individually before static uniform/mixed-precision ensembles are compared. Preparation cost and full deployment inference cost remain distinct.

## Historical generative contract and deferred ICL

Earlier GGUF/llama.cpp experiments requested exactly `{"relations": ["positive ID", ...]}` and no surrounding text; `{"relations": []}` denotes abstention. Unknown IDs, malformed/truncated JSON, repeated object keys, wrong types, and explicit NA outputs were invalid. The original 150-bag strict condition invalidated duplicate relation strings; the later development condition deduplicated otherwise valid known labels and recorded duplicate counts. Invalid answers were scored as empty positive sets with invalid rate reported, not silently repaired. Transport failures were distinct from model-format failures. Historical scores are not recomputed under a different parser.

ICL is deferred. If resumed, demonstrations must come only from training and be fixed across compatible conditions. The three-bag rich/simple/NA suggestion and earlier four-example alternative are unselected proposals. Encoder-informed decoder context and new fine-tuning are separate future conditions, not retroactive changes to existing experiments.

## Publication and final freezing

Default [Full_validation_relex.ipynb](../../../source/Full_validation_relex.ipynb) Run All displays a compact published report. Local inference, checkpoint downloads, and local report regeneration require separate explicit switches. The public repository includes code, self-contained notebook outputs, documentation, annotations, and compact aggregate provenance; datasets, weights, detailed results, environment, secrets, and unrelated output files stay local.

Before the first official test-model comparison, verify the API workflow on validation and freeze base, large, one selected static ensemble, the API LLM, label descriptions, input selection, thresholds, aggregation, output/failure rules, and API configuration on development evidence. Count all member calls, retries, and actual API usage; report loading/preparation/training costs separately from inference where relevant.

## Staged test evaluation

The first test comparison is a released-model baseline study. Later quantization/pruning decisions must use training/validation evidence, not that first test's errors or rankings. Version and predeclare later systems before their test evaluation and disclose the sequential use of the same benchmark. Keeping later choices on validation reduces feedback leakage; it does not make a previously evaluated test newly untouched.
