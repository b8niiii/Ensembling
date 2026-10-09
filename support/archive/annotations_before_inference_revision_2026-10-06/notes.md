# Thesis working notes

Updated 6 October 2026. This file records the current scope, protocol, evidence, and open decisions. Pilot tables and settings are consolidated in [Preliminary_experiments.ipynb](../../../source/Preliminary_experiments.ipynb); paper summaries remain in [literature.md](../../literature.md).

## 1. Research question and scope

**When does combining small language models improve news relation extraction enough to justify the extra computation, compared with individual small models and a large model accessed through an API?**

The agreed question is retained. The current study focuses on released, primarily encoder-based relation extractors under or around one billion parameters, static ensembles, and feasible reductions of a leading checkpoint. An API LLM remains an external quality–cost comparator; it is not the only relevant baseline. Every ensemble must also be compared with its strongest individual member under competitive development-selected settings.

The 1 October supervision discussion prioritised quantization/pruning and ensembling of reduced variants. On 6 October, the execution order was revised: first establish a released-SLM/static-ensemble versus API-LLM comparison; reduction follows as a separately declared phase. GLiNER-relex large is the **provisional parent checkpoint** for that branch. Quantization changes numerical precision and may reduce memory; it does not reduce the number of parameters. Pruning requires an implementation that actually reduces executable work or storage before an efficiency gain can be claimed. No random-forest analogy establishes error diversity or savings automatically. Five variants and “different splits” remain hypotheses whose construction must be specified, not requirements for the first feasibility test.

ICL, an encoder-informed decoder, routing/cascades, and new task fine-tuning are deferred extensions. A two-member system is acceptable; an additional vote is useful only if its errors and cost improve the measured frontier.

## 2. Dataset and EDA

NYT10m is a news **bag-level** relation extraction benchmark, distributed through [OpenNRE](https://github.com/thunlp/OpenNRE#data) and described by [Gao et al.](https://aclanthology.org/2021.findings-acl.112/). A record contains sentence text, head and tail entity names/IDs/half-open character spans, and a relation label. A bag groups records with the same **ordered** `(h.id, t.id)` within one split. A record count and a bag count describe different units.

| Split | Sentence records | Ordered-pair bags | Role |
|---|---:|---:|---|
| Train | 417,893 | 255,526 | Optional future adaptation or train-only demonstrations/calibration inputs |
| Validation | 46,422 | 36,266 | Development, error analysis, thresholds, ensemble and reduction decisions |
| Official test | 11,086 | Not recomputed during cleanup | Final evaluation after the method is frozen |

Train and validation use noisy distant supervision, not confirmed sentence-level gold. The official test has manual relation annotations (`anno_relation_list`); its evaluation must follow the benchmark's defined label aggregation. `NA` means no positive dataset label, not proof that no relation is expressed. Predictions use a set of the 24 positive relation IDs; an empty set is abstention. Bags can support multiple distinct positive relations. EDA keeps repeated records and distinguishes record labels, combined bag labels, and distinct sentence texts.

The provisional cap is 20 sentence records per bag. In validation it shortens **75/36,266 bags (0.21%)**, supplies **44,751/46,422 records**, and omits 1,671 records from model inputs. Selection is deterministic, evenly spaced in original record order, and includes the first/last records. It uses neither labels nor model scores. The dataset itself is unchanged. A paired cap sensitivity check remains open for the affected long bags.

No official test-model evaluation has been run. Limited earlier input-overlap and aggregate label-convention audits are retained as provenance in the historical evidence snapshot; they were not model-selection runs. The official test remains reserved for the final frozen comparison.

## 3. Active supplied-entity protocol

The completed experiment uses local **GLiNER-relex base and large**, sequential CPU inference, the same supplied sentence records, and the dataset's exact head → tail character spans. Entity recognition, entity confidence thresholds, and predicted-name filtering are bypassed. The adapter retains the pretrained span/relation layers and refines model word boundaries at supplied entity offsets when needed; sentence text is unchanged. Offline checks passed for both checkpoints on all 44,751 selected records; 320 records required boundary refinement.

Every selected sentence stores scores for all 24 directed positive relations. A bag score is the maximum over its selected sentences. A relation is accepted when its score is **strictly greater than** the cutoff. The primary recorded cutoff is 0.5; cached 0.7/0.9 diagnostics do not rerun inference or freeze the final cutoff. Scores are model confidences, not established calibrated probabilities.

The current run contains **72,532 valid bag results**, with fingerprint `c9a68bf3bb8d0af2830fabbaa7948bb165077e4bea69e90b7233a744f4743793`. Dataset, source, checkpoint, settings, and selected-input hashes remain tied to that run. `run_extended_gliner_relex.py` remains unchanged because it provides the frozen relation descriptions imported by the active workflow. Its historical filename does not make it disposable.

The active notebook is [Full_validation_relex.ipynb](../../../source/Full_validation_relex.ipynb). Default Run All displays the published [compact summary](../../../results/full_validation_summary.json), without download or model execution. Detailed local scores permit further cutoff and ensemble analysis. NYT10m train-seen/unseen cohorts are omitted because membership in that benchmark split does not establish the released checkpoints' actual training exposure. Model cards lack a full checkpoint-specific training inventory; the framework paper describes FineWeb-derived synthetic training but does not certify absence of overlap with NYT10m.

Precision = TP/(TP+FP); recall = TP/(TP+FN); micro-F1 = 2TP/(2TP+FP+FN), pooling positive label decisions across bags. True negatives do not contribute to positive-label F1. The scorer is deterministic evaluation code, not a model. A five-case constructed scorer check was previously passed; the retained adapter tests additionally cover direction, boundaries, multiple relations, and strict cutoff semantics.

## 4. Current evidence and its limits

### Primary cutoff 0.5

| System | Distant P | Distant R | Distant micro-F1 | Reviewed-30 micro-F1 | Total member inference seconds |
|---|---:|---:|---:|---:|---:|
| Base | 0.2319 | 0.5258 | 0.3219 | 0.5600 | 3,157.34 |
| Large | 0.2241 | 0.6642 | 0.3351 | 0.7213 | 10,078.57 |
| Intersection | 0.3503 | 0.4444 | 0.3918 | 0.6829 | 13,235.91 |
| Union | 0.1882 | 0.7457 | 0.3005 | 0.6286 | 13,235.91 |

The reviewed 30 are a selected diagnostic sample (12 older replacement bags plus 18 confirmed additions), with 23 positive label instances. They are neither independent of development choices nor a representative population estimate. Their separate text reference remains useful because distant labels can disagree with explicit evidence.

Intersection reduces distant-negative errors at 0.5 but loses recall. Union adds recall and many false positives. On reviewed-30, large alone has higher F1 than either combined system. Deploying a combination costs both inferences: 13,235.91 seconds here, approximately **31% more than large alone**. These recorded seconds exclude checkpoint loading and do not replace repeated controlled latency/memory measurements.

### Fair cutoff comparison

| Cutoff | Base distant F1 | Large distant F1 | Intersection distant F1 | Union distant F1 | Large reviewed-30 F1 |
|---|---:|---:|---:|---:|---:|
| 0.5 | 0.3219 | 0.3351 | 0.3918 | 0.3005 | 0.7213 |
| 0.7 | 0.2625 | 0.3964 | 0.2550 | 0.3892 | 0.7170 |
| 0.9 | 0.0292 | 0.4079 | 0.0276 | 0.4069 | 0.6486 |

An ensemble advantage cannot be claimed from intersection at 0.5 versus large at 0.5 alone: large at 0.7 or 0.9 is a competitive cheaper alternative against distant labels. Higher cutoffs sharply reduce base recall and change the precision/recall balance of large. Distant F1 and reviewed-text F1 favor different settings; no final choice follows automatically from either noisy labels or a small selected review sample.

At 0.5, base/large disagree on **8,206 bags**. Among 32,392 distant-NA bags, positive predictions occur in 4,873 bags for base, 5,601 for large, 2,564 for intersection, and 7,611 for union. These are mismatches against distant labels. The separate analysis of the existing confirmed 30 explains examples but cannot establish how all 8,206 disagreements should be labeled.

### Mixed-member cutoff grid — 6 October 2026

Section8 was executed successfully on the unchanged 72,532 saved results. Six individual settings and eighteen union/intersection settings were evaluated using strict score > member cutoff, with separate references. Cache integrity/completeness and parity with the existing scorer passed. No new inference or test access occurred; aggregates are saved in [validation_mixed_cutoff_grid_v1.json](../../../results/validation_mixed_cutoff_grid_v1.json).

| System | Base cutoff | Large cutoff | Distant micro-F1 | Reviewed-30 micro-F1 |
|---|---:|---:|---:|---:|
| Base | 0.5 | — | 0.3219 | 0.5600 |
| Large | — | 0.5 | 0.3351 | 0.7213 |
| Large | — | 0.7 | 0.3964 | 0.7170 |
| Large | — | 0.9 | 0.4079 | 0.6486 |
| Union | 0.9 | 0.9 | 0.4069 | 0.6486 |
| Intersection | 0.5 | 0.7 | 0.4056 | 0.6667 |
| Intersection | 0.5 | 0.5 | 0.3918 | 0.6829 |

No ensemble exceeds the strongest individual micro-F1 within either reference across this grid. Union at 0.9/0.9 has the highest distant ensemble F1 but remains below large at 0.9; union at base0.9/large0.5 ties large0.5 on reviewed-30 counts at greater cost. Mixed intersection at base0.5/large0.7 achieves reviewed precision 0.9231 but recall 0.5217 (12 TP, 1 FP, 11 FN). Large0.7 has 19 TP, 11 FP, 4 FN: filtering removes errors and supported relations together.

Recorded full-validation member inference totals are 52.62 minutes for base, 167.98 for large, and 220.60 for either ensemble (**31.33% extra versus large**), excluding loading and analysis. These are reused timings, not new benchmarks. Large0.9 maximizes distant F1 while large0.5 maximizes reviewed F1; large0.7 is an intermediate precision/recall setting. Base0.9 nearly eliminates recall (0.0153 against distant labels). The current grid does not establish an ensemble F1 gain that justifies its extra inference, nor an independent final-test conclusion. Reviewed-30 findings follow below; no final cutoff or fusion rule is frozen.

### Reviewed-30 error analysis — 6 October 2026

Section9 completed successfully: 60 saved model results were checked against the frozen inputs, sentence maxima, and six completed-grid metric rows. All 35 supplied sentences were inspected against unchanged confirmed annotations. No new model inference or test access occurred. Of 30 bags, 22 have at least one FP/FN in one of the six configurations; 13 have different confirmed and distant label sets.

| Configuration | TP | FP | FN | Micro-F1 | Exact-match bags |
|---|---:|---:|---:|---:|---:|
| Base0.5 | 14 | 13 | 9 | 0.5600 | 12 |
| Large0.5 | 22 | 16 | 1 | 0.7213 | 17 |
| Large0.7 | 19 | 11 | 4 | 0.7170 | 18 |
| Large0.9 | 12 | 2 | 11 | 0.6486 | 18 |
| Intersection base0.5/large0.5 | 14 | 4 | 9 | 0.6829 | 17 |
| Intersection base0.5/large0.7 | 12 | 1 | 11 | 0.6667 | 18 |

At 0.5, base contributes **zero correct labels absent from large**, and nine exclusive false positives. Large contributes eight exclusive correct labels and twelve exclusive false positives. Intersection removes those twelve errors but also all eight correct labels. This is filtering, without a correct-label recovery benefit, on this selected sample. With large0.7, base does contribute two correct labels missed at that stricter cutoff (N01 country-capital and N05 place-of-death); complementarity is therefore conditional on thresholds, not absent in every setting. These counts alone do not validate another fusion rule.

Observed errors include related-ontology confusion (capital versus administrative division/region capital; birth/burial versus residence/death), unsupported relations inferred from context, and incompatible entity-role assignments. Examples: N09 correctly predicts company-founder but also company-place-founded for the person Barry Schwartz (large score 0.9698); N18 predicts nationality from a national-team sentence (0.9119) without explicit nationality evidence under the confirmed policy. Neither is removed by cutoff0.9. In N05, death-in-Mumbai scores 0.6870 while unsupported residence scores 0.8334; cutoff0.7 discards the supported relation and retains the error. C11 explicitly states death in Canterbury, but both scores fall below0.5 (base0.4777, large0.4966). C07 explicitly states son-of; large scores the ordered parent→child relation0.9955 while base scores0.2519. These observations do not establish the models' internal causes or a general direction-error rate.

Large0.5 retains 22/23 annotated positive instances; large0.7 removes five FP at the cost of three TP. Large0.9 and mixed intersection retain only12/23. Exact-match bag counts alone conceal this recall loss. Large remains the provisional leading individual checkpoint; 0.5 versus0.7 and the predeclared ensemble comparator remain open decisions. The selected, repeatedly used sample supports error diagnosis, not independent population estimates or an exclusively reviewed-30 threshold search. An API-LLM validation workflow is the next planned implementation.

## 5. Historical model decisions

Generative pilots showed that valid JSON does not imply correct relations; many outputs overpredicted positives or failed the format contract. Qwen's strict-parser duplicate failures and later deduplication are distinct protocol conditions. Their settings and numbers remain in the self-contained historical notebook.

The 1,012-bag development comparison contained 500 random, 500 enriched, and 12 reviewed bags, with separate references. The four-encoder reviewed-30 screen and the subsequent GLiFormer screen are complete. GLiFormer's sparse exact-pair output and an exploratory broader location-tail mapping did not establish a cost-effective third vote over relex large. Its checkpoint cache is removed; the [pair-matching audit](../../gliformer_pair_matching_audit_2026-10-01.md) remains as a compact record. No further GLiFormer run is pending in the main plan.

GenTune is a generative Qwen LoRA adapter, not an encoder. It is excluded from the main roster because its native output contract did not reliably map to the fixed NYT10m ontology and published NYT11 inputs overlap development data. Corpus overlap does not prove that every matching row was used by the released checkpoint's capped training subset. NYT-trained checkpoints generally require a credible training/split audit and compatible ontology before a clean comparison; specialization itself is not disqualifying. GenTune retraining/mapping and new task fine-tuning are outside the current scope.

Historical detailed results and other model caches were removed after aggregate tables, settings, revisions, and weight hashes were preserved. Data, both active checkpoints, all current scores, confirmed annotations, and small reporter dependencies remain local. The unrelated Excel workbook in `outputs/` is retained outside Git. See the [cleanup inventory](../../cleanup_2026-10-05.md).

## 6. Literature and writing

[evidence_matrix.md](../../evidence_matrix.md) retains the compact six-paper comparison; [literature.md](../../literature.md) retains paper-by-paper summaries. Gao et al. and OpenNRE are direct benchmark/toolkit references. RE adaptation and encoder papers are task/architecture evidence. General SLM ensembles, mixed precision, ICL, and cascades provide adjacent methodological evidence; they do not establish gains for this bag-level task.

The verified initial [BibTeX file](../../references.bib) covers NYT10m/manual evaluation, OpenNRE, GLiNER-Relex, and the mixed-precision precedent. Remaining digest references must receive checked complete entries before thesis citation. [thesis_outline.md](../../thesis_outline.md) is a local outline for transfer to Overleaf; no external sharing or Overleaf edit was performed during cleanup.

## 7. Updated work order — agreed 6 October

**Analysis status:** section8 of [Full_validation_relex.ipynb](../../../source/Full_validation_relex.ipynb) completed the six individual/eighteen ensemble grid and saved its aggregate results. Runtime integrity/completeness/scorer-parity checks passed. Execution was explicitly enabled locally; no new inference occurred. Findings are recorded in section4; final method selection remains open.

Section9 completed reviewed-30 error analysis after explicit local activation. The runtime checks passed and its six-configuration tables and expandable text/score reports were inspected. Confirmed annotations and original score caches remain unchanged; no result files were written by the section. Findings are recorded in section4. Final method selection and the API-LLM comparison remain pending.

1. **Fair SLM comparison from caches.** Compare base/large at 0.5/0.7/0.9 and intersection/union under all nine per-member cutoff combinations. This is a new versioned analysis of unchanged scores, not new inference. Report separate distant-label/text-reviewed metrics, negative-bag mismatches, and summed member cost; compare fusion with competitive individual cutoffs.
2. **Existing reviewed-30 error analysis.** Read all supplied sentences, confirmed labels, and current supplied-entity scores/predictions. Classify unsupported positives, missed explicit relations, direction errors, annotation-source mismatches, and unique contributions to fusion. Do not treat these repeatedly used, selected bags as independent gold for a population estimate or optimize a global threshold solely on them. No additional manual annotation is scheduled. The prepared [audit-20](../../validation_audit_20_review_2026-10-06.md) remains unannotated, inactive, and excluded from reference scoring.
3. **Verify an API LLM on validation.** Select and verify one model/version and its price/budget. Prepare the same directed-pair/evidence/24-ID task with the fixed JSON/empty-set contract, without demonstrations or reference answers in inputs. Use the reviewed30 for setup/error checks; if a broader cost/setup pilot is needed, reuse the fixed random100 validation inputs already prepared. Fix prompt, parsing, decoding, retries, and logging on development data before test use.
4. **Freeze the first comparison.** Record models, individual/member cutoffs, one static ensemble, supplied-input selection, label mapping, API configuration, scorer/manual-test aggregation, failure handling, and cost measures. Keep base, large, the selected ensemble, and the LLM in the predeclared comparison so individual–ensemble added value remains observable.
5. **Run and interpret the official test.** Evaluate the frozen systems on identical test evidence; reuse member predictions for fusion. Report quality and complete model/API costs, including negative or inconclusive ensemble results. Do not use test errors or rankings to revise prompts, cutoffs, fusion, or subsequent reduction designs.
6. **Continue reduction and writing as separate work.** Quantization/pruning remain a later branch. The fixed random100/cap20 preparation is retained; no reduced checkpoint exists. Design and select variants on training/validation, separately freeze later comparisons, and document that the official test has already been used. Repository sharing, bibliography completion, and Overleaf writing can progress in parallel.
