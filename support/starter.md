# Thesis roadmap

Updated 8 October 2026. Detailed decisions are in [notes.md](notes.md), the active contract in [protocol.md](protocol.md), and historical results in [Preliminary_experiments.ipynb](../source/Preliminary_experiments.ipynb).

**Research question:** When does combining small language models improve news relation extraction enough to justify the extra computation, compared with individual small models and a large model accessed through an API?

## 1. Scope, data, and evaluation reference

- [x] Retain the agreed quality–cost research question and sub-1B, encoder-first focus.
- [x] Record the 1 October supervision decisions: static ensembling, feasible quantization/pruning of a leading model, Overleaf writing; defer ICL and new fine-tuning.
- [x] Download NYT10m and inspect train/validation schema, directed bags, NA, multi-label relations, repeated records, lengths, and provenance in EDA.
- [x] Establish train/validation/test roles and an explicit sentence-evidence criterion for working annotations.
- [x] Retain cap 20 with deterministic label-independent selection; measure 75 shortened validation bags and 44,751 supplied records.
- [x] Confirm 12 replacement and 18 additional validation annotations, yielding the reviewed-30 diagnostic reference.
- [x] Verify the manual-test description and aggregation from the paper, official release bytes and loader; document selected-only references and the distinction between label rows and unique sentence/mention units.
- [ ] If needed before final freezing, perform a paired cap sensitivity check on long validation bags, including retained/omitted evidence and full cost.

**Challenge:** distant supervision can miss textual relations; reviewed development samples are selected diagnostics, not population estimates. The first frozen test evaluation is complete; preserve its settings and disclose any later use of the same benchmark.

## 2. Reproducible implementation and model screening

- [x] Set up the local Python/Jupyter environment and pinned model artifacts.
- [x] Complete generative pilots, specialist screening, GenTune contract checks, and the 1,012-bag comparison.
- [x] Complete four-encoder reviewed-30 validation and **run GLiFormer large**, compare it with the other members, and audit directed pair matching.
- [x] Exclude GenTune from the main roster; retain corpus-overlap and output-contract evidence.
- [x] Implement supplied-entity GLiNER-relex inference, bypass NER, and verify native relation parity, directed scores, and mention boundaries.
- [x] Complete base/large inference on all 36,266 validation bags; retain 72,532 complete score caches.
- [x] Regenerate the analysis-only report without NYT10m train-membership cohorts; keep reviewed-30 separate.
- [x] Consolidate eight historical experiments, clean obsolete files/caches, retain active artifacts, and make default Run All report-only.
- [ ] Commit and push the reviewed public file set; share the repository/notebook with the supervisor.

**Challenge:** preserve the frozen runner/adapter/relation-description hashes; changes require a new run version. Report-only execution must work without local weights or detailed data.

## 3. Completed baseline comparison: released SLMs versus an API LLM

The agreed order was revised on 6 October: compare competitive SLM settings, examine the existing reviewed 30, validate the API LLM setup, freeze the first comparison, then evaluate on the official test. No additional manual annotation is scheduled. Reduction remains a later research phase.

### 3.1 Fair cached SLM comparison

- [x] Produce existing equal-cutoff tables at 0.5/0.7/0.9 for base, large, intersection, and union.
- [x] Write optional section8 in Full_validation_relex.ipynb for the nine base/large cutoff combinations and both fusion rules; no analysis execution or frozen-source changes.
- [x] Execute section8 with RUN_CACHED_GRID_ANALYSIS=True and verify the runtime cache/completeness/scorer-parity checks; save the compact aggregate grid.
- [x] Compare each combination with competitive individual-model cutoffs on identical supplied inputs. Report P/R/micro-F1, distant-NA mismatches, and summed member inference cost; keep distant and reviewed-text references separate. The original two-member grid establishes no F1 gain over the strongest individual; the completed three-member assessment is recorded below.
- [x] Select large0.7 and majority base0.5/large0.7/GLiDRE0.7 for the first test comparison alongside DeepSeek v2 low. Retain base/GLiDRE test outputs for fusion/audit; use their validation evidence rather than separate test leaderboard rows. The shared test specification is recorded in section3.4.

**Deliverable:** a compact comparison table with member cutoffs, fusion rule, separate references, and full inference cost. Selection uses validation only; the distant-label and selected reviewed-text evidence must not be conflated.

### 3.2 Error analysis on the existing reviewed 30

- [x] Prepare an additional blinded 20-bag packet; retained as inactive preparation history, not an annotation task or reference.
- [x] Write and execute optional section9 for the confirmed 30: six fixed configurations, per-bag errors, member contributions, and expandable text/score reports; runtime consistency and grid parity passed.
- [x] Examine all 35 supplied sentences, confirmed labels, and current supplied-entity scores/predictions for the existing reviewed 30.
- [x] Compare errors across selected cutoffs and rules; record unsupported predictions, missed explicit relations, ontology/entity-role confusion, and confirmed-versus-distant mismatches. Do not infer a direction-error rate from aggregate FP/FN counts.
- [x] Record which correct/incorrect labels each member uniquely contributes and which errors fusion removes or introduces; findings are in notes.md.
- [ ] Summarize implications for the chosen settings without optimizing a general threshold solely on these 30 selected development bags. Preserve the confirmed annotation reference; any justified correction must be documented separately.

**Deliverable:** a short error-analysis table and interpretation using existing annotations. No new 20-bag selection or annotation is required. The reviewed 30 are diagnostic, not an independent representative estimate.

### 3.2a Additional GLiDRE comparison — completed

Before closing candidate selection, the requested [additional sub-1B audit](model_candidate_audit_2026-10-06.md) identified GLiDRE. Its complete validation run and 90-configuration analysis passed, retaining the original relex results. Majority improves reviewed-30 F1, but its highest distant F1 is almost tied with competitive large at 91% extra inference time. This does not freeze member cutoffs or replace the API comparison.

- [x] Inspect released sub-1B alternatives using primary sources; record parameter counts, training evidence, task compatibility, revisions and limitations. GLiDRE is the first new RE candidate; one entailment encoder is a second option.
- [x] Add isolated, pinned GLiDRE loading/scoring, initial asset prefetch, independent per-model flags, resumable outputs, fresh confirmed-annotation scoring and the three-model fixed grid. Validate imports, the sub-1B meta architecture, notebook syntax, 15 synthetic/stub tests and attention-routing/native-path parity on a tiny synthetic architecture; preserve original relex caches and source hashes. No pretrained checkpoint inference was executed.
- [x] Run Full_validation_relex.ipynb locally: strict weight loading, native/cached-label parity, all supplied boundaries and the longest-input check passed before full inference on 36,266 bags.
- [x] Inspect current reviewed-30 errors and full distant-label comparisons separately; record GLiDRE's exclusive correct/incorrect predictions, pairwise fusion, majority and summed cost against competitive individual settings in notes.md. All 108,798 caches and 180 metric rows were verified. Other candidates remain inactive.

### 3.3 API LLM selection and validation

- [x] Select DeepSeek V4.1 Flash (`deepseek-flash`) for a paired development pilot; verify public weights, advertised API release, tariffs and input-based cost estimates. Actual account access and served-version behavior remain runtime checks.
- [x] Prepare [DeepSeek_flash_validation.ipynb](../source/DeepSeek_flash_validation.ipynb) with all helpers inside the notebook: supplied pair/spans, unchanged cap20 evidence, 24 IDs, text-supported evidence/inference rule, no demonstrations, JSON/empty-set contract, paired non-thinking/thinking-high settings, off-peak guard, budget, attempt accounting and resumable outputs. Notebook syntax/schema and 12 offline fixture checks passed; no API or notebook execution occurred.
- [x] Run and analyse the original two-mode pilot: 130 bags/mode, 260 calls. Thinking-high leads under strict scoring; non-thinking format errors and high-effort truncation remain material limitations. Confirmed answers remain excluded from requests.
- [x] Use the fixed random100 for measured tokens/time, reconstructed charges and a full-validation projection; compare saved encoder scores without inference. Keep distant scores separate from reviewed30.
- [x] Prepare v2 in the same notebook: non-thinking/low/high flags, a shared positive JSON syntax example, separate cache/fingerprint, all pairwise diagnostics, explicit truncation counts and cached three-member majority comparisons. Schema/syntax and 18 offline fixture checks passed; no real API calls occurred during preparation.
- [x] Execute v2 on the same 130 bags: 390 calls, complete references, no HTTP failures or unknown charges. Recompute metrics and cost accounting from responses/journals. Low has the highest reviewed F1 (0.8936), fewer truncations and lower latency/projected token cost than high; record the single-distant-prediction random100 difference and small-sample limitations in notes.md. Preserve v1/v2 results.
- [x] Fix prompt, parser, retry/failure policy, model/API settings, and cost logging using validation only. Include every attempt's tokens, spend, and latency.

**Deliverable:** a verified API inference/scoring workflow and a validation-based setup/cost report. A higher-capacity model is a comparator; superior task performance is not assumed.

### 3.4 Freeze and run the first official-test comparison

- [x] Select DeepSeek v2 thinking-low, large0.7 and majority base0.5/large0.7/GLiDRE0.7; skip full-validation API inference (decision, 7 October).
- [x] Verify the official manual-release bytes and `relation` serialization; reconstruct multilabel sentences and ordered bags, select at most20 distinct sentence/mention units without label dependence, and use only their manual labels as reference. Verify full-fact parity and synthetic scorer/parser/cache/accounting cases.
- [x] Record `nyt10m_selected_evidence_test_v1_20261007` with checkpoints, cutoffs, fusion, label mapping, input/reference and source hashes, exact API prompt, runtime and accounting rules. Prepare shared modules plus SLM_test.ipynb and DeepSeek_test.ipynb, with execution disabled.
- [x] Run SLM_test.ipynb; complete mandatory offline asset/native-parity/all-selected-input preflight and resumable base/large/GLiDRE inference on all 5,174 bags.
- [x] Run DeepSeek_test.ipynb; complete low requests on the same test inputs under the explicit budget, without off-peak waiting. Preserve two acknowledged interrupted-attempt records and 78 invalid completions.
- [x] Evaluate large0.7, the selected majority ensemble and DeepSeek v2 low on identical official test evidence. Retain every member's scores/timings for fusion and audit; omit separate base/GLiDRE test leaderboard rows.
- [x] Independently verify and report P/R/micro-F1, invalid rate, per-relation errors, total/per-bag time and full member cost. Retain API tokens/reconstructed spend and distinguish local CPU time from API network/service latency.
- [x] Interpret the frozen comparison: F1 large 0.5228, majority 0.5010 and DeepSeek low 0.6736. Majority improves precision but loses recall and F1 at 1.94× local time. Do not use test outcomes to revise thresholds, prompts, fusion or reduction designs.

- [x] Complete the saved-result statistical analysis: 20,000 paired whole-bag bootstrap replicates; quality–latency and precision/recall plots; relation-level voting diagnostics; separate API/local-time/memory reporting. Preserve Test_results_analysis.ipynb, compact counts, numerical report and PDF/SVG/PNG figures for thesis drafting. No new predictions or configuration changes.

**Deliverable:** the first frozen individual–ensemble–LLM test comparison, including an ensemble that does not win if that is the observed outcome. The test serves evaluation, not the next development round.

## 4. Later phase: reduction and ensembles of variants

The reduction branch is deferred until after the first released-model/API comparison. It remains part of the agreed thesis direction, not a prerequisite for that first test run.

- [x] Prepare a uniform random 100-bag validation pilot, seed 20261006, cap 20, and record the source fingerprint and provisional large parent checkpoint.
- [ ] Verify one CPU-compatible quantization path without changing the current frozen environment or original checkpoint.
- [ ] Compare that reduced variant with original large on identical inputs: directed score validity, text/distant quality separately, time, peak memory, and real artifact size.
- [ ] Audit practical pruning options: weight masks, sparse execution, structured removal, implementation/parity constraints, and any train-only recovery compute.
- [ ] If feasibility passes, choose controlled precision/pruning levels, record each parent/variant hash and preparation settings, and evaluate each member individually.
- [ ] Compare uniform-precision and mixed-precision static ensembles, plus feasible pruned variants, with the best individual alternative and complete cost.
- [ ] Define what “different splits” means before using it: parameter masks, layer allocation, or data/calibration partitions must not be conflated.

**Challenge:** design and select any later reductions on training/validation evidence, not first-test errors or rankings. Record the sequence of test evaluations; the test cannot be treated as newly untouched after the first comparison. Quantization need not accelerate this CPU/runtime; masked zeros do not automatically reduce dense compute; variants of one checkpoint may have highly correlated errors. Do not construct five variants before one reduction works.

## 5. Final evaluation, writing, and publication

- [x] Preserve paper summaries/evidence matrix and separate direct NYT10m precedents from adjacent ensemble, reduction, ICL, and routing evidence.
- [x] Prepare an initial source-verified BibTeX set and chapter outline for Overleaf.
- [ ] Verify and add complete bibliography entries for the remaining cited papers, including preprints and technical reports.
- [ ] Draft introduction, task/dataset, related work, and methods in Overleaf; share chapters incrementally with the supervisor.
- [x] Freeze, version and report the first large/majority/API comparison under section3.4.
- [ ] Freeze later reduction conditions separately before any subsequent test evaluation; document repeated use of the same official test and avoid test-driven method changes.
- [x] Clean redundant downloads/reports and Python caches, preserve scientific artifacts, disable default model/API execution, and prepare compact public reports and a publication checker.
- [ ] Prepare final reproducibility instructions, limitations, charts, and thesis conclusions; check the 31 October submission requirements.

**Challenge:** distinguish exploratory validation from frozen test evidence, and measured gains from hypotheses. Completed API and official-test results are documented; the reduction branch is still unexecuted.

## 6. Deferred extensions

- [ ] If the main comparison is complete, reconsider ICL using only training-split demonstrations; evaluate three example bags (rich positive, simple positive, NA) against the earlier four-example proposal before freezing.
- [ ] Optionally supply base/large outputs as context to a small decoder; report the decoder and both encoder calls in cost.
- [ ] Consider routing/cascades only after static-system evidence; include selector cost and every invoked model.
- [ ] Consider new train-only task adaptation with noise-aware objectives; report training/recovery compute separately and preserve untuned baselines.
- [ ] Reopen discarded candidates only with a new justified compatibility/provenance protocol.
