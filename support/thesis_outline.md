# Thesis chapter outline for Overleaf

Prepared 5 October 2026. This is a local writing outline, not an uploaded Overleaf project. The agreed research question remains unchanged. Initial source-verified keys are in [references.bib](references.bib); verify remaining digest citations before use.

## 1. Introduction

- Motivation: news relation extraction under local memory and compute constraints.
- Agreed question: when does combining small models improve quality enough to justify extra computation, relative to individuals and an API LLM?
- Proposed contributions: supplied-entity evaluation, fair static-ensemble comparisons, controlled model reductions, and measured quality–cost trade-offs. Describe these as proposed until experiments support them.
- Scope: bag-level closed-ontology extraction, under/around-1B models, released checkpoints, deferred ICL/new fine-tuning.

## 2. Task, data, and evaluation

- NYT10m and distant supervision; manual test protocols (`gao-etal-2021-manual`).
- OpenNRE distribution/toolkit (`han-etal-2019-opennre`).
- Ordered entities, character spans, positive-label sets, NA, multi-label bags, repeated records, cap 20.
- Split roles, existing reviewed-30 diagnostics; inactive audit-20 preparation history, reference differences, and final frozen evaluation.

## 3. Related work

- Direct benchmark and news-RE precedents, distinguished from document-level extraction.
- Released encoder specialists and GLiNER-Relex architecture (`stepanov2026glinerrelex`).
- SLM ensembles and quality–cost evidence: digest entries require complete verified bibliography.
- Mixed-precision voting precedent (`lu2025mixedprecision`), plus quantization/pruning methods actually used.
- ICL and routing/cascades as deferred adjacent evidence; do not imply they were evaluated.

## 4. Methodology

- Exact supplied spans, direction-preserving adapter, word-boundary handling, and native-layer verification.
- Deterministic sentence selection; checkpoint/code/input hashes and resumable offline scoring.
- Individual thresholds; union/intersection and any later frozen fusion rules.
- Reduction feasibility, parent/variant provenance, mixed precision, pruning and recovery cost.
- API baseline contract, complete member/call cost, latency and memory methodology.

## 5. Experiments and results

- Historical screening, with contract failures distinct from semantic performance.
- Completed full-validation experiment; distant and reviewed evidence separated.
- Fair individual/member-cutoff comparison and deeper error analysis on existing reviewed30.
- First frozen released-SLM/static-ensemble/API comparison, followed by a separately declared reduction study chosen on training/validation.
- Final frozen manual-test comparison and uncertainty; this result is currently absent.

## 6. Discussion and conclusion

- Conditions under which an ensemble gain justifies cost, including cases where one member is preferable.
- Annotation noise, purposeful review sampling, training-provenance limits, correlated variant errors, runtime specificity, and cap sensitivity.
- Deferred extensions and research directions.

## Writing and sharing checklist

- [ ] Transfer this outline and the bibliography into Overleaf.
- [ ] Draft chapters 1–3 first; share readable chapters incrementally with the supervisor.
- [ ] Add verified full citations for all retained papers, preprints, model/runtime artifacts, and implemented reduction methods.
- [ ] Keep completed results separate from planned claims and absent API/test results.
- [ ] Review repository publication, then commit/push and share its URL explicitly.
