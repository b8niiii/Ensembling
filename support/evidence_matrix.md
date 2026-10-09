# Evidence matrix for Background and Related Work

**Checked:** 6 October 2026. Read with [the paper analyses and chapter outline](literature.md) and [references.bib](references.bib). P01–P13 are the original corpus; P14–P22 are the agreed additions. S1/S2 are brief AWQ/GPTQ supplements inside P10, not additional deep-review cards.

This matrix compares evidence, not pooled performance. **D** = direct NYT10m; **R** = other RE; **A** = other tasks; **T** = theory, survey or position. Cost labels mean **M** author-measured, **E** estimated/analytical, **N** absent for the specified dimension. Published scores concern the source protocol; local adaptations remain separately evaluable. N never means zero cost.

## Task, scale, adaptation, and mechanism

| ID / citation key | Evidence and evaluation unit | Input/output contract | Scale and adaptation | Mechanism / diversity |
|---|---|---|---|---|
| P01 / `mojarradi2024ensemble` | A; sentence/pair classification, medical-subject questions | Class label, not RE/NA sets | 25M–770M tuned members plus Llama3-8B; medical transfer from MNLI | Predictions/confidence supplied in context; different architectures |
| P02 / `cho2025cosmosfl` | A; Defects4J bug/repository | Ranked methods, acc@k counts | 7B–12B, 4-bit; tool/prompt adaptation and weight optimization | Static vote over members/repeated runs; heterogeneous checkpoints |
| P03 / `li2024biomedical` | R; PubMed abstracts/concept pairs/novelty | NER then relation/novelty classification; modified JAC | Domain encoders plus T0pp-11B; target tuning, ten folds, extra annotation/rules | Majority over contexts/folds/backbones; separate novelty model |
| P04 / `li2024multiagent` | T; heterogeneous agent workflows | No common RE contract | Diverse LLMs/tools/roles; no controlled scale comparison | Interaction, memory, planning, feedback; roles can reuse one model |
| P05 / `folino2025defense` | A; full-document fake-news classification | Binary probability; no relation ontology | Compact four-layer Small BERT; supervised/incremental self-training | Weighted score fusion of sequential snapshots |
| P06 / `chan2026smc` | T+A; structured generated strings | JSON/SQL/word sorting; expected correctness | 7B–14B; no new tuning; prompt/model pairs | Global distributional ensemble, SMC/byte alignment; differs from hard voting |
| P07 / `zhang2025refiner` | R; DocRED/Re-DocRED document–pair decisions | Given entities, directed multilabel relations, NA threshold | Trained encoders plus Llama3-8B/API, no new LLM tuning | Pair-level selective refinement, score addition, second verification |
| P08 / `chanthran-etal-2026-document` | R; MEN/DocRED/Re-DocRED unseen-label DocRE | Entity side information; highest-score label; NA/multilabel underspecified | BERT-base plus GPT-4o-mini side-information calls | Feature similarity fusion; not checkpoint ensemble |
| P09 / `tang2026collaboration` | R; classical Chinese joint triples | Predicted entities/directed triples; micro/macro metrics | BERT-guwen/SpERT tuning plus proprietary LLMs; Alpaca2-7B control | Context guidance; union/filter/relation-type replacement variants |
| P10 / `lu2025mixedprecision` | A; trustworthiness/classification | Mapped answers and task-specific refusal filtering | Llama2-13B at 3/4/8 bits; dense controls | Whole-checkpoint precision voting; same backbone |
| P11 / `gao-etal-2021-manual` | D; NYT10m sentences/bags/facts; also Wiki20 | Given entities, directed multilabel, empty-set NA | CNN/PCNN/BERT task training under DS | Sentence ONE/AVG/ATT inside a model; not multi-checkpoint ensemble |
| P12 / `han-etal-2019-opennre` | R; original NYT10/sentence/few-shot tasks | Usually given entities; sentence/bag/document distinctions | CNN/BERT-family trained baselines | Toolkit; attention/adversarial/RL modules |
| P13 / `christou2026subbillion` | R; nine processed pair-classification datasets; NYT11 | Given pair, one generated label; invalid labels penalized | 362M/494M/~3B decoders; QLoRA/domain mixtures; RoBERTa controls | Specialization/matched demonstrations; no ensemble |
| P14 / `stepanov2026glinerrelex` | R; joint CoNLL04/DocRED/FewRel/CrossRE | Predicted spans, ordered pairs, independent labels | DeBERTa-v3; synthetic multitask supervision; large recipe detailed | Shared label/text encoder and pair heads; local bags change contract |
| P15 / `armingaud2025glidre` | R; Re-DocRED/FREDo/ReFREDo | Gold mentions/coreference, ordered pairs, multiple labels | ~800M total dual encoders; synthetic pretraining, supervised/support tuning | Dual encoding/localized context; no ensemble |
| P16 / `zafrir2019q8bert` | A; GLUE/SQuAD development | Classifier/QA task metrics | BERT-base/large; 8-bit QAT versus dynamic quantization | Quantization training intervention |
| P17 / `wei2022outlier` | A; GLUE/SQuAD/summarization | Encoder/encoder–decoder metrics | BERT/RoBERTa/BART; PTQ/QAT/optional KD | Gamma Migration/Token-Wise Clipping; explicit tensor bits |
| P18 / `sanh2020movement` | A; SQuAD/MNLI/QQP | Task-tuned encoder predictions | BERT-base; encoder sparsity excludes embeddings; recovery/KD | First-order unstructured masks |
| P19 / `xia-etal-2022-structured` | A; GLUE/SQuAD | Classifier/QA predictions | Mainly BERT-base; pruning denominator excludes embeddings | Joint layer/head/dimension pruning plus distillation/recovery |
| P20 / `mintz-etal-2009-distant` | R; Wikipedia/Freebase pair facts | Detected entities; one relation/unrelated output | Logistic classifier, features/NER/parser | KG/text alignment and pooled co-mention features |
| P21 / `dietterich2000ensemble` | T+A; classical/synthetic classifiers | Binary/multiclass, no RE contract | Classical learners, no LM scale | Statistical/computational/representational motivations; bagging/boosting |
| P22 / `schwartz2019green` | T; position/historical descriptive analysis | No extraction experiment | No controlled model roster | Efficiency objective and lifecycle accounting |
| S1 / `lin2024awq` | A; brief supplement | Decoder quantization | Activation-informed weight-only PTQ | Channel scaling; TinyChat kernels separate |
| S2 / `frantar2023gptq` | A; brief supplement | Layer reconstruction | Weight-only second-order PTQ | Approximate Hessian/error compensation; no NYT10m evidence |

## Principal results and uncertainty

Numbers are author-reported, with their original scales. Additional sample sizes, results and ablations have primary locators in the corresponding P-card. A documented source conflict is traceable, but not independently resolved ground truth.

| ID | Result and comparator | Metric / condition / locator | Uncertainty or comparison limit |
|---|---|---|---|
| P01 | SST-2 97.13 vs ELECTRA 96.56; medical 84.29 vs ICL 79.43 | Accuracy %, selected configurations; Tables 1–2/7–8, pp.3–4/11–12 | Selection sweep; demonstration-seed sensitivity; CoLA is MCC |
| P02 | Ensemble does not exceed best single acc@1 at equal total calls | Figure 5, p.5; 353 bugs; acc@k is a count | Plot coordinates omitted; cached-run resampling; API call budget differs |
| P03 | RE 0.628 vs T0pp 0.600; NER ensemble 0.907 vs best single 0.914 | Modified JAC, public leaderboard; Tables 1–2, p.1908 | API figures use whole test; exact supplementary metric unverified |
| P04 | No pooled numerical effect retained | Survey §§3–5 | No common benchmark, controlled ablation or shared budget |
| P05 | PolitiFact F1 0.68±0.04 vs supervised 0.60±0.05 | DEFENSE-acc, 2.5% labels; Table 3, p.11 | Five runs; formula/weighting/sensitivity inconsistencies |
| P06 | Phi JSON 87.4±0.3 vs stronger prompt 83.5±1.9; other combinations lose | Expected accuracy %, Table 4, p.23; token-SMC minimum | Sampling CI; global-operator assumptions; not hard-vote F1 |
| P07 | DREEAM Re-DocRED 80.73→81.69; DocRED 65.30→65.82 | Test F1 %, Table 4, p.6299 | Imported baseline cells; no repeated-seed CI; abstract uses weak LLM comparator |
| P08 | Re-DocRED five-label 50.05 vs descriptions-only 28.14 | Macro-F1 %, Table 1, p.4675 | Three label sets; ± called variance; NA contract underspecified |
| P09 | Qwen main-table micro 64.84 vs SpERT 64.92; macro 53.93 vs 43.19 | F1 %, Table 6, p.11 | Tradeoff; no main-table seed CI; Table 8 limits universal micro-loss claim |
| P10 | Voting can lose to best precision copy | Figure 3, p.6, AWQ ethics | Exact chart coordinates/headline aggregate omitted; no filtering-only ablation |
| P11 | BERT+sent+ONE micro/macro 62.9/36.1 vs BERT+bag+ATT 54.1/25.8 | Manual bag F1 %, Table 4, p.1311/PDF p.6 | Best test-P–R point; not frozen validation threshold; no seed CI |
| P12 | CNN-ATT F1 0.397/AUC 0.333 | Original DS NYT10, Table 4, p.173 | Cannot transfer to manual NYT10m; no seed CI |
| P13 | Qwen-0.5B general avg 0.828 vs GPT-5.4 0.693; RoBERTa-base 0.826 | Tables 4/11, pp.15/22; positive micro-F1 averaged over datasets | Qwen tuned/two-shot vs zero-shot API; Llama zero-shot reference 0.821; one seed; best-SLM average is oracle |
| P14 | Average micro-F1 25.6 vs GPT-5-mini 22.1 | Table 2, p.12; four joint-extraction tasks, % | Prose/table conflict; GLiREL has gold entities; no error bars |
| P15 | Supervised 77.83±0.23 vs DREEAM 80.20±0.45; zero-shot 17.32 | Re-DocRED test micro-F1 %, Tables 3–4, pp.7–8 | Different adaptation conditions; supervised five-seed spread; episodic macro-F1 separate |
| P16 | SQuAD QAT 87.74±0.15 vs FP 88.46±0.15 / DQ 80.02±2.38 | F1 0–100, development; Table 1, p.3 | Five runs; task dependent; not RE/runtime evidence |
| P17 | Four-bit QAT 81.13; +KD 83.56 vs FP 83.83 | BERT GLUE mixed-metric aggregate; Table 5, p.9 | Near-FP result needs KD qualifier; no error bars |
| P18 | Three-percent SQuAD 79.9 vs magnitude 54.5 / FP 88.1; KD 82.3 | F1 %, Tables 2–3, pp.6–7 | Encoder-weight denominator; no main-table multi-seed CI; not lossless |
| P19 | MNLI 80.6 vs FP 84.8, speedup 12.1× | Accuracy %, Tables 1/2/4, pp.1513/1518/1520 | V100 batch128 length128; setting specific, not universal >10× |
| P20 | Top100-per-relation precision 0.69 combined vs 0.67 syntactic / 0.66 lexical | Human ranked sample; Table 5, p.1010 | Ten frequent relations; fact discovery, not text-support micro-F1 |
| P21 | Independent 21-member vote at error 0.3 gives ~0.026 | Binomial illustration; Figure 1, p.2 | Independence/binary assumptions; not correlated multilabel ensembles |
| P22 | No RE/compression gain retained | Position paper Eq.1/§§2–3 | Historical estimates are not current prices/local energy |
| S1/S2 | No decoder quality/speed number transferred to encoders | Primary AWQ method; GPTQ §§3–4/6 | Tensor, architecture and kernel conditions must match |

## Cost scope

| ID | Preparation / training | Inference / memory / spend | Thesis implication |
|---|---|---|---|
| P01 | N: prior tuning/search not priced | N: GPU environment, no measured bill/latency | Count members, demonstrations and final decoder |
| P02 | N: weight-selection full cost | M: RTX3090 time, tokens, GPU energy; Figures 8–9 | Sum runs; GPU energy excludes whole system; infrastructure outliers matter |
| P03 | M: per fold T0pp ~18h/eight A100s; encoders ~20/150min/one GPU | N: deployment benchmark | Include folds, human annotation/rules and novelty stage |
| P04 | N: surveyed heterogeneous preparation | N: communication/call costs discussed | Prospective concerns, not measured routing savings |
| P05 | M: RTX3080 training time; E: FLOP formula | N: snapshot-ensemble deployment | Training savings do not remove member inference |
| P06 | No new task tuning | M: JSON Llama 0.76s single, 2.34s local, 15.24s token-SMC, 63.45s byte-SMC; Table 3 | Include approximation/alignment overhead; hardware specific |
| P07 | M: encoder training hours in Table 11 | M: API $100.91→$4.96; Llama 28.69h→1.21h; Table 12 | Referral saves expensive calls; training/inference denominator must be separated |
| P08 | N: full preparation bill | N: auxiliary GPT/BERT latency/bill | Include descriptions, hypernyms and scoring |
| P09 | N: definitions/training/search | N: complete latency/API/energy bill | Short prompts imply potential, not measured savings |
| P10 | N: conversion/calibration | N: three-copy work/memory | Parallel schematic does not prove affordability |
| P11 | N: settings and sampling motivation | N: deployment cost frontier | Direct quality evidence does not show ensemble efficiency |
| P12 | N: training bill | N: qualitative efficiency | Historical toolkit evidence, not laptop benchmark |
| P13 | M: ~600 grid +~50 DAPT GPU-hours | E: GPU/CPU latency and NF4/GGUF footprints; some storage M | Estimates are not local measurements; NF4 quality does not validate Q4 |
| P14 | N: synthetic teacher/training cost | M: ~0.9s L4 vs ~64s API; 50 FineWeb docs; §5.1 | Different workload/default API reasoning, not NYT10m frontier |
| P15 | M: ~24h pretrain/~3.5h tuning H100; episodic tuning extra | M: 500 docs ~100s/<10GB one A100 vs ~600s/>300GB four A100s | Device-time differs from wall time; no Mac/full lifecycle bill |
| P16 | N: complete QAT bill | E: ~4× weight storage; N: model latency | Stored weights differ from runtime peak memory |
| P17 | M: calibration 135.73s vs 439.29/1754s; supplement Table15 | M: FP 417.6MB vs 8-bit 104.8MB storage; N: deployment latency | Preparation/storage measurements do not prove bag speed |
| P18 | N: total training/KD bill | N: no significant ordinary-PyTorch speedup; supplementary warning | Need sparse serialization/kernels; zeros alone are not acceleration |
| P19 | M: task procedure ≤20 GPU-hours; E: ~350 TinyBERT comparator | M: V100 batch128 speedups | Account for teacher/recovery; portability unproven |
| P20 | N: parsing/features cost | N: runtime/bill | DS scale does not establish cost advantage |
| P21 | N: modern LM preparation cost | N: cost frontier | Diversity motivation, not affordability evidence |
| P22 | E: conceptual work × data × trials | N: local energy/runtime | Separate lifecycle phases; negative results legitimate |
| S1/S2 | Calibration/reconstruction described, no local estimate | Decoder kernel evidence external to local encoders | No encoder transfer without implementation measurement |

Every M/E cost is tied to its source hardware/workload in the P-card. Historical API bills are not current prices. Device-hours are not electricity/carbon estimates. Total static-ensemble work is additive even when wall time is parallelized; resident memory depends on execution strategy.

## Chapter coverage and limits

| Section | Main sources | Supporting or adjacent sources | Claim boundary |
|---|---|---|---|
| 2.1 Settings | P11/P12/P20 | P03/P07/P08/P09/P13/P14/P15 | Bags differ from articles, single-label pairs and joint detection |
| 2.2 DS/manual evaluation | P11/P20 | P07; annotation caveats P03/P09 | World truth and textual support differ; motivated-inference policy is local |
| 2.3 SLMs/encoders | P11/P12/P13/P14/P15 | P01/P03/P05/P08/P09 | Count full system; distinguish provenance/adaptation |
| 2.4 Ensembles | P21; P01/P02/P03/P05/P06/P07/P09/P10 | P04/P08; internal bag aggregation P11/P12 | Mechanisms, member construction, recoveries/errors differ |
| 2.5 Compression | P16/P17/P18/P19 | P10 with S1/S2; NF4 context P13 | Other-task encoder evidence; bits/sparsity do not prove speed/diversity |
| 2.6 Quality–cost | P22/P02/P19 | P03/P05/P06/P07/P13/P14/P15/P17; metrics P11/P09 | Separate measured/estimated/absent and lifecycle phases |
| 2.7 Positioning | P11/P13/P14/P15 and P02/P10/P16–P22 | Brief ICL P01/P09/P13; routing P07/P02/P04 | Gap limited to reviewed corpus; best single can win |

## Corrections and verification boundaries

- **Corrected:** Christou citation; unqualified calibrated-confidence language; mechanism/routing conflations; IgnF1 as pure unseen-fact quality; Chan mixture bound applied to votes.
- **Preserved and flagged:** DEFENSE loss/weighting inconsistencies; GLiNER-Relex table/prose disagreement; DocZSRE-SI confidence/role ambiguity; Tang character counts; Gao/Han published/preprint author differences.
- **Unavailable:** biomedical supplementary modified-JAC equation and supplement-only implementation details. Main-paper comparisons are verified; ordinary Jaccard/F1 is not substituted.
- **Excluded:** unverifiable graph coordinates/headline aggregates, current-price inventions, new local performance claims, absolute novelty and test-driven method selection.
- **Traceability:** each retained numerical comparison above names a metric, comparator, condition and primary locator; additional numbers remain attached to the corresponding P-card. Bibliography records use inspected versions and verified venues.

The bounded question is whether compact RE ensembles improve quality enough to justify total cost against the strongest validation-selected single and a documented API comparator. This matrix motivates that test without pre-deciding its outcome.
