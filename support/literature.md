# Background and Related Work: analytical review and writing outline

**Review date:** 6 October 2026. **Deliverable:** source-grounded analytical notes and a paragraph plan for Sections 2.1–2.7; this is not the finished chapter. The original thirteen papers are retained as P01–P13. GLiNER-Relex, GLiDRE v2, four encoder-compression studies and three targeted foundations bring the deep-review corpus to **22 papers**. AWQ and GPTQ receive short primary-source notes inside P10 and separate bibliography records.

Use this document with [the evidence matrix](evidence_matrix.md) and [the verified bibliography](references.bib). Citation keys in backticks match BibTeX keys. Paper IDs remain stable across the three documents. Published/preprint results are author-reported; no new local experiment, annotation revision, notebook edit or test-set inspection is included.

## Reading the evidence

There are four evidence classes:

- **Direct NYT10m:** P11 supplies the revised benchmark, manual annotation and bag-level evaluation evidence.
- **Other RE:** P03, P07, P08, P12–P15 and P20 concern biomedical, document, sentence, original NYT10 or NYT11 settings. Transfer to the thesis requires checking the input, output, training and evaluation contracts.
- **Other tasks:** P01, P02, P05, P10 and P16–P19 motivate ensemble/compression mechanisms and cost measurement. P06 includes other-task experiments alongside distributional theory.
- **Theory, survey or position:** P04, P06, P21 and P22 provide conceptual framing; they do not substitute for an empirical NYT10m result.

Classes describe claims, so a theoretical paper with experiments can belong to more than one class. Results are not ranked across different tasks or metrics. The source notes identify full-text sections, PDF pages and tables; formulas introduced for explanation are marked as such. References in the reviewed papers do not automatically become reviewed sources.

## Chapter writing outline

### 2.1 Relation Extraction: Sentence-Level, Bag-Level, and Document-Level Settings

**Purpose:** define the experiment's task before comparing models.

1. **Define relation extraction and the supplied-entity variant.** A relation is an ontology label attached to ordered arguments. Distinguish recognizing entity mentions from classifying relations between given mentions; joint extraction includes both. Cite `han-etal-2019-opennre` §2.1 and `stepanov2026glinerrelex` §§3–5. Explain that the thesis uses supplied spans and preserves head/tail order.
2. **Separate sentence, bag and document units.** Sentence RE uses one textual instance; a pair bag collects co-mention sentences; DocRE operates over a coherent document, frequently with mention/coreference chains and cross-sentence reasoning. Cite `han-etal-2019-opennre` §§2.2–2.4, `gao-etal-2021-manual` §4.1/§5.2, and `armingaud2025glidre` §3. Define \(B_{h,t}=\{s_j:(h,t)\text{ co-occur in }s_j\}\). A bag is not necessarily a complete article, nor a preserved discourse sequence.
3. **Define outputs and absence.** For positive ontology \(\mathcal R\), predict \(\widehat R(B_{h,t})\subseteq\mathcal R\); several labels may hold and the empty set means NA. Distinguish single-label pair classification, candidate-triple extraction and exhaustive multilabel prediction. Cite Gao, GLiDRE, Christou and Tang; their contracts differ. State direction and inverse labels according to the benchmark rather than silently treating relations as symmetric.
4. **Locate NYT10m precisely.** Gao's manually evaluated revision retains ordered entity pairs and aggregates sentence annotations into bag gold. It does not turn NYT into a full-article DocRE resource. Distinguish it from Han's original DS NYT10, Christou's processed NYT11, and MEN/DocRED articles in Chanthran. Use P11 as the direct anchor and P08/P12/P13 as boundary examples.
5. **Identify comparison consequences.** Supplied entities remove detection from the metric; bag aggregation, entity normalization, candidate selection, relation definitions and NA prevalence still affect difficulty. Joint GLiNER-Relex scores and supervised document GLiDRE scores cannot be transferred to the thesis's local adapters. Cite P14/P15 and explicitly mark this as a methodological inference.

**Transition to 2.2:** after defining what a prediction means, explain how labels are acquired and what counts as a correct relation.

### 2.2 Distant Supervision and Manual Evaluation

**Purpose:** justify the reference labels and resolve world truth versus textual support.

1. **Introduce distant supervision.** Align knowledge-graph edges with entity co-mentions to obtain large training sets. Explain the sentence-expression assumption and later bag-level mitigation. Cite `mintz-etal-2009-distant` §§3–6 and Han §2.3; avoid assigning later multi-instance neural mechanisms to the original algorithm.
2. **Explain both noise directions.** A true KG edge may be unsupported by a particular sentence; a relation expressed in text may be absent from the graph. Cite Mintz §6.3 and `gao-etal-2021-manual` §§1/3/5.3. Missing labels and wrong positive labels have different effects on measured precision/recall. Gao's numerical DS diagnostics describe its annotated construction, including model-selected NA examples; they are not population prevalence estimates.
3. **Explain manual sentence and bag gold.** Describe independent relation annotation, majority/adjudication, multilabel decisions and the union \(R(B)=\bigcup_{s\in B}R(s)\). Cite Gao §§3.1/5.2. A relation can be supported by one sentence; that does not license combining arbitrary unrelated facts into a new relation. Record the supporting sentence or inference chain where manual review is performed.
4. **Distinguish the two evaluation targets.** Mintz's new-fact human evaluation asks whether a relation holds; Gao asks what the text expresses. Zhang Appendix F also discusses world-knowledge answers unsupported by documents. Cite P20/P11/P07. The presence of `nationality` in the ontology makes it an eligible output, but does not by itself provide evidence for any particular nationality. A world-true output may be appropriate under fact discovery and unsupported under text-grounded extraction.
5. **State the thesis's agreed operational rule.** Count explicit support and justified linguistic/contextual inference as text-supported; store world-knowledge-only correctness separately. For an inference, identify the textual premise, ontology interpretation, direction and inferential step. Demonyms, explicit citizenship or contextually resolved references may supply evidence; birthplace, a club affiliation or a name alone do not automatically establish citizenship. Likewise, `contains` requires evidence for the appropriate geographic/part–whole direction under the ontology. These are proposed annotation controls for the chosen target, **not a universal guideline experimentally validated by the reviewed papers**. Do not adjudicate Barzagli without its actual bag. This review does not change existing labels.

**Transition to 2.3:** once the target is fixed, introduce models that can predict it and the origins of their task competence.

### 2.3 Small Language Models and Encoder-Based Relation Extractors

**Purpose:** define “small,” describe the actual model families, and expose adaptation/provenance differences.

1. **Define scale operationally.** The thesis's main member constraint is below one billion parameters, including deployed task modules and all encoders; it is not a universal community threshold. Contrast P01's sub-1B members plus an 8B final decoder, P02/P06's 7–14B systems and P03's 11B component. Cite their model tables and `christou2026subbillion` Table 21. Compressed precision reduces representation size, not parameter count.
2. **Explain encoder and generative predictors.** Encoder RE pools contextual representations of supplied entities/spans and produces class/label scores; autoregressive decoders generate labels or structures token by token. Decoder-only, encoder-only and encoder–decoder models are distinct architectures. Cite Han, Christou and biomedical Li. Parsing/schema validity is separate from relation correctness; no general speed claim follows solely from architecture.
3. **Discuss specialization and provenance.** Separate general pretraining, domain-adaptive pretraining, synthetic extraction supervision, target-task fine-tuning and inference-only label prompting. Cite P03/P05/P13–P15. Christou supports the strength of specialized singles under a particular API comparison; it does not establish a level training budget. Audit dataset overlap, task preprocessing and synthetic teacher sources before interpreting local results.
4. **Describe GLiNER-Relex.** Shared contextual encoding of labels/text, span scoring and ordered-pair relation heads; main published evaluation includes predicted entities. Cite P14 Equations 7–9/§§3.8–5. Explain the thesis's supplied-span and bag adaptation as a separate protocol. Preserve the paper's table/prose conflict and the base/large recipe distinction.
5. **Describe GLiDRE v2 and comparator regimes.** Text/label dual encoders, gold mention/coreference pooling, pair-label dot products and independent thresholds. Cite P15 §§3–4, Tables 1/3/4. Approximately 800M refers to both encoders; supervised and zero-shot scores are different conditions. Neither published model establishes its NYT10m adaptation's performance.

**Transition to 2.4:** competent compact singles are the necessary reference for asking whether combining them creates useful gains.

### 2.4 Language Model Ensembles and Error Complementarity

**Purpose:** distinguish mechanisms and measure useful diversity rather than assuming it.

1. **Introduce conditional ensemble benefits.** Dietterich's statistical, computational and representational explanations motivate accurate members with differently located errors. Its independent binary-vote illustration is a restricted case, not a theorem about NYT10m multilabel F1. Cite `dietterich2000ensemble` §§1–3 and P02's correct-set overlaps.
2. **Define the mechanism taxonomy.** Output voting combines final labels/triples (P02/P03/P10); score fusion combines aligned scores (P05/P07); context fusion exposes predictions to another decoder (P01/P09); routing selectively invokes a stage (P07); joint decoding combines distributions during generation (P06). Sentence-evidence aggregation inside one model (P11/P12) is distinct from combining checkpoints. P04 adds interacting agents as adjacent workflow literature, not a synonym for any of these.
3. **Separate how members are constructed.** Different pretrained/task checkpoints, seeds/data subsets, sequential snapshots and precision copies have different dependencies. Cite P01, P05 and P10. Precision variants and pruning variants may alter errors but cannot be presumed independent. Lu motivates evaluating same-backbone copies but supplies no encoder RE or net-cost demonstration.
4. **Specify complementarity diagnostics.** For member positives \(P_i\) and gold facts G, inspect exclusive correct facts \((P_i\setminus P_j)\cap G\), exclusive incorrect additions \((P_i\setminus P_j)\setminus G\), shared correct predictions and shared misses. Disagreement alone is neither quality nor achievable gain. An oracle union of correct facts is an upper-bound diagnostic; an actual union/intersection/vote can lose recovered facts or add errors. Cite Dietterich, Cho and Zhang; these set diagnostics are the thesis's application of their principles.
5. **Keep negative results and theoretical scope.** P02's equal-call acc@1, P03's NER JAC, P09's main-table micro-F1 and P10's ethics comparison all qualify unconditional success. Chan's global-mixture expected-correctness bound does not cover hard voting, argmax or learned selection. Its printed weighted maximum is inconsistent with its preceding convex sum; document the correction in P06. No unsupported voting ceiling should enter the chapter.

**Transition to 2.5:** combining more predictors increases workload; compression may change the feasible set but must be evaluated as a separate intervention.

### 2.5 Model Compression: Quantization and Pruning

**Purpose:** explain compression mechanisms that are relevant to encoders without assuming decoder recipes transfer.

1. **Define quantization by tensors and adaptation.** Weight precision, activation precision, accumulator/bias precision and retained floating-point operations are separate. PTQ operates on an already trained checkpoint, usually with calibration; dynamic scaling is computed at runtime. QAT simulates quantization while adapting weights. Cite `zafrir2019q8bert` §§2–3 and `wei2022outlier` §§3–4. Christou's QLoRA/NF4 setup is not a PTQ/QAT comparative experiment.
2. **Explain outlier handling.** Learned LayerNorm scales and token-dependent activation ranges can make low-bit encoder quantization difficult. Gamma Migration is an equivalent reparameterization before quantization; Token-Wise Clipping searches downstream reconstruction error. Cite P17's main/supplemental ablations and distinguish PTQ, QAT and QAT+KD results. Near-FP quality at one task/bit configuration is not universal losslessness.
3. **Place AWQ/GPTQ briefly.** Activation-informed weight-only scaling and second-order reconstruction are relevant decoder precedents reviewed inside Lu. Cite `lin2024awq` and `frantar2023gptq` without broadening them into core encoder evidence. Activations used for calibration do not imply activation quantization. TinyChat/kernel speedups are separate from the quantizer's accuracy claim.
4. **Define pruning granularity.** Unstructured pruning masks individual weights; Movement Pruning learns importance during task adaptation. Structured CoFi removes layers/heads/dimensions and adds distillation/recovery. Cite `sanh2020movement` §§3–5 and `xia-etal-2022-structured` §§3–4. Sparsity denominators may exclude embeddings; teacher and recovery costs matter.
5. **Connect compression to deployment and diversity.** Sparse zeros in dense tensors need not accelerate execution, whereas CoFi supplies hardware-specific measured speedups. Lower precision requires compatible kernels and buffers. Each compressed model needs its own quality/cost measurement before considering ensembles of variants. No reviewed compression paper establishes NYT10m complementarity. Cite P16–P19 and P10.

**Transition to 2.6:** move from architectural potential to a consistent measured quality–cost comparison.

### 2.6 Quality–Cost Evaluation

**Purpose:** define how the thesis will decide whether an ensemble is worthwhile.

1. **Define quality on positive facts.** Compute \(P=TP/(TP+FP)\), \(R=TP/(TP+FN)\), \(F_1=2TP/(2TP+FP+FN)\); aggregate counts across bag–relation decisions for micro scores and average per-relation F1 for macro scores. NA is an empty positive set, so correct NA is not added as a positive true positive. Document zero-denominator conventions. Cite Gao and Tang; distinguish biomedical JAC, CoLA MCC, Cho acc@k and Chan expected accuracy. IgnF1 is a benchmark overlap adjustment, not a universal leakage detector.
2. **Separate selection and uncertainty.** Choose thresholds, members, prompts, caps and fusion settings on train/validation, then freeze them for final evaluation. Gao's best-test-P–R operating point is a published convention, not the thesis's deployable selection method. Different bootstrap/seed/demonstration intervals cover different uncertainty. Cite P11/P01/P13/P15; compare predictions on the same bags and preserve individual outputs.
3. **Account for total execution.** For a static m-member ensemble, total compute/call expenditure is \(C_{\rm ens}=\sum_i C_i+C_{\rm fusion}\). Parallel wall time can approach the slowest member plus scheduling, while total work remains additive and simultaneous memory can increase. Report elapsed time/throughput, peak process memory and API token spend separately. Cached offline fusion avoids rerunning members during analysis, but deployment still incurs their inference. Cite P02/P06/P10/P22, marking these accounting equations as thesis definitions.
4. **Separate lifecycle phases and measurement status.** Report training, synthetic annotation, calibration/pruning/distillation, configuration search and inference distinctly. Mark every cost measured, estimated, inherited from another work, or absent. Use P03/P05/P13/P17/P19 as examples. Zhang's mixed training-plus-inference denominator must not become deployment overhead. Hardware, batch size, token/word length, bag cap, precision, software and cold/warm loading affect comparability.
5. **Define Pareto comparison and baselines.** A configuration is dominated if another is no worse in quality/cost and strictly better in at least one axis. Present quality–latency and quality–memory/spend frontiers rather than assuming one exchange rate for F1. Compare with the strongest validation-selected single and a documented API baseline using matching inputs/ontology. API dates/prices/providers and failure/retry costs must be explicit. Cite `schwartz2019green` and P02/P19. Claims about energy/carbon require measured or transparent estimated energy, not parameter counts alone.

**Transition to 2.7:** identify which intersection of task, scale, mechanism and cost remains to be tested in this reviewed corpus.

### 2.7 Research Gap and Positioning of This Thesis

**Purpose:** formulate a bounded empirical contribution that permits an ensemble to lose.

1. **Summarize the evidence intersection.** Gao anchors manually evaluated bag RE; Christou and GLiNER-Relex/GLiDRE support specialization and compact encoder alternatives; other RE collaborations operate with different input/adaptation/scale contracts; ensemble and compression studies provide mechanisms on adjacent tasks. Cite P11/P13–P15/P03/P07/P09/P02/P10/P16–P19. These strands motivate the experiment but do not establish its outcome.
2. **State the reviewed-corpus gap.** Among these examined sources, no study jointly establishes a static ensemble of sub-1B RE members on the thesis's NYT10m supplied-pair bag protocol, with separate text-supported manual review, comparisons to the strongest tuned single and an API baseline, and measured local total-member latency/memory. This is a claim about this review's coverage, **not an exhaustive claim that no such work exists**.
3. **Frame the contribution and questions.** Ask whether an ensemble improves useful relation recovery enough to justify its extra errors and total deployment cost; whether compression changes that answer; and whether any quality–cost gain remains against the best single and API comparator. Different members and precision/pruning variants are hypotheses to test. A strong single winning, or no ensemble Pareto improvement, is an informative result under the cost-aware objective. Cite P02/P09/P10/P22.
4. **Keep extensions brief and separate.** ICL can supply training-set demonstrations or member outputs in context, but adds retrieval/prompt/decoder cost and may depend on matched training format (P01/P09/P13). Routing may avoid unnecessary expensive calls, but needs validated referral rules and complete selector/verification costs (P07; suggestions in P02/P04). GLiDRE's episodic support adaptation is not ICL. Neither extension is a required outcome of the static-ensemble experiments.
5. **Bridge to the methods chapter.** Specify the dataset splits, manual evidence policy, supplied spans/direction, bag aggregation, model provenance, member/fusion choices, compression settings, API protocol, frozen selection rules and cost instrumentation. Related work motivates these choices; the methods chapter must define them exactly and the results chapter must measure them.

**Transition out of the chapter:** the next chapter turns these bounded research questions into a reproducible experimental protocol.

## Corrections and unresolved source issues

| Issue | Resolution for chapter writing | Source notes |
|---|---|---|
| Christou identity and benchmark | Use Christou/Tsoumakas 2026 v1; NYT11/pair preprocessing, not direct NYT10m | P13 |
| “Calibrated confidence” | Use confidence/score unless a calibration procedure and diagnostic are actually supplied | P01/P02/P07/P08/P09/P10 |
| DEFENSE loss formula | Preserve printed 2(1−u); flag range/direction contradiction; do not invent corrected implementation | P05 |
| Routing | Zhang selectively refines entity-pair decisions; Cho/survey routing is prospective | P02/P04/P07 |
| IgnF1 | Treat as task-specific training-fact overlap adjustment, without claims of pure unseen reasoning | P07/P15 |
| Chan bound | Correct convex-mixture bound is max expert expectation; applies to that distributional operator, not hard votes | P06 |
| GLiNER-Relex results | Use table-backed cells and flag conflict with prose | P14 |
| Biomedical metric | Keep modified JAC; exact supplement-only formula is unverified because attachment was inaccessible | P03 |
| Other reproduction details | Document source ambiguities instead of repairing them silently: DEFENSE weighting, DocZSRE-SI role/confidence, Tang counts, published/preprint author differences | P05/P08/P09/P11/P12 |

All principal quantitative claims retained below have primary-source locators. Source conflicts are flagged and restricted; unlabelled graph coordinates and unverifiable headline averages are omitted. The inaccessible biomedical supplement is not needed for the bounded related-work comparisons, but is needed to reproduce that metric exactly. The review does not certify the reviewed models' training data as contamination-free.

## Paper analyses

| Paper | Citation key | Analysis |
|---|---|---|
| P01 | `mojarradi2024ensemble` | Ensemble SuperICL: predictions as in-context evidence |
| P02 | `cho2025cosmosfl` | COSMosFL: voting under measured inference budgets |
| P03 | `li2024biomedical` | Li and Wei et al.: biomedical extraction with context, folds, and models |
| P04 | `li2024multiagent` | Li and Wang et al.: multi-agent systems as adjacent workflow literature |
| P05 | `folino2025defense` | DEFENSE: sequential encoder snapshots and scarce labels |
| P06 | `chan2026smc` | Chan et al.: distributional ensembles and Sequential Monte Carlo |
| P07 | `zhang2025refiner` | Zhang et al.: selective document-pair refinement and probability fusion |
| P08 | `chanthran-etal-2026-document` | Chanthran et al.: document zero-shot RE with side information |
| P09 | `tang2026collaboration` | Tang et al.: collaboration in classical Chinese joint extraction |
| P10 | `lu2025mixedprecision` | Lu et al.: whole-model precision-ensemble voting |
| P11 | `gao-etal-2021-manual` | Gao et al.: Manual Evaluation Matters |
| P12 | `han-etal-2019-opennre` | Han et al.: OpenNRE task and toolkit boundaries |
| P13 | `christou2026subbillion` | Christou and Tsoumakas: specialization, scale, and matched prompting |
| P14 | `stepanov2026glinerrelex` | GLiNER-Relex: joint entity and relation representations |
| P15 | `armingaud2025glidre` | GLiDRE v2: dual-encoder document relation extraction |
| P16 | `zafrir2019q8bert` | Q8BERT: quantization-aware encoder adaptation |
| P17 | `wei2022outlier` | Outlier Suppression: making low-bit encoder quantization viable |
| P18 | `sanh2020movement` | Movement Pruning: task-adaptive unstructured sparsity |
| P19 | `xia-etal-2022-structured` | CoFi: structured pruning with measured latency |
| P20 | `mintz-etal-2009-distant` | Mintz et al.: distant supervision and the evaluation target |
| P21 | `dietterich2000ensemble` | Dietterich: why ensembles can help and why diversity is conditional |
| P22 | `schwartz2019green` | Green AI: efficiency as an evaluation objective |

## P01. Ensemble SuperICL: predictions as in-context evidence

**Citation key:** `mojarradi2024ensemble`. **Source/version:** M. Mehdi Mojarradi, Lingyi Yang, Robert McCraith and Adam Mahdi, *Improving In-Context Learning with Small Language Model Ensembles*, arXiv:2410.21868v2, 20 December 2024; presented at the NeurIPS 2024 Workshop on Adaptive Foundation Models. [Primary version record](https://arxiv.org/abs/2410.21868v2). Local source: `Mehdi Mojarradi - Lingyi Yang.pdf`, 13 pages, including Appendices A-G. All pages read; Tables 2-3 additionally inspected visually. **Evidence class:** adjacent text-classification experiments; direct evidence for a context-fusion mechanism, not for NYT10m or relation extraction.

### Question, method and adaptation

The question is whether a large decoder can use predictions from several specialized small models to improve ICL, including transfer to a task on which those small models were not fine-tuned. The method extends single-plugin SuperICL. For each labeled demonstration, every member supplies a predicted label and confidence score. The same information accompanies the unlabeled query. The decoder receives the instruction, augmented demonstrations and augmented query, and emits the final class using greedy decoding. In compact explanatory notation, not an equation printed in the paper,

\[
\hat y=G\left(I,\{(x_j,\{(\hat y_{ij},c_{ij})\}_{i=1}^{m},y_j)\}_{j=1}^{k},x,\{(\hat y_i,c_i)\}_{i=1}^{m}\right).
\]

This is **fusion through the decoder's input context**. It neither averages encoder logits nor implements an explicit learned router. The description of confidence in Section 2, p.3, is a sigmoid applied to the model's logit-derived output. The paper provides no calibration procedure, reliability diagram, ECE/Brier evaluation or guarantee that different members' scores are comparable. Therefore use *confidence scores*, not *calibrated confidence*. Its explanation that the decoder learns which member to trust is a proposed interpretation, rather than a measured identification of a trust mechanism.

The pool in Table 1 (p.3) contains MobileBERT 25M, flan-t5-base 248M, ELECTRA-large 335M, DeBERTa-large 350M, RoBERTa-large 356M, BART-large 407M and T5-large 770M. The term SLM is explicitly defined as a task-tuned model below 1B parameters (p.2, footnote 2); the final predictor still includes **Llama3-8B-Instruct**. Released GLUE-task-fine-tuned members are used for the corresponding GLUE task. Only the medical case study avoids target-task tuning of the members, by reusing MNLI-fine-tuned checkpoints. Thus the method is not generally training-free, although it does not newly fine-tune the decoder. The Section 3 prose says flan-t5-large, whereas Tables 1 and 8 say flan-t5-base; retain the table-backed base/248M description and record the source inconsistency.

### Task and evaluation contract

The inputs are sentences or sentence pairs, not ordered entity pairs. There is no RE ontology, relation direction, multi-label fact set or NA class. Appendix A, Table 4 (p.8), reports SST-2 sentiment (872 evaluation examples), MRPC paraphrase classification (408), MNLI natural-language inference (9,815), CoLA grammatical acceptability (1,043), and a balanced 700-question medical-subject classification subset. The medical labels are *Dental* and *Surgery*: this is not answering MedMCQA's multiple-choice medical questions. SST-2/MRPC/MNLI/MedMCQA use accuracy; **CoLA uses Matthews correlation coefficient (MCC)** despite several main-text references to accuracy.

Sections 3-4 and Appendices D-F evaluate all eligible member combinations of size two to five across zero/eight/sixteen/twenty-four/thirty-two demonstrations. Demonstrations come from training data and are shared between compared configurations. The best configurations are reported after a broad sweep; an independent selection-versus-final-test procedure is not established. Do not interpret maxima as estimates from a preregistered configuration.

### Verified results, ablation and uncertainty

Representative comparisons below keep metrics and conditions attached (Tables 1-2, pp.3-4; Appendices D-F, pp.10-13):

| Task | Relevant individual / plain ICL baseline | Selected Ensemble SuperICL result | Condition and locator |
|---|---|---|---|
| SST-2 | ELECTRA-large 96.56% accuracy; plain ICL 94.15% | 97.13% | RoBERTa + ELECTRA, 32 demonstrations; Tables 1-2 and 8 |
| MRPC | ELECTRA-large 89.95% accuracy; plain ICL 75.25% | 91.42% | Selected three/four-member configurations; Tables 1-2, 5 and 8 |
| MNLI | Best listed individual DeBERTa-large 90.39%; majority vote 91.39% | 91.27% accuracy | Five members, zero demonstrations; Tables 1-2, 5 and 8; ensemble loses to majority vote |
| CoLA | ELECTRA-large 67.43 MCC x100; plain ICL 55.43 | 70.36 MCC x100 | Two-member 16-shot or three-member 24-shot selections; Tables 1-2, 6 and 8 |
| Medical subject | Best MNLI-tuned member DeBERTa-large 71.43%; plain ICL 79.43%; single-plugin SuperICL 82.71% | 84.29% accuracy | DeBERTa + flan-t5-base, 16 demonstrations; Tables 1-2, 7 and 8 |

The summary table's two-member MNLI value should not be described as the overall maximum: Appendix D includes stronger zero-shot two-member rows. The results above avoid that inconsistent cell.

Table 3 (p.5) shows MNLI 91.14% with all reported components versus 84.76% when both demonstration predictions and confidence are absent; medical subject accuracy falls from 84.29% to 78.57%. Removing confidence alone gives CoLA 67.25 versus 70.36 MCC x100. However, **every ablation row retains test-input predictions**. The previous digest's assertion that removing test predictions is least harmful is unsupported. Moreover, MRPC is unchanged at 91.42 when demonstration predictions are absent, so the narrative that every component is necessary should be qualified. Appendix G, Table 9 (p.13), varies five demonstration seeds: the selected medical configuration ranges from 80.00% to 84.29%, with reported variance 2.44. This is sensitivity to demonstration selection, not a confidence interval or proof of significance. Weak-member assistance is configuration-dependent: Appendix D's MobileBERT + T5 reaches 62.21 MCC x100 on CoLA, although both listed individual MCC values are lower than plain ICL.

### Cost, limitations and chapter use

Experiments run on two A100 80GB GPUs (p.4). The paper does **not** report measured end-to-end latency, energy, monetary cost or a quality-cost frontier. Existing checkpoint fine-tuning, member inference, labeled demonstrations, longer prompts, decoder inference and configuration search remain relevant costs. Larger or more numerous member sets do not monotonically improve results. Section 4.3 (p.5) explicitly acknowledges task-dependent choices and restriction to classification.

**Supports:** heterogeneous sub-1B predictions can improve a decoder's classification when supplied as context; transfer from an adjacent supervised task can work in the small medical case study; compare against strong single members and majority vote. **Does not establish:** local sub-1B-only inference, calibrated score fusion, a bag-level RE gain, a general training-free workflow, or lower total inference cost. **Placement:** 2.3 for operational SLM definitions and encoder/decoder distinction; 2.4 for context fusion and non-monotonic ensemble gains; 2.6 for missing cost evidence; a brief optional ICL extension in 2.7.

## P02. COSMosFL: voting under measured inference budgets

**Citation key:** `cho2025cosmosfl`. **Source/version:** Hyunjoon Cho, Sungmin Kang, Gabin An and Shin Yoo, *COSMosFL: Ensemble of Small Language Models for Fault Localisation*, IEEE/ACM International Workshop on Large Language Models for Code (LLM4Code), 2025, pp.17-24, DOI [10.1109/LLM4Code66737.2025.00007](https://doi.org/10.1109/LLM4Code66737.2025.00007). [Author-institution proceedings record](https://pure.kaist.ac.kr/en/publications/cosmosfl-ensemble-of-small-language-models-for-fault-localisation/). The analyzed local `Hyunjoon Cho - Sungmin Kang.pdf` is the eight-page arXiv:2502.02908v1 preprint dated 5 February 2025; its pagination supplies locators below. All pages read, including references; Figures 2-10 inspected visually. **Evidence class:** adjacent software-engineering task, with measured ensemble cost.

### Question, method and member construction

The research question is whether heterogeneous local language models can replace repeated samples from one model in fault localization while offering useful cost-quality choices. COSMosFL wraps the AutoFL repository agent, which uses failing-test coverage, code and comments to rank suspicious methods. It removes AutoFL's class-coverage tool because smaller models called it redundantly (Section II.B, p.2). This adaptation matters when comparing to the original GPT agent.

A run predicts a set of suspicious methods, giving each equal share of that run's vote. Rewriting the definition in Section II.A-B (pp.2-3), let \(S_{ir}\) be the method set returned by member \(i\) on run \(r\). Then

\[
v_i(a)=\frac1{R_i}\sum_{r=1}^{R_i}\frac{\mathbf1[a\in S_{ir}]}{|S_{ir}|},\qquad
v(a)=\sum_i w_i v_i(a).
\]

Methods are ranked by \(v\). The maximum aggregated score is termed confidence: it measures vote concentration, with no probability-calibration guarantee. Empty-output handling is not specified by this mathematical summary and should not be invented. Equal weights are compared with differential-evolution weights optimized for top-one accuracy, using wasted effort as a tie-breaker. Section III.D (p.4) specifies population 40, 30 generations, differential weight 1.5, crossover probability 0.8, and ten-fold cross-validation. This learns **instance-independent voting weights**, not an input-dependent routing policy.

Seven candidate 4-bit Ollama models are screened: CodeLlama 7B, Gemma2 9B, Granite3 8B, Llama3 8B, Llama3.1 8B, Mistral NeMo 12B and Qwen2.5-Coder 7B (Section III.B, p.4). Four are selected for complementary correct bug sets: Llama3, Llama3.1, Mistral NeMo and Qwen2.5-Coder. Their parameter scale is **7B-12B**, outside this thesis's sub-1B definition. The claimed individual quantized files below 8GB are a deployment motivation, not a measured bound on total runtime memory or CPU speed. No task-specific parameter fine-tuning is described; adaptation is prompting, tool design, membership and voting weights.

### Dataset, evaluation and verified findings

The evaluation uses 353 bugs from five Defects4J projects: Chart 26, Closure 131, Lang 64, Math 106 and Time 26 (Table I, p.4). A unit is a bug/repository, and a prediction is a ranked method list. There are no entity spans, directed relations, RE ontology or NA abstention contract. The paper's acc@k is a **count** of bugs with a faulty method in the top k, not automatically a percentage. Converting to a percentage would require division by 353 and explicit labeling.

Each selected member is run 30 times. For each run budget, 20 subsets of those cached runs are sampled; ensemble budgets have one to six runs per member, so \(R=4R_i\), from four to twenty-four total runs (Section III.B, p.4). Consequently, assigning all ensemble weight to one member would still use only \(R/4\) of that member's calls, whereas its single-model baseline receives \(R\). This explains why membership selection and allocation policy must be evaluated together.

Specific evidence is:

- The initial Llama3/Gemma2 comparison shares 71 correct top-one bugs, with 37 and 41 exclusive successes respectively (Section II.B, p.2). These are explicit correct-set overlaps; they do not claim independent errors. The final four-model correct-set union covers 180 bugs in the initial five-run screening (Section IV.A, Figure 2, p.4), an oracle coverage statistic, not the actual ensemble's top-one score.
- The ensemble **does not surpass the best single model at acc@1 for equal total runs** (Figure 5 and Section IV.A, p.5). At larger k, the ensemble tends to outperform members (Figure 3, p.4). This is a useful negative result against the strongest relevant individual baseline.
- Equal and optimized weighting are similar; DE does not consistently improve the equal-weight baseline. Mean optimized weights remain around 0.15-0.37 (Figure 6 and text, p.5). Pairwise weight grids in Figure 10 (p.7) show optima often near equal weighting, with a more irregular landscape as runs increase (Section V, p.6).
- Figure 7 (p.5) displays median and empirical 50/80/95% intervals over sampled runs. These show resampling variability, not independent test replications or formal pairwise significance.
- GPT-3.5 results imported from AutoFL use five runs, while Figure 3's COSMosFL configurations use twenty. Section IV.A explicitly calls the direct comparison unfair; do not cite it as a controlled claim that small models outperform the commercial system.

Because key quality and cost values are unlabelled plot coordinates, exact curve points are not transcribed into the numerical evidence matrix.

### Measured cost and applicability

Sections III.D and IV.B (pp.4,6), Figures 8-9 (pp.6-7), measure **GPU energy through EnergyMeter/nvidia-smi**, execution time, and input-plus-output tokens on a single RTX3090, 252GB RAM and 40 Xeon CPU cores. This excludes whole-system/CPU energy and says nothing quantitative about training cost. Token count is a workload measure; it is not a complete API-price calculation and treats input/output equally despite potentially different rates.

Figure 8 places ensembles on useful parts of the quality-cost frontier between cheaper and stronger members. However, Llama3.1 and Qwen2.5-Coder exhibit severe time/energy outliers, which the authors suspect stem from an Ollama endless-generation issue (Section IV.B, p.6). This is a suspected infrastructure cause, not demonstrated intrinsic model inefficiency; it affects the interpretation of aggregate cost and Pareto positions. Use the work's multi-axis reporting method without transferring its numeric GPU costs to encoder inference on a Mac.

**Supports:** correct-set complementarity analysis, an equal-weight baseline, cost-budget comparisons, accounting for every member/run, and the possibility that the best single remains preferable. **Does not establish:** NYT10m improvement, sub-1B efficacy, independence of errors, or an implemented router. Routing and cascades are suggestions in Section V, not evaluated components. Membership is selected from the same benchmark family before final evaluation, so an independent frozen selection split is not established. **Placement:** 2.4 for output-level voting and complementarity; 2.6 for measured quality-cost analysis and infrastructure effects; 2.7 for short routing prospects.

## P03. Li and Wei et al.: biomedical extraction with context, folds, and models

**Citation key:** `li2024biomedical`. **Source:** Zhao Li, Qiang Wei and colleagues, *Ensemble pretrained language models to extract biomedical knowledge from literature*, *Journal of the American Medical Informatics Association* 31(9), 1904–1911, [DOI:10.1093/jamia/ocae061](https://doi.org/10.1093/jamia/ocae061), published online 23 March 2024. All eight supplied article pages were read; result tables were also inspected visually. **Evidence class:** other RE, biomedical document-level extraction; hybrid encoder/11B decoder ensemble.

### Question, method, and input/output contract

The LitCoin system asks how domain-specialized pretrained models, additional annotation and ensemble learning can extract entities, relations and scientific novelty from PubMed abstracts. The first phase predicts entity spans/types; the second classifies pairs of normalized biomedical concepts in an abstract. The dataset has 400 development and 100 evaluation abstracts, six entity types and eight positive relation types. Novelty means whether the relation is presented as a new discovery **in that publication**, not whether a general-purpose model has previously encountered the fact (§Methods, pp.1905–1906).

NER uses BioBERT, PubMedBERT and BioM-ELECTRA-Large with BIO tagging and cross-sentence contexts. Predictions are combined at context, ten-fold and backbone levels, retaining mentions supported by a majority at each tier (Figure 2, p.1907). Seventy additional manually annotated abstracts and rules resolve organism/cell-line type inconsistencies; these are separate interventions, not effects of voting alone.

RE considers every two-concept candidate combination. Encoder inputs contain title and relevant mention sentences; T0pp uses a task prompt and abstract information. One-step prediction combines relation and novelty into 17 labels, including absence. Two-step prediction first identifies relation type, then uses a random forest with sentence-location features to predict novelty. The main article says equal-weight majority voting combines PubMedBERT, BioM-ELECTRA-Large and **T0pp-11B** outputs (§Novel RE, pp.1906–1907). In explanatory notation for its categorical vote, \(\hat y=\arg\max_c\sum_i\mathbf1[\hat y_i=c]\). This is not independent thresholding of the thesis's NYT10m multilabel set; directionality and multi-relation handling are not specified sufficiently to generalize the procedure to that ontology. Exact fold/context merge details depend on the supplement.

Ten-fold supervised training creates adapted checkpoints. Domain pretraining, target fine-tuning, cross-sentence context, location filtering, rules and voting all contribute to the final pipeline. Although two encoder backbones are compact, the complete RE ensemble contains an 11B decoder and is outside the thesis's sub-1B member constraint. The article does not provide a complete checkpoint-specific encoder parameter accounting.

### Results, ablations, and comparison limits

The competition metric is a **modified Jaccard similarity (JAC)**, not ordinary relation micro-F1. Table 1 (p.1908) reports public-leaderboard NER JAC **0.914** for BioM-ELECTRA-Large with added annotation/rules versus **0.907** for the final three-backbone ensemble. The paper submits the ensemble for hoped-for robustness despite this observed loss; robustness is not separately demonstrated. Cross-sentence context and extra annotation explain 0.890→0.896→0.912 for that backbone before rules.

Table 2 (p.1908) gives novel-RE public-leaderboard JAC **0.628** for the ensemble plus rules versus **0.600** for two-step T0pp. Under the alternative BioRED reference in Table 3 (p.1909), the location-filtered two-step scores are **0.6090** for the ensemble, **0.5935** for T0pp and **0.5910** for BioM-ELECTRA-Large. These are different evaluation conditions and must not be pooled. Public/private leaderboard partitioning is 50/50. The ChatGPT 3.5/4 results are computed on the **whole** test because the authors lack the organizer's partition; they therefore are not identical-subset comparisons with the public-leaderboard systems.

Table 3 separates relation-only and novelty-dependent contributions to JAC and shows that two-step prediction is not uniformly beneficial for every backbone. Table 4 shows strongly uneven relation support. A paired t-test is reported for one NER cross-context comparison across the ten folds; this does not demonstrate statistical significance of the RE ensemble's improvement or training-seed independence. No uncertainty interval for the final RE JAC is reported.

### Cost, verification gap, and chapter use

The discussion (pp.1909–1910) reports approximately **18 hours per fold on eight A100 40GB GPUs** for T0pp fine-tuning, versus approximately **20 minutes** for PubMedBERT and **150 minutes** for BioM-ELECTRA-Large on one GPU per fold. Ten-fold ensembling multiplies training workload. Approximately ten annotation hours per annotator are also reported for the added abstracts. These are training/human preparation costs, not measured deployment latency, memory peaks, energy or API expenditure.

**Explicit verification gap:** the publisher/PMC supplementary attachment could not be retrieved as a valid document; the apparent DOCX response was an HTML access page. The exact modified-JAC formula, supplement-only architecture settings and fine-grained checkpoint/fold merge instructions remain unverified. No ordinary Jaccard formula is substituted. The main article is sufficient for the bounded conclusions above; the supplementary file would be required before reproducing its metric or exact pipeline.

**Supports:** RE-specific output combination can help in a specialized supervised pipeline; NER and RE gains can differ; cost includes folds, rules, annotation and all deployed members. **Does not establish:** sub-1B-only RE, NYT10m improvement, calibrated confidence or unconditional ensemble robustness. **Placement:** 2.1, 2.3, 2.4 and 2.6; comparison boundary in 2.7.

## P04. Li and Wang et al.: multi-agent systems as adjacent workflow literature

**Citation key:** `li2024multiagent`. **Source:** Xinyi Li, Sai Wang, Siqi Zeng, Yu Wu and Yi Yang, *A survey on LLM-based multi-agent systems: workflow, infrastructure, and challenges*, *Vicinagearth* 1, article 9 (2024), [DOI:10.1007/s44336-024-00009-2](https://doi.org/10.1007/s44336-024-00009-2). All 43 supplied pages, including references, were read. **Evidence class:** survey of other tasks and workflows; no direct RE experiment.

### Question and conceptual contribution

The survey organizes LLM-based multi-agent systems by workflow, infrastructure and unresolved problems. Section 3 (pp.4–26) identifies five components: **profile, perception, self-action, mutual interaction and evolution**. Profiles specify roles, goals and capabilities; perception turns environment information into usable observations; self-action encompasses memory, reasoning, planning and tools; interaction governs communication and cooperation; evolution covers feedback and adaptation. Section 4 (pp.26–30) groups applications into problem solving and world simulation. These components give vocabulary for distinguishing an agentic workflow from a static predictor ensemble.

A multi-agent system may assign different roles to repeated instances of the **same** language model. Different prompts, memories and tool access can yield behavioral diversity without distinct model parameters. Conversely, a classifier ensemble can combine several checkpoints in one fixed rule without communication, planning, memory or autonomous task decomposition. Neither configuration implies the other. The survey's module taxonomy is not a voting formula or an implemented relation extractor.

### Mechanisms, evidence, and applicability

Section 3.4 (pp.21–24) discusses centralized, decentralized and hierarchical interaction structures, cooperation and competition, and communication/task/environment scenarios. Repeated conversations can change later outputs, unlike independently computed member predictions. The self-action and evolution sections discuss tool use, memory and feedback; they do not establish that a final majority vote has calibrated confidence or independent errors. No single equation should be imposed as the method of this heterogeneous survey.

The reviewed systems cover software development, reasoning, decision support, simulations and multimodal interaction, with many different LLM/API sizes and adaptation regimes. There is no shared dataset, model roster, budget, test unit or pooled benchmark. Entity spans, directed relation labels, multilabel decoding and NA are not a common evaluated contract. Individual systems' reported outcomes remain secondary evidence unless their primary papers are examined. The review therefore retains **no cross-paper numerical performance ranking or aggregate ensemble gain**.

The survey provides neither a controlled ablation of the five modules nor a matched-cost test showing that agent interaction beats a strong single predictor. Claims about role-based collaboration are conceptual explanations and summaries of cited work. They cannot be transferred as measured evidence that several compact RE models have complementary errors.

### Cost, limits, and thesis use

Section 5 (pp.30–33) discusses hallucination, coordination, scalability, benchmarks, communication overhead and the cost of larger-model calls. Adaptive allocation between cheaper and more powerful models is an opportunity discussed in the survey, rather than an evaluated router with a quantified saving. The document has no uniform measured latency, memory, energy or API-cost comparison. For an actual agentic extension, repeated calls, tool invocations, message tokens, failures and feedback loops would need to be counted.

**Supports:** concise terminology for agent workflows; distinguishing role diversity from parameter diversity; recognizing communication and auxiliary-call costs. **Does not establish:** NYT10m gains, a particular RE fusion rule, sub-1B feasibility, compression benefits or a measured routing advantage. **Placement:** a brief boundary paragraph in 2.4 and optional extensions in 2.7, with cost caveats in 2.6. The survey should not dominate the chapter's RE evidence.

## P05. DEFENSE: sequential encoder snapshots and scarce labels

**Citation key:** `folino2025defense`. **Source/version:** Gianluigi Folino, Massimo Guarascio, Luigi Pontieri and Paolo Zicari, *Discovering ensembles of small language models out of scarcely labelled data for fake news detection*, *Applied Soft Computing* 171 (2025), article 112794, DOI [10.1016/j.asoc.2025.112794](https://doi.org/10.1016/j.asoc.2025.112794). The local `Folino et al.pdf` is the sixteen-page publisher article, available online 28 January 2025. [Author's publication listing](https://gfolino.github.io/). All pages read, including the computational-cost appendix and references; Equation 2, algorithm and Tables 2-8/Figures 4-6 visually checked. **Evidence class:** adjacent binary news classification, encoder self-training/snapshot ensemble, with measured training time and approximate training FLOPs.

### Question and architecture

The question is whether limited genuine labels, unlabeled documents and a small encoder can produce reliable fake-news classifiers without expensive repeated full-corpus training. DEFENSE combines class-balanced pseudo-label selection, incremental fine-tuning and reliability-weighted aggregation of classifiers from successive self-training rounds. These are **related snapshots produced by one training trajectory**, not independent architecture experts, different precision copies or independently trained runs.

The backbone is distilled Small BERT with four hidden layers, hidden size 512 and eight attention heads, pretrained on Wikipedia and BookCorpus (Section 2.1, p.4). A single sigmoid-output dense classifier is added (Section 4.2, p.8). The paper does not provide an exact deployed parameter count; do not infer one from the name alone. An initial classifier is supervised on labeled data, followed by at most five configured pseudo-label rounds. Each round selects twenty additional documents, balanced across predicted fake/real labels, with uncertainty threshold 0.3. Fine-tuning uses AdamW, batch size 32, learning rate 3e-5 and at most thirty epochs with early stopping (Section 5.1, p.10). New rounds start from the preceding checkpoint.

Definition 1 (p.8) averages class probabilities with fixed member weights,

\[
\mathcal M(x)=\frac{\sum_i w_i M_i(x)}{\sum_i w_i}.
\]

DEFENSE-acc weights members by validation accuracy. DEFENSE-ps is intended to use pseudo-label reliability; its specification is inconsistent across prose and pseudocode, as detailed below. Both are static, input-independent rules. A single-model pseudo-label baseline retains only a self-trained classifier rather than aggregating the trajectory.

### Essential equations and source consistency

Equation 1 (p.9) defines the binary least-confidence uncertainty

\[
u(x)=0.5-|M(x)-0.5|=\min\{M(x),1-M(x)\},\quad u\in[0,0.5].
\]

The published Equation 2 **literally prints**

\[
\operatorname{cost}(u)=2(1-u).
\]

It therefore ranges over **[1,2] and decreases with uncertainty**, contradicting the stated [0,1] range and the prose that higher uncertainty increases the cost. This was checked on the rendered PDF, not inferred from text-extraction noise. Equation 3 (p.10) multiplies per-example binary cross-entropy by this cost before averaging the batch. Describe the intended uncertainty-aware training approach, but **do not silently replace the published formula with 2u or another conjecture**. Its actual implementation is not verified by the available paper.

There are related reproducibility issues: Section 4.1.1 says DEFENSE-ps uses one minus average pseudo-label uncertainty; Algorithm 1 line 23 instead averages one minus the stored *cost*, which, with Equation 2, would produce non-positive weights contrary to Definition 1. Section 4.2's final prose gives yet another description. The algorithm's member indexing and Definition 1's summation bounds also do not cleanly match the initial member plus newly trained members. Avoid claiming an unambiguous exact member count from the pseudocode. The thesis is not adopting this procedure, so these issues can be summarized as limits on reconstructing the algorithm rather than resolved by speculation.

Section 1.3 (p.3) states that BERT-base has 110M parameters and requires more than 10GB merely to store. The storage interpretation is incorrect: **110M x 4 bytes is approximately 0.44GB of raw FP32 weights**, an arithmetic estimate that excludes gradients, optimizer state, activations and runtime allocations. The thesis must distinguish raw weight storage from training/peak runtime memory. No measured BERT-base inference-memory baseline supports the article's claim.

### Task, data, comparisons and results

The unit is a full news document labeled fake or real; there is no entity-pair input, relation direction, multi-label ontology or RE NA class. The retrieved FakeNewsNet subsets contain 814 PolitiFact articles and 4,719 GossipCop articles (Table 2, p.11). The labeled fraction combines training and validation and varies from 2.5%-20% for PolitiFact and 1.25%-20% for GossipCop (Section 5.3, p.11). Exact held-out split counts and label-stratification details are not fully disclosed; do not reconstruct them from these fractions.

The strongest directly comparable single is supervised Small BERT with the same backbone, followed by the pure pseudo-label variant. Other competitors include semi-supervised temporal CNN/DEFD-SSL and text-only adaptations of originally multimodal classifiers. Some literature-baseline values are imported from earlier publications rather than all being rerun under one identical pipeline (Section 5.4, p.12).

Verified comparisons retain the paper's reported plus/minus dispersion; it is not labeled as a confidence interval:

| Setting | Supervised Small BERT F1 | Pure pseudo-label F1 | DEFENSE-acc F1 | Locator |
|---|---|---|---|---|
| PolitiFact, 2.5% labeled | 0.60 +/- 0.05 | 0.60 +/- 0.14 | 0.68 +/- 0.04 | Table 3, p.11 |
| PolitiFact, 20% labeled | 0.81 +/- 0.03 | 0.81 +/- 0.05 | 0.85 +/- 0.01 | Table 3, p.11 |
| GossipCop, 1.25% labeled | 0.62 +/- 0.16 | 0.62 +/- 0.14 | 0.67 +/- 0.02 | Table 4, p.12 |
| GossipCop, 20% labeled | 0.72 +/- 0.03 | 0.75 +/- 0.05 | 0.77 +/- 0.01 | Table 4, p.12 |

Section 5.1 states five runs; Section 5.3 uses Friedman and Nemenyi comparisons at 5% significance and says the two DEFENSE weighting strategies do not significantly differ. The results do not imply universal superiority of weighting by validation accuracy. Table 7's PolitiFact 5%-label sensitivity gives F1 0.75 for both k=10 and k=20, versus 0.67/0.69 at k=30/40 (p.13); adding more pseudo-labels can hurt. Table 8 gives GossipCop 0.76 at k=20, versus Table 4's 0.73 at the same nominal 5%-label/default-k setting. The source does not reconcile that discrepancy, and Section 5.5 reverses the dataset table references. Preserve the main-comparison values and explicitly flag the sensitivity discrepancy.

### Cost scope, limits and placement

Section 5.6 (p.13) measures **learning/self-training time**, on an RTX3080 10GB GPU with an i7-12700KF and 64GB system RAM. Figures 4-6 compare training time and F1. They do not establish lower ensemble inference latency, whole-system energy or peak memory. Appendix (p.14) approximates training as six trainable-parameter operations per token and derives

\[
6pet\left(sn_L+\frac{k s(s-1)}2\right),
\]

with linear-in-initial-labeled-token-size behavior under a bound on k and s (Property 1, p.10). This approximation abstracts away operations and the repeated inference scan of unlabeled data; it is not a measured whole-workflow FLOP or energy count. Its assumed nL bound is not reconciled with the smallest labeled fractions. Sequential warm starts and early stopping explain possible training savings, but every ensemble member still needs inference at deployment unless distilled.

**Supports:** cheap encoder specialization, explicit limits on pseudo-label rounds, comparing a snapshot ensemble against matching singles, and separating training from inference costs. **Does not establish:** entity-relation extraction, independent error diversity, a reproducible uncertainty-loss formula, pruning/quantization gains, or a superior NYT10m quality-cost frontier. **Placement:** 2.3 for compact pretrained encoders and adaptation; 2.4 for snapshot versus heterogeneous ensembles; 2.6 for cost accounting; 2.7 as adjacent evidence with bounded transfer.

## P06. Chan et al.: distributional ensembles and Sequential Monte Carlo

**Citation key:** `chan2026smc`. **Evidence class:** adjacent distributional theory and structured-generation experiments. **Source:** [arXiv:2603.05432v1](https://arxiv.org/abs/2603.05432v1), submitted 5 March 2026. All 29 supplied PDF pages were read, including Appendices A–G. The author list is Robin Shing Moon Chan, Tianyu Liu, Samuel Kiegeland, Clemente Pasti, **Jacob Hoover Vigly**, Timothy J. O'Donnell, Ryan Cotterell, and Tim Vieira. Vigly's first names were absent from the previous digest. The reviewed artifact and bibliography remain the preprint; a proceedings record was not independently established.

### Question, formal contribution, and assumptions

The paper studies how language-model distributions should be combined during generation, and whether better approximation of a globally defined ensemble improves task performance. Combining next-token scores and renormalising at every step generally produces a different string distribution from normalising aggregated probabilities of complete strings. The GPT-2 prompt-intersection illustration (§2, pp. 2–3) finds correlation 0.958 with an explicit intersected prompt for global product scoring, versus 0.090 locally; these concern 200 beam-generated completions, not an RE benchmark or a universal agreement theorem.

Definition 4.1 (p. 3) defines the ensemble distribution

\[
\Phi(x)=\frac{f(p_1(x),\ldots,p_K(x))}{Z},\qquad
Z=\sum_{x'}f(p_1(x'),\ldots,p_K(x')),
\]

with nonnegative potentials and a finite positive normaliser. Theorem 4.1/Appendix D.3 links weighted alpha-divergence minimisation to the generalised mean

\[
\Phi_\tau(x)\propto\left(\sum_k w_k p_k(x)^\tau\right)^{1/\tau},\qquad \tau=1-\alpha.
\]

The product is the limit at zero; mixture corresponds to one; minimum/maximum are limiting cases. For active experts with positive weights, nonpositive-power means concentrate on common support. Negative powers at zero are defined **by continuity**, setting the mean to zero if an active expert probability is zero (Appendix D.3, p. 18). Avoid evaluating a literal negative power of zero or applying an unqualified formula outside these support conditions. If overlap is empty, the required positive normaliser fails. A zero-weight expert should not constrain the active support. This is a statement about string distributions, not intersection/union of predicted RE label sets.

### Inference and the mixture-bound correction

Importance sampling weights complete outputs by unnormalised target/proposal; sequential importance sampling uses tractable prefix shaping; SMC resamples weighted partial strings (Algorithm 1, p. 5). The locally optimal proposal is proportional to the next-symbol shaping function (Proposition 5.2). Appendix D.1 requires **r(x)=0 implies phi(x)=0**, so the proposal covers target support. Footnote 6 on p. 4 prints a tautological relation involving phi and Phi rather than this proposal condition; the appendix supplies the intended requirement. Annihilativity of the generalised means makes the proposed shaping compatible with absolute continuity (§D.2–D.3). The estimated normaliser is unbiased; self-normalised distribution estimates have finite-particle bias and consistency under the stated assumptions. “Consistent” does not mean exact at ten particles.

Byte-level inference maps tokenised models into a common decoded-string probability space, marginalising over token sequences that decode to the same bytes (Appendix C, pp. 13–14). It therefore addresses tokenizer mismatch at substantial cost. It is unnecessary for already normalised encoder relation IDs.

For normalised expert distributions and fixed convex weights, expected correctness of a **global mixture** is the weighted mean of expert expected correctness and cannot exceed the highest expert expectation. Equation 14a–b establishes this by linearity. However, the supplied v1's Eq. 14c visibly prints `max_k(w_k E_k)`, an invalid bound for the preceding weighted sum. The correct upper bound is `max_k E_k`; this correction is a mathematical inference, not a quotation of the printed line. Neither bound covers majority voting, argmax decoding, locally renormalised token mixtures, nonlinear aggregation, or learned input-dependent selection. The notes must not extend this theorem to the thesis's static multilabel vote.

### Experimental contract and results

Section 6.1 (pp. 6–7) uses instruction-tuned Llama-3.1-8B, Qwen2.5-7B, and Phi-4 14B, without new task training. Two experts combine either two prompts of one checkpoint or two checkpoints under one prompt. Appendix F gives prompts; some include demonstrations, so these experiments are not uniformly zero-shot. One hundred sampled examples per task evaluate JSON-schema conformance, BBH word sorting, and SPIDER SQL execution correctness. There are no entity spans, directional relations, RE ontology, NA bags, or multilabel facts in these tasks.

The outcome is **expected accuracy**: probability mass assigned to correct strings, estimated from weighted particles (Eq. 13). It is not single greedy-output accuracy or RE F1. Baselines include the strongest base prompt/model, local probability averaging, and local/global alternatives. Five sampling seeds, 95% confidence intervals, ten default particles, equal expert weights, and effective-sample-size resampling threshold 0.9 are reported. Bold results indicate nonoverlapping intervals, not a paired hypothesis test. Prompt and best-operator comparisons warrant selection caveats; these intervals do not cover training variance or the whole dataset population.

Table 4 (Appendix G.1, p. 23) reports Phi JSON expected accuracy 83.5±1.9 for its stronger prompt, 82.0±4.7 for local probability averaging, 87.4±0.3 for token-SMC minimum, and 86.4±1.0 for product. Table 2 (p. 7) gives best cross-model byte-SMC accuracy 41.1±1.0 for Phi+Llama on word sorting and 55.5±1.1 for Qwen+Llama on SQL. JSON cross-model 92.6±0.9 is below Qwen's strongest single prompt 95.6±0.8: ensemble benefit is not universal.

Appendix G.2/Figure 4 (p. 25) finds gains chiefly where both prompt distributions assign moderate correct-answer mass, rather than where one dominates. This is distributional complementarity, not a count of exclusive hard-label recoveries. Particle ablations (Figures 5–7, pp. 26–27) show diminishing approximation improvements around 10–25 particles. Table 6 reports average **per-example Spearman correlations** with standard errors; scatter plots report different, aggregated **Pearson** coefficients. Product's word-sorting Llama correlation is 0.28±0.04. More accurate approximation helps consensus operators in several settings, but not all models/tasks; more computation is not a guaranteed accuracy improvement.

### Measured cost, limitations, and chapter use

Appendix E (pp. 18–19) serves bfloat16 models with vLLM on two RTX 4090 24GB GPUs and runs the SMC loop on an RTX 3090 24GB GPU, at temperature one. Table 3 measures average JSON seconds per instance: Llama single 0.76, local ensemble 2.34, token-SMC ten particles 15.24, byte-SMC ten particles 63.45. Phi's corresponding byte cost is 166.76 seconds. These are measured hardware-specific runtimes; no joules, monetary spend, peak memory, complete training cost, CPU portability, or matched wall-time budget benefit is demonstrated. Appendix B names compute overhead, only two experts, 7–14B models, and structured-task restrictions.

**Supports:** separating post-output voting from global decoding ensembles; identifying operator/objective/support assumptions; examining conditional complementarity; including inference approximation overhead in cost.

**Does not establish:** a hard-voting accuracy ceiling; sub-1B or NYT10m gains; calibrated encoder probabilities; compressed-variant diversity; cheap CPU deployment; universal superiority of consensus rules.

**Chapter mapping:** mechanism taxonomy and conditional complementarity in 2.4; measured inference-overhead example in 2.6; task/scale positioning in 2.7. It supplies no direct DS/manual benchmark evidence for 2.2, no encoder RE experiment for 2.3, and no weight compression experiment for 2.5.

## P07. Zhang et al.: selective document-pair refinement and probability fusion

**Citation key:** `zhang2025refiner`. **Source:** Fu Zhang, Xinlong Jin, Jingwei Cheng, Hongsen Yu and Huangming Xu, *Rethinking the Role of LLMs for Document-level Relation Extraction: a Refiner with Task Distribution and Probability Fusion*, NAACL 2025, pp.6293–6312, [primary paper](https://aclanthology.org/2025.naacl-long.319/), [DOI](https://doi.org/10.18653/v1/2025.naacl-long.319). All 20 supplied pages, including Appendices A–G, were read. **Evidence class:** other RE; hybrid document-level routing and score fusion.

### Question, method, and task contract

The study asks how a generative LLM can refine a trained document extractor while controlling false positives and multilabel decoding. Experiments use DocRED/Re-DocRED with supplied entities/coreference and ordered candidate entity pairs. A pair may have several labels, while an NA score provides the extractor's decision threshold. This is full-document RE, not NYT10m pair bags.

ATLOP, Eider, DREEAM and AA are specialized encoder baselines. The LLM stages use unfine-tuned Llama3-8B and, in some analyses, GPT-3.5-turbo-0613. “No additional fine-tuning” applies to the LLM refinement stage: the encoders themselves are task-trained. The system is not wholly sub-1B and the paper does not provide an exact total parameter count for every encoder configuration.

Task distribution selects difficult **entity-pair decisions**, particularly those with a leading NA score close to the best positive score. Equation 7 (p.6296) defines

\[
\gamma(h,t)=\frac{P_{\rm slm}(NA)-\max_rP_{\rm slm}(r)}{P_{\rm slm}(NA)},\qquad \gamma(h,t)\le\delta.
\]

The LLM receives the document and a multiple-choice prompt built from the encoder's top-k relation candidates. First-output-token probabilities for option tokens provide LLM scores. Equations 8–9 temperature-scale the vocabulary distribution and constrain the difference between the dispersion of the selected encoder and LLM score vectors. Equations 10–11 add aligned scores and keep relations above the fused NA score. A second true/false LLM query verifies each retained relation. This is **pair-level selective refinement plus score fusion and verification**, not a router choosing an independently optimal model for every relation label. Matching score dispersion is not an established probability-calibration procedure.

### Results, ablations, and uncertainty

Table 4 (p.6299) gives Re-DocRED test F1 **80.73→81.69** for DREEAM with Refiner and **81.20→82.03** for AA. DREEAM's DocRED test F1 is **65.30→65.82**. These modest increments over strong encoders are distinct from the abstract's much larger comparison with weak LLM-only methods. Baseline cells are mainly imported from original papers; dagger-marked cells are reproduced. All numbers are on the 0–100 F1 scale.

Table 5's Re-DocRED test ablation reports F1 81.69 for the complete DREEAM system, 81.48 without self-verification, 80.01 with simple score addition, 79.24 without probability fusion and 46.78 without task distribution. Removing task distribution changes which pairs reach the LLM, so this large loss should not be read as the isolated causal value of a general router. Table 6 (p.6300) reports top-k candidate hit rates; these are candidate coverage, not final extraction recall. Figures 6–8 show non-monotonic effects of temperature and difficulty threshold. More referrals recover more correct pairs **and introduce more errors**.

F1 and IgnF1 are both reported. IgnF1 adjusts for benchmark training-fact overlap; it is not an evaluation restricted to provably unseen semantic relations and does not remove all pretraining contamination. No repeated-seed intervals or paired significance test establish the small encoder-refinement increments. Appendix F (pp.6305–6306) describes LLM answers based on internal knowledge when the document does not support the relation. This reinforces the distinction between world truth and extraction correctness.

### Cost and accounting correction

Table 7 (p.6300) reports average inference time across the two document datasets and specified templates; Appendix D/Table 12 (p.6305) separately reports API cost **$100.91→$4.96** with task distribution, and Llama3-8B time **28.69→1.21 hours**. These are reported experiment costs, not current prices or local laptop projections. They do not imply a comparable reduction in the already cheap encoder's full-pipeline cost.

Appendix D/Table 11 adds encoder **training hours** to LLM **inference hours** and calls the latter's share an inference proportion. That mixed denominator is unsuitable as a deployment overhead estimate. Training, selector/encoder inference, referred LLM calls, and verification should be separated. No complete peak-memory, energy or matched-budget Pareto comparison is supplied; A100 40GB is the reported Llama environment.

**Supports:** NA prevalence as a routing concern, explicit recovery/error accounting, score-fusion requirements and careful selective-call cost reporting. **Does not establish:** a static compact ensemble gain, per-label adaptive model selection, calibrated scores or NYT10m transfer. **Placement:** 2.1, 2.2, 2.4, 2.6; briefly as a routing extension in 2.7.

## P08. Chanthran et al.: document zero-shot RE with side information

**Citation key:** `chanthran-etal-2026-document`. **Evidence class:** adjacent document-level RE, including Malaysian English news; no model ensemble. **Source:** [EACL paper and record](https://aclanthology.org/2026.eacl-long.216/), EACL 2026 Volume 1 Long Papers, pp. 4670–4680. All 11 PDF pages, including Appendices A–D, were read. Formula, result, role-template, and sentence-gap pages were visually inspected. The official BibTeX record renders author Ong Huey Fang as `Huey Fang, Ong`; this spelling/order is retained explicitly.

### Question and actual pipeline

DocZSRE-SI asks whether side information can replace full-document encoding when assigning previously unseen relation labels. Its contrast is with methods that generate synthetic labelled examples for unseen labels. It **does not eliminate LLM use**: GPT-4o-mini generates entity descriptions from document context, and then hypernyms from mention, entity type, and description (§§3.1.1–3.1.2, p. 3/4672; acknowledged again in §8). Appendix A supplies a description example; it is contextual information generation, not a demonstrated evidence-verification guarantee.

`bert-base-uncased` encodes concatenated head/tail descriptions, each entity's hypernym/type, role templates, context template, and relation-label names (§3.2, pp. 3–5). The method uses the named pretrained encoder; no target-task BERT fine-tuning is described. GPT-4o-mini's parameter count is not disclosed here, so the complete pipeline must not be described as a wholly sub-1B model system. Entity mentions/types are treated as available side-information inputs in the experiments; an end-to-end NER evaluation is not reported.

### Scoring, direction, and contract limitations

The chosen label maximises a weighted combination of cosine similarities. Equation 2 (p. 5/4674) gives description similarity weight 0.4 and weights of 0.1 for each of two hypernym similarities, two type similarities, a role score, and a context score, multiplied by a consistency-based confidence factor. The authors tested description weights 0.2/0.4/0.6, and §7 says fixed coefficients were selected through preliminary validation experiments rather than separately for every domain. This is a hand-set multi-feature score, not voting among independent language models or an established calibrated probability.

There are reproducibility ambiguities. Equation 1 averages items labelled role-based **embeddings**, whereas the surrounding prose discusses averaging similarity scores. The confidence factor is described in terms of mean similarity and standard deviation, but an exact algebraic function, scaling, and the stated number of contributing measures are not specified consistently. Appendix B (p. 10/4679) labels **both** head and tail as acting as a subject, despite §3.2.1 motivating distinct subject/object roles. Preserve these as source issues; do not silently rewrite the tail template as object or invent a confidence formula.

The method returns the highest-scoring unseen label for a queried pair. No explicit multilabel decoding, NA acceptance threshold, or abstention algorithm is specified in §3.2.4. Consequently its reported macro-F1 must not be equated to the thesis's exhaustive 24-label set prediction across heavily negative bags. The paper does not clearly document how negatives participate in each unseen-label evaluation.

### Data, results, and uncertainty

The datasets are coherent MEN news articles and Wikipedia-derived DocRED/Re-DocRED, not NYT10m entity-pair bags. Unseen label sets of five, ten, or fifteen are randomly selected three times; macro-F1 is averaged (§4.1). This label-level zero-shot condition differs from no NYT10m task fine-tuning of a released extractor. Section 4.2 says a DocRED ablation uses 20% of documents, given as 21,577, but does not clearly identify which DocRED component that large count belongs to. Do not substitute the familiar human-annotated split size or treat this as a settled count of human-labelled documents.

Table 1 (p. 6/4675) provides the clearest feature ablation. For five unseen labels, description-only macro-F1 is 28.14 on Re-DocRED, 36.34 on DocRED, and 27.65 on MEN; full scoring obtains 50.05, 48.83, and 40.25. Hypernyms outperform coarse entity types; adding all unweighted information can add noise. For ten unseen labels, the full method obtains 44.98/43.43/36.02; for fifteen, 33.91/32.55/24.28. The paper labels the accompanying ± quantities “variance”; their definition should not be silently changed to standard deviation, standard error, or confidence interval. Three random label sets do not establish training-seed robustness or population significance.

Table 2 (p. 8/4677) compares published GenRDK results with the proposed method. On Re-DocRED test, five-label macro-F1 is 47.81 versus 41.3; ten-label F1 is 43.06 versus 30.1. Those are gains of **6.51 and 12.96 percentage points**, with relative increases about 15.76% and 43.05%. DocRED comparisons concern development data because its test labels are blind. Do not convert the abstract's aggregate “11.6% improvement” into a universal effect. Baseline numbers are taken from another paper, not a demonstrated matched-cost rerun.

Appendix D/Table 3 (p. 11/4680) associates large sentence gaps with worse performance, but the narrative overstates monotonicity. MEN five-label accuracy is **50.65% at gap zero and 56.69% at gap one**; MEN ten-label accuracy rises from 26.26% at gap two to 37.58% at gap three. Thus neither “gap zero is always highest” nor “every increase lowers accuracy” is supported. The drop to 16.93% at gap ≥5 for MEN's five-label condition is a useful limited observation.

### Cost, limits, and thesis mapping

The paper supplies no measured full-pipeline latency, memory, energy, API expenditure, or matched-budget comparison. The full cost includes descriptions, hypernyms, and repeated BERT computations. “Low complexity” is a design claim; §7 acknowledges computational burden, similar-label confusion, generic entity types, and manual coefficient choices.

**Supports:** article/document versus bag boundaries; label-level zero-shot definitions; context compression with side information; separate entity roles and distance-sensitive analysis; careful accounting of auxiliary LLM stages.

**Does not establish:** LLM-free or fully sub-1B inference; complete text-grounding of generated descriptions; correct NYT10m multilabel/NA handling; ensemble or compression gains; a measured efficiency advantage.

**Chapter mapping:** adjacent 2.1 and 2.3; feature-score fusion distinction in 2.4; missing cost boundary in 2.6; task-boundary positioning in 2.7. It is not a direct manual DS-evaluation source for 2.2 or a compression source for 2.5.

## P09. Tang et al.: collaboration in classical Chinese joint extraction

**Citation key:** `tang2026collaboration`. **Source:** Xuemei Tang, Linxu Wang and Jun Wang, *Language model collaboration for relation extraction from classical Chinese historical documents*, *Information Processing & Management* 63(1), article 104286 (2026), online 1 August 2025, [DOI:10.1016/j.ipm.2025.104286](https://doi.org/10.1016/j.ipm.2025.104286). All 26 supplied pages, including appendices, were read; main and ablation tables were visually checked. **Evidence class:** other RE; joint entity/relation extraction, encoder–LLM collaboration.

### Question, method, data, and adaptation

SLCoLM asks whether a specialized model can guide a larger decoder on rare relations in classical Chinese. ChisRE contains six entity and 40 relation types; its RE component has approximately 4.1K samples and 9.7K triples. The narrative reports 22.3K characters while Table 3 reports 141.0K, an unresolved dataset-count discrepancy. The study uses an 8:1:1 train/validation/test split (§5.1, p.11).

SpERT jointly predicts entities and triples with BERT-guwen, a backbone further pretrained on classical Chinese, then fully task-fine-tuned. Candidate relation definitions are selected using similar training examples and trigger-word heuristics. An LLM sees the sample, SpERT predictions/probabilities, task instructions, relevant definitions and optionally two demonstrations, then revises/supplements triples (§4, pp.7–10). The comparison includes GPT-3.5, ERNIE-4.0, DeepSeek-V3, Qwen-max and a LoRA-tuned Alpaca2-7B baseline. The article does not provide complete exact parameter counts for proprietary models or every deployed encoder head; the collaboration is not a sub-1B-only system.

The four merge modes must be distinguished. Mode 1 takes a union of triples. Mode 2 keeps high-scoring SpERT triples and combines them with LLM triples. Mode 3 replaces SpERT outputs for its K weakest relation types with LLM outputs of those types. Mode 4 retains SpERT triples for types it predicts and adds LLM triples only for other types. If W denotes the chosen weak types, Mode 3 is

\[
\widehat T=\{t\in T_S:r(t)\notin W\}\cup\{t\in T_L:r(t)\in W\}.
\]

It is **relation-type replacement**, not voting and not an unrestricted union. It requires performance estimates for choosing W; a thesis implementation must obtain these on validation. The setting extracts typed entities and directed triples from samples, rather than accepting a fixed pair and predicting its NYT10m label set. Several triples per input are possible; an empty triple set is absence, while a detailed independent NA classifier is not the proposed main mechanism.

### Results, ablations, and interpretation

Table 6 (p.11) reports SpERT micro-F1 **64.92** and macro-F1 **43.19**. Qwen-max with ICL+SLCoLM reaches **64.84 micro-F1 / 53.93 macro-F1**; ERNIE-4.0 reaches **64.11 / 51.40**. Every collaboration row **in this main table** has lower micro-F1 than SpERT while some improve macro-F1 substantially. The positive finding is better rare-class balance with a tradeoff in aggregate precision/quality, not unqualified superiority over the strongest single model.

Table 7 (p.12) ablates raw LLM outputs **before merging**: GPT-3.5 micro-F1 18.79 with demonstrations rises to 61.30 when SpERT predictions are supplied and 61.39 with candidate definitions. This identifies a strong effect of the supplied specialized predictions in that prompting pipeline, not the final merge's independent effect. Tables 8–10 compare demonstration retrieval, definition selection and merge modes. Table 8 includes an ERNIE random-demonstration micro-F1 of 64.96, so the main-table loss is not a universal claim across every configuration. Section 5.7 and Figure 8 examine per-relation improvements; Section 5.8/Figure 9 addresses long-tail relations. Threshold h and replacement-count K sweeps are shown in Appendix D. Main tables do not report training-seed intervals or formal significance tests; use the observed tradeoffs without upgrading them to established population effects.

### Cost, limits, and thesis use

Restricting definitions can shorten prompts, but there is no measured end-to-end latency, GPU energy, peak-memory or complete API-bill comparison. Task fine-tuning, retrieval/heuristics, prompt assembly, encoder and LLM calls, definition writing and configuration search all contribute. Historical-domain language, joint entity detection, ontology size and decoder scale differ from the thesis. A probability used to filter SpERT triples is not reported as calibrated.

**Supports:** RE-specific context collaboration; relation-dependent complementarity; separating macro gains, micro losses and extra errors; choosing between replacement and union. **Does not establish:** improvements from a static sub-1B vote, NYT10m transfer, model-independent thresholds or measured net savings. **Placement:** 2.1, 2.3, 2.4 and 2.6; short ICL extension in 2.7.

## P10. Lu et al.: whole-model precision-ensemble voting

**Citation key:** `lu2025mixedprecision`. **Source/version:** Guanxi Lu, Hao Mark Chen, Zhiqiang Que, Wayne Luk and Hongxiang Fan, *Enhancing Trustworthiness with Mixed Precision: Benchmarks, Opportunities, and Challenges*, arXiv:2511.22483v1, 27 November 2025. [Version record](https://arxiv.org/abs/2511.22483v1); [full primary HTML](https://arxiv.org/html/2511.22483v1). The local `Guanxi Lu et al.pdf` has seven pages; all pages read and Figure 3 inspected visually. The official [ASP-DAC 2026 program](https://www.aspdac.com/aspdac2026/pdf/ASP-DAC_2026_Full_Program.pdf), printed p.41, also lists the same authors at a talk with a slightly changed title, starting proceedings page716. The present analysis cites the inspected **2025 v1 preprint**, rather than merging unverified final-proceedings metadata into it. **Evidence class:** adjacent decoder trustworthiness/general classification; empirical same-backbone precision ensemble, not RE.

### Question and controlled variants

The paper asks whether weight quantization can change trustworthiness despite preserving general accuracy, and whether voting among differently quantized copies can reduce failures. The dense baselines are LLaMA-2-Chat 7B and 13B. The 13B checkpoint is quantized with AWQ or GPTQ at three, four and eight weight bits (Section III.A, p.3). The ensemble experiment combines three precision variants within each quantization framework, rather than mixing independently pretrained checkpoints (Section IV.C, p.5). The experimental scale is **13B**, not sub-1B. No new target-task fine-tuning is reported; the exact calibration data, checkpoint conversion settings, generation settings and independently repeated evaluation seeds are not fully disclosed.

The measured mechanism is **output voting across whole-model precision variants**. It is not assigning different bit widths to different layers in a single model, token-probability fusion, progressive bit-width scheduling during generation, or a pruning experiment. Those belong to the background or future-opportunity discussion. Quantization reduces representation precision and packed storage; it does not reduce the number of learned parameters or guarantee independent errors between copies.

### Procedure and task contract

Section IV.B and Algorithm 1 (p.5) define quantization, independent response generation, output-to-label mapping, refusal/invalid-output filtering and unweighted majority voting. The prose gives a highest-precision tie-break. If all candidates are filtered, the algorithm returns REFUSED. For ordinary classification refusals are removed from the candidate vote; in out-of-scope tasks a refusal is retained as a legitimate label. In explanatory notation,

\[
C'(x)=\{h(g_i(x)):h(g_i(x))\text{ is valid under the task-specific filter}\},\quad
\hat y=\operatorname{mode}C'(x),
\]

with the task-specific all-invalid/tie behavior above. Output mapping can use heuristics or an LLM judge; a paid judge would add cost, but the paper does not itemize such calls. A neutral or absent answer is different from an RE NA label. The distinction matters: for a given-entity RE task, *no supported relation* is a valid semantic prediction and should not be removed as an invalid refusal.

Section III.B (p.3) evaluates MMLU's 57 subjects and DecodingTrust-based adversarial robustness, fairness, machine ethics and out-of-distribution robustness. Adversarial tasks use AdvGLUE/AdvGLUE++ SST-2, QQP and MNLI; fairness uses Adult demographic parity and equalized-odds disparities; machine ethics uses 2,109 ETHICS samples with zero/few-shot and adversarial wording; OOD includes style changes and knowledge-out-of-scope queries. These are classifications or composite benchmark scores, without supplied entity spans, directed relation ontology, bags or multi-label facts. Fairness is excluded from precision-ensemble evaluation because it measures group disparity rather than directly comparable prediction accuracy (Section IV.C.1, p.5). Do not report four evaluated ensemble trustworthiness dimensions when only three are compared.

### Verified empirical interpretation and limitations

Figures 1 (p.4) and 3 (p.6), together with Sections III.C-IV.C, support qualitative findings: eight-bit variants generally preserve the dense baseline; lower-bit AWQ is more stable than lower-bit GPTQ in these models/settings; MMLU behavior does not fully predict trustworthiness behavior; filtered voting often improves over a compressed variant. The dense 7B baseline remains stronger than the ensemble on some robustness dimensions, demonstrating that more parameters and more member calls need not be preferable.

**Figure 3 does not support universal gains over the strongest individual precision copy.** In particular, the AWQ machine-ethics voting bar is visibly below AWQ-best. The caption's broad improvement language and the surrounding text should be qualified by the per-method/per-dimension bars. There are no printed exact numeric result tables, confidence intervals, independent repetitions or controlled filtering-only-versus-voting-only ablations. Filtering and aggregation are changed together, so their separate contributions are not identified. The paper's headline maximum improvement is not retained in the quantitative comparison because the exact plotted values and whether the gain is relative percent or score points are not disclosed clearly enough for reconstruction. Similarly, explanatory claims about sensitivity to lexical cues or damaged calibration are hypotheses; calibration itself is not evaluated.

### AWQ and GPTQ: checked background, not an encoder recipe

**AWQ (`lin2024awq`).** [The MLSys 2024 primary record](https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html) describes weight-only PTQ: calibration activations identify important channels, and channel scaling reduces quantization error without backpropagation or reconstruction training. Its TinyChat acceleration is a separate kernel/runtime contribution. Activations inform weight quantization; this does not mean activations themselves are stored at the same low precision. Lu's framework description is consistent with this distinction. AWQ's reported GPU advantages do not establish compatibility or speed for the thesis's DeBERTa/GLiNER encoder runtime.

**GPTQ (`frantar2023gptq`).** [The ICLR 2023 author's v2 paper](https://arxiv.org/html/2210.17323v2), Sections 3-4, minimizes layer-output reconstruction error, \(\|WX-\widehat W X\|^2\), using approximate second-order information and blockwise error compensation after rounding. It is weight-only PTQ, not target-task retraining or activation quantization. Section 6 attributes measured speed gains to reduced memory movement and notes hardware/operand limitations; lower bits alone do not guarantee faster arithmetic. An Appendix comparison includes BERT-base on SQuAD, an adjacent encoder result; it does not validate GLiNER-Relex or NYT10m.

### Cost scope and thesis use

Lu does not report measured generation latency, peak memory, joules, money, conversion time or a cost-quality frontier. Figure 2 depicts parallel generation, but that is a proposed execution structure, not evidence of affordable concurrency on a 16GB laptop. Three members require three forward-generation workloads; sequential execution adds work, while concurrent execution also raises resident-memory demands. Any optional judge, conversion/calibration and loading costs must be accounted for separately. Section V.C (p.6) explicitly leaves efficient system/hardware co-design as a research opportunity.

**Supports:** testing compressed copies separately, checking task-specific behavioral changes, and a same-backbone precision-voting hypothesis. **Does not establish:** encoder compatibility, effective layerwise mixed precision, net inference savings, independence of variants, reliable gains over the best copy on every metric, pruning benefits or NYT10m improvement. **Placement:** 2.4 for precision-copy member construction; 2.5 for PTQ, tensors and distinct meanings of mixed precision; 2.6 for missing system-level cost evidence; 2.7 to position the local RE study as an empirical test rather than a presumed transfer of success.

## P11. Gao et al.: Manual Evaluation Matters

**Citation key:** `gao-etal-2021-manual`. **Evidence class:** direct revised NYT benchmark and manual-evaluation evidence. **Source:** [published paper and record](https://aclanthology.org/2021.findings-acl.112/), Findings of ACL-IJCNLP 2021, pp. 1306–1318. The supplied PDF is arXiv:2105.09543v1; its entire 13 pages were read. The published PDF was additionally checked for its author order and the main construction/evaluation/results passages. The local preprint lists Qiu before Bai, whereas the published PDF and ACL record list Bai before Qiu; the bibliography uses the published order.

### Question, contribution, and task contract

The paper asks whether DS-RE rankings remain trustworthy when automatically assigned knowledge-graph facts are replaced by text-grounded human annotations. Distant supervision can attach a true fact to text that does not express it, and incomplete knowledge graphs can miss facts the text does express. The contribution is a common manually labelled test resource, plus systematic comparisons of neural encoders, training units, and aggregation strategies. It is not an ensemble of independently deployed small models.

The input contains sentences and identified entity pairs. A bag groups mentions of the same pair; the prediction unit is a relational fact comprising a pair and a relation. Multiple relations can hold for a sentence or bag. Head and tail are relation arguments, so directional ontology labels must retain their ordering. Entity detection is not the evaluated task. Section 4.1 and Figure 2 (PDF p. 4/printed p. 1309) define the multi-instance, multi-label setting. Section 5.2 (p. 6/1311) constructs bag gold as the union of all sentence annotations in the bag; N/A applies only if that union contains no positive relation.

### Dataset and annotation evidence

The revised ontology has 25 labels including N/A, rather than the original 53. Twelve country-specific region-capital labels are merged, three organisation-location labels are merged, and relations absent from one split are removed (§3.1, pp. 3–4; Appendix A/Table A.1, p. 12). Table 1 reports 417,893 training sentences and 46,422 validation sentences. Its 17,137 training and 4,062 validation **facts** are positive relational facts, not total bags.

The authors annotate all originally positive test sentences and the 5,000 highest-scoring originally N/A sentences identified by a fine-tuned BERT model. Four annotators independently decide which relations are expressed without seeing suggested DS or model labels. A relation receives a positive annotation when more than half vote for it; the authors adjudicate conflicts. The resulting resource has 9,744 sentence units, 5,174 pairs, 3,899 verified positive facts, and 32% N/A sentences (§3.1; Table 1). The model-selected N/A component is not a uniform prevalence sample. In the checked set, DS fact precision is 69.1%, fact recall 33.9%, and exact pair-label accuracy 47.1% (p. 4).

The thesis repository's **previously saved public aggregate audit**, rather than a new test read, records 11,086 released rows and 9,744 unique sentence–pair units in `results/preliminary_experiments.json`. Both the supplied preprint and published Table 1 report 9,744. Distinguish release rows, sentence–pair units, bags, and positive facts explicitly. The matching unique-unit count supports an accounting explanation; it does not reconstruct the release's serialization history or justify silently substituting one count for another.

### Models, results, and interpretation

Sections 4.2–5.1 (p. 5) compare CNN, PCNN, and pretrained `bert-base-uncased`, with sentence-level or bag-level DS training. AVG combines sentence representations; ONE takes each relation's maximum sentence score; ATT weights sentence representations by relation-conditioned attention. RL-DSRE, BGWA, and RESIDE are additional trained baselines. These are sentence-evidence aggregation mechanisms within models, distinct from the thesis's output combination across checkpoints. BERT uses batch size 64, sampled training bag size four, learning rate 2e-5, and three epochs. Sampling addresses training memory; **all sentences are used at evaluation**. Table 3's training comparison does not validate a 20-sentence inference cap.

Table 4 (p. 6; visually verified) demonstrates protocol-sensitive rankings: RESIDE obtains held-out micro-F1 40.5% versus PCNN+bag+ATT's 39.1%, but manual bag micro-F1 43.3% versus 56.5%. BERT+sent+ONE obtains manual bag AUC 61.3%, micro-F1 62.9%, and macro-F1 36.1%; BERT+bag+ATT obtains 51.2%, 54.1%, and 25.8%. Thus attention's benefit with PCNN does not automatically transfer to BERT. Table 5's Wiki20 comparison is supporting evidence on a different ontology and N/A construction, not a second NYT10m score.

Section 5.3 (p. 7/1312) explicitly treats a real birthplace fact as a false positive when the sentence only places a story in that city. The explanation involving pretrained knowledge and entity-name shortcuts is a hypothesis supported by masking diagnostics, not a proof of the internal mechanism. BERT-M masks entity mentions during training and inference, improving precision near the top of the P–R curve while sacrificing coverage (Figure 4). Figure 5 (p. 8) shows severe relation imbalance and zero-F1 rare relations, motivating macro and per-relation reporting.

### Evaluation, cost, and limits

Section 5.2 selects the **best micro-F1 on each test P–R curve** and computes macro-F1 at that threshold. This is a published oracle operating-point convention, not the thesis's deployable development-selected threshold policy. AUC requires ranked scores. The paper reports no seed variation, confidence intervals, or formal paired significance test for Table 4. It reports training settings and a memory-motivated sampling design, but no measured deployment latency, peak memory, energy, API spend, or multi-model cost frontier.

**Supports:** text-supported manual gold; separate DS and manual references; sentence-to-bag union; maximum-score aggregation as an established bag mechanism; careful NA semantics and per-relation evaluation.

**Does not establish:** correctness of arbitrary recalled world facts; a universal rule for motivated pragmatic inference; frozen released encoders' expected scores; a cap-20 inference policy; benefits or affordability of a sub-1B static ensemble or reduced variants.

**Chapter mapping:** primary evidence for 2.1 and 2.2; historical encoder evidence for 2.3; aggregation-versus-ensemble distinction in 2.4; metric/protocol boundaries in 2.6; direct benchmark anchor and remaining cost gap in 2.7. No quantization/pruning experiment for 2.5.

## P12. Han et al.: OpenNRE task and toolkit boundaries

**Citation key:** `han-etal-2019-opennre`. **Evidence class:** direct task/toolkit background; original DS NYT10 experiments. **Source:** [ACL paper and bibliographic record](https://aclanthology.org/D19-3029/), EMNLP-IJCNLP 2019 System Demonstrations, pp. 169–174. All six PDF pages were read, including references. The ACL record spells the fourth author's name Deming Ye; the supplied PDF renders Demin Ye. The bibliography follows the official record and the difference is documented.

### Question and contribution

OpenNRE addresses the practical difficulty of implementing, training, deploying, and comparing neural RE methods in a common framework. Its contribution is a modular toolkit and online demonstration, rather than a new SLM ensemble or an efficiency benchmark. The five components are tokenization, reusable neural modules, encoders, models, and scenario-specific training/evaluation frameworks (§3, pp. 3–4/171–172; Figure 2). Encoders include convolutional, recurrent, and pretrained BERT representations; model modules include attention, adversarial training, and reinforcement-learning methods. TensorFlow/PyTorch and GPU support are infrastructure features, not measured proof of low deployment cost.

### RE settings and input assumptions

Section 2.1 (p. 2/170) explicitly distinguishes normal RE, where entity mentions are already annotated, from optional NER and entity linking. Therefore using supplied entity spans in the thesis is a recognised RE setup; NER need not be repeated simply because the toolkit includes it.

Sentence-level RE classifies a relation between two annotated mentions in one sentence (§2.2). Bag-level RE groups multiple sentences mentioning the same pair to mitigate wrong DS labels: some mentions may express the relation even when others do not (§2.3, pp. 2–3). Document-level RE can require evidence across several sentences and several entities in one coherent document (§2.4). Figure 1 illustrates a sibling relation requiring family context, rather than independent pair co-mentions. These settings must not be collapsed: NYT10m bags need not preserve a whole article's discourse or all supporting intermediate entities. Few-shot RE concerns learning a relation from a small support set (§2.5), not necessarily prompt-based in-context learning in a frozen generative model.

The paper is descriptive about pair relations but does not state one universal NA/multi-label output contract across all toolkit scenarios. Sentence and few-shot benchmarks generally evaluate class predictions; the bag setting evaluates extracted facts. The thesis's ordered-pair, 24-positive-label set and empty-set NA convention must therefore be specified through the benchmark and its own protocol, rather than attributed wholesale to this toolkit paper.

### Experiments and competitive comparisons

Sections 4.1–4.3 (pp. 4–5) evaluate implemented models on separate tasks. Wiki80 has 80 relations and 56,000 Wikipedia/Wikidata instances; its reported scores use the validation set because it was not an official benchmark. SemEval 2010 Task-8 has 19 relation labels and 10,717 instances, with 17.4% no-relation examples. BERT inputs use entity markers: the plain BERT variant takes the CLS representation, whereas BERT-Entity takes entity-start representations following Soares et al. Table 1 reports accuracy of 63.93%, 84.57%, and 86.61% on Wiki80 for CNN, BERT, and BERT-Entity. Table 2 separately reports SemEval micro-F1 0.880 and 0.883, with the original BERT-Entity result 0.892. Do not interchange these accuracy and F1 tables.

For **original distantly supervised NYT10**, CNN-ATT, CNN-ADV, and CNN-RL use a common CNN encoder; ADV and RL are based on instance attention. Table 4 (p. 5/173) reports AUC/F1 0.333/0.397, 0.337/0.406, and 0.276/0.429. The original CNN-ATT comparison is 0.318/0.380. These establish replication quality in the earlier DS setting. They are **not** results on Gao's later revised 25-label manually annotated NYT10m test.

FewRel's five-way one-shot/five-shot accuracies in Table 3 are 74.5/88.4 for Prototype-CNN, 80.7/89.6 for Prototype-BERT, and 88.3/93.2 for BERT-PAIR. BERT-PAIR scores whether a query and supporting sentence share a relation; this is another trained matching architecture, not evidence that a static encoder vote improves NYT10m.

### Cost, uncertainty, and thesis use

No measured inference times, parameter-count comparison, memory/energy/API costs, repeated-run confidence intervals, or significance tests are supplied. No document-level benchmark results are reported, despite the toolkit's conceptual support for that scenario. Its qualitative claims of efficiency and extensibility should not be converted into quantitative resource conclusions or claims of universal CPU feasibility.

**Supports:** relation extraction with supplied entities; sentence/bag/document/few-shot distinctions; modular separation of encoding, evidence aggregation, classification, and evaluation; historical trained encoder baselines.

**Does not establish:** manual NYT10m performance; current released-model training provenance; sub-1B deployment cost; ensemble complementarity; compression gains; routing or ICL benefits.

**Chapter mapping:** foundational 2.1, supporting DS background for 2.2, encoder vocabulary for 2.3, the aggregation-versus-checkpoint-ensemble boundary in 2.4, and historical comparison limits in 2.7. Absence of cost/compression measurements is relevant to 2.6/2.5, not positive evidence for either.

## P13. Christou and Tsoumakas: specialization, scale, and matched prompting

**Citation key:** `christou2026subbillion`. **Source:** Despina Christou and Grigorios Tsoumakas, *Sub-Billion, Super-Frontier: Small Language Models Rival Zero-Shot Frontier LLMs on General and Literary Relation Extraction*, [arXiv:2606.22606v1](https://arxiv.org/abs/2606.22606v1), 21 June 2026, 41 pages including appendices. **Evidence class:** other RE; supervised specialization and API comparisons. This replaces the previous incorrect Christou citation.

### Question, method, and evaluation contract

Can task-specialized compact decoders rival general frontier APIs, and how much do domain mixture and demonstration format matter? Five decoder backbones are trained with QLoRA, using NF4 base weights, floating-point computation and LoRA adapters. Three mixtures cover general, literary, or balanced general/literary training. Zero-shot and two-shot formats are applied at both training and inference, producing 30 configurations. General training is capped at 200,000 instances; balancing is between domains, rather than equalizing every constituent dataset (§§3–4; Appendices A–B).

This is supplied-entity-pair, **single-label** classification over sentences or passages. Bag datasets and document datasets are transformed into pair-classification instances. Seven general datasets are TACRED, SemEval, CoNLL04, **NYT11**, GIDS, ReDocRED and REBEL; Biographical and PG-Fiction are literary datasets. It is not the revised NYT10m bag-level multilabel evaluation. Providing entity names/mentions in the generative template also differs from joint entity detection. Label-enumeration instructions are advisory: generation is not schema-constrained. Invalid labels are penalized by exact matching. Negative-class cases contribute false positives/negatives to positive-class micro-F1; an average across datasets is an unweighted average of their F1 scores, not pooled micro-F1 (§5; Tables 17–20).

Table 21 (p.39) gives actual sizes of 362M for SmolLM2-360M and 494M for Qwen2.5-0.5B; the other decoders are approximately 3B. RoBERTa-base/large are 125M/355M comparisons. Two epochs, seed 42 and final checkpoints are used; the broad grid does not supply repeated-training-seed evidence. LoRA settings depend on model scale. NF4 is part of training, rather than a controlled comparison of compression methods.

### Results, ablations, uncertainty, and provenance

Table 4 (p.15) reports general-average F1 **0.828** for Qwen2.5-0.5B General-Tuned/two-shot, versus **0.693** for GPT-5.4 and **0.662** for Claude Sonnet 4.6, which are evaluated **zero-shot**. The Qwen NYT11 score is 0.792. Llama3.2-3B General-Tuned/two-shot reaches 0.844; its zero-shot-tuned/zero-shot-evaluated row reaches 0.821 and provides the demonstration-matched reference. The two-shot compact-model headline therefore differs from the APIs in both supervision and demonstrations. The main-text description of a common zero-shot protocol must be qualified by the explicit row conditions and caption. These results support specialization under the specified comparison; they do not compare equal training budgets or establish universal sub-1B superiority.

The encoder control is particularly relevant: Table 11 (p.22) gives RoBERTa-base general-average F1 0.826, comparable to the selected Qwen result. The “best SLM” average 0.853 selects a different winner per dataset: it is an **oracle envelope**, not a deployable single checkpoint or an evaluated ensemble. CoNLL04 reaches 1.000 under a supplied-entity-type lookup; this reflects an unusually informative task schema, not a general RE reasoning ceiling.

Table 7 (p.17) separates training-format and inference-demonstration effects using

\[
F_{2,2}-F_{0,0}=(F_{2,0}-F_{0,0})+(F_{2,2}-F_{2,0}).
\]

For the reported sub-1B aggregate, the scores are 0.647, 0.373 and 0.778. Two-shot training therefore creates a substantial dependence on matching inference demonstrations; the matched gain is 0.132, although changing training alone reduces this aggregate by 0.274. This is evidence about prompt-format interaction, not probability calibration. Appendix D/Table 25 (p.41) compares generic and label-enumerating prompts across 164 retained matched cases: average 0.798 versus 0.766. The paired sampling analysis does not replace training-seed replication. Literary domain-adaptive pretraining produces only a small average change, 0.826 to 0.827 in Table 8 (p.20).

The validity discussion matters as much as the headline. Appendix A reports approximately 23% exact train/test overlap in Biographical. PG-Fiction is synthetically labelled; overlap with LitBank used in domain adaptation and ontology inconsistencies complicate interpretation. General and literary specialists are evaluated within their domains. Table 13's bootstrap intervals condition on the fitted systems. API runs use default providers/settings rather than a pinned hardware/effort comparison. These are limits on generalization and comparability, not reasons to discard the entire study.

### Cost and thesis use

The paper reports approximately **600 GPU-hours** for the tuning grid and approximately **50 GPU-hours** for domain-adaptive pretraining on an RTX4090. Table 14's roughly 22ms GPU and 180ms CPU figures for Qwen are **estimates** for a short specified prompt/completion, not a measured NYT10m end-to-end benchmark. Table 21 distinguishes measured BF16 model/adapter storage from estimated NF4/GGUF footprints and runtime memory. Quality is measured with NF4; Q4 GGUF quality is not demonstrated by the estimated CPU footprint. No full monetary or energy comparison is supplied.

**Supports:** a sub-1B operational scale, strong specialized single-model baselines, encoder controls, matched input/prompt requirements, and provenance audits. **Does not establish:** NYT10m bag performance, ensemble complementarity, benefits of precision variants, or a measured local quality–cost advantage. A local adapter converting generated pair labels to bag-level predictions is a thesis adaptation and must be evaluated independently. Training overlap must be audited before using released task-tuned adapters. **Placement:** 2.3, 2.6 and 2.7; brief ICL discussion in 2.7.

## P14. GLiNER-Relex: joint entity and relation representations

**Citation key:** `stepanov2026glinerrelex`. **Source:** Ihor Stepanov, Oleksandr Lukashov, Mykhailo Shtopko and Vivek Kalyanarangan, *GLiNER-Relex: A Unified Framework for Joint Named Entity Recognition and Relation Extraction*, [arXiv:2605.10108v1](https://arxiv.org/abs/2605.10108v1), 11 May 2026, 19 pages. **Evidence class:** other RE; open-schema encoder extraction.

### Question, architecture, and adaptation

The framework asks whether one encoder can jointly identify entities and directed relations specified through textual labels. A bidirectional DeBERTa-v3 encoder contextualizes entity labels marked with `[ENT]`, relation labels marked with `[REL]`, and the input text together. Span representations combine boundary information and width, followed by a BiLSTM and task heads. Ordered head/tail span representations are concatenated and passed through a pair MLP. In schematic notation for Equations 7–8,

\[
z_{h,t}=\operatorname{MLP}([s_h;s_t]),\qquad
p(r\mid h,t,x)=\sigma(z_{h,t}^{\top}u_r).
\]

Independent relation scores permit several labels above a threshold. Direction is encoded by the ordered pair; swapping head and tail changes the representation. No supported positive label can be represented by an empty set rather than a separate generated refusal.

The framework discusses several adjacency decoders, but the released large configuration forms all ordered pairs of detected entities and has zero adjacency-loss weight. Their separate effectiveness is not evaluated. Likewise, optional knowledge-graph scoring modules should not be attributed to the reported main model. Equation 9 supplies focal loss for entity/relation heads; the released settings use gamma zero and alpha 0.75, reducing this component to weighted cross-entropy. This is score learning, not a reported calibration evaluation.

Sections 3.8–3.9 describe synthetic supervision from one million FineWeb sentences and 50,000 full-text samples annotated by Qwen3-32B, followed by 3,000 entity-rich examples produced and corrected using Gemini. The detailed schedule applies to the **large** configuration, with one first-stage epoch and five second-stage epochs. It should not be copied as a verified recipe for every released base checkpoint. The paper does not give a sufficiently explicit deployed parameter-count breakdown for every variant; model size requires a checkpoint-specific count, including added task modules.

### Evaluation, results, and source conflict

The target experiments are zero-shot joint extraction on CoNLL04, DocRED, FewRel and CrossRE. Entity detection participates in the GLiNER-Relex metric. GLiREL's comparison uses gold entities, introducing an input-contract difference. These are not supplied-pair NYT10m bag experiments. The PDF does not establish a complete source/target overlap audit or a training-inventory guarantee for released variants.

**Table 2 (p.12, visually checked)** reports micro-F1 percentages 40.4/31.3/12.5/18.1 for GLiNER-Relex, averaging 25.6. GPT-5-mini is 42.4/18.6/15.0/12.4, averaging 22.1. GLiNER2 is 32.9/11.7/20.8/6.0, averaging 17.8. The surrounding prose gives different GLiNER2 values and also differs for some GLiREL cells. The review retains **table values with this conflict explicitly marked**, rather than treating them as independently resolved ground truth. There are no reported error bars or controlled ablations isolating multitask training, synthetic-data stages, pair enumeration, or label conditioning.

Section 5.1 (p.13) measures average latency on 50 FineWeb documents, approximately 288 words each, with six entity labels and 50 relation labels. GLiNER-Relex takes approximately **0.9 seconds** per document on an NVIDIA L4; GPT-5-mini takes approximately **64 seconds**, including API/reasoning behavior. The roughly 70-fold ratio is specific to that pipeline and workload. It is not a matched quality–cost result on the four target datasets, a CPU measurement, or a full preparation/training/API-bill comparison.

### Thesis relevance and local boundary

The thesis uses supplied NYT10m entity spans, preserves head-to-tail ordering, scores labels for those pairs, and aggregates sentence evidence into bags. Bypassing detection changes the published contract and can remove detection errors; bag aggregation introduces another choice. Therefore the published joint-extraction scores cannot be assigned to the local adaptation. This review adds no local performance estimate. Released-score thresholds also require validation under the new ontology and negative prevalence.

**Supports:** encoder-based open-schema RE, joint versus supplied-entity distinctions, ordered-pair scoring, and reporting actual workload-specific latency. **Does not establish:** NYT10m effectiveness, identical base/large training, calibrated confidence, safe ontology transfer, or useful ensemble gains. **Placement:** 2.1, 2.3, 2.6 and 2.7.

## P15. GLiDRE v2: dual-encoder document relation extraction

**Citation key:** `armingaud2025glidre`. **Source:** Robin Armingaud and Romaric Besançon, *GLiDRE: Generalist Lightweight model for Document-level Relation Extraction*, [arXiv:2508.00757v2](https://arxiv.org/abs/2508.00757v2), revised 7 October 2025, [full v2 HTML](https://arxiv.org/html/2508.00757v2), 15 pages. The review and bibliography use this version; an accepted proceedings venue is not established by the inspected record. **Evidence class:** other RE; document-level encoder model. Author/title metadata come from this primary record, not Christou's inconsistent secondary reference.

### Question, method, task, and scale

GLiDRE asks whether label-conditioned extraction can support supervised, low-resource, few-shot and zero-shot DocRE without a large generative decoder. A DeBERTa-v3-large text encoder and BGE-large-en-v1.5 label encoder form a dual-encoder system, approximately **800M parameters in total** (§4.2, p.5). Counting only the document encoder would misstate the deployed system's scale. Labels can be encoded separately and cached; that benefit is architectural rather than a universal latency guarantee.

Gold entity mentions and coreference chains are supplied. Word representations pool into mentions, mentions into entities, and ordered entity representations feed a pair network. A label embedding is obtained by mean pooling its token representations. Equations 1–4 can be summarized as

\[
z_{h,t}=\operatorname{FFN}([e_h;e_t]),\qquad
p(r\mid h,t,D)=\sigma(z_{h,t}^{\top}u_r),\quad
\widehat R_{h,t}=\{r:p(r\mid h,t,D)>0.5\}.
\]

Focal loss trains independent positive-label decisions. Optional localized context pooling combines head/tail attention to gather document context (Equations 5–8). Ordered pairs support direction and multiple simultaneous labels; no positive prediction corresponds to no extracted relation. The model evaluates a coherent document with coreference, unlike a set of NYT10m pair co-mentions. Its encoder still has a token limit, and enumerating ordered candidate pairs scales quadratically in entity count.

Section 3.5 describes 136,404 synthetic FineWeb documents annotated by Mistral-Small-24B, with 76,497 verbalized relation labels. Synthetic pretraining and subsequent supervised adaptation are distinct sources of competence. Low-resource experiments use five shared sampled training subsets for each size and report means with standard deviations. Episodic support-set fine-tuning is parameter adaptation, **not frozen-model ICL**.

### Results, ablations, and version-specific conditions

Table 1 (p.6) gives low-resource Re-DocRED F1 **41.73 ± 2.94** at ten training documents, versus DREEAM **27.07 ± 5.81** and ATLOP **29.48 ± 3.91**. At 500 documents, DREEAM's 69.32 is slightly above GLiDRE's 69.26. Thus gains depend on the data regime. Table 2's FREDo/ReFREDo values are episodic **macro-F1**, with support adaptation, rather than comparable full-supervision micro-F1.

Table 3 (p.7) reports fully supervised Re-DocRED test micro-F1 **77.83 ± 0.23**, against ATLOP **77.81 ± 0.71** and DREEAM **80.20 ± 0.45**. GLiDRE's IgnF1 is **76.80 ± 0.22**. IgnF1 adjusts for training-set fact overlap under the benchmark convention; it does not isolate reasoning on genuinely unseen facts or correct all pretraining contamination.

Table 4 (p.8) evaluates a different, zero-shot regime: GLiDRE obtains **17.32** micro-F1, compared with Mistral-Large-123B 18.61, Qwen2.5-72B 18.00 and Llama3.3-70B 15.81. The 17.32 and 77.83 results concern different adaptation conditions; a released pretraining-only checkpoint must not inherit the supervised score.

Appendix A.3 (pp.13–14) reports 77.83 ± 0.23 with mean entity pooling versus 76.86 ± 0.15 with log-sum-exp pooling; removing pretraining gives 77.15 ± 0.42, and removing localized context pooling gives 77.61 ± 0.09. Adaptive-threshold loss gives 75.74 ± 1.05 versus focal loss 77.15 ± 0.42 **in the no-pretraining comparison**. This does not show that adaptive thresholds universally underperform. Dispersion across five seeds is useful but not a formal matched significance test.

### Cost and thesis boundary

Sections 4.2–4.3 report approximately 24 hours of synthetic pretraining and 3.5 hours of supervised fine-tuning on an H100, separate from teacher annotation cost. Episodic evaluation adds repeated support-set tuning. Section 4.3.4 (p.8) measures processing 500 documents: GLiDRE takes approximately **100 seconds on one A100 80GB**, using less than 10GB VRAM; Mistral-Large takes approximately **600 seconds on four A100s**, requiring over 300GB total VRAM. The approximately 24-fold GPU-time comparison multiplies time by device count, not wall time alone. The different hardware allocations, absent energy/API bills and preparation costs constrain interpretation. No Apple/CPU measurement is given.

**Supports:** dual-encoder label generalization, sub-1B parameter accounting, explicit adaptation regimes, low-resource uncertainty and hardware-specific efficiency measurement. **Does not establish:** NYT10m bag performance, universal superiority over supervised encoders, zero-shot performance equal to supervised performance, compressed-variant diversity or local ensemble affordability. Supplied-pair/bag adaptation and locally prepared checkpoints require independent evaluation; this review reports no new local result. **Placement:** 2.1, 2.3, 2.6 and 2.7.

## P16. Q8BERT: quantization-aware encoder adaptation

**Citation key:** `zafrir2019q8bert`. **Source:** [arXiv:1910.06188v2](https://arxiv.org/abs/1910.06188v2), five pages, read in full; published in the 2019 Fifth Workshop on Energy Efficient Machine Learning and Cognitive Computing—NeurIPS edition, pp. 36–39, [DOI](https://doi.org/10.1109/EMC2-NIPS53020.2019.00016). Locators below refer to the inspected preprint. **Evidence class:** other NLP tasks; encoder compression.

### Research question and mechanism

Can BERT retain downstream quality when most weights and matrix-multiplication activations use eight-bit integers? The contribution is a task-fine-tuning procedure with fake quantization, compared with ordinary floating-point fine-tuning followed by dynamic quantization. This is useful evidence about the importance of adaptation to quantization error, rather than a relation-extraction experiment.

Sections 2–3 define symmetric quantization as \(q=\operatorname{clip}(\operatorname{round}(xS),-M,M)\), with \(M=2^{b-1}-1\), and approximate reconstruction \(\hat x=q/S\). At eight bits, \(M=127\). Weight scales use the largest absolute weight; activation scales during training use an exponential moving average of the maximum activation magnitude. The dynamic comparison computes activation scales at inference. Quantization-aware training (QAT) keeps full-precision master weights, simulates rounding/clipping in the forward pass, and uses a straight-through estimator for the rounding gradient. The model therefore learns under quantization noise during supervised task adaptation.

The tensor contract matters: fully connected and embedding weights are Int8, biases are Int32, and LayerNorm, softmax and GELU remain floating point (§3). Over 99% of weights are quantized. This is not an integer-only implementation, and eight-bit weight storage does not mean every intermediate tensor or arithmetic operation is eight-bit. QAT also cannot be described as a frozen-checkpoint post-processing operation.

### Experiments and verified results

BERT-base and BERT-large are evaluated on seven GLUE tasks and SQuAD v1.1, with additional large-model rows for selected tasks. The paper describes their sizes as 110M and 334M parameters respectively (§3); do not silently standardize its large-model count to a different paper's convention. Evaluation uses public development data and reports means and standard deviations over five runs (Table 1, p. 3). Baselines are task-fine-tuned floating point (FP) and dynamic quantization (DQ), rather than modern encoder RE models.

Representative results demonstrate both preservation and residual losses:

| Task / metric (0–100 scale) | FP | QAT | DQ | Locator |
|---|---:|---:|---:|---|
| CoLA / Matthews correlation | 58.48 ± 1.54 | 58.48 ± 1.32 | 56.74 ± 0.61 | Table 1, p. 3 |
| MRPC / F1, BERT-base | 90.00 ± 0.23 | 89.56 ± 0.18 | 87.88 ± 2.03 | Table 1, p. 3 |
| RTE / accuracy | 69.70 ± 1.50 | 68.78 ± 3.52 | 63.32 ± 4.58 | Table 1, p. 3 |
| SQuAD v1.1 / F1 | 88.46 ± 0.15 | 87.74 ± 0.15 | 80.02 ± 2.38 | Table 1, p. 3 |

These are task-dependent scores, not a single universal “accuracy.” QAT largely closes the dynamic-quantization gap in this setup; the RTE spread and SQuAD loss prevent a blanket claim of lossless compression. Table 2 (p. 4) summarizes relative changes, but the thesis should prefer the directly reported scores or explicitly computed percentage-point differences. The experiment comparing FP, DQ and QAT is the central adaptation ablation; it does not disentangle every tensor's contribution or compare contemporary calibration methods.

### Cost, limits, and thesis use

The approximately fourfold weight-storage reduction follows the 32-to-8-bit representation; it is not a measurement of peak application memory. No end-to-end model latency, energy, total training bill, or ensemble cost is benchmarked. The cited matrix-multiplication acceleration example is evidence from another implementation, not a Q8BERT hardware experiment. Calibration/training and runtime support must be included in any local proposal.

**Supports:** the PTQ/QAT distinction, quantized tensor accounting, and the need to evaluate task quality after compression. **Does not establish:** NYT10m transfer, zero-shot GLiNER preservation, faster execution on Apple hardware, or useful diversity between precision copies. **Placement:** 2.3 for encoder specialization; 2.5 for eight-bit QAT versus dynamic quantization; 2.6 for the distinction between storage and measured latency.

## P17. Outlier Suppression: making low-bit encoder quantization viable

**Citation key:** `wei2022outlier`. **Source:** [NeurIPS 2022 record and main paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/6f6db140de9c9f111b12ef8a216320a9-Abstract-Conference.html); thirteen-page main paper and [ten-page supplement](https://proceedings.neurips.cc/paper_files/paper/2022/file/6f6db140de9c9f111b12ef8a216320a9-Supplemental-Conference.pdf). Supplement printed pages continue at 14–23. **Evidence class:** other NLP tasks; encoder/encoder–decoder compression. Authors are Xiuying Wei, Yunchen Zhang, Xiangguo Zhang, Ruihao Gong, Shanghang Zhang, Qi Zhang, Fengwei Yu and Xianglong Liu.

### Question, method, and essential equations

Why do conventional quantizers degrade language-model quality sharply at low activation precision, and can the problematic ranges be reduced without changing the full-precision function? Sections 3–4 identify structured activation outliers, especially in LayerNorm outputs and GELU, and distinguish outlier magnitude from task importance. LayerNorm's learned scaling parameter \(\gamma\) amplifies particular embedding dimensions; a few frequent/special tokens exhibit especially large ranges. The observations motivate two complementary interventions rather than indiscriminate clipping.

Uniform affine quantization uses \(q=\operatorname{clip}(\operatorname{round}(x/s)+z,0,2^b-1)\), reconstructed as \(s(q-z)\). Gamma Migration factors the scale out of LayerNorm and absorbs it into following linear transformations: \(W(\gamma\odot x)=(W\operatorname{diag}(\gamma))x\). Residual branches must be handled as well. The supplement's Appendix A proves the transformation and illustrates the different attention/FFN paths. It is an equivalent reparameterization before quantization, not deleting an allegedly unimportant learned parameter.

Token-Wise Clipping first builds per-token extrema, searches quantiles of these ranges, and then refines step sizes to minimize final-output reconstruction error, \(\|\hat f(s)-f\|_F^2\) (§4.2). It therefore searches from the token perspective and optimizes downstream error, unlike minimizing each activation's local rounding error alone. PTQ uses a small calibration set without supervised task-weight retraining. The QAT variant combines the method with learned-step-size training, and some reported settings additionally use knowledge distillation (KD). These variants have different preparation costs and must remain separate.

### Protocol, results, and ablations

The paper evaluates BERT and RoBERTa encoders and BART encoder–decoder models on GLUE, SQuAD v1.1/v2.0, CNN/DailyMail and XSum. The notation W–E–A denotes weight, embedding and activation bits. Supplement B/F describes symmetric per-channel weights and asymmetric per-layer activations, with embeddings and matrix-multiplication inputs quantized. Matching activation nodes is part of a fair comparison: a method that leaves a troublesome branch in FP is not equivalent to quantizing it.

PTQ calibration uses 256 examples; fine-grained scale optimization uses three epochs (Supplement F, printed p. 22). Baselines include MinMax, OMSE, Percentile, EasyQuant and PEG for PTQ, and PACT/LSQ+ for QAT. Main Table 4 (p. 8) reports the paper's aggregate GLUE score for BERT: FP 83.83, eight-bit method 83.96, six-bit method 81.19, versus six-bit OMSE 73.52. This aggregate mixes task-specific metrics; it is neither RE F1 nor uniform accuracy. Main Table 5 (p. 9), at 4–4–4 bits, gives LSQ+ 71.49, the method 81.13, and method+KD 83.56 against FP 83.83. Near-FP quality at four bits must therefore carry the KD qualifier.

The component ablation is particularly informative: RoBERTa six-bit QNLI accuracy rises from MinMax 62.13 to 78.56 with Gamma Migration, 79.66 with Token-Wise Clipping, and 86.82 with both; FP is 92.68 (Table 3, p. 8). Both interventions contribute, but this setting retains a substantial gap. Supplement B.2 disables selected quantization nodes and shows why LayerNorm/GELU deserve attention; Supplement D compares coarse clipping against other calibration searches. The main checklist explicitly reports no error bars, so small improvements over FP should not be called significant.

### Cost and bounded transfer

There is measured **preparation** cost: Supplement Table 15 (printed p. 22) reports 135.73 seconds for the coarse clipping search on 256 samples versus 439.29 seconds for OMSE golden-section search and 1,754 seconds for its grid search. This is activation-calibration timing, not end-to-end inference latency, and the table does not supply a portable local-machine guarantee. Supplement Table 17 (p. 23) gives BERT GLUE model size 417.6 MB at FP versus 104.8 MB at 8–8–8, and 52.6 MB at 4–4–4. Storage does not include the full running application's activations and buffers. QAT also incurs task training, hyperparameter trials and optional teacher computation.

**Supports:** encoder-specific evidence that activation outliers matter; careful tensor/bit definitions; calibration and KD accounting. **Does not establish:** that generic AWQ/GPTQ weight-only recipes preserve a DeBERTa RE system, that all four-bit encoder results are lossless, or that reparameterization alone accelerates an ensemble. **Placement:** 2.5 for PTQ/QAT and outlier handling; 2.6 for preparation-versus-inference cost.

## P18. Movement Pruning: task-adaptive unstructured sparsity

**Citation key:** `sanh2020movement`. **Source:** [NeurIPS 2020 record](https://proceedings.neurips.cc/paper/2020/hash/eae15aabaa768ae4a5993a8a4f4fa6e4-Abstract.html), twelve-page main paper and two-page supplementary appendix (printed pp. 13–14). **Evidence class:** other NLP tasks; encoder pruning.

### Question and mechanism

Can pruning exploit the movement of pretrained weights during task fine-tuning instead of ranking their inherited absolute magnitudes? This targets the transfer-learning regime, where a large pretrained weight need not be important for the new task. Each weight has a separately learned importance score. The masked linear layer is \(a=(W\odot M)x\), with hard masks \(M=\operatorname{Top}_v(S)\) retaining the top fraction of signed scores. The straight-through score gradient is \(\partial L/\partial S_{ij}=(\partial L/\partial a_i)W_{ij}x_j\) (Eqs. 1–2). Accumulated negative gradients give high scores to connections moving away from zero during adaptation; this is first-order information, distinct from magnitude pruning.

Soft Movement Pruning uses a threshold mask and a regularizer on sigmoid scores, avoiding a fixed per-matrix top fraction. A gradual schedule and final cooldown allow recovery as connections are removed. Optional output-distribution distillation combines teacher and supervised losses. This is **unstructured** weight pruning: it produces sparse matrices rather than physically removing full Transformer layers or heads.

Supplement A.1 presents a local, first-order loss-decrease argument under smoothness and small learning-rate assumptions; it is not a generalization theorem, a global convergence result or an accuracy guarantee. Signed importance is essential to that argument: applying absolute values to the scores invalidates the stated reasoning.

### Experiments, results, and scope

BERT-base is task-fine-tuned on SQuAD v1.1, MNLI and QQP, with frozen embeddings. The reported percentages refer to the roughly 84–85M **encoder weights**, excluding embeddings; the three-percent model is not a three-percent copy of every parameter in the complete checkpoint (§5, Table 2). All methods receive the same task-specific update budget, using six to ten epochs. Experiments run on one 16GB V100 (Supplement A.2). Comparisons include gradual magnitude pruning, unstructured L0 regularization, movement variants, and published RPP, LayerDrop and mini-BERT results; not all baselines have identical implementation provenance.

At three percent remaining encoder weights, SQuAD development F1 is 54.5 for magnitude pruning, 73.3 for L0, 76.3 for hard movement and 79.9 for soft movement, against dense BERT 88.1 (Table 2, p. 6). With teacher distillation the soft variant reaches 82.3 at three percent and 84.9 at ten percent (Table 3, p. 7). These demonstrate an advantage at high sparsity, not no accuracy loss. Low-sparsity comparisons reverse part of the ranking: retaining over 70% of weights favors magnitude pruning (Figure 2). The main sensitivity analyses compare local/global masks and layer allocations (Figures 4–6, pp. 7–8); global selection offers only limited further gains in most settings. The paper provides no multi-seed uncertainty intervals for the principal score tables.

### Cost interpretation and thesis use

The paper explicitly warns that standard PyTorch stores matrices with zeros and does not yield significant inference speedup (Supplement A.3). Efficient sparse serialization and a low nonzero-weight count are different from smaller dense tensors, lower peak memory and lower wall time. The distillation stage is “free” only in the final pruned architecture, not in preparation: training scores, recovery fine-tuning and teacher forward passes consume compute. Complete training time, joules, and an end-to-end quality–cost frontier are not supplied.

**Supports:** task-aware first-order pruning, adaptation costs, and the distinction between sparsity and acceleration. **Does not establish:** a useful frozen zero-shot RE pruning recipe, fast local sparse kernels, or complementary ensemble errors. **Placement:** 2.5 for unstructured pruning; 2.6 for requiring measured system performance.

## P19. CoFi: structured pruning with measured latency

**Citation key:** `xia-etal-2022-structured`. **Source:** [ACL 2022 paper and metadata](https://aclanthology.org/2022.acl-long.107/), pp. 1513–1528; sixteen pages including Appendices A–J. Authors: Mengzhou Xia, Zexuan Zhong and Danqi Chen. **Evidence class:** other NLP tasks; encoder pruning and distillation.

### Question, contribution, and method

Can task-specific pruning achieve the large speedups associated with compact distilled architectures while avoiding their expensive general-distillation phase? CoFi jointly learns masks at five granularities: MHA layers, FFN layers, attention heads, FFN intermediate dimensions and shared hidden dimensions. A parameter survives only when its relevant coarse and fine masks survive (§3.1, pp. 1516–1517). Explicit layer masks make it easier to remove whole computation blocks than waiting for every constituent fine unit to vanish.

Masks are sampled from hard-concrete distributions and trained under a Lagrangian target-size constraint (Appendix B, p. 1525). Prediction distillation uses the paper's stated \(D_{KL}(p_s\|p_t)\) objective. Dynamic layer distillation selects the surviving student layer closest to each selected teacher layer and minimizes \(L_{layer}=\sum_{i\in T}\operatorname{MSE}(W_{layer}H_s^{m(i)},H_t^i)\). The mapping \(m\) changes with the pruned architecture; \(L_{distil}=\lambda L_{pred}+(1-\lambda)L_{layer}\) (§3.2). Unlike training a random small student on a vast unlabeled corpus, this procedure starts from a task-fine-tuned teacher and uses task data. It nevertheless requires teacher availability, mask search and further fine-tuning.

### Evaluation and verified results

BERT-base is the principal teacher; Appendix I probes RoBERTa. Eight GLUE tasks and SQuAD v1.1 are evaluated on development splits. Embeddings are frozen and excluded from the paper's parameter/sparsity denominator (§4.1, p. 1517). Thus “5M parameters” refers to the non-embedding component rather than the complete deployment model. Task data and augmentation conditions are matched where possible; some additional baselines use different teachers and are explicitly marked as indirect comparisons.

Table 1 (p. 1513) gives MNLI accuracy 84.8 for dense BERT versus 80.6 for a 4.4M non-embedding CoFi model, with measured 12.1× speedup. Table 2 (p. 1518) compares high-compression CoFi and TinyBERT4 without task augmentation: MNLI 80.6 versus 78.8, but QNLI 86.1 versus 86.7. SQuAD F1 is 82.6 versus 82.1 at 8.7× speedup for both. Therefore the abstract's “over 10×” result is setting-specific, not universal across tasks. Metrics differ: CoLA uses Matthews correlation, STS-B Spearman correlation, SQuAD F1 and the other listed GLUE tasks accuracy (Appendix D).

The ablations show why structure and recovery matter. Removing layer masks reduces high-sparsity MNLI speedup from 12.1× to 8.4× while leaving rounded accuracy at 80.6; removing layer and hidden masks yields 7.0× and 78.4 (Table 4, p. 1520). Without both distillation objectives, high-sparsity SQuAD falls from 82.6 to 75.8 F1 (Table 5). Dynamic layer distillation generally helps, but SST-2 at 95% sparsity is an explicit exception (Table 11, p. 1528). Three repetitions characterize learned structures in §4.4/Table 6; the primary accuracy tables do not report comparable seed uncertainty intervals for every result.

### Measured versus estimated cost

All inference speedups are measured relative to dense BERT on one NVIDIA V100, batch size 128, sequence length 128 for GLUE and 384 for SQuAD (§4.1, p. 1518). This is concrete GPU evidence, not a CPU batch-one result. CoFi preparation is reported as at most 20 GPU-hours for the task-specific procedure. TinyBERT's approximately 350-GPU-hour comparison includes an **estimated** general-distillation component: Appendix J scales a timed 10.6M-token sample to the 2,500M-token original corpus. Teacher pretraining and every development trial are not included in that simplified total. Appendix A also describes final-subnetwork recovery fine-tuning, which cannot be omitted when reproducing the procedure.

**Supports:** the distinction between structured and unstructured pruning; hardware-controlled latency comparisons; explicit recovery/distillation cost. **Does not establish:** tenfold acceleration of GLiNER or GLiDRE on a Mac, a pretrained-model drop-in pruning transformation, or ensemble Pareto gains. **Placement:** 2.5 and 2.6; discuss it as a relevant encoder precedent whose transfer still requires measurement.

## P20. Mintz et al.: distant supervision and the evaluation target

**Citation key:** `mintz-etal-2009-distant`. **Source:** [ACL-IJCNLP 2009 paper](https://aclanthology.org/P09-1113/), pp. 1003–1011, nine pages read in full. The ACL author record expands the PDF's “Dan Jurafsky” to Daniel Jurafsky. **Evidence class:** other RE; foundational distant supervision.

### Question and method

Can a knowledge base provide supervision for relation extraction without manually labeled training sentences? The authors align Freebase entity pairs with Wikipedia co-mentions, extract lexical and dependency features, and train a regularized multiclass logistic classifier. Relations are ordered binary predicates; features record entity order and entity types (§3–5). For a pair occurring repeatedly, sentence features are combined into one pair-level representation. In explanatory notation, \(p(r\mid h,t,B)=\operatorname{softmax}_r(w_r^\top\phi(B))\), where \(\phi(B)\) aggregates features from co-mention sentences; this restates the architecture, rather than reproducing a numbered paper equation.

The original DS intuition assumes co-mentions can express the database relation. It permits very large noisy training sets, but a real-world relation between the pair need not be expressed in every sentence. Conversely, an absent KB edge need not mean the pair is unrelated. Section 6.3 explicitly acknowledges incorrectly omitted relations among negative examples. Later bag-level multi-instance methods weaken the sentence assumption; those should not be retroactively described as Mintz's exact algorithm.

### Data, evaluation, results, and ablation

The experiment retains 102 relations and uses 800,000 Wikipedia articles for training and 400,000 different articles for testing (§6, pp. 1008–1009). It recognizes/chunks entities with a four-class Stanford NER tagger and dependency-parses text with MINIPAR; it is not a supplied-span encoder experiment. The classifier outputs one relation or an unrelated class for a pair. Held-out KB evaluation removes half the relation instances from training; human evaluation instead trains with all retained instances and checks newly discovered pairs. These are distinct supervision/evaluation conditions, not interchangeable splits.

Figure 2 (p. 1009) compares lexical, syntactic and combined features under automatic held-out evaluation. Table 5 (p. 1010) reports human precision for stratified samples of the top 100 and 1,000 predictions **per relation**, over ten frequent relations. At the former cutoff the combined-feature mean is 0.69 versus 0.67 syntactic and 0.66 lexical; at the latter, 0.67 versus 0.68 and 0.67. No feature family wins every relation/cutoff. The abstract's 67.6% precision is a ranked-discovery summary; it must not be relabeled micro-F1 on a fixed balanced test set. Human labels use one to three workers, majority vote and random tie resolution. No uncertainty interval accompanies the principal precision table.

The human protocol asks whether the relation holds between the entities (§7), which aligns more closely with **fact discovery** than a strict sentence-support benchmark. Gao's later manual reannotation instead asks what the text expresses. This difference is directly relevant to the professor's nationality example: a true KB/world relation can be correct for fact discovery and still unsupported as textual extraction. The choice of target must therefore precede changing judgments. The thesis's agreed textual-support target, including justified linguistic/contextual inference, is a local operational policy; it is not uniquely mandated by all historical RE papers.

### Cost, limitations, and placement

Parsing/NER and large feature aggregation are necessary preparation/inference components, but no hardware-specific latency, energy or full bill is reported. Sampling one percent of negative pairs is a computational expedient (§6.3), not a present-day encoder cost comparison. Wikipedia/Freebase alignment, ambiguous mentions and incompleteness bound evaluation validity. The authors' broad claims of avoiding domain dependence do not establish domain-invariant performance; the experiment is conducted within one textual source.

**Supports:** the origin of DS, both noise directions, ordered pair aggregation, and distinguishing knowledge-base truth from textual evidence. **Does not establish:** the NYT10m label inventory or annotation policy, bag-union neural scoring, sub-1B ensemble utility, or a cost advantage. **Placement:** 2.1–2.2, with the evaluation-target distinction motivating manual review.

## P21. Dietterich: why ensembles can help and why diversity is conditional

**Citation key:** `dietterich2000ensemble`. **Source:** [author-hosted full paper](https://web.engr.oregonstate.edu/~tgd/publications/mcs-ensembles.pdf), fifteen pages, *Multiple Classifier Systems*, LNCS 1857, pp. 1–15, [DOI](https://doi.org/10.1007/3-540-45014-9_1). The [author's tutorial listing](https://web.engr.oregonstate.edu/~tgd/projects/tutorials.html) identifies the paper. **Evidence class:** theory/tutorial with classical classification experiments.

### Question, framework, and assumptions

The review asks why combining classifiers can outperform a single classifier and how useful members can be constructed. It distinguishes statistical uncertainty among hypotheses consistent with limited data, computational failure to locate the best hypothesis, and representational limitations of a single model (§1, pp. 1–4). Voting can reduce the first, multiple starting points can address the second, and weighted combinations can enlarge effective decision boundaries for the third. These are explanations, not an unconditional guarantee for every fusion rule.

For odd \(L\), equally accurate binary classifiers with **independent** error probability \(p<1/2\) have majority-vote error \(\sum_{k=(L+1)/2}^{L}\binom{L}{k}p^k(1-p)^{L-k}\). The illustrative 21-member, \(p=0.3\) example yields approximately 0.026 ensemble error (Figure 1, p. 2). This formula assumes a binary majority decision and independent errors. It does not apply directly to correlated same-checkpoint precision copies, two-member intersection, calibrated-score averaging, or multilabel bag F1. The paper's broad accurate/diverse framing should be cited as motivation, with these restrictions, rather than as a necessary-and-sufficient practical test based on pairwise disagreement.

Section 2 surveys Bayesian averaging, bagging and boosting, feature subsets, output coding and randomized learning. Bayesian averaging weights hypotheses by their posterior, \(P(y\mid x,S)=\sum_h P(y\mid x,h)P(h\mid S)\); arbitrary validation weights are not automatically Bayesian posteriors. Bagging resamples training examples; boosting successively emphasizes residual errors. Neither is equivalent to collecting sentence evidence in one RE bag. Different seeds/data subsets, different checkpoints, training snapshots and quantization copies are different sources of member variation, whose error correlation must be observed.

### Empirical content and limits

The paper combines reviewed studies with decision-tree comparisons and a synthetic AdaBoost experiment. Table 1 (p. 10) compares C4.5, randomized trees, bagging and boosting across 33 datasets: boosting is often strongest in the low-noise setting. Table 2 adds synthetic label noise to nine domains and shows a different ranking, with bagging more resistant than boosting. The synthetic study in §3/Figure 5 examines an aggressive reoptimization variant against stagewise AdaBoost and illustrates overfitting. These experiments demonstrate conditional outcomes, not a modern NLP or relation-extraction benchmark. Parameter scale, pretrained encoders, NA, entity direction and task-specific transfer do not form part of this study.

The numerical binomial illustration is a theoretical/simulated result. Classical statistical comparisons do not furnish uncertainty bounds for the thesis's local ensemble; their noise conditions and learner families differ. No measured inference latency, application memory, monetary cost or energy frontier is supplied.

### Thesis implications

Disagreement identifies where members differ; it cannot tell which member is correct. A useful local analysis should distinguish gold relations found only by one member, incorrect relations added only by one member, shared true positives and shared misses. Even exclusive true positives may be rejected by an intersection rule, while a union may recover them but add false positives. This diagnostic decomposition is a thesis application of the accurate/diverse principle, not a formula experimentally tested in this paper.

**Supports:** principled motivations for member construction and the importance of differently located errors. **Does not establish:** that an ensemble always beats its best member, that independence follows from different bit widths, or that more members improve quality per unit cost. **Placement:** 2.4 foundation, with a transition to measured quality–cost in 2.6 and the empirical research question in 2.7.

## P22. Green AI: efficiency as an evaluation objective

**Citation key:** `schwartz2019green`. **Source/version:** [arXiv:1907.10597v3](https://arxiv.org/abs/1907.10597v3), twelve-page August 2019 position paper, read in full. Authors: Roy Schwartz, Jesse Dodge, Noah A. Smith and Oren Etzioni. The bibliography deliberately identifies the inspected preprint rather than merging it with a later journal version. **Evidence class:** methodological position paper, with descriptive analyses of earlier AI research.

### Question and conceptual contribution

What would change if efficiency, as well as predictive performance, counted as a research contribution? The paper argues for reporting the computational price of developing, training and running models, both to address resource use and to make participation possible for researchers with smaller budgets. It values efficient results without treating the largest possible accuracy result as the only worthwhile outcome. This supplies a methodological rationale for a cost-aware thesis and for accepting a strong single model as the best outcome.

Its simplified Eq. 1 (p. 3) is \(\operatorname{Cost}(R)\propto E\cdot D\cdot H\): per-example work, dataset size and number of development experiments. The authors explicitly acknowledge that it ignores factors such as epochs and that different configurations have different per-example costs. It is a conceptual decomposition, not a calculator for the local experiment's electricity bill. The thesis should use separate measured preparation, selection and inference accounts instead of treating this product as an exact estimate.

Section 3.1 (pp. 5–7) compares carbon emissions, electricity, elapsed time, parameter count and floating-point operation counts (FPO). Hardware, geography, implementation and workload change what each measure means. FPO is advocated as a relatively portable description of computational work, but the discussion acknowledges that it overlooks memory limits and depends on implementation. For low-bit and sparse execution, a floating-point count alone further fails to capture kernel efficiency and data movement; this is a thesis inference supported by the tensor/runtime distinctions in the compression papers, not a new Green AI experimental result.

### Evidence, uncertainty, and applicability

The descriptive paper sample covers 60 papers from ACL, NeurIPS and CVPR, used to classify stated contributions (Figure 2/§2, p. 3). It is a small historical sample of what authors claimed, not a current prevalence survey of all AI research. Figures 3–4 draw on prior vision results to illustrate diminishing returns and show that parameter count and computational work can order models differently. No NYT10m, encoder RE or SLM ensemble is trained or evaluated; no comparative RE F1 result or compression ablation exists. Historical expenditure estimates are not retained as current prices in this review.

Sections 3.2–3.3 discuss per-inference work and budget–performance curves. The useful implication is to report how much computation was needed to **find** a setting as well as run it. Reusing released pretrained models can reduce local preparation expense, but it does not erase their original training cost. Distillation, calibration, pruning recovery and repeated validation trials belong to preparation/development accounts even if their final checkpoints run cheaply.

### Cost interpretation and chapter use

The paper proposes cost reporting and reuses illustrative estimates rather than measuring this thesis's energy or total bill. Local elapsed time, API token spend and peak memory are different dimensions. An ensemble has the sum of its members' inference work even if parallel wall time approximates the slowest member; simultaneous execution can also increase resident memory. Carbon savings cannot be inferred from reduced parameter count without measuring or transparently estimating electricity and the relevant emission factors.

**Supports:** joint quality–cost evaluation, transparent development budgets, and efficient or negative results as legitimate contributions. **Does not establish:** a universal exchange rate between F1 and cost, actual carbon savings from small RE ensembles, or a particular Pareto frontier. **Placement:** 2.6 for the motivation and accounting principles; 2.7 for positioning the thesis around empirical convenience rather than guaranteed accuracy improvement.
