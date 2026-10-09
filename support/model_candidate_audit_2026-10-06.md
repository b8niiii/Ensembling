# Additional sub-1B candidate audit

Checked 6 October 2026. This is a source-based shortlist, not an executed experiment or a frozen roster. Public model cards, author code, papers, and Hugging Face metadata/configurations were inspected. No weights were downloaded and no candidate inference was run.

## Selection criteria

Candidates require released weights below one billion total parameters, a plausible CPU/Python implementation, and a credible route to the same 24 directed NYT10m relations. Supplied entities, multi-label decisions, abstention, sentence selection/cap20, and separate distant versus reviewed references must be preserved. Quantization does not change parameter count. A different architecture or training objective motivates a complementarity hypothesis; it does not establish an ensemble gain.

Parameter counts below include embeddings and checkpoint heads. Exact figures come from public Hugging Face `safetensors.total` metadata; GLiDRE's approximate total comes from its paper. ModernBERT's count is rounded from its card. Runtime memory and CPU speed have not been measured locally.

## Shortlist and reserves

| Released checkpoint | Parameters | Task/training evidence | Compatibility and limitation | Screening priority |
|---|---:|---|---|---|
| [cea-list-ia/glidre_large](https://huggingface.co/cea-list-ia/glidre_large) | Approx. 800M, both encoders combined | Document RE; synthetic FineWeb pretraining. The paper separately studies Re-DocRED fine-tuning. | Native supplied mentions, custom relation labels, multi-label output. Boundary/context and complete score access require verification; the released checkpoint's exact training stage is insufficiently specified. | First new RE candidate |
| [MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c) | 435,063,810 | Entailment classification; Mixtral synthetic classification data, MNLI and FEVER-NLI. | Reformulate each directed relation as an entity-specific hypothesis. No RE benchmark establishes NYT10m quality. Up to 24 premise/hypothesis evaluations per sentence; 512-token limit. | Second, distinct objective |
| [knowledgator/gliner-relex-multi-v1.0](https://huggingface.co/knowledgator/gliner-relex-multi-v1.0) | 318,859,779 | Multilingual RE with mDeBERTa-v3-base; incomplete checkpoint-specific training inventory. | Custom relations; the supplied-span adapter needs its own parity check. Shared framework and multilingual training do not guarantee independent errors or better English RE. | Direct RE reserve |
| [MoritzLaurer/ModernBERT-large-zeroshot-v2.0](https://huggingface.co/MoritzLaurer/ModernBERT-large-zeroshot-v2.0) | Approx. 400M | NLI/classification with the author's broader zero-shot training mix. | Alternative to the DeBERTa entailment candidate: longer context and author-reported GPU efficiency, slightly lower general classification accuracy. These are not Mac CPU or RE measurements. | Cost/context reserve; not an additional default trial |
| [knowledgator/gliclass-large-v3.0](https://huggingface.co/knowledgator/gliclass-large-v3.0) | 438,668,801 | Zero-shot classification, extended with logical reasoning tasks. | Can score relation hypotheses, but the card recommends a single hypothesis for best NLI behavior. No direct NYT10m RE evidence or complete training inventory. | General-classifier reserve |
| [numind/NuExtract-1.5-tiny](https://huggingface.co/numind/NuExtract-1.5-tiny) | 494,032,768 | Qwen2.5-based structured extraction on a private dataset. | Requires its native template format. Extractive values/triples are not automatically equivalent to the fixed relation-ID classifier; provenance and mapping remain unresolved. | Decoder reserve |
| [s-nlp/enoki-openie-encoder](https://huggingface.co/s-nlp/enoki-openie-encoder) | 395,098,170 | September 2026 ModernBERT OpenIE; distilled EnokiQA training. | Produces text-anchored free-form triples, rather than supplied-pair/24-label scores. Entity alignment, ontology mapping, short default context and license verification add integration work. | Recent research reserve |

The [GLiDRE paper](https://arxiv.org/html/2508.00757v1) reports Re-DocRED test F1 **0.1732 in zero-shot** and **0.7783 after supervised fine-tuning**. Neither is directly comparable with the selected NYT10m reviewed-30 scores. The latter must not be advertised as the released candidate's expected no-adaptation performance. Its synthetic pretraining uses 136,404 FineWeb documents annotated by Mistral-Small-24B. The [configuration](https://huggingface.co/cea-list-ia/glidre_large/blob/main/glidre_config.json) uses DeBERTa-v3-large for text and BGE-large-en-v1.5 for labels; the [native implementation](https://github.com/cea-list-lasti/glidre/blob/main/glidre/model.py) accepts supplied mentions. Different label encoding and pair representations motivate testing, without proving complementary errors.

For entailment, a parent-to-child relation can be expressed as “Mark Mays is a child of Lowry Mays”, with the original sentence as the premise. All 24 templates must explicitly encode head/tail direction. Classification and a verifier cascade are different experiments: a verifier called only on large's predictions cannot demonstrate independent full-label recall. Both stages and all premise/hypothesis evaluations count toward cost. The [author's collection](https://huggingface.co/collections/MoritzLaurer/zeroshot-classifiers) documents the DeBERTa versus ModernBERT accuracy/efficiency distinction; general classification scores do not establish relation-extraction performance.

No inspected candidate has directly comparable published evidence demonstrating parity with or superiority to the current supplied-entity GLiNER-relex large on NYT10m. The reviewed cards do not document NYT10m task training for the first two candidates, but that is not proof of zero web-pretraining exposure. GLiDRE checkpoint-stage ambiguity and any incomplete training inventory must remain explicit. License declarations do not certify dataset non-overlap.

## Other families and historical candidates

- The previously evaluated GLiREL and GLiFormer releases remain at the same inspected revisions, `40a523e12a8432d6da364cf2a195a28755ff04d3` and `d0a4e53d09cebe6bc963dd9be319d4279084bb2d`. Historical joint-recognition/name-matching results do not establish their supplied-entity performance. Reopening them requires a concrete protocol correction, not an assumed newer checkpoint.
- [ReLiK](https://github.com/SapienzaNLP/relik) has an NYT-specialized release requiring a dataset/split audit, and a Wikipedia release requiring Wikidata-to-NYT ontology coverage and full retriever/reader accounting. Neither is an established drop-in candidate for all 24 relations.
- [REBEL](https://huggingface.co/Babelscape/rebel-large) and [KnowGL](https://huggingface.co/ibm-research/knowgl-large) are structured triple generators with their own Wikidata-based schemas. Full relation coverage and supplied-pair mapping need evidence before a fair comparison.
- Fixed-head TACRED/DocRED classifiers such as ATLOP/DREEAM-style checkpoints cannot acquire the 24 NYT10m labels merely by changing a prompt. New fine-tuning remains deferred. A pretrained BERT/ModernBERT backbone without an appropriate trained head is not a ready RE model.

## Revisions inspected

| Checkpoint | Hugging Face revision |
|---|---|
| GLiDRE large | `d1ba1cf0f41d2849f1695c6acc67546700d721ba` |
| DeBERTa zero-shot v2.0-c | `b2730f16019076bb0009481121efbe4705e0e378` |
| GLiNER-relex multi v1.0 | `e990d9ba6f471b846f7d78bf7e4b4dab11761ada` |
| GLiClass large v3.0 | `e065d1844f913a9aa611cf33623a9538b8aa8841` |
| NuExtract 1.5 tiny | `63e2e80c804d9c97f3f19a4aa25613e7beca83c9` |
| Enoki OpenIE encoder | `ffa8f0078cbda099b66b7475ebdcb6057f8d688c` |

## Proposed bounded validation gate

1. Screen GLiDRE first; consider one entailment encoder as the second experiment. Other rows remain reserves. Verify loading, total parameters, complete 24-score inventory, multi-label abstention, exact spans/direction and length handling before scoring. No silent truncation or extra entity/type annotations derived from reference labels are permitted.
2. Reuse the current confirmed 30 for error diagnosis and the already fixed random100 validation sample for a separate distant-label/cost pilot. Preserve its current annotation version. No new annotation or official-test access is required. Compare each candidate with cached large on identical evidence; any necessary input change requires a separately versioned fair comparison.
3. Inspect both correct labels absent from large and false positives a combination can remove without sacrificing too many supported labels. Compare P/R/F1 and total cost with competitive individual cutoffs. Different confidence scales do not justify averaging raw scores without validation.
4. Extend only a compatible, practically promising candidate to broader validation before freezing the roster/fusion. Small reused samples establish a screening signal, not a general ensemble advantage. If neither candidate supports a useful quality–cost trade-off, keep large as the leading individual and report existing ensemble outcomes; the later reduction branch remains separate.

This bounded gate was the initial research proposal. A subsequent request prepared a full-validation GLiDRE extension in [Full_validation_relex.ipynb](../source/Full_validation_relex.ipynb), with unchanged supplied evidence and mandatory offline checks before inference. The architecture-only count is 800,241,152 parameters; pinned imports, native processor collation and synthetic adapter/report tests were checked without weights or model inference. Actual strict checkpoint loading, parity and full validation remain pending the local run. Existing frozen relex caches and sources are unchanged; candidate selection remains open.
