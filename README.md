# Cost-Aware Ensembling for News Relation Extraction

A Data Science master's thesis at the Università degli Studi di Milano investigating:

**When does combining small language models improve news relation extraction enough to justify the extra computation, compared with individual small models and a large model accessed through an API?**

Given an ordered entity pair and a bag of news sentences, the task is to predict a set of the 24 positive relations in **NYT10m**. An empty set represents no supported positive relation. Entity mentions and their direction are supplied; the encoder adapters bypass entity recognition.

## Completed comparison

Development used all **36,266 validation bags** with noisy distant labels, plus a separately reviewed 30-bag diagnostic reference. Checkpoints, thresholds, fusion, prompt and scoring were frozen before the first official manual-test comparison.

The completed test contains **5,174 bags**, **9,001 selected sentence/mention units**, and **3,899 positive manual bag facts**. Repeated rows that serialize multiple manual labels are merged before deterministic selection of at most 20 evidence units per pair. References use only selected evidence; in this release the selected and full-bag fact unions happen to coincide.

| Frozen system | Precision | Recall | Micro-F1 | Accumulated time |
|---|---:|---:|---:|---:|
| GLiNER-relex large, cutoff 0.7 | 0.5552 | 0.4940 | 0.5228 | 32.63 min |
| Majority: base0.5 / large0.7 / GLiDRE0.7 | 0.6902 | 0.3932 | 0.5010 | 63.40 min |
| DeepSeek thinking-low | 0.8481 | 0.5586 | 0.6736 | 6.12 h |

The tested majority improves precision but loses recall and F1 at 1.94× local inference time. DeepSeek has the highest measured quality, with 78 truncated/invalid answers retained in scoring. Its completed-response token charge reconstructs to approximately **$2.41**, with two interrupted attempts' charges unknown. This is not an invoice. Local CPU inference time excludes loading/preflight; API time includes network/service and request attempts, so these are distinct cost measurements.

See the [frozen specification](results/test_comparison_spec.json), [combined results](results/test_comparison_summary.json), [results audit](results/test_results_audit_20261008.json), and [working notes](support/notes.md). No test outcome is used to revise the evaluated configuration. Quantization/pruning and ensembles of reduced variants remain a separate, training/validation-based phase. ICL, routing, extra decoders and new fine-tuning are deferred.

The [statistical analysis](source/Test_results_analysis.ipynb) adds paired whole-bag confidence intervals and quality–latency/precision–recall figures using saved predictions only. Majority minus large is **−2.18 F1 percentage points [95% CI: −3.26, −1.11]**; DeepSeek minus large is **+15.08 [13.65, 16.51]**. These intervals describe sampled-bag uncertainty, not repeated API-generation variability. API expenditure, local CPU time and unmeasured peak memory remain separate. See the [analysis write-up](support/test_results_analysis_2026-10-08.md).

## Review the notebooks

| Notebook | Purpose | Default execution |
|---|---|---|
| [Test_results_analysis.ipynb](source/Test_results_analysis.ipynb) | Paired F1 intervals, quality/latency figures, voting diagnostics and separate costs | Published analysis; no data, models or API access required |
| [SLM_test.ipynb](source/SLM_test.ipynb) | Completed large/majority test and matching three-system comparison | Published reports; no local data or models required |
| [DeepSeek_test.ipynb](source/DeepSeek_test.ipynb) | Completed low-reasoning test, API accounting and matching comparison | Published reports; no API key or local data required |
| [Full_validation_relex.ipynb](source/Full_validation_relex.ipynb) | Three-encoder validation, competitive cutoffs and ensemble costs | Published report; no checkpoint loading |
| [DeepSeek_flash_validation.ipynb](source/DeepSeek_flash_validation.ipynb) | Completed non-thinking/low/high development pilot | Paid calls disabled; execution requires local validation inputs/caches |
| [Preliminary_experiments.ipynb](source/Preliminary_experiments.ipynb) | Eight historical experiments with embedded aggregate results and limitations | Analysis only; no datasets or models required |
| [EDA.ipynb](source/EDA.ipynb) | Train/validation schema, bags, labels, lengths and duplicates | Requires local NYT10m data |
| [data.ipynb](source/data.ipynb) | Obtain and inspect the official distribution | Downloads require `DOWNLOAD_DATA=True` |

GitHub displays the saved notebook tables. For interactive report viewing, install `jupyterlab`, `pandas`, `nbformat` and `nbclient` in a Python environment and open a report notebook. Local inference uses the separately verified Python 3.14.3 environment pinned in [requirements.txt](requirements.txt); compatibility must be checked when installing it on another platform.

## Local execution and reproducibility

1. Create a virtual environment and install [requirements.txt](requirements.txt) for inference, or the smaller analysis dependency set above for report viewing.
2. Obtain NYT10m using the download notebook and the [official OpenNRE distribution](https://github.com/thunlp/OpenNRE#data), described by [Gao et al. (2021)](https://aclanthology.org/2021.findings-acl.112/).
3. Use the explicit notebook controls. In test notebooks, `ANALYZE_LOCAL_RESULTS=True` verifies retained local caches without new inference; `RUN_LOCAL_INFERENCE=True` or `RUN_API=True` enables execution. Model flags select encoder workers, and asset prefetch is separately enabled before offline inference.
4. Supply `DEEPSEEK_API_KEY` through the environment or a local `.env` based on [.env.example](.env.example) only for paid execution. The API notebook retains a $10 usage/reserve budget, durable request journals and explicit uncertain-attempt recovery.

The shared modules implement test reconstruction/scoring (`nyt10m_test_protocol.py`), isolated offline encoder workers (`nyt10m_test_encoders.py`), and fixed-prompt API execution/accounting (`nyt10m_test_api.py`). Published reports can be read separately through `publication_utils.py`.

Checkpoint revisions, weight identities, tokenizer/config metadata and source hashes are recorded in the frozen specification. The GLiDRE implementation uses pinned author code, an isolated GLiNER 0.2.13 helper, and a compatibility bridge preserving pretrained layers. Native-parity and supplied-span preflights passed before inference. Original hash-bound runners/adapters remain unchanged.

Datasets, weights, detailed score caches, API journals, environments and credentials are **local only and excluded from Git**. A fresh clone contains compact aggregate evidence and cannot reconstruct every per-bag result without obtaining assets and running inference. Cached results may be resumed only with matching frozen settings; a different runtime or method requires an explicitly versioned experiment.

Run the retained synthetic/stub checks and publication scan with:

```bash
python -m unittest discover -s tests -v
python source/check_publication.py
```

## Documentation

- [Protocol](support/protocol.md), [notes and interpretation](support/notes.md), and [roadmap](support/starter.md).
- [Literature review](support/literature.md), [evidence matrix](support/evidence_matrix.md), and [BibTeX references](support/references.bib).
- [Thesis outline](support/thesis_outline.md) and [background chapter](support/background_related_work.md).
- [Latest cleanup and publication checks](support/cleanup_2026-10-08.md).
