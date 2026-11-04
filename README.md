# Cost-Aware Ensembling for News Relation Extraction

Data Science master's thesis at the Università degli Studi di Milano. The study evaluates when combining local relation extractors improves quality enough to justify the extra inference cost. Given news sentences and an ordered pair of supplied entity mentions, each system predicts a set of directed relations from a 24-label inventory.

## Results at a glance

Four separately frozen studies use the same **5,174 officially annotated NYT10m bags**, **9,001 selected evidence units** and **3,899 positive facts**. The table shows selected conditions. Full results include every declared control.

| Study | Selected comparison | Test micro-F1 |
|---|---|---:|
| Released models | GLiNER-relex large → three-member majority | 52.28% → 50.10% |
| Numerical reduction | Matched ONNX FP32 → dynamic INT8 | 51.41% → 51.97% |
| LoRA adaptation | Matched original large → Last6 → Last6/Random6 union | 52.28% → 58.88% → 60.60% |
| Heuristic fusion | Last6 → equal mean of Last6 and ModernBERT-NLI | 58.88% → 63.37% |
| API reference | DeepSeek thinking-low, reused across studies | 67.36% |

The tested majority lost recall and took 1.94× the individual large model's recorded local inference time. Dynamic INT8 reduced matched FP32 benchmark latency by about 18%, while averaging three reduced variants fell below INT8 alone. The Last6–NLI equal mean had the highest observed local F1, at 5.20× Last6's **estimated** sequential benchmark time. Its paired gain was +4.49 F1 points with a nominal 95% interval of [+3.50, +5.50]. The late fusion study reused an already inspected test benchmark and is exploratory. Local CPU timing and API request timing have different scopes. See the [full results and limitations](docs/results.md).

## Repository guide

- [Protocol](docs/protocol.md) covers evidence selection, scoring, controls and the four frozen studies.
- [Result inventory](results/README.md) links specifications, aggregate reports, audits and compact TP/FP/FN archives.
- [Reproducibility guide](docs/reproducibility.md) explains how to verify all four sets of test estimates from a fresh clone. The original released-model analysis also has a [report notebook](source/Test_results_analysis.ipynb).
- [Source](source/) and [tests](tests/) contain the released-model pipeline and its analysis and validation checks. [Support](support/README.md) documents the original development annotations. [References](docs/references.bib) lists the research sources.

The later studies are published as frozen evidence. Their execution notebooks, checkpoints, detailed data and API journals are maintained locally. Viewing the reports and verifying their counts requires neither inference nor API access.
