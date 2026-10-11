# Cost-Aware Ensembling for News Relation Extraction

Data Science master's thesis at the Università degli Studi di Milano. The study asks **when combining small relation extractors improves quality enough to justify the additional computation, compared with individual models and a large model accessed through an API**.

The current optimization objective is to measure **how closely compact systems approach DeepSeek's measured F1 while reducing local inference time, memory and storage**. The next main experiment uses **ONNX Runtime to quantize GLiNER-relex large**, investigating the requested **4-, 8- and 16-bit** levels against the unquantized parent. **PEFT on limited modules and differentiated training subsets** is an exploratory extension; neither optimization branch has produced results yet.

The task is bag-level, multi-label relation extraction on **NYT10m**: given news sentences and an ordered pair of supplied entity mentions, predict a set of 24 positive relation IDs. An empty set represents `NA`. The encoder adapters use the supplied spans and preserve head → tail direction.

## Main result

The completed, frozen comparison evaluates **5,174 officially annotated test bags** under the same evidence and scoring rules.

| System | Precision | Recall | Micro-F1 | Recorded time |
|---|---:|---:|---:|---:|
| GLiNER-relex large, cutoff 0.7 | 0.5552 | 0.4940 | 0.5228 | 32.63 min |
| Majority: base 0.5 / large 0.7 / GLiDRE 0.7 | 0.6902 | 0.3932 | 0.5010 | 63.40 min |
| DeepSeek thinking-low | 0.8481 | 0.5586 | 0.6736 | 6.12 h |

The tested majority improves precision but reduces recall and F1, using **1.94×** the large model's local inference time. DeepSeek achieves the highest measured F1; all **78 invalid/truncated answers** remain in scoring. Its completed-response token charge reconstructs to approximately **USD 2.41**, with two interrupted attempts' charges unknown. Local CPU inference and remote request time have different measurement scopes.

The paired bag-bootstrap 95% interval for majority minus large is **−2.18 F1 percentage points [−3.26, −1.11]**. This conclusion concerns the evaluated members, thresholds and voting rule. On the matched relative scale `100 × F1(system) / F1(DeepSeek)`, large scores **77.62**, majority **74.38** and DeepSeek **100.00**. This empirical upper reference is not a mathematical ceiling or perfect F1; absolute scores remain visible. See the [reporting convention](docs/protocol.md#deepseek-relative-f1).

## Next experiments

1. Validate ONNX FP32 export of the complete supplied-entity relation path and measure paired original PyTorch/ONNX FP32 controls.
2. Implement dynamic INT8 and assess the requested 4-/16-bit levels, recording exact weight/activation formats and supported operations. ORT's documented INT4 route is weight-only; FP16 is floating-point conversion. Support for dynamic activation quantization at each requested level must be verified.
3. Compare feasible variants on validation using P/R/F1, repeated timing, peak memory, artifact size and preparation cost; select and freeze before later test evaluation.
4. Study compatible PEFT methods on limited modules and train-only subsets, comparing untuned controls and separating training cost from inference. Isolate adaptation from quantization before a combined experiment or ensemble.

The [remaining-work protocol](docs/protocol.md#remaining-experimental-work) defines the experiment boundaries. Pruning remains optional later work; ICL, routing and extra decoders remain deferred.

## Reading guide

1. [Experimental protocol](docs/protocol.md): data splits, supplied entities, evidence selection, models and scoring.
2. [Results and interpretation](docs/results.md): uncertainty, voting diagnostics, figures and separate cost measurements.
3. [Reproducibility guide](docs/reproducibility.md): report viewing, local execution, dependencies and checks.

For the complete tables and figures, start with [Test_results_analysis.ipynb](source/Test_results_analysis.ipynb). The [SLM test](source/SLM_test.ipynb), [DeepSeek test](source/DeepSeek_test.ipynb) and [full-validation](source/Full_validation_relex.ipynb) notebooks also display published reports by default, without datasets, checkpoints or API access.

## Repository structure

| Location | Contents |
|---|---|
| [docs/](docs/) | Public protocol, results, reproduction instructions and [bibliography](docs/references.bib) |
| [source/](source/) | Inference/scoring modules, analysis scripts and notebooks |
| [results/](results/README.md) | Frozen specifications, aggregate reports, bootstrap counts and figures |
| [support/](support/README.md) | Reviewed development samples, annotation evidence and historical versions |
| [tests/](tests/) | Synthetic checks for direction, reconstruction, parsing, caching and statistics |

The published repository contains compact evidence. Datasets, model weights, detailed per-bag caches, request journals, credentials and personal thesis drafts are excluded from Git. A fresh clone can display the main reports and reproduce their bootstrap analysis; full inference requires the assets and runtime described in the [reproducibility guide](docs/reproducibility.md).

## Quick start: view the reports

```bash
python -m venv .venv-report
source .venv-report/bin/activate
python -m pip install -r requirements-report.txt
jupyter lab
```

Open `source/Test_results_analysis.ipynb` and run its cells with the default flags. For verification in the full inference environment:

```bash
python -m unittest discover -s tests -v
python source/check_publication.py
```

The full synthetic suite requires [requirements.txt](requirements.txt). Report-only dependencies support the published analysis and its statistical checks; see the [reproducibility guide](docs/reproducibility.md). Dataset references are [Gao et al. (2021)](https://aclanthology.org/2021.findings-acl.112/) and the [OpenNRE distribution](https://github.com/thunlp/OpenNRE#data).
