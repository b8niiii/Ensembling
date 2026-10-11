# Reproducibility guide

## What a fresh clone can reproduce

The main reports, figures, reviewed development labels, frozen specifications and per-bag TP/FP/FN bootstrap counts are public. Datasets, checkpoints, detailed score caches, API responses/journals and the original environment are excluded from Git.

A fresh clone can display the primary reports, validate their shared experiment identity and reproduce the paired bootstrap from the compact count archive. Reconstructing predictions or auditing raw request journals requires the corresponding local assets. The API alias cannot guarantee identical future generations; runtimes and source hashes are part of the frozen identity.

## View the published reports

Use an isolated report environment from the repository root:

```bash
python -m venv .venv-report
source .venv-report/bin/activate
python -m pip install -r requirements-report.txt
jupyter lab
```

Select this environment's Python kernel. The dependency pins reflect the recorded project environment; their installation on a different Python version/platform must be checked separately.

| Notebook | Purpose | Default requirements |
|---|---|---|
| [Test_results_analysis](../source/Test_results_analysis.ipynb) | Primary estimates, paired intervals, voting diagnostics and figures | Published JSON/NPZ/PNG files |
| [SLM_test](../source/SLM_test.ipynb) | Individual large, majority and combined test comparison | Published summaries |
| [DeepSeek_test](../source/DeepSeek_test.ipynb) | API test quality, validity and accounting | Published summaries |
| [Full_validation_relex](../source/Full_validation_relex.ipynb) | Three-encoder validation, cutoffs and fusion | Published reports and reviewed annotations |
| [Preliminary_experiments](../source/Preliminary_experiments.ipynb) | Eight historical screening experiments and their limitations | Embedded aggregate tables |
| [DeepSeek_flash_validation](../source/DeepSeek_flash_validation.ipynb) | Development non-thinking/low/high pilot | Original local validation manifests and response caches; paid calls disabled |
| [EDA](../source/EDA.ipynb) | Train/validation schema, bags, labels and lengths | Local NYT10m files |
| [data](../source/data.ipynb) | Dataset download and inspection | Downloads explicitly enabled; inspection requires local data |

In `Test_results_analysis`, `RECOMPUTE_BOOTSTRAP=True` recomputes statistics from the public count archive without inference. `REGENERATE_FIGURES=True` also requires [requirements-analysis.txt](../requirements-analysis.txt), which adds Matplotlib to the report dependencies. Neither flag requires an API key or checkpoint. Default viewing uses saved figures.

## Local inference and cached-result verification

The recorded inference environment is Python 3.14.3 on an Apple M4 CPU. Create the project `.venv` and install [requirements.txt](../requirements.txt) only when reproducing the inference workflow; the pinned runners require that interpreter location. Preserve the original environment when evaluating a new runtime.

1. Obtain NYT10m with [data.ipynb](../source/data.ipynb), setting `DOWNLOAD_DATA=True`. It writes `data/nyt10m/`; the official test SHA-256 is checked before preparation. EDA and method decisions use train/validation.
2. Prefetch pinned checkpoint/tokenizer/helper assets before offline execution. The preparation stages need the recorded model metadata as well as dataset files. Downloads and model inference are separate stages.
3. Use the notebook flags below to choose published viewing, local verification or inference. Detailed results are saved under versioned directories inside `data/`, and identical completed caches can be resumed.
4. For API execution, supply `DEEPSEEK_API_KEY` through the environment or a local `.env` based on [.env.example](../.env.example). The test notebook's recorded budget is USD 10; the development notebook's is USD 2. Budget accounting includes retained attempts and uncertain-charge reserves. These are explicit execution controls, not current pricing estimates.

| Notebook | Read existing local results | Enable inference | Asset controls |
|---|---|---|---|
| SLM_test | `ANALYZE_LOCAL_RESULTS=True` | `RUN_LOCAL_INFERENCE=True`, member flags | `PREFETCH_ASSETS=True`; metadata must exist before frozen-input preparation |
| DeepSeek_test | `ANALYZE_LOCAL_RESULTS=True` | `RUN_API=True` | Prepared shared test inputs/model metadata |
| Full_validation_relex | `ANALYZE_LOCAL_SCORES=True` | `RUN_RELEX_BASE`, `RUN_RELEX_LARGE`, `RUN_GLIDRE` | `FETCH_ENABLED_MODEL_ASSETS=True` for enabled members |
| DeepSeek_flash_validation | Defaults analyse the original local pilot | `RUN_API=True`, mode flags | Original validation manifests, reviewed labels and fixed random100 inputs |

Missing test model assets can be downloaded explicitly from the project `.venv` before notebook preparation:

```bash
python source/run_full_validation_relex_given_entities.py prefetch
python source/run_full_validation_glidre.py prefetch --glidre
```

GLiDRE's pinned helper is isolated from the main GLiNER 0.2.29 installation. Runtime preflights remain mandatory. A changed runtime, method, input or hash-bound source requires a separately versioned experiment; bypassing identity checks would invalidate cache reuse. The original API development notebook's implementation cells also enter its cache fingerprint.

## Code entry points

| Module | Responsibility |
|---|---|
| `nyt10m_test_protocol.py` | Manual-row reconstruction, frozen shared inputs, scoring and voting |
| `nyt10m_test_encoders.py` | Sequential offline encoder workers, preflights and score caches |
| `nyt10m_test_api.py` | Frozen API prompt/parser, request journals and accounting |
| `relex_given_entities.py`, `glidre_given_entities.py` | Directed supplied-span adapters |
| `run_full_validation_relex_given_entities.py`, `run_full_validation_glidre.py` | Pinned validation preparation and execution |
| `report_full_validation_relex.py`, `validation_model_comparison.py` | Cached validation analysis |
| `reviewed_validation_reference.py` | Confirmed C/N annotation identity and historical change inventory |
| `analyze_test_results.py`, `plot_test_results.py` | Saved-prediction statistics and scientific figure export |
| `publication_utils.py` | Identity-checked reading of primary public summaries |
| `prepare_reduction_pilot.py` | Reproduce the fixed validation-only 100-bag input sample; no model execution |
| `run_extended_gliner_relex.py` | Historical extended pilot and hash-bound relation-description dependency |

Modules are in [source/](../source/). The compact [result inventory](../results/README.md) maps reports to stages; [support/README.md](../support/README.md) explains current and historical annotations.

## Planned optimization experiments

The [remaining-work protocol](protocol.md#remaining-experimental-work) prioritises ONNX Runtime quantization of GLiNER-relex large at requested 4/8/16-bit levels, with exact tensor formats and paired PyTorch/ONNX FP32 controls. The existing inference pins do not establish ONNX export or reduced-kernel compatibility. Record exporter/runtime/provider versions in a separate reduction environment, preserve the frozen baseline and use new graph/artifact/run hashes. Pilot preparation is not quantization execution.

PEFT on limited modules and differentiated training subsets is exploratory: record method, target modules, trainable parameters, train-only subset manifests, hyperparameters and training cost. Verify adaptation alone and any later merge/export before combining it with quantization. Both branches need validation selection and a separate freeze before later test evaluation. Final matched [DeepSeek-relative F1](protocol.md#deepseek-relative-f1) accompanies absolute metrics; no validation/test denominator mixing is allowed.

## Verification

In the inference environment, run:

```bash
python -m unittest discover -s tests -v
python source/check_publication.py
```

The tests use synthetic data, stubs and temporary files. They check supplied direction, score inventories, manual reconstruction, invalid-output handling, cache identity, voting and whole-bag statistics, without pretrained inference or real API calls. In a report-only environment, run the statistical subset with `python -m unittest discover -s tests -p 'test_test_results_analysis.py' -v`.

The publication checker scans candidate files and reachable Git history for credentials/oversized blobs, validates Python and notebook syntax, checks public documentation links, rejects local-only files and verifies that notebook execution is disabled by default. It makes no provider requests and does not stage, commit or push.
