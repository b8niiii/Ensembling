# Reproducibility guide

## Verify the published results

The [result inventory](../results/README.md) links the four specifications, summaries, audits and paired-count archives. The archives let a fresh clone recompute test precision, recall and micro-F1 without model checkpoints, NYT10m texts or API access.

For the released, ONNX and LoRA archives, the NPZ member counts has axes [bag, system, TP/FP/FN]. The systems member gives the column names. The heuristic archive instead stores one [bag, TP/FP/FN] array per system. All rows preserve the frozen bag order. Sum counts over bags before computing micro metrics.

With NumPy installed, the following example verifies the late equal-mean condition against the published summary. Run it from the repository root.

```python
import json
from pathlib import Path

import numpy as np

results = Path("results")
system = "nli__equal_mean"
with np.load(results / "heuristic_ensembles_test_counts_20261022.npz", allow_pickle=False) as archive:
    tp, fp, fn = archive[system].sum(axis=0)
report = json.loads((results / "heuristic_ensembles_test_summary_20261022.json").read_text())
row = next(row for row in report["quality"] if row["system"] == system)
assert (tp, fp, fn) == (row["TP"], row["FP"], row["FN"])
assert np.isclose(2 * tp / (2 * tp + fp + fn), row["micro_f1"])
```

The bootstrap settings and contrasts for each study are recorded in the linked reports or specifications. Recomputing an interval requires sampling the same bags for both compared systems in each replicate and calculating the difference of micro-F1 values. The released-model implementation is in [analyze_test_results.py](../source/analyze_test_results.py). It targets the three systems from that study.

## View the released-model notebooks

Use an isolated report environment.

```bash
python -m venv .venv-report
source .venv-report/bin/activate
python -m pip install -r requirements-report.txt
jupyter lab
```

Select this environment's kernel. The pins reflect the recorded environment and platform compatibility may vary.

| Notebook | Report |
|---|---|
| [Test results analysis](../source/Test_results_analysis.ipynb) | Estimates, paired intervals, voting diagnostics and figures |
| [SLM test](../source/SLM_test.ipynb) | Individual large and released majority |
| [DeepSeek test](../source/DeepSeek_test.ipynb) | API quality, validity and accounting |
| [Full validation](../source/Full_validation_relex.ipynb) | Three-encoder validation, cutoffs and fusion |
| [Preliminary experiments](../source/Preliminary_experiments.ipynb) | Historical screening tables |

The test analysis notebook displays saved evidence by default. Its optional RECOMPUTE_BOOTSTRAP flag reruns the released-model statistics. REGENERATE_FIGURES also requires [analysis dependencies](../requirements-analysis.txt).

## Execution boundary

The published [released-model pipeline](../source/) includes test reconstruction, supplied-entity adapters, scoring, caching and API parsing. Full reproduction requires local NYT10m data, pinned checkpoint assets and the recorded [inference environment](../requirements.txt). The original runs used Python 3.14.3 on an Apple M4 CPU. The frozen specifications bind source, inputs and checkpoint identities. Future calls to the API alias may produce different outputs.

The [data notebook](../source/data.ipynb) supports explicit dataset download. The SLM and DeepSeek test notebooks require explicit inference or API flags and prepared local inputs. Use [.env.example](../.env.example) to identify the local environment variable for the API key. The later ONNX, LoRA and fusion studies expose report-level reproducibility through their frozen artifacts. Their execution code, full caches and environment receipts live in the local workspace.

For the published synthetic tests and publication checks, use the relevant dependencies and run:

```bash
python -m unittest discover -s tests -v
python source/check_publication.py
```
