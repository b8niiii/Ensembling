# Cost-Aware Ensembling of Small Language Models for Relation Extraction from News

This repository contains work in progress for a Data Science master's thesis. The research question is: **When does combining small language models improve news relation extraction enough to justify the extra computation, compared with individual small models and a large model accessed through an API?**

## Task and data

The initial benchmark is **NYT10m**, distributed through [OpenNRE](https://github.com/thunlp/OpenNRE) and described in [Gao et al. (2021)](https://aclanthology.org/2021.findings-acl.112/). Each prediction concerns an *ordered pair of entities* and a bag of sentences mentioning them. The system must return zero or more relations from the dataset's fixed inventory; an empty set means no positive relation is predicted. This is bag-level relation extraction, not extraction from a complete news article.

Training and validation labels are produced by distant supervision and can be noisy. In the first phase, model weights will **not** be fine-tuned: training data supplies candidate in-context examples, validation supports pilot experiments and method choices, and the manually annotated test set is reserved for final evaluation after those choices are fixed.

## Planned experiments

1. **Zero-shot baseline:** Run three distinct small language models (SLMs) individually and compare them with a larger API model. Combine the three SLM predictions using relation-wise majority voting and compare the ensemble with each member and the API baseline.
2. **In-context learning (ICL):** Repeat the individual and ensemble comparisons with a small, fixed set of examples selected from training data. Measure whether any quality gain justifies the additional prompt tokens and inference time.
3. **Quantization:** Run each SLM at feasible GGUF precision levels. Compare individual variants, ensembles whose members use the same precision, and ensembles with different precisions assigned to different members. Keep the voting rule fixed to isolate the effect of quantization.

The SLMs will run locally with [llama.cpp](https://github.com/ggml-org/llama.cpp); Python and Jupyter notebooks will prepare inputs and analyze saved predictions. The API model is a separate baseline, not a member of the SLM ensemble. Planned reporting includes precision, recall, micro/macro F1, invalid-output rate, latency, memory, token use, and API spending, with energy measured where feasible. Ensemble cost includes all member calls even though voting itself requires no additional model inference.

## Current status

The NYT10m data has been acquired and training/validation exploratory analysis is available in [`source/EDA.ipynb`](source/EDA.ipynb). The [`source/data.ipynb`](source/data.ipynb) notebook records the initial data download and inspection workflow. The model roster, API baseline, ICL examples, and final quantization assignments are **not yet selected**; no model comparison results are reported here yet.

Dataset files, model weights, virtual environments, and local research notes are excluded from Git. Obtain NYT10m from the [OpenNRE benchmark source](https://github.com/thunlp/OpenNRE/tree/master/benchmark) before running the notebooks. The test set is not used to choose prompts, models, ensemble rules, or quantization settings.
