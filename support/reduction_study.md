# First reduction feasibility study

Prepared for 6 October 2026; deferred under the updated work order until after the first frozen released-model/API comparison. Variants will be designed and selected on training/validation, with later test use separately declared; first-test errors will not drive reduction design. Large is a provisional parent, not the frozen final model. The prepared local pilot has 100 uniformly sampled validation bags / 118 selected records, seed `20261006`, cap 20. No quantization, pruning, recovery training, or reduced-model inference has run.

## Quantization gate

Inspect the actual installed CPU/ARM runtime and model graph first. Candidate implementation routes include the [PyTorch AO quantization project](https://github.com/pytorch/ao) and an explicitly exported/validated [ONNX Runtime](https://github.com/microsoft/onnxruntime) graph. Their existence does not establish compatibility or acceleration for GLiNER-relex, the supplied-span adapter, Python 3.14, or this Mac. Do not convert these encoders to GGUF merely because earlier decoder pilots used that format.

Before choosing a route, record supported layers, dtype/kernel availability, exporter constraints, and whether the adapter's relation outputs remain accessible. Keep the current environment/checkpoint intact; any necessary dependency changes belong in a separate recorded reduction environment. Verify one reduced artifact against original large on the fixed inputs before creating additional precision levels.

## Pruning options to evaluate

| Option | What changes | Evidence needed before an efficiency claim | Potential recovery cost |
|---|---|---|---|
| Unstructured weight masking | Some stored weights become zero | Real sparse kernel/storage support; a dense mask alone does not establish fewer operations | Optional train-only recovery, measured separately |
| Structured layer/head/channel removal | Executable dimensions or modules change | Architecture changes, shape/parity checks, complete relation output, actual size/time/memory reduction | Likely needs a separately tested recovery procedure; not assumed necessary or sufficient |
| Backend-supported sparse pattern | Weights follow a hardware/runtime constraint | Verify the pattern and a usable CPU execution path on this machine | Calibration/recovery, if used, must be recorded |

The official [PyTorch pruning tutorial](https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html) documents mask reparameterization and structured/unstructured methods; it is a method reference, not proof that a pruned transformer will accelerate this CPU pipeline. Removing a pruning reparameterization does not itself shrink layer dimensions.

## Paired measurement contract

1. Record original parent/revision/hash, variant method/settings/hash, versions, selected-input hash, and supported operations.
2. Check supplied-span direction, complete 24-label scores, failures, and any numerical changes.
3. Compare quality on identical inputs and a separately identified reference; distinguish distant labels and reviewed text.
4. Measure repeatable latency with loading/warmup distinguished, peak process memory, artifact bytes, and any preparation/calibration/recovery cost.
5. Continue only if one usable variant has a defensible measured trade-off. Then choose controlled levels and compare each variant alone before uniform/mixed-precision ensembling.

The 100-bag random sample is a feasibility/cost pilot, not a sufficient estimate for rare relations or a full validation substitute. Larger paired validation and targeted error review follow only after compatibility is established.
