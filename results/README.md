# Result inventory

All four official comparisons use the same ordered 5,174 NYT10m test bags and 3,899 positive selected-evidence facts. The published archives contain per-bag TP/FP/FN contributions. See the [protocol](../docs/protocol.md), [results](../docs/results.md) and [verification guide](../docs/reproducibility.md).

| Study | Frozen specification | Results | Paired counts | Verification |
|---|---|---|---|---|
| Released models and API | [Specification](test_comparison_spec.json) | [Summary](test_comparison_summary.json), [statistical analysis](test_statistical_analysis_20261008.json) | [NPZ](test_bag_counts_20261008.npz) | [Data audit](test_data_protocol_audit.json), [result audit](test_results_audit_20261008.json) |
| ONNX reduction | [Specification](onnx_reduction_test_spec.json) | [Summary](onnx_reduction_test_summary.json) | [NPZ](onnx_reduction_test_counts_74acea576df35ee8.npz) | [Audit](onnx_reduction_test_audit_20261013.json) |
| LoRA adaptation | [Specification](lora_test_spec_20261019.json) | [Summary](lora_test_summary_20261019.json) | [NPZ](lora_test_counts_20261019.npz) | [Audit](lora_test_audit_20261019.json) |
| Heuristic fusion | [Specification](heuristic_ensembles_test_spec_20261022.json) | [Summary](heuristic_ensembles_test_summary_20261022.json) | [NPZ](heuristic_ensembles_test_counts_20261022.npz) | [Audit](heuristic_ensembles_test_audit_20261023.json) |

The [SLM summary](test_slm_summary.json) and [API summary](test_llm_summary.json) contain timing and cost accounting for the released-model comparison. Its [figures](figures/) include a hash manifest.

## Development and historical evidence

| Artifact | Role |
|---|---|
| [Settings](full_validation_settings.json), [preflight](full_validation_preflight.json), [summary](full_validation_summary.json) | Supplied-entity base/large validation with separate distant and reviewed references |
| [Three-member comparison](validation_relex_glidre_comparison_fb9972e04f46.json), [revised cutoff grid](validation_mixed_cutoff_grid_v2_annotations.json) | Released encoder and ensemble development results |
| [Earlier cutoff grid](validation_mixed_cutoff_grid_v1.json), [archive](archive/) | Prior annotation conditions for historical interpretation |
| [DeepSeek v2 pilot](deepseek_flash_pilot_summary_6b356cadbc56715d.json), [v1 pilot](deepseek_flash_pilot_summary_d17dec454f4f71a9.json) | Separate development settings and prompt versions |
| [Preliminary experiments](preliminary_experiments.json) | Eight historical screening experiments and their inference-contract limitations |
| [Reduction pilot](reduction_pilot.json) | Preparation record for the fixed 100-bag validation input sample |

Development references, timings and official test results retain their own scopes. Detailed caches, execution journals and later development artifacts are kept in the local research workspace.
