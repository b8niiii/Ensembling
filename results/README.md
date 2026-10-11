# Result inventory

Start with the [results interpretation](../docs/results.md) and [analysis notebook](../source/Test_results_analysis.ipynb). JSON files retain their original names and bytes where they form frozen experimental evidence.

## Official manual-test comparison

| Artifact | Role |
|---|---|
| [test_comparison_spec.json](test_comparison_spec.json) | Frozen inputs, references, members, thresholds, API configuration and source identities |
| [test_comparison_summary.json](test_comparison_summary.json) | Primary three-system P/R/micro-F1, validity and recorded time |
| [test_slm_summary.json](test_slm_summary.json), [test_llm_summary.json](test_llm_summary.json) | Member timing and API accounting supporting the combined report |
| [test_data_protocol_audit.json](test_data_protocol_audit.json) | Official manual-row reconstruction and capped-reference counts |
| [test_results_audit_20261008.json](test_results_audit_20261008.json) | Completed-result verification, failures, accounting and per-relation counts |
| [test_statistical_analysis_20261008.json](test_statistical_analysis_20261008.json) | Paired bootstrap, voting diagnostics and cost dimensions |
| [test_bag_counts_20261008.npz](test_bag_counts_20261008.npz) | Public per-bag TP/FP/FN sufficient statistics for all three systems |
| [figures/](figures/) | PNG, PDF and SVG figures, with a hash manifest |

## Development and historical evidence

| Artifact | Role |
|---|---|
| [full_validation_settings.json](full_validation_settings.json), [full_validation_preflight.json](full_validation_preflight.json) | Frozen supplied-entity base/large validation and successful runtime checks |
| [full_validation_summary.json](full_validation_summary.json) | Two-member full-validation report with separate distant and reviewed references |
| [validation_relex_glidre_comparison_fb9972e04f46.json](validation_relex_glidre_comparison_fb9972e04f46.json) | Completed three-member validation and 90 configurations per reference |
| [validation_mixed_cutoff_grid_v2_annotations.json](validation_mixed_cutoff_grid_v2_annotations.json) | Two-member cutoff grid with the confirmed revised annotations |
| [validation_mixed_cutoff_grid_v1.json](validation_mixed_cutoff_grid_v1.json), [archive/](archive/) | Prior annotation conditions, preserved to interpret historical results |
| [deepseek_flash_pilot_summary_6b356cadbc56715d.json](deepseek_flash_pilot_summary_6b356cadbc56715d.json) | Completed v2 non-thinking/low/high development pilot |
| [deepseek_flash_pilot_summary_d17dec454f4f71a9.json](deepseek_flash_pilot_summary_d17dec454f4f71a9.json) | Earlier v1 non-thinking/high pilot with its original prompt |
| [preliminary_experiments.json](preliminary_experiments.json) | Eight historical screening experiments, including contract limitations and GLiFormer pair-matching evidence |
| [reduction_pilot.json](reduction_pilot.json) | Prepared validation-only inputs; no quantization, pruning or reduced-model result |

Distant-supervision development metrics, selected reviewed diagnostics and official manual-test metrics remain separate. Earlier joint-recognition experiments do not use the current supplied-span contract. Dataset files, per-bag model scores and request journals are local assets, rather than published aggregate reports.

## Next experiment and reporting

The [protocol](../docs/protocol.md#remaining-experimental-work) prioritises ONNX Runtime quantization of GLiNER-relex large at requested 4/8/16-bit levels; PEFT on limited modules and differentiated train-only subsets is exploratory. Quantization/adaptation outputs are pending and must be distinguished from prepared inputs. The [results write-up](../docs/results.md#deepseek-relative-reporting-and-next-experiment) derives the completed baseline's DeepSeek-relative F1 (77.62/74.38/100.00) while preserving absolute scores and uncertainty. Later variants are selected/frozen on validation before matched test reporting.
