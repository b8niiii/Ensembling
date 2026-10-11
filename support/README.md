# Reviewed development annotations

This directory contains machine-readable development evidence. The canonical paths and file bytes are preserved because saved reports and annotation loaders reference their identities. The experimental rules are explained in the [public protocol](../docs/protocol.md).

## Active reviewed-30 diagnostic

| Cohort | Selected inputs | Confirmed labels |
|---|---|---|
| C01–C12 | [validation_clean_sample_2026-09-29.json](validation_clean_sample_2026-09-29.json) | [validation_clean_annotations_2026-09-29.json](validation_clean_annotations_2026-09-29.json) |
| N01–N18 | [validation_expanded_18_2026-10-01.json](validation_expanded_18_2026-10-01.json) | [validation_expanded_18_annotations_2026-10-01.json](validation_expanded_18_annotations_2026-10-01.json) |

The version is `text_supported_inference_v2_20261006`. These 30 selected validation bags are a repeatedly used development diagnostic, not an independent benchmark or official test gold. The C sample's train-overlap filter does not establish that the released models never saw these sentences during training.

The [confirmed reassessment](validation_annotation_reassessment_2026-10-06.json) records evidence, assumptions, uncertainty and decisions for all 42 previously reviewed B/C/N bags. It documents N18's national-team nationality inference and historical B08's containment revision. Only N18 is in the active reviewed30; this project convention does not alter official test annotations.

## Historical evidence

The original B01–B12 diagnostic remains in [its input sample](validation_blind_sample_2026-09-28.json) and [current annotation file](validation_blind_annotations_2026-09-28.json). It is separate from C/N reviewed30.

[archive/annotations_before_inference_revision_2026-10-06/](archive/annotations_before_inference_revision_2026-10-06/) preserves the prior B/C/N annotations and reassessment proposal. Those originals support historical report interpretation, previous-version hashes and the portable annotation-change inventory. Current reports load confirmed C/N files through [reviewed_validation_reference.py](../source/reviewed_validation_reference.py), checking versions, sample identities and hashes.

Personal writing outlines, chapter drafts, reading notes and internal workflow records are kept in the ignored local workspace. They are not part of the published evidence set.

## Use in the next experiments

The [current protocol](../docs/protocol.md#remaining-experimental-work) prioritises ONNX Runtime quantization of GLiNER-relex large at requested 4/8/16-bit levels and exploratory PEFT on limited modules/differentiated training subsets. These annotations remain selected validation diagnostics; they must not become PEFT training targets or official test gold. The [DeepSeek=100 convention](../docs/protocol.md#deepseek-relative-f1) requires matched inputs/references and keeps absolute F1 visible. No annotation change is introduced by the optimization plan.
