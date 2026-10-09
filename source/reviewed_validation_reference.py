"""Read the current human-confirmed validation annotations, without model imports.

Experiment-local reviewed_reference.json files remain historical snapshots.
Active reports use the public annotation files and record their provenance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REFERENCE_VERSION = "text_supported_inference_v2_20261006"
ANNOTATION_FILES = (
    "support/validation_clean_annotations_2026-09-29.json",
    "support/validation_expanded_18_annotations_2026-10-01.json",
)


def load_reviewed_reference(root: Path, manifest: list[dict] | None = None) -> tuple[dict, dict]:
    """Validate sample identity and return the current C/N label map and metadata."""
    root = Path(root)
    labels, inputs, sources = {}, {}, {}
    # The label inventory is published in the frozen settings; no dataset/test read.
    settings = json.loads((root / "results/full_validation_settings.json").read_text())
    allowed = set(settings["settings"]["label_name"])
    for relative in ANNOTATION_FILES:
        path = root / relative
        raw = path.read_bytes()
        annotation = json.loads(raw)
        if (annotation.get("review_status") != "human_confirmed"
                or annotation.get("annotation_version") != REFERENCE_VERSION):
            raise RuntimeError(f"Unconfirmed or unexpected annotation version: {relative}")
        sample_path = root / annotation["sample_file"]
        sample_raw = sample_path.read_bytes()
        if hashlib.sha256(sample_raw).hexdigest() != annotation["sample_sha256"]:
            raise RuntimeError(f"Annotation sample changed: {sample_path}")
        sample = json.loads(sample_raw)
        sample_items = {item["sample_id"]: item for item in sample["items"]}
        if len(sample_items) != len(sample["items"]):
            raise RuntimeError(f"Duplicate sample ID: {sample_path}")
        annotated_ids = set()
        for item in annotation["annotations"]:
            sample_id = item["sample_id"]
            if sample_id in labels or sample_id not in sample_items:
                raise RuntimeError(f"Duplicate or missing reviewed sample: {sample_id}")
            source = sample_items[sample_id]
            if item["pair_ids"] != source["pair_ids"]:
                raise RuntimeError(f"Annotation pair differs from supplied sample: {sample_id}")
            relations = item.get("relations", item.get("proposed_relations"))
            if (not isinstance(relations, list) or any(not isinstance(r, str) for r in relations)
                    or len(set(relations)) != len(relations) or set(relations) - allowed):
                raise RuntimeError(f"Invalid reviewed relations: {sample_id}")
            labels[sample_id] = sorted(relations)
            inputs[sample_id] = source
            annotated_ids.add(sample_id)
        if annotated_ids != set(sample_items):
            raise RuntimeError(f"Incomplete annotations: {relative}")
        sources[relative] = hashlib.sha256(raw).hexdigest()
        sources[str(sample_path.relative_to(root))] = hashlib.sha256(sample_raw).hexdigest()
    expected_ids = {f"C{i:02d}" for i in range(1, 13)} | {f"N{i:02d}" for i in range(1, 19)}
    if set(labels) != expected_ids or len({tuple(i["pair_ids"]) for i in inputs.values()}) != 30:
        raise RuntimeError("Expected 30 distinct reviewed C/N bags")
    if manifest is not None:
        manifest_ids = [item["sample_id"] for item in manifest]
        if len(manifest_ids) != 30 or set(manifest_ids) != expected_ids:
            raise RuntimeError("Historical reviewed manifest has an unexpected inventory")
        for item in manifest:
            source = inputs[item["sample_id"]]
            if any(item[key] != source[key] for key in ("pair_ids", "head", "tail", "sentences")):
                raise RuntimeError(f"Reviewed input differs from annotation sample: {item['sample_id']}")
    encoded = json.dumps(labels, sort_keys=True, separators=(",", ":")).encode()
    # Compute the change inventory from the retained original labels rather than
    # hard-coding N18: subsequent confirmed edits also change report provenance.
    original_path = root / "data/pilot_runs/reviewed_validation_30_encoder_v1_20261001/reviewed_reference.json"
    archive = root / "support/archive/annotations_before_inference_revision_2026-10-06"
    archived_files = [archive / Path(relative).name for relative in ANNOTATION_FILES]
    changed = None
    if all(path.is_file() for path in archived_files):
        # The published annotation snapshots make metadata identical in a
        # portable checkout and a local experiment with detailed caches.
        original = {}
        for path in archived_files:
            for item in json.loads(path.read_text())["annotations"]:
                original[item["sample_id"]] = item.get("relations", item.get("proposed_relations"))
    elif original_path.is_file():
        original = json.loads(original_path.read_text(encoding="utf-8"))
    else:
        original = None
    if original is not None:
        if set(original) != expected_ids:
            raise RuntimeError("Original reviewed reference has an unexpected inventory")
        changed = sorted(key for key in labels if set(labels[key]) != set(original[key]))
    metadata = {
        "version": REFERENCE_VERSION,
        "labels_sha256": hashlib.sha256(encoded).hexdigest(),
        "source_hashes": sources,
        "confirmed_date": "2026-10-06",
        "policy": "Text-supported relations including documented motivated inference; external facts kept separate.",
        "changed_samples_from_original_reviewed30": changed,
        "historical_B08_in_current_cohort": False,
    }
    return labels, metadata
