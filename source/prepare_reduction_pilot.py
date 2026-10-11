"""Reproduce the fixed validation-only reduction inputs without model execution.

The former preparation workflow also created an abandoned review packet.
This entry point retains only the uniformly sampled 100-bag input pilot.
Existing inputs are verified and never overwritten with different content.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data/pilot_runs/relex_full_validation_given_entities_v1_20261004"
DESTINATION = ROOT / "data/reduction_pilot/validation_100_20261006"
SEED = 20261006


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_stable(path: Path, value: object) -> None:
    content = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise RuntimeError(f"Existing inputs differ; choose a new version: {path}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def prepare() -> None:
    settings = json.loads((RUN / "settings.json").read_text())
    published = json.loads((ROOT / "results/full_validation_settings.json").read_text())
    fingerprint = hashlib.sha256(json.dumps(settings["settings"], sort_keys=True).encode()).hexdigest()
    manifest = json.loads((RUN / "manifest.json").read_text())
    if settings != published or settings["fingerprint"] != fingerprint:
        raise RuntimeError("Local and published validation identities differ")
    if len(manifest) != 36266 or len({tuple(b["pair_ids"]) for b in manifest}) != 36266:
        raise RuntimeError("Expected the complete distinct validation-bag inventory")
    chosen = random.Random(SEED).sample(sorted(manifest, key=lambda b: tuple(b["pair_ids"])), 100)
    fields = ("pair_ids", "head", "tail", "full_bag_size", "sentences",
              "selected_indices_zero_based", "entity_positions")
    inputs = [{key: bag[key] for key in fields} for bag in chosen]
    pilot_settings = {
        "seed": SEED, "split": "validation", "bags": 100, "bag_cap": 20,
        "selection": "uniform sampling without replacement from sorted full-validation pairs",
        "sentence_selection": settings["settings"]["selection"],
        "run_fingerprint": fingerprint, "source_manifest_sha256": digest(RUN / "manifest.json"),
        "parent_model": settings["settings"]["models"][1],
        "purpose": "CPU reduction compatibility, paired quality, time, and memory pilot",
        "status": "inputs prepared; no reduced checkpoint or inference run",
    }
    write_stable(DESTINATION / "manifest.json", inputs)
    write_stable(DESTINATION / "settings.json", pilot_settings)
    report = {
        "date": "2026-10-06", "status": pilot_settings["status"],
        "split": "validation", "bags": 100, "selected_records": sum(len(b["sentences"]) for b in inputs),
        "seed": SEED, "bag_cap": 20, "selection": pilot_settings["selection"],
        "source_run_fingerprint": fingerprint,
        "source_manifest_sha256": pilot_settings["source_manifest_sha256"],
        "manifest_sha256": digest(DESTINATION / "manifest.json"),
        "settings_sha256": digest(DESTINATION / "settings.json"),
        "parent_model": pilot_settings["parent_model"], "inference_run": False,
    }
    write_stable(ROOT / "results/reduction_pilot.json", report)
    print(f"Verified/prepared 100 validation bags / {report['selected_records']} selected records; no inference")


if __name__ == "__main__":
    prepare()
