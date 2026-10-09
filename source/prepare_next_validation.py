"""Prepare blinded review inputs and a fixed reduction pilot from validation caches.

No checkpoint loading, model inference, training input, or official test access.
Sampling provenance containing selection strata remains in the ignored data tree.
The review packet is an unannotated proposal, not an evaluation reference.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data/pilot_runs/relex_full_validation_given_entities_v1_20261004"
REVIEWED = ROOT / "data/pilot_runs/reviewed_validation_30_encoder_v1_20261001"
SEED = 20261006


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_stable(path: Path, value: object) -> None:
    """Do not overwrite an existing proposal or annotation with changed content."""
    content = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise RuntimeError(f"Existing material differs; choose a new version: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def supplied_input(bag: dict) -> dict:
    """Remove distant labels, predictions, and split-membership metadata."""
    fields = ("pair_ids", "head", "tail", "full_bag_size", "sentences",
              "selected_indices_zero_based", "entity_positions")
    return {field: bag[field] for field in fields}


def prepare() -> None:
    manifest = json.loads((RUN / "manifest.json").read_text(encoding="utf-8"))
    settings = json.loads((RUN / "settings.json").read_text(encoding="utf-8"))
    report = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
    reviewed = json.loads((REVIEWED / "manifest.json").read_text(encoding="utf-8"))
    disagreements = json.loads((RUN / "disagreements.json").read_text(encoding="utf-8"))
    if report["run_fingerprint"] != settings["fingerprint"] or len(manifest) != 36266:
        raise RuntimeError("The completed validation report and inputs do not match.")
    reviewed_pairs = {tuple(b["pair_ids"]) for b in reviewed}
    disagreeing = {tuple(b["pair_ids"]) for b in disagreements}
    if len(reviewed_pairs) != 30 or len(disagreeing) != report["disagreeing_bags"]:
        raise RuntimeError("Unexpected reviewed reference or disagreement inventory.")
    available = sorted((b for b in manifest if tuple(b["pair_ids"]) not in reviewed_pairs),
                       key=lambda b: tuple(b["pair_ids"]))
    pools = {
        "distant_NA_disagreement": [b for b in available if not b["ds_positive_labels"]
                                   and tuple(b["pair_ids"]) in disagreeing],
        "distant_positive_disagreement": [b for b in available if b["ds_positive_labels"]
                                         and tuple(b["pair_ids"]) in disagreeing],
    }
    rng = random.Random(SEED)
    chosen, strata = [], {}
    for stratum, pool in pools.items():
        if len(pool) < 8:
            raise RuntimeError(f"Insufficient bags in {stratum}")
        for bag in rng.sample(pool, 8):
            chosen.append(bag)
            strata[tuple(bag["pair_ids"])] = stratum
    chosen_pairs = {tuple(b["pair_ids"]) for b in chosen}
    remainder = [b for b in available if tuple(b["pair_ids"]) not in chosen_pairs]
    for bag in rng.sample(remainder, 4):
        chosen.append(bag)
        strata[tuple(bag["pair_ids"])] = "random_remaining_validation"
    rng.shuffle(chosen)
    packet = [{"sample_id": f"A{i:02d}", **supplied_input(bag)}
              for i, bag in enumerate(chosen, 1)]
    if len({tuple(b["pair_ids"]) for b in packet}) != 20:
        raise RuntimeError("Duplicate audit pair")
    if reviewed_pairs & {tuple(b["pair_ids"]) for b in packet}:
        raise RuntimeError("Audit overlaps the reviewed 30")
    forbidden = {"ds_positive_labels", "relations", "base_relations", "large_relations",
                 "ordered_pair_seen_in_train", "stratum"}
    if any(forbidden & set(bag) for bag in packet):
        raise RuntimeError("Blinding failure")
    write_stable(ROOT / "support/validation_audit_20_2026-10-06.json", packet)
    write_stable(ROOT / "data/validation_audits/audit_20_20261006/selection.json", {
        "seed": SEED, "cutoff": report["primary_relation_threshold"],
        "run_fingerprint": settings["fingerprint"],
        "manifest_sha256": digest(RUN / "manifest.json"),
        "disagreements_sha256": digest(RUN / "disagreements.json"),
        "pool_sizes": {name: len(pool) for name, pool in pools.items()},
        "selection": [{"sample_id": b["sample_id"], "pair_ids": b["pair_ids"],
                       "stratum": strata[tuple(b["pair_ids"])]} for b in packet],
    })
    lines = ["# Validation audit: 20 blinded bag proposals", "",
             "**Status:** unannotated; confirmation is required before reference scoring. "
             "Only the ordered entity pair and selected sentence texts are shown. "
             "Predictions, distant labels, and selection strata are withheld.", "",
             "Annotate all explicit positive relations for head → tail from the supplied texts. "
             "Use `[]` when none is expressed. Do not infer unstated world knowledge. "
             "List uncertain cases separately. The complete allowed inventory is in "
             "[protocol.md](protocol.md#annotation-conventions).", ""]
    for bag in packet:
        lines.extend([f"## {bag['sample_id']}: {bag['head']} → {bag['tail']}", "",
                      f"Ordered IDs: `{bag['pair_ids'][0]}` → `{bag['pair_ids'][1]}`. "
                      f"Selected records: {len(bag['sentences'])}/{bag['full_bag_size']}.", ""])
        for i, sentence in enumerate(bag["sentences"], 1):
            lines.extend([f"**Sentence {i}:** {sentence}", ""])
        lines.extend(["**Proposed relations:** pending review.", "",
                      "**Evidence / uncertainty:** pending review.", ""])
    review_path = ROOT / "support/validation_audit_20_review_2026-10-06.md"
    # Preserve any human edits if this preparation script is run again.
    if not review_path.exists():
        review_path.write_text("\n".join(lines), encoding="utf-8")

    # Use an independent seeded stream for an unstratified compatibility pilot.
    random_100 = random.Random(SEED).sample(
        sorted(manifest, key=lambda b: tuple(b["pair_ids"])), 100)
    reduction_dir = ROOT / "data/reduction_pilot/validation_100_20261006"
    write_stable(reduction_dir / "manifest.json", [supplied_input(b) for b in random_100])
    write_stable(reduction_dir / "settings.json", {
        "seed": SEED, "split": "validation", "bags": 100, "bag_cap": 20,
        "selection": "uniform sampling without replacement from sorted full-validation pairs",
        "sentence_selection": settings["settings"]["selection"],
        "run_fingerprint": settings["fingerprint"],
        "source_manifest_sha256": digest(RUN / "manifest.json"),
        "parent_model": settings["settings"]["models"][1],
        "purpose": "CPU reduction compatibility, paired quality, time, and memory pilot",
        "status": "inputs prepared; no reduced checkpoint or inference run",
    })
    write_stable(ROOT / "results/next_step_preparation.json", {
        "date": "2026-10-06", "seed": SEED,
        "audit": {"bags": 20, "selection_counts": {"distant_NA_disagreement": 8,
                  "distant_positive_disagreement": 8, "random_remaining_validation": 4},
                  "selected_records": sum(len(b["sentences"]) for b in packet),
                  "review_status": "pending confirmation", "excludes_reviewed_30": True,
                  "inputs_sha256": digest(ROOT / "support/validation_audit_20_2026-10-06.json")},
        "reduction_pilot": {"bags": 100, "bag_cap": 20,
                            "selected_records": sum(len(b["sentences"]) for b in random_100),
                            "manifest_sha256": digest(reduction_dir / "manifest.json"),
                            "inference_run": False},
        "source_run_fingerprint": settings["fingerprint"],
    })
    print(f"Prepared {len(packet)} blinded review bags and {len(random_100)} reduction-pilot bags.")
    print(f"Review packet: {review_path.relative_to(ROOT)}")


if __name__ == "__main__":
    prepare()
