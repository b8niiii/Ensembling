"""Run the cached GLiNER-relex specialist on the frozen extended validation manifest.

This is a separate encoder baseline. It uses the same displayed sentences and
ordered pairs as the generative comparison, with thresholds fixed before this run.
"""

import hashlib
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from gliner import GLiNER

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "data/pilot_runs/slm_extended_validation_q8_v1_20260930"
MODEL_DIR = RUN_DIR / "gliner_relex_base"
WEIGHTS = (ROOT / "data/model_cache/models--knowledgator--gliner-relex-base-v1.0/snapshots/"
           "e6a880049a19c5cc222a7a479c32e84b0d8cdd9a")
MODEL_NAME = "GLiNER-relex-base"
REPO_ID = "knowledgator/gliner-relex-base-v1.0"
ENTITY_THRESHOLD = 0.3
RELATION_THRESHOLD = 0.5
ENTITY_LABELS = ["person", "organization", "location", "other"]
LABEL_NAME = {
    "/business/company/advisors": "company has advisor",
    "/business/company/founders": "company has founder",
    "/business/company/majorshareholders": "company has major shareholder",
    "/business/company/place_founded": "company was founded in place",
    "/business/location": "business is located in place",
    "/business/person/company": "person is associated with company",
    "/film/film/featured_film_locations": "film features location",
    "/location/administrative_division/country": "administrative division belongs to country",
    "/location/country/administrative_divisions": "country has administrative division",
    "/location/country/capital": "country has capital city",
    "/location/location/contains": "location contains location",
    "/location/neighborhood/neighborhood_of": "neighborhood belongs to location",
    "/location/region/capital": "region has capital city",
    "/location/us_county/county_seat": "US county has county seat",
    "/people/deceasedperson/place_of_burial": "deceased person was buried in place",
    "/people/deceasedperson/place_of_death": "deceased person died in place",
    "/people/ethnicity/geographic_distribution": "ethnicity is distributed in location",
    "/people/person/children": "person has child",
    "/people/person/ethnicity": "person belongs to ethnicity",
    "/people/person/nationality": "person has nationality",
    "/people/person/place_lived": "person lived in place",
    "/people/person/place_of_birth": "person was born in place",
    "/people/person/religion": "person follows religion",
    "/time/event/locations": "event occurred in location",
}
NAME_LABEL = {name: label for label, name in LABEL_NAME.items()}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def normalized(value):
    return " ".join(str(value).casefold().split())


def result_path(bag):
    key = "|".join(bag["pair_ids"])
    return MODEL_DIR / "bags" / (hashlib.sha256(key.encode()).hexdigest()[:20] + ".json")


def main():
    if Path(sys.executable).absolute().parent.parent != ROOT / ".venv":
        raise RuntimeError("Use the project's .venv Python")
    if not (WEIGHTS / "model.safetensors").is_file():
        raise FileNotFoundError(WEIGHTS)
    manifest_path = RUN_DIR / "manifest.json"
    settings_path = RUN_DIR / "settings.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parent_settings = json.loads(settings_path.read_text(encoding="utf-8"))
    rel2id = json.loads((ROOT / "data/nyt10m/nyt10m_rel2id.json").read_text(encoding="utf-8"))
    if set(LABEL_NAME) != set(rel2id) - {"NA"}:
        raise RuntimeError("Relation descriptions do not cover NYT10m")
    frozen = {"model": MODEL_NAME, "revision": WEIGHTS.name,
              "parent_run_fingerprint": parent_settings["fingerprint"],
              "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "entity_threshold": ENTITY_THRESHOLD, "relation_threshold": RELATION_THRESHOLD,
              "entity_labels": ENTITY_LABELS, "label_name": LABEL_NAME,
              "linking": "exact casefolded whitespace-normalized head and tail names, directed",
              "aggregation": "union across selected sentences"}
    fingerprint = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    meta_path = MODEL_DIR / "settings.json"
    if meta_path.exists() and json.loads(meta_path.read_text(encoding="utf-8"))["fingerprint"] != fingerprint:
        raise RuntimeError("Existing specialist run has different settings")
    if not meta_path.exists():
        save_json(meta_path, {"fingerprint": fingerprint, "settings": frozen,
                              "created_utc": datetime.now(timezone.utc).isoformat()})
    pending = [bag for bag in manifest if not result_path(bag).exists()]
    print("GLiNER-relex pending bags:", len(pending), "of", len(manifest), flush=True)
    if not pending:
        return
    load_start = time.perf_counter()
    model = GLiNER.from_pretrained(str(WEIGHTS))
    import torch
    if torch.backends.mps.is_available():
        model = model.to("mps")
    save_json(MODEL_DIR / "model.json", {"repo": REPO_ID,
                                            "revision": WEIGHTS.name, "load_wall_s": round(time.perf_counter() - load_start, 2),
                                            "device": "mps" if torch.backends.mps.is_available() else "cpu",
                                            "weights_sha256": hashlib.sha256((WEIGHTS / "model.safetensors").read_bytes()).hexdigest()})
    for index, bag in enumerate(pending, 1):
        start = time.perf_counter()
        evidence, labels = [], set()
        for sentence in bag["sentences"]:
            _, relations = model.inference(texts=[sentence], labels=ENTITY_LABELS,
                                           relations=list(NAME_LABEL), threshold=ENTITY_THRESHOLD,
                                           relation_threshold=RELATION_THRESHOLD,
                                           return_relations=True, flat_ner=False)
            for relation in relations[0]:
                head = relation["head"]["text"]
                tail = relation["tail"]["text"]
                name = relation["relation"]
                kept = (normalized(head) == normalized(bag["head"])
                        and normalized(tail) == normalized(bag["tail"])
                        and name in NAME_LABEL)
                evidence.append({"sentence": sentence, "raw": relation, "kept": kept})
                if kept:
                    labels.add(NAME_LABEL[name])
        result = {"model": MODEL_NAME, "pair_ids": bag["pair_ids"], "cohort": bag["cohort"],
                  "status": "valid", "relations": sorted(labels), "evidence": evidence,
                  "sentences_processed": len(bag["sentences"]),
                  "ds_positive_labels": bag["ds_positive_labels"],
                  "full_bag_size": bag["full_bag_size"],
                  "parent_run_fingerprint": parent_settings["fingerprint"],
                  "run_fingerprint": fingerprint, "wall_s": round(time.perf_counter() - start, 3),
                  "created_utc": datetime.now(timezone.utc).isoformat()}
        save_json(result_path(bag), result)
        if index <= 4 or index % 50 == 0 or index == len(pending):
            print(f"  GLiNER-relex: {index}/{len(pending)}; labels={len(labels)}", flush=True)
    print("GLiNER-relex complete:", dict(Counter(x["cohort"] for x in manifest)), flush=True)


if __name__ == "__main__":
    main()
