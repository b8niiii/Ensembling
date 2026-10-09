"""Offline-capable NYT10m relation inference using supplied entity spans.

The prefetch stage resolves pinned local checkpoints. All subsequent stages
are offline. Entity recognition, entity thresholds, and name-based output
matching are bypassed; directed relations use the dataset's mention positions.
The completed joint-recognition run is preserved in its separate run folder.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "data/nyt10m/nyt10m_val.txt"
TRAIN = ROOT / "data/nyt10m/nyt10m_train.txt"
REL2ID = ROOT / "data/nyt10m/nyt10m_rel2id.json"
CACHE = ROOT / "data/model_cache"
RUN = ROOT / "data/pilot_runs/relex_full_validation_given_entities_v1_20261004"
PREVIOUS_RUN = ROOT / "data/pilot_runs/relex_full_validation_v1_20261002"
ADAPTER = ROOT / "source/relex_given_entities.py"
BAG_CAP = 20
RELATION_THRESHOLD = 0.5
DEVICE = "cpu"  # Same backend for both members; MPS is unavailable in this venv.
MIN_FREE_GIB = 2
REQUIRED_FILES = ("model.safetensors", "gliner_config.json", "tokenizer.json",
                  "tokenizer_config.json", "spm.model")
MODELS = (
    {
        "slug": "gliner_relex_base",
        "name": "GLiNER-relex-base",
        "repo": "knowledgator/gliner-relex-base-v1.0",
        "revision": "e6a880049a19c5cc222a7a479c32e84b0d8cdd9a",
        "weights_sha256": "7186a83eb61b067bfc9fdfbe542a07c69cd39a446be0cfec170bd83f8a5b12b5",
    },
    {
        "slug": "gliner_relex_large",
        "name": "GLiNER-relex-large",
        "repo": "knowledgator/gliner-relex-large-v1.0",
        "revision": "4aedc9226a5ac9e2f6b5ea3e91c1ee577c88a290",
        "weights_sha256": "7c5bd751e1b24e4254d70fe4355a986cd65400676ce3735f7752429fcc26960a",
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def write_once(path: Path, value: object) -> None:
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise RuntimeError(f"Frozen file differs; preserve it and use a new run version: {path}")
    else:
        save_json(path, value)


def normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        return json_safe(value.item())
    return repr(value)


def evenly_spaced_indices(count: int, cap: int) -> list[int]:
    """Use the project's existing first/last-inclusive 20-sentence selection."""
    if count <= cap:
        return list(range(count))
    den = cap - 1
    indices = [(2 * index * (count - 1) + den) // (2 * den) for index in range(cap)]
    if len(set(indices)) != cap or indices[0] != 0 or indices[-1] != count - 1:
        raise RuntimeError("Non-unique or incomplete sentence selection")
    return indices


def snapshot_path(spec: dict) -> Path:
    repo_cache_name = "models--" + spec["repo"].replace("/", "--")
    return CACHE / repo_cache_name / "snapshots" / spec["revision"]


def local_weights(spec: dict) -> Path:
    path = snapshot_path(spec)
    missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{spec['name']} missing local files {missing} in {path}")
    actual = sha256(path / "model.safetensors")
    if actual != spec["weights_sha256"]:
        raise RuntimeError(f"{spec['name']} weight hash mismatch: {actual}")
    return path


def prefetch() -> None:
    """The sole stage permitted to contact Hugging Face; skip fully local models."""
    for spec in MODELS:
        try:
            path = local_weights(spec)
            print(f"LOCAL: {spec['name']} → {path}", flush=True)
            continue
        except FileNotFoundError:
            pass
        from huggingface_hub import snapshot_download
        print(f"Downloading/checking {spec['repo']} at {spec['revision']}...", flush=True)
        snapshot_download(repo_id=spec["repo"], revision=spec["revision"],
                          cache_dir=CACHE, max_workers=2)
        print(f"VERIFIED: {spec['name']} → {local_weights(spec)}", flush=True)
    for spec in MODELS:
        local_weights(spec)
    print("Both pinned model snapshots are complete and hash-verified.", flush=True)


def require_offline() -> None:
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("Launch this stage with HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1")


def offline_check() -> None:
    """Verify local loading, supplied-span inference, and all selected boundaries."""
    require_offline()
    from gliner import GLiNER
    from relex_given_entities import verify_supplied_span_adapter, align_supplied_sentence
    from run_extended_gliner_relex import ENTITY_LABELS, NAME_LABEL
    manifest, settings = prepare()
    checks = {}
    for spec in MODELS:
        model = GLiNER.from_pretrained(str(local_weights(spec))).to(DEVICE).eval()
        try:
            verification = verify_supplied_span_adapter(model, ENTITY_LABELS, NAME_LABEL)
            total = max_words = max_entity_words = refined_sentences = 0
            for bag in manifest:
                for sentence, positions in zip(bag["sentences"], bag["entity_positions"], strict=True):
                    prepared, _ = align_supplied_sentence(
                        model, sentence, positions["head"], positions["tail"], ENTITY_LABELS, list(NAME_LABEL),
                    )
                    max_words = max(max_words, len(prepared["tokens"][0]))
                    max_entity_words = max(max_entity_words, max(
                        end - start + 1 for start, end in prepared["word_input_spans"][0]))
                    refined_sentences += prepared["boundary_refined"]
                    total += 1
            checks[spec["slug"]] = {**verification, "selected_sentences_checked": total,
                                    "max_sentence_words": max_words,
                                    "max_entity_words": max_entity_words,
                                    "sentences_with_word_boundary_refinement": refined_sentences,
                                    "span_mode": model.config.span_mode}
            print(f"OFFLINE SUPPLIED-SPAN CHECK OK: {spec['name']}; "
                  f"{total:,} sentences; native relation parity; NER bypassed", flush=True)
        finally:
            del model
            gc.collect()
    save_json(RUN / "offline_preflight.json", {"run_fingerprint": settings["fingerprint"],
                                              "models": checks})
    print("OFFLINE READY: both checkpoints and all supplied boundaries verified.", flush=True)


def read_pairs(path: Path) -> set[tuple[str, str]]:
    pairs = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                pairs.add((row["h"]["id"], row["t"]["id"]))
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path.name}:{line_number}: malformed record") from exc
    return pairs


def build_manifest() -> list[dict]:
    rel2id = json.loads(REL2ID.read_text(encoding="utf-8"))
    if len(rel2id) != 25 or "NA" not in rel2id:
        raise RuntimeError("Unexpected NYT10m relation inventory")
    train_pairs = read_pairs(TRAIN)
    grouped = defaultdict(list)
    with VAL.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                pair = (row["h"]["id"], row["t"]["id"])
                if row["relation"] not in rel2id:
                    raise ValueError("unknown relation")
                grouped[pair].append(row)
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{VAL.name}:{line_number}: malformed record") from exc
    manifest = []
    for pair in sorted(grouped):
        records = grouped[pair]
        if any((r["h"]["name"], r["t"]["name"]) !=
               (records[0]["h"]["name"], records[0]["t"]["name"]) for r in records):
            raise RuntimeError(f"Inconsistent entity names for ordered pair {pair}")
        indices = evenly_spaced_indices(len(records), BAG_CAP)
        positions = []
        for index in indices:
            row = records[index]
            for role in ("h", "t"):
                start, end = row[role]["pos"]
                if (type(start) is not int or type(end) is not int or
                        not 0 <= start < end <= len(row["text"]) or
                        row["text"][start:end] != row[role]["name"]):
                    raise ValueError(f"Invalid dataset entity span: {pair}, record {index}")
            if row["h"]["pos"] == row["t"]["pos"]:
                raise ValueError(f"Identical head and tail spans: {pair}")
            positions.append({"head": row["h"]["pos"], "tail": row["t"]["pos"]})
        manifest.append({
            "pair_ids": list(pair), "head": records[0]["h"]["name"],
            "tail": records[0]["t"]["name"],
            "full_bag_size": len(records),
            "selected_indices_zero_based": indices,
            "sentences": [records[index]["text"] for index in indices],
            "entity_positions": positions,
            "ds_positive_labels": sorted({r["relation"] for r in records} - {"NA"}),
            "ordered_pair_seen_in_train": pair in train_pairs,
        })
    if len(manifest) != 36266 or sum(b["full_bag_size"] for b in manifest) != 46422:
        raise RuntimeError("NYT10m validation counts differ from the recorded EDA")
    if sum(len(b["sentences"]) for b in manifest) != 44751:
        raise RuntimeError("20-sentence selection differs from the recorded protocol")
    return manifest


def settings_for(manifest: list[dict]) -> dict:
    from run_extended_gliner_relex import ENTITY_LABELS, LABEL_NAME
    from relex_given_entities import GLINER_VERSION
    import gliner
    rel2id = json.loads(REL2ID.read_text(encoding="utf-8"))
    if set(LABEL_NAME) != set(rel2id) - {"NA"}:
        raise RuntimeError("Relation descriptions do not cover the 24 positive labels")
    versions = {package: importlib.metadata.version(package) for package in
                ("gliner", "torch", "transformers", "huggingface_hub")}
    if versions["gliner"] != GLINER_VERSION:
        raise RuntimeError(f"This inference adapter requires gliner=={GLINER_VERSION}")
    gliner_root = Path(gliner.__file__).parent
    return {
        "purpose": "Full NYT10m validation using supplied head/tail spans without NER",
        "inference_mode": "supplied_dataset_spans_no_entity_recognition",
        "adapter_sha256": sha256(ADAPTER),
        "label_module_sha256": sha256(ROOT / "source/run_extended_gliner_relex.py"),
        "gliner_implementation_sha256": {name: sha256(gliner_root / name) for name in
                                         ("model.py", "modeling/base.py", "modeling/span_rep.py")},
        "validation_sha256": sha256(VAL), "train_sha256": sha256(TRAIN),
        "rel2id_sha256": sha256(REL2ID),
        "runner_sha256": sha256(Path(__file__)),
        "manifest_count": len(manifest), "bag_cap": BAG_CAP,
        "selection": "same first-and-last-inclusive evenly spaced record indices as extended pilot",
        "entity_threshold": None,
        "primary_relation_threshold": RELATION_THRESHOLD,
        "saved_scores": "all 24 directed positive-relation scores per sentence and bag maxima",
        "entity_labels": ENTITY_LABELS, "label_name": LABEL_NAME,
        "mapping": "dataset half-open character spans; head index 0 to tail index 1; union over sentences",
        "entity_labels_role": "unchanged encoder prompts only; no entity-type classification",
        "boundary_alignment": "split model words at supplied mention boundaries; sentence text unchanged",
        "device": DEVICE, "models": list(MODELS), "versions": versions,
        "manual_test_access": False,
    }


def prepare() -> tuple[list[dict], dict]:
    manifest = build_manifest()
    frozen = settings_for(manifest)
    fingerprint = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    settings = {"fingerprint": fingerprint, "settings": frozen}
    write_once(RUN / "manifest.json", manifest)
    write_once(RUN / "settings.json", settings)
    print(f"Prepared {len(manifest):,} validation bags / "
          f"{sum(len(b['sentences']) for b in manifest):,} selected sentence records; "
          f"run {fingerprint[:16]}", flush=True)
    print(f"Train-seen ordered pairs: {sum(b['ordered_pair_seen_in_train'] for b in manifest):,}",
          flush=True)
    return manifest, settings


def bag_result_path(spec: dict, bag: dict) -> Path:
    pair_token = "|".join(bag["pair_ids"])
    short_hash = hashlib.sha256(pair_token.encode()).hexdigest()[:20]
    return RUN / spec["slug"] / "bags" / (short_hash + ".json")


def valid_saved(path: Path, spec: dict, bag: dict, fingerprint: str) -> bool:
    if not path.exists():
        return False
    row = json.loads(path.read_text(encoding="utf-8"))
    if (row.get("model") != spec["name"] or row.get("pair_ids") != bag["pair_ids"]
            or row.get("run_fingerprint") != fingerprint):
        raise RuntimeError(f"Existing result does not belong to this frozen run: {path}")
    if row.get("status") != "valid":
        return False
    from run_extended_gliner_relex import NAME_LABEL
    scores = row.get("relation_score_max", {})
    if (set(scores) != set(NAME_LABEL.values()) or
            row.get("sentences_processed") != len(bag["sentences"])):
        raise RuntimeError(f"Incomplete supplied-span score cache: {path}")
    import math
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or
           not 0 <= value <= 1 for value in scores.values()):
        raise RuntimeError(f"Invalid supplied-span score cache: {path}")
    return True


def run_one(spec: dict, manifest: list[dict], settings: dict) -> None:
    require_offline()
    from gliner import GLiNER
    import torch
    from run_extended_gliner_relex import ENTITY_LABELS, NAME_LABEL
    from relex_given_entities import install_supplied_span_adapter, score_supplied_sentence, labels_at_threshold

    fingerprint = settings["fingerprint"]
    pending = [bag for bag in manifest if not valid_saved(
        bag_result_path(spec, bag), spec, bag, fingerprint)]
    print(f"{spec['name']}: {len(pending):,}/{len(manifest):,} bags pending", flush=True)
    if not pending:
        return
    if shutil.disk_usage(ROOT).free < MIN_FREE_GIB * 1024**3:
        raise RuntimeError("Less than 2 GiB free; stop before writing more results")
    weights = local_weights(spec)
    load_started = time.perf_counter()
    model = GLiNER.from_pretrained(str(weights)).to(DEVICE).eval()
    install_supplied_span_adapter(model)
    save_json(RUN / spec["slug"] / "model.json", {
        "repo": spec["repo"], "revision": spec["revision"],
        "weights_sha256": spec["weights_sha256"], "device": DEVICE,
        "load_wall_s": round(time.perf_counter() - load_started, 3),
        "weight_bytes": (weights / "model.safetensors").stat().st_size,
    })
    consecutive_failures = 0
    run_started = time.perf_counter()
    try:
        for index, bag in enumerate(pending, 1):
            path = bag_result_path(spec, bag)
            previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
            started = time.perf_counter()
            try:
                scores_max = {label: 0.0 for label in NAME_LABEL.values()}
                evidence = []
                with torch.inference_mode():
                    for sentence_index, (sentence, positions) in enumerate(zip(
                            bag["sentences"], bag["entity_positions"], strict=True)):
                        scores, word_spans = score_supplied_sentence(
                            model, sentence, positions["head"], positions["tail"],
                            ENTITY_LABELS, NAME_LABEL,
                        )
                        for label, value in scores.items():
                            scores_max[label] = max(scores_max[label], value)
                        evidence.append({"selected_sentence_index": sentence_index,
                                         "source_record_index": bag["selected_indices_zero_based"][sentence_index],
                                         "head_char_span": positions["head"],
                                         "tail_char_span": positions["tail"],
                                         "head_word_span_inclusive": list(word_spans[0]),
                                         "tail_word_span_inclusive": list(word_spans[1]),
                                         "relation_scores": scores})
                result = {"status": "valid", "relations": labels_at_threshold(scores_max, RELATION_THRESHOLD),
                          "relation_score_max": scores_max, "evidence": evidence,
                          "sentences_processed": len(bag["sentences"])}
                consecutive_failures = 0
            except Exception as exc:
                history = list(previous.get("failures", [])) if previous else []
                history.append(f"{type(exc).__name__}: {exc}")
                result = {"status": "failed", "relations": [], "evidence": [],
                          "sentences_processed": 0, "failures": history}
                consecutive_failures += 1
            result.update({
                "model": spec["name"], "pair_ids": bag["pair_ids"],
                "run_fingerprint": fingerprint,
                "full_bag_size": bag["full_bag_size"],
                "selected_indices_zero_based": bag["selected_indices_zero_based"],
                "wall_s": round(time.perf_counter() - started, 3),
                "created_utc": datetime.now(timezone.utc).isoformat(),
            })
            save_json(path, result)
            if index == 1 or index % 250 == 0 or index == len(pending):
                elapsed = time.perf_counter() - run_started
                print(f"  {spec['name']}: {index:,}/{len(pending):,} pending bags processed "
                      f"({elapsed / 3600:.2f} h); last={result['status']}", flush=True)
                if shutil.disk_usage(ROOT).free < MIN_FREE_GIB * 1024**3:
                    raise RuntimeError("Less than 2 GiB free; saved outputs can be resumed later")
            if consecutive_failures >= 3:
                raise RuntimeError(f"Three consecutive failures for {spec['name']}; inspect saved rows")
    finally:
        del model
        gc.collect()
    print(f"Finished {spec['name']}; released model before loading the next.", flush=True)


def run_all() -> None:
    require_offline()
    manifest, settings = prepare()
    preflight = RUN / "offline_preflight.json"
    if (not preflight.is_file() or
            json.loads(preflight.read_text())["run_fingerprint"] != settings["fingerprint"]):
        raise RuntimeError("Matching offline supplied-span preflight is required before inference")
    for spec in MODELS:
        run_one(spec, manifest, settings)
    print("Both models complete. The report stage reads saved files only.", flush=True)


def score(bags: list[dict], gold: dict, predictions: dict) -> dict:
    tp = fp = fn = negative_bags_with_fp = 0
    for bag in bags:
        pair = tuple(bag["pair_ids"])
        expected = set(gold[pair])
        actual = set(predictions[pair])
        tp += len(expected & actual)
        fp += len(actual - expected)
        fn += len(expected - actual)
        negative_bags_with_fp += bool(not expected and actual)
    return {
        "bags": len(bags), "TP": tp, "FP": fp, "FN": fn,
        "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
        "recall": round(tp / (tp + fn), 4) if tp + fn else 0.0,
        "micro_f1": round(2 * tp / (2 * tp + fp + fn), 4) if 2 * tp + fp + fn else 0.0,
        "negative_bags_with_fp": negative_bags_with_fp,
    }


def report() -> None:
    """Score full validation distant labels and reviewed 30 separately."""
    manifest, settings = prepare()
    fingerprint = settings["fingerprint"]
    from relex_given_entities import labels_at_threshold
    predictions, score_cache, wall_s = {}, {}, {}
    for spec in MODELS:
        by_pair, by_pair_scores, total_wall = {}, {}, 0.0
        for bag in manifest:
            path = bag_result_path(spec, bag)
            if not valid_saved(path, spec, bag, fingerprint):
                raise RuntimeError(f"Incomplete result for {spec['name']} and {bag['pair_ids']}")
            row = json.loads(path.read_text(encoding="utf-8"))
            pair = tuple(bag["pair_ids"])
            by_pair_scores[pair] = row["relation_score_max"]
            by_pair[pair] = labels_at_threshold(by_pair_scores[pair], RELATION_THRESHOLD)
            total_wall += row["wall_s"]
        predictions[spec["slug"]] = by_pair
        score_cache[spec["slug"]] = by_pair_scores
        wall_s[spec["slug"]] = round(total_wall, 2)

    base = predictions["gliner_relex_base"]
    large = predictions["gliner_relex_large"]
    pairs = [tuple(b["pair_ids"]) for b in manifest]
    systems = {
        "relex_base": base,
        "relex_large": large,
        "intersection": {p: sorted(set(base[p]) & set(large[p])) for p in pairs},
        "union": {p: sorted(set(base[p]) | set(large[p])) for p in pairs},
    }
    ds_gold = {tuple(b["pair_ids"]): b["ds_positive_labels"] for b in manifest}
    cohorts = {
        "all_validation_distant_labels": manifest,
        "distant_NA_only": [b for b in manifest if not b["ds_positive_labels"]],
        "distant_positive": [b for b in manifest if b["ds_positive_labels"]],
        "pair_seen_in_train": [b for b in manifest if b["ordered_pair_seen_in_train"]],
        "pair_not_seen_in_train": [b for b in manifest if not b["ordered_pair_seen_in_train"]],
        "singleton_bags": [b for b in manifest if b["full_bag_size"] == 1],
        "bags_with_2_to_20_records": [b for b in manifest if 2 <= b["full_bag_size"] <= 20],
        "bags_above_20_records": [b for b in manifest if b["full_bag_size"] > 20],
    }
    summary = {
        "run_fingerprint": fingerprint,
        "inference_mode": "supplied_dataset_spans_no_entity_recognition",
        "primary_relation_threshold": RELATION_THRESHOLD,
        "reference_warning": "Full validation relation labels are distant supervision, not manual gold",
        "inference_wall_s_by_model": wall_s,
        "distant_label_scores": {name: {system: score(bags, ds_gold, pred)
                                       for system, pred in systems.items()}
                                 for name, bags in cohorts.items()},
    }

    # The original 30 text-reviewed references are kept separate from distant labels.
    reviewed_dir = ROOT / "data/pilot_runs/reviewed_validation_30_encoder_v1_20261001"
    reviewed_manifest = json.loads((reviewed_dir / "manifest.json").read_text(encoding="utf-8"))
    reviewed_labels = json.loads((reviewed_dir / "reviewed_reference.json").read_text(encoding="utf-8"))
    full_by_pair = {tuple(b["pair_ids"]): b for b in manifest}
    reviewed_bags, human_gold = [], {}
    for item in reviewed_manifest:
        pair = tuple(item["pair_ids"])
        full_bag = full_by_pair[pair]
        if full_bag["sentences"] != item["sentences"]:
            raise RuntimeError(f"Reviewed and full-run selected texts differ for {item['sample_id']}")
        reviewed_bags.append(full_bag)
        human_gold[pair] = reviewed_labels[item["sample_id"]]
    summary["reviewed_30_text_scores"] = {
        system: score(reviewed_bags, human_gold, pred) for system, pred in systems.items()
    }

    # These fixed diagnostic cutoffs reuse all saved relation scores. No model
    # inference or automatic threshold selection occurs during reporting.
    summary["threshold_diagnostics"] = {}
    for cutoff in (0.5, 0.7, 0.9):
        at_cutoff = {slug: {pair: labels_at_threshold(values, cutoff)
                            for pair, values in by_pair.items()}
                     for slug, by_pair in score_cache.items()}
        b, l = at_cutoff["gliner_relex_base"], at_cutoff["gliner_relex_large"]
        combined = {"relex_base": b, "relex_large": l,
                    "intersection": {pair: sorted(set(b[pair]) & set(l[pair])) for pair in pairs},
                    "union": {pair: sorted(set(b[pair]) | set(l[pair])) for pair in pairs}}
        summary["threshold_diagnostics"][str(cutoff)] = {
            "all_validation_distant_labels": {system: score(manifest, ds_gold, pred)
                                               for system, pred in combined.items()},
            "reviewed_30_text": {system: score(reviewed_bags, human_gold, pred)
                                 for system, pred in combined.items()},
        }

    # Retain the prior completed experiment as a distinct protocol comparison.
    prior_summary = PREVIOUS_RUN / "summary.json"
    if prior_summary.is_file():
        prior_settings = json.loads((PREVIOUS_RUN / "settings.json").read_text())
        prior = json.loads(prior_summary.read_text())
        previous = prior_settings["settings"]
        current = settings["settings"]
        for key in ("validation_sha256", "train_sha256", "rel2id_sha256", "bag_cap", "selection", "label_name", "entity_labels", "models", "device", "versions"):
            if previous[key] != current[key]:
                raise RuntimeError(f"Prior joint experiment is not comparable: {key}")
        if (previous["relation_threshold"] != RELATION_THRESHOLD or
                prior["run_fingerprint"] != prior_settings["fingerprint"]):
            raise RuntimeError("Prior report settings do not match the primary comparison")
        summary["previous_joint_run"] = {
            "run_directory": str(PREVIOUS_RUN.relative_to(ROOT)),
            "run_fingerprint": prior["run_fingerprint"], "entity_threshold": previous["entity_threshold"],
            "relation_threshold": previous["relation_threshold"],
            "all_validation_distant_labels": prior["distant_label_scores"]["all_validation_distant_labels"],
            "reviewed_30_text": prior["reviewed_30_text_scores"],
            "inference_wall_s_by_model": prior["inference_wall_s_by_model"],
        }

    disagreements = []
    for bag in manifest:
        pair = tuple(bag["pair_ids"])
        if set(base[pair]) != set(large[pair]):
            disagreements.append({
                "pair_ids": bag["pair_ids"], "head": bag["head"], "tail": bag["tail"],
                "full_bag_size": bag["full_bag_size"],
                "ordered_pair_seen_in_train": bag["ordered_pair_seen_in_train"],
                "ds_positive_labels": bag["ds_positive_labels"],
                "base_relations": base[pair], "large_relations": large[pair],
                "sentences": bag["sentences"],
            })
    summary["disagreeing_bags"] = len(disagreements)
    save_json(RUN / "summary.json", summary)
    save_json(RUN / "disagreements.json", disagreements)

    rows = []
    for system, pred in systems.items():
        for relation in sorted(set(json.loads(REL2ID.read_text(encoding="utf-8"))) - {"NA"}):
            tp = fp = fn = 0
            for pair in pairs:
                g, p = relation in ds_gold[pair], relation in pred[pair]
                tp += g and p
                fp += not g and p
                fn += g and not p
            rows.append({"system": system, "relation": relation, "TP": tp, "FP": fp,
                         "FN": fn, "micro_f1_for_relation":
                         round(2 * tp / (2 * tp + fp + fn), 4) if 2 * tp + fp + fn else 0.0})
    with (RUN / "per_relation_distant_scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("Full-validation scores below use noisy distant labels; reviewed-30 scores use text labels.")
    print(json.dumps({"all_validation": summary["distant_label_scores"]["all_validation_distant_labels"],
                      "reviewed_30": summary["reviewed_30_text_scores"],
                      "disagreeing_bags": len(disagreements),
                      "inference_wall_s_by_model": wall_s}, indent=2), flush=True)
    print(f"Saved report and disagreements in {RUN}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prefetch", "prepare", "offline-check", "run", "report"))
    args = parser.parse_args()
    if Path(sys.executable).resolve() != (ROOT / ".venv/bin/python").resolve():
        raise RuntimeError(f"Select the project .venv Python; found {sys.executable}")
    if args.stage == "prefetch":
        prefetch()
    elif args.stage == "prepare":
        for spec in MODELS:
            local_weights(spec)
        prepare()
    elif args.stage == "offline-check":
        offline_check()
    elif args.stage == "run":
        run_all()
    else:
        report()


if __name__ == "__main__":
    main()
