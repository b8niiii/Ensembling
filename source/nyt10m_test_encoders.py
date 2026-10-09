"""Sequential offline test workers using the existing supplied-entity adapters.

Each checkpoint has a separate process, including GLiDRE's isolated helper
runtime. Importing this module neither loads models nor starts inference.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from nyt10m_test_protocol import (
    ROOT, CUTOFFS, MODEL_IDS, atomic_json, digest, file_hash, labels_above,
    majority, prepare_experiment, read_json, score_predictions, selected_gold,
    verify_context,
)


def model_locations():
    return {name: ROOT / "data/model_cache" /
            ("models--" + spec["repo"].replace("/", "--")) / "snapshots" / spec["revision"]
            for name, spec in MODEL_IDS.items()}


def prefetch_assets(enabled=False):
    """Optional online stage; all later encoder stages prohibit Hub access."""
    if enabled:
        from run_full_validation_glidre import prefetch
        prefetch(True, ["base", "large"])
    return [{"model": name, "path": str(path), "exists": path.is_dir()}
            for name, path in model_locations().items()]


@contextlib.contextmanager
def keep_awake(enabled=True):
    process = None
    if enabled and sys.platform == "darwin" and shutil.which("caffeinate"):
        process = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        yield
    finally:
        if process is not None:
            process.terminate()
            process.wait(timeout=10)


@contextlib.contextmanager
def exclusive_worker(context, member):
    import fcntl
    directory = context["run"] / "encoders" / member
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "worker.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def load_runtime(member, descriptions, *, for_preflight=False):
    """Reuse pinned checkpoints and the already validated relation-layer bridges."""
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("Encoder workers require explicit offline mode")
    if member == "glidre":
        from run_full_validation_glidre import verify_assets, load_model
        verify_assets()
        model, parameters = load_model()
        from glidre_given_entities import (
            encode_relation_names, prepare_sentence, score_sentence, verify_native_parity,
        )
        names = {name.upper().replace(" ", "_"): label for label, name in descriptions.items()}
        label_started = time.perf_counter()
        embeddings = encode_relation_names(model, names)
        label_time = time.perf_counter() - label_started

        def score(text, positions):
            scores, aligned = score_sentence(model, text, positions["head"], positions["tail"], names, embeddings)
            return scores, aligned["word_spans"]

        def prepare(text, positions):
            aligned, _, _ = prepare_sentence(model, text, positions["head"], positions["tail"], names, cached_labels=embeddings)
            return {"words": len(aligned["tokens"]), "subtokens": aligned["subtokens"],
                    "refined": aligned["boundary_refined"]}

        def parity():
            return verify_native_parity(model, names, embeddings)
    else:
        from run_full_validation_relex_given_entities import MODELS, local_weights
        spec = next(s for s in MODELS if s["slug"] == "gliner_relex_" + member)
        from gliner import GLiNER
        from run_extended_gliner_relex import ENTITY_LABELS, NAME_LABEL
        from relex_given_entities import (
            install_supplied_span_adapter, prepare_supplied_sentence,
            score_supplied_sentence, verify_supplied_span_adapter,
        )
        if set(NAME_LABEL.values()) != set(descriptions):
            raise RuntimeError("Encoder label inventory changed")
        model = GLiNER.from_pretrained(str(local_weights(spec))).to("cpu").eval()
        parameters = sum(p.numel() for p in model.parameters())
        label_time = 0.

        def score(text, positions):
            return score_supplied_sentence(model, text, positions["head"], positions["tail"], ENTITY_LABELS, NAME_LABEL)

        def prepare(text, positions):
            prepared, batch, _ = prepare_supplied_sentence(
                model, text, positions["head"], positions["tail"], ENTITY_LABELS, list(NAME_LABEL))
            return {"words": len(prepared["tokens"][0]),
                    "subtokens": int(batch["input_ids"].shape[-1]),
                    "refined": bool(prepared.get("boundary_refined", False))}

        def parity():
            result = verify_supplied_span_adapter(model, ENTITY_LABELS, NAME_LABEL)
            install_supplied_span_adapter(model)
            return result
        if not for_preflight:
            install_supplied_span_adapter(model)
    if not 0 < parameters < 1_000_000_000:
        raise RuntimeError("Expected a sub-billion encoder")
    return model, score, prepare, parity, {"parameters": parameters, "label_encoding_wall_s": label_time}


def preflight(context, member):
    destination = context["run"] / "encoders" / member / "preflight.json"
    if destination.exists():
        previous = read_json(destination)
        if previous.get("success") and previous.get("fingerprint") == context["fingerprint"]:
            print(f"{member}: matching test preflight retained", flush=True)
            return
    import torch
    started = time.perf_counter()
    model, score, prepare, parity, runtime = load_runtime(member, context["settings"]["label_descriptions"], for_preflight=True)
    count = refined = max_words = max_subtokens = 0
    longest = None
    try:
        with torch.inference_mode():
            parity_result = parity()
            for bag in context["manifest"]:
                for text, pos in zip(bag["sentences"], bag["entity_positions"], strict=True):
                    info = prepare(text, pos)
                    count += 1
                    refined += int(info["refined"])
                    max_words = max(max_words, info["words"])
                    max_subtokens = max(max_subtokens, info["subtokens"])
                    if longest is None or info["subtokens"] > longest[0]:
                        longest = (info["subtokens"], text, pos)
            scores, _ = score(longest[1], longest[2])
            labels_above(scores, CUTOFFS[member], context["settings"]["label_descriptions"])
        if count != context["audit"]["selected_sentence_mentions"]:
            raise RuntimeError("Incomplete test input preflight")
        atomic_json(destination, {"success": True, "fingerprint": context["fingerprint"],
                    "member": member, "selected_sentences_checked": count,
                    "boundary_refined_sentences": refined, "max_words": max_words,
                    "max_subtokens": max_subtokens, "native_parity": parity_result,
                    "longest_input_forward_pass": True, "ner_bypassed": True,
                    "wall_s": time.perf_counter() - started, **runtime})
        print(f"{member}: OFFLINE TEST PREFLIGHT OK, {count:,} supplied sentences", flush=True)
    finally:
        del model
        gc.collect()


def cache_path(context, member, bag):
    return context["run"] / "encoders" / member / "bags" / (bag["bag_id"] + ".json")


def read_encoder_result(context, member, bag):
    path = cache_path(context, member, bag)
    if not path.exists():
        return None
    row = read_json(path)
    if (row.get("fingerprint") != context["fingerprint"] or row.get("member") != member or
            row.get("input_sha256") != digest(bag) or row.get("pair_ids") != bag["pair_ids"]):
        raise RuntimeError(f"Mismatched encoder cache: {path}")
    if row.get("status") == "failed":
        return row
    if row.get("status") != "valid" or len(row.get("evidence", [])) != len(bag["sentences"]):
        raise RuntimeError("Incomplete encoder score cache")
    allowed = context["settings"]["label_descriptions"]
    maxima = {label: 0. for label in allowed}
    for i, evidence in enumerate(row["evidence"]):
        if (evidence.get("selected_sentence_index") != i or
                evidence.get("source_lines") != bag["selected_source_lines"][i] or
                evidence.get("head_char_span") != bag["entity_positions"][i]["head"] or
                evidence.get("tail_char_span") != bag["entity_positions"][i]["tail"]):
            raise RuntimeError("Cached selected mention identity changed")
        labels_above(evidence["relation_scores"], CUTOFFS[member], allowed)
        for label in maxima:
            maxima[label] = max(maxima[label], evidence["relation_scores"][label])
    if maxima != row.get("relation_score_max"):
        raise RuntimeError("Bag maxima disagree with sentence scores")
    expected = sorted(labels_above(maxima, CUTOFFS[member], allowed))
    if row.get("relations") != expected:
        raise RuntimeError("Cached cutoff predictions changed")
    if any(not isinstance(row.get(k), (int, float)) or not math.isfinite(row[k]) or row[k] < 0
           for k in ("wall_s", "total_attempt_wall_s")):
        raise RuntimeError("Invalid encoder timing")
    return row


def run_member(context, member):
    receipt = context["run"] / "encoders" / member / "preflight.json"
    if not receipt.is_file():
        raise RuntimeError("Run the offline test preflight first")
    check = read_json(receipt)
    if not check.get("success") or check.get("fingerprint") != context["fingerprint"]:
        raise RuntimeError("Preflight differs from this frozen experiment")
    pending = []
    for bag in context["manifest"]:
        row = read_encoder_result(context, member, bag)
        if row is None or row["status"] == "failed":
            pending.append((bag, row))
    print(f"{member}: {len(pending):,}/{len(context['manifest']):,} test bags pending", flush=True)
    if not pending:
        return
    if shutil.disk_usage(ROOT).free < 2 * 1024**3:
        raise RuntimeError("Less than 2 GiB free space")
    import torch
    loaded = time.perf_counter()
    model, score, _, _, runtime = load_runtime(member, context["settings"]["label_descriptions"])
    sessions_path = context["run"] / "encoders" / member / "runtime.json"
    sessions = read_json(sessions_path) if sessions_path.exists() else []
    sessions.append({"load_including_label_encoding_wall_s": time.perf_counter() - loaded, **runtime})
    atomic_json(sessions_path, sessions)
    failures = 0
    last_notice = time.perf_counter()
    try:
        for index, (bag, previous) in enumerate(pending, 1):
            started = time.perf_counter()
            row = {"fingerprint": context["fingerprint"], "member": member,
                   "input_sha256": digest(bag), "pair_ids": bag["pair_ids"]}
            interrupted = None
            try:
                maxima = {label: 0. for label in context["settings"]["label_descriptions"]}
                evidence = []
                with torch.inference_mode():
                    for i, (text, positions) in enumerate(zip(bag["sentences"], bag["entity_positions"], strict=True)):
                        scores, word_spans = score(text, positions)
                        labels_above(scores, CUTOFFS[member], maxima)
                        for label in maxima:
                            maxima[label] = max(maxima[label], scores[label])
                        evidence.append({"selected_sentence_index": i,
                                         "source_lines": bag["selected_source_lines"][i],
                                         "head_char_span": positions["head"], "tail_char_span": positions["tail"],
                                         "word_spans_inclusive": [list(span) for span in word_spans],
                                         "relation_scores": scores})
                row.update(status="valid", relation_score_max=maxima, evidence=evidence,
                           relations=sorted(labels_above(maxima, CUTOFFS[member], maxima)))
                failures = 0
            except BaseException as exc:
                if not isinstance(exc, Exception):
                    interrupted = exc
                history = list((previous or {}).get("failures", []))
                history.append(f"{type(exc).__name__}: {exc}")
                row.update(status="failed", relations=[], failures=history)
                failures += 1
            row["wall_s"] = time.perf_counter() - started
            row["total_attempt_wall_s"] = (previous or {}).get("total_attempt_wall_s", 0.) + row["wall_s"]
            atomic_json(cache_path(context, member, bag), row)
            if interrupted is not None:
                raise interrupted
            if index == 1 or index == len(pending) or time.perf_counter() - last_notice > 30:
                print(f"{member}: {index:,}/{len(pending):,} pending bags processed; {row['status']}", flush=True)
                last_notice = time.perf_counter()
                if shutil.disk_usage(ROOT).free < 2 * 1024**3:
                    raise RuntimeError("Less than 2 GiB free; resume saved work later")
            if failures >= 3:
                raise RuntimeError("Three consecutive encoder failures; inspect saved records")
    finally:
        del model
        gc.collect()


def run_local(context, flags, *, preflight_only=False, enabled=False, prevent_sleep=True):
    """Run enabled members in isolated subprocesses, one checkpoint at a time."""
    if not enabled:
        print("Local model execution disabled; enable RUN_LOCAL_INFERENCE to execute.")
        return
    verify_context(context)
    if set(flags) != set(CUTOFFS):
        raise ValueError("Specify independent base/large/glidre flags")
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    with keep_awake(prevent_sleep):
        for member in ("base", "large", "glidre"):
            if not flags[member]:
                print(f"{member}: disabled; any complete matching cache remains available")
                continue
            for stage in (["preflight"] if preflight_only else ["run"]):
                subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()),
                                "--member", member, "--stage", stage], env=env, cwd=ROOT, check=True)


def encoder_status(context):
    rows = []
    for member in CUTOFFS:
        counts = {"valid": 0, "failed": 0, "missing": 0}
        for bag in context["manifest"]:
            result = read_encoder_result(context, member, bag)
            counts["missing" if result is None else result["status"]] += 1
        rows.append({"member": member, "expected_bags": len(context["manifest"]), **counts})
    return rows


def report_slm(context):
    verify_context(context)
    allowed = context["settings"]["label_descriptions"]
    members = {m: {} for m in CUTOFFS}
    seconds = {m: 0. for m in CUTOFFS}
    for bag in context["manifest"]:
        for member in CUTOFFS:
            row = read_encoder_result(context, member, bag)
            if row is None or row["status"] != "valid":
                raise RuntimeError("Complete valid score inventories for all members are required")
            members[member][bag["bag_id"]] = row
            seconds[member] += row["total_attempt_wall_s"]
    label_setup = {}
    for member in CUTOFFS:
        sessions = read_json(context["run"] / "encoders" / member / "runtime.json")
        label_setup[member] = sum(r["label_encoding_wall_s"] for r in sessions)
        seconds[member] += label_setup[member]
    ensemble = {bag_id: {"status": "valid", "relations": sorted(majority(
                {m: members[m][bag_id]["relations"] for m in CUTOFFS}))} for bag_id in members["large"]}
    gold = selected_gold(context)
    rows = []
    for name, predictions, elapsed in (
            ("large_0.7", members["large"], seconds["large"]),
            ("majority_0.5_0.7_0.7", ensemble, sum(seconds.values()))):
        rows.append({"system": name, **score_predictions(gold, predictions, allowed),
                     "total_wall_s": elapsed, "mean_wall_s": elapsed / len(gold),
                     "timing_scope": "sum of CPU bag attempts plus session label encoding; loading/preflight separate"})
    report = {"fingerprint": context["fingerprint"], "reference": "selected_sentence_manual_union",
              "rows": rows, "member_inference_seconds_for_accounting_only": seconds,
              "member_label_encoding_seconds": label_setup}
    atomic_json(context["run"] / "slm_summary.json", report)
    atomic_json(ROOT / "results/test_slm_summary.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--member", choices=list(CUTOFFS), required=True)
    parser.add_argument("--stage", choices=["preflight", "run"], required=True)
    args = parser.parse_args()
    context = prepare_experiment()
    verify_context(context)
    with exclusive_worker(context, args.member):
        (preflight if args.stage == "preflight" else run_member)(context, args.member)
