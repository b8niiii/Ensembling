"""Pinned GLiDRE download, isolated offline preflight and resumable inference.

The existing relex runner/adapter/descriptions and 72,532 caches are untouched.
The GLiNER 0.2.13 helper wheel is extracted into a private import directory;
it is never installed over the project's GLiNER 0.2.29 environment.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.machinery
import json
import os
import shutil
import sys
import time
import types
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from validation_model_comparison import (
    ROOT, RELEX_RUN, GLIDRE_RUN, bag_path, file_hash, frozen_inputs,
    read_json, save_json, validate_result,
)

REPO = "cea-list-ia/glidre_large"
REVISION = "d1ba1cf0f41d2849f1695c6acc67546700d721ba"
WEIGHTS_SHA256 = "dc2fb3cf239ba898161fa9a9d6c27d8f4f51714b409d094f58c343ea12c52225"
CODE_REVISION = "5c3bee6698b4365224576459f9c8d48c9a6d3b6d"
LABEL_REPO = "BAAI/bge-large-en-v1.5"
LABEL_REVISION = "d4aa6901d3a41ba39fb536a557fa166f842b0e09"
GLINER_WHEEL_URL = "https://files.pythonhosted.org/packages/2b/aa/9f08dddffac7f85f61a2a180a64ec76a88dda082cc7de61c6fd2f7fb449b/gliner-0.2.13-py3-none-any.whl"
GLINER_WHEEL_SHA256 = "a8798aef78b8a7322dc4f785c15d58c526f86698ec95feb892b500b5be507437"
GLIDRE_SOURCE_HASHES = {'glidre/__init__.py': 'e03792c2b621dfdefeffb009b9974796d91140cb85263c8e531b0edbee898424', 'glidre/base_model.py': 'a4b4a327fea140d8123f9d6531fab245df702c3f296cc63510e3a0cf1897b7a5', 'glidre/collator.py': '2655d58122e1dfb50403e6bab9fdbd435e8ae001f8f750c2786e6ada37e4a8d2', 'glidre/config.py': 'db681c87563cd54d2cd85e20884e4327319337e2dfdc70f0cfaf2546198186aa', 'glidre/decoder.py': '1ba34e26a138c29469af2c7794718ec1a399f395f66b748a5bdda2b2ed1d7604', 'glidre/encoder.py': '294b81e286c338001568728b9107b29f3c10d5af383aa6812ad10cff9e4e28ae', 'glidre/loss_functions.py': 'ca8fa97a84d55a9752a4ecde04e13a29060b546a6433d3c2c40649c06e82fcb7', 'glidre/model.py': 'ab93477bc692a100a48e198f6e619267245a9d87fd4c181b16837e73294e701f', 'glidre/processor.py': '1e710801a57dcab8c13acff7a9a669d03986efa41c9ea1ebecbb49cd419087a2', 'glidre/rel_rep.py': '454e713082eab5d21f7d83664f071d09ee312c4a6304db40119f18b2d24bc566', 'glidre/training_utils.py': 'd223203c5ae567dc48da19beb1e85142c4a6c823ca8486a5dfeec6ce0020aa62'}
CACHE = ROOT / "data/model_cache"
VENDOR = CACHE / "glidre_runtime" / CODE_REVISION
CHECKPOINT = CACHE / "models--cea-list-ia--glidre_large/snapshots" / REVISION
LABEL_TOKENIZER = CACHE / "models--BAAI--bge-large-en-v1.5/snapshots" / LABEL_REVISION
ASSET_RECEIPT = VENDOR / "assets.json"
MAX_TEXT_WORDS = 2048
EXPECTED_PARAMETERS = 800241152
MIN_FREE_GIB = 2


def require_offline():
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("This stage requires explicit Hugging Face/Transformers offline mode")


def download_small(url, path, expected_hash=None):
    """Download a source archive/helper wheel; never execute an installer."""
    path = Path(path)
    if path.is_file() and (expected_hash is None or file_hash(path) == expected_hash):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".download")
    with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    if expected_hash and file_hash(temporary) != expected_hash:
        raise RuntimeError("Downloaded helper package checksum does not match")
    temporary.replace(path)


def prefetch(include_glidre, relex_members):
    """Fetch every enabled experiment asset before any offline model stage."""
    from huggingface_hub import snapshot_download
    CACHE.mkdir(parents=True, exist_ok=True)
    if include_glidre:
        required_free = MIN_FREE_GIB * 1024**3
        if not (CHECKPOINT / "pytorch_model.bin").is_file():
            required_free += 3201265131
        if shutil.disk_usage(ROOT).free < required_free:
            raise RuntimeError("Not enough free disk for GLiDRE download plus the 2 GiB reserve")
        print("Prefetching pinned GLiDRE weights (3.20 GB), tokenizer and isolated source helpers.", flush=True)
        snapshot_download(REPO, revision=REVISION, cache_dir=CACHE, max_workers=1,
                          allow_patterns=["pytorch_model.bin", "glidre_config.json", "tokenizer.json",
                                          "tokenizer_config.json", "special_tokens_map.json", "spm.model"])
        # Both encoder weights are already inside GLiDRE; BGE downloads are tokenizer-only.
        snapshot_download(LABEL_REPO, revision=LABEL_REVISION, cache_dir=CACHE, max_workers=1,
                          allow_patterns=["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"])
        VENDOR.mkdir(parents=True, exist_ok=True)
        archive = VENDOR / "glidre_source.zip"
        download_small(f"https://codeload.github.com/cea-list-lasti/glidre/zip/{CODE_REVISION}", archive)
        with zipfile.ZipFile(archive) as source:
            for relative, expected in GLIDRE_SOURCE_HASHES.items():
                matches = [name for name in source.namelist() if name.endswith("/" + relative)]
                if len(matches) != 1:
                    raise RuntimeError(f"Pinned GLiDRE source missing: {relative}")
                raw = source.read(matches[0])
                if hashlib.sha256(raw).hexdigest() != expected:
                    raise RuntimeError(f"Pinned GLiDRE source checksum differs: {relative}")
                destination = VENDOR / "source" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(raw)
        wheel = VENDOR / "gliner-0.2.13.whl"
        download_small(GLINER_WHEEL_URL, wheel, GLINER_WHEEL_SHA256)
        with zipfile.ZipFile(wheel) as package:
            for name in package.namelist():
                if not name.startswith("gliner/") or name.endswith("/"):
                    continue
                if ".." in Path(name).parts:
                    raise RuntimeError("Invalid helper wheel path")
                destination = VENDOR / "helper" / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(package.read(name))
        files = [CHECKPOINT / name for name in ("pytorch_model.bin", "glidre_config.json", "tokenizer.json",
                                                 "tokenizer_config.json", "special_tokens_map.json", "spm.model")]
        files += [LABEL_TOKENIZER / name for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt")]
        files += sorted((VENDOR / "source/glidre").glob("*.py"))
        files += sorted((VENDOR / "helper/gliner").rglob("*.py"))
        if file_hash(CHECKPOINT / "pytorch_model.bin") != WEIGHTS_SHA256:
            raise RuntimeError("GLiDRE weight checksum differs from the pinned Hub LFS object")
        save_json(ASSET_RECEIPT, {"repo": REPO, "revision": REVISION, "code_revision": CODE_REVISION,
                                 "label_tokenizer_revision": LABEL_REVISION,
                                 "helper_version": "0.2.13", "helper_wheel_sha256": GLINER_WHEEL_SHA256,
                                 "files": {str(p.relative_to(ROOT)): file_hash(p) for p in files}})
        print("GLiDRE assets complete; no package was installed or downgraded.", flush=True)
    if relex_members:
        from run_full_validation_relex_given_entities import MODELS, REQUIRED_FILES
        for name in relex_members:
            spec = next(s for s in MODELS if s["slug"] == "gliner_relex_" + name)
            snapshot_download(spec["repo"], revision=spec["revision"], cache_dir=CACHE,
                              allow_patterns=list(REQUIRED_FILES), max_workers=1)
    print("PREFETCH COMPLETE. All subsequent stages use offline mode.", flush=True)


def verify_assets():
    if not ASSET_RECEIPT.is_file():
        raise FileNotFoundError("GLiDRE assets missing. Enable the notebook's asset-prefetch flag while online.")
    receipt = read_json(ASSET_RECEIPT)
    if (receipt.get("revision") != REVISION or receipt.get("code_revision") != CODE_REVISION
            or receipt.get("helper_wheel_sha256") != GLINER_WHEEL_SHA256):
        raise RuntimeError("GLiDRE asset receipt belongs to a different runtime")
    if receipt["files"].get(str((CHECKPOINT / "pytorch_model.bin").relative_to(ROOT))) != WEIGHTS_SHA256:
        raise RuntimeError("Incorrect pinned GLiDRE weight identity")
    for relative, expected in receipt["files"].items():
        if not (ROOT / relative).is_file() or file_hash(ROOT / relative) != expected:
            raise RuntimeError(f"Missing or changed offline asset: {relative}")
    for relative, expected in GLIDRE_SOURCE_HASHES.items():
        if file_hash(VENDOR / "source" / relative) != expected:
            raise RuntimeError("Inspected GLiDRE source changed")
    return receipt


def prepare():
    """Bind GLiDRE to the existing selected evidence, independently of annotations."""
    require_offline()
    manifest, original, _ = frozen_inputs()
    receipt = verify_assets()
    descriptions = original["settings"]["label_name"]
    name_to_id = {name.upper().replace(" ", "_"): label for label, name in descriptions.items()}
    if len(name_to_id) != 24:
        raise RuntimeError("GLiDRE relation verbalizations are not unique")
    settings = {
        "model": "GLiDRE-large", "repo": REPO, "revision": REVISION,
        "weights_sha256": WEIGHTS_SHA256, "code_revision": CODE_REVISION,
        "source_run_fingerprint": original["fingerprint"],
        "source_manifest_sha256": file_hash(RELEX_RUN / "manifest.json"),
        "asset_receipt_sha256": file_hash(ASSET_RECEIPT),
        "runner_sha256": file_hash(Path(__file__)),
        "adapter_sha256": file_hash(ROOT / "source/glidre_given_entities.py"),
        "device": "cpu", "bag_cap": 20, "name_to_relation_id": name_to_id,
        "label_format": "same frozen descriptions; uppercase with spaces replaced by underscores",
        "native_word_cap": 512, "full_text_word_guard": MAX_TEXT_WORDS,
        "length_handling": "lift preprocessing word cap; no truncation; relative-position text encoder; longest actual input runtime check required",
        "relation_scores": "all 24 sigmoid scores, supplied head-to-tail pair, maximum over unchanged selected sentences",
        "label_embeddings": "native fixed label embeddings cached once per worker session",
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "huggingface_hub", "tokenizers", "safetensors")},
        "isolated_gliner_helper": "0.2.13", "manual_test_access": False,
        "native_compatibility_bridge": "repair BiEncoder and word-attention tuple plumbing; unchanged pretrained layers and weights",
    }
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    wrapper = {"fingerprint": fingerprint, "settings": settings}
    destination = GLIDRE_RUN / "settings.json"
    if destination.exists() and read_json(destination) != wrapper:
        raise RuntimeError("GLiDRE frozen inputs/runtime changed; preserve this run and use a new version")
    if not destination.exists():
        save_json(destination, wrapper)
    print(f"GLiDRE prepared: {len(manifest):,} bags / 44,751 unchanged sentence records; {fingerprint[:16]}", flush=True)
    return manifest, wrapper, receipt


def load_model():
    """Load the complete checkpoint strictly, using embedded encoder configs."""
    require_offline()
    # Import isolation lasts only for this subprocess and does not change .venv.
    sys.path[:0] = [str(VENDOR / "source"), str(VENDOR / "helper")]
    # The historical top-level __init__ imports an unrelated ONNX backend.
    # GLiDRE needs only the unchanged PyTorch helper submodules; expose that
    # package namespace without importing its unused public GLiNER model.
    if "gliner" in sys.modules:
        raise RuntimeError("GLiDRE worker already imported a non-isolated GLiNER")
    helper = VENDOR / "helper/gliner"
    namespace = types.ModuleType("gliner")
    namespace.__path__ = [str(helper)]
    namespace.__file__ = str(helper / "__init__.py")
    namespace.__spec__ = importlib.machinery.ModuleSpec("gliner", loader=None, is_package=True)
    namespace.__version__ = "0.2.13"
    sys.modules["gliner"] = namespace
    import gliner
    if not Path(gliner.__file__).resolve().is_relative_to((VENDOR / "helper").resolve()):
        raise RuntimeError("GLiDRE loaded the wrong GLiNER helper package")
    from glidre import GLiDRE, GLiDREConfig
    from glidre.processor import SpanPairBiEncoderProcessor
    from gliner.data_processing.tokenizer import WordsSplitter
    from transformers import AutoConfig, AutoTokenizer
    import torch
    from glidre_given_entities import UntruncatedTokenizer, wire_native_biencoder
    from glidre.base_model import extract_word_embeddings
    raw = read_json(CHECKPOINT / "glidre_config.json")
    def embedded_config(name, **kwargs):
        field = "encoder_config" if name == raw["model_name"] else "labels_encoder_config" if name == raw["labels_encoder"] else None
        if field is None:
            raise RuntimeError("Unexpected remote backbone configuration request")
        values = dict(raw[field])
        model_type = values.pop("model_type")
        return AutoConfig.for_model(model_type, **values)
    with patch.object(AutoConfig, "from_pretrained", side_effect=embedded_config):
        config = GLiDREConfig(**raw)
    if (config.encoder_config.model_type != "deberta-v2"
            or config.encoder_config.position_biased_input
            or not config.encoder_config.relative_attention
            or config.loss in ("atloss", "afloss")):
        raise RuntimeError("Unexpected checkpoint context or adaptive-threshold architecture")
    # This changes only a processor guard, not positions, layers or parameters.
    config.max_len = MAX_TEXT_WORDS
    text_tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT), local_files_only=True)
    label_tokenizer = AutoTokenizer.from_pretrained(str(LABEL_TOKENIZER), local_files_only=True)
    processor = SpanPairBiEncoderProcessor(config, UntruncatedTokenizer(text_tokenizer),
                                          WordsSplitter(config.words_splitter_type), label_tokenizer)
    model = GLiDRE(config, tokenizer=text_tokenizer, data_processor=processor, encoder_from_pretrained=False)
    state = torch.load(CHECKPOINT / "pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True)
    model.model.load_state_dict(state, strict=True)
    del state
    wire_native_biencoder(model.model, extract_word_embeddings)
    model = model.to("cpu").eval()
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != EXPECTED_PARAMETERS or not 0 < parameters < 1_000_000_000:
        raise RuntimeError(f"GLiDRE total parameter count differs from the audited sub-1B architecture: {parameters}")
    return model, parameters


def preflight():
    """Validate every selected input and perform native/long-input runtime checks."""
    manifest, settings, _ = prepare()
    destination = GLIDRE_RUN / "offline_preflight.json"
    if destination.is_file():
        previous = read_json(destination)
        if previous.get("run_fingerprint") == settings["fingerprint"] and previous.get("success") is True:
            print("Reusing the matching successful GLiDRE preflight.", flush=True)
            return
    model, parameters = load_model()
    from glidre_given_entities import encode_relation_names, prepare_sentence, score_sentence, verify_native_parity
    names = settings["settings"]["name_to_relation_id"]
    longest = None
    refined = above_native_cap = max_words = max_subtokens = total = 0
    started = last_message = time.perf_counter()
    try:
        labels = encode_relation_names(model, names)
        parity = verify_native_parity(model, names, labels)
        for bag in manifest:
            for sentence, positions in zip(bag["sentences"], bag["entity_positions"], strict=True):
                aligned, _, _ = prepare_sentence(model, sentence, positions["head"], positions["tail"], names, cached_labels=labels)
                length = len(aligned["tokens"])
                max_words, max_subtokens = max(max_words, length), max(max_subtokens, aligned["subtokens"])
                if longest is None or aligned["subtokens"] > longest[0]:
                    longest = (aligned["subtokens"], sentence, positions)
                refined += aligned["boundary_refined"]
                above_native_cap += length > 512
                total += 1
            if time.perf_counter() - last_message >= 30:
                print(f"GLiDRE boundary/tokenization preflight: {total:,}/44,751 sentences checked", flush=True)
                last_message = time.perf_counter()
        # Relative-position support must work on the actual longest validation sentence.
        longest_scores, _ = score_sentence(model, longest[1], longest[2]["head"], longest[2]["tail"], names, labels)
        if len(longest_scores) != 24 or total != 44751:
            raise RuntimeError("Incomplete full-input GLiDRE preflight")
        report = {"run_fingerprint": settings["fingerprint"], "success": True,
                  "total_parameters": parameters, "strict_checkpoint_loading": True,
                  "selected_sentences_checked": total, "max_words": max_words, "max_subtokens": max_subtokens,
                  "sentences_above_native_512_word_guard": above_native_cap,
                  "boundary_refined_sentences": refined, "full_text_word_guard": MAX_TEXT_WORDS,
                  "longest_input_forward_pass": True, "native_parity": parity,
                  "offline": True, "wall_s": time.perf_counter() - started}
        save_json(destination, report)
        print(f"GLiDRE OFFLINE READY: {parameters:,} parameters; all {total:,} inputs retained; native/cached-label parity and longest-input inference passed.", flush=True)
    finally:
        del model
        gc.collect()


def run_glidre():
    manifest, settings, _ = prepare()
    verified = read_json(GLIDRE_RUN / "offline_preflight.json")
    if verified.get("success") is not True or verified.get("run_fingerprint") != settings["fingerprint"]:
        raise RuntimeError("A matching successful GLiDRE preflight is required")
    names = settings["settings"]["name_to_relation_id"]
    pending = []
    for bag in manifest:
        path = bag_path(GLIDRE_RUN, "glidre", bag)
        if not path.is_file() or not validate_result(read_json(path), bag, "glidre", settings["fingerprint"], names.values()):
            pending.append(bag)
    print(f"GLiDRE large: {len(pending):,}/{len(manifest):,} bags pending", flush=True)
    if not pending:
        return
    if shutil.disk_usage(ROOT).free < MIN_FREE_GIB * 1024**3:
        raise RuntimeError("Less than 2 GiB free disk space")
    loaded = time.perf_counter()
    model, parameters = load_model()
    load_s = time.perf_counter() - loaded
    from glidre_given_entities import encode_relation_names, score_sentence
    label_started = time.perf_counter()
    labels = encode_relation_names(model, names)
    label_s = time.perf_counter() - label_started
    runtime_path = GLIDRE_RUN / "model_runtime.json"
    runtime = read_json(runtime_path) if runtime_path.is_file() else {"sessions": []}
    runtime["sessions"].append({"started_utc": datetime.now(timezone.utc).isoformat(),
                                "load_wall_s": load_s, "label_encoding_wall_s": label_s,
                                "parameters": parameters, "pending_bags": len(pending)})
    save_json(runtime_path, runtime)
    consecutive_failures = 0
    last_message = time.perf_counter()
    try:
        for index, bag in enumerate(pending, 1):
            path = bag_path(GLIDRE_RUN, "glidre", bag)
            previous = read_json(path) if path.is_file() else {}
            started = time.perf_counter()
            try:
                maxima = {label: 0.0 for label in names.values()}
                evidence = []
                for i, (sentence, positions) in enumerate(zip(bag["sentences"], bag["entity_positions"], strict=True)):
                    scores, aligned = score_sentence(model, sentence, positions["head"], positions["tail"], names, labels)
                    for label in maxima:
                        maxima[label] = max(maxima[label], scores[label])
                    evidence.append({"selected_sentence_index": i,
                                     "source_record_index": bag["selected_indices_zero_based"][i],
                                     "head_char_span": positions["head"], "tail_char_span": positions["tail"],
                                     "head_word_span_inclusive": list(aligned["word_spans"][0]),
                                     "tail_word_span_inclusive": list(aligned["word_spans"][1]),
                                     "relation_scores": scores})
                result = {"status": "valid", "relation_score_max": maxima,
                          "relations": sorted(k for k, v in maxima.items() if v > .5),
                          "evidence": evidence, "sentences_processed": len(bag["sentences"])}
                consecutive_failures = 0
            except Exception as exc:
                history = list(previous.get("failures", []))
                history.append(f"{type(exc).__name__}: {exc}")
                result = {"status": "failed", "relations": [], "evidence": [],
                          "sentences_processed": 0, "failures": history}
                consecutive_failures += 1
            elapsed = time.perf_counter() - started
            result.update(model="GLiDRE-large", pair_ids=bag["pair_ids"], run_fingerprint=settings["fingerprint"],
                          full_bag_size=bag["full_bag_size"], selected_indices_zero_based=bag["selected_indices_zero_based"],
                          wall_s=elapsed, total_attempt_wall_s=previous.get("total_attempt_wall_s", previous.get("wall_s", 0.0)) + elapsed,
                          created_utc=datetime.now(timezone.utc).isoformat())
            save_json(path, result)
            if index == 1 or index == len(pending) or time.perf_counter() - last_message >= 30:
                print(f"GLiDRE: {index:,}/{len(pending):,} pending bags processed; last={result['status']}", flush=True)
                last_message = time.perf_counter()
                if shutil.disk_usage(ROOT).free < MIN_FREE_GIB * 1024**3:
                    raise RuntimeError("Less than 2 GiB free; completed outputs can be resumed")
            if consecutive_failures >= 3:
                raise RuntimeError("Three consecutive GLiDRE failures; inspect saved failure records")
    finally:
        import resource
        runtime["sessions"][-1]["peak_worker_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        save_json(runtime_path, runtime)
        del model
        gc.collect()
    print("GLiDRE full validation completed; model released.", flush=True)


def run_relex(member):
    """Explicitly rerun one old member in a separate cache, preserving original outputs."""
    require_offline()
    manifest, settings, _ = frozen_inputs()
    from run_full_validation_relex_given_entities import MODELS
    import run_full_validation_relex_given_entities as runner
    for package, expected in settings["settings"]["versions"].items():
        if importlib.metadata.version(package) != expected:
            raise RuntimeError(f"Relex rerun requires the original {package}=={expected}")
    spec = next(s for s in MODELS if s["slug"] == "gliner_relex_" + member)
    destination = ROOT / "data/pilot_runs/relex_explicit_reruns_20261006" / member
    if destination.is_dir() and all(bag_path(destination, member, b).is_file() and
            validate_result(read_json(bag_path(destination, member, b)), b, member, settings["fingerprint"], settings["settings"]["label_name"])
            for b in manifest):
        archive = destination.with_name(member + "_completed_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"))
        destination.rename(archive)
    save_json(destination / "settings.json", settings)
    save_json(destination / "manifest.json", manifest)
    runner.RUN = destination
    runner.run_one(spec, manifest, settings)
    print(f"Rerun saved separately in {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prefetch", "prepare", "offline-check", "run", "run-base", "run-large"))
    parser.add_argument("--glidre", action="store_true")
    parser.add_argument("--relex-members", nargs="*", choices=("base", "large"), default=[])
    args = parser.parse_args()
    if args.stage == "prefetch":
        prefetch(args.glidre, args.relex_members)
    elif args.stage == "prepare":
        prepare()
    elif args.stage == "offline-check":
        preflight()
    elif args.stage == "run":
        run_glidre()
    else:
        run_relex(args.stage.removeprefix("run-"))


if __name__ == "__main__":
    main()
