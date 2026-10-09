"""DeepSeek low test execution and durable accounting outside the notebook.

Prompt, parser, tariff helpers and request journaling are extracted unchanged
from the completed v2 development notebook. Scope/configuration and final-test
reporting are separate. Importing this module never sends a request.
"""
from __future__ import annotations
import contextlib
import hashlib
import json
import math
import os
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from getpass import getpass
from pathlib import Path
from zoneinfo import ZoneInfo
from nyt10m_test_protocol import (
    ROOT, canonical, digest, file_hash, read_json, score_predictions,
    selected_gold, verify_context,
)

API_SETTINGS = {
    "model": "deepseek-flash", "advertised_release": "DeepSeek-V4.1-Flash",
    "endpoint": "https://api.deepseek.com/chat/completions",
    "thinking": {"type": "enabled"}, "reasoning_effort": "low",
    "max_tokens": 4096, "response_format": {"type": "json_object"},
    "sampling": "provider defaults; no temperature/top_p override",
    "max_http_retries": 2, "request_timeout_seconds": 180,
    "duplicate_rule": "deduplicate known relation strings and count repeats",
    "invalid_rule": "empty positive set; separate invalid failure; no generation retry",
    "price_version": "deepseek_flash_pricing_20261006",
    "rates_per_million": {
        "off_peak": {"input_miss": .15, "input_hit": .003, "output": .60},
        "peak": {"input_miss": .30, "input_hit": .006, "output": 1.20}},
}
ROME = ZoneInfo("Europe/Rome")


def system_prompt(descriptions):
    return ('Extract relations from the supplied news sentences for the ordered HEAD -> TAIL pair. '
    'Entity mentions are supplied; do not detect or substitute other entities. '
    'Character spans are zero-based half-open positions [start,end) in the exact sentence. '
    'Preserve head-to-tail direction even when the tail occurs first in the text. '
    'Use all supplied sentences. Accept directly expressed relations and linguistic or pragmatic '
    'inferences justified by the supplied text. Do not add relations from unsupported outside '
    'knowledge, mere co-occurrence, or an assumed database fact. Multiple relations may be supported. '
    'Return exactly one JSON object with only the key "relations". Its value must be a flat '
    'list of distinct allowed relation-ID strings, never a dictionary, a list of objects or nested lists. '
    'Syntax-only positive example: {"relations": ["/people/person/children"]}. '
    'This example illustrates the output format only; it supplies no evidence about the current pair. '
    'Do not copy the example label unless the supplied sentences support it. '
    'Return {"relations": []} when no allowed positive relation is supported. '
    'Never output NA, an invented label, Markdown, explanations, confidence values or extra keys. '
    'List each accepted relation once. Allowed relations and their head-to-tail meaning:\n'
    + '\n'.join(f'{label}: {descriptions[label]}' for label in sorted(descriptions)))


def messages_for(bag):
    evidence = {'head': bag['head'], 'tail': bag['tail'], 'sentences': [
        {'text': text, 'head_span': pos['head'], 'tail_span': pos['tail']}
        for text, pos in zip(bag['sentences'], bag['entity_positions'])]}
    return [{'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': canonical(evidence)}]

def no_duplicate_keys(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError('Repeated JSON object key.')
        obj[key] = value
    return obj

def parse_answer(content, finish_reason):
    if finish_reason != 'stop':
        return {'status': 'invalid', 'relations': [], 'duplicates': 0,
                'error': f'finish_reason={finish_reason}'}
    try:
        if not isinstance(content, str) or not content.strip():
            raise ValueError('Empty or non-string content.')
        obj = json.loads(content, object_pairs_hook=no_duplicate_keys)
        if not isinstance(obj, dict) or set(obj) != {'relations'}:
            raise ValueError('Expected only the relations key.')
        labels = obj['relations']
        if not isinstance(labels, list) or any(not isinstance(x, str) for x in labels):
            raise ValueError('Relations must be a list of strings.')
        if set(labels) - ALLOWED:
            raise ValueError('Unknown or NA relation ID.')
        unique = sorted(set(labels))
        return {'status': 'valid' if unique else 'valid_empty', 'relations': unique,
                'duplicates': len(labels) - len(unique), 'error': None}
    except (ValueError, TypeError) as exc:
        return {'status': 'invalid', 'relations': [], 'duplicates': 0, 'error': str(exc)}

def request_for(bag, mode):
    return {'model': MODEL_ID, 'messages': messages_for(bag),
            'response_format': {'type': 'json_object'}, 'stream': False,
            'max_tokens': MAX_COMPLETION_TOKENS, **MODE_SPECS[mode]}

def is_peak(moment):
    utc = moment.astimezone(timezone.utc)
    return utc.weekday() < 5 and (1 <= utc.hour < 4 or 6 <= utc.hour < 10)

def interval_has_peak(start, end):
    if end < start:
        raise ValueError('Invalid request interval.')
    cursor = start.replace(minute=0, second=0, microsecond=0)
    while cursor <= end:
        if is_peak(cursor):
            return True
        cursor += timedelta(hours=1)
    return is_peak(start) or is_peak(end)

def off_peak_ready(moment):
    return not interval_has_peak(moment, moment + timedelta(seconds=BOUNDARY_BUFFER_SECONDS))

def wait_for_window():
    last_notice = 0.0
    while OFF_PEAK_ONLY and not off_peak_ready(datetime.now(timezone.utc)):
        if not WAIT_FOR_OFF_PEAK:
            raise RunStopped('Outside the buffered off-peak window; resume later.')
        if time.monotonic() - last_notice >= 60:
            print('Waiting for off-peak:', datetime.now(timezone.utc).astimezone(ROME).isoformat(timespec='seconds'))
            last_notice = time.monotonic()
        time.sleep(30)

def normalized_usage(raw):
    if not isinstance(raw, dict):
        raise ValueError('Missing API token usage.')
    def integer(value, name):
        if type(value) is not int or value < 0:
            raise ValueError(f'Invalid usage field: {name}')
        return value
    prompt = integer(raw.get('prompt_tokens'), 'prompt_tokens')
    completion = integer(raw.get('completion_tokens'), 'completion_tokens')
    hit = raw.get('prompt_cache_hit_tokens', (raw.get('prompt_tokens_details') or {}).get('cached_tokens'))
    hit = 0 if hit is None else integer(hit, 'cached_tokens')
    miss = raw.get('prompt_cache_miss_tokens', prompt - hit)
    miss = integer(miss, 'prompt_cache_miss_tokens')
    if hit + miss != prompt:
        raise ValueError('Cache-hit/miss counts do not sum to prompt tokens.')
    details = raw.get('completion_tokens_details') or {}
    reasoning = details.get('reasoning_tokens')
    if reasoning is not None:
        reasoning = integer(reasoning, 'reasoning_tokens')
        if reasoning > completion:
            raise ValueError('Reasoning exceeds total completion tokens.')
    if raw.get('total_tokens', prompt + completion) != prompt + completion:
        raise ValueError('Total usage inventory differs.')
    return {'prompt_tokens': prompt, 'cache_hit_tokens': hit, 'cache_miss_tokens': miss,
            'completion_tokens': completion, 'reasoning_tokens': reasoning,
            'final_tokens': completion - reasoning if reasoning is not None else None}

def usage_cost(usage, tariff, uncached=False):
    rate = RATES[tariff]
    input_cost = (usage['prompt_tokens'] * rate['input_miss'] if uncached else
                  usage['cache_miss_tokens'] * rate['input_miss'] + usage['cache_hit_tokens'] * rate['input_hit'])
    return (input_cost + usage['completion_tokens'] * rate['output']) / 1_000_000

class RunStopped(RuntimeError):
    pass

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, encoding='utf-8', delete=False) as handle:
        temp = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    os.replace(temp, path)

def credential():
    key = os.environ.get('DEEPSEEK_API_KEY', '').strip()
    if not key and (ROOT / '.env').is_file():
        for line in (ROOT / '.env').read_text().splitlines():
            name, separator, value = line.strip().removeprefix('export ').partition('=')
            if separator and name.strip() == 'DEEPSEEK_API_KEY':
                values = shlex.split(value, comments=True)
                if len(values) != 1:
                    raise ValueError('DEEPSEEK_API_KEY must be a single .env value.')
                key = values[0].strip()
    if not key or key.lower() in {'...', 'your_api_key', 'your-key-here'}:
        key = getpass('DeepSeek API key (hidden input): ').strip()
    if not key:
        raise RunStopped('No API key supplied.')
    return key

def pair_slug(pair):
    return digest(list(pair))[:24]

def result_path(pair, mode):
    return RESULTS / mode / f'{pair_slug(pair)}.json'

def read_result(bag, mode):
    path = result_path(tuple(bag['pair_ids']), mode)
    if not path.is_file():
        return None
    row = json.loads(path.read_text())
    if (row.get('inference_fingerprint') != INFERENCE_FP or row.get('mode') != mode or
            row.get('pair_ids') != bag['pair_ids'] or
            row.get('request_sha256') != digest(request_for(bag, mode))):
        raise RuntimeError(f'Stale API cache: {path}')
    if row.get('status') not in {'valid', 'valid_empty', 'invalid', 'http_error'}:
        raise RuntimeError(f'Unexpected terminal API status: {path}')
    if row['status'] != 'http_error':
        response = row['response']
        parsed = parse_answer(response['choices'][0]['message'].get('content'),
                              response['choices'][0].get('finish_reason'))
        if any(row[key] != parsed[key] for key in ('status', 'relations', 'duplicates', 'error')):
            raise RuntimeError(f'API parser/cache mismatch: {path}')
    elif row.get('relations') != []:
        raise RuntimeError('An HTTP failure must not invent predictions.')
    attempt_path = ATTEMPTS / f'{row['attempt_id']}.json'
    if not attempt_path.is_file():
        raise RuntimeError('A saved API result has no durable billing journal.')
    attempt = json.loads(attempt_path.read_text())
    if any(attempt.get(k) != row.get(k) for k in ('inference_fingerprint', 'pair_ids', 'mode', 'request_sha256')):
        raise RuntimeError('Result and billing journal identities differ.')
    if row['status'] != 'http_error':
        if attempt.get('state') != 'completed' or attempt.get('response') != row['response']:
            raise RuntimeError('Result differs from its journaled response.')
        normalized_usage(row['response'].get('usage'))
    elif attempt.get('state') != 'http_error':
        raise RuntimeError('HTTP failure differs from its journal state.')
    return row

def read_ledger():
    ledger = {}
    for path in ATTEMPTS.glob('*.json'):
        row = json.loads(path.read_text())
        if row.get('inference_fingerprint') != INFERENCE_FP:
            raise RuntimeError('Attempt ledger fingerprint mismatch.')
        ledger[row['attempt_id']] = row
    return ledger

def ledger_budget(ledger):
    return sum(row.get('cost_usd_estimate', row['reserve_usd']) for row in ledger.values())

def save_attempt(ledger, row):
    atomic_json(ATTEMPTS / f'{row["attempt_id"]}.json', row)
    ledger[row['attempt_id']] = row

@contextlib.contextmanager
def exclusive_run():
    import fcntl
    API_RUN.mkdir(parents=True, exist_ok=True)
    with (API_RUN / 'run.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RunStopped('Another notebook process already owns this API run.')
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)

@contextlib.contextmanager
def awake_context():
    process = None
    if KEEP_AWAKE and sys.platform == 'darwin' and shutil.which('caffeinate'):
        process = subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        yield
    finally:
        if process is not None:
            process.terminate(); process.wait(timeout=10)

def completion_record(bag, mode, response, attempt_id, elapsed):
    choices = response.get('choices')
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0].get('message'), dict):
        raise ValueError('Unexpected completion envelope.')
    returned_model = response.get('model')
    if not isinstance(returned_model, str) or returned_model.lower() not in {
            'deepseek-flash', 'deepseek-v4.1-flash'}:
        raise ValueError('The returned model ID does not identify the requested Flash service.')
    usage = normalized_usage(response.get('usage'))
    if MODE_SPECS[mode]['thinking']['type'] == 'disabled' and (
            choices[0]['message'].get('reasoning_content') or (usage['reasoning_tokens'] or 0) > 0):
        raise ValueError('The non-thinking condition returned reasoning; stop and verify API controls.')
    parsed = parse_answer(choices[0]['message'].get('content'), choices[0].get('finish_reason'))
    return {'inference_fingerprint': INFERENCE_FP, 'pair_ids': bag['pair_ids'], 'mode': mode,
            'request_sha256': digest(request_for(bag, mode)), 'attempt_id': attempt_id,
            'wall_s': elapsed, 'response': response, **parsed}

def call_once(bag, mode, key, ledger):
    payload = request_for(bag, mode)
    raw_payload = canonical(payload).encode('utf-8')
    # Byte count deliberately overestimates text tokens; reserve the peak rate.
    input_upper = len(raw_payload) + 512
    reserve = (input_upper * RATES['peak']['input_miss'] +
               MAX_COMPLETION_TOKENS * RATES['peak']['output']) / 1_000_000
    if ledger_budget(ledger) + reserve > MAX_BUDGET_USD:
        raise RunStopped('Model-usage budget reached; increase it explicitly to continue.')
    wait_for_window()
    if shutil.disk_usage(ROOT).free < 256 * 1024**2:
        raise RunStopped('Less than 256 MiB free disk space.')
    started = datetime.now(timezone.utc)
    row = {'attempt_id': uuid.uuid4().hex, 'inference_fingerprint': INFERENCE_FP,
           'pair_ids': bag['pair_ids'], 'mode': mode, 'request_sha256': digest(payload),
           'started_utc': started.isoformat(), 'state': 'inflight', 'reserve_usd': reserve}
    save_attempt(ledger, row)
    request = urllib.request.Request(API_URL, data=raw_payload, method='POST', headers={
        'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'})
    tick = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as handle:
            raw_response = handle.read()
        ended = datetime.now(timezone.utc)
        elapsed = time.monotonic() - tick
        response = json.loads(raw_response)
        row['response'] = response
        usage = normalized_usage(response.get('usage'))
        tariff = 'peak' if interval_has_peak(started, ended) else 'off_peak'
        row.update(state='completed', ended_utc=ended.isoformat(), wall_s=elapsed,
                   response=response, usage=usage, tariff_estimate=tariff,
                   cost_usd_estimate=usage_cost(usage, tariff),
                   common_uncached_offpeak_usd=usage_cost(usage, 'off_peak', uncached=True))
        save_attempt(ledger, row)
        result = completion_record(bag, mode, response, row['attempt_id'], elapsed)
        atomic_json(result_path(tuple(bag['pair_ids']), mode), result)
        return result, None
    except urllib.error.HTTPError as exc:
        # Store a sanitized error body, never Authorization headers or the API key.
        body = exc.read(4096).decode('utf-8', errors='replace').replace(key, '[REDACTED]')
        row.update(state='http_error', ended_utc=datetime.now(timezone.utc).isoformat(),
                   wall_s=time.monotonic() - tick, http_status=exc.code, error_body=body)
        save_attempt(ledger, row)  # Unknown billing retains the conservative reserve.
        return None, row
    except BaseException:
        # Unknown completion/charge state must not trigger an automatic duplicate call.
        if row['state'] != 'completed':
            row.update(state='uncertain', ended_utc=datetime.now(timezone.utc).isoformat(),
                       wall_s=time.monotonic() - tick)
            save_attempt(ledger, row)
        raise

def call_with_retries(bag, mode, key, ledger):
    for retry in range(MAX_HTTP_RETRIES + 1):
        result, error = call_once(bag, mode, key, ledger)
        if result is not None:
            return result
        retryable = error['http_status'] in {429, 502, 503, 504}
        if retryable and retry < MAX_HTTP_RETRIES:
            time.sleep(min(2 ** (retry + 1), 30))
            continue
        failure = {'inference_fingerprint': INFERENCE_FP, 'pair_ids': bag['pair_ids'],
                   'mode': mode, 'request_sha256': digest(request_for(bag, mode)),
                   'attempt_id': error['attempt_id'], 'wall_s': error['wall_s'],
                   'status': 'http_error', 'relations': [], 'duplicates': 0,
                   'error': f'HTTP {error["http_status"]}'}
        atomic_json(result_path(tuple(bag['pair_ids']), mode), failure)
        if error['http_status'] in {400, 401, 402, 403, 404, 422}:
            raise RunStopped(f'HTTP {error["http_status"]}; check credentials/balance/API settings before continuing.')
        return failure

def recover_completed_journal(ledger):
    # Restore a result after a successful response was journaled but the result write was interrupted.
    for row in ledger.values():
        if row['state'] != 'completed':
            continue
        bag = full_by_pair[tuple(row['pair_ids'])]
        if result_path(tuple(row['pair_ids']), row['mode']).exists():
            continue
        result = completion_record(bag, row['mode'], row['response'], row['attempt_id'], row['wall_s'])
        atomic_json(result_path(tuple(row['pair_ids']), row['mode']), result)

def mode_order(index):
    # Rotate the first mode so each configuration occupies every request position.
    modes = list(MODE_SPECS)
    shift = index % len(modes)
    return modes[shift:] + modes[:shift]


def configure(context, *, run_api=False, max_budget_usd=10., keep_awake=True,
              allow_resubmit_uncertain_attempt_ids=()):
    """Bind only input evidence and fixed low settings; never put gold in requests."""
    global RUN_API, MAX_BUDGET_USD, KEEP_AWAKE, ALLOW_RESUBMIT_UNCERTAIN_ATTEMPT_IDS
    global LABEL_DESCRIPTIONS, ALLOWED, SYSTEM_PROMPT, MODE_SPECS, SELECTED
    global MAX_COMPLETION_TOKENS, MODEL_ID, ADVERTISED_RELEASE, API_URL
    global MAX_HTTP_RETRIES, REQUEST_TIMEOUT_SECONDS, STOP_AFTER_CONSECUTIVE_FAILURES
    global OFF_PEAK_ONLY, WAIT_FOR_OFF_PEAK, BOUNDARY_BUFFER_SECONDS, RATES
    global INFERENCE_SETTINGS, INFERENCE_FP, API_RUN, ATTEMPTS, RESULTS
    global full_by_pair, reviewed_id, SCOPE, CONTEXT
    verify_context(context)
    if not isinstance(max_budget_usd, (int, float)) or not math.isfinite(max_budget_usd) or max_budget_usd <= 0:
        raise ValueError("Positive finite API budget required")
    CONTEXT = context
    RUN_API, MAX_BUDGET_USD, KEEP_AWAKE = run_api, max_budget_usd, keep_awake
    ALLOW_RESUBMIT_UNCERTAIN_ATTEMPT_IDS = tuple(allow_resubmit_uncertain_attempt_ids)
    LABEL_DESCRIPTIONS = context["settings"]["label_descriptions"]
    ALLOWED = set(LABEL_DESCRIPTIONS)
    SYSTEM_PROMPT = system_prompt(LABEL_DESCRIPTIONS)
    MODE_SPECS = {"thinking_low": {"thinking": {"type": "enabled"}, "reasoning_effort": "low"}}
    SELECTED = context["manifest"]
    full_by_pair = {tuple(b["pair_ids"]): b for b in SELECTED}
    reviewed_id = {}
    SCOPE = "official_test_selected20"
    MAX_COMPLETION_TOKENS = API_SETTINGS["max_tokens"]
    MODEL_ID, ADVERTISED_RELEASE = API_SETTINGS["model"], API_SETTINGS["advertised_release"]
    API_URL = API_SETTINGS["endpoint"]
    MAX_HTTP_RETRIES, REQUEST_TIMEOUT_SECONDS = API_SETTINGS["max_http_retries"], API_SETTINGS["request_timeout_seconds"]
    STOP_AFTER_CONSECUTIVE_FAILURES = 3
    OFF_PEAK_ONLY = WAIT_FOR_OFF_PEAK = False
    BOUNDARY_BUFFER_SECONDS = 300
    RATES = API_SETTINGS["rates_per_million"]
    INFERENCE_FP = context["fingerprint"]
    INFERENCE_SETTINGS = {"shared_test_fingerprint": INFERENCE_FP,
                          "api": API_SETTINGS, "system_prompt": SYSTEM_PROMPT}
    API_RUN = context["run"] / "api"
    ATTEMPTS, RESULTS = API_RUN / "attempts", API_RUN / "bags"
    return {"configuration": "thinking_low", "bags": len(SELECTED),
            "paid_calls_enabled": RUN_API, "budget_USD": MAX_BUDGET_USD,
            "cache_directory": str(API_RUN), "off_peak_wait_enabled": False}


def run_experiment():
    """Execute missing low completions; cached invalid answers are never regenerated."""
    verify_context(CONTEXT)
    if not RUN_API:
        print("Paid requests disabled. Set RUN_API=True in the LLM test notebook to execute.")
        return
    with exclusive_run(), awake_context():
        config_path = API_RUN / "settings.json"
        if config_path.exists() and read_json(config_path) != INFERENCE_SETTINGS:
            raise RuntimeError("API settings differ from this cache")
        atomic_json(config_path, INFERENCE_SETTINGS)
        atomic_json(API_RUN / f"selection_{SCOPE}.json", {
            "scope": SCOPE, "pair_ids": [b["pair_ids"] for b in SELECTED],
            "input_hashes": {pair_slug(b["pair_ids"]): digest(messages_for(b)) for b in SELECTED}})
        ledger = read_ledger()
        recover_completed_journal(ledger)
        acknowledged = set(ALLOW_RESUBMIT_UNCERTAIN_ATTEMPT_IDS)
        if acknowledged - set(ledger):
            raise RunStopped("An acknowledged attempt ID does not exist")
        for attempt_id in acknowledged:
            row = ledger[attempt_id]
            if row["state"] in {"inflight", "uncertain"}:
                row.update(state="acknowledged_unknown", recovery_note="Explicit resubmission; original charge unknown")
                save_attempt(ledger, row)
        unresolved = [r["attempt_id"] for r in ledger.values() if r["state"] in {"inflight", "uncertain"}]
        if unresolved:
            raise RunStopped("Review unresolved journal states before duplicate submission: " + ", ".join(unresolved[:5]))
        pending = []
        for bag in SELECTED:
            row = read_result(bag, "thinking_low")
            if row is None or row["status"] == "http_error":
                pending.append(bag)
        print(f"DeepSeek low: {len(pending):,}/{len(SELECTED):,} test bags pending")
        if not pending:
            return
        key = credential()
        failures = 0
        last_notice = time.monotonic()
        for index, bag in enumerate(pending, 1):
            row = call_with_retries(bag, "thinking_low", key, ledger)
            failures = failures + 1 if row["status"] in {"invalid", "http_error"} else 0
            if index == 1 or index == len(pending) or time.monotonic() - last_notice > 30:
                print(f"DeepSeek low: {index:,}/{len(pending):,} pending processed; {row['status']}; "
                      f"usage/reserve ${ledger_budget(ledger):.4f}", flush=True)
                last_notice = time.monotonic()
            if failures >= STOP_AFTER_CONSECUTIVE_FAILURES:
                raise RunStopped("Three consecutive failed answers/requests; completed results are retained")
        print("Selected test request pass finished; verify completion status before scoring.")


def status_and_cost():
    counts = Counter()
    for bag in SELECTED:
        row = read_result(bag, "thinking_low")
        counts["missing" if row is None else row["status"]] += 1
    ledger = read_ledger()
    measured = [r for r in ledger.values() if "cost_usd_estimate" in r]
    unknown = [r for r in ledger.values() if "cost_usd_estimate" not in r]
    return {
        "expected_bags": len(SELECTED), "status_counts": dict(counts),
        "attempts": len(ledger), "completed_with_usage": len(measured),
        "unknown_charge_attempts": len(unknown),
        "unknown_charge_reserve_USD": sum(r["reserve_usd"] for r in unknown),
        "token_charge_USD_estimate": sum(r["cost_usd_estimate"] for r in measured),
        "prompt_tokens": sum(r["usage"]["prompt_tokens"] for r in measured),
        "completion_tokens_including_reasoning": sum(r["usage"]["completion_tokens"] for r in measured),
        "request_wall_s_including_attempts": sum(r.get("wall_s", 0.) for r in ledger.values()),
        "peak_rate_attempts": sum(r.get("tariff_estimate") == "peak" for r in measured),
        "backend_fingerprints": sorted({str(r["response"].get("system_fingerprint")) for r in measured}),
    }


def report_api():
    """Score all test bags under the shared reference; never drop invalid answers."""
    verify_context(CONTEXT)
    inventory = {}
    for bag in SELECTED:
        row = read_result(bag, "thinking_low")
        if row is None or row["status"] == "http_error":
            raise RuntimeError("Missing/HTTP requests remain; no partial final-test score")
        inventory[bag["bag_id"]] = row
    accounting = status_and_cost()
    metrics = score_predictions(selected_gold(CONTEXT), inventory, ALLOWED)
    row = {"system": "deepseek_low", **metrics,
           "total_wall_s": accounting["request_wall_s_including_attempts"],
           "mean_wall_s": accounting["request_wall_s_including_attempts"] / len(SELECTED),
           "timing_scope": "sum of API request attempts; includes service/network; excludes backoff and idle time",
           "token_charge_USD_estimate": accounting["token_charge_USD_estimate"]}
    report = {"fingerprint": INFERENCE_FP, "reference": "selected_sentence_manual_union",
              "rows": [row], "accounting": accounting,
              "cost_note": "tariff/cache-based reconstruction, not an invoice; unknown charges remain separate"}
    atomic_json(CONTEXT["run"] / "llm_summary.json", report)
    atomic_json(ROOT / "results/test_llm_summary.json", report)
    return report


def combined_report(context):
    """Read the two completed summaries; no models or API requests are involved."""
    verify_context(context)
    rows = []
    for filename in ("slm_summary.json", "llm_summary.json"):
        path = context["run"] / filename
        if not path.is_file():
            raise RuntimeError("Both SLM and LLM summaries are required")
        report = read_json(path)
        if report["fingerprint"] != context["fingerprint"] or report["reference"] != "selected_sentence_manual_union":
            raise RuntimeError("Systems use different test inputs/references")
        rows.extend(report["rows"])
    report = {"fingerprint": context["fingerprint"], "rows": rows,
              "reference": "selected_sentence_manual_union",
              "timing_note": "local CPU member sums and API network/service latency are distinct measurements"}
    atomic_json(ROOT / "results/test_comparison_summary.json", report)
    return report
