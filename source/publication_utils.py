"""Read compact published reports without datasets, checkpoints or API access."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def load_test_report(root: Path, filename: str) -> dict:
    """Check the shared frozen identity before displaying a published summary."""
    expected_systems = {
        "test_slm_summary.json": ["large_0.7", "majority_0.5_0.7_0.7"],
        "test_llm_summary.json": ["deepseek_low"],
        "test_comparison_summary.json": [
            "large_0.7", "majority_0.5_0.7_0.7", "deepseek_low"
        ],
    }
    if filename not in expected_systems:
        raise ValueError("Choose a published primary test summary")
    directory = Path(root) / "results"
    specification = json.loads((directory / "test_comparison_spec.json").read_text())
    canonical = json.dumps(specification["settings"], ensure_ascii=False,
                           sort_keys=True, separators=(",", ":"), allow_nan=False)
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    report = json.loads((directory / filename).read_text())
    if (specification["fingerprint"] != fingerprint
            or report["fingerprint"] != fingerprint
            or report["reference"] != "selected_sentence_manual_union"
            or [row["system"] for row in report["rows"]] != expected_systems[filename]):
        raise ValueError("Published report and frozen test specification disagree")
    return report
